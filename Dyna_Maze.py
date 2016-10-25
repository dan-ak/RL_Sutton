# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:18:07 2016

@author: Dan
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
import maze_func
import Queue
from random import randint


def dynaQ_maze(ALPHA, EPSILON, GAMMA, nPlanningSteps, maze_fn, s_starts, s_end, MAX_N_STEPS):

    maze = maze_fn(0)
    nStates = int(maze.shape[0]*maze.shape[1]) # the maximal number of states:
    nActions = 4 # of actions
     
    Q = 0.0*np.ones([nStates,nActions])# Action-Value function
    
    seen_states = [] # storage for the sequence of states we have observed
    
    act_taken   = np.zeros([nStates,nActions], dtype=int) # binary storage for the sequence of actions taken in each state 
     
    Model_ns = np.zeros([nStates,nActions], dtype=int) # <- Model of next state we will obtain 
    Model_nr = np.zeros([nStates,nActions], dtype=int) # <- Model of next reward we will obtain 
    
    ets = [] # keep track of how many timestep we take per episode
    ts = 0 
    cr = np.zeros(MAX_N_STEPS + 1)
    
    s_t = s_start # initialize the starting state
    
    for t in range(MAX_N_STEPS):
        
        if(t % 500 == 0):
            print "working on step" , t 
                 
        if s_t not in seen_states: # keep track of the states/action seen
            seen_states.append(s_t)
        
        a_t = get_action(s_t, Q, EPSILON) # pick action using an epsilon greedy policy derived from Q:    
        act_taken[s_t, a_t] = 1         
        s_t1, r_t1 = next_state(s_t, a_t, maze) # propagate to state s_t1 and collect a reward r_t1     
          
           
        if is_terminal(s_t1, maze): # update our action-value function:
            Q[s_t,a_t] = Q[s_t,a_t] + ALPHA*(r_t1 - Q[s_t,a_t]) 
        else:  
            Q[s_t,a_t] = Q[s_t,a_t] + ALPHA*(r_t1 + GAMMA*np.max(Q[s_t1]) - Q[s_t,a_t]) 
        
        
        Model_ns[s_t, a_t] = s_t1 # update our model of the environment:
        Model_nr[s_t, a_t] = r_t1
          
           
        for pi in range(nPlanningSteps): # perform some planning 
        
             
            tmp = np.random.permutation(len(seen_states)) # pick a random state we have seen "ran_s_t"
            p_s_t = seen_states[tmp[0]] 
            
            pro_action = act_taken[p_s_t]*1.0 / sum(act_taken[p_s_t]) #pick a random action 
            p_a_t = np.random.choice(range(len(pro_action)), 1, p = list(pro_action))
            
            model_s_t1   = Model_ns[p_s_t,p_a_t] #use model to get s_t1 and r_t1
            model_r_t1   = Model_nr[p_s_t,p_a_t]
                    
            if is_terminal(model_s_t1, maze): # update our action-value function
                Q[p_s_t,p_a_t] = Q[p_s_t,p_a_t] + ALPHA*(model_r_t1 - Q[p_s_t,p_a_t])  
            else:
                Q[p_s_t,p_a_t] = Q[p_s_t,p_a_t] + ALPHA*(model_r_t1 + GAMMA*np.max(Q[model_s_t1]) - Q[p_s_t,p_a_t]) 
                                                              
    
        s_t = s_t1 
        ts = ts + 1 
             
        if is_terminal(s_t, maze): # If we have solved start over
    
            s_t = s_start 
            ets.append(ts)
            ts = 0
            cr[t] += 1

        cr[t+1] = cr[t]
        
    return Q, ets, cr 


def p_sweeping_maze(ALPHA, EPSILON, GAMMA, THETA, nPlanningSteps, maze_fn, s_start, s_end, MAX_N_STEPS):

    maze = maze_fn(0)
    nStates = int(maze.shape[0]*maze.shape[1]) # the maximal number of states:
    nActions = 4 # of actions
     
    Q = 0.0*np.ones([nStates,nActions])# Action-Value function
    
    seen_states = [] # storage for the sequence of states we have observed
    
    act_taken   = np.zeros([nStates,nActions], dtype=int) # binary storage for the sequence of actions taken in each state 
     
    Model_ns = np.zeros([nStates,nActions], dtype=int) # <- Model of next state we will obtain 
    Model_nr = np.zeros([nStates,nActions], dtype=int) # <- Model of next reward we will obtain 
    
    p_queue = Queue.PriorityQueue()
    p_q = np.zeros([nStates,nActions])
    
    ets = [] # keep track of how many timestep we take per episode
    ts = 0 
    cr = np.zeros(MAX_N_STEPS + 1)    
    
    s_t = s_start # initialize the starting state
    
    for t in range(MAX_N_STEPS):
        
       
        if(t % 500 == 0):
            print "working on step" , t 
                 
        if s_t not in seen_states: # keep track of the states/action seen
            seen_states.append(s_t)
        
        a_t = get_action(s_t, Q, EPSILON) # pick action using an epsilon greedy policy derived from Q:    
        act_taken[s_t, a_t] = 1         
        s_t1, r_t1 = next_state(s_t, a_t, maze) # propagate to state s_t1 and collect a reward r_t1     
        
        Model_ns[s_t, a_t] = s_t1 # update our model of the environment:
        Model_nr[s_t, a_t] = r_t1
        
        p = np.abs(r_t1 + GAMMA*np.max(Q[s_t1]) - Q[s_t,a_t])
        
        if p > THETA and p > p_q[s_t, a_t]:
            p_queue.put([s_t, a_t], 1.0/p)
            p_q[s_t, a_t] = p
 
        
        for pi in range(nPlanningSteps): # perform some planning 
        
            #if p_queue.empty():
            #    break
            if np.max(p_q) == 0:
                break
            
            #print p_q
            
            p_s_t, p_a_t = np.unravel_index(p_q.argmax(), p_q.shape)
            p_q[p_s_t, p_a_t] = 0.0
        
            #pop = p_queue.get()
            #p_s_t = pop[0]
            #p_a_t = pop[1]
            
            model_s_t1   = Model_ns[p_s_t,p_a_t] #use model to get s_t1 and r_t1
            model_r_t1   = Model_nr[p_s_t,p_a_t]
                    
            if is_terminal(model_s_t1, maze): # update our action-value function
                Q[p_s_t, p_a_t] = Q[p_s_t, p_a_t] + ALPHA*(model_r_t1 - Q[p_s_t, p_a_t])  
            else:
                Q[p_s_t, p_a_t] = Q[p_s_t, p_a_t] + ALPHA*(model_r_t1 + GAMMA*np.max(Q[model_s_t1]) - Q[p_s_t, p_a_t]) 
            
            pre_states = get_pre_states(Model_ns, s_t)
            
            for states in pre_states:
                l_s_t = states[0]
                l_a_t = states[1]
                l_r_t1 = Model_nr[l_s_t, l_a_t]
                p = np.abs(l_r_t1 + GAMMA*np.max(Q[s_t]) - Q[l_s_t, l_a_t])
                
                if p > THETA and p > p_q[l_s_t, l_a_t]:
                    p_queue.put([l_s_t, l_a_t], 1.0/p)
                    p_q[l_s_t, l_a_t] = p
   
    
        s_t = s_t1 
        ts = ts + 1 
             
        if is_terminal(s_t, maze): # If we have solved start over
    
            s_t = s_start 
            ets.append(ts)
            ts = 0
            cr[t] += 1

        cr[t+1] = cr[t]
        
    return Q, ets, cr
