# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:18:07 2016

@author: Dan
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
import maze_func
from random import randint


def dynaQ_maze(ALPHA, EPSILON, GAMMA, nPlanningSteps, maze_fn, s_starts, s_end, MAX_N_STEPS):

    maze = maze_fn(0)
    nStates = int(maze.shape[0]*maze.shape[1]) # the maximal number of states:
    nActions = 4 # of actions
     
    Q = -3.0*np.ones([nStates,nActions])# Action-Value function
    
    seen_states = [] # storage for the sequence of states we have observed
    
    act_taken   = np.zeros([nStates,nActions], dtype=int) # binary storage for the sequence of actions taken in each state 
     
    Model_ns = np.zeros([nStates,nActions], dtype=int) # <- Model of next state we will obtain 
    Model_nr = np.zeros([nStates,nActions], dtype=int) # <- Model of next reward we will obtain 
    
    ets = [] # keep track of how many timestep we take per episode
    ts = 0 
    
    s_t = s_start # initialize the starting state
    
    
    for t in range(MAX_N_STEPS):
      
        if(t % 100 == 0):
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
            ran_s_t = seen_states[tmp[0]] 
            
            pro_action = act_taken[ran_s_t]*1.0 / sum(act_taken[ran_s_t]) #pick a random action 
            ran_a_t = np.random.choice(range(len(pro_action)), 1, p = list(pro_action))
            
            model_s_t1   = Model_ns[ran_s_t,ran_a_t] #use model to get s_t1 and r_t1
            model_r_t1   = Model_nr[ran_s_t,ran_a_t]
                    
            if is_terminal(model_s_t1, maze): # update our action-value function
                Q[ran_s_t,ran_a_t] = Q[ran_s_t,ran_a_t] + ALPHA*(model_r_t1 - Q[ran_s_t,ran_a_t])  
            else:
                Q[ran_s_t,ran_a_t] = Q[ran_s_t,ran_a_t] + ALPHA*(model_r_t1 + GAMMA*np.max(Q[model_s_t1]) - Q[ran_s_t,ran_a_t]) 
                                                              
    
        s_t = s_t1 
        ts = ts + 1 
             
        if is_terminal(s_t, maze): # If we have solved start over
            s_t = s_start 
            ets.append(ts)
            ts = 0 

    return Q, ets 


ALPHA = 0.1
EPSILON = 0.05
GAMMA = 0.95
s_start = state_to_index([0,3], maze)
s_end = state_to_index([8,5], maze)
nPlanningSteps = 5
MAX_N_STEPS = 5000


Q, ets = dynaQ_maze(ALPHA, EPSILON, GAMMA, nPlanningSteps, make_maze, s_start, s_end, MAX_N_STEPS)