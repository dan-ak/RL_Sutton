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


ALPHA = 0.1
EPSILON = 0.1
GAMMA = 0.95
maze = make_maze(0)
s_start = state_to_index([0,3], maze)
s_end = state_to_index([8,5], maze)
MAX_N_STEPS = 5000
PLOT_STEPS = 0
nPlanningSteps = 5 

# the maximal number of states: 
nStates = int(maze.shape[0]*maze.shape[1]); 

# on each grid we can choose from among at most this many actions:
nActions = 4 

# An array to hold the values of the action-value function ... with planning 
# it seems that the RL algorithm is very sensitive to the initial conditions 
# we take for Q ... this value seems to work for planning steps 0 ... 50 
Q = -3.0*np.ones([nStates,nActions]);

seen_states = []# storage for the sequence of states we have observed

act_taken   = np.zeros([nStates,nActions], dtype=int) # storage for the sequence of actions taken in each state 

# Some arrays to hold the model of the environment (assuming it is determinastic): 
Model_ns = np.zeros([nStates,nActions], dtype=int) # <- next state we will obtain 
Model_nr = np.zeros([nStates,nActions], dtype=int) # <- next reward we will obtain 

ets = [] # keep track of how many timestep we take per episode
ts = 0 

# keep track of how many times we reach the end of our maze:
numFinishes = 0 

# keep track of how many times we have solved our problem in this number of timesteps:
cr = np.zeros(MAX_N_STEPS+1)
cr[0] = 0; 

# initialize the starting state
s_t = s_start


for tsi in range(MAX_N_STEPS):
  if(tsi % 100 == 0):
      print "working on step" , tsi

  # pick action using an epsilon greedy policy derived from Q: 
  a_t = get_action(s_t, Q, EPSILON)
  
  # keep track of the states/action seen 
  if s_t not in seen_states:
      seen_states.append(s_t)
  
  act_taken[s_t, a_t] = 1; 
  

  # propagate to state stp1 and collect a reward rew
  s_t1, r_t1 = next_state(s_t, a_t, maze)     
  
  # update our action-value function: 
  if is_terminal(s_t1, maze): # s_t1 in terminal state
    Q[s_t,a_t] = Q[s_t,a_t] + ALPHA*(r_t1 - Q[s_t,a_t]) 
  else:  
    Q[s_t,a_t] = Q[s_t,a_t] + ALPHA*(r_t1 + GAMMA*np.max(Q[s_t1]) - Q[s_t,a_t]) 

    
  # update our model of the environment: 
  Model_ns[s_t, a_t] = s_t1 
  Model_nr[s_t, a_t] = r_t1;
  
  # perform some planning steps: 
  for pi in range(nPlanningSteps):

    # pick a random state we have seen "r_sti": 
    tmp = np.random.permutation(len(seen_states))
    ran_s_t = seen_states[tmp[0]] 
    
    # pick a random action from the ones that we have seen (in this state) "r_at": 
    pro_action = act_taken[ran_s_t]*1.0 / sum(act_taken[ran_s_t]) #<- the probabilty of each specific action ... 
    ran_a_t = np.random.choice(range(len(pro_action)), 1, p = list(pro_action))
    
    # get our models predition of the next state (and reward) "model_sprimei", "model_rew": 
    model_s_t1   = Model_ns[ran_s_t,ran_a_t]
    model_r_t1   = Model_nr[ran_s_t,ran_a_t]
    
    # update our action-value function: 
    if is_terminal(model_s_t1, maze): # model_s_t1 is the terminal state
        Q[ran_s_t,ran_a_t] = Q[ran_s_t,ran_a_t] + ALPHA*(model_r_t1 - Q[ran_s_t,ran_a_t])  
    else:
        Q[ran_s_t,ran_a_t] = Q[ran_s_t,ran_a_t] + ALPHA*(model_r_t1 + GAMMA*np.max(Q[model_s_t1]) - Q[ran_s_t,ran_a_t]) 
                                                      
 
  
  # shift everything by one (this completes one "step" of the algorithm): 
  s_t = s_t1 
  ts=ts+1 

  # for continual planning ... if we have "solved" our maze we will start over:
  if is_terminal(s_t, maze): # stp1 is the terminal state
    s_t = s_start 
    ets.append(ts)
    ts = 0
    numFinishes += 1 # record that we got to the end: 
    cr[tsi+1] = cr[tsi]+1 # record that we got to the end:
  else:  
    cr[tsi+1] = cr[tsi]  #record that we did not get to the end and our cummulative reward count does not change:
 
