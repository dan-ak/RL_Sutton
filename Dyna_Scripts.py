# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:41:43 2016

@author: Dan
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
import maze_func
import Dyna_Maze
from random import randint

ALPHA = 0.1
EPSILON = 0.1
GAMMA = 0.95
THETA = 0.0001
maze = make_maze(0)
s_start = state_to_index([0,3], maze)
s_end = state_to_index([8,5], maze)
nPlanningSteps = 50
MAX_N_STEPS = 10000

nSteps = [0,5,50]
results = np.zeros([3,50])


#for i, n in enumerate(nSteps):
#    for j in range(10):
#        if n == 5:
#            MAX_N_STEPS = 6000
#        if n == 50:
#           MAX_N_STEPS = 4000
 #           
 #       Q, ets = dynaQ_maze(ALPHA, EPSILON, GAMMA, n, make_maze, s_start, s_end, MAX_N_STEPS)
  #      results[i] += (1.0/10.0)*np.array(ets[:50])


#Q, ets, cr = dynaQ_maze(ALPHA, EPSILON, GAMMA, 0, make_maze, s_start, s_end, MAX_N_STEPS)        

Q, ets, cr = p_sweeping_maze(ALPHA, EPSILON, GAMMA, THETA, nPlanningSteps, make_maze, s_start, s_end, MAX_N_STEPS)