# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:01:47 2016

@author: Dan
"""

import numpy as np
import matplotlib.pyplot as plt
from random import randint

def is_terminal(s_t, maze):
    try:
        s = index_to_state(s_t, maze)
    except TypeError:
        s = s_t 
    
    if maze[s[0],s[1]] == 2:
        return True
    else:
        return False

def make_maze(ts):
    A = np.zeros([9,6], dtype=np.int) 
    A[2,2:5] = 1 
    A[7,3:6] = 1 
    A[5,1] = 1
    A[8,5] = 2
    return A

def make_maze2(ts):
    A = np.zeros([9,6], dtype=np.int) 
    if( ts < 3000 ):
      A[1:8,3] = 1 # the first maze
    else:
      A[1:7,3] = 1 # the second maze
    return A

def index_to_state(s_t, maze):
    shape = maze.shape 
    s = np.zeros(2, dtype=np.int)
    s[0] = int(s_t % shape[0])
    s[1] = int(s_t / (1.0*shape[0]))
    return s

def state_to_index(s, maze):
    shape = maze.shape
    s_t = int(s[0] + shape[0]*s[1])
    return s_t

def get_action(s, Q, epsilon):

    num_a = len(Q[s])
    
    if np.random.rand() < epsilon:
        return np.int(np.random.rand()*num_a)
     
    return np.argmax(Q[s])
    
def next_state(s, a, maze):
  
    shape = maze.shape    
    s_t = index_to_state(s, maze)
   
    s_t1 = s_t.copy()
    r_t1 = 0
    out_of_bounds = False
    
    if a == 0:
        s_t1[0] += 1
    elif a == 1:
        s_t1[1] += 1
    elif a == 2:
        s_t1[0] += -1
    elif a == 3:
        s_t1[1] += -1
    
    if s_t1[0] > (shape[0]-1) or s_t1[1] > (shape[1]-1) or s_t1[0] < 0 or s_t1[1] < 0:
        s_t1 = s_t
        out_of_bounds = True
    
    if not out_of_bounds and maze[s_t1[0],s_t1[1]] == 1:
        s_t1 = s_t
        
    if not out_of_bounds and is_terminal(s_t1, maze):
        r_t1 = 1
    
    s_new = state_to_index(s_t1, maze)    
    return s_new, r_t1

def get_pre_states(Model_ns, s_t):
    
    pre_states = []     
    
    for i in range(Model_ns.shape[0]):
        for j in range(Model_ns.shape[1]):
            if Model_ns[i][j] == s_t:
                pre_states.append([i,j])
                       
    return pre_states
    
    
    
def display_Q(Q,maze):
    disp = np.zeros([9,6])
    for i in range(9):
        for j in range(6):
            disp[i,j] = 100.0*np.max(Q[state_to_index([i,j], maze)])
            print np.max(Q[state_to_index([i,j], maze)])
    plt.figure()
    ax = plt.gca()
    ax.set_xlim([-1,9])
    ax.set_ylim([-1,6])
    plt.imshow(np.transpose(disp), interpolation='nearest')    
    plt.draw()
    plt.show()
    
def plot_Q(Q,maze):
    vector_list = []
    
    for i in range(9):
        for j in range(6):
            a = np.argmax(Q[state_to_index([i,j], maze)])
            if a == 0:
                x = 1
                y = 0
            if a == 1:
                x = 0
                y = 1
            if a == 2:
                x = -1
                y = 0
            if a == 3:
                x = 0
                y = -1
            if Q[state_to_index([i,j], maze)][a] == -3:
                x = 0
                y = 0
            p = np.array([i,j,x,y])
            vector_list.append(p)  
    
    soa = np.array(vector_list) 
    X,Y,U,V = zip(*soa)
    plt.figure()
    ax = plt.gca()
    ax.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1)
    ax.set_xlim([-1,9])
    ax.set_ylim([-1,6])
    plt.imshow(np.transpose(maze), interpolation='nearest')    
    plt.draw()
    plt.show()