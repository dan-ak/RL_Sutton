# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 18:01:44 2016

@author: Dan
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from random import randint

#Q = np.ones([10*7,8])*0

EPSILON = 0.1
ALPHA = 0.2
GAMMA = 1

def graphPi(Q):
    S = []
    s = stateNum([0,3])
    r = -1
    while r == -1:
        S.append(s)
        a = getAction(s, Q, 0)
        s, r = nextState(s,a)
        
    S.append(s)
    return S    
    

def stateNum(S):
    if S[0] > 9 or S[0] < 0 or S[1] > 6 or S[1] < 0:
       return stateNum([0,0])
       
    s = S[0] + 10*S[1]
    return s

def stateVal(s):
    S = np.zeros(2)
    S[0] = s%10
    S[1] = np.int(s/10)
    
    return S

def getAction(s, Q, epsilon):
    
    
    num_a = len(Q[s])
    
    if np.random.rand() < epsilon:
        return np.int(np.random.rand()*num_a)

    max_q = Q[s][0]
    
    a = 0
    
    for i in range(num_a):
        
        if Q[s][i] > max_q:
            a = i
            max_q = Q[s][i]
    
    return a

def valPi(Q):
    v = np.ones([10,7])*-50
    for i in range(10):
        for j in range(7):
            for a in range(8):
                if Q[stateNum([i,j])][a] > v[i][j]:
                    v[i][j] = Q[stateNum([i,j])][a] 
    return v
     
def nextState(s,a):
    S = stateVal(s)
    r = -1
    rand = np.random.rand()
    
    if S[0] > 2 and S[0] < 9:
        S[1] += 1
        
        if rand < .33:
            S[1] += -1
        if rand > .66:
            S[1] += 1
        
    if S[0] > 5 and S[0] < 8:
        S[1] += 1  
    
    if a == 0:
        S[0] = S[0] + 1
    if a == 1:
        S[1] = S[1] + 1
    if a == 2:
        S[0] = S[0] - 1
    if a == 3:
        S[1] = S[1] - 1
    if a == 4:
        S[0] = S[0] + 1
        S[1] = S[1] + 1
    if a == 5:
        S[0] = S[0] - 1
        S[1] = S[1] + 1
    if a == 6:
        S[0] = S[0] - 1
        S[1] = S[1] - 1
    if a == 7:
        S[0] = S[0] + 1
        S[1] = S[1] - 1
        
    
    
    if S[1] > 6:
        S[1] = 6
    if S[1] < 0:
        S[1] = 0
    if S[0] > 9:
        S[0] = 9
    if S[0] < 0:
        S[0] = 0
        
    
    if stateNum(S) == stateNum([7,3]):
        r = 0
    
    return stateNum(S), r


t_steps = 0
episodes = []
for i in range(2000):
    s = stateNum([0,3])
    a = getAction(s, Q, EPSILON)
    while True:
        t_steps += 1
        if t_steps % 1000 == 0:
            episodes.append(i)
        s_p, r = nextState(s,a)
        a_p = getAction(s_p, Q, EPSILON)
        Q[s][a] += ALPHA*(r + GAMMA*Q[s_p][a_p] - Q[s][a])
        s = s_p
        a = a_p
        if r == 0:
            break    

#S = graphPi(Q)            