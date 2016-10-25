# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 16:54:15 2016

@author: Dan
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#V = np.zeros(101)
V[0]=0
V[100]= 1
A = np.zeros(101)
A_V = np.zeros([101,51])
a_EV = np.zeros([101,51])
prob_h = 0.45
gamma = 1

for i in range (300):
    
    for s in range (100):
        v = V[s]
        
        for a in range (min(s,100-s)+1):
            v_p = prob_h*gamma*V[s+a] + (1-prob_h)*gamma*V[s-a]
            if (v_p > V[s]):
                V[s] = v_p
        
for s in range (100):

    for a in range (min(s,100-s)):
        v_p = prob_h*gamma*V[s+a+1] + (1-prob_h)*gamma*V[s-a-1]
        A_V[s][a] = v_p
        if ((V[s] - v_p)<.0005):
            A[s] = a+1
            a_EV[s][a] = 1
            
A_norm = np.zeros([101,51]) 
           
for s in range (100):
    sum1 = 0
    for a in range (min(s,100-s)):
        sum1 += A_V[s][a]
    avg = sum1 / max(1,min(s,100-s))
    for a in range (min(s,100-s)):
        A_norm[s][a] = A_V[s][a] - avg