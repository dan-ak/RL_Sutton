# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 17:26:40 2016

@author: Dan
"""

import numpy as np
import matplotlib.pyplot as plt

def run_task_kbandit(eps, steps, alpha, mean_payouts, var_payouts, drift_payouts):
    
    OptAct = np.zeros(steps)
    payouts = mean_payouts
    k = np.size(mean_payouts)
    Q = np.zeros(k)
    n = np.zeros(k)
    R = np.zeros(steps)
    al = alpha
    
    for t in range(steps):
        
        if np.random.random() < eps:
            a = np.int(np.random.random()*10)
        else:
            er = np.random.rand(k)/100000
            Q = Q + er
            a = np.argmax(Q)
            Q = Q - er

        if a == np.argmax(payouts): 
            OptAct[t] = 1
        
        R[t] = np.random.normal(payouts[a],var_payouts[a])         
        n[a] = n[a] + 1
        if alpha == 0: 
            al = 1.0/n[a]
        Q[a] = Q[a] + al * (R[t] - Q[a])
        payouts = payouts + drift_payouts*np.random.normal(0,1,k)

    
    return OptAct, R
    
def average(eps,alpha):
    
    trials = 5000
    steps = 4000
    avgOA = np.zeros(steps)
    avgR = np.zeros(steps)
    
    
    for r in range(trials):
        
        OptAct, R = run_task_kbandit(eps,steps,alpha,np.random.normal(0,.01,10),np.ones(10),np.ones(10))
    
        
        avgOA = avgOA + 1.0/(r+1.0) * (OptAct - avgOA) 
        avgR = avgR + 1.0/(r+1.0) * (R - avgR)
        
    return avgOA, avgR
    

eps = 0.1
alpha = [0,0.1]


avgOA1, avgR1 = average(eps, 0)
plt.figure(1)
plt.plot(avgR1, color='r')
plt.figure(2)
plt.plot(avgOA1, color='r')

avgOA2, avgR2 = average(eps, 0.1)
plt.figure(1)
plt.plot(avgR2, color='b')
plt.figure(2)
plt.plot(avgOA2, color='b')