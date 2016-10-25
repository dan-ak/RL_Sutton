# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 16:16:45 2016

@author: Dan
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt

num_cars = 21
rewards = 20
max_rentals = 11
max_returns = 11
numactions = 11

lambdas = np.array([[3,4],[3,2]])
#A = np.ones([num_cars,num_cars])*5
#V = np.zeros([num_cars,num_cars])

gamma = 0.9

def poisson(n,l):
    return np.exp(-l)*np.power(l,n)/scipy.math.factorial(n)

def get_poisson_prob():
    p_prob = np.zeros([2,2,16])
    for i in range(2):
        for j in range(2):
            for n in range(16):
                p_prob[i][j][n] = poisson(n,lambdas[i][j])
    return p_prob
    
def update_Values(A,oldV,reward,prob):
    V = oldV
    for s1 in range(num_cars):
        for s2 in range(num_cars):
            print (s1,s2)
            a = A[s1][s2]
            V[s1][s2] = 0
            
            for ns1 in range(num_cars):
                for ns2 in range(num_cars):
                    V[s1][s2] += prob[s1][s2][ns1][ns2][a]*(reward[s1][s2][ns1][ns2][a] + gamma*V[ns1][ns2])                  
    return V                

def best_a(V,s1,s2,reward,prob):
    best_a = 5
    best_v = 0
    for a in range(numactions):
        temp_v = 0
           
        for ns1 in range(num_cars):
            for ns2 in range(num_cars):
                temp_v += prob[s1][s2][ns1][ns2][a]*(reward[s1][s2][ns1][ns2][a] + gamma*V[ns1][ns2])       
                
        if (temp_v > best_v):
            best_a = a
            best_v = temp_v
    
    return best_a

def update_Actions(oldA,V,reward,prob):
    newA = oldA
    for s1 in range(num_cars):
        for s2 in range(num_cars):
            print (s1,s2)
            newA[s1][s2] = best_a(V,s1,s2,reward,prob)
            
    return newA
    
def expected_reward(s1,s2,a,reward,prob):
    r = 0.0
    p = 0.0    
    for ns1 in range(num_cars):
        for ns2 in range(num_cars):
            p = p + prob[s1][s2][ns1][ns2][a]
            r += prob[s1][s2][ns1][ns2][a]*reward[s1][s2][ns1][ns2][a]        
    return r
    
def get_prob_reward():
    
    prob = np.zeros([num_cars, num_cars, num_cars, num_cars, numactions])
    reward = np.zeros([num_cars, num_cars, num_cars, num_cars, numactions])
    
    #loop over all states
    for s1 in range(num_cars):
        for s2 in range(num_cars):
            print (s1,s2)
            
            #loop over all actions
            for a in range(numactions):
                move = a - 5
                
                #Select legal actions
                if (move < -1*s1): 
                    continue
                if (move > s2): 
                    continue 
                
                nsa1 = min(20, s1 + move)
                nsa2 = min(20, s2 - move)
                
                #loop over all rentals            
                for i1 in range(max_rentals):
                    for i2 in range(max_rentals):
                    
                    
                        #loop over all returns
                        for j1 in range(max_returns):
                            for j2 in range(max_returns):
                                
                                p = p_prob[0][0][i1]*p_prob[1][0][i2]*p_prob[1][0][j1]*p_prob[1][1][j2]
                                                              
                                r = 10*min(nsa1,i1) + 10*min(nsa2,i2) - 2*np.absolute(move)
                                
                                ns1 = max(0,nsa1 - i1)
                                ns2 = max(0,nsa2 - i2)
                                
                                ns1 = min(20,ns1+j1)
                                ns2 = min(20,ns2+j2)

                                oldp = prob[s1][s2][ns1][ns2][a]
                                oldr = reward[s1][s2][ns1][ns2][a]
                                
                                prob[s1][s2][ns1][ns2][a] = oldp + p
                                reward[s1][s2][ns1][ns2][a] = (oldp*oldr + r*p)/(oldp + p)

                
    return prob, reward



