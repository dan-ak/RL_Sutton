# -*- coding: utf-8 -*-
"""
Created on Sun Oct 09 13:35:13 2016

@author: Dan
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt
from random import randint

track = np.array([[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
                [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0]])

gamma = 1        

#Q = np.ones([32*17*6*11,9])*1
#for i in range(32*17*6*11):
#    Q[i] = np.ones(9)*(np.int(i%32))/-3
#C = np.zeros([32*17*6*11,9])
#pi = np.ones([32*17*6*11])*5


def nextState(S,a):
    s = stateVal(S)
    R = -1

    s[2] = min(5,max(0,s[2] - 1 + np.int(a/3.0)))
    s[3] = min(10,max(0,s[3] + a%3 -1))
    
    s[0] = s[0]-s[2]
    s[1] = s[1] + s[3] - 5
    
    if s[0] > -1 and s[0]<6 and s[1]>15:
        R = 0
        s[1] = 16
        return stateNum(s), R
    
    if s[0]<0 or s[0]>31 or s[1]<0 or s[1]>16 :
        s = [31,randint(3,8),0,5]
        R = -1
        return stateNum(s), R
    
    if s[1] > 9 and s[0]+s[2] > 7 and s[0]+(s[1] - 9)*s[2]/(s[3]-5) > 7 :
        s = [31,randint(3,8),0,5]
        R = -1
        return stateNum(s), R
    
    if track[np.int(s[0])][np.int(s[1])] == 0:
        s = [31,randint(3,8),0,5]
        R = -1
        return stateNum(s), R
    
    return stateNum(s), R
    
def stateNum(S):
    s = S[0]+32*S[1]+32*17*S[2]+32*17*6*S[3]
    return s

def stateVal(s):
    S = np.zeros(4)
    S[0] = np.int(s%(32))
    S[1] = np.int(s/32)%17
    S[2] = np.int(s/(32*17))%6
    S[3] = np.int(s/(32*17*6)) 
    return S 
    
def episode_generate():
    S = []
    A = []
    R = [np.NaN]
    
    s = stateNum([31,randint(3,8),0,5])
    a = randint(0,8)
    
    S.append(s)
    A.append(a)
    
    while True:
        s, r = nextState(s,a)
        
        S.append(s)
        R.append(r)
        
        if(r==0):
            A.append(np.NaN)
            break
        
        a = randint(0,8)
        A.append(a)
    
    return S, A, R

def sample_pi():
    path = []
    s = stateNum([31,randint(3,8),0,5])
    path.append(stateVal(s))
    
    while True:
        s,r = nextState(s,pi[s])
        path.append(stateVal(s))
        if(r==0):
            return path
    

testPath = []
longestPath = []
    
for i in range(500):
    S,A,R = episode_generate()
    G = 0
    W = 1
    testPath = []
    
    
    for t in range(len(S)-2,-1,-1):
        G = gamma*G + R[t+1]
        C[int(S[t]),int(A[t])] += W
        Q[int(S[t]),int(A[t])] += W/C[int(S[t]),int(A[t])]*(G-Q[int(S[t]),int(A[t])])
        max_a = -10000
        for a in range(9):
            if (Q[int(S[t]),a] > max_a):
                max_a = Q[int(S[t]),a]
                pi[int(S[t])] = a
        if A[t] != pi[int(S[t])]:
            break
        W = W*8
        
        testPath.append(stateVal(S[t]))

    if len(testPath) > len(longestPath):
        longestPath = testPath
        
#plot_car(sample_pi(),track)