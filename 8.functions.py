# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:01:47 2016

@author: Dan
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from random import randint

def make_maze(ts):
    A = np.zeros([9,6]) 
    A[2,2:5] = 1 
    A[7,3:6] = 1 
    A[5,1] = 1
    return A

def make_maze2(ts):
    A = np.zeros([9,6]) 
    if( ts < 3000 ):
      A[1:8,3] = 1 # the first maze
    else:
      A[1:7,3] = 1 # the second maze
    return A
    
