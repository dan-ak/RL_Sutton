# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:45:19 2016

@author: Dan
"""

first = np.zeros(1001)
first[0] = 1.

for i in range(1001):
    for d in range(6):
        if (i - d - 1) >= 0 :
            first[i] += first[i-d-1]*1./6.

second =     