# -*- coding: utf-8 -*-
"""
Created on Sun Oct 09 17:37:12 2016

@author: Dan
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_car(path,track):

    vector_list = []
    
    for i in range(len(path)):
        p = np.array([path[i][0],path[i][1], -0.5*path[i][2], 0.5*(path[i][3]-5)])
        vector_list.append(p)
        
    
    soa = np.array(vector_list) 
    Y,X,V,U = zip(*soa)
    plt.figure()
    ax = plt.gca()
    ax.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1)
    ax.set_xlim([-1,17])
    ax.set_ylim([31,-1])
    fig_size = plt.rcParams["figure.figsize"]
    print fig_size
    plt.imshow(track, interpolation='nearest')
    
    plt.draw()
    plt.show()