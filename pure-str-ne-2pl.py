#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script allows you to find pure strategy NE of a 2 player bimatrix game. 

@author: amyrhee@nyu.edu
"""

# Finding pure strategy NE in an (n x n) 2 player bimatrix game

import numpy as np

class bimatrix:
    def __init__(self, row_payoffs , col_payoffs):
        self.mat_r = np.array(row_payoffs)
        self.mat_c = np.array(col_payoffs)
        self.n_str = len(row_payoffs)
    
    def pure_ne(self):
        max_r = np.zeros((self.n_str,1))
        ind_r = np.zeros((self.n_str,2))
        
        max_c = np.zeros((self.n_str,1))
        ind_c = np.zeros((self.n_str,2))
        
        pure_ne = []
        
        for i in range(self.n_str):
            max_r[i] = np.max(self.mat_r[:,i])
            ind_r[i] = np.argmax(self.mat_r[:,i]), i
            
            max_c[i] = np.max(self.mat_c[i,:])
            ind_c[i] = i, np.argmax(self.mat_c[i,:])
            
            if (ind_r[i,:] == ind_c[i,:]).all() == True:
                pure_ne.append(ind_r[i,:])
        self.pure_ne = pure_ne
        return pure_ne
    
    def pure_ne_payoffs(self):
        big_mat = np.dstack(((self.mat_r, self.mat_c)))
        
        ne_payoffs = []
        
        for i in range(len(self.pure_ne)):
            str_ind = list(map(int, self.pure_ne[i]))
            ne_payoffs.append(big_mat[str_ind[0], str_ind[1]])
        
        self.ne_payoffs = ne_payoffs
        return ne_payoffs
    
    
# Example with 3 x 3 game

mat_r = np.array([[10, 0, 0], [6, 1, -1], [12, -1, 5]])
mat_c = np.array([[10, 6, 12], [0, 5, -1], [0, -1, 1]])

gamma = bimatrix(mat_r, mat_c)

# Assuming that we've enumerated the strategies from top (0) to bottom (n) for the row player, and left (0) to right (n) for the column player
print('The pure NE are the following strategy profiles (row pl. strategy index, column pl. strategy index): ', 
      gamma.pure_ne())

print('The respective payoffs are: ', gamma.pure_ne_payoffs())
