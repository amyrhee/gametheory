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
        max_r, max_c = np.zeros((self.n_str,1)), np.zeros((self.n_str,1))
        ind_r, ind_c = np.zeros((self.n_str,2)), np.zeros((self.n_str,2))
        
        ne = []
        
        for i in range(self.n_str):
            max_r[i] = np.max(self.mat_r[:,i])
            ind_r[i] = np.argmax(self.mat_r[:,i]), i
            
            max_c[i] = np.max(self.mat_c[i,:])
            ind_c[i] = i, np.argmax(self.mat_c[i,:])
            
            if (ind_r[i,:] == ind_c[i,:]).all() == True:
                ne.append(ind_r[i,:])
                
        self.pure_neq = ne
        
        if len(ne) == 0:
            print("No pure NE found.")
        else:
            return ne
    
    def pure_ne_payoffs(self):
        big_mat = np.dstack(((self.mat_r, self.mat_c)))
        
        ne_payoffs = []
        
        for i in range(len(self.pure_ne())):
            str_ind = list(map(int, self.pure_neq[i]))
            ne_payoffs.append(big_mat[str_ind[0], str_ind[1]])
        
        self.ne_payoffs = ne_payoffs
        return ne_payoffs
