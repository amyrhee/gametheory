#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script allows you to find pure strategy NE of a 2 player (n x n) bimatrix game. 

@author: amyrhee@nyu.edu
"""

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
            ne = 'No pure NE found.'
        return ne
    
    def pure_ne_payoffs(self):
        big_mat = np.dstack(((self.mat_r, self.mat_c)))
        
        ne_payoffs = []
        
        for i in range(len(self.pure_ne())):
            str_ind = list(map(int, self.pure_neq[i]))
            ne_payoffs.append(big_mat[str_ind[0], str_ind[1]])
        
        self.ne_payoffs = ne_payoffs
        return ne_payoffs
    
    def mixed_ne(self): # Need to redo and add in constraints
        def row_pl():
            k = 0
            a_n = -1 * self.mat_r[:,k].reshape((self.mat_r.shape[0],1))
            a_tilde = self.mat_r - ((-1*a_n).T @ np.eye(self.mat_r.shape[0])).T
            a_tilde[:,k] = -1*np.ones((self.mat_r.shape[0],))
            
            while np.linalg.det(a_tilde) == 0:
                k += 1
                a_n = -1 * self.mat_r[:,k].reshape((self.mat_r.shape[0],1))
                a_tilde = self.mat_r - ((-1*a_n).T @ np.eye(self.mat_r.shape[0])).T
                a_tilde[:,k] = -1*np.ones((self.mat_r.shape[0],))
                
                if k >= len(self.mat_r) - 1:
                    print('Error: Payoff matrix singular')
                    break
            
            x_r = np.linalg.solve(a_tilde, a_n)
            y_r = np.copy(x_r)
            x_r[k] = 1 - np.delete(y_r, k, axis=0).sum()
            
            return x_r
        
        def col_pl():
            k = 0
            mat_c = self.mat_c.T
            a_n = -1*mat_c[:,k].reshape((mat_c.shape[0],1))
            a_tilde = mat_c - ((-1*a_n).T @ np.eye(mat_c.shape[0])).T
            a_tilde[:,k] = -1*np.ones((mat_c.shape[0],))
            
            while np.linalg.det(a_tilde) == 0:
                k += 1
                a_n = -1*mat_c[:,k].reshape((mat_c.shape[0],1))
                a_tilde = mat_c - ((-1*a_n).T @ np.eye(mat_c.shape[0])).T
                a_tilde[:,k] = -1*np.ones((mat_c.shape[0],))
                
                if k >= len(mat_c) - 1:
                    print('Error: Payoff matrix singular')
                    break
            
            x_c = np.linalg.solve(a_tilde, a_n)
            y_c = np.copy(x_c)
            x_c[k] = 1 - np.delete(y_c, k, axis=0).sum()
            
            return x_c
        
        print('Note: Output only valid post iterated deletion of dominated strategies:')
        
        self.mixed_neq = np.array([row_pl(), col_pl()])
        return self.mixed_neq
    
    def del_dom_str(self): # add setting for printing what is the dominating strategy
        mat_r, mat_c = self.mat_r, self.mat_c
        
        def row_pl(mat_r):
            support_r = np.unique( [np.argmax(mat_r[:,i]) for i in range(mat_r.shape[0])] )
            total_str = list(range(mat_r.shape[0]))
            dominated_str_r = [x for x in total_str if x not in support_r]
            
            if len(dominated_str_r) == 0:
                print('No dominated strategies remaining for row player.')
                return True
            else:
                return support_r, dominated_str_r
        
        def col_pl(mat_c):
            support_c = np.unique( [np.argmax(mat_c[i,:]) for i in range(mat_c.shape[1])] )
            total_str = list(range(mat_c.shape[1]))
            dominated_str_c = [x for x in total_str if x not in support_c]
            
            if len(dominated_str_c) == 0:
                print('No dominated strategies remaining for col player.')
                return True
            else:
                return support_c, dominated_str_c

        remaining_str_r = []
        remaining_str_c = []
        
        while row_pl(mat_r) != True or col_pl(mat_c) != True:
            if row_pl(mat_r) != True and col_pl(mat_c) != True:
                remaining_str_r.append(row_pl(mat_r)[0])
                remaining_str_c.append(col_pl(mat_c)[0])
    
                mat_r = np.delete(mat_r, row_pl(mat_r)[1], axis=0)
                mat_c = np.delete(mat_c, col_pl(mat_c)[1], axis=1)
            
            elif row_pl(mat_r) != True:
                remaining_str_r.append(row_pl(mat_r)[0])
                mat_r = np.delete(mat_r, row_pl(mat_r)[1], axis=0)
            
            elif col_pl(mat_c) != True:
                remaining_str_c.append(col_pl(mat_c)[0])
                mat_c = np.delete(mat_c, col_pl(mat_c)[1], axis=1)
        
        if len(remaining_str_r) != 0:
            self.mat_c = np.array( list(map(lambda x: mat_c[x,:], np.array(remaining_str_r)))[0] )
        else:
            print('Row player has no dominated strategies.')
        
        if len(remaining_str_c) != 0:
            self.mat_r = np.array( list(map(lambda x: mat_r[:,x], np.array(remaining_str_c)))[0] )
        else:
            print('Col player has no dominated strategies.')

        return self.mat_r, self.mat_c
