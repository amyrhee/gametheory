#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script is meant to test if other scripts are working. 

@author: amyrhee@nyu.edu

"""

import numpy as np
import bimatrix 
import sim_mat

class test:
    def __init__(self, n_str=10):
        self.n_str = n_str
    
    def default_test(self):
        mat_r = np.array([[10, 0, 0], [6, 1, -1], [12, -1, 5]])
        mat_c = np.array([[10, 6, 12], [0, 5, -1], [0, -1, 1]])
        
        gamma = bimatrix.bimatrix(mat_r, mat_c)
        
        self.default_test_result = gamma.pure_ne() # Strategies: (1,1), (2,2), Payoffs: (1,5), (5,1) 
        return print('Strategies: ', gamma.pure_ne(), 'Payoffs: ', gamma.pure_ne_payoffs())
    
    def sim_test(self, n_runs, dist='normal', lambda_p=1, mu=0, sigma=1, a=0, b=1):
        
        if dist == 'poisson' or 'Poisson' or 'poisson_dist':
            for i in range(n_runs):
                mat_r, mat_c = sim_mat.sim_mat(self.n_str).poisson_dist(lambda_p)
                
                g2 = bimatrix.bimatrix(mat_r, mat_c)
                if type(g2.pure_ne()) == str:
                    continue
    
                else:
                    print('Found pure NE on iteration', i,'. ',
                          'The pure NE strategy profile(s): ', 
                          g2.pure_ne(), 'and payoffs ', g2.pure_ne_payoffs())
                
        elif dist == 'normal' or 'Normal' or 'normal_dist':
            for i in range(n_runs):
                mat_r, mat_c = sim_mat.sim_mat(self.n_str).normal_dist(mu, sigma)
                
                g2 = bimatrix.bimatrix(mat_r, mat_c)
                if type(g2.pure_ne()) == str:
                    continue
    
                else:
                    print('Found pure NE on iteration', i,'. ',
                          'The pure NE strategy profile(s): ', 
                          g2.pure_ne(), 'and payoffs ', g2.pure_ne_payoffs())
        
        elif dist == 'exponential' or 'exp' or 'exp_dist':
            for i in range(n_runs):
                mat_r, mat_c = sim_mat.sim_mat(self.n_str).exp_dist(lambda_p)
                
                g2 = bimatrix.bimatrix(mat_r, mat_c)
                if type(g2.pure_ne()) == str:
                    continue
    
                else:
                    print('Found pure NE on iteration', i,'. ',
                          'The pure NE strategy profile(s): ', 
                          g2.pure_ne(), 'and payoffs ', g2.pure_ne_payoffs())
        elif dist == 'uniform' or 'unif' or 'unif_dist':
            for i in range(n_runs):
                mat_r, mat_c = sim_mat.sim_mat(self.n_str).unif_dist(a, b)
                
                g2 = bimatrix.bimatrix(mat_r, mat_c)
                if type(g2.pure_ne()) == str:
                    continue
    
                else:
                    print('Found pure NE on iteration', i,'. ',
                          'The pure NE strategy profile(s): ', 
                          g2.pure_ne(), 'and payoffs ', g2.pure_ne_payoffs())


