#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script allows you to generate an (n x n) matrix with payoffs drawn from a specified distribution. 
It is meant to be used in tandem with the bimatrix script, which computes solutions to (n x n) 2 player games. 

@author: amyrhee@nyu.edu
"""

from numpy import random

class sim_mat:
    def __init__(self, n_str):
        self.n_str = n_str
        
    def poisson_dist(self, lambda_p=1):
        mat_r = random.poisson(lambda_p, (self.n_str, self.n_str))
        mat_c = random.poisson(lambda_p, (self.n_str, self.n_str))
        self.mat_r = mat_r
        self.mat_c = mat_c
        return mat_r, mat_c
    
    def normal_dist(self, mu=0, sigma=1):
        mat_r = random.normal(mu, sigma, (self.n_str, self.n_str))
        mat_c = random.normal(mu, sigma, (self.n_str, self.n_str))
        self.mat_r = mat_r
        self.mat_c = mat_c
        return mat_r, mat_c
    
    def exp_dist(self, lambda_p=1):
        mat_r = random.exponential(lambda_p, (self.n_str, self.n_str))
        mat_c = random.exponential(lambda_p, (self.n_str, self.n_str))
        self.mat_r = mat_r
        self.mat_c = mat_c
        return mat_r, mat_c
    
    def unif_dist(self, a=0, b=1):
        mat_r = random.uniform(a, b, (self.n_str, self.n_str))
        mat_c = random.uniform(a, b, (self.n_str, self.n_str))
        self.mat_r = mat_r
        self.mat_c = mat_c
        return mat_r, mat_c 

