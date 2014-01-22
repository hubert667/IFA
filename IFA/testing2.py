# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:58:25 2014

@author: AlbertoEAF
"""

from hmm import *
import numpy as np
import sys


S = 3 # states
T = 2000 # Time samples
M = 2 # microphones
N = M # sources


#Y =  np.ones((M,T))

a = HMM(S,T)
x = [ Gsample(20,0.5) for i in range(T) ]
#x = [ Gsample(0,4) for i in range(T/2) ] + [ Gsample(20,4) for i in range(T/2) ] # requires even T


iterations = 10


for i in range(iterations):
    a.update(x)
    print "------------------"
    print "alpha", a.alpha
    print "beta", a.beta
    print "gamma", a.gamma
    print "mu", a.mu_states
    print "var", a.var_states
    print "a", a.a
    

