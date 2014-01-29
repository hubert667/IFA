# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:22:18 2014

@author: AlbertoEAF
"""

from hmm import *
import numpy as np
import matplotlib.pyplot as plt

S = 2 # states
T = 1000# Time samples
M = 2 # microphones
N = M # sources

#example of H matrix
H=np.identity(N)
H[0,1]=0.5
H[1,0]=0.25
H /= np.linalg.norm(H)

G =  np.random.random((N, N))

        
HMMs = [ HMM(S,T) for n in range(N) ]


mean1=0.
var1=6.
mean2=0.
var2=0.5


HMMs[0].gamma[:,:] = 1/float(S)
HMMs[0].mu_states[:] = mean1
HMMs[0].var_states[:] = var1


HMMs[1].gamma[:,:] = 1/float(S)
HMMs[1].mu_states[:] = mean2
HMMs[1].var_states[:] = var2



HMMs[0].gamma[0,0] = 0.75
HMMs[0].gamma[1,0] = 0.25
HMMs[0].mu_states[1] = 0.3
HMMs[0].var_states[1] = 0.4


HMMs[1].gamma[0,0] = 0.75
HMMs[1].gamma[1,0] = 0.25
HMMs[1].mu_states[1] = 0.2
HMMs[1].var_states[1] = 0.6

HMMs[0].p()
HMMs[1].p()

x = np.array([0.2,0.9])

