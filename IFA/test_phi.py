# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:22:18 2014

@author: AlbertoEAF
"""

from hmm import *
import numpy as np
import matplotlib.pyplot as plt

S = 2 # states
T = 3# Time samples
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

X = np.array([[0.2, 0.3, 0.4],[1.2,0.2,0.5]])

print
print "X:\n\r", X

Phi = np.ones((N,T))

for t in range(T):
    Phi[:,t] = Calc_phi(HMMs,t,X[:,t])

def Psi(Phi, t, X):
    Psi = np.ones((2,2))
    
    Psi[0,0] = Phi[0,t]*X[0,t]
    Psi[0,1] = Phi[0,t]*X[1,t]
    Psi[1,0] = Phi[1,t]*X[0,t]
    Psi[1,1] = Phi[1,t]*X[1,t]
    
    return Psi
    

print "\n\rPhi:\n\r", Phi

print 
psi = Psi(Phi,0,X)+Psi(Phi,1,X)+Psi(Phi,2,X)

print 0, Psi(Phi,0,X)
print 1, Psi(Phi,1,X)
print 2, Psi(Phi,2,X)

G = np.array([[0.1,0.3],[0.2,0.5]])

print psi
print G

print np.dot(psi,G)


calculated_G = Calc_G(G,HMMs,X)

print "Calc_G"
print calculated_G

print G + 0.1*(G-1/float(T)*np.dot(psi,G))