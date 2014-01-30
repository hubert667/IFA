# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:12:33 2014

@author: AlbertoEAF
"""

from hmm import *
import numpy as np
import matplotlib.pyplot as plt

S = 2 # states
T = 500# Time samples
M = 2 # microphones
N = M # sources

#example of H matrix
H=np.identity(N)
H[0,1]=0.5
H[1,0]=0.25
H /= np.linalg.norm(H)
#H^-1=[1.73,-0.86;-0.43,1,73]

#G =  np.random.random((N, N))
G=  np.identity((N))
        
HMMs = [ HMM(S,T) for n in range(N) ]
realHMMs = [ HMM(S,T) for n in range(N) ]

Eps=0.02 #learning rate for the G matrix

mean1=0.
var1=6.
mean2=0.
var2=0.5


HMMs[0].gamma[:,0] = 1/float(S)
HMMs[0].mu_states = np.array([0., 1.2])
HMMs[0].var_states = np.array([0.3, 2.])


HMMs[1].gamma[:,:] = 1/float(S)
HMMs[1].mu_states = np.array([0., 2.])
HMMs[1].var_states = np.array([0.6, 5.])


#original sources
x0=np.zeros((N,T))
HMMs[0].gamma[:,:] = 0.
HMMs[1].gamma[:,:] = 0.
for t in range(T):
    g = int(np.random.rand(1)>0.5)
    HMMs[0].gamma[g,t] = 1.
    HMMs[1].gamma[g,t] = 1.
    x0[0,t] = Gsample(HMMs[0].mu_states[g],np.sqrt(HMMs[0].var_states[g]))
    x0[1,t] = Gsample(HMMs[1].mu_states[g],np.sqrt(HMMs[1].var_states[g]))

#mixing
y = np.dot(H, x0)
        

iterations=300
egs =  [np.inf]
negs = [np.inf]
increased_eG = False
increased_NeG = False
for itM in range(iterations):
    
    HMMs[0].gamma[:,:] = 0.
    HMMs[1].gamma[:,:] = 0.
    for t in range(T):
        g = int(np.random.rand(1)>0.5)
        HMMs[0].gamma[g,t] = 1.
        HMMs[1].gamma[g,t] = 1.
        x0[0,t] = Gsample(HMMs[0].mu_states[g],np.sqrt(HMMs[0].var_states[g]))
        x0[1,t] = Gsample(HMMs[1].mu_states[g],np.sqrt(HMMs[1].var_states[g]))
    y = np.dot(H, x0)
    
    x = unmix(G, y)  
    for i in range(len(HMMs)):
        realHMMs[i].update(x[i])
    G = Calc_G(G,realHMMs,x, Eps)
    #G = Calc_G(G,HMMs,x, Eps)

    print "-------------------"
    print "G:", G/G[0,0]
    eG = G - np.linalg.inv(H)
    egs.append(np.linalg.norm(eG))
    NeG = G/np.linalg.norm(G) - np.linalg.inv(H)/np.linalg.norm(np.linalg.inv(H))    
    negs.append(np.linalg.norm(NeG))
    print "eG", eG, egs[-1]
    print "NeG", NeG, negs[-1]
    for hmm_i in range(len(HMMs)):
        print "mu" ,   realHMMs[hmm_i].mu_states
        print "var",   realHMMs[hmm_i].var_states
        #print "LogLs", HMMs[hmm_i].log_likelihood()
        #HMMs[hmm_i].log_likelihood_check()
        
    if egs[-1] > egs[-2] and len(egs)>5: # sometimes fails right at the first step, does it fail after? yes and probably when it fails after it would fail at the beginning as well.
        increased_eG = True
        
    if negs[-1] > negs[-2] and len(negs)>5:
        increased_NeG = True
        
    
if increased_eG:  print "Error in G increased."
if increased_NeG: print "Error in normalized G increased."
    
plt.plot(egs[1:])
#plt.plot(negs[1:])
