# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:12:33 2014

@author: AlbertoEAF
"""

from hmm import *
import numpy as np
import matplotlib.pyplot as plt

S = 2 # states
T = 600# Time samples
M = 2 # microphones
N = M # sources

#example of H matrix
H=np.identity(N)
H[0,1]=0.5
H[1,0]=0.5
H1=np.linalg.inv(H)
#H /= np.linalg.norm(H)
#H^-1=[1.73,-0.86;-0.43,1,73]
#H^-1=[1,-0.5;-0.5,1]

#G =  np.random.random((N, N))
G=  np.identity((N))
        
HMMs = [ HMM(S,T) for n in range(N) ]
realHMMs = [ HMM(S,T) for n in range(N) ]

Eps=0.05 #learning rate for the G matrix

mean1=0.
var1=6.
mean2=0.
var2=0.5


HMMs[0].gamma[:,0] = 1/float(S)
HMMs[0].mu_states = np.array([0., 0.3])
HMMs[0].var_states = np.array([0.8, 1])




HMMs[1].gamma[:,:] = 1/float(S)
HMMs[1].mu_states = np.array([0., 0.5])
HMMs[1].var_states = np.array([0.6, 0.3])


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

iterations=50
egs =  [np.inf]
negs = [np.inf]
increased_eG = False
increased_NeG = False
difs=[]
for itM in range(iterations):
    
    HMMs[0].gamma[:,:] = 0.
    HMMs[1].gamma[:,:] = 0.
    for t in range(T):
        g = int(np.random.rand(1)>0.5)
        HMMs[0].gamma[g,t] = 1.
        HMMs[1].gamma[g,t] = 1.
        #x0[0,t] = Gsample(HMMs[0].mu_states[g],np.sqrt(HMMs[0].var_states[g]))
            
    #y = np.dot(H, x0)
    
    x = unmix(G, y)  
    for i in range(len(HMMs)):
        realHMMs[i].update(x[i])
    G = Calc_G(G,realHMMs,x, Eps)
    #G/=np.max(G,axis=1)
    dif=dAmari(G,H1)
    difs.append(dif)
    print "-------------------progress:"+str(itM*100/iterations)
    print "G:", G/G[0,0]
    for hmm_i in range(len(HMMs)):
        print "mu" ,   realHMMs[hmm_i].mu_states
        print "var",   realHMMs[hmm_i].var_states
        #print "LogLs", HMMs[hmm_i].log_likelihood()
        #HMMs[hmm_i].log_likelihood_check()
        

    
plt.show(plt.plot(difs[1:]))
