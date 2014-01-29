# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:12:33 2014

@author: AlbertoEAF
"""

from hmm import *
import numpy as np
import matplotlib.pyplot as plt

def ICA(Y, hmms, learning_rate=0.01, max_iterations = 100000):
   
    N = Y.shape[0]   
    T = Y.shape[1]
   
    W = 0.01 * numpy.random.rand(N,N)

    Z = np.empty((N,T))  
    for i in range(max_iterations):
        X  = W.dot(Y)
        #Z  = activation_function(X)
        for t in range(T):
            Z[:,t] = Calc_phi(hmms, t, X[:,t])
        
        
        Yp = W.T.dot(X)

        dW = W - Z.dot(Yp.T) / T
        W += learning_rate * dW
       
        Wsum = numpy.absolute(dW).sum()

        print W
        print np.linalg.norm(dW)       
       
        print "E:", np.linalg.norm(W-np.eye(N))       
       
        if numpy.isnan(Wsum) or numpy.isinf(Wsum):
            print "Failed convergence!"
            break
        if np.linalg.norm(dW) < 1e-5:
            print "Early solution! :)"
            break
 
    return W


S = 2 # states
T = 1000# Time samples
M = 2 # microphones
N = M # sources

#example of H matrix
H=np.identity(N)
#H[0,1]=0.5
#H[1,0]=0.25
#H /= np.linalg.norm(H)

G =  np.random.random((N, N))

        
HMMs = [ HMM(S,T) for n in range(N) ]

Eps=0.1 #learning rate for the G matrix

mean1=0.
var1=6.
mean2=0.
var2=0.5


HMMs[0].gamma[:,0] = 1/float(S)
HMMs[0].mu_states = np.array([0., 1.2])
HMMs[0].var_states = np.array([0.3, 20.])


HMMs[1].gamma[:,:] = 1/float(S)
HMMs[1].mu_states = np.array([0., 2.])
HMMs[1].var_states = np.array([0.6, 50.])


#original sources
x0=np.zeros((N,T))
HMMs[0].gamma[:,:] = 0.
HMMs[1].gamma[:,:] = 0.
for t in range(T):
    g = int(np.random.rand(1)>0.5)
    HMMs[0].gamma[g,t] = 1.
    HMMs[1].gamma[g,t] = 1.
    x0[0,t] = Gsample(HMMs[0].mu_states[g],sqrt(HMMs[0].var_states[g]))
    x0[1,t] = Gsample(HMMs[1].mu_states[g],sqrt(HMMs[1].var_states[g]))

#mixing
y = np.dot(H, x0)
        

iterations=5000
egs =  [np.inf]
negs = [np.inf]
increased_eG = False
increased_NeG = False
print "Running..."

G = ICA(y, HMMs)

print G

e(0)

for itM in range(iterations):
    x = unmix(G, y)  
#    for i in range(len(HMMs)):
#        HMMs[i].update(x[i])
    G = Calc_G(G,HMMs,x, Eps)

    #print "-------------------"
    #print "G:", G/G[0,0]
    eG = G - np.linalg.inv(H)
    egs.append(np.linalg.norm(eG))
    #NeG = G/np.linalg.norm(G) - np.linalg.inv(H)/np.linalg.norm(np.linalg.inv(H))    
    #negs.append(np.linalg.norm(NeG))
    #print "eG", eG, egs[-1]
    #print "NeG", NeG, negs[-1]
#    for hmm_i in range(len(HMMs)):
#        print "mu" ,   HMMs[hmm_i].mu_states
#        print "var",   HMMs[hmm_i].var_states
        #print "LogLs", HMMs[hmm_i].log_likelihood()
        #HMMs[hmm_i].log_likelihood_check()
        
    if egs[-1]-egs[-2] > 0.001:
        increased_eG = True
        break
        
        
    
if increased_eG:  print "Error in G increased."
#if increased_NeG: print "Error in normalized G increased."
    
plt.plot(egs[1:])
#plt.plot(negs[1:])
plt.xlabel("Iterations")
plt.ylabel("G error $| G - H^{-1}|$")