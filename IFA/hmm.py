# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 02:25:10 2014

@author: AlbertoEAF
"""

import numpy as np
import math as mt
import scipy.stats

def Gsample(mean,stddev):
    """ Returns a sample from a normal with parameters mean and stddev. """

    return np.random.standard_normal(1)*stddev + mean

def gauss_prob(x,mean,var):
    """Returns probability of sampling x from the gaussian"""
    return scipy.stats.norm(mean,var).pdf(x)


def unmix(G, Y):
    """ Unmixes the sound recordings with the unmixing matrix G and returns the estimated X values. """
    X = np.dot(G,Y)
    return X

def AbsTotalError(A,B):
    """ Returns the absolute elementwise error between 2 matrices. """
    return np.sum(np.abs(A-B))

def Calc_G(G,hmms,X):
    """Returns a new G matrix after update. Each column of X contain data from different sources for the same timestep """
    T=X.shape[1]
    sum=0
    for t in range(0,T):
        phi=Calc_phi(hmms,t,X[:,t])
        sum+=phi*X[:,t].T*G
    G=G+Eps*G-Eps*1/T*sum
    return G
    
def Calc_phi(hmms,t,X):
    """Calculates phi for X for particular timestep for all HMMs"""
    phi = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        phi[i] = np.sum(hmms[i].gamma[:,t]*(x[i]-hmms[i].mu_state[:])/hmms[i].var_state[:])

    return phi

S = 4 # states
T = 9 # Time samples
M = 2 # microphones
N = M # sources
Eps=0.00000000001 #learning rate for the G matrix

Y =  np.ones((M,T))



class HMM:
    def __init__(self, states, length):
        self.S   = states
        self.T = length
        
        # store the states of each node
        self.state = np.zeros(length,dtype=int)        
        
        # store mu and var for each state
        self.mu_state  = np.random.randn(states)*10
        self.var_state = np.random.gamma(1,1,states)
#        self.w_state =  np.random.gamma(1,1,())       
        
        self.alpha = np.empty((states, length))        
        self.beta  = np.ones((states, length))
        
        
        self.pi = np.ones(states)  / states # uniform prior   
        
        self.a = np.ones((states,states)) / states # rows: s', cols: s
      
    def _calc_alphas(self,x):
        # t=1 (t = 0)
        self.alpha[:,0] = np.multiply(self.pi, gauss_prob(x[0], self.mu_state, self.var_state))
        
        # t=2,...,T (t = 1,...,T-1)
        for t in range(1, self.T):
            for s in range(self.S):
                x_prob = gauss_prob(x[t], self.mu_state[s], self.var_state[s])
                self.alpha[s,t] = x_prob * np.dot(self.alpha[:,t-1], self.a[:,s])
                
    def _calc_betas(self,x):
        # t = T (t = T-1)
        # self.beta[:,self.T-1] = 1 # already done by the initialization
        
        # t = 1,...,T-1 (t = 0,...,T-2)
        for t in range(self.T-2, -1, -1):
            for s in range(self.S):
                self.beta[s,t] = np.dot(self.beta[:,t+1], np.multiply(gauss_prob(x[t+1], self.mu_state, self.var_state), self.a[s,:]))
                
    def _calc_gamma(self):
        self.gamma = np.multiply(self.alpha, self.beta)
        
    def xi(self, x, s_prime, s, t):
        return self.alpha[s_prime,t-1]*self.beta[s,t]*gauss_prob(x[t],self.mu_state[s],self.var_state[s])*self.a[s_prime,s]

    def update(self,x):
        """Updates a,mean and variance. x contains data only for particular source"""
        
        self._calc_alphas(x)
        self._calc_betas(x)
        
        self._calc_gamma()
        
        for s in range(self.S):
            sum_gamma = np.sum(self.gamma[s])
            
            self.mu_state[s] = np.dot(self.gamma[s], x) / sum_gamma
            
            self.var_state[s]= np.dot(self.gamma[s], (x-self.mu_state[s])**2) / sum_gamma
            
            for s_prime in range(self.S):
                #should for t-1 so from 0 to T-1 for denominator?????????? 
                self.a[s_prime,s]=np.sum(self.xi(x, s_prime, s, np.arange(self.T))) / np.sum(self.gamma[s_prime, np.arange(self.T-1)])

        self.pi = self.gamma[:,0]

    def likelihood(x):
        return np.sum(self.alpha[:,-1])

"""
    
G = np.ones((M,N))    
        
HMMs = []
for n in range(N):
    HMMs.append(HMM(S,T))
    
a = HMMs[0]

#for i in range(10):
#    print i
#    a._update_messages()

a._update_messages()





a._calc_xi()

s = np.arange(0, S)
t = np.arange(1, T)
c = np.ones((3,1))
d = np.arange(1,4)
e = (c.T*d).T
alpha = a.alpha
beta = a.beta

A = np.ones((S,T))
B = np.ones((S,T))
for i in range(S):
    for j in range(T):
        A[i,j] = i*j
for j in range(T):
    B[:,j] = j

s = np.arange(0, S)
t = np.arange(1, T)

C = A[:,t-1]*B[:,t]

g =  C * Gsample(s,s).reshape(4,1)

print C
print g

print np.tile(np.array(Gsample(s, 0)),(2,3)).shape
"""

a = HMM(S,T)
x = [ Gsample(0,2) for i in range(T) ]
#x = np.arange(T)
a.update(x)