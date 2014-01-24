# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 02:25:10 2014

@author: AlbertoEAF
"""

import numpy as np
#import math as mt
import scipy.stats
import sys

def Gsample(mean,stddev):
    """ Returns a sample from a normal with parameters mean and stddev. """

    return float(np.random.standard_normal(1)*stddev + mean)

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

def Calc_G (G, hmms, X):
    """Returns a new G matrix after update. Each column of X contain data from different sources for the same timestep """
    T = X.shape[1]
    
    Sum=0    
    for t in range(T):
        phi = Calc_phi(hmms,t,X[:,t])
        result=np.reshape(phi,(phi.shape[0],1))*X[:,t].T
        Sum += np.dot(result,G)
    G += Eps*(G-Sum/T)
    return G
    
def Calc_phi(hmms,t,X):
    """Calculates phi for X for particular timestep for all HMMs"""
    phi = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        phi[i] = np.sum(hmms[i].gamma[:,t]*(X[i]-hmms[i].mu_states[:])/hmms[i].var_states[:])

        if np.isnan(phi[i]):
            print "phi[i]=nan", hmms[i].var_states[:], hmms[i].gamma[:,t]*(X[i]-hmms[i].mu_states[:])

    return phi

Eps=0.1 #learning rate for the G matrix

class HMM:
    def __init__(self, states, length, mu_init=None, var_init=None):
        self.S   = states
        self.T = length
        
        # store mu and var for each state
        self.mu_states  = np.random.randn(states)
        self.var_states = np.random.gamma(1,10,states)
        
        if mu_init!=None:
            self.mu_states = mu_init
        if var_init!=None:
            self.var_states = var_init
            
        if mu_init!=None and var_init!=None:
            print "Overriding random initialization..."
        
        self.alpha = np.empty((states, length))        
        self.beta  = np.ones((states, length))
        
        self.c = np.empty(length)
        
        self.pi = np.ones(states)  / states # uniform prior   
        
        self.a = np.ones((states,states)) / states # rows: s', cols: s
      
        self.xi_sum_t = np.empty((states,states), dtype=float)      
      
        self.s_range = np.arange(self.S)
      
    def _calc_alphas(self,x):
        # t=1 (t = 0)
        self.alpha[:,0] = self.pi * gauss_prob(x[0], self.mu_states, self.var_states)
        
        self.c[0] = np.sum(self.alpha[:,0])
        self.alpha[:,0] /= self.c[0]
        
        # t=2,...,T (t = 1,...,T-1)
        for t in range(1, self.T):
            for s in self.s_range:
                x_prob = gauss_prob(x[t], self.mu_states[s], self.var_states[s])
                self.alpha[s,t] = x_prob * np.dot(self.alpha[:,t-1], self.a[:,s])
                
            self.c[t] = np.sum(self.alpha[:,t])
            self.alpha[:,t] /= self.c[t]
                
    def _calc_betas(self,x):
        # t = T (t = T-1)
        # self.beta[:,self.T-1] = 1 # already done by the initialization
        
        # t = 1,...,T-1 (t = 0,...,T-2)
        for t in range(self.T-2, -1, -1):
            for s in self.s_range:
                self.beta[s,t] = np.dot(self.beta[:,t+1], gauss_prob(x[t+1], self.mu_states, self.var_states) * self.a[s,:])
            self.beta[:,t] /= self.c[t+1]
                
    def _calc_gamma(self):
        self.gamma = self.alpha * self.beta
        
    def _calc_xi(self, x):
        for s in self.s_range:        
            for s_prime in self.s_range:
                # the sum of xi_s's is the same as the unnormalized a_s's
                self.xi_sum_t[s_prime,s] = np.sum(self.xi(x, s_prime, s, np.arange(1,self.T))) 
    
    def xi(self, x, s_prime, s, t):
        """ IMPORTANT!: t>0 """           
        return self.alpha[s_prime,t-1]*self.beta[s,t]*gauss_prob(x[t],self.mu_states[s],self.var_states[s])*self.a[s_prime,s] / self.c[t]
        

    def update(self,x):
        """Updates a,mean and variance. x contains data only for particular source"""
        
        # E-step  
        
        self._calc_alphas(x)
        self._calc_betas(x)
        
        self._calc_gamma()
        self._calc_xi(x)
        
        # M-step        
        
        for s in self.s_range:
            sum_gamma_s = np.sum(self.gamma[s])
            
            self.mu_states[s] = np.dot(self.gamma[s], x) / sum_gamma_s
            
            self.var_states[s]= np.dot(self.gamma[s], np.power(x-self.mu_states[s],2)) / sum_gamma_s
               
        # the sum of xi_s's is the same as the unnormalized a_s's
        self.a = np.copy(self.xi_sum_t) # depending on the situation / upgrade to the final algorithm, copy may be avoided
        for s_prime     in self.s_range:    
            self.a[s_prime] /= np.sum(self.a[s_prime])
            
        
        self.pi = self.gamma[:,0] / np.sum(self.gamma[:,0])
        
        

    def likelihood(self):
        return np.prod(self.c)
        
    def log_likelihood(self):
        return np.sum(np.log(self.c))


def GSeqSample(T, persistence, w, mu, var):
    """ Generates T samples from a set of gaussians with probabilities w and parameters (mu,var). 
        Persistence is the probability of sampling the next sample from the same state. """
    w = np.asarray(w)    
    w /= np.sum(w) # to normalize, just in case

    x = []
    i = 0
    u_sample = np.random.rand(1)
    while i < T:
        if persistence > np.random.rand(1):
            u_sample = u_sample
        else:
            u_sample = np.random.rand(1)
        for g in range(len(w)):
            g_max = w[g]
            if u_sample < g_max:
                x.append(Gsample(mu[g],var[g]))
                i += 1
                break
    return np.array(x)
















import sys
import matplotlib.pyplot as plt

S = 4 # states
T = 100 # Time samples
M = 2 # microphones
N = M # sources


#Y =  np.ones((M,T))

#mu_init = np.array([0,10.,5])
#var_init = np.array([2,10.,12])

a = HMM(S,T)#, mu_init, var_init)
x = np.array([ Gsample(0,5) for i in range(T) ])




# probabilities of selecting each gaussian
w = [0.4, 0.4] 
mu_w = [0., 23.]
var_w = [2.2, 4.]

x = np.array([ Gsample(0,3) for i in range(T/2) ] + [ Gsample(50,4) for i in range(T/2) ]) # requires even T
#x = GSeqSample(T, 0.1, w, mu_w, var_w)

iterations = 20

log_likelihoods = []
for i in range(iterations):
    a.update(x)
    
    print "----------   --------"
#    print "alpha", a.alpha
#    print "beta", a.beta
#    print "gamma", a.gamma
    print "mu", a.mu_states
    print "var", a.var_states
    print "p(s)", np.sum(a.a, axis=0)/np.sum(a.a)
#    print "a", a.a
    print "a normalization", np.sum(a.a, axis=1)
    print "log-likelihood =", a.log_likelihood()
    log_likelihoods.append(a.log_likelihood())
    

plt.plot(log_likelihoods)
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood")
    

