# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 02:25:10 2014

@author: AlbertoEAF
"""

import numpy as np
#import math as mt
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

def Calc_G (G, hmms, X):
    """Returns a new G matrix after update. Each column of X contain data from different sources for the same timestep """
    T = X.shape[1]
    
    Sum=0    
    for t in range(T):
        phi = Calc_phi(hmms,t,X[:,t])
        result=np.reshape(phi,(phi.shape[0],1))*X[:,t].T
        Sum += result*G
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

Eps=0.01 #learning rate for the G matrix

MIN_variance = 1e-30

class HMM:
    def __init__(self, states, length):
        self.S   = states
        self.T = length
        
        # store the states of each node
        self.states = np.zeros(length,dtype=int)        
        
        # store mu and var for each state
        self.mu_states  = np.random.randn(states)
        self.var_states = np.random.gamma(1,1,states)    
        
        self.alpha = np.empty((states, length))        
        self.beta  = np.ones((states, length))
        
        #self.beta[:, self.T-1] /= states # we assume beta should still be 1 even when rescaled - right or wrong?
        
        self.c = np.empty(length)
        
        self.pi = np.ones(states)  / states # uniform prior   
        
        self.a = np.ones((states,states)) / states # rows: s', cols: s
      
    def _calc_alphas(self,x):
        # t=1 (t = 0)
        self.alpha[:,0] = np.multiply(self.pi, gauss_prob(x[0], self.mu_states, self.var_states))
        
        self.c[0] = np.sum(self.alpha[:,0])
        self.alpha[:,0] /= self.c[0]
        
        # t=2,...,T (t = 1,...,T-1)
        for t in range(1, self.T):
            for s in range(self.S):
                x_prob = gauss_prob(x[t], self.mu_states[s], self.var_states[s])
                self.alpha[s,t] = x_prob * np.dot(self.alpha[:,t-1], self.a[:,s])
                
            self.c[t] = np.sum(self.alpha[:,t])
            self.alpha[:,t] /= self.c[t]
                
    def _calc_betas(self,x):
        # t = T (t = T-1)
        # self.beta[:,self.T-1] = 1 # already done by the initialization
        
        # t = 1,...,T-1 (t = 0,...,T-2)
        for t in range(self.T-2, -1, -1):
            for s in range(self.S):
                self.beta[s,t] = np.dot(self.beta[:,t+1], np.multiply(gauss_prob(x[t+1], self.mu_states, self.var_states), self.a[s,:]))
            self.beta[:,t] /= self.c[t+1]
                
    def _calc_gamma(self):
        self.gamma = np.multiply(self.alpha, self.beta)
        
    def xi(self, x, s_prime, s, t_):
        t_ = np.array(t_)
        xi = np.zeros(t_.size)
        for i in range(len(t_)):
            t = t_[i]
            assert t>0
            assert t < self.T
            assert s_prime < self.S and s < self.S
            xi[i] = self.alpha[s_prime,t-1]*self.beta[s,t]*gauss_prob(x[t],self.mu_states[s],self.var_states[s])*self.a[s_prime,s]
            xi[i] /= self.c[t]
        return xi

    def update(self,x):
        """Updates a,mean and variance. x contains data only for particular source"""
        
        self._calc_alphas(x)
        self._calc_betas(x)
        
        self._calc_gamma()
        
        sumA=[0]*self.S
        for s in range(self.S):
            sum_gamma = np.sum(self.gamma[s])
            
            self.mu_states[s] = np.dot(self.gamma[s], x) / sum_gamma
            
            self.var_states[s]= np.dot(self.gamma[s], (x-self.mu_states[s])**2) / sum_gamma
            # Let's update the variance but with a minimum threshold
            self.var_states = np.maximum(self.var_states, MIN_variance)            
            
            for s_prime in range(self.S):
                #should for t-1 so from 0 to T-1 for denominator?????????? 
                self.a[s_prime,s] = np.sum(self.xi(x, s_prime, s, np.arange(1,self.T))) / np.sum(self.gamma[s_prime, np.arange(self.T-1)])
                sumA[s_prime]+=self.a[s_prime,s]   
        for s_prim in range(self.S):    
            self.a[s_prim,:]/=sumA[s_prim]
        self.pi = self.gamma[:,0]/sum(self.pi)

    def likelihood(self):
        return np.prod(self.c)
