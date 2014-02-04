# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 02:25:10 2014

@author: AlbertoEAF
"""

import numpy as np
import scipy.stats
import sys
import random

def e(s=1):
    sys.exit(s)

def Gsample(mean,stddev):
    """ Returns a sample from a normal with parameters mean and stddev. """

    #return np.random.standard_normal(1)*stddev + mean
    return random.gauss(mean,stddev)

def gauss_prob(x,mean,variance):
    """Returns probability of sampling x from the gaussian"""
    std_dev=np.sqrt(variance)
    return scipy.stats.norm(mean,std_dev).pdf(x)


def unmix(G, Y):
    """ Unmixes the sound recordings with the unmixing matrix G and returns the estimated X values. """
    X = np.dot(G,Y)
    return X

def AbsTotalError(A,B):
    """ Returns the absolute elementwise error between 2 matrices. """
    return np.sum(np.abs(A-B))

def Calc_G (G, hmms, X, epsilon=0.1):
    """Returns a new G matrix after update. Each column of X contain data from different sources for the same timestep """
    N = X.shape[0]        
    T = X.shape[1]
        
    psi=np.zeros((N,N))
    for t in range(T):
        phi = Calc_phi(hmms,t,X[:,t])
        #phi = Calc_phi_other_way(hmms,t,X[:,t])
        psi += phi.reshape(N,1)*X[:,t].T
    return G + epsilon*(G-np.dot(psi,G)/T) 
    
def Calc_phi(hmms,t,x):
    """Calculates phi for x for particular timestep for all HMMs"""
    phi = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        phi[i] = np.sum(hmms[i].gamma[:,t]*(x[i]-hmms[i].mu_states)/hmms[i].var_states)

        if np.isnan(phi[i]):
            print "phi[%d]=nan" % i, hmms[i].var_states, hmms[i].mu_states, hmms[i].gamma[:,t], hmms[i].gamma[:,t]*(x[i]-hmms[i].mu_states[:])
            e(i)

    return phi

def Calc_phi_other_way(hmms,t,X):
    phi = np.zeros(X.shape[0])
    
    phi=np.tanh(X) # positive since our formula has a different sign than the original ICA
    return phi



class HMM:
    def __init__(self, states, length, mu_init=None, var_init=None):
        self.S   = states
        self.T = length
        
        # store mu and var for each state
        self.mu_states  = np.random.randn(states)
        self.var_states = np.random.gamma(1,10,states)
        self.last_log=-10000000000000
        
        if mu_init!=None:
            self.mu_states = mu_init
        if var_init!=None:
            self.var_states = var_init
            
        if mu_init!=None and var_init!=None:
            print "Overriding random initialization..."
        
        self.alpha = np.empty((states, length))        
        self.beta  = np.ones ((states, length))
        self.gamma = np.empty((states, length))        
        
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
            #self.mu_states[s]=0
            
            self.var_states[s]= np.dot(self.gamma[s], np.power(x-self.mu_states[s],2)) / sum_gamma_s
               
        # the sum of xi_s's is the same as the unnormalized a_s's
        self.a = np.copy(self.xi_sum_t) # depending on the situation / upgrade to the final algorithm, copy may be avoided
        for s_prime     in self.s_range:   
            self.a[s_prime] /= np.sum(self.gamma[s_prime,:-1]) 
            self.a[s_prime] /= np.sum(self.a[s_prime])
            
        
        self.pi = self.gamma[:,0] / np.sum(self.gamma[:,0])
        
        

    def likelihood(self):
        return np.prod(self.c)
    
    def log_likelihood(self):
        return np.sum(np.log(self.c))    
    
    def log_likelihood_check(self):
        like=np.sum(np.log(self.c))
        if like<self.last_log:
            print "Error-likelihood goes down"
            sys.exit(0)
        elif like-self.last_log<1e-5:
            print "Likelihood is not changing"
            sys.exit(0)
        self.last_log=like
        return like
    def p(self):
        print "mu:",self.mu_states,"var:",self.var_states,"gamma:"
        print self.gamma


def discretePDF2CDF(w):
    for i in range(1,w.size):
        w[i] += w[i-1]
    return w/w[-1] # normalizes in case the input wasn't



def discreteInvCDF(CDF,x):
    " Returns the class of the CDF to which x belongs. "
    for i in range(CDF.size):
        if x < CDF[i]:
            return i

def GSeqSample(T, persistence, w, mu, var):
    """ Generates T samples from a set of gaussians with probabilities w and parameters (mu,var). 
        Persistence is the probability of sampling the next sample from the same state. """
    mu = np.array(mu); var = np.asarray(var)
    
    w = discretePDF2CDF(np.asarray(w)) # and normalizes
    
    # Uniform samples i.i.d. in [0,1[.
    u_samples = np.random.rand(T) 
    # Array that stores which gaussians get chosen.
    g = [discreteInvCDF(w, u_samples[0])]
    
    for i in range(1,T):
        if persistence > np.random.rand(1):
            g.append(g[-1]) # Mantain the state.
        else:
            g.append(discreteInvCDF(w, u_samples[i])) # Sample from a new normal chosen through the probabilities/weights w.
        
    return Gsample(mu[g], var[g])  
    


