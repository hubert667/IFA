# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 19:50:09 2014

@author: AlbertoEAF
"""


import numpy as np
#import math as mt
import scipy.stats
import sys



def Gsample(mean,stddev):
    """ Returns a sample from a normal with parameters mean and stddev. """

    #return float(np.random.standard_normal(1)*stddev + mean)
    return np.random.standard_normal(1)*stddev + mean

def gauss_prob(x,mean,var):
    """Returns probability of sampling x from the gaussian"""
    return scipy.stats.norm(mean,var).pdf(x)

def logsumexp(a):
    B = np.max(a)
    
    return B + np.log(np.sum(np.exp(a-B)))

def e(s=1):
    sys.exit(s)

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
        
        self.log_alpha = np.empty((self.S, self.T))        
        self.log_beta  = np.zeros((self.S, self.T))
        
        
        
        self.pi = np.ones(self.S)  / self.S # uniform prior   
        
        self.A = np.ones((self.S,self.S)) / self.S # rows: s', cols: s
      
        self.xi_sum_t = np.empty((self.S,self.S), dtype=float)      
      
        self.s_range = np.arange(self.S)
      
    def _calc_alphas(self,x):
        # t=1 (t = 0)
        self.log_alpha[:,0] = np.log(self.pi) + np.log( gauss_prob(x[0], self.mu_states, self.var_states) )
        
        # t=2,...,T (t = 1,...,T-1)
        for t in range(1, self.T):
            for s in self.s_range:
                self.log_alpha[s,t] = np.log(gauss_prob(x[t],self.mu_states[s],self.var_states[s])) + logsumexp(self.log_alpha[:,t-1] + np.log(self.A[:,s]))
        
                
    def _calc_betas(self,x):
        # t = T (t = T-1)
        # self.log_beta[:,self.T-1] = 0 # already done by the initialization
        
        # t = 1,...,T-1 (t = 0,...,T-2)
        for t in range(self.T-2, -1, -1):
            for s in self.s_range:
                self.log_beta[s,t] = logsumexp(self.log_beta[:,t+1] + np.log(gauss_prob(x[t+1],self.mu_states,self.var_states)) + np.log(self.A[s]))
            
            
        
        
    def _calc_gamma(self):
        self.log_gamma = self.log_alpha + self.log_beta
        
    def _calc_xi_sum_t(self, x):
        for s in self.s_range:        
            for s_prime in self.s_range:
                # the sum of xi_s's is the same as the unnormalized a_s's
                self.xi_sum_t[s_prime,s] = np.exp(logsumexp(self.log_xi(x, s_prime, s, np.arange(1,self.T))))
    
    def log_xi(self, x, s_prime, s, t):
        """ IMPORTANT!: t>0 """           
        return self.log_alpha[s_prime,t-1] + self.log_beta[s,t] + np.log(gauss_prob(x[t],self.mu_states[s],self.var_states[s])) + np.log(self.A[s_prime,s])
        

    def update(self,x):
        """Updates a,mean and variance. x contains data only for particular source"""
        
        # E-step  
        
        self._calc_alphas(x)
        self._calc_betas(x)
        
        self._calc_gamma()
        self._calc_xi_sum_t(x) # only the sum of xi for all time is relevant
        
        # M-step        
        
        for s in self.s_range:
            sum_gamma_s = np.sum(np.exp(self.log_gamma[s])) # with np.exp(logsumexp(self.log_gamma[s])) doesn't work either
            print "IT IS 0!!! ", sum_gamma_s
            self.mu_states[s] = np.dot(np.exp(self.log_gamma[s]), x) / sum_gamma_s
            
            self.var_states[s]= np.dot(np.exp(self.log_gamma[s]), np.power(x-self.mu_states[s],2)) / sum_gamma_s
               
        # the sum of xi_s's is the same as the unnormalized a_s's
        self.A = np.copy(self.xi_sum_t) # depending on the situation / upgrade to the final algorithm, copy may be avoided
        for s_prime in self.s_range:    
            self.A[s_prime] /= np.sum(self.A[s_prime])
            
        
        self.pi = np.exp(self.log_gamma[:,0]) / np.sum(np.exp(self.log_gamma[:,0]))
        
        
        
    def log_likelihood(self):
        return logsumexp(self.log_alpha[:,-1])


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
    














import sys
import matplotlib.pyplot as plt

S = 4 # states
T = 1000 # Time samples
M = 2 # microphones
N = M # sources


#Y =  np.ones((M,T))

#mu_init = np.array([0,10.,5])
#var_init = np.array([2,10.,12])

a = HMM(S,T)#, mu_init, var_init)





# probabilities of selecting each gaussian
w = [.4, 0.4] 
mu_w = [0., 50.]
var_w = [3., 4.]

x = np.array([ Gsample(0,5) for i in range(T) ])
#x = np.asarray([ Gsample(20.,3) for i in range(T/2) ] + [ Gsample(50,4) for i in range(T/2) ]) # requires even T
#x = GSeqSample(T, .999, w, mu_w, var_w) # persistence of 0.8 creates reasonably noticeable chains when w is close to uniform

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
    print "p(s)", np.sum(a.A, axis=0)/np.sum(a.A)
#    print "a", a.a
#    print "a normalization", np.sum(a.a, axis=1)
    print "log-likelihood =", a.log_likelihood()
    log_likelihoods.append(a.log_likelihood())
    

plt.plot(log_likelihoods)
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood")
    

