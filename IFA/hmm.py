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
    
def Calc_phi(hmms,t,x):
    """Calculates phi for X for particular timestep for all HMMs"""
    phi=[0]*N
    for s in range(0,hmms[0].S):
        for i in range(0,N):
            phi[i]+=hmms[i].gamma[s,t]*((x[i]-hmms[i].mu_state[s])/hmms[i].var_state[s])

    return phi

S = 4 # states
T = 9 # Time samples
M = 2 # microphones
N = M # sources
Eps=0.001 #learning rate for the G matrix

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
        
        
        self.pi = np.ones((states,1))  / states # uniform prior   
        
        self.a = np.ones((states,states)) / states # rows: s', cols: s
      
    def _calc_alphas(self,x):
        # t=1 (0)
        for s in range(self.S): 
            self.alpha[s,0] = self.pi[s] * gauss_prob(x[0],self.mu_state[s],self.var_state[s])
            #self.alpha[s,0] = self.pi[s] * Gsample(self.mu_state[s],self.var_state[s])
        # t=2,...,T (1,...,T-1)
        for t in range(1, self.T):
            for s in range(self.S):
                #x_sample = Gsample(self.mu_state[s],self.var_state[s])
                x_sample=x[t]
                x_prob=gauss_prob(x[t],self.mu_state[s],self.var_state[s])
#                sum_transitions = 0        
#                for s_left in range(self.S):
#                    sum_transitions += self.alpha[s_left,t-1]*self.a[s_left,s]
#                self.alpha[s,t] = x_sample*sum_transitions
                s_left = np.arange(0,self.S)
                #self.alpha[s,t] = x_sample * np.dot(self.alpha[s_left,t-1], self.a[s_left,s])
                self.alpha[s,t] = x_prob * np.dot(self.alpha[s_left,t-1], self.a[s_left,s])
                
    def _calc_betas(self,x):
        # t = T (T-1)
        # The beta message is already 1 for any beta(s_T).
    
        # t = 1,...,T-1 (0,...,T-2)
        for t in range(self.T-2, -1, -1):
            for s in range(self.S):
#                self.beta[s,t] = 0
#                for s_right in range(self.S):
#                    self.beta[s,t] += self.beta[s_right,t+1]*Gsample(self.mu_state[s_right],self.var_state[s_right])*self.a[s,s_right]
                s_right = np.arange(0,self.S)
                #self.alpha[s,t] = np.sum(self.beta[s_right,t+1]*Gsample(self.mu_state[s_right],self.var_state[s_right])*self.a[s,s_right])
                self.beta[s,t] = np.sum(self.beta[s_right,t+1]*gauss_prob(x[t],self.mu_state[s_right],self.var_state[s_right])*self.a[s,s_right])
                
        #print self.beta
    def _update_messages(self,x):
        self._calc_alphas(x)
        self._calc_betas(x)
        
    def _calc_gamma(self):

        self.gamma = np.multiply(self.alpha,self.beta)
        #print self.gamma
        #this trick with arange won't work:TypeError: only integer arrays with one element can be converted to an index
        
    def _calc_xi(self,x):
        
        #s = np.arange(0, self.S)
        #s_prime = s        
        #t = np.arange(1, self.T)
        #return self.alpha[s,t-1]*Gsample(self.mu_state[s], self.var_state[s]) * self.a[s_prime, s] * self.beta[s,t]
        #res = self.alpha[:,t-1]*self.beta[:,t] * Gsample(self.mu_state[s], self.var_state[s]).reshape(self.S,1) #* self.a[s_prime, s]  
        res=np.zeros((self.S,self.S,self.T))
        for s in range(self.S):
            for s_prime in range(self.S):
                for t in range(self.T):
                    res[s_prime,s,t]=self.alpha[s_prime,t]*self.beta[s,t]*gauss_prob(x[t],self.mu_state[s],self.var_state[s])*self.a[s_prime,s]
                    #res[s_prime,s,t]=self.alpha[s_prime,t]*self.beta[s,t]*Gsample(self.mu_state[s],self.var_state[s])*self.a[s_prime,s]

        #print res, res.shape
        self.xi=res
        return res

    def _calc_gauss_param(self,x):
        """Updates a,mean and variance. x contains data only for particular source"""
        
        self._update_messages(x)
        self._calc_gamma()
        self._calc_xi(x)
        
        for s in range(0,S):
            numerator=0;
            for t in range(0,T):
                numerator+=np.sum(self.gamma[s,t]*x[t])
            denominator=np.sum(self.gamma[s,:])
            self.mu_state[s]=numerator/denominator
            
            sum=0
            for tt in range(0,T):
                sum+=self.gamma[s,tt]*mt.pow((x[t]-self.mu_state[s]),2)
            self.var_state[s]=sum/denominator
            
            for s_prime in range(0,S):
                numerator=0
                denominator=0
                for t in range(0,T):
                    numerator+=self.xi[s_prime,s,t]
                    denominator+=self.gamma[s_prime,t] #should for t-1 so from 0 to T-1 for denominator?????????? 
            self.a[s_prime,s]=numerator/denominator
        #print self.a
        
    def _mu(self):
        pass

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
