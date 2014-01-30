# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:12:33 2014

@author: AlbertoEAF
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sys

activation_functions = [lambda a: -np.tanh(a),    lambda a: -a + np.tanh(a),                 lambda a: -a ** 3.0,             lambda a: - (6 * a) / ( a ** 2.0 + 5.0)]
p_from_act_functions = [lambda a: 1 / np.cosh(a), lambda a: np.cosh(a) * np.exp(-0.5 * a ** 2.), lambda a: np.exp(-0.25 * a ** 4.), lambda a: (a ** 2. + 5.0) ** (-3.)     ]

def dAmari(W,W0):
    """ 
    Returns the Amari distance between two permutation-invariant matrices and row-scale invariant. 
    W0 is the perfect unmixing matrix and W the estimated one. 
    """
    r = np.abs(np.dot(W0,np.linalg.inv(W))) # L1 element-wise norm
        
    d1 = np.sum(np.sum(r,axis=1)/np.max(r,axis=1) - 1)
    d2 = np.sum(np.sum(r,axis=0)/np.max(r,axis=0) - 1)
    
    return 1/(2*float(r.shape[0])) * (d1 + d2)


def GetData(filepath, start=0, size=100, plot_hist=1, hist_bins=50):
    """ Gets normalized amplitude data from a .wav file. """
    sample_rate, wave_data = scipy.io.wavfile.read(filepath)
    print "File %s has %d samples at a sample rate of %d Hz" % (filepath, len(wave_data), sample_rate)

    if start+size > len(wave_data):
        print "File doesn't contain that many samples!"
        sys.exit(3)
    
    wave_data = wave_data[start:start+size]
    
    # Hist can only draw int values
    if plot_hist:
        plt.figure()
        plt.hist(wave_data, bins = hist_bins, alpha=1.0)  # no need to set lower alphas  
        plt.xlabel("x(t) unnormalized")

    if np.sum(wave_data) == 0:
        print "No data in this section!"
        sys.exit(2)
    
    wave_data = wave_data.astype(float)
    #wave_data/=np.max(np.abs(wave_data))

    return wave_data


def ICA(X, W0, activation_function=lambda a: -np.tanh(a), learning_rate=0.01, max_iterations = 10000):
   
    W = 1e-2 * np.eye(X.shape[0]) # 1e-2 * np.random.rand(X.shape[0],X.shape[0])
    
    error = [np.inf]  
  
    for i in range(max_iterations):
        A  = W.dot(X)
        
        Z  = activation_function(A)
        #print Z
        Xp = W.T.dot(A)

        dW = W + Z.dot(Xp.T) / X.shape[1]
        W += learning_rate * dW
       
        Wsum = np.absolute(dW).sum()
       
        #print W/np.sum(W,axis=0)
        
        error.append(dAmari(W,W0))

        if error[-1] - error[-2] > 0.1:
            print "Error increasing!"
            break       
        if np.isnan(Wsum) or np.isinf(Wsum):
            print "Failed convergence!"
            break
        if np.linalg.norm(dW) < 1e-5:
            print "Early solution! :)"
            break
 
    return W, error[1:]

S = 2 # states
T = 150000# Time samples
M = 2 # microphones
N = M # sources
      
iterations=5000

H = np.eye(N)
#H[1,0] = 0.34
#H[1,1] = 0.2
#H[1,0] = 0.3
X = np.empty((N,T))
X[0] = GetData("mike.wav", 6000, T)
X[1] = GetData("beet.wav", 6000, T)
y = np.dot(H,X)

G,eG = ICA(y, np.linalg.inv(H))#, activation_functions[3])

plt.figure()
plt.plot(eG)
plt.xlabel("Iterations")
plt.ylabel("G error")

print G/np.sum(G,axis=1)
