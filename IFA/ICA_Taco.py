# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:10:33 2014

@author: AlbertoEAF
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:43:26 2014

@author: AlbertoEAF
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:12:33 2014

@author: AlbertoEAF
"""

from hmm import *
import numpy as np
import matplotlib.pyplot as plt
import pylab as P
import scipy.io.wavfile

activation_functions = [lambda a: -tanh(a),    lambda a: -a + tanh(a),                 lambda a: -a ** 3,             lambda a: - (6 * a) / ( a ** 2 + 5.0)]
p_from_act_functions = [lambda a: 1 / cosh(a), lambda a: cosh(a) * exp(-0.5 * a ** 2), lambda a: exp(-0.25 * a ** 4), lambda a: (a ** 2 + 5.0) ** (-3)     ]

def GetData(filepath, start=0, size=100, plot_hist=1, hist_bins=50):
    """ Gets normalized amplitude data from a .wav file. """
    sample_rate, wave_data = scipy.io.wavfile.read(filepath)
    print "File %s has %d samples at a sample rate of %d Hz" % (filepath, len(wave_data), sample_rate)

    if start+size > len(wave_data):
        print "File doesn't contain that many samples!"
        e(3)
    
    wave_data = wave_data[start:start+size]
    
    # Hist can only draw int values
    if plot_hist:
        plt.figure()
        plt.hist(wave_data, bins = hist_bins, alpha=1.0)  # no need to set lower alphas  
        plt.xlabel("x(t) unnormalized")

    if np.sum(wave_data) == 0:
        print "No data in this section!"
        e(2)
    
    wave_data = wave_data.astype(float)
    wave_data/=np.max(wave_data)

    
    return wave_data


def ICA(X, perfectW, activation_function, learning_rate=0.01, max_iterations = 1000000):
   
    W = 0.01 * numpy.random.rand(X.shape[0],X.shape[0])
  
    error = [np.inf]  
  
    for i in range(max_iterations):
        A  = W.dot(X)
        Z  = activation_function(A)
        Xp = W.T.dot(A)

        dW = W + Z.dot(Xp.T) / X.shape[1]
        W += learning_rate * dW
       
        Wsum = np.absolute(dW).sum()
       
        print W
        
        error.append(np.linalg.norm(W-perfectW))     
        if error[-1] - error[-2] > 0.0001:
            print "Error increasing!"
            break       
        if numpy.isnan(Wsum) or numpy.isinf(Wsum):
            print "Failed convergence!"
            break
        if np.linalg.norm(dW) < 1e-5:
            print "Early solution! :)"
            break
 
    return W, error

S = 2 # states
T = 150000# Time samples
M = 2 # microphones
N = M # sources
       

iterations=5000
print "Running..."

H = np.eye(N)
X = np.empty((N,T))
X[0] = GetData("mike.wav", 6000, T)
X[1] = GetData("beet.wav", 6000, T)
y = np.dot(H,X)

G,eG = ICA(y, np.linalg.inv(H), activation_functions[1])

plt.figure()
plt.plot(eG)

plt.xlabel("Iterations")
plt.ylabel("G error $| G - H^{-1}|$")

print G

e(0)
