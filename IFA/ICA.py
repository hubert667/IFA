"""
Created on Wed Jan 29 15:12:33 2014

@author: AlbertoEAF
"""

#from hmm import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sys
from hmm import *

def e(i):
    sys.exit(i)

activation_functions = [lambda a: -np.tanh(a),    lambda a: -a + np.tanh(a),                 lambda a: -a ** 3.0,             lambda a: - (6 * a) / ( a ** 2.0 + 5.0)]
p_from_act_functions = [lambda a: 1 / np.cosh(a), lambda a: np.cosh(a) * np.exp(-0.5 * a ** 2.), lambda a: np.exp(-0.25 * a ** 4.), lambda a: (a ** 2. + 5.0) ** (-3.)     ]





def ICA(X, W0, activation_function=lambda a: -np.tanh(a), learning_rate=0.01, max_iterations = 1000000):
   
    W = 0.01*np.eye(X.shape[0]) # 1e-2 * np.random.rand(X.shape[0],X.shape[0])
    
    error = [np.inf]  
  
    for i in range(max_iterations):
        A  = W.dot(X)
        Z  = activation_function(A)
        Xp = W.T.dot(A)

        dW = W + Z.dot(Xp.T) / X.shape[1]
        W += learning_rate * dW
       
        Wsum = np.absolute(dW).sum()
              
       
        print W/W[0,0]
        
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

