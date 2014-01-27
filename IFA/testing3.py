import sys
import matplotlib.pyplot as plt
from hmm import *
import numpy as np

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

#x = np.array([ Gsample(0,4) for i in range(T) ])
x = np.asarray([ Gsample(2,2) for i in range(T/2) ] + [ Gsample(100,4) for i in range(T/2) ]) # requires even T
#x = GSeqSample(T, .999, w, mu_w, var_w) # persistence of 0.8 creates reasonably noticeable chains when w is close to uniform

iterations = 100

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
    print "a", a.a
#    print "a normalization", np.sum(a.a, axis=1)
    print "log-likelihood =", a.log_likelihood()
    log_likelihoods.append(a.log_likelihood())
    

plt.plot(log_likelihoods)
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood")