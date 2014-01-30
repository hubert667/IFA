import sys
import matplotlib.pyplot as plt
from hmm import *
import numpy as np

S = 2 # states
T = 600 # Time samples
M = 2 # microphones
N = M # sources


#Y =  np.ones((M,T))

#mu_init = np.array([0,10.,5])
#var_init = np.array([2,10.,12])

a = HMM(S,T)#, mu_init, var_init)





# probabilities of selecting each gaussian
w = [0.5, 0.5] 
mu_w = [0., 2.]
var_w = [4., 2.]

#x = np.array([ Gsample(0,4) for i in range(T) ])
#x = np.asarray([ Gsample(2,2) for i in range(T/2) ] + [ Gsample(0,4) for i in range(T/2) ]) # requires even T
#x = GSeqSample(T, 0.99, w, mu_w, var_w) # persistence of 0.8 creates reasonably noticeable chains when w is close to uniform


mean1=0
stddev1=0.8
mean2=0.3
stddev2=1
x=np.zeros(T)
for t in range(T):
    val=np.random.rand(1)
    if val<0.5:
        x[t]=Gsample(mean1,np.sqrt(stddev1))
    else:
        x[t]=Gsample(mean2,np.sqrt(stddev2))

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
    like=a.log_likelihood()
    a.log_likelihood_check()
    print "log-likelihood =", like
    log_likelihoods.append(like)
    

plt.plot(log_likelihoods)
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood")