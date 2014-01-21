from hmm import *
import numpy as np


S = 2 # states
T = 1000# Time samples
M = 2 # microphones
N = M # sources

G = np.random.random((N, N))  
        
HMMs = []
for n in range(N):
    HMMs.append(HMM(S,T))
    

mean1=2
stddev1=4
mean2=0
stddev2=1

y=np.zeros((N,T))
for t in range(T):
    y[0,t]=Gsample(mean1,stddev1)
    y[1,t]=Gsample(mean2,stddev2)

#so it is like using I matrix as a mixing matrix

iterations=10
x=np.zeros((N,T))
for it in range(iterations):
    for t in range(T):
        x[:,t]=unmix(G, y[:,t])
        
    for i in range(len(HMMs)):
        HMMs[i].update(x[i,:])
    G = Calc_G(G,HMMs,x)

    print "-------------------"
    print G
    print HMMs[0].mu_states
    print HMMs[0].var_states
    print HMMs[1].mu_states
    print HMMs[1].var_states
#print HMMs[0].gamma

