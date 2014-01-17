from hmm import *
import numpy as np

G = np.random.random((N, N))  
        
HMMs = []
for n in range(N):
    HMMs.append(HMM(S,T))
    

mean1=1
stddev1=1
mean2=0
stddev2=0.5

y=np.zeros((N,T))
for t in range(0,T):
    y[0,t]=Gsample(mean1,stddev1)
    y[1,t]=Gsample(mean2,stddev2)

#so it is like using I matrix as a mixing matrix

iterations=10
x=np.zeros((N,T))
for it in range(iterations):
    for t in range(0,T):
        x[:,t]=unmix(G, y[:,t])
        
    for i in range(len(HMMs)):
        HMMs[i]._calc_gauss_param(x[i,:])
    G=Calc_G(G,HMMs,x)

print "-------------------"
print G
print HMMs[0].mu_state
print HMMs[0].var_state
print HMMs[1].mu_state
print HMMs[1].var_state
#print HMMs[0].gamma
