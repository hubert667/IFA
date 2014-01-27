from hmm import *
import numpy as np


S = 2 # states
T = 1000# Time samples
M = 2 # microphones
N = M # sources

#example of H matrix
H=np.identity(N)
#H[0,1]=0.5
#H[1,0]=1.5

G = np.random.random((N, N))  
#G=np.identity(N)
        
HMMs = []
for n in range(N):
    HMMs.append(HMM(S,T))
    

mean1=0
stddev1=4
mean2=1
stddev2=2

#oryginal sources
yy=np.zeros((N,T))
for t in range(T):
    yy[0,t]=Gsample(mean1,stddev1)
    yy[1,t]=Gsample(mean2,stddev2)

#mixing
y=np.zeros((N,T))   
for t in range(T):
        y[:,t]=unmix(H, yy[:,t])

#so it is like using I matrix as a mixing matrix

iterations=20
freezeIterations=20
x=np.zeros((N,T))
for itM in range(iterations):
    for t in range(T):
        x[:,t]=unmix(G, y[:,t])
        
    for it in range(freezeIterations):  
        for i in range(len(HMMs)):
            HMMs[i].update(x[i,:])
    for it in range(freezeIterations):
        G = Calc_G(G,HMMs,x)

    print "-------------------"
    print "G:", G
    print "mu",HMMs[0].mu_states
    print "var",HMMs[0].var_states
    print "mu",HMMs[1].mu_states
    print "var",HMMs[1].var_states
    

#print HMMs[0].gamma

