from hmm import *
from wave_hist import *
import numpy as np
import matplotlib.pyplot as plt

S = 2 # states
T = 3000# Time samples
M = 2 # microphones
N = M # sources

#example of H matrix
H=np.identity(N)
#H[0,1]=0.5
#H[1,0]=0.25
H /= np.linalg.norm(H)
#H^-1=[1.73,-0.86;-0.43,1,73]

G = np.random.random((N, N))
#G=np.identity(N)
        


mean1=0
stddev1=4
mean2=0
stddev2=1

#oryginal sources
yy=np.zeros((N,T))
yy[0,:]=GetData(0,T)
yy[1,:]=GetData(1,T)

variances=np.zeros((N,S))
for i in range(M):
    variances[i]=np.abs(np.random.randn(S)*max(yy[i,:])*10)
HMMs = [ HMM(S,T,[0]*S,variances[n]) for n in range(N) ]
#for t in range(T):
    #yy[0,t]=Gsample(mean1,stddev1)
    #yy[1,t]=Gsample(mean2,stddev2)
    

#mixing
 
y = np.dot(H, yy)
        

#so it is like using I matrix as a mixing matrix

iterations=500
egs = []
negs = []
for itM in range(iterations):
    x = unmix(G, y)  
    for i in range(len(HMMs)):
        HMMs[i].update(x[i])
    G = Calc_G(G,HMMs,x)

    print "-------------------"
    print "mu",HMMs[0].mu_states
    print "var",HMMs[0].var_states
    print "mu",HMMs[1].mu_states
    print "var",HMMs[1].var_states
    print "G:", G
    eG = G - H
    egs.append(np.linalg.norm(eG))
    NeG = G/np.linalg.norm(G) - H/np.linalg.norm(H)    
    negs.append(np.linalg.norm(NeG))
    print "eG", eG, egs[-1]
    print "NeG", NeG, negs[-1]
    print "mu",HMMs[0].mu_states
    print "var",HMMs[0].var_states
    print "mu",HMMs[1].mu_states
    print "var",HMMs[1].var_states
    

plt.plot(egs)
plt.plot(negs)

#print HMMs[0].gamma

