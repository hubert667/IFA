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
stddev1=6
mean2=0
stddev2=0.5

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


iterations=50
egs =  [10000]
negs = [10000]

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
    eG = G - np.linalg.inv(H)
    egs.append(np.linalg.norm(eG))
    NeG = G/np.linalg.norm(G) - np.linalg.inv(H)/np.linalg.norm(np.linalg.inv(H))    
    negs.append(np.linalg.norm(NeG))
    print "eG", eG, egs[-1]
    print "NeG", NeG, negs[-1]
    for hmm_i in range(len(HMMs)):
        print "mu" ,   HMMs[hmm_i].mu_states
        print "var",   HMMs[hmm_i].var_states
        print "LogLs", HMMs[hmm_i].log_likelihood()
        HMMs[hmm_i].log_likelihood_check()
        
    if egs[-1] > egs[-2] and len(egs)>5: # sometimes fails right at the first step, does it fail after? yes and probably when it fails after it would fail at the beginning as well.
        print "Error in G increased."
        break
    if negs[-1] > negs[-2] and len(egs)>5:
        print "Error in normalized G increased."
        break
    

plt.plot(egs[1:])
plt.plot(negs[1:])

#print HMMs[0].gamma

