from hmm import *
import numpy as np
import matplotlib.pyplot as plt

S = 1 # states
T = 1000# Time samples
M = 2 # microphones
N = M # sources

#example of H matrix
H=np.identity(N)
H[0,1]=0.5
H[1,0]=0.25
H /= np.linalg.norm(H)

G =  np.random.random((N, N))
#G=np.identity(N)
        
HMMs = [ HMM(S,T) for n in range(N) ]

Eps=0.01 #learning rate for the G matrix

mean1=0.
stddev1=6.
mean2=0.
stddev2=0.5


HMMs[0].gamma[:,:] = 1.0
HMMs[0].mu_states[:] = mean1
HMMs[0].var_states[:] = stddev1


HMMs[1].gamma[:,:] = 1.0
HMMs[1].mu_states[:] = mean2
HMMs[1].var_states[:] = stddev2


#oryginal sources
yy=np.zeros((N,T))
for t in range(T):
    yy[0,t]=Gsample(mean1,stddev1)
    yy[1,t]=Gsample(mean2,stddev2)

#mixing
y = np.dot(H, yy)
        

iterations=50
egs =  [np.inf]
negs = [np.inf]
increased_eG = False
increased_NeG = False
for itM in range(iterations):
    x = unmix(G, y)  
#    for i in range(len(HMMs)):
#        HMMs[i].update(x[i])
    G = Calc_G(G,HMMs,x, Eps)

    print "-------------------"
    print "G:", G/G[0,0]
    eG = G - np.linalg.inv(H)
    egs.append(np.linalg.norm(eG))
    NeG = G/np.linalg.norm(G) - np.linalg.inv(H)/np.linalg.norm(np.linalg.inv(H))    
    negs.append(np.linalg.norm(NeG))
    print "eG", eG, egs[-1]
    print "NeG", NeG, negs[-1]
    for hmm_i in range(len(HMMs)):
        print "mu" ,   HMMs[hmm_i].mu_states
        print "var",   HMMs[hmm_i].var_states
        #print "LogLs", HMMs[hmm_i].log_likelihood()
        #HMMs[hmm_i].log_likelihood_check()
        
    if egs[-1] > egs[-2] and len(egs)>5: # sometimes fails right at the first step, does it fail after? yes and probably when it fails after it would fail at the beginning as well.
        increased_eG = True
        break
    if negs[-1] > negs[-2] and len(negs)>5:
        increased_NeG = True
        break
    
if increased_eG:  print "Error in G increased."
if increased_NeG: print "Error in normalized G increased."
    
plt.plot(egs[1:])
plt.plot(negs[1:])

#print HMMs[0].gamma

