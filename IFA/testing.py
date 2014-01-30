from hmm import *
#from wave_hist import *
from loadingFiles import *
import numpy as np
import matplotlib.pyplot as plt

S = 2 # states
T = 600# Time samples
M = 2 # microphoneerratics
N = M # sources
Eps=0.05 #learning rate for the G matrix

#example of H matrix
H=np.identity(N)
H[0,1]=0.5
H[1,0]=0.5
H /= np.linalg.norm(H)
H1=np.linalg.inv(H)
#H^-1=[1,-0.5;-0.5,1]

G=np.identity(N)
        
mean1=0
stddev1=6
mean2=0
stddev2=0.5

#oryginal sources

sample_rate=load_data()
yy=np.zeros((N,T))
yy[0,:]=GetData(0,T,0)
yy[1,:]=GetData(1,T,0)
y = np.dot(H, yy)  


variances=np.zeros((N,S))
for i in range(M):
    variances[i]=np.abs(np.random.randn(S)*max(yy[i,:]))
HMMs = [ HMM(S,T,[0]*S,variances[n]) for n in range(N) ]

iterations=40
difs =  []


for itM in range(iterations):
    if itM==50:
        Eps=0.01
    yy[0,:]=GetData(0,T,itM)
    yy[1,:]=GetData(1,T,itM)
    y = np.dot(H, yy)
    x = unmix(G, y)  
    for i in range(len(HMMs)):
        HMMs[i].update(x[i])
    G = Calc_G(G,HMMs,x,Eps)

    print "-------------------progress:"+str(itM*100/iterations)
    print "mu",HMMs[0].mu_states
    print "var",HMMs[0].var_states
    print "mu",HMMs[1].mu_states
    print "var",HMMs[1].var_states
    print "G:", G/np.max(G,axis=1)
    G/=np.max(G,axis=1)
    dif=dAmari(G,H1)
    difs.append(dif)

    for hmm_i in range(len(HMMs)):
        print "mu" ,   HMMs[hmm_i].mu_states
        print "var",   HMMs[hmm_i].var_states
        print "LogLs", HMMs[hmm_i].log_likelihood()
        #HMMs[hmm_i].log_likelihood_check()
        
    if len(difs)>5 and difs[-1] > difs[-2] : # sometimes fails right at the first step, does it fail after? yes and probably when it fails after it would fail at the beginning as well.
        print "Error in dAmari increased."
        #break
    else:
        print " "


#saving data:
temp=GetAllData(0)
yy=np.zeros((N,temp.size))
yy[0,:]=GetAllData(0)
yy[1,:]=GetAllData(1)
y = np.dot(H, yy)  
save_data(y, "mixed",sample_rate)
x = unmix(G, y)     
save_data(x, "unmixed",sample_rate)
plt.show(plt.plot(difs[1:]))



#print HMMs[0].gamma

