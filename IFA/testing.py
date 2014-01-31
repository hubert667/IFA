from hmm import *
#from wave_hist import *
from loadingFiles import *
from ICA import *
import numpy as np
import matplotlib.pyplot as plt
#from ccm.Utils import Label

source_files = ['mike.wav', 'street.wav']
S = 2 # states
T = 2000# Time samples
M = 2 # microphoneerratics
N = M # sources
Eps=0.01 #learning rate for the G matrix
iterations=20

#example of H matrix
H=np.identity(N)
H[0,1]=0.5
H[1,0]=0.5
#H /= np.linalg.norm(H)
H1=np.linalg.inv(H)
#H^-1=[1,-0.5;-0.5,1]

G=np.identity(N)

sample_rate=load_data(source_files)
yy=np.zeros((N,T))
yy[0,:]=GetData(0,T,0,0)
yy[1,:]=GetData(1,T,0,0)
y = np.dot(H, yy)  

G_ICA, error_ICA=ICA(y, H1,activation_function=lambda a: -np.tanh(a), learning_rate=Eps, max_iterations = iterations)
#G_ICA, error_ICA=ICA(y, H1,activation_function=lambda a: - (6 * a) / ( a ** 2.0 + 5.0), learning_rate=Eps, max_iterations = iterations)
#T = 1000# Time samples
Eps=0.05
yy=np.zeros((N,T))
yy[0,:]=GetData(0,T,0)
yy[1,:]=GetData(1,T,0)
variances=np.zeros((N,S))
for i in range(M):
    variances[i]=np.abs(np.random.randn(S)*max(yy[i,:]))
HMMs = [ HMM(S,T,[0]*S,variances[n]) for n in range(N) ]

difs =  []
for itM in range(iterations):
    #if itM==50:
    #    Eps=0.05
    #yy[0,:]=GetData(0,T,itM)
    #yy[1,:]=GetData(1,T,itM)
    y = np.dot(H, yy)
    x = unmix(G, y)  
    for i in range(len(HMMs)):
        HMMs[i].update(x[i])
    G = Calc_G(G,HMMs,x,Eps)

    print "-------------------progress:"+str(itM*100/iterations)
    print "G:", G/G[0,0]
    #G/=np.max(G,axis=1)
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
#yy[0,:]=GetAllData(0)
#yy[1,:]=GetAllData(1)
y = np.dot(H, yy)  
save_data(y, "mixed",sample_rate)
x = unmix(G, y)     
save_data(x, "unmixed",sample_rate)
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("G error (Amari)")
plt.plot(error_ICA,label="ICA")
plt.plot(difs,label="IFA with HMMs")
plt.legend( loc='upper left', numpoints = 1 )
plt.show()




#print HMMs[0].gamma

