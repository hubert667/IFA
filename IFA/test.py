# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:44:49 2014

@author: AlbertoEAF
"""
import numpy as np

s = np.arange(0, S)
t = np.arange(1, T)
c = np.ones((3,1))
d = np.arange(1,4)
e = (c.T*d).T
alpha = a.alpha
beta = a.beta

A = np.ones((S,T))
B = np.ones((S,T))
for i in range(S):
    for j in range(T):
        A[i,j] = i*j
for j in range(T):
    B[:,j] = j

C = np.ones((S,T-1))

for s in range(S):
    for t in range(1,T):
        C[s,t-1] = (s*(t-1))*(t)


s = np.arange(0, S)
t = np.arange(1, T)
print (A[:,t-1]*B[:,t]).shape

print C.shape

print AbsTotalError(A[:,t-1]*B[:,t],C)