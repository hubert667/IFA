# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:58:25 2014

@author: AlbertoEAF
"""

from hmm import *
import numpy as np
import sys


a = HMM(S,T)
x = [ Gsample(0,2) for i in range(T) ]


iterations = 2

print a.alpha[0]

print a.alpha[0]/a.likelihood(x)

exit(1)

for i in range(iterations):
    a.update(x)
    print a.alpha[0]

