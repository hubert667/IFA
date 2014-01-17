# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:55:52 2014

@author: AlbertoEAF
"""

import numpy as np
import matplotlib.pyplot as plt
from math import *

import sys

f = 1
phi  =0

f2 = 1.03
phi2 = 0

items = 500



X  = np.linspace(0,10,items)
Y = np.sin(2*pi*f * X + phi)
Z = np.sin(2*pi*f2*X + phi2)

#plt.plot(X,Y)
#plt.plot(X,Z)

#plt.figure()
plt.hist(Y,alpha=0.2)
plt.hist(Z,alpha=0.2)

#print Y
#print Z

#print Y-Z

error = Y-Z

#plt.hist(error)

print "<Error/item> = ", np.sum(np.abs(error))/items
