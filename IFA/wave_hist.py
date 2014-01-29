# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:33:21 2014

@author: AlbertoEAF
"""
from __future__ import division
import numpy as np
import scipy.io.wavfile
import pylab as P


wave_filepaths = ["mike.wav", "beet.wav"]
max_size=100000
wave_datas=np.zeros((2,max_size))
def ReadFile():
    for i in range(2):
        filepath=wave_filepaths[i]
        sample_rate, temp = scipy.io.wavfile.read(filepath)
        wave_datas[i,:]=temp[10000:max_size+10000]

def GetData(source,size,period):
    """"""

    wave_data = wave_datas[source,:]
    wave_data=wave_data[period*size:period*size+size]
    wave_data=wave_data.astype(float)
    wave_data/=np.max(wave_data)
    #g=P.hist(wave_data[:], bins = 50)
    #P.show(g)
    #print wave_data.size
    return wave_data


