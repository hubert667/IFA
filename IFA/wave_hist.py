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

def GetData(source,size):
    """"""
    filepath=wave_filepaths[source]
    sample_rate, wave_data = scipy.io.wavfile.read(filepath)
    wave_data=wave_data[5000:5000+size]
    wave_data=wave_data.astype(float)
    wave_data/=np.mean(wave_data)
    g=P.hist(wave_data[:], bins = 50)
    #P.show(g)
    #print wave_data.size
    return wave_data


