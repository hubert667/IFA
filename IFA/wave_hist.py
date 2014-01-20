# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:33:21 2014

@author: AlbertoEAF
"""

import scipy.io.wavfile

sample_rate, wave_data = scipy.io.wavfile.read("mike.wav")

print sample_rate

print wave_data