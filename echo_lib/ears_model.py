#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:36:19 2021

@author: thinh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, gammatone

emit_freq = 4e4


def generatebankfrom1channel(freq, banksize):
    # generate and place for output to be stored
    """ https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html """
    erb_step = 1
    freq_in_erb = 21.4*np.log10(.00437*freq + 1)
    bank = np.concatenate(( 
        np.flip(np.arange(freq_in_erb - erb_step, freq_in_erb - erb_step*(np.floor(banksize/2) +1 ), -erb_step)) , 
        np.arange(freq_in_erb, freq_in_erb + erb_step*np.ceil(banksize/2), erb_step) 
        ))
    bank = ( np.power(10,bank/21.4) - 1 ) / 0.00437
    return bank


def cochlear_model(data, channels):
    """ make sure that data is a 1-D row vector, otherwise it will not work """
    num_of_channels = len(channels)
    fs=300000
    """ broad bandpass filter 10kHz to 200kHz"""
    b, a = butter(4,[20000, 80000],'band',fs=fs)
    temp0 = lfilter(b,a,data)
    """ gammatone filterbanks """
    if num_of_channels == 1:
        fc = channels[0]
        b, a = gammatone(fc,'fir',fs=fs)
        temp = lfilter(b, a, temp0)
    else:
        temp = np.empty((num_of_channels, len(data))) # 
        for idx, fc in enumerate(channels):
            b, a = gammatone(fc, 'fir', fs=fs)
            temp[idx,:] = lfilter(b,a,temp0)
    """" halfwave rectify, exp compress, lowpass """
    temp[temp<0] = 0
    temp = np.power(temp,0.4)
    b, a = butter(2, 1000, 'low', fs=fs)
    y = lfilter(b,a,temp)
    return y


def ear_filter(data, single_channel=True, banksize=10, center_freq=None):
    if center_freq==None:
        freq = np.copy(emit_freq)
    else:
        freq = np.copy(center_freq)
    if not single_channel:
        channel_array = generatebankfrom1channel(freq,banksize)
    else:
        channel_array = np.array([freq])
    y = cochlear_model(data, channel_array)
    return y


if __name__ == '__main__':
    fullecho_raw = np.load('/home/thinh/Dropbox/statics_sonar/Spring_2021/smallears/smallears_wood90_p5m/measurement0004.npy')
    wave = fullecho_raw[:,0,0,2]
    #chan_array = np.array([40000])
    chan_array = generatebankfrom1channel(40000,10)
    result = cochlear_model(wave,chan_array)
    plt.imshow(result,cmap=plt.cm.hot,origin='lower',interpolation = 'nearest',extent = [0, 7000*(.01/3), chan_array[0], chan_array[-1]], aspect = 'auto')
    #plt.plot(wave/np.max(wave))
    #plt.plot(result/np.max(result))