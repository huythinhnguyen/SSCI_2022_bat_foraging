#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List of utility functions for processing and transforming echo waveform

@author: thinh
"""
"""  LIST OF COMPLETED FUNCTION:
    ----------------------------
def get_raw_echo(klass, dist, angle, dataset):
    ... take in klass, dist, angle and put it in a tuple "spec"
    ... randomly select an echo that meet the spec
    ... output --> echo = {"left", "right"}
    return echo

def subtract_bg(echo, angle, dataset):
    ... for a given input angle, select the approriate background echo then subtract the input echo for the background
    ... the background is randomly select from different sample in the dataset    
    return echo_nobg


reference: https://en.wikibooks.org/wiki/Engineering_Acoustics/Outdoor_Sound_Propagation
def timeshift_echo(echo, from_dist, to_dist, keepfront=True):
    ... shift the echo assuming speed of sound is 340 m/s
    ... sampling frequency should be 300,000
    return echo_shifted

----THIS FUNCTION IS FUNCTIONING BUT THE CORRECTNESS IS IFFY--------
----NEED TO CHECK THE PHYSICS OF GEOMETRIC SPREADING LOSS!!!--------
def attenuate_echo(echo, from_dist, to_dist,alpha = 1.31, spreadloss = True):
    ... 2 loss phenomena: atmospheric absorption and geometric spreading
    ... attenuation coefficient of air is set to 1.31 db/m
    ... geometric loss is assumed to be sqrt(r_1/r_2) (from r1 to r2)
    return echo_att
def echo_dist_transform(echo,from_dist,to_dist):
    ... call timeshift then call attenuate
    return echo_dist_trans

"""
import numpy as np
from scipy.signal import butter, lfilter
from echo_lib.ears_model import ear_filter
import pathlib
current_path = str(pathlib.Path().absolute())
root = current_path + '/echo_lib/dataset/'
k_dict = {0: 'background', 1: 'pole', 2: 'planter'}
sample_freq = 3e5
speed_of_sound = 340
d_axis = np.arange(1,7001) * 0.5 * (1/sample_freq) * speed_of_sound
main_starts_dict = {0.25: 0.13, 0.5: 0.38, 0.75: 0.62, 1.0: 0.88, 1.25: 1.12, 1.5: 1.36, 1.75: 1.62, 2.0: 1.87, 2.25: 2.11, 2.5: 2.35}
main_ends_dict ={0.25: 0.47, 0.5: 0.66, 0.75: 0.88, 1.0: 1.1, 1.25: 1.31, 1.5: 1.53, 1.75: 1.76, 2.0: 1.98, 2.25: 2.22, 2.5: 2.48}
reverb_starts_dict ={0.25: 0.89, 0.5: 0.79, 0.75: 0.94, 1.0: 1.13, 1.25: 1.33, 1.5: 1.55, 1.75: 1.78, 2.0: 2.01, 2.25: 2.24}
reverb_ends_dict ={0.25: 1.13, 0.5: 0.99, 0.75: 1.15, 1.0: 1.33, 1.25: 1.51, 1.5: 1.71, 1.75: 1.92, 2.0: 2.14, 2.25: 2.35}
planter_starts_dict = {0.5: 0.26, 0.75: 0.49, 1.0: 0.76, 1.25: 0.99, 1.5: 1.24, 1.75: 1.49, 2.0: 1.74, 2.25: 1.91, 2.5: 2.24}
planter_ends_dict = {0.5: 1.29, 0.75: 1.35, 1.0: 1.57, 1.25: 1.71, 1.5: 1.76, 1.75: 1.88, 2.0: 2.38, 2.25: 2.47, 2.5: 2.72}

def retrieve_echo(klass, dist, angle, random_mode=True, index=None):
    path = root + k_dict[klass] + '/' + str(dist) + '_' + str(angle+0) + '/'
    left_set = np.load(path + 'left.npy')
    right_set = np.load(path + 'right.npy')
    if not random_mode:
        np.random.seed(1)
    number_of_results = left_set.shape[0]
    sel_idx = np.random.randint(number_of_results)
    if index is not None:
        sel_idx == index
    l_echo = left_set[sel_idx, :]
    r_echo = right_set[sel_idx, :]
    echo = {'left': l_echo,
            'right': r_echo}
    return echo


def get_raw_echo(klass, dist, angle, random_mode=True, index=None):
    echo = retrieve_echo(klass, dist, angle, random_mode=random_mode, index=index)
    return echo


def add_echo(echo1, echo2):
    l_echo = echo1['left'] + echo2['left']
    r_echo = echo1['right'] + echo2['right']
    echo = {'left': l_echo,
            'right': r_echo}
    return echo


def subtract_bg(echo, angle, random_mode=True, index=None):
    bg = retrieve_echo(0, 0.0, angle, random_mode=random_mode, index=index)
    l_nobg = echo["left"] - bg['left']
    r_nobg = echo["right"] - bg['right']
    echo_nobg = {
        "left"  : l_nobg ,
        "right" : r_nobg
    }
    return echo_nobg


def get_raw_echo_nobg(klass,dist,angle, random_mode=True, index=None):
    spec = (klass, dist, angle)
    echo = get_raw_echo(*spec, random_mode=random_mode, index=index)
    echo = subtract_bg(echo,angle)
    return echo


def snip_raw(echo, from_dist, type='main'):
    if type=='main' or type=='m':
        start = main_starts_dict[from_dist]
        end = main_ends_dict[from_dist]
    elif type=='planter' or type=='p':
        start = planter_starts_dict[from_dist]
        end = planter_ends_dict[from_dist]
    elif type=='reverb' or type=='r':
        start = reverb_starts_dict[from_dist]
        end = reverb_ends_dict[from_dist]
    else:
        print('Error in inputting type for snip_echo(). type="main" or "reverb"')
    start_idx = np.argmin(np.abs(d_axis - start))
    end_idx = np.argmin(np.abs(d_axis - end))
    l_snip = echo['left'][start_idx:end_idx]
    r_snip = echo['right'][start_idx:end_idx]
    snip = {
        'left' : l_snip,
        'right': r_snip
    }
    return snip


def paste_snip(snip, from_dist, to_dist, type='main'):
    #if (to_dist<from_dist) and (type!='reverb') : print("WARNING: WAVE IS TRANSFORM BACKWARD!")
    l_echo = np.zeros(7000)
    r_echo = np.zeros(7000)

    if type=='main' or type=='m':
        start = main_starts_dict[from_dist] + (to_dist-from_dist)
    elif type=='planter' or type=='p':
        start = planter_starts_dict[from_dist] + (to_dist-from_dist)
    elif type=='reverb' or type=='r':
        (a,b,c) = (1.134, 2.532, 0.14)
        start = main_starts_dict[from_dist] + (to_dist-from_dist) + a*np.exp(-b*to_dist)+c
    else:
        print('Error in inputting type for paste_snip(). type="main" or "reverb" or "planter"')
    start_idx = np.argmin(np.abs(d_axis - start))
    end_idx = start_idx + len(snip['left'])

    l_echo[start_idx:end_idx] = snip['left']
    r_echo[start_idx:end_idx] = snip['right']

    echo = {
        'left': l_echo,
        'right': r_echo
    }
    return echo


def get_attenuation(from_dist,to_dist,type='main', alpha=1.31,outward_spread=1, inward_spread=0.5, debug_print=False):
    if type=='main' or type=='m':
        start = main_starts_dict[from_dist]
        end = main_ends_dict[from_dist]
        start_idx = np.argmin(np.abs(d_axis - start))
        end_idx = np.argmin(np.abs(d_axis - end))
        from_dist_array = d_axis[start_idx:end_idx]
        start = main_starts_dict[from_dist] + (to_dist - from_dist)
        start_idx = np.argmin(np.abs(d_axis - start))
        end_idx = start_idx + len(from_dist_array)
        to_dist_array = d_axis[start_idx:end_idx]
    elif type == 'planter' or type == 'p':
        start = planter_starts_dict[from_dist]
        end = planter_ends_dict[from_dist]
        start_idx = np.argmin(np.abs(d_axis - start))
        end_idx = np.argmin(np.abs(d_axis - end))
        from_dist_array = d_axis[start_idx:end_idx]
        start = planter_starts_dict[from_dist] + (to_dist - from_dist)
        start_idx = np.argmin(np.abs(d_axis - start))
        end_idx = start_idx + len(from_dist_array)
        to_dist_array = d_axis[start_idx:end_idx]
    elif type=='reverb' or type=='r':
        start = reverb_starts_dict[from_dist]
        end = reverb_ends_dict[from_dist]
        start_idx = np.argmin(np.abs(d_axis - start))
        end_idx = np.argmin(np.abs(d_axis - end))
        from_dist_array = d_axis[start_idx:end_idx]
        (a, b, c) = (1.134, 2.532, 0.14)
        start = main_starts_dict[from_dist] + (to_dist - from_dist) + a * np.exp(-b * to_dist) + c
        start_idx = np.argmin(np.abs(d_axis - start))
        end_idx = start_idx + len(from_dist_array)
        to_dist_array = d_axis[start_idx:end_idx]
    else:
        print('Error: type="main" or "reverb" or "planter"')
    atmospheric = np.power(10, -alpha*2*(to_dist-from_dist)/20)
    spreading = np.divide(from_dist_array,to_dist_array) ** (outward_spread + inward_spread)
    attenuation = atmospheric * spreading
    if debug_print==True:
        print('atmospheric is:')
        print(atmospheric)
        print('spreading is:')
        print(spreading)
        print('attenuation is:')
        print(attenuation)
        print('length of attenuation array = '+str(len(attenuation)))
    return attenuation


def attenuate_snip(snip, from_dist, to_dist, type='main', alpha = 1.31, outward_spread=1, inward_spread=0.5):
    l_snip = snip['left']
    r_snip = snip['right']
    if type=='main' or type=='m' or type=='planter' or type=='p':
        attenuate = get_attenuation(from_dist=from_dist, to_dist=to_dist, type=type, alpha=alpha, outward_spread=outward_spread, inward_spread=inward_spread)
    elif type=='reverb' or type=='r':
        if to_dist >= 1:
            attenuate = get_attenuation(from_dist=from_dist, to_dist=to_dist, type=type, alpha=alpha, outward_spread=outward_spread, inward_spread=inward_spread)
        elif 0.5 <= to_dist < 1:
            attenuate = 1.2 * (to_dist - from_dist) + 0.7
        else:
            attenuate=1
    l_snip = attenuate * l_snip
    r_snip = attenuate * r_snip
    snip_att = {
        'left': l_snip,
        'right': r_snip
    }
    return snip_att


def get_echo_dist_trans(klass, from_dist, angle, to_dist, reverb_mode=True, alpha=1.31, outward_spread=1, inward_spread=0.5):
    raw = get_raw_echo_nobg(klass=klass, dist=from_dist, angle=angle)
    if klass == 1:
        main = snip_raw(raw, from_dist, type='main')
        main = attenuate_snip(main,from_dist, to_dist, type='main', alpha=alpha, outward_spread=outward_spread, inward_spread=inward_spread)
        main_echo = paste_snip(main,from_dist,to_dist, type='main')
        l_echo = main_echo['left']
        r_echo = main_echo['right']
        if reverb_mode and (to_dist < 2.5):
            if 0.5 <= to_dist < 1:
                temp_ref = 0.75 if to_dist < 0.75 else 1.0
                reverb = snip_raw(raw,temp_ref,type='reverb')
                reverb = attenuate_snip(reverb, temp_ref, to_dist, type='reverb', alpha=alpha, outward_spread=outward_spread, inward_spread=inward_spread)
                reverb_echo = paste_snip(reverb, temp_ref, to_dist, type='reverb')
            else:
                reverb = snip_raw(raw, from_dist, type='reverb')
                reverb = attenuate_snip(reverb,from_dist, to_dist, type='reverb', alpha=alpha, outward_spread=outward_spread, inward_spread=inward_spread)
                reverb_echo = paste_snip(reverb, from_dist, to_dist, type='reverb')
            l_echo = l_echo + reverb_echo['left']
            r_echo = r_echo + reverb_echo['right']
    elif klass>=2:
        main = snip_raw(raw, from_dist, type='planter')
        main = attenuate_snip(main, from_dist, to_dist, type='planter', alpha=alpha, outward_spread=outward_spread, inward_spread=inward_spread)
        main_echo = paste_snip(main, from_dist, to_dist, type='planter')
        l_echo = main_echo['left']
        r_echo = main_echo['right']
    else:
        print("cannot transform background. klass should not be 0")
    echo = {
        'left': l_echo,
        'right': r_echo
    }
    return echo


def get_echo_at_dist(klass, dist, angle, reverb_mode=True, alpha=1.31, outward_spread=1, inward_spread=0.5):
    d4x = 4*dist
    d4x_floor = np.floor(d4x)
    from_dist = d4x_floor/4 if (d4x < 10) else 2.5
    # if klass = planter and distance < 0.5 --> transform from distance of 0.5
    if klass >= 2 and dist < 0.5:
        from_dist = 0.5
    if klass == 1 and dist < 0.25:
        from_dist = 0.25
    echo = get_echo_dist_trans(klass, from_dist, angle, dist, reverb_mode=reverb_mode, alpha=alpha, outward_spread=outward_spread, inward_spread=inward_spread)
    return echo


def echo_floor_ceil_angle_echo(klass, dist, angle, reverb_mode=True, alpha=1.31, outward_spread=1, inward_spread=0.5):
    sign = -1 if angle < 0 else 1
    floor_angle = sign*np.floor(np.abs(angle))
    ceil_angle = sign*np.ceil(np.abs(angle))
    floor_gap = np.abs(angle - floor_angle)
    ceil_gap = np.abs(ceil_angle - angle)
    echo_floor = get_echo_at_dist(klass, dist, floor_angle, reverb_mode=reverb_mode, alpha=alpha, outward_spread=outward_spread, inward_spread=inward_spread)
    echo_ceil = get_echo_at_dist(klass, dist, ceil_angle, reverb_mode=reverb_mode, alpha=alpha, outward_spread=outward_spread, inward_spread=inward_spread)
    return echo_floor, echo_ceil, floor_gap, ceil_gap


def get_echo_trans(klass, dist, angle, interpolate='linear', reverb_mode=True, alpha=1.31, outward_spread=1, inward_spread=0.5):
    echo_floor, echo_ceil, floor_gap, _ = echo_floor_ceil_angle_echo(klass, dist, angle, reverb_mode=reverb_mode, alpha=alpha, outward_spread=outward_spread, inward_spread=inward_spread)
    ceil_gap = 1 - floor_gap
    if interpolate == 'closest':
        ceil_gap = 0 if ceil_gap > floor_gap else 1
        floor_gap = 1 - ceil_gap
    l_echo = ceil_gap * echo_floor['left'] + floor_gap * echo_ceil['left']
    r_echo = ceil_gap * echo_floor['right'] + floor_gap * echo_ceil['right']
    echo = {
        "left": l_echo,
        "right": r_echo
    }
    return echo


def get_background(mode):
    if mode == 'sample':
        echo = retrieve_echo(0,0.0,0.0)
    else:
        (sigma, mu) = (0.57307, 0.00049)
        bg = retrieve_echo(0,0.0,0.0)
        bg_idx = np.argmin(np.abs(d_axis - 0.33)) + 1
        l_bg_snip = bg['left'][0:bg_idx]
        r_bg_snip = bg['right'][0:bg_idx]
        l_echo = np.concatenate((l_bg_snip, np.random.normal(mu,sigma, 7000-bg_idx) ))
        r_echo = np.concatenate((r_bg_snip, np.random.normal(mu,sigma, 7000-bg_idx) ))
        echo = {
            "left": l_echo,
            "right": r_echo
        }
    return echo


def get_total_echo(inView, inView_dist, inView_angle, bandpass=True, interpolate='linear', reverb_mode=True, alpha=1.31, outward_spread=1, inward_spread=0.5):
    if inView.shape[0] == 0:
        bg = get_background('sample')
        l_echo = bg['left']
        r_echo = bg['right']
    else:
        echo_set_left = np.zeros((inView.shape[0], 7000))
        echo_set_right = np.zeros((inView.shape[0], 7000))
        bg = get_background('artificial') # select 'sample' or 'artificial'
        for i in range(echo_set_left.shape[0]):
            echo_i = get_echo_trans(int(inView[i,2]), inView_dist[i,0], inView_angle[i,0], interpolate, reverb_mode, alpha, outward_spread, inward_spread)
            echo_set_left[i, :] = echo_i['left']
            echo_set_right[i, :] = echo_i['right']
        l_echo = np.sum(echo_set_left, axis=0) + bg ['left']
        r_echo = np.sum(echo_set_right, axis=0) + bg['right']
        if bandpass:
            b, a = butter(2, [2e4, 8e4], 'band',fs=3e5)
            l_echo = lfilter(b,a,l_echo)
            r_echo = lfilter(b,a,r_echo)
    echo = {
        "left": l_echo,
        "right": r_echo
    }
    return echo


def echo2envelope(echo, single_channel=True, banksize=10, center_freq=None):
    l_echo = np.copy(echo['left'])
    r_echo = np.copy(echo['right'])
    l_envelope = ear_filter(l_echo, single_channel=single_channel, banksize=banksize, center_freq=center_freq)
    r_envelope = ear_filter(r_echo, single_channel=single_channel, banksize=banksize, center_freq=center_freq)
    envelope = {
        'left': l_envelope,
        'right': r_envelope
    }
    return envelope


def compress_envelope(envelope, subsample=125):
    l_envelope = envelope['left']
    r_envelope = envelope['right']
    l_z_envelope = np.mean(l_envelope.reshape(-1,subsample),axis=1)
    r_z_envelope = np.mean(r_envelope.reshape(-1,subsample),axis=1)
    z_envelope = {
        'left': l_z_envelope,
        'right': r_z_envelope
    }
    return z_envelope


def echo2envelope_z(echo, single_channel=True, banksize=10, center_freq=None, subsample=125):
    envelope = echo2envelope(echo, single_channel=single_channel, banksize=banksize, center_freq=center_freq)
    z_envelope = compress_envelope(envelope, subsample=subsample)
    return z_envelope


def quiet_envelope(envelope, threshold=0.75):
    l_envelope = envelope['left']
    r_envelope = envelope['right']
    mask_l = l_envelope >= threshold
    mask_r = r_envelope >= threshold
    l_result = l_envelope * mask_l
    r_result = r_envelope * mask_r
    result = {
        'left' : l_result,
        'right': r_result
        }
    return result


def get_total_envelope_z(inView, inView_dist, inView_angle, bandpass=True, interpolate='linear', reverb_mode=True, alpha=1.31, outward_spread=1, inward_spread=0.5, subsample=125, quiet=True):
    echo = get_total_echo(inView, inView_dist, inView_angle, bandpass, interpolate, reverb_mode, alpha, outward_spread, inward_spread)
    z_envelope = echo2envelope_z(echo, subsample=subsample)
    if quiet:
        result = quiet_envelope(z_envelope)
    return result


def get_measurement_envelope_z(klass,dist,angle,random_mode=True, index=None,window=False,subsample=125):
    echo = retrieve_echo(klass,dist,angle,random_mode,index)
    if window:
        bg = get_background('artificial')
        if klass==0:
            echo = get_background('artificial')
        elif klass==1:
            snip_m = snip_raw(echo, dist, type='m')
            echo_m = paste_snip(snip_m,dist,dist,type='m')
            if dist<2.5:
                snip_r = snip_raw(echo, dist, type='r')
                echo_r = paste_snip(snip_r,dist,dist,type='r')
                echo = add_echo(echo_m, echo_r)
                echo = add_echo(echo, bg)
            elif klass==2:
                snip = snip_raw(echo, dist, type='planter')
                echo = paste_snip(snip,dist,dist,type='planter')
                echo = add_echo(echo, bg)
    elif window=='same':
        bg = retrieve_echo(0,0.0,0.0)
        bg_idx = np.argmin(np.abs(d_axis - 0.33)) + 1
        l_bg_snip = bg['left'][:bg_idx]
        r_bg_snip = bg['right'][:bg_idx]
        l_echo_snip = echo['left'][bg_idx:]
        r_echo_snip = echo['right'][bg_idx:]
        l_echo = np.concatenate((l_bg_snip, l_echo_snip))
        r_echo = np.concatenate((r_bg_snip, r_echo_snip))
        echo = {'left': l_echo, 'right': r_echo}
    z_envelope = echo2envelope_z(echo, subsample=subsample)
    return z_envelope
