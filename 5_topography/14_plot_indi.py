#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:32:15 2024

@author: mtrubshaw
"""

import numpy as np
import pandas as pd
from osl_dynamics.analysis import power
import os
import mne
import matplotlib.pyplot as plt
os.makedirs('plots/topo/ind_activations',exist_ok=True)
participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
group = participants["Group"].values
regions_ = np.load('data/regions.npy',allow_pickle=True)[:,[4,5,6,7,8,9]]

p_groups = ['ALS', 'HC']
pos = ['hyper', 'hypo']
freqs = ['beta', 'gamma', 'hgamma']

col = 'seismic'
stdc = 1

splits = 60
samples = 1500
sos = samples/splits

freqs = ['hgamma']

for g, fr in enumerate(freqs):
    freq = np.load(f'data/psd_tf_{fr}_range_meg.npy',allow_pickle=True)[:,:52]
    time = np.arange(freq.shape[2])/250
    freq_bl = mne.baseline.rescale(freq,time,[0,0.6])[:,:52]
    freq_bin = []

    cutoff = stdc*np.std(freq_bl[group=='HC'],axis=(0,2))
    cutoffs = np.tile(cutoff[:,np.newaxis],(1,splits))
    cutoffs_ = np.tile(cutoffs[:,:,np.newaxis],(1,freq_bl.shape[0])).swapaxes(0, 2).swapaxes(2, 1)  

    # loc = freq_bl>cutoffs_
    
    

    
    freq_bl_av = []
    for n in range(splits):
        a= int(sos*n)
        b = int((sos*n)+sos)

        freq_bl_av.append(np.mean(np.squeeze(freq_bl[:,:,[np.arange(a,b)]]),axis=2))
    freq_bl_av = np.array(freq_bl_av).swapaxes(0, 2).swapaxes(0, 1)

    loc = freq_bl_av>cutoffs_
    
    freq_ones = np.zeros(freq_bl_av.shape)
    freq_ones[loc] = 1
    
    freq_act = np.zeros(freq_bl_av.shape)
    freq_act[loc] = freq_bl_av[loc]
    

    for s in range(freq_act.shape[0]):
        if s>39 & s<77:
            f = freq_act[s].T
        
        
        
            power.save(
                f,
                mask_file="MNI152_T1_8mm_brain.nii.gz",
                parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
                show_plots=False,
                plot_kwargs={
                    "cmap": f"{col}",
                    "bg_on_data": 1,
                    "darkness": 0.3,
                    "alpha": 1,
                    "views": ["lateral"],
                    "vmax": 1,
                    "vmin":-1,
                    "n_jobs": -1
                },
                filename=f"plots/topo/ind_activations/sub{s}_{fr}_plot.png",
            )
            plt.close()

            
# reg_m_sigs = np.array(reg_m_sigs)        
# reg_m_sigs = reg_m_sigs.reshape((len(p_groups),len(pos),len(freqs)))


