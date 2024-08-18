#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:23:04 2024

@author: mtrubshaw
"""

import mne
import numpy as np
from osl.preprocessing import osl_wrappers
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt


# Directories
epoch_dir = Path("/ohba/pi/knobre/mtrubshaw/ALS_task/data/emg_grip_epoched")

participants = pd.read_csv("../demographics/task_demographics.csv")
subjects = participants["Subject"].values
missing_task2 = participants["Missing_task2"].values


task = 'task1'
freqs = [[13,30],[30,48],[52,80]]
freqs_n = ['beta','gamma','hgamma']

fsample = 250

# # Fif files containined parcellated data
data_emg = []
data_meg = []
for n, subject in enumerate(subjects):
    epoch_file_emg = epoch_dir / f'{task}_s0{subject}_emg_grip_epo.fif'
    epochs_emg = mne.read_epochs(epoch_file_emg)
    d_emg = epochs_emg.get_data()
    if task == 'task1':
        if missing_task2[n] == 'No' or missing_task2[n] == 'Mri':
            epoch_file_emg = epoch_dir / f'task2_s0{subject}_emg_grip_epo.fif'
            epochs_emg = mne.read_epochs(epoch_file_emg)
            d_emg2 = epochs_emg.get_data()
            d_emg = np.concatenate((d_emg,d_emg2))
            
    data_emg.append(d_emg)
    
    epoch_file_meg = epoch_dir / f'{task}_s0{subject}_meg_epo.fif' 
    epochs_meg = mne.read_epochs(epoch_file_meg)
    d_meg = epochs_meg.get_data()
    if task == 'task1':
        if missing_task2[n] == 'No' or missing_task2[n] == 'Mri':
            epoch_file_meg = epoch_dir / f'task2_s0{subject}_meg_epo.fif'
            epochs_meg = mne.read_epochs(epoch_file_meg)
            d_meg2 = epochs_meg.get_data()
            d_meg = np.concatenate((d_meg,d_meg2))
    data_meg.append(d_meg)
    
#Standardise timecourses prior to power calculation
data_meg_s = []    
for sub in range(len(data_meg)):
    d = data_meg[sub]
    mean_last_dim = np.mean(d, axis=-1, keepdims=True)
    std_last_dim = np.std(d, axis=-1, keepdims=True)
    data_meg_s.append((d-mean_last_dim)/std_last_dim)
data_meg = data_meg_s
data_meg_s = []


# calculate power tfr
for freq, freq_n in zip(freqs,freqs_n):
    psd_tf_meg = []
    psd_tf_emg = []
    for n, subject in enumerate(subjects):
        
        fr = np.arange(freq[0],freq[1])
        psd_tf_meg.append(np.mean(mne.time_frequency.tfr_array_morlet(data_meg[n][:,:,:], fsample, fr, n_jobs=10, output='power'),axis=(0,2)))
    

    np.save(f'data/psd_tf_{freq_n}_range_meg.npy',psd_tf_meg)    