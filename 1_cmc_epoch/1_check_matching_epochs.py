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
from osl_dynamics.analysis import static


# Directories
epoch_dir = Path("/ohba/pi/knobre/mtrubshaw/ALS_task/data/emg_grip_epoched")


participants = pd.read_csv("../demographics/task_demographics.csv")
subjects = participants["Subject"].values


task = 'task1'

fsample = 250

# # Fif files containined parcellated data
data_emg = []
data_meg = []
for subject in subjects:
    epoch_file_emg = epoch_dir / f'{task}_s0{subject}_emg_grip_epo.fif'
    epoch_file_meg = epoch_dir / f'{task}_s0{subject}_meg_epo.fif'
        
    epochs_emg = mne.read_epochs(epoch_file_emg)
    epochs_meg = mne.read_epochs(epoch_file_meg)
    
    data_emg.append(epochs_emg.get_data())
    data_meg.append(epochs_meg.get_data())

psd_tf_meg = []
psd_tf_emg = []
emg_motor = []
match = []
for n, subject in enumerate(subjects):
    
    freqs = np.arange(2,47)
    
    meg_sub = data_meg[n]
    meg_sti = np.squeeze(meg_sub[:,[52]])
    
    emg_sub = data_emg[n]
    emg_sti = np.squeeze(emg_sub[:,4])
    
    if np.array_equal(emg_sti , meg_sti):
        print(f'{subject} ----- matched stim channel -----')
    else:
        print(f'{subject} - unmatched!!')
    

    
    

