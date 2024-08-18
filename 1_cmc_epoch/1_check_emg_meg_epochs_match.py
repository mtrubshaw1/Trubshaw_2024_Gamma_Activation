
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
src_dir = Path("/home/mtrubshaw/Documents/ALS_task/data/src")
preproc_dir = Path("/home/mtrubshaw/Documents/ALS_task/data/preproc")
epoch_dir = Path("/ohba/pi/knobre/mtrubshaw/ALS_task/data/emg_grip_epoched")
os.makedirs('data',exist_ok=True)
os.makedirs(epoch_dir,exist_ok=True)

participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
smris = participants["MRI_no"].values
missing_task2 = participants["Missing_task2"].values

preproc_files = []
smri_files = []
subject_list = []
parc_list = []
tasks = ['task1','left','right','task2']
tasks2 = ['task1','left','right']
tasks1 = ['left','right','task2']
n_subjects = len(subjects)
for n, subject in enumerate(subjects):
    if missing_task2[n] == "No":
        for task in tasks:
            subject_list.append(f'{task}_s0{subject}')
            parc_list.append(f'sub-{subject}_{task}')
    elif missing_task2[n] == "Yes":
        for task2 in tasks2:
            subject_list.append(f'{task2}_s0{subject}')
            parc_list.append(f'sub-{subject}_{task2}')
    elif missing_task2[n] == "Mri":
        for task in tasks:
            subject_list.append(f'{task}_s0{subject}')
            parc_list.append(f'sub-{subject}_{task}')
    elif missing_task2[n] == "Yes_Mri":
        for task2 in tasks2:
            subject_list.append(f'{task2}_s0{subject}')
            parc_list.append(f'sub-{subject}_{task2}')



# # Fif files containined parcellated data
match = []
for subject, parc in zip(subject_list,parc_list):
    preproc_file = preproc_dir / f'{subject}_raw_tsss/{subject}_tsss_preproc_raw.fif'
    parc_file = src_dir / f'{parc}/sflip_parc-raw.fif'
    if not preproc_file.exists():
        print(f'File not found: {preproc_file}')
        continue
    # Read continuous parcellated data
    raw = mne.io.read_raw_fif(preproc_file, preload=True)
    raw_chn = raw.info['ch_names']
    raw_parc = mne.io.read_raw_fif(parc_file, preload=True)
    raw_parc_chn = raw_parc.info['ch_names']
    
    raw_data = raw.get_data()
    raw_parc_data = raw_parc.get_data()
    
    raw_sti = raw_data[324]
    raw_parc_sti = raw_parc_data[52]
    
    
    if np.array_equal(raw_sti, raw_parc_sti):
        match.append(f'{subject} - match')
    else:
        match.append(f'{subject} - no match')
    
   