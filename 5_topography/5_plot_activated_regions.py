#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:26:04 2024

@author: mtrubshaw
"""
import numpy as np
import pandas as pd

from osl_dynamics.analysis import power
import os

os.makedirs('plots/topo/activations',exist_ok=True)
participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
group = participants["Group"].values
        
pos = ['hyper', 'hypo']
freqs = ['beta', 'gamma', 'hgamma']


pos = ['hyper']
freqs = [ 'hgamma']

for h, po in enumerate(pos):
    for g, freq in enumerate(freqs):
        

        
        data = np.load(f'data/zeros_{freq}.npy',allow_pickle=True)[:,h]
        data_ = np.mean(data,axis=-1)
        data_s = (data_-np.mean(data_,axis=2,keepdims=True))/np.std(data_,axis=2,keepdims=True)
        data_s = np.nan_to_num(data_s)
        
        data_s = data_
        # for ppt in range(data.shape[1]):
        #     data_.append((data[:,ppt]-np.mean(data[:,ppt]))/np.std(data[:,ppt]))
        # data = np.array(data_).swapaxes(0, 1)
        # data=data_

        
        
        region_ws_als = np.mean(data_s[:,group=='ALS'],axis=1)
        region_ws_hc = np.mean(data_s[:,group=='HC'],axis=1)
        als_hc = region_ws_als-region_ws_hc
        
        col = 'seismic'
        if po == 'hypo':
            col = 'seismic_r'
        power.save(
            region_ws_als,
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            plot_kwargs={
                "cmap": f"{col}",
                "bg_on_data": 1,
                "darkness": 0.3,
                "alpha": 1,
                "views": ["lateral"],
                "vmax": 0.2,
                "vmin":-0.2,
                "n_jobs": -1
            },
            filename=f"plots/topo/activations/{freq}_{po}_plot_als.png",
        )
        
        power.save(
            region_ws_hc,
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            plot_kwargs={
                "cmap": f"{col}",
                "bg_on_data": 1,
                "darkness": 0.3,
                "alpha": 1,
                "views": ["lateral"],
                "vmax": 0.2,
                "vmin":-0.2,
                "n_jobs": -1
            },
            filename=f"plots/topo/activations/{freq}_{po}_plot_hc.png",
        )
        
        # power.save(
        #     als_hc,
        #     mask_file="MNI152_T1_8mm_brain.nii.gz",
        #     parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        #     plot_kwargs={
        #         "cmap": f"{col}",
        #         "bg_on_data": 1,
        #         "darkness": 0.3,
        #         "alpha": 1,
        #         "views": ["lateral"],
        #         "vmax": 0.2,
        #         "vmin":-0.2,
        #         "n_jobs": -1
        #     },
        #     filename=f"plots/topo/activations/{freq}_{po}_plot_als_hc.png",
        # )