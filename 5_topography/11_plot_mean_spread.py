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

os.makedirs('plots/topo/shared_regs',exist_ok=True)
participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
group = participants["Group"].values
regions_ = np.load('data/regions.npy',allow_pickle=True)[:,[4,5,6,7,8,9]]

p_groups = ['ALS', 'HC']
pos = ['hyper', 'hypo']
freqs = ['beta', 'gamma', 'hgamma']



reg_m_sigs = []
for p_group in p_groups:
    for h, po in enumerate(pos):
        for g, freq in enumerate(freqs):
            regions = regions_[:,:,h,g]
            
            regions_als_ = regions[group==p_group]
            
            for r in range(regions.shape[1]):
                regions_als = regions_als_[:,r]
                
                regs = np.zeros((len(regions_als),52))
                for s in range(len(regions_als)):
                    regs[s,regions_als[s]] = 1
    
                # for p in range(len(regions_als)):
             
            regs_m = np.mean(regs,axis=0)
            regs_m_sig = np.where(regs_m>0.75)
            reg_m_sigs.append(regs_m_sig)
            
            
            plot = np.zeros(52)
            plot[regs_m_sig[0]]= 1
            power.save(
                regs_m,
                mask_file="MNI152_T1_8mm_brain.nii.gz",
                parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
                plot_kwargs={
                    "cmap": "hot_r",
                    "bg_on_data": 1,
                    "darkness": 0.3,
                    "alpha": 1,
                    "views": ["lateral"],
                    "vmax": 1,
                    "vmin":0
                },
                filename=f"plots/topo/shared_regs/{p_group}_{po}_{freq}.png",
            )
            
reg_m_sigs = np.array(reg_m_sigs)        
reg_m_sigs = reg_m_sigs.reshape((len(p_groups),len(pos),len(freqs)))


