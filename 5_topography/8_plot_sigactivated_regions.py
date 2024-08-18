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

os.makedirs('plots/topo/sig_activations',exist_ok=True)

participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
group = participants["Group"].values

data = np.load(f'data/contrast_0.npy')
pvalues = np.load('data/contrast_0_pvalues.npy')

power.save(
    data,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={
        "cmap": "bwr",
        "bg_on_data": 1,
        "darkness": 0.3,
        "alpha": 1,
        "views": ["lateral"],
        "vmax": 0.5
    },
    filename=f"plots/topo/sig_activations/als_hc_copes.png",
)


power.save(
    pvalues,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={
        "cmap": "Greens_r",
        "bg_on_data": 1,
        "darkness": 0.3,
        "alpha": 1,
        "views": ["lateral"],
        "vmax": 0.1
    },
    filename=f"plots/topo/sig_activations/als_hc_pvals.png",
)
