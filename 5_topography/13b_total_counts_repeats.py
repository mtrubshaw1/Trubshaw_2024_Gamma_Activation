#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:16:06 2024

@author: mtrubshaw
"""

"""Fit a GLM and perform statistical significance testing.

"""

import numpy as np
import os
import pandas as pd
from scipy import stats

import glmtools as glm
from osl_dynamics.analysis import power
from scipy.stats import ttest_ind

os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('plots/compiled_plots', exist_ok=True)
# Load target data


data = np.load('data/total_counts.npy',allow_pickle=True)
data =  np.swapaxes(data, 0, 2)

for a in [1,2]:
    print(f'freq: {a}')
    data_ = data[:,[a],0]

    data_als = data_[[7,12,15,18,29,45],:]
    data_rals= data_[[81,82,83,84,86,87],:]
    print(np.mean(data_als))
    print(np.mean(data_rals))
    print(f'diff ALS: {np.mean(data_rals)-np.mean(data_als)}')
    t, p =ttest_ind(data_rals,data_als)
    print(f't={t}, p={p}')
    print('')
    data_fdr = data_[21,:]
    data_rfdr = data_[85,:]
    data_fdr_diff = data_rfdr-data_fdr
    print(f'diff FDR: {data_fdr_diff}')
    print('')
    print('-----')