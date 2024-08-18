#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:33:32 2024

@author: mtrubshaw
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns

metrics = ["Disease_duration", "ALSFRS-R", "ALSFRS_progression_rate", "UMN_score", "ECAS_total", "Serum_NFL"]
units = ["(months)","","(∆ALSFRS-R/year)","","","(pg/ml)"]
# Load regressor data
demographics = pd.read_csv("../demographics/task_demographics.csv")
group_ = demographics["Group"].values
subjects = demographics["Subject"].values


metric = demographics["Serum_NFL"].values

data = np.squeeze(np.load('data/region_count.npy', allow_pickle=True)[:,:,0,2][:,5:10])
data = np.mean(data,axis=1)

loc_reps = []
for i, subject in enumerate(subjects):
    sub = str(subject)
    if len(sub)==4:
        loc_reps.append(i)


loc_first = []
for loc_rep in loc_reps:
    loc_first.append(np.where(subjects==int(subjects[loc_rep]/10)))
    
loc_first = np.squeeze(np.array(loc_first))

data_fir = data[loc_first]
data_rep = data[loc_reps]

data_m = (data_fir+data_rep)/2

metric_fir = metric[loc_first]
metric_rep = metric[loc_reps]

metric_m = (metric_fir+metric_rep)/2

data_diff = data_rep - data_fir
metric_diff = metric_rep - metric_fir

data_diff_n = (data_diff-np.mean(data_m))/np.std(data_m)
metric_diff_n = (metric_diff-np.mean(metric_m))/np.std(metric_m)






plt.plot(data_diff_n, label='gamma activations')
plt.plot(metric_diff_n, label='metric')
plt.legend()