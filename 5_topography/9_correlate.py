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

metrics = ["ALSFRS-R", "ALSFRS_progression_rate", "UMN_score", "Serum_NFL", "ECAS_total"]
units = ["","(∆ALSFRS-R/year)","","(pg/ml)",""]
# Load regressor data
demographics = pd.read_csv("../demographics/task_demographics.csv")
group_ = demographics["Group"].values

# Create a single figure to hold all plots


for f, f_name in enumerate(['gamma','hgamma']):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for i, (unit,m) in enumerate(zip(units,metrics)):
        metric_ = demographics[f"{m}"].values
        
        idx_ = np.isnan(metric_)
        idx = np.where(idx_==False)
        
        data = np.squeeze(np.load('data/region_count.npy', allow_pickle=True)[:,:,0,(f+1)][idx,:])
        data = np.mean(data[:,[5,6,7,8,9]],axis=1)
        data_hc = np.squeeze(np.load('data/region_count.npy', allow_pickle=True)[:,:,0,(f+1)][group_=='HC',9])
        
        metric = np.squeeze(metric_[idx])
        metric_hc = np.zeros(len(data_hc))
        if (m =='ECAS_total') or (m =='Serum_NFL'):
            metric_hc = np.squeeze(metric_[group_=='HC'])
        elif m=='ALSFRS-R':
            metric_hc = metric_hc+48

            
        
        slope, intercept, r, p, _ = linregress(data, metric)
        
        p = p * len(metrics)
        if p > 1:
            p = 1
        
        row = i // 3
        col = i % 3
        ax = axes[col,row ]
        sns.scatterplot(x=data, y=metric, color="#89ABE3", ax=ax,label='ALS')
        # sns.scatterplot(x=data_hc, y=metric_hc, color="green", ax=ax, label='HC')
        sns.lineplot(x=data, y=slope*data + intercept, label=f'r = {r:.2f}\np = {p:.3f}', color="#EA738D", ax=ax)
        ax.set_title(f"{m}")
        ax.set_ylabel(f'{m} {unit}')
        ax.set_xlabel('Number of activated regions')
        ax.legend()


    plt.tight_layout()

    plt.savefig(f'plots/correlations_{f_name}.png',dpi=300)
    plt.show()