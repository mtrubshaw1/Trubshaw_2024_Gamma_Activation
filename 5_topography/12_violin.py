#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:42:52 2024

@author: mtrubshaw
"""



import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
from scipy.stats import ttest_ind

os.makedirs('plots/topo/num_regs',exist_ok=True)

demographics = pd.read_csv("../demographics/task_demographics.csv")

group = demographics["Group"].values
group[group == "HC"] = "HC"
group[group == "ALS"] = "ALS"
group[group == "rALS"] = "ALS"
group[group == "PLS"] = "PLS"
group[group == "rPLS"] = "PLS"
group[group == "FDR"] = "FDR"
group[group == "rFDR"] = "FDR"

pos = ['hyper','hypo']
freqs = [ 'beta','gamma','hgamma']

total_counts = []
for h, po in enumerate(pos):
    for g, freq in enumerate(freqs):
        metric = demographics["ALSFRS_progression_rate"]
        metric[np.isnan]=0
        data = np.squeeze(np.load('data/region_count.npy', allow_pickle=True)[:,:,h,g][:,:])
        data = np.mean(data[:,[5,6,7,8,9]],axis=1)
        tips = sns.load_dataset("tips")
        
        dict_ = {
            "Number of activated regions": data,
            "Group": group,
            "Metric":metric,
        }
        df = pd.DataFrame(dict_)
        filtered_df = df[df["Group"].isin(["ALS", "HC"])]
        
        t, p = ttest_ind(data[group=="HC"],data[group=="ALS"])
        p = p*6
        print(f't = {t}, p = {p:.8f}')
        # Create a violin plot
        sns.swarmplot(x="Group", y="Number of activated regions", data=filtered_df,hue="Group",legend=False)
        plt.title(f"Number of {po}activated regions in {freq} band \n during tonic grip (1-3s) ")
        plt.ylim(0,59)
        plt.savefig(f"plots/topo/num_regs/{freq}_{po}_num_activated_regs.png",dpi=300)
        # Show the plot
        plt.show()
        total_counts.append(data)
        
total_counts = (np.array(total_counts)).reshape(len(pos),len(freqs),len(data))
np.save('data/total_counts.npy',total_counts)
