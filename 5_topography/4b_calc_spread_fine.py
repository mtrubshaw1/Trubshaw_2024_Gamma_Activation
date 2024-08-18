#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:13:22 2024

@author: mtrubshaw
"""


import numpy as np
import pickle

import mne
import pandas as pd
import scipy.stats as stats 

participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
missing_task2 = participants["Missing_task2"].values
group = participants["Group"].values

step = 25
stdc = 2
freqs = ['beta', 'gamma','hgamma']

region_count = []
regions = []

l_times = np.arange(0,1500,step)
h_times= np.arange(step,1500+step,step)

for f in freqs:
    freq = np.load(f'data/psd_tf_{f}_range_meg.npy',allow_pickle=True)
    zeros = []
    
    freq_av=freq
    # for ppt in range(len(freq)):
    #     freq_av.append(np.mean(freq[ppt],axis=0))
    # freq_av = np.array(freq_av)
    
    time = np.arange(freq_av.shape[2])/250
    freq_bl = mne.baseline.rescale(freq_av,time,[0,0.6])[:,:52]
    for l_time, h_time in zip(l_times,h_times):

        
        
        freq_bl_toi = np.squeeze(freq_bl[:,:,[np.arange(l_time,h_time)]])
        
        negatives = np.where(freq_bl_toi<0)
        
        ns = freq_bl_toi[negatives]
        
        cutoff = stdc*np.std(freq_bl[group=='HC'],axis=(0,2))
        # cutoff = np.zeros(52) +1
        cutoff_proj_pos = np.tile(cutoff[:,np.newaxis],(1,len(freq_bl_toi[0][1])))
        cutoff_proj_neg = -np.tile(cutoff[:,np.newaxis],(1,len(freq_bl_toi[0][1])))
        
        cutoff_projs = [cutoff_proj_pos,cutoff_proj_neg]
        names = ['cutoff_proj_pos','cutoff_proj_neg']
        
    

        for cutoff_proj, name in zip(cutoff_projs,names):
            unique_regions = []
            num_uniques = []
            ppt_zeros = []
            for ppt in range(len(freq_bl_toi)):
                if name == 'cutoff_proj_neg':
                    negatives = np.where(freq_bl_toi[ppt]<cutoff_proj)
                else:
                    negatives = np.where(freq_bl_toi[ppt]>cutoff_proj)
                ns = freq_bl_toi[ppt][negatives]
                unique_regions.append(np.unique(negatives[0]))
                num_uniques.append(len(np.unique(negatives[0])))
                
                ppt_zeros_t = np.zeros((freq_bl_toi[ppt].shape))
                ppt_zeros_t[negatives] += 1
                ppt_zeros.append(ppt_zeros_t)
                
            num_uniques = np.array(num_uniques)
            ppt_zeros = np.array(ppt_zeros)
            num_als = num_uniques[group=='ALS']
            num_hc = num_uniques[group=='HC']
            region_count.append(num_uniques)
            regions.append(unique_regions)
            zeros.append(ppt_zeros)
            # stat, pval = stats.ttest_ind(num_als,num_hc,equal_var=False, permutations=10000)
            # print(f'{l_time}, {f} - {name}')
            # print(f'ALS mean: {np.mean(num_als)}')
            # print(f'HC mean: {np.mean(num_hc)}')
            # print(f'stat = {stat}, pval (bonf corrected)= {pval*4}')
            # print('')

    zeros_ = np.array(zeros)
    zeros_ = np.reshape(zeros_,(len(l_times),len(cutoff_projs),zeros_.shape[1],zeros_.shape[2],zeros_.shape[3]))
    np.save(f'data/zeros_{f}.npy',zeros_)
            
            
region_count = np.array(region_count).reshape((len(freqs),len(l_times),len(cutoff_projs),len(freq))).swapaxes(0,3)
regions = np.array(regions).reshape((len(freqs),len(l_times),len(cutoff_projs),len(freq))).swapaxes(0,3)
np.save('data/region_count.npy',region_count)
np.save('data/regions.npy',regions)
