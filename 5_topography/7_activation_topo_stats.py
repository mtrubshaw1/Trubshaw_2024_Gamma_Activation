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
import pandas as pd
from scipy import stats

import glmtools as glm


data = np.load('data/regions.npy',allow_pickle=True)

participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
group = participants["Group"].values

region_ws = np.zeros((data.shape[0],data.shape[1],52))
for sub in range(data.shape[0]):
    for time in range(data.shape[1]):
        region_ws[sub,time,data[sub,time,0,1]] += 1 #change last dim to 2 for high gamma and 1 for gamma
    





os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('plots/compiled_plots', exist_ok=True)
# Load target data


data = region_ws


# Load regressor data
demographics = pd.read_csv("../demographics/task_demographics.csv")


category_list = demographics["Group"].values
category_list[category_list == "HC"] = 1
category_list[category_list == "ALS"] = 2
category_list[category_list == "rALS"] = 2
category_list[category_list == "PLS"] = 3
category_list[category_list == "rPLS"] = 3
category_list[category_list == "FDR"] = 3
category_list[category_list == "rFDR"] = 3
uniques = np.unique(category_list)

age = demographics["Age"].values

gender = []
for g in demographics["Sex"].values:
    if g == "M":
        gender.append(0)
    else:
        gender.append(1)
gender = np.array(gender)


missing_struc = demographics["Missing_struc"].values


# Create GLM dataset
data = glm.data.TrialGLMData(
    data=data,
    category_list=category_list,
    age=age,
    gender=gender,
    dim_labels=["Subjects", "time", "Parcels"],
    missing_struc=missing_struc,
)

# Design matrix
DC = glm.design.DesignConfig()
DC.add_regressor(name="HC", rtype="Categorical", codes=1)
DC.add_regressor(name="ALS", rtype="Categorical", codes=2)
# DC.add_regressor(name="FDR", rtype="Categorical", codes=3)
DC.add_regressor(name="Sex", rtype="Parametric", datainfo="gender", preproc="z")
DC.add_regressor(name="Age", rtype="Parametric", datainfo="age", preproc="z")
DC.add_regressor(name="Missing Structural", rtype="Parametric", datainfo="missing_struc", preproc="z")



DC.add_contrast(name="ALS-HC", values=[-1, 1, 0, 0, 0])
# DC.add_contrast(name="FDR-HC", values=[-1, 0, 1, 0, 0, 0])
# DC.add_contrast(name="ALS-FDR", values=[0, 1, -1, 0, 0, 0])


design = DC.design_from_datainfo(data.info)
design.plot_summary(savepath="plots/glm_design.png", show=False)
design.plot_leverage(savepath="plots/glm_leverage.png", show=False)
design.plot_efficiency(savepath="plots/glm_efficiency.png", show=False)

# Fit the GLM
model = glm.fit.OLSModel(design, data)

def do_stats(contrast_idx, metric="tstats"):
    # Max-stat permutations
    perm = glm.permutations.MaxStatPermutation(
        design=design,
        data=data,
        contrast_idx=contrast_idx,
        nperms=1000,
        metric=metric,
        tail=0,  # two-tailed t-test
        pooled_dims=(1,2),  # pool over channels
        nprocesses=16,
    )
    null_dist = perm.nulls

    # Calculate p-values
    if metric == "tstats":
        tstats = abs(model.tstats[contrast_idx])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        copes = abs(model.copes[contrast_idx])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100

    return pvalues

for i in range(model.copes.shape[0]):
    cope = model.copes[i]
    pvalues = do_stats(contrast_idx=i)
    print(cope)
    print(pvalues)
    mask = pvalues<0.05
    np.save(f"data/contrast_{i}.npy", cope)
    np.save(f"data/contrast_{i}_pvalues.npy", pvalues)

for unique in uniques:
    count = np.count_nonzero(category_list==unique)
    print('Group',unique,' - ',count)
np.save('data/beta_pval_mask.npy',mask)