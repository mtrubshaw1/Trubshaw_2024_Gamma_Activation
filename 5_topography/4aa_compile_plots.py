#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:08:26 2024

@author: mtrubshaw
"""
import numpy as np
import matplotlib.pyplot as plt

cols = 3
comparisons = ['beta', 'gamma','hgamma']


fig, ax = plt.subplots(nrows=2, ncols=cols, figsize=(cols,2))
for i, name in enumerate(comparisons):
    im1 = plt.imread(f"plots/topology_spread_hyper_{name}.png")
    im2 = plt.imread(f"plots/topology_spread_hypo_{name}.png")
    ax[0, i].imshow(im1)
    ax[1, i].imshow(im2)
    # ax[0, i].set_title(name.upper(),fontsize=5)
    ax[0, i].axis("off")
    ax[1, i].axis("off")

filename = f"plots/compiled_plots/topology_spread.png"
print("Saving", filename)
plt.tight_layout()
plt.savefig(filename, dpi=1000)