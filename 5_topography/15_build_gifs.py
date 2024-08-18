#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:48:01 2024

@author: mtrubshaw
"""

from PIL import Image
import glob
import os

os.makedirs('plots/topo/ind_activations/1_gifs',exist_ok=True)
freqs = ['gamma','hgamma']
for fr in freqs:
    for s in range(88):

        # for f in freqs:
        # List all image filenames in the directory
        image_files = []
        for i in range(60):
            image_files.append(f'plots/topo/ind_activations/sub{s}_{fr}_plot{i:02d}.png')
        
        # image_files = sorted(glob.glob())  # Change '*.png' to match your image format
        
        # Open all images and append to a list
        images = []
        for filename in image_files:
            images.append(Image.open(filename))
        
        # Save the images as a GIF
        images[0].save(f'plots/topo/ind_activations/1_gifs/sub_{s}_{fr}.gif',
                    save_all=True,
                    append_images=images[1:],
                    duration=600,  # Adjust duration as needed (in milliseconds)
                    loop=0)  # Set loop to 0 for infinite loop, or any other number for a finite loop

                # for i in range(60):
                #     image_files.append(f'plots/topo/{group}/{f}{i:02d}_ubl.png')
                
                # # Open all images and append to a list
                # images = []
                # for filename in image_files:
                #     images.append(Image.open(filename))
                
                # # Save the images as a GIF
                # images[0].save(f'plots/topo/gifs/{f}_ubl_{group}.gif',
                #                save_all=True,
                #                append_images=images[1:],
                #                duration=600,  # Adjust duration as needed (in milliseconds)
                #                loop=0)  # Set loop to 0 for infinite loop, or any other number for a finite loop
                
