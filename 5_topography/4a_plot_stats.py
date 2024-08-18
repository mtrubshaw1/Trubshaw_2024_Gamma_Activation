
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.stats import kruskal
import numpy as np
import pandas as pd
import os

os.makedirs('plots/topo/activations',exist_ok=True)

# Load regressor data
demographics = pd.read_csv("../demographics/task_demographics.csv")
group_ = demographics["Group"].values

category_list = demographics["Group"].values
category_list[category_list == "HC"] = 1
category_list[category_list == "ALS"] = 2
category_list[category_list == "rALS"] = 2
category_list[category_list == "PLS"] = 3
category_list[category_list == "rPLS"] = 3
category_list[category_list == "FDR"] = 3
category_list[category_list == "rFDR"] = 3


# data structure is "times (early, mid,late)", "positivity (higher, lower)", "frequencies(beta,gamma)"
adj = 0.5
plot_gif=True
pos = ['hyper', 'hypo']
freqs = ['beta', 'gamma', 'hgamma']


for h, po in enumerate(pos):
    for g, freq in enumerate(freqs):
        group_labels = np.arange(0,1600,100)
        tstats = np.squeeze(np.load(f'data/contrast_0_tstats.npy'))[:,h,g]
        pvalues = np.squeeze(np.load('data/contrast_0_pvalues.npy'))[:,h,g]
        data = np.load('data/region_count.npy',allow_pickle=True)[:,:,h,g]
        data_als = np.mean(data[group_==2],axis=0)
        data_hc = np.mean(data[group_==1],axis=0)
        data_diff = data_als-data_hc

        
        #can also plot tstats
        values = data_diff
        n = len(values)
        
        x = range(-n // 2, n // 2)
        
        barplot = sns.barplot(x=list(x), y=values, palette='pastel')
        
        max_abs_value = max(max(values), abs(min(values)))
        plt.ylim(-max_abs_value * 1.2, max_abs_value * 1.2)
        plt.axhline(0, color='black', linewidth=0.8)
        
        # plt.xlabel('Contrasts', fontsize=12)
        # plt.title(f'Difference in total number of {po}activated regions \n- {freq} band (ALS-HC)',fontsize=14)
        plt.ylabel('Difference (ALS-HC)',fontsize=12)
        plt.ylim(-10,20)
        plt.xlabel('Time (s)',fontsize=12)
        
        plt.axvline(x=2.5-adj, color='red', linestyle='--', linewidth=1, label='Trigger ON')
        plt.axvline(x=10-adj, color='#eb6134', linestyle=':', linewidth=1, label='Trigger OFF')

        plt.xticks(np.arange(len(group_labels))-adj, group_labels/250, fontsize=8)
        
        legend_labels = ['* = p < 0.05', '** = p < 0.01', '*** = p < 0.001']
        handles = [plt.Line2D([0], [0], color='w', markerfacecolor='red', markersize=10, label=label) for label in legend_labels]
        
        # plt.legend(handles=handles, labels=legend_labels, loc='lower right', fontsize=8)
        
        # d_als = data[group_==2]
        # std_diff = np.std(d_als,axis=0)
        # plt.errorbar(x=np.arange(15), y=data_diff, yerr=std_diff, fmt='none', capsize=5, color='black')
        
        for i in range(n):
            if pvalues[i] < 0.001:
                plt.text(i, values[i], '***', ha='center', va='baseline', color='red', fontsize=12)
            elif pvalues[i] < 0.01:
                plt.text(i, values[i], '**', ha='center', va='baseline', color='red', fontsize=12)
            elif pvalues[i] < 0.05:
                plt.text(i, values[i], '*', ha='center', va='baseline', color='red', fontsize=12)
        # plt.tight_layout()
        # plt.legend(fontsize=8)
        plt.savefig(f"plots/topology_spread_{po}_{freq}.png",dpi=300)

        # plt.tight_layout()
        plt.show()
        # plt.close()  # Close the figure to release memory and avoid overlap in the next figure
        
        if plot_gif==True:
            for j in range(60):
            
                group_labels = np.arange(0,1600,100)
                tstats = np.squeeze(np.load(f'data/contrast_0_tstats.npy'))[:,h,g]
                pvalues = np.squeeze(np.load('data/contrast_0_pvalues.npy'))[:,h,g]
                values = data_diff
                n = len(values)
                
                x = range(-n // 2, n // 2)
                
                barplot = sns.barplot(x=list(x), y=values, palette='pastel')
                
                max_abs_value = max(max(values), abs(min(values)))
                plt.ylim(-max_abs_value * 1.2, max_abs_value * 1.2)
                plt.axhline(0, color='black', linewidth=0.8)
                
                # plt.xlabel('Contrasts', fontsize=12)
                plt.title(f'Difference in total number of {po}activated regions \n- {freq} band (ALS-HC)',fontsize=14)
                plt.ylabel('Difference',fontsize=12)
                plt.ylim(-10,20)
                plt.xlabel('Time (seconds)',fontsize=12)
                
                plt.axvline(x=2.5-adj, color='red', linestyle='--', linewidth=1, label='Trigger')
                plt.axvline(x=10-adj, color='red', linestyle='--', linewidth=1, label='Trigger')
            
                plt.axvline(x=(j/4)-adj, color='green', linewidth=1)
                plt.xticks(np.arange(len(group_labels))-adj, group_labels/250, fontsize=8)
                
                legend_labels = ['* = p < 0.05', '** = p < 0.01', '*** = p < 0.001']
                handles = [plt.Line2D([0], [0], color='w', markerfacecolor='red', markersize=10, label=label) for label in legend_labels]
                
                plt.legend(handles=handles, labels=legend_labels, loc='lower right', fontsize=5)
                
                for i in range(n):
                    if pvalues[i] < 0.001:
                        plt.text(i, values[i], '***', ha='center', va='baseline', color='red', fontsize=12)
                    elif pvalues[i] < 0.01:
                        plt.text(i, values[i], '**', ha='center', va='baseline', color='red', fontsize=12)
                    elif pvalues[i] < 0.05:
                        plt.text(i, values[i], '*', ha='center', va='baseline', color='red', fontsize=12)
                # plt.tight_layout()
            
                plt.savefig(f"plots/topo/activations/{freq}_{po}_plot_lines{j:02d}.png",dpi=300)
                # plt.tight_layout()
                plt.close()
