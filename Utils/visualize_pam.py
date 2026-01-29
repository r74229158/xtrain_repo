import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

sys.path.append(os.getcwd())
from Utils.model_utils import *; 
from Training.Data.Modules.datasets.pam_50 import PAM50Dataset
from XAI_Applications.evaluation import benchmark_on_batch
from Utils.visualize import SaveAndVisualize, calc_vmin_vmax


class VisualizePAM(SaveAndVisualize):


    def __init__(self, samples, labels, epochs, classes, run_pth, num_save=20):

        super().__init__(samples, labels, epochs, classes, run_pth, num_save)
        self.gene_names = PAM50Dataset().get_gene_names()

    
    def save_logs(self, epoch, epoch_acc, epoch_class_hist):

        with open(os.path.join(self.tot_pth, "accuracy.txt"), "a+") as f:
            for i, ep in enumerate(range(self.epochs)):  

                f.write(f"\nEpoch : {ep+1}\n")
                f.write(f"Accuracy : {epoch_acc[i]} \n")

                if epoch_class_hist != None:
                    if len(epoch_class_hist[ep]) > 0:
                        for k in range(len(epoch_class_hist[1])):
                            f.write(f"\tClass {k}, accuracy: {epoch_class_hist[ep][k]}\n")


def save_heatmaps_simple(r_scores, r_targ, indices, tot_pth, plot_together=False):

    r_minmax = calc_vmin_vmax([r_scores, r_targ])
    if plot_together:

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 3), 
                                sharex=True,  # Share x-axis
                                gridspec_kw={'hspace': 0.1})
        
        custom_plot(r_scores, r_minmax[0], indices[0], ax1)
        custom_plot(r_targ, r_minmax[1], indices[1], ax2, True)

        # Set titles and labels
        plt.tight_layout(pad=4.0)

        plt.savefig(tot_pth + f"/xtrain_group.pdf", transparent=True, 
                    bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    else:
        for id, r in enumerate([r_scores, r_targ]):

            plt.figure(figsize=(20, 6), facecolor='none')
            custom_plot(r, r_minmax[id], indices[id], None)

            # Set titles and labels
            plt.xlabel('Gene Features', fontsize=32)
            plt.tight_layout(pad=4.0)

            lab = "rel" if id==0 else "rel_target_pos"

            plt.savefig(tot_pth + f"/{lab}.pdf", transparent=True, 
                        bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()


def custom_plot(r_scores, r_minmax, idx_labels, ax, last_ax=False):

    df_heatmap = pd.DataFrame(r_scores, index=idx_labels, columns=PAM50Dataset().get_gene_names())
    custom_cmap = sns.diverging_palette(350, 150, s=80, l=55, as_cmap=True)

    mappable = sns.heatmap(df_heatmap, ax=ax, cbar=False, cmap=custom_cmap, 
                vmin=r_minmax[0], 
                vmax=r_minmax[1])
    
    if ax==None:
        return
    
    if last_ax==False:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', length=0)
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10,  fontfamily='DejaVu Sans')
    if last_ax: 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12,  fontfamily='DejaVu Sans')

    return mappable

def save_heatmaps(r_scores, r_targ, indices, tot_pth, probe_num=4, normalize=True):

    # Normalize scores
    r_scores[0, ::] = r_scores[0, ::]/8
    r_scores[1, ::] = r_scores[1, ::]/70
    r_scores[2, ::] = r_scores[2, ::]*2
    r_scores[3, ::] = r_scores[3, ::]*2
    r_scores[4, ::] = r_scores[4, ::]/5
    r_scores[5, ::] = r_scores[5, ::]*5

    r_minmax = calc_vmin_vmax([r_scores, r_targ])

    if normalize:
        
        r_minmax[0] = [-1, 1]
        for i in range(r_scores.shape[0]):
            
            for j in range(r_scores.shape[1]):
                r_scores[i, j, :] = 2 * (r_scores[i, j, :] - torch.min(r_scores[i, :, :])) / (torch.max(r_scores[i, :, :]) - torch.min(r_scores[i, :, :])) - 1

    f, ax_list = plt.subplots(probe_num+1, 1, figsize=(18, 24), 
                            sharex=True,  # Share x-axis
                            gridspec_kw={'hspace': 0.1})

    for i in range(probe_num):
        mappable = custom_plot(r_scores[:, i, :], r_minmax[0], indices[0], ax_list[i])
    f.colorbar(mappable.collections[0], ax=ax_list[:-1], fraction=0.05, pad=0.02)

    mappable = custom_plot(r_targ, r_minmax[1], indices[1], ax_list[-1], True)
    f.colorbar(mappable.collections[0], ax=ax_list[-1], fraction=0.05, pad=0.02)

    # Set titles and labels
    plt.tight_layout(pad=4.0)
    plt.savefig(tot_pth + f"/group.pdf", transparent=True, 
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def compare_attributions(run_num=0, probe_num=3, pos_num=10):
        
    # Load model
    model = get_model_architecture("DNNandPAM50", [50, 32, 8, 2])
    model.load_state_dict(torch.load(os.getcwd() + f"/Training/Trainer/Checkpoints/DNNandPAM50_run_{run_num}.pt"))

    # Load data
    data_pth = os.getcwd() + f"/Results/Datasets/pam50/run_{run_num}"
    data = torch.load(data_pth+"/data.pt")
    X, labels = data['samples'], data['labels']

    R_xtr_pos = torch.load(data_pth+"/relevance_target_pos.pt")
    R_xtr_pos = R_xtr_pos[:pos_num, :]

    methods = ['SHAP', 'LRP', 'DeepLIFT', 'IG', 'Xlinear', 'Xtrain'] 
    R = benchmark_on_batch(model, X, labels, methods, 0, data_pth, 0, 
                           calc_auc=False, calc_ad=False)

    methods = list(R.keys())
    r_scores = torch.zeros(len(methods), probe_num, 50)

    for i, key in enumerate(list(R.keys())):
        r_scores[i, :, :] = R[key][:probe_num, :]
    
    indices = [methods, range(R_xtr_pos.shape[0])]
    save_heatmaps(r_scores, R_xtr_pos, indices, data_pth, probe_num)
            