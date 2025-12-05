import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

sys.path.append(os.getcwd())
from Utils.model_utils import *; 
from Training.Data.Modules.datasets.pam_50 import PAM50Dataset
from XAI_Applications.deletion_AUC import benchmark_on_batch


class VisualizePAM:


    def __init__(self, samples, labels, epochs, 
                 classes, run_pth):

        self.samples = samples
        self.labels = labels

        self.tot_pth = run_pth

        self.epochs = epochs
        self.classes = classes

        self.gene_names = PAM50Dataset().get_gene_names()
        self.save_data()


    def save_data(self):

        print("Saving results in", self.tot_pth)
        os.makedirs(self.tot_pth, exist_ok=True)

        for i in range(self.classes):
            os.makedirs(self.tot_pth+f"/{i}", exist_ok=True)

        self.data = {
            'samples': self.samples,
            'labels': self.labels,
        }

        torch.save(self.data, os.path.join(self.tot_pth, "data.pt"))
    
    def save_logs(self, epoch, epoch_acc, epoch_class_hist):

        with open(os.path.join(self.tot_pth, "accuracy.txt"), "a+") as f:
            for i, ep in enumerate(range(self.epochs)):  

                f.write(f"\nEpoch : {ep+1}\n")
                f.write(f"Accuracy : {epoch_acc[i]} \n")

                if epoch_class_hist != None:
                    if len(epoch_class_hist[ep]) > 0:
                        for k in range(len(epoch_class_hist[1])):
                            f.write(f"\tClass {k}, accuracy: {epoch_class_hist[ep][k]}\n")

    def save_scores(self, r_scores, r_target, r_target_pos, r_non_target, 
                    samples_output):
        
        self.r_scores = r_scores
        self.r_target = r_target

        self.r_target_pos = r_target_pos
        self.r_non_target = r_non_target

        self.samples_output = samples_output
        if samples_output != None:
            torch.save(samples_output, os.path.join(self.tot_pth, "samples_output.pt"))
    
        torch.save(r_scores, os.path.join(self.tot_pth, f"relevance.pt"))
        torch.save(r_target, os.path.join(self.tot_pth, f"relevance_target.pt"))
        torch.save(r_target_pos, os.path.join(self.tot_pth, f"relevance_target_pos.pt"))
        torch.save(r_non_target, os.path.join(self.tot_pth, f"relevance_non_target.pt"))


def save_heatmaps(r_scores, r_targ, idx, tot_pth, plot_together=False, st=4):

    num_samples = len(idx[0])
    r_scores = r_scores[:num_samples, :]
    r_targ = r_targ[:num_samples, :]
    index = idx # [str(i+1) for i in range(num_samples)]

    r_minmax = calc_vmin_vmax(r_scores, r_targ)
    if plot_together:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 3), 
                                sharex=True,  # Share x-axis
                                gridspec_kw={'hspace': 0.1})
        # ax1.set_aspect('auto') 
        # ax2.set_aspect('auto')
        
        df_heatmap = pd.DataFrame(r_scores, index=idx[0], columns=PAM50Dataset().get_gene_names())
        custom_cmap = sns.diverging_palette(350, 150, s=80, l=55, as_cmap=True)

        sns.heatmap(df_heatmap, ax=ax1, cbar=True, cmap=custom_cmap, 
                    vmin=r_minmax[0][0], 
                    vmax=r_minmax[0][1])
        
        ax1.set_xticklabels([])
        ax1.tick_params(axis='x', which='both', length=0) 
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10,  fontfamily='DejaVu Sans')

        df_heatmap2 = pd.DataFrame(r_targ, index=idx[1], columns=PAM50Dataset().get_gene_names())
        custom_cmap = sns.diverging_palette(350, 150, s=80, l=55, as_cmap=True)

        sns.heatmap(df_heatmap2, ax=ax2, cbar=True, cmap=custom_cmap, 
                    vmin=r_minmax[1][0], 
                    vmax=r_minmax[1][1])
        
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=12,  fontfamily='DejaVu Sans')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=10,  fontfamily='DejaVu Sans')

        # ax1.set_aspect(2.0)
        # ax2.set_aspect(2.0)

        # Set titles and labels
        plt.tight_layout(pad=4.0)

        plt.savefig(tot_pth + f"/group.pdf", transparent=True, 
                    bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    else:
        for id, r in enumerate([r_scores, r_targ]):

            # Put it in a DataFrame (this is the easiest way to feed seaborn)
            df_heatmap = pd.DataFrame(r, index=index[id], columns=PAM50Dataset().get_gene_names())
            custom_cmap = sns.diverging_palette(350, 150, s=80, l=55, as_cmap=True)

            plt.figure(figsize=(20, 6), facecolor='none')
            ax = sns.heatmap(df_heatmap, cmap=custom_cmap, 
                                vmin=r_minmax[id][0], 
                                vmax=r_minmax[id][1],
                                cbar=True, annot=False)
            
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=24,  fontfamily='DejaVu Sans')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=24,  fontfamily='DejaVu Sans')

            # Set titles and labels
            plt.xlabel('Gene Features', fontsize=32)
            plt.tight_layout(pad=4.0)

            lab = "rel" if id==0 else "rel_target_pos"

            plt.savefig(tot_pth + f"/{lab}.pdf", transparent=True, 
                        bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()


def compare_attributions(run_num=0):
        
    # Load model
    model_name = "DNNandPAM50"
    model_conf = [50, 32, 8, 2]
    model = get_model_architecture(model_name, model_conf)
                    
    model_pth = os.getcwd() + f"/Training/Trainer/Checkpoints/DNNandPAM50_run_{run_num}.pt"
    model.load_state_dict(torch.load(model_pth))

    # Load data
    data_pth = os.getcwd() + f"/Results/Datasets/pam50/run_{run_num}"
    data = torch.load(data_pth+"/data.pt")
    X, labels = data['samples'], data['labels']

    R_xtr = torch.load(data_pth+"/relevance.pt")
    R_xtr_pos = torch.load(data_pth+"/relevance_target_pos.pt")

    methods = ['IG', 'DeepLIFT', 'LRP', 'XtrAIn', 'GradSHAP', 'SHAP'] 

    R = benchmark_on_batch(model, X, labels, methods, False, data_pth, 0, skip_auc=True)
    R['XtrAIn'] = R_xtr

    methods = list(R.keys())
    R_gather = torch.zeros(len(methods), 50)

    idx = 6
    for i, key in enumerate(list(R.keys())):

        R_gather[i, :] = R[key][idx, :]
    

    indices = [methods, range(len(methods))]
    save_heatmaps(R_gather, R_xtr_pos, indices, data_pth, True)
            

def calc_vmin_vmax(*attrs):

    l = []
    for attr in attrs:
        min, max = torch.min(attr), torch.max(attr)
        
        if min*max > 0:
            
            if max > 0:
                vmin = -max
                vmax = max
            else:
                vmin = min
                vmax = -min
        
        else:
            vmin = torch.min(min, -max)
            vmax = torch.max(-min, max)

        l.append((vmin, vmax))
        
    return l