import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import torch
from matplotlib.collections import LineCollection
import math

from Utils.visualize import SaveAndVisualize, calc_input_edges, bar_plot_scores, calc_vmin_vmax
from Utils.model_utils import *; 

class VisualizeMNIST(SaveAndVisualize):


    def __init__(self, samples, labels, epochs, 
                 break_per_epochs, classes, 
                 run_pth, num_save=20, 
                 save_pdf=False):

        super().__init__(samples, labels, epochs, classes, run_pth, num_save)

        self.save_pdf = save_pdf
        self.break_per_epochs = break_per_epochs
    

    def save_logs(self, epoch, epoch_acc, epoch_class_hist):

        if self.break_per_epochs != None:
            res = epoch%self.break_per_epochs
            diff = res if res>0 else epoch - self.break_per_epochs
            
            self.start_epoch = epoch - diff
            self.end_epoch = epoch
        
        else:
            self.start_epoch = 0
            self.end_epoch = self.epochs-1

        j = -1
        with open(os.path.join(self.tot_pth, "accuracy.txt"), "a+") as f:
            for i, ep in enumerate(range(self.start_epoch, self.end_epoch+1)):  

                if len(epoch_acc)<(self.end_epoch+1 - self.start_epoch) and i == 0:
                    continue 

                j += 1
                f.write(f"\nEpoch : {ep+1}\n")
                f.write(f"Accuracy : {epoch_acc[j]} \n")

                if epoch_class_hist != None:
                    if len(epoch_class_hist[ep]) > 0:
                        for k in range(len(epoch_class_hist[0])):
                            f.write(f"\tClass {k}, accuracy: {epoch_class_hist[ep][k]}\n")


    def save_imgs(self, save_all_images=False, plot_edges=False):
        """
        Saves all images (for all steps of the process), or only images from the
        last step.
        """

        if plot_edges:
            self.edges = calc_input_edges(self.samples)

        for i in range(self.r_scores.shape[0]):

            self.size = int(math.sqrt(self.samples.shape[1]))

            lbl = self.labels[i].to(torch.int16).item() 
            new_pth = os.path.join(self.tot_pth, f"{lbl}/sample_{i}")
            os.makedirs(new_pth, exist_ok=True)

            # Save image
            plt.figure(figsize=(8, 6), facecolor='none')
            plt.imshow(self.samples[i, :].view(self.size, self.size))
            plt.tight_layout(); plt.axis('off') 

            if self.save_pdf:
                plt.savefig(new_pth + f"/sample_{i}.pdf", transparent=True, bbox_inches='tight', 
                            pad_inches=0, dpi=300)

            else:
                plt.savefig(new_pth + f"/sample_{i}.png", transparent=True, bbox_inches='tight', 
                            pad_inches=0, dpi=300)
            plt.close()

            # Save images for each step
            if save_all_images:
                
                img_pth = new_pth + "/all_images"
                os.makedirs(img_pth, exist_ok=True)
                
                self.save_all_images(i, img_pth)
            
            # Save images for only the last step
            else:

                rel_norm = self.r_scores[i, :]
                rel_targ = self.r_target[i, :]

                rel_no_t = self.r_non_target[i, :]
                rel_t_po = self.r_target_pos[i, :]
                rel_linr = self.r_linr[i, :]
                
                self.save_rel_data(rel_norm, rel_targ, 
                                   rel_no_t, rel_t_po,
                                   rel_linr,
                                   i=i, ep=None, 
                                   img_pth=new_pth,
                                   add_ep=False)

                    
    def save_all_images(self, i, img_pth):

        for id, ep in enumerate(range(self.start_epoch, self.end_epoch+1)):

            rel_norm = self.r_scores[i, :, id]
            rel_targ = self.r_target[i, :, id]

            rel_no_t = self.r_non_target[i, :, id]
            rel_t_po = self.r_target_pos[i, :, id]

            self.save_rel_data(rel_norm, rel_targ, 
                               rel_no_t, rel_t_po,
                            i=i, ep=ep, img_pth=img_pth)


    def save_rel_data(self, *args, i=0, ep=0, img_pth=None, add_ep=False):
        
        if add_ep:
            activations = self.samples_output[ep, i, :]
            bar_plot_scores(activations, self.classes, img_pth, ep)

        v_minmax = calc_vmin_vmax(args)
        
        for j, r in enumerate(args):

            # vmin = torch.min(torch.tensor([min_vals, min_max_vals]))
            # vmax = torch.max(torch.tensor([max_vals, max_min_vals]))

            # if self.samples_output != None and bat==None:
            #     self.plot_activations(activations, r, vmin, vmax); continue

            plt.figure(figsize=(8, 6), facecolor='none')
            plt.imshow(r.view(self.size, self.size), vmin=v_minmax[j][0], 
                       vmax=v_minmax[j][1], cmap='coolwarm')
            plt.axis('off')
                
            new_lab = "rel" if j==0 else "rel_target" if j==1 else "rel_non_target" \
                if j==2 else "rel_target_pos" if j==3 else "rel_lin"
            
            new_lab += f"_{i}"
            if add_ep: new_lab += f"_ep_{ep}"

            plt.tight_layout()
            if self.save_pdf:
                plt.savefig(img_pth + f"/{new_lab}.pdf", transparent=True, bbox_inches='tight', 
                            pad_inches=0, dpi=300)
            else:
                plt.savefig(img_pth + f"/{new_lab}.png", transparent=True, bbox_inches='tight', 
                            pad_inches=0, dpi=300)
            plt.close()
    

    ### ---------------------------------------------------------
    ### ---------------------------------------------------------

    @DeprecationWarning
    def attr_all_params(self):

        self.relevance_scores = {}
        self.relevance_target = {}
        self.relevance_non_target = {}

        for i in range(len(self.r_scores)):
            self.relevance_scores[f'layer_{i}'] = self.r_scores[i]
            self.relevance_target[f'layer_{i}'] = self.r_target[i]
            self.relevance_non_target[f'layer_{i}'] = self.r_non_target[i]

        self.r_scores = self.r_scores[0]
        self.r_target = self.r_target[0]
        self.r_non_target = self.r_non_target[0]

        torch.save(self.relevance_scores, os.path.join(self.tot_pth, f"relevance.pt"))
        torch.save(self.relevance_target, os.path.join(self.tot_pth, f"relevance_target.pt"))
        torch.save(self.relevance_non_target, os.path.join(self.tot_pth, f"relevance_non_target.pt"))

    @DeprecationWarning
    def plot_activations(self, activations, r, vmin, vmax):
    
        fig, (ax1, ax2) = plt.subplots(2, 1, 
                            figsize=(7, 9), 
                            gridspec_kw={'height_ratios': [5, 1], 
                                            'hspace':0})

        
        ax1.imshow(r.view(self.size, self.size), vmin=vmin, vmax=vmax, cmap='coolwarm')
        # ax1.imshow(self.edges[i], cmap='gray', alpha=0.5); 
        ax1.axis('off')

        bars = ax2.bar(range(len(activations)), 
                    activations, color='skyblue', 
                    edgecolor=None, alpha=0.7)
        
        ax2.grid(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False) 
        
        ax2.set_xticks(range(self.classes))
        ax2.set_xticklabels(range(self.classes), ha='center')
        ax2.set_ylim(0, 1)
        ax2.axes.get_yaxis().set_visible(False)
        ax2.set_xlim(-0.5, len(activations) - 0.5)

        max_idx = np.argmax(activations)
        bars[max_idx].set_color('red')
        bars[max_idx].set_alpha(0.8)

    ### ---------------------------------------------------------

