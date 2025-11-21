import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import torch
from matplotlib.collections import LineCollection
import math

import sys
sys.path.append(os.getcwd())
from Utils.model_utils import *; 

class SaveAndVisualize:


    def __init__(self, samples, labels, epochs, 
                 break_per_epochs, classes, 
                 run_pth, keep_all_changes=False, 
                 save_pdf=False):

        self.samples = samples
        self.labels = labels

        self.tot_pth = run_pth

        self.epochs = epochs
        self.break_per_epochs = break_per_epochs
        self.classes = classes

        self.save_pdf = save_pdf
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
                plt.savefig(new_pth + "/sample.pdf", transparent=True, bbox_inches='tight', 
                            pad_inches=0, dpi=300)

            else:
                plt.savefig(new_pth + "/sample.png", transparent=True, bbox_inches='tight', 
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
                
                self.save_rel_data(rel_norm, rel_targ, 
                                   rel_no_t, rel_t_po,
                                   i, None, 
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
                            i, ep, img_pth=img_pth)


    def save_rel_data(self, r, r_t, r_nt, r_t_pos, i, ep, img_pth=None, add_ep=True):

        min_vals = torch.min(torch.tensor([torch.min(r), torch.min(r_t), torch.min(r_nt)]))
        max_vals = torch.max(torch.tensor([torch.max(r), torch.max(r_t), torch.max(r_nt)]))

        min_max_vals = torch.min(torch.tensor([-torch.abs(torch.max(r)), 
                        -torch.abs(torch.max(r_t)), -torch.abs(torch.max(r_nt))]))
        max_min_vals = torch.max(torch.tensor([torch.abs(torch.min(r)), 
                        torch.abs(torch.min(r_t)), torch.abs(torch.min(r_nt))]))
        
        if add_ep:
            activations = self.samples_output[ep, i, :]
            bar_plot_scores(activations, self.classes, img_pth, ep)

        v_minmax = calc_vmin_vmax(r, r_t, r_nt, r_t_pos)
        
        for j, r in enumerate([r, r_t, r_nt, r_t_pos]):

            # vmin = torch.min(torch.tensor([min_vals, min_max_vals]))
            # vmax = torch.max(torch.tensor([max_vals, max_min_vals]))

            # if self.samples_output != None and bat==None:
            #     self.plot_activations(activations, r, vmin, vmax); continue

            plt.figure(figsize=(8, 6), facecolor='none')
            plt.imshow(r.view(self.size, self.size), vmin=v_minmax[j][0], 
                       vmax=v_minmax[j][1], cmap='coolwarm')
            plt.axis('off')

            lab = "rel" if j==0 else "rel_target" if j==1 else "rel_non_target" \
                if j==2 else "rel_target_pos"
            
            if add_ep: lab += f"_ep_{ep}"

            plt.tight_layout()
            if self.save_pdf:
                plt.savefig(img_pth + f"/{lab}.pdf", transparent=True, bbox_inches='tight', 
                            pad_inches=0, dpi=300)
            else:
                plt.savefig(img_pth + f"/{lab}.png", transparent=True, bbox_inches='tight', 
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
    ### ---------------------------------------------------------


def bar_plot_scores(activations, classes, pth, epoch, save_pdf=False):

    fig, ax = plt.subplots()

    bars = ax.bar(range(len(activations)), 
                activations, color='skyblue', 
                edgecolor=None, alpha=0.7)

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) 

    ax.set_xticks(range(classes))
    ax.set_xticklabels(range(classes), ha='center')
    ax.set_ylim(0, 1)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(-0.5, len(activations) - 0.5)

    max_idx = np.argmax(activations)
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(0.8)

    plt.tight_layout()
    
    if save_pdf:
        plt.savefig(pth+f"/activs_{epoch}.pdf",dpi=300, bbox_inches='tight')
    else:
        plt.savefig(pth+f"/activs_{epoch}.png")
        
    plt.close()

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

def process_rel_feats(rel_scores):

    max_val = torch.abs(rel_scores).amax(dim=1)
    max_val = max_val.view(max_val.shape[0], 1, max_val.shape[1])

    # add_mask = max_val < 1e-8
    # max_val[add_mask] += 1e-8
  
    return rel_scores / max_val

def calc_input_edges(X):
    """Edge detection for input X"""

    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if X.shape[0] == 1:
        X = X.squeeze(0)

    if len(X.shape)==2:
        X = X.reshape(X.shape[0], 28, 28)

    X_edg = ((X + 1)/2)
    X_edg = (X_edg * 255).astype(np.uint8)
    _, binary_digit = cv2.threshold(
        X_edg, 30, 255, cv2.THRESH_BINARY
    )
    
    binary_digit = binary_digit.astype(float) / 255.0  # Normalize to [0, 1]

    X_edg = X_edg.transpose(1,2,0) 
    X_resized = cv2.resize(X_edg, (56, 56), interpolation=cv2.INTER_LINEAR)

    edges = cv2.Canny(X_resized, 50, 100)
    edges = cv2.resize(edges, (28, 28))

    edges[(0 < edges) & (edges < 128)] = 128
    edges[(128 <= edges) & (edges < 256)] = 255

    return edges * binary_digit


### --------------
### --- DEPREC ---
### --------------

def test_interm_layers(samples_R):
    """
    Only in case when relevance scores of intermediate layers
    are calculated.
    """

    R_layer_2 = samples_R[2]
    R_layer_3 = samples_R[3]

    pth = os.getcwd() + "/Results/model"
    os.makedirs(pth, exist_ok=True)

    for nm, layer in zip(["layer_2", "layer_3"], [R_layer_2, R_layer_3]):

        layer_pth = pth+f"/{nm}"
        os.makedirs(layer_pth, exist_ok=True)

        for i in range(layer.shape[0]):
            
            vals = layer[i, :, :]
            fig, ax = plt.subplots(figsize=(10, 6))

            n, feats = layer.shape[-1], layer.shape[1]

            # Vertical spacing between lines
            offset = 2
            y_offsets = np.arange(n * offset) 

            # Colormap setup
            norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
            cmap = plt.cm.coolwarm

            # Plot each line
            for j in range(n):

                x = np.arange(feats)
                y = (np.ones(feats) + y_offsets[j])/4  # Shift line vertically
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Assign colors to segments
                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=10)
                lc.set_array(vals[:, j].numpy())
                ax.add_collection(lc)

            # Add colorbar
            plt.colorbar(lc, ax=ax, label='Value')

            # Adjust axes
            ax.set_xlim(0, layer.shape[1])
            ax.set_ylim(-1, layer.shape[-1] * offset + 1)
            ax.set_yticks(y_offsets + 1)  # Center ticks on lines

            ax.axis('off')
            plt.savefig(layer_pth+f"/sample_{i}")
            plt.close()

def visualize_gif(samples, labels, rel_scores_feat, edges, pth):

    for i in range(samples.shape[0]):

        lbl = labels[i] 

        new_pth = os.path.join(pth, f"{lbl}/sample_{i}")
        os.makedirs(new_pth, exist_ok=True)

        fig = plt.figure(figsize=(10, 5))

        # Left subplot (static)
        ax_left = fig.add_subplot(1, 2, 1)
        left_display = ax_left.imshow(samples[i, :].view(28, 28), cmap='gray', animated=True)
        ax_left.set_title("Sample")
        ax_left.axis('off')

        # Right subplot (dynamic)
        rel_scores = rel_scores_feat[i, :, :]

        ax_right = fig.add_subplot(1, 2, 2)
        right_display = ax_right.imshow(rel_scores[::, 0].view(28, 28), 
                                        cmap='coolwarm', 
                                        animated=True,
                                        vmin=-1,
                                        vmax=1)
        ax_right.set_title("Relevance Evolution")
        ax_right.axis('off')

        # Animation update function
        def update(frame):
            right_display.set_data(rel_scores[::, frame].view(28, 28))
            right_display.set_clim(vmin=-1.0, vmax=1.0)
            return left_display, right_display

        # Create animation
        anime = ani.FuncAnimation(
            fig, update, frames=rel_scores.shape[-1], 
            interval=2000,  # Time delay in ms
            blit=True
        )

        # Save as GIF
        anime.save(os.path.join(new_pth, f"sample_{i}_rel.gif"), writer='pillow', fps=2)
        plt.close()
