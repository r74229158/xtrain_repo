import os,  cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.collections import LineCollection

from Utils.model_utils import *; 

class SaveAndVisualize:
    """A custom class for basic visualization utilities."""
    
    def __init__(self, samples, labels, 
                 epochs, classes, run_pth, 
                 num_save=20):

        self.samples = samples
        self.labels = labels

        self.tot_pth = run_pth

        self.epochs = epochs
        self.classes = classes

        self.num_save = num_save
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
    
    def save_scores(self, r_scores, 
                    r_target, r_target_pos, 
                    r_non_target, r_linr,
                    samples_output):
        
        if r_scores.shape[0] >= self.num_save:
            
            self.r_scores = r_scores[:self.num_save]
            self.r_target = r_target[:self.num_save]

            self.r_target_pos = r_target_pos[:self.num_save]
            self.r_non_target = r_non_target[:self.num_save]

            self.r_linr = r_linr[:self.num_save]
            self.samples_output = samples_output[:self.num_save]

        if samples_output != None:

            torch.save(samples_output, os.path.join(self.tot_pth, "samples_output.pt"))
    
        torch.save(r_scores, os.path.join(self.tot_pth, f"xt_relevance.pt"))
        torch.save(r_target, os.path.join(self.tot_pth, f"relevance_target.pt"))
        torch.save(r_target_pos, os.path.join(self.tot_pth, f"relevance_target_pos.pt"))
        torch.save(r_non_target, os.path.join(self.tot_pth, f"relevance_non_target.pt"))
        torch.save(r_linr, os.path.join(self.tot_pth, f"xlin_relevance.pt"))


def calc_vmin_vmax(attrs):

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
