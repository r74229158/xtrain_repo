import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import torch
import math

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from Utils.model_utils import load_model
from Utils.visualize_mnist import calc_vmin_vmax
from XAI_Applications.Utils.lrp import LRP_Implem
from XAI_Applications.Utils.captum import *
from XAI_Applications.attribution_methods import calculate_relevance_scores
from XAI_Applications.deletion_AUC import deletion_auc
from XAI_Applications.average_drop import evaluate_pixel_erasure

def benchmark_on_batch(model, data, labels, 
                       methods_list, 
                       num_R_save, results_pth, 
                       baseline_val, calc_auc=True, 
                       calc_ad=True):
    """
    Runs the deletion AUC benchmark and the Average Drop on a batch of examples.

    Args:
        model: The PyTorch model.
        data_loader: A DataLoader yielding (images, labels).
        methods_list (list): List of methods to test, e.g., ['integrated_gradients', 'my_method'].
        num_examples (int): Number of examples to run on.
        num_R_save (int): number of samples to save heatmaps.
        results_pth (str): Path to results.
        baseline_val (float): replacement value
        calc_auc (bool): If True, Deletion AUC is calculated
        calc_ad (bool): If True, Average Drop is calculated

    Returns:
        dict: A dictionary of results for each method.
    """
    root = math.sqrt(data.shape[1])
    root = int(root)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    r_scores = {method: [] for method in methods_list}
    
    auc_results = {method: [] for method in methods_list}
    ad_results = {method: [] for method in methods_list}

    data, labels = data.to(device), labels.to(device)
    model = model.to(device); model.eval()
    target_class = labels.to(torch.int)

    gather_path = results_pth + f"/Attr_Methods/"
    os.makedirs(gather_path, exist_ok=True)

    for method in methods_list:
        
        # Calculate relevance scores
        r_scores[method] = calculate_relevance_scores(method, model,
                                                      data, labels, 
                                                      results_pth)

        # Save visualization
        if num_R_save>0:
            save_heatmaps(method, r_scores[method][:num_R_save], root, gather_path)

        # Calculate Average Drop
        if calc_ad:
            ad_results[method] = evaluate_pixel_erasure(model, data, labels, r_scores[method])

        # Calculate AUC
        if calc_auc:
            for i, x in enumerate(data):
        
                auc_value, _ = deletion_auc(model, x, r_scores[method][i, :], 
                                            target_class[i], 
                                            baseline_val=baseline_val, 
                                            steps=200)
                try:
                    auc_results[method].append(auc_value)
                    
                except Exception as e:
                    print(f"Error with {method} on example {i}: {e}")
                    auc_results[method].append(np.nan) # Append NaN if there's an error
        
    save_results(calc_auc, calc_ad, auc_results, ad_results, results_pth)
    return r_scores


def save_heatmaps(method, r_scores, root, method_res_pth):
    """
    Saves heatmaps from the attribution scores, resulting from the application
    of some attribution method.
    """
    os.makedirs(method_res_pth, exist_ok=True) 
    for i in range(r_scores.shape[0]):

        R_minmax = calc_vmin_vmax([r_scores[i]])

        plt.figure(figsize=(8, 6), facecolor='none')
        plt.imshow(r_scores[i].reshape(root, root), cmap='coolwarm',
                    vmin=R_minmax[0][0], 
                    vmax=R_minmax[0][1])
        
        plt.tight_layout(); plt.axis('off')
        plt.savefig(method_res_pth+f"/{method}_sample_{i}.pdf", 
                    transparent=True, bbox_inches='tight', 
                    pad_inches=0, dpi=300); plt.close()


def save_results(calc_auc, calc_ad, auc_results, ad_results, results_pt):
    """Saves results from Average Drop and Average Deletion AUC in related files."""

    # Save AUC scores
    if calc_auc:
        with open(f"{results_pt}/auc_compare.txt", "a+") as f:

            for method, auc_list in auc_results.items():
                avg_auc = np.nanmean(auc_list)
                f.write(f"Average Deletion AUC for {method:20s}: {avg_auc:.4f} \n")
    
    # Save AD scores
    if calc_ad:
        with open(f"{results_pt}/ad_compare.txt", "a+") as f:

            for method, ad_score in ad_results.items():
                f.write(f"Average Drop for {method:20s}: {ad_score:.4f} \n")        


def test_methods(model_pth="DNNandTMNIST_run_0.pt", 
                 model_layers=[784, 400, 100, 10], 
                 data_pth="/typeface_mnist/run_0",
                 samples_num=64,
                 baseline_val=0.6,
                 methods = ['SHAP', 'LRP', 'GS', 'IG', 'DeepLIFT', 'Xtrain', 'Xlinear']):
    
    """Calculates the attribution scores for different attribution methods and 
    the Deletion AUC and Average Drop metrics.
    
    Args:
        model_pth (str): Path of the model
        model_layers [int]: model's architecture (needed to load the model)
        data_pth (str): Path of the data (saved by Xtrain algorithm)
        samples_num (int): Number of samples to save their resulting attribution scores as heatmaps
        baseline_val (float): the value replacing that of the features removed by Deletion AUC
        methods [str]: a list of attribution methods to calculate their"""

    data_pth = os.getcwd()+f"/Results/Datasets/{data_pth}"
    model = load_model(os.getcwd()+f"/Training/Trainer/Checkpoints/{model_pth}", model_layers)
    
    data = torch.load(data_pth+"/data.pt", weights_only=False)
    data = {'samples': data['samples'], 'labels': data['labels']}
     
    _ = benchmark_on_batch(model, data['samples'], data['labels'], 
                       methods, samples_num, data_pth, baseline_val)
