import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import torch
import math

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from Utils.model_utils import load_model
from XAI_Applications.Utils.lrp import LRP_Implem
from XAI_Applications.Utils.captum import *
from Utils.visualize_utils import calc_vmin_vmax


def deletion_auc(model, input_tensor, attribution_map, target_class, baseline_val=-1, steps=200):
    """
    Computes the Deletion AUC metric.
    A lower AUC (faster performance drop) means a better attribution map.

    Args:
        model: The PyTorch model.
        input_tensor (torch.Tensor): Original input image (1, C, H, W).
        attribution_map (np.ndarray): Attribution scores (H, W) or (C, H, W). Will be summed across channels.
        target_class (int): The class index to monitor.
        baseline_val (int): The baseline value that will replace that of the targeted features
        steps (int): Number of steps in the deletion process.

    Returns:
        float: The AUC of the deletion curve.
        np.ndarray: The deletion curve values.
    """
    with torch.no_grad():

        input_batch = input_tensor.clone().requires_grad_(False)
        root = math.sqrt(input_batch.shape[0]) if len(input_batch.shape)==1 else math.sqrt(input_batch.shape[1])
        root = int(root)
        if len(input_batch.shape)==1: 
            input_batch = input_batch.reshape(root, root).unsqueeze(0)
        
        target_class = target_class.to(torch.int)
        original_pred = model(input_batch, apply_softmax=False)
        original_score = torch.softmax(original_pred, dim=1)[0, target_class.item()]
        
        attribution_map = attribution_map.reshape(root, root)
        attr_flat = attribution_map.flatten().cpu()
        
        # Get the indices of the most important pixels first
        sorted_indices = torch.flip(np.argsort(attr_flat), [0]) # Descending order: most important first

        # Create a mask of ones (all pixels present)
        h, w = attribution_map.shape
        curve_scores = []

        # Gradually delete pixels from most to least important
        for i in range(0, steps + 1):
            
            # Calculate how many pixels to delete at this step
            pixels_to_delete = int(len(sorted_indices) * (i / steps))

            mask = torch.ones(h * w).to(input_batch.device)
            if pixels_to_delete > 0:
                mask[sorted_indices[:pixels_to_delete]] = 0
            
            mask = mask.view(1, h, w)
            current_image = input_batch * mask + baseline_val * (1 - mask)

            # Get the model's prediction on the modified image
            pred = model(current_image, apply_softmax=False)
            current_score = torch.softmax(pred, dim=1)[0, target_class].item()

            normalized_score = current_score / original_score
            curve_scores.append(normalized_score.item())

        curve_scores = np.array(curve_scores)
        x = np.linspace(0, 1, steps + 1)
        auc_value = auc(x, curve_scores)

        return auc_value, curve_scores
    

def benchmark_on_batch(model, data, labels, methods_list, 
                       save_R_results, results_pth, baseline_val,
                       skip_auc=False):
    """
    Runs the deletion AUC benchmark on a batch of examples.

    Args:
        model: The PyTorch model.
        data_loader: A DataLoader yielding (images, labels).
        methods_list (list): List of methods to test, e.g., ['integrated_gradients', 'my_method'].
        num_examples (int): Number of examples to run on.
        save_R_results (bool): True to save R heatmaps of other methods
        results_pth (str): Path to results.
        baseline_val (float): replacement value

    Returns:
        dict: A dictionary of results for each method.
    """
    root = math.sqrt(data.shape[1])
    root = int(root)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    r_scores = {method: [] for method in methods_list}
    results = {method: [] for method in methods_list}

    data, labels = data.to(device), labels.to(device)
    model = model.to(device); model.eval()
    target_class = labels.to(torch.int)

    os.makedirs(results_pth + "/Attr_Methods/", exist_ok=True)

    for method in methods_list:

        R = get_ig_attributions(model, data, labels) if method == 'IG' else \
            get_deeplift_attributions(model, data, labels) if method == 'DeepLIFT' else \
            get_xtrain_res(method, results_pth) if method == 'XtrAIn' else \
            get_gradient_shap_attribution(model, data, target_class) if method == "GradSHAP" else \
            get_lrp_res(model, data, target_class) if method == 'LRP' else \
            get_shapley_sampling_attribution(model, data, target_class)
        
        r_scores[method] = R

        method_res_pth = results_pth + f"/Attr_Methods/{method}"
        if save_R_results and os.path.exists(method_res_pth)==False:

            if method.startswith('XtrAIn')==False:
                
                os.makedirs(method_res_pth, exist_ok=True) 
                for i in range(R.shape[0]):

                    sample = R[i, :]
                    R_minmax = calc_vmin_vmax(sample)

                    plt.figure(figsize=(8, 6), facecolor='none')
                    plt.imshow(R[i, :].reshape(root, root), cmap='coolwarm',
                               vmin=R_minmax[0][0], 
                               vmax=R_minmax[0][1])
                    
                    plt.tight_layout(); plt.axis('off')
                    plt.savefig(method_res_pth+f"/sample_{i}.pdf", 
                                transparent=True, bbox_inches='tight', 
                                pad_inches=0, dpi=300); plt.close()

        if skip_auc==False:
            for i, x in enumerate(data):
        
                auc_value, _ = deletion_auc(model, x, R[i, :], target_class[i], baseline_val=baseline_val, steps=200)
                try:
                    results[method].append(auc_value)
                    
                except Exception as e:
                    print(f"Error with {method} on example {i}: {e}")
                    results[method].append(np.nan) # Append NaN if there's an error

    # Calculate average AUC for each method
    if skip_auc==False:

        with open(f"{results_pth}/compare_methods.txt", "a+") as f:

            for method, auc_list in results.items():
                avg_auc = np.nanmean(auc_list)
                f.write(f"Average Deletion AUC for {method:20s}: {avg_auc:.4f} \n")
        
    return r_scores


def get_lrp_res(model, data, target_class):

    shp = data.shape[1]
    
    meth = LRP_Implem(model, None)
    R = meth.lrp(data)[0]

    target_expanded = target_class.unsqueeze(1)\
        .expand(-1, shp).to(torch.int64)
    
    result = torch.gather(R, dim=2, index=target_expanded.unsqueeze(2))
    
    return result.squeeze(2).detach().cpu()

def get_xtrain_res(method, results_pth):

    if method=='XtrAIn':
        R = torch.load(f"{results_pth}/relevance.pt", weights_only=False)
    
    return R


def test_methods(model_pth="DNNandTMNIST_run_0.pt", 
                 model_layers=[784, 400, 100, 10], 
                 data_pth="/typeface_mnist/run_0",
                 save_methods_r=False, 
                 baseline_val=0.6,
                 methods = ['SHAP', 'LRP', 'GradSHAP', 'IG', 'DeepLIFT', 'XtrAIn']):
    
    """Calculates the attribution scores for different attribution methods and the Deletion 
    AUC criterion.
    
    Args:
        model_pth (str): Path of the model
        model_layers [int]: model's architecture (needed to load the model)
        data_pth (str): Path of the data (saved by xtrain algorithm)
        save_methods_r (bool): If True, it saves the resulting attribution scores as heatmaps
        baseline_val (float): the value replacing that of the features removed by Deletion AUC
        methods [str]: a list of attribution methods to calculate their"""

    data_pth = os.getcwd()+f"/Results/Datasets/{data_pth}"
    model = load_model(os.getcwd()+f"/Training/Trainer/Checkpoints/{model_pth}", model_layers)
    
    data = torch.load(data_pth+"/data.pt", weights_only=False)
    data = {'samples': data['samples'][:64, :], 'labels': data['labels'][:64]}
     
    _ = benchmark_on_batch(model, data['samples'], data['labels'], 
                       methods, save_methods_r, data_pth, baseline_val)
