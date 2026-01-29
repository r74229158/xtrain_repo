import numpy as np
from sklearn.metrics import auc
import torch
import math


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