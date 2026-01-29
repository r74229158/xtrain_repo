import numpy as np
import torch
import math

def evaluate_pixel_erasure(model, x, target, attr_map, baseline_val=0.6, p=0.2):
    """
    Evaluate the drop in model score after erasing the top p% most important pixels.
    
    Args:
        model: The PyTorch model.
        x (torch.Tensor): Original input image (1, C, H, W).
        target (int): The class index to monitor.
        attr_map (np.ndarray): Attribution scores (H, W) or (C, H, W). Will be summed across channels.
        baseline_val (int): The baseline value that will replace that of the targeted features
        p: Fraction of pixels to erase (0.0 to 1.0).

    Returns:
        float: Drop in model score (original score - erased score).
    """

    with torch.no_grad():

        x = x.to(model.device)
        attr_map = attr_map.to(model.device)

        target = target.to(torch.int).to(model.device)
        original_score = model(x)[torch.arange(x.shape[0]), target]
                
        # Get the indices of the most important pixels first
        sorted_indices = torch.flip(np.argsort(attr_map.cpu()), [1]).to(x.device) 

        # Create a mask of ones (all pixels present)
        num_to_erase = int(p * attr_map.shape[1])

        mask = torch.zeros_like(x, dtype=torch.int).to(model.device)
        for i in range(mask.shape[0]):
            mask[i, sorted_indices[i, :num_to_erase]] = 1
        mask = mask.to(torch.bool)

        erased_image = x.clone()
        erased_image[mask] = baseline_val

        erased_score = model(erased_image)[torch.arange(x.shape[0]), target]

    # Return drop in score
    max_val = torch.maximum(torch.zeros_like(original_score), original_score - erased_score)
    res = torch.sum(max_val/original_score)/(original_score.shape[0])
    
    return res.item()
