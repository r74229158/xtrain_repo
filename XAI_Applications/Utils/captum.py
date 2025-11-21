from captum.attr import IntegratedGradients
from captum.attr import DeepLift, GuidedGradCam
from captum.attr import Saliency, Lime, LRP
from captum.attr import GradientShap, FeatureAblation, ShapleyValueSampling

import torch


def get_lrp_attribution(model, input_batch, target_class):

    lrp = LRP(model)
    attribution = lrp.attribute(input_batch, target=target_class)

    return attribution

def get_saliency_attribution(model, input_batch, target_class):

    saliency = Saliency(model)
    attr = saliency.attribute(input_batch, target=target_class.to(torch.int64))

    return attr.cpu().detach()

def get_lime_attribution(model, input_batch, target_class, denorm=True, denorm_a=0.6, denorm_b=0.5):

    indices = range(len(input_batch))
    
    all_attributions = []
    
    for idx in indices:
        sample = input_batch[idx].unsqueeze(0)

        if denorm:
            sample = ((sample-denorm_a)*denorm_b * 255)

        # sample = sample.squeeze().permute(1, 2, 0).cpu().numpy()
        target = target_class[idx]
        
        lime_explainer = Lime(model)
        attributions = lime_explainer.attribute(
            sample,
            target=target,
            n_samples=2000,
            return_input_shape=True
        )
        
        all_attributions.append(attributions.squeeze())
    
    return torch.stack(all_attributions).detach().cpu()

    
def get_ig_attributions(model, input_batch, target_class):
    """
    A helper function to get attribution maps from Integrated Gradients.
    """
    model.eval()

    input_batch.requires_grad_(True)
    target_class = target_class.to(torch.int64)

    ig = IntegratedGradients(model)
    attribution = ig.attribute(input_batch, target=target_class)

    return attribution.squeeze(0).cpu().detach()

def get_deeplift_attributions(model, input_batch, target_class, **kwargs):
    """
    A helper function to get attribution maps from DeepLift.
    """
    model.eval()

    dl = DeepLift(model)
    target_class = target_class.to(torch.int64)
    attribution = dl.attribute(input_batch, target=target_class, **kwargs)
    
    return attribution.squeeze(0).cpu().detach()

def get_guided_grad_cam_attributions(model, input_batch, target_class):
    
    # TODO. Only works with Convs
    grad_cam = GuidedGradCam(model, layer=model.net[1])

    target_class = target_class.to(torch.int64)
    attribution = grad_cam.attribute(input_batch, target=target_class)

    return attribution.squeeze(0).cpu().detach()

def get_gradient_shap_attribution(model, input_batch, target_class):
    """
    Compute GradientSHAP attributions for flattened input
    """
    model.eval()

    # input_batch = input_batch.to(torch.int64)
    target_class = target_class.long()
    
    # Create multiple baselines for better approximation
    baseline = torch.ones_like(input_batch) * 0.7  # Mid-point baseline
    baseline_2 = torch.randn_like(input_batch) * 0.1 + 0.7  # Random baseline around mean
    
    # Combine baselines
    baselines = torch.cat([baseline, baseline_2], dim=0)
    
    gradient_shap = GradientShap(model)
    
    attributions = gradient_shap.attribute(
        input_batch,
        baselines=baselines,
        target=target_class,
        n_samples=200,  # Reduced for efficiency
        stdevs=0.02,    # Small noise for normalized data
    )
    
    return attributions.squeeze(0).cpu().detach()


def get_shapley_sampling_attribution(model, input_batch, target_class, feature_group_size=5):
    """
    Compute Shapley Value Sampling attributions for flattened input
    Use larger feature groups due to computational complexity
    """
    model.eval()

    input_batch = input_batch.float()
    target_class = target_class.long()
    
    batch_size, input_size = input_batch.shape
    
    # Create feature groups - use larger groups for Shapley due to computational cost
    # feature_mask = torch.zeros(batch_size, input_size, dtype=torch.long, device=input_batch.device)
    
    # for i in range(batch_size):
    #     for j in range(input_size):
    #         feature_mask[i, j] = j // feature_group_size
    
    shapley_sampling = ShapleyValueSampling(model)
    
    attributions = shapley_sampling.attribute(
        input_batch,
        target=target_class,
        feature_mask=None,
        baselines=0.7,  # Use midpoint of normalization range
        n_samples=50,   # Reduced for practicality
        perturbations_per_eval=4,
        show_progress=True
    )
    
    return attributions.squeeze(0).cpu().detach()
