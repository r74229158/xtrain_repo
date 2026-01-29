import os, sys
import torch
# sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from XAI_Applications.Utils.lrp import LRP_Implem
from XAI_Applications.Utils.captum import *


class AttributionNotImplemented(Exception):
    """Exception raised for custom error in the application."""

    def __init__(self, method):
        super().__init__("Method not found.")
        self.method = method

    def __str__(self):
        return f"Attribution method {self.method} is not implemented. The catalogue only contains \
        IG, DL, XT, GS, LRP, Shap. Consider using one of those."


def calculate_relevance_scores(method, model, samples, labels, results_pt=None):
    """
    Given a model and a data batch, this method calculates the relevance scores 
    of popular attribution methods.
    
    Args:
        method (str): name of the attribution method to use
        model (nn.Module)
        samples (torch.tensor)
        labele (torch.tensor)
        results_pt (str): In case of Xtrain (should be already calculated)

    Returns
        torch.tensor: the calculated attribution score.
    """

    labels = labels.to(torch.int)

    if method in ['IG', 'IntegratedGradients']:
        return get_ig_attributions(model, samples, labels) 
    
    elif method in ['DL', 'DeepLIFT']:
        return get_deeplift_attributions(model, samples, labels)
        
    elif method in ['XT', 'Xtrain']:
        return torch.load(f"{results_pt}/xt_relevance.pt", weights_only=False)
    
    elif method in ['XL', 'Xlinear']:
        return torch.load(f"{results_pt}/xlin_relevance.pt", weights_only=False)

    elif method in ['GS', 'GradShap']:
        return get_gradient_shap_attribution(model, samples, labels) 
    
    elif method in ['LRP']:
        return get_lrp_res(model, samples, labels) 
    
    elif method in ['SHAP', "Shapley"]:
        return get_shapley_sampling_attribution(model, samples, labels)
    
    else:
        raise AttributionNotImplemented(method)


def get_lrp_res(model, data, target_class):

    shp = data.shape[1]
    
    meth = LRP_Implem(model, None)
    R = meth.lrp(data)[0]

    target_expanded = target_class.unsqueeze(1)\
        .expand(-1, shp).to(torch.int64)
    
    result = torch.gather(R, dim=2, index=target_expanded.unsqueeze(2))
    return result.squeeze(2).detach().cpu()
