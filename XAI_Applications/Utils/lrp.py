import os
import torch; import torch.nn as nn
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlus, EpsilonPlusFlat
from zennit.core import Stabilizer
# from Utils.DEPREC.stats_utils import calc_similarity


#### Official Implementation of LRP




class LRP_Implem:
    """This is the class implementing the LRP algorithm."""

    def __init__(self, model, data, epsilon=1e-6):

        self.model = model
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.epsilon = epsilon
        self.data = data


    def zennit_lrp(self, X, labels):

        X.requires_grad_(True)
        target = torch.eye(10)[[labels.to('cpu')]].to(X.device)
        # target = target[[labels]]

        composite = EpsilonPlus(stabilizer=Stabilizer(epsilon=1e-5, clip=True),
                                zero_params=['bias'])
        with Gradient(self.model, composite) as attributor:
            _, relevance = attributor(X, target)

        return relevance


    def lrp(self, X, train=False, add_bias=False, relative_lrp=False):

        R_scores, R_tot  = [], self.get_R_tot(X)
        mods = [l for n, l in self.model.named_modules() if '.' in n]

        # squeeze channels if image is gray
        # X = X.squeeze()

        if X.device.type == 'cpu':
            X = X.to(self.device)

        for i, layer in enumerate(mods):

            if isinstance(layer, nn.Linear):

                W = layer.weight.T
                b = layer.bias

                if not train:
                    W, b = W.detach(), b.detach()

                # Perform diag(X)*W multiplication
                R = XtimesW(X, W, b, add_bias)
                
                # Define normalizing term
                dm = 0 if len(R.size()) == 2 else 1
                Z = R.shape[1] if relative_lrp else torch.sum(R, dim=dm)
                
                # Add epsilon
                if not relative_lrp:
                    Z[torch.abs(Z)<1e-8] += self.epsilon
                
                if not relative_lrp and add_bias: Z += b

                # Normalize by the total
                if len(R.size()) > 2 and not relative_lrp:
                    Z = Z.unsqueeze(1)
                R /= Z

                if torch.sum(torch.isnan(R))>0:
                    print('nan values')

                R_scores.append(R)
                R_tot = torch.matmul(R_tot, R)
            
            # elif isinstance(layer, nn.Sequential):
            #     continue

            X = layer(X)

            # Last layer
            if i == len(mods):
                R_scores.append(X)
        
        return R_tot, X, R_scores
    

    def get_R_tot(self, X, no_batch=False):
        
        if len(X.squeeze().shape)==2 and no_batch:
            R_tot = torch.eye(X.numel()).to(self.device)  
        else:
            R_tot = torch.eye(X[0, ::].numel()).repeat(X.shape[0], 1, 1).to(self.device)
        
        return R_tot
    

    def lrp_interm_attrib(self, R_scores, label):
        
        scores = {}
        for i, sc in enumerate(R_scores):

            for j in range(i+1, len(R_scores)):
                sc = torch.matmul(sc, R_scores[j])
            
            scores[i] = sc.squeeze()[:, label]

        return scores    

    def calculate_intermediate_attributions(self, input, target=torch.eye(10)[[1]]):
        """This method calculates intermediate relative scores for a 
        FCNN or a CNN model, given one or multiple target classes.
        
        Args:
            target (torch.tensor): A tensor with ones in target neuron(s), zeros elsewhere
        
        Returns:
            dict: A dictionary of (layer, attribution)
        """

        if type(target)==int:
            target = torch.eye(self.model.output_layer)[[target]]
        
        elif type(target)==list:
            target = torch.sum(torch.eye(self.model.output_layer)[target], dim=0).unsqueeze(0)

        target = target.to(self.device)

        # create a composite instance
        composite = EpsilonPlus()

        # create a gradient attributor
        attributor = Gradient(self.model, composite)

        def store_hook(module, input, output):
            """Create a hook to keep track of intermediate outputs"""
            # set the current module's attribute 'output' to the its tensor
            module.output = output
            # keep the output tensor gradient, even if it is not a leaf-tensor
            output.retain_grad()


        with attributor:
            """Enter the attributor's context to register the rule-hooks"""
            # register the store_hook AFTER the rule-hooks have been registered (by
            # entering the context) so we get the last output before the next module
            handles = [
                module.register_forward_hook(store_hook) for module in self.model.modules()
            ]
            # compute the relevance wrt. target neuron
            output, relevance = attributor(input, target)

        # remove the hooks using `store_hook`
        for handle in handles: 
            handle.remove()

        # print the gradient tensors
        interm_rel = {}
        for name, module in self.model.named_modules():
            if type(module)==torch.nn.modules.linear.Linear:
                interm_rel[name] = module.output.grad.detach().cpu()

        return interm_rel
    

    def calculate_relevance_similarity(self):

        num_examples = self.model.output_layer
        L1 = torch.zeros(64, 4); L2 = torch.zeros(64, 4)
        L3 = torch.zeros(64, 4); L4 = torch.zeros(64, 4)

        gather_rels = {}

        for batch in self.data:

            X = batch[0]; X = X.to(self.device)
            for c, x in enumerate(X):
    
                for i in range(num_examples):
                    
                    _, _, R_SCORES = self.lrp(x)
                    interm_rel = self.lrp_interm_attrib(R_SCORES, i)    

                    for n, v in list(interm_rel.items()):
                        
                        if len(gather_rels) == 0:
                            for n, v in list(interm_rel.items()):
                                gather_rels[n] = torch.zeros(num_examples, v.shape[0])
                    
                        gather_rels[n][i, :] = interm_rel[n].squeeze()

                pth = os.getcwd() + "/Results/lrp"
                # l = calc_similarity(gather_rels, pth)

                # L1[c, :] = torch.tensor([i[0] for i in l]); L2[c, :] = torch.tensor([i[1] for i in l])
                # L3[c, :] = torch.tensor([i[2] for i in l]); L4[c, :] = torch.tensor([i[3] for i in l])

            break

        print("Mean evolution of Cosine similarity", torch.mean(L1, dim=0))
        print("Mean evolution of Euclidean distance", torch.mean(L2, dim=0))
        print("Mean evolution of variance", torch.mean(L3, dim=0))
        print("Mean evolution of centroid distance", torch.mean(L4, dim=0))
        


### ----- Utilities -----
### ---------------------

def XtimesW(X, W, b, add_bias=False):

    if len(X.shape) == 1 or X.shape[0]==3:
        R = torch.matmul(torch.diag(X), W)
    else:
        R = torch.matmul(torch.diag_embed(X), W)

    if add_bias:
        R += b / X.shape[1]

    return R

