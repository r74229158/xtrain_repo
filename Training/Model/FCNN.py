import torch
from torch import nn
import torch.nn.init as init
from copy import deepcopy
from random import random

class simpleDNN4l(nn.Module):
    """A simple implementation of a Fully Connected Neural Network."""

    def __init__(self, neurons_per_layer, rand_weights=True):
        """
        Args:
            neurons_per_layer [int]: a list of integers specifying the neurons of each layer. The 
                lenght of the list is the number of layers. It should contain the input and output
                layer as well.
            rand_weights (bool): Random or uniform initialization for the model.
        """

        super().__init__()
        
        self.layers = neurons_per_layer
        self.output_layer = self.layers[-1]

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.activations = {} 

        func_layers = [nn.Flatten()]
        for i in range(len(self.layers)-1):
            func_layers.append(nn.Linear(self.layers[i], 
                                    self.layers[i+1]))
            if i!= len(self.layers) - 2:
                func_layers.append(nn.ReLU())

        self.net = nn.Sequential(*func_layers)
        self.forward(torch.rand((1, self.layers[0])))
        
        self._init_weights(rand_weights)
        self.net.to(self.device)
        

    def forward(self, X, apply_softmax=True):
        
        if X[0, 0].dtype == torch.int64: X = X.to(torch.float32)
        X = self.net(X)
        return torch.softmax(X, dim=1) if apply_softmax else X
    

    def _init_weights(self, rand=True):
    
        i = 0
        for m in [mod for mod in self.net if isinstance(mod, nn.Linear)]:

            if rand:
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            else: 
                c = 1 if i!=0 else -1
                init.constant_(m.weight, c*torch.rand(1).item())
                i+=1

            init.zeros_(m.bias)


    def modified_forward_hook(self, delta_w, idx):

        """ 
        Use forward hooks to temporarily modify weight behavior without changing weights. In
        this step, the output of the first layer is been altered, to simulate a change in weight
        parameters for feature `i` of the input. To achieve this, we simply add
            (w_upd - w) * X 
        to the output of the first linear layer, in order to replace `w*X` with `w_upd*X`.
        """

        def modify_first_layer(module, input, output):

            X = input[0][:, idx]
            with torch.no_grad():
                modified_output = output + delta_w.unsqueeze(0) * X.unsqueeze(1)
            
            return modified_output
        
        # Register hook on first layer
        hook = self.net[1].register_forward_hook(modify_first_layer)
        return hook 
    
    def modified_forward_ts_op_hook(self, delta_w):

        """ """
        def modify_first_layer(module, input, output):

            with torch.no_grad():                
                return output.unsqueeze(1) + delta_w.unsqueeze(0) * input[0].unsqueeze(-1)
                    
        # Register hook on first layer
        return self.net[1].register_forward_hook(modify_first_layer)

    @DeprecationWarning
    def get_layers_activations(self, X):
        
        X = X.to(self.device)
        self.activations.clear()

        _ = self.net(X)
        return self.activations

    @DeprecationWarning
    def _register_hooks(self):

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        # Clear old hooks (if any)
        for handle in getattr(self, '_hook_handles', []):
            handle.remove()
        self._hook_handles = []

        # Register hooks for ReLU and Softmax layers
        idx = 0
        while True:
            idx += 1
            try:
                label = f'linear_{idx}' if idx%2==1 else f'relu_{idx}'
                self._hook_handles.append(self.net[idx].register_forward_hook(get_activation(label))) 
            except IndexError: break