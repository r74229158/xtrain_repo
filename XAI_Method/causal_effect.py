from tqdm import tqdm
import torch


class Causal:
    """
    The base class for calculating Xtrain. 
    """

    def __init__(self, model, model_state_no_upd, samples, labels, effect='norm'):
        """
        Args:
            model (nn.Module): the UPDATED model (after one training epoch)
            model_state_no_upd (dict): the NON-UPDATED state (before the training epoch)
            samples (torch.tensor): samples to calculate their attribution score
            labels (torch.tensor): labels of samples
            effect (str): A controller of $I$'s effect on f(x). Choices: ['norm', 'pos', 'neg', 'neut'] 
        """

        self.model = model
        self.device = torch.device('cuda')

        self.model_state_no_upd = model_state_no_upd
        
        self.samples = samples.to(self.device)
        self.num_classes = self.model(self.samples[:1, :]).shape[1]

        self.val = 0 if effect=='pos' else 1 if effect=='neut' else -1  
        self.I = self.val*torch.ones(labels.shape[0], self.num_classes)
        
        if effect == 'neg':
            self.I[torch.arange(labels.shape[0]), labels.to(torch.int)] = 0
            
        else:
            self.I[torch.arange(labels.shape[0]), labels.to(torch.int)] = 1
        self.I = self.I.to(self.device)

        self.weight_diff = self.model_state_no_upd['net.1.weight'].T - \
                            self.model.net[1].weight.T.data 

    def upd_calc(self, a, b):
        """Performs the basic calculation for Xtrain.

        Args:
            a (int): controls weight sign
            b (int): controls subtr. sign
        
        Returns:
            the aggegated score
        """
        
        # Forward pass for the model in hand
        f_X = self.model(self.samples, apply_softmax=True)

        # Adding hooks for simulating parameter change
        f_hook = self.model.modified_forward_ts_op_hook(a*self.weight_diff)
        
        # Forward pass for the model with updated target parameters
        f_X_upd = self.model(self.samples, apply_softmax=False)
        f_X_upd = torch.softmax(f_X_upd, dim=2)

        f_hook.remove()

        # Subtraction and aggregation.
        calc = b*(f_X_upd - f_X.unsqueeze(1))*self.I.unsqueeze(1)

        return torch.sum(calc, dim=-1)
  
    def update(self, disent=False):
        """The main pipeline for calculating Xtrain.
        
        Args:
            disent (bool): if True, X_train is broken into df, df'
        
        Returns: attribution scores for Xtrain.
        """

        # Calculation of df'
        with torch.no_grad():
            df_prev = self.upd_calc(a=1, b=-1) 

        # Calculation of df
        self.model.load_state_dict(self.model_state_no_upd)
        with torch.no_grad():
            df_next = self.upd_calc(a=-1, b=1) 
        
        if disent:
            return (df_prev + df_next).detach().cpu(), \
                df_prev.detach().cpu(), \
                df_next.detach().cpu() 
        else:
            return (df_prev + df_next).detach().cpu()
