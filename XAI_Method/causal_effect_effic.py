from tqdm import tqdm
import torch


class Causal:
    """
    The base class for calculating XtrAIn. This requires many model transformations
    and forward calls. To minimize models in GPU's memory, only one model is used.
    The actions needed (`theoretical` and `practical` terms of XtrAIn) are handled 
    in series, after the model's state update. 
    
    Steps:
    1. The model comes with its updated state after training. Thus, the `practical` 
       step is being calculated first: df' - f'_W'(x) - f'_{W'_i = W_i}(x)
    2. The model is altered to its previous state
    3. The `theoretical` step is calculated: df = f_{W_i=W'_i}(x) - f_W(x) 
    """

    def __init__(self, model, model_state_upd, samples, labels, nt_effect=True):
        """
        model: the UPDATED model (after one training epoch)
        model_state_upd: the NON-UPDATED state (before the training epoch)
        """

        self.model = model
        self.device = torch.device('cuda')

        self.model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model_state_upd = model_state_upd
        
        self.samples = samples.to(self.device)
        self.num_classes = self.model(self.samples[:1, :]).shape[1]

        self.val = -1 if nt_effect else 1
        self.label_ids = self.val*torch.ones(labels.shape[0], self.num_classes)
        
        self.label_ids[torch.arange(labels.shape[0]), labels.to(torch.int)] = 1
        self.label_ids = self.label_ids.to(self.device)

        self.weight_diff = self.model_state_upd['net.1.weight'].T - \
                            self.model.net[1].weight.T.data 

    def upd_calc(self, a, b):
        """Performs the basic calculation for df or df': f'_W - f'_W'.
          
        It simulates a change in model's parameters (as required by the
        mathematical expression for df, df') with the use of hooks, so that
        the model's input is altered to simulate this change. Then, it performs
        forward passes, subtraction and aggregation (all computations are 
        matrix calculations for speed).

        Args:
            a (int): controls weight sign
            b (int): controls subtr. sign
        
        Returns:
            the calculated aggegated score (= df or df' based on the configuration)
        """
        
        # Forward pass for the model in hand
        f_X = self.model(self.samples, apply_softmax=True)

        # Adding hooks for simulating parameter change
        f_hook = self.model.modified_forward_ts_op_hook(a*self.weight_diff)
        
        # Forward pass for the "updated" model
        f_X_upd = self.model(self.samples, apply_softmax=False)
        f_X_upd = torch.softmax(f_X_upd, dim=2)

        f_hook.remove()

        # Subtraction and aggregation.
        calc = b*(f_X_upd - f_X.unsqueeze(1))*self.label_ids.unsqueeze(1)

        return torch.sum(calc, dim=-1)
  
    def update(self, disent=False):
        """The main pipeline for calculating XtrAIn.
        
        Args:
            disent (bool): if True, X_train is broken into df, df'
        
        Returns: attribution scores for XtrAIn, optionally for df, df'.
        """

        # Calculation of df'
        with torch.no_grad():
            df_prev = self.upd_calc(a=1, b=-1) 

        # Calculation of df
        self.model.load_state_dict(self.model_state_upd)
        with torch.no_grad():
            df_next = self.upd_calc(a=-1, b=1) 
        
        if disent:
            return (df_prev + df_next).detach().cpu(), \
                df_prev.detach().cpu(), \
                df_next.detach().cpu() 
        else:
            return (df_prev + df_next).detach().cpu()
