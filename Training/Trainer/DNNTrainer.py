import torch; import torch.nn as nn
from torch.amp import autocast
from copy import deepcopy

from Training.Trainer.BaseTrainer import BaseTrainer
from Training.Utils.train_utils import accuracy
from XAI_Method.causal_effect_effic import Causal

class simpleDNNTrainer(BaseTrainer):
    """
    Implements the training step of the algorithm. 
    Check the BaseTrainer for more information.
    """

    def __init__(self, model, config, data, 
                 n_test=64, 
                 break_per_epochs=5,
                 run_simple_acc=False,
                 save_r_scores='last',
                 save_pdf=False
                 ):

        super().__init__(model, config, data, n_test, 
                         break_per_epochs=break_per_epochs,
                         run_simple_acc=run_simple_acc,
                         save_r_scores=save_r_scores,
                         save_pdf=save_pdf
                        )
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss = nn.CrossEntropyLoss()


    def apply_loss(self, y_pred, y):

        y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        y = y.reshape((-1,))

        if y[0].dtype == torch.int or torch.float32: y = y.long()

        return self.loss(y_pred, y)
    
    def apply_loss_target(self, y_pred, y):

        if y[0].dtype == torch.int or torch.float32: y = y.long()

        logits_sel = y_pred[torch.arange(y_pred.size(0)), y]
        return -logits_sel.mean()

    def apply_loss_non_target(self, y_pred):

        logsum = torch.logsumexp(y_pred, dim=1)
        return logsum.mean()
    
    def train_on_batch(self, batch, loss):
        """
        The main step of DNNTrainer. Depending on the loss, the method
        calls the suitable loss function. The model is then trained and
        the XtrAIn calculation is being applied. This computation reverses
        the state of the model back to its original state.

        Args:
            batch: a list of (idx, (samples, labels))
            loss (str): choices: ['normal', 'target', 'non-target']
        
        Returns:
            The loss along with the attribution score.
        """

        if loss == 'normal':
            self.model_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        # Get data from batch, calculate loss and update
        images, labels = batch[1][0], batch[1][1]
        images = images.to(self.device); labels = labels.to(self.device)

        self.optimizer.zero_grad()
        with autocast(device_type=self.dev):  # Enable mixed-precision

            output = self.forward(images)
            if loss == 'normal':
                l = self.apply_loss(output, labels)

            elif loss == 'target':

                with torch.no_grad(): 
                    old_output = self.forward(self.X, apply_softmax=True)\
                        [torch.arange(self.labels.shape[0]), self.labels]

                l = self.apply_loss_target(output, labels)

            else:
                l = self.apply_loss_non_target(output)

        self.scaler.scale(l).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if loss == 'normal':
            self.upd_model_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        
        # Track positive contributions of target updates
        if loss == 'target':
            with torch.no_grad():

                new_output = self.forward(self.X, apply_softmax=True)\
                    [torch.arange(self.labels.shape[0]), self.labels]
                catch_pos = (new_output>old_output).unsqueeze(1).detach().cpu()

        R = self.calculate_R()          
        if loss == 'target':
            return l, (R, catch_pos)

        return l, R

    def validate_on_batch(self, batch):

        images, labels = batch[0], batch[1]
        images = images.to(self.device)
        labels = labels.to(self.device)

        output = self.forward(images)
        l = self.apply_loss(output, labels)

        acc = accuracy(output, labels, averaged=False)
        loss = {"loss": l, "metric": acc, "metric_name": "acc"}

        return loss

    def calculate_R(self):
        """This method calls the mechanism for calculating XtrAIn"""
        
        causal = Causal(self.model, self.model_state, self.X, self.labels)
        return causal.update()

