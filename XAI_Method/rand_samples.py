import torch

from Training.Model.FCNN import simpleDNN4l
from XAI_Method.causal_effect import Causal


class RandomSamples:

    def __init__(self, 
                 model, 
                 data_loader, 
                 epochs, 
                 classes,
                 break_per_epochs=5,
                 num_samples=64,
                 save_r_scores='last'):
        """
        This class performs the first step of the method. It creates a uniform (baseline) 
        model, selects random samples for the algorithm to track and performs the first, 
        artificial step (tracking changes from the uniform model to the random one).

        Args:
            model (torch.nn.Module): The (untrained) random model
            data_loader (torch.dataset.Dataloader): The test data to draw samples randomly
            epochs (int): the number of training epochs
            classes (int): the number of classes
            break_per_epochs (int): length of steps to track until saving results and restarting 
            num_samples (int): corresponds to the number of samples for the algorithm to track
            save_r_scores (str): a string indicator for initializing R scores properly
        """

        self.const_model = simpleDNN4l(model.layers, False)
        self.const_model_state = {k: v.detach().clone() for k, v in self.const_model.state_dict().items()}
        self.model = model
        self.dev = model.device.type

        self.layers = model.layers
        n_features = self.layers[0]
        
        self.X = torch.zeros(num_samples, n_features)
        self.lbls = torch.zeros(num_samples)
        self.data_loader = data_loader

        self.num_samples = num_samples
        self.epochs = epochs
        self.classes = classes
        self.break_per_epochs = break_per_epochs

        self.save_r_scores = save_r_scores
        if save_r_scores=='all':

            sz = min(epochs, break_per_epochs)
            self.R = torch.zeros(num_samples, self.layers[0], sz)
        
        else:
            self.R = torch.zeros(num_samples, self.layers[0])
        
        self.select_random_samples()
        

    def select_random_samples(self):

        i = 0
        for batch in self.data_loader: 

            # Data are randomly loaded
            data, labels = batch[0], batch[1] 
            batch_size = len(labels)

            data = data.reshape(batch_size, -1)
            j = min(batch_size, self.num_samples - i)

            self.lbls[i:i+j] = labels[:j]
            self.X[i:i+j, :] = data[:j]

            if j < batch_size or batch_size==(self.num_samples - i):
                break
            
            i += batch_size


    def artificial_step(self):
        
        model_state = {k: v.detach().clone() for k, v in self.const_model.state_dict().items()} 
        rand_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        upd_attrib = Causal(self.model, model_state, self.X, self.lbls, 'neut').update()  
        
        if self.save_r_scores=='all':
            self.R[:, :, 0] = upd_attrib
        else:
            self.R[:, :] = upd_attrib
        
        self.model.load_state_dict(rand_state)
        return self.R
