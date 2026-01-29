import os, gc; from os.path import join
from tqdm import tqdm; import numpy as np
import torch; import torch.optim as optim
from torch.amp import GradScaler

from XAI_Method.rand_samples import RandomSamples
from Utils.visualize_mnist import VisualizeMNIST
from Utils.visualize_pam import VisualizePAM, save_heatmaps_simple
from XAI_Method.causal_effect import Causal


class BaseTrainer:
    """
    This is the base class working as the pipeline for all processes regarging
    XtrAIn: loading model, defining training parameters, keeping training logs, 
    calling the training step, gathering Xtrain scores and saving results. 
    
    Specific functionalities are completed by other parts of code (e.g. calculating
    attribution scores). This method organizes the workflow. 

    Args:
        model: a randomly initialized pytorch model
        config (dict): a configuration of the training parameters
        data (CustomLoader): a dataloader with training and test data
        n_test (int): number of samples for Xtrain to track their attribution scores
        break_per_epochs (int): breaking epochs and saving results according to the 
            epoch passed. Useful when the model is trained for many epochs and the 
            results start to ramp up in memory
        run_simple_acc (bool): True to capture accuracy between epochs. False: 
            also captures accuracy for each class for an epoch. 
        save_r_scores (str): Whether or not to save attributions and heatmaps of all 
            the attribution scores (for all epochs), only for the last, or none at all 
            (last attribution scores are always saved as .pt)
                Should be a value from ['None', 'last', 'all']
        save_pdf (bool): Save relevance scores as heatmaps in pdf, else png,
        save_results (bool): If True, results are saved (False only in case of capturing training statistics)
        num_to_save (int): number of samples to save their corresponding heatmaps (attribution scores).
    """

    def __init__(self, model, config, 
                 data, n_test, 
                 break_per_epochs=5,
                 keep_all_changes=False,
                 run_simple_acc=False,
                 save_r_scores='last',
                 save_pdf=False,
                 save_results=True,
                 num_to_save=20
                 ):

        super().__init__()

        self.model = model
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
        
        ## Data-related variables
        self.data = data
        self.dataset = data.dataset_name

        self.train_load = data.train_load 
        self.test_load = data.evalu_load
        
        ## Train-related variables
        self.config = config
        self.epochs = config["epochs"]
        
        self.classes = config['num_classes']
        self.lr = config['lr']

        self.init_optimizer()
        self.init_scheduler()
        self.scaler = GradScaler()

        # Store Train logs
        self.flag_acc_simple = run_simple_acc
        self.train_logs = []
        
        self.epoch_acc = {}
        for i in range(self.epochs): 
            self.epoch_acc[i] = {}
                
        # Path-related variables
        self.save_results = save_results
        if save_results:
            self.create_results_dir()

        # Initializing R and performing the first step
        self.keep_all_changes = keep_all_changes
        self.break_per_epochs = break_per_epochs

        self.save_r_scores = save_r_scores
        self.upd_model_state = None
        self.n_test = n_test

        self.save_pdf = save_pdf
        self.num_to_save = num_to_save

        self.init_rand_step()

    ## Create directory for results
    def create_results_dir(self):

        self.run_on = join(os.getcwd(), f"Results/Datasets/{self.dataset}")
        os.makedirs(self.run_on, exist_ok=True)
        
        run_num = [int(i.rsplit("_")[1]) for i in \
                    os.listdir(self.run_on) if i.startswith("run")]
        
        self.run_num = 0 if len(run_num)==0 else max(run_num)+1
        self.data_pth = self.run_on + f"/run_{self.run_num}"

        self.ckpt_pth = "Training/Trainer/Checkpoints/" + \
                f"{self.config['model_name']}_run_{self.run_num}.pt"

    ## Load and Save Model
    def load_ckpt(self):

        tot_pth = join(os.getcwd(), self.ckpt_pth)
        ckpt = torch.load(tot_pth)

        print('Loading model from', self.ckpt_pth)
        self.model.load_state_dict(ckpt)

    def save_ckpt(self):
        
        tot_pth = join(os.getcwd(), self.ckpt_pth)

        print("Saving trained model at", self.ckpt_pth)
        torch.save(self.model.state_dict(), tot_pth)
        
        return tot_pth

    def init_optimizer(self):

        if self.config["optimizer"] == "adamW":
            self.optimizer = optim.AdamW(self.model.parameters(), 
                                         lr=self.lr, weight_decay=0) 

        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def init_scheduler(self):
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

    def init_rand_step(self):
        """Performs the random step and initializes samples, labels and variables 
        for capturing their attribution scores.
        """
        
        # Perform first, artificial step.
        self.rand_samples = RandomSamples(self.model, self.test_load, 
                                     self.epochs, 
                                     self.classes,
                                     self.break_per_epochs,
                                     self.n_test,
                                     self.save_r_scores)
        
        self.X, self.labels = self.rand_samples.X, self.rand_samples.lbls
        self.X_out = torch.zeros(self.epochs+1, self.labels.shape[0], self.classes)

        self.X = self.X.to(self.device)
        self.labels = self.labels.int().to(self.device)

        # Initialize relevances.
        self.R_norm = self.rand_samples.artificial_step()
        self.R_targ = torch.zeros_like(self.R_norm)
        self.R_no_t = torch.zeros_like(self.R_norm)
        self.R_t_po = torch.zeros_like(self.R_norm)

        if self.save_results:
            
            self.run_pth = self.run_on + f"/run_{self.run_num}"

            if self.dataset == 'pam50':
                self.save_viz = VisualizePAM(self.X.detach().cpu(), 
                                             self.labels, self.epochs,
                                             self.classes, self.run_pth,
                                             self.num_to_save)

            else:
                self.save_viz = VisualizeMNIST(self.X.detach().cpu(), 
                                                 self.labels, self.epochs, 
                                                 self.break_per_epochs, self.classes,
                                                 self.run_pth, self.num_to_save,
                                                 self.save_pdf
                                                )

    ## Propagation Methods
    def loss(self, y, y_pred):
        raise NotImplementedError

    def forward(self, X, apply_softmax=False):
        return self.model(X, apply_softmax=apply_softmax)

    ## Train-Val Methods
    def train_on_batch(self, batch):
        raise NotImplementedError

    def validate_on_batch(self, batch):
        raise NotImplementedError
    
    ## Training Loop
    def train(self):
        """The main training loop."""

        print(f"Starting the training process...")

        train_losses, test_losses = [], []
        self.X_out[0, ::] = self.model(self.X.to(self.device), 
                                apply_softmax=True).detach().cpu()

        self.break_idx = 0
        for epoch in range(0, self.epochs):
            
            ## --- TRAINING ---
            ## ----------------

            train_bar = tqdm(enumerate(self.train_load), 
                        desc=f"Epoch: {epoch+1}/{self.epochs}",
                        total=len(self.train_load)
                    )
            
            self.model.train()
            train_loss, batch_size = 0, 0
            
            self.next_ep, self.prev_ep = 2*[epoch%self.break_per_epochs] \
                if type(self.break_per_epochs) == int else 2*[epoch] 
            
            for batch in train_bar:

                if batch[0]==0 and self.prev_ep!=0: self.prev_ep -= 1     
                
                l, attr = self.train_on_batch(batch, "normal")
                _, attr_t = self.train_on_batch(batch, "target")
                _, attr_nt = self.train_on_batch(batch, "non-targ")

                batch_size += batch[1][0].size(0)
                train_loss += l*batch[1][0].size(0)

                self.update_R(attr, attr_t, attr_nt)
                self.model.load_state_dict(self.upd_model_state)

                torch.cuda.empty_cache()

            self.break_idx += 1
            self.scheduler.step()

            train_loss /= batch_size; train_losses.append(train_loss)
            print(f"Training loss: {train_loss:.4f}")

            # Evaluation
            test_losses = self.evaluate(epoch, test_losses)

            # Save intermediate relevance scores
            if epoch==self.epochs-1 or \
                (self.break_idx == self.break_per_epochs and self.save_r_scores=='all'):
                
                self.calc_and_save_r_linear()
                self.save_vizualization(epoch)

                if self.save_r_scores=='all' and epoch!=self.epochs-1:
                    self.track_interm_results(epoch)
                
                self.break_idx = 0

        model_pth = self.save_ckpt()
        print("Training process ended...")

        return train_losses, test_losses, model_pth
    
    def evaluate(self, epoch, test_losses):

        self.model.eval()

        metrics = np.zeros(len(self.test_load.dataset))
        all_labels = np.zeros(len(self.test_load.dataset))

        test_loss, batch_size = 0, 0

        test_bar = tqdm(enumerate(self.test_load), 
                    desc=f"Epoch: {epoch + 1}/{self.epochs}",
                    total=len(self.test_load)
                )

        with torch.no_grad():

            self.X_out[epoch, ::] = self.model(self.X.to(self.device), 
                                        apply_softmax=True).detach().cpu()
            
            for batch in test_bar:

                l = self.validate_on_batch(batch[1])

                bs = batch[1][0].size(0)
                test_loss += l["loss"]*bs

                if l["metric_name"] == 'acc':
                    
                    accuracies = l['metric'].cpu().numpy()
                    metrics[batch_size: batch_size+bs] = accuracies
                    all_labels[batch_size: batch_size+bs] = batch[1][1].cpu().numpy()
                    
                batch_size += batch[1][0].size(0)

        if not self.flag_acc_simple:  
            self.get_epoch_acc(epoch, metrics, all_labels)  

        test_loss /= batch_size
        test_losses.append(test_loss)
        print(f"Evaluation loss: {test_loss:.4f}")
        
        metric = np.mean(metrics)
        print(f"Model Accuracy: {metric}")
        
        self.train_logs.append(metric)
        return test_losses
    
    def update_R(self, attr, attr_t, attr_nt):

        if self.save_r_scores == 'all':

            flag = self.break_idx > 0 or self.next_ep==self.prev_ep

            r_norm = self.R_norm[:, :, self.prev_ep] if flag else self.r_norm
            r_targ = self.R_targ[:, :, self.prev_ep] if flag else self.r_targ
            r_no_t = self.R_no_t[:, :, self.prev_ep] if flag else self.r_no_t
            r_t_po = self.R_t_po[:, :, self.prev_ep] if flag else self.r_t_po
            
            self.R_norm[:, :, self.next_ep] = r_norm + attr  
            self.R_no_t[:, :, self.next_ep] = r_no_t + attr_nt

            self.R_targ[:, :, self.next_ep] = r_targ + attr_t[0]
            self.R_t_po[:, :, self.next_ep] = r_t_po + attr_t[0]*attr_t[1]

            if not flag:
                self.prev_ep += 1 
        
        else:
            self.R_norm += attr  
            self.R_no_t += attr_nt

            self.R_targ += attr_t[0]
            self.R_t_po += attr_t[0]*attr_t[1]


    def track_interm_results(self, epoch):
        """Tracks intermediate results, saves their values and images,
        zeroes tracking variables."""

        self.train_logs = []

        self.r_norm, self.r_targ = self.R_norm[:, :, self.next_ep], self.R_targ[:, :, self.next_ep]
        self.r_no_t, self.r_t_po = self.R_no_t[:, :, self.next_ep], self.R_t_po[:, :, self.next_ep]
        
        self.R_norm = torch.zeros(self.R_norm.shape[0], self.R_norm.shape[1], \
                min(self.break_per_epochs, self.epochs - 1 - epoch))
        
        self.R_targ = torch.zeros_like(self.R_norm)
        self.R_no_t = torch.zeros_like(self.R_norm)
        self.R_t_po = torch.zeros_like(self.R_norm)

    def get_epoch_acc(self, epoch, accuracies, all_labels):

        for i in range(self.classes):

            msk = all_labels == i
            self.epoch_acc[epoch][i] = np.mean(accuracies[msk])

    def calc_and_save_r_linear(self):
        """
        Calculate the relevance score for X_linAIr.
        """

        self.R_linr = Causal(self.model, self.rand_samples.const_model_state, 
                        self.X, self.labels).update()
        self.model.load_state_dict(self.upd_model_state)

        
    def save_vizualization(self, epoch):
        """Save logs and attribution scores."""
        
        X_out = self.X_out if not self.flag_acc_simple else None
        X_out = X_out[-1, ::] if self.save_r_scores=='last' else X_out

        self.save_viz.save_logs(epoch, self.train_logs, self.epoch_acc)
        save_all_images = False if self.save_r_scores=='last' else True

        self.save_viz.save_scores(self.R_norm, self.R_targ, 
                                  self.R_t_po, self.R_no_t,
                                  self.R_linr,
                                  X_out)
        
        if self.save_r_scores != 'none':

            if self.dataset == 'pam50':
                
                save_heatmaps_simple(self.save_viz.r_scores, self.save_viz.r_target_pos, 
                              [range(self.num_to_save), range(self.num_to_save)], self.save_viz.tot_pth, 
                              True)
                
            else:
                self.save_viz.save_imgs(save_all_images=save_all_images)