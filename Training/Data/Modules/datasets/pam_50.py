import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class PAM50Dataset(Dataset):

    def __init__(self):

        self._get_mnist_data()

        self.X = torch.from_numpy(self.samples.values).float()
        self.y = torch.from_numpy(self.labels.values).squeeze().float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def _get_mnist_data(self):

        target_path = os.getcwd() + "/Training/Data/Datasets/PAM50/"        
        
        self.samples = pd.read_csv(target_path+"PAM50_clean_samples.csv")
        self.labels = pd.read_csv(target_path+"PAM50_clean_labels.csv")

    
    def get_gene_names(self):
        return self.samples.columns.tolist()