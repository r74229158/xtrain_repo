import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import os
import kagglehub
import shutil


class TMNISTDataset(Dataset):
    
    def __init__(self, transform=None):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self._get_mnist_data()
        self.transform = transform
        

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Extract the image pixels (columns 2:786) and reshape to 28x28
        image = self.data.iloc[idx, 2:].values.astype(np.uint8).reshape(28, 28)
        label = self.data.iloc[idx, 1]
        digit_id = self.data.iloc[idx, 0]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, digit_id
    
    def _get_mnist_data(self):

        target_path = os.getcwd() + "/Training/Data/Datasets/TMNIST/TMNIST_Data.csv"
        data_path = Path(target_path)

        if data_path.exists()==False:

            download_path = kagglehub.dataset_download("nimishmagre/tmnist-typeface-mnist")
            shutil.copytree(download_path, target_path)
        
        self.data = pd.read_csv(target_path)
