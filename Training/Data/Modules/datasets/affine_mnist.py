import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import os
import kagglehub
import shutil
import scipy


class AffineMNISTDataset(Dataset):

    def __init__(self, transform=None, max_samples=30000):
        """
        Args:
            transform: Transformations for images
            target_transform: Transformations for labels
        """
        data_path = self._get_affine_data()
        
        self.transform = transform
        self.max_samples = max_samples
        self.images, self.labels = self._load_mat_data(data_path)
        
    def _load_mat_data(self, data_path):
        """Load data from .mat file(s)"""
        
        data_path = Path(data_path)
        
        # Load multiple .mat files (training batches)
        all_images = []
        all_labels = []
        
        for mat_file in sorted(data_path.glob('*.mat')):

            mat_data = scipy.io.loadmat(mat_file)
            
            # Common key patterns in these files
            images = mat_data['affNISTdata']['image'][0][0]
            labels = mat_data['affNISTdata']['label_int'][0][0]
            
            # Ensure correct shape and type
            images = images.astype(np.float32)
            labels = labels.astype(np.int32)
            
            # Affine MNIST has shape (1600, N)
            if images.shape[0] == 1600:
                images = images.T
            
            all_images.append(images)
            all_labels.append(labels.flatten())
        
        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        if len(images) > self.max_samples:
            images = images[:self.max_samples]
            labels = labels[:self.max_samples]
        
        return images, labels.flatten()
    
    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, idx):
        
        image = self.images[idx]
        label = self.labels[idx]
        
        # Reshape to 40x40
        image = image.reshape(40, 40)
        
        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label, dtype=torch.long)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    
    def _get_affine_data(self):

        target_path = os.getcwd() + "/Training/Data/Datasets/AffineMNIST"
        data_path = Path(target_path)

        if data_path.exists()==False:

            download_path = kagglehub.dataset_download("kmader/affinemnist")
            shutil.copytree(download_path, target_path)
            
        return target_path

