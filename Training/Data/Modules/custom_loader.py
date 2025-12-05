import os
import torch; from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from Training.Data.Modules.datasets.tmnist import TMNISTDataset
from Training.Data.Modules.datasets.corrupted_mnist import get_corrupted_data
from Training.Data.Modules.datasets.affine_mnist import AffineMNISTDataset
from Training.Data.Modules.datasets.pam_50 import PAM50Dataset
from Training.Data.Modules.transformations import AddCustomTransformations


class CustomLoader:
    """A loader for loading data from different sources. A predefined set of transformations
    is performed on each dataset. Then data are broken into train and test set if on train mode.
    Data are then loaded to dataloaders.
    
    Args
        dataset_name (str): name of the dataset, selected from some predefined datasets
        train (bool): If True, data are split into train and test
        batch_size (int): size of the batch, 
        num_workers=4, 
        test_size=0.2, 
        seed=None, 
        shuffle_test
    """

    def __init__(self, dataset_name, train=True, 
                 batch_size=64, num_workers=4, 
                 test_size=0.2, seed=None, 
                 shuffle_test=False):

        self.dataset_name = dataset_name
        self.train = train
        self.batch_size = batch_size
        self.num_workers= num_workers

        if self.dataset_name=="typeface_mnist":

            dataset = TMNISTDataset(transform=transforms.Compose([
                transforms.ToPILImage(), 
                transforms.ToTensor(),                 # Normalize to [0, 1]
                transforms.Lambda(lambda x: x * 0.2 + 0.6), # Normalize to [0.6, 0.8]
            ]))

        elif self.dataset_name=="typeface_mnist_augment":

            dataset = TMNISTDataset(transform=transforms.Compose([
                transforms.ToPILImage(), 
                transforms.ToTensor(),                 
                transforms.Lambda(lambda x: x * 0.2 + 0.6), # Normalize to [0.6, 0.8]
                AddCustomTransformations(add_grid=True, add_lines=False)
            ]))

        elif self.dataset_name=="corrupted_mnist":

            root = get_corrupted_data()
            dataset = datasets.ImageFolder(
                root=root,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Grayscale(),
                    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
                    transforms.Lambda(lambda x: x * 0.2 + 0.6) # Normalize to [0.4, 0.8]
                ])
            )

        elif self.dataset_name=="affine_mnist":

            dataset = AffineMNISTDataset(transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(), 
                transforms.Grayscale(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x * 0.2 + 0.6)  # Normalize to [0.4, 0.8]
            ]))
        
        elif self.dataset_name=="pam50":

            dataset = PAM50Dataset()


        if train:

            train_size = int((1 - test_size) * len(dataset))
            test_size = len(dataset) - train_size

            if type(seed) == int: torch.manual_seed(seed)

            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            self.train_load = self.train_loader(train_dataset)
            self.evalu_load = self.evalu_loader(test_dataset, shuffle_test)

        else:
            self.data_load = DataLoader(dataset, 
                                        batch_size=batch_size, 
                                        shuffle=train, 
                                        num_workers=num_workers)
            

    def train_loader(self, train_dataset):

        return DataLoader(train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
        
    def evalu_loader(self, eval_dataset, shuffle=False):

        return DataLoader(eval_dataset, batch_size=self.batch_size,
                          shuffle=shuffle, num_workers=self.num_workers)
