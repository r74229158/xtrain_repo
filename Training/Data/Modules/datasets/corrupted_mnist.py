import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import os
import kagglehub
import shutil


def get_corrupted_data():

    target_path = os.getcwd() + "/Training/Data/Datasets/CorruptedMNIST/mnist"
    data_path = Path(target_path)

    if data_path.exists()==False:

        download_path = kagglehub.dataset_download("shreyasi2002/corrupted-mnist")
        shutil.copytree(download_path, target_path)
    
    return target_path

