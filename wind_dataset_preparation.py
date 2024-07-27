#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:20:57 2024

@author: forootan
"""

import numpy as np
import sys
import os


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir


root_dir = setting_directory(1)

from pathlib import Path
import torch
from scipy import linalg
import torch.nn as nn
import torch.nn.init as init

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time

warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(7)
#CUDA support


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod






#############################

class DataPreparation(ABC):
    def __init__(self, coords, data):
        self.data = data
        self.coords = coords

    @abstractmethod
    def prepare_data_random(self, test_data_size):
        pass
    
    
    #@abstractmethod
    #def prepare_data_DEIM(self):
    #    pass

#####################################################
#####################################################


class WindDataGen(DataPreparation):
    def __init__(self, coords, data, noise_level = None):
        
        
        super().__init__(coords, data)
        
        self.coords = coords
        self.data = data
        
    
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        
        print(type(self.coords))
        print(type(self.data))
        
        
        u_trains = []
        x_trains = []
        
        
        (
            x_train,
            u_train,
            x_test,
            u_test,
        ) = self.train_test_split(self.coords, self.data, test_data_size)
        
        print(f"Type of X: {x_train.dtype}")
        print(f"Type of Y: {u_train.dtype}")

        
       

        batch_size_1 = x_train.shape[0]

        train_loader = self.data_loader(x_train, u_train, 2500)
        test_loader = self.data_loader(x_test, u_test, 2500)
       
        

        train_test_loaders = [
            train_loader,
            test_loader,
       
        ]
        
        
        
        return x_train, u_train, train_test_loaders

    

    def train_test_split(self, x, data, test_data_size):
        

        x_train, x_test, data_train, data_test = train_test_split(
            x, data, test_size=test_data_size, random_state=42
        )
        
        x_train = np.array(x_train, dtype=np.float32)
        data_train = np.array(data_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)
        data_test = np.array(data_test, dtype=np.float32)
        

        return x_train, data_train, x_test, data_test
    
     
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader

    