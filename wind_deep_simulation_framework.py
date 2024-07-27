#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:42:41 2023

@author: forootani
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

root_dir = setting_directory(0)

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


from siren_modules import Siren


warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod


#######################################
#######################################    


class DeepSimulation(ABC):
    def __init__(self, ):            
        pass
    
    @abstractmethod
    def nn_models(self, ):
        pass
    
    @abstractmethod
    def optimizer_func(self, ):
        pass
    
    @abstractmethod
    def scheduler_setting(self,):
        pass
    pass

    
class WindDeepModel(DeepSimulation):
    def __init__(self,  in_features, out_features,
                 hidden_features_str, 
                 hidden_layers,  learning_rate_inr=1e-5
                 ):            
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.hidden_features_str = hidden_features_str
        self.learning_rate_inr = learning_rate_inr
    
    
    def nn_models(self, ):
        
        # siren model initialization
        self.model_str_1 = Siren(
            self.in_features,
            self.hidden_features_str,
            self.hidden_layers,
            self.out_features,
            outermost_linear=True,
        ).to(device)


        
        models_list = [self.model_str_1, 
                       ]
        
        return models_list
        
    
    def optimizer_func(self, ):
        
        self.optim_adam = torch.optim.Adam(
            [
                {
                    "params": self.model_str_1.parameters(),
                    "lr": self.learning_rate_inr,
                    "weight_decay": 1e-6,
                },
               
                
            ]
        )
        
        return self.optim_adam
        
    
    def scheduler_setting(self):
            
        scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optim_adam,
                base_lr=0.1 * self.learning_rate_inr,
                max_lr=10 * self.learning_rate_inr,
                cycle_momentum=False,
                mode="exp_range",
                step_size_up=1000,
            )
            
        return scheduler
        
        
    def run(self):
            
        models_list = self.nn_models()
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()
        
        return models_list, optimizer, scheduler

        
################################################
################################################ 






















    
    