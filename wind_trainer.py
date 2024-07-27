#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:52:55 2024

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:35:53 2023
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

from siren_modules import Siren



from wind_loop_process import WindLoopProcessor

from wind_loss import wind_loss_func

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
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod

##################################

class WindTrain(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def train_func(self):
        pass


################################################
################################################


"""
Train_inst = Trainer(
    model_str,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
    loss_func = wind_loss_func
)
"""

class Trainer(WindTrain):
    def __init__(
        self,
        model_str,
        optim_adam,
        scheduler,
        wind_loss_func,
        num_epochs=1500,
    ):
        """
        # Usage
        # Define your train loaders, features_calc_AC, calculate_theta_AC, loss_func_AC, etc.
        # Create optimizer and scheduler objects
        # Instantiate the EnsembleTrainer class
        # Call the train method on the instance
        
        # Example Usage:
        # ensemble_trainer = EnsembleTrainer(model_str, num_epochs, optim_adam, scheduler)
        # ensemble_trainer.train(train_loader, features_calc_AC, calculate_theta_AC, loss_func_AC)
        """
        
        super().__init__()
        self.model_str = model_str
        self.num_epochs = num_epochs
        self.optim_adam = optim_adam
        self.scheduler = scheduler
        self.wind_loss_func = wind_loss_func

        self.loss_total = []
        self.coef_s = []
        

    def train_func(
        self,
        train_loader,
    ):
        #loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        loop = tqdm(range(self.num_epochs), leave=False)
        
        for epoch in loop:
            #tqdm.write(f"Epoch: {epoch}")

            loss_data = 0
            start_time = time.time()

            ####################################################
            wind_loss_instance = WindLoopProcessor(
                self.model_str, self.wind_loss_func
            )
            
            
            loss_data= wind_loss_instance(train_loader)
          

            ####################################################
            loss = loss_data
            self.loss_total.append(loss.cpu().detach().numpy())
            self.optim_adam.zero_grad()
            loss.backward()
            self.optim_adam.step()

            # scheduler step
            self.scheduler.step()
            
           
            
            #loop.set_description(f"Epoch [{epoch}/{self.num_epochs}]")
            loop.set_postfix(
                training_loss=loss.item(),)
            

        self.loss_total = np.array(self.loss_total)
        

        loss_func_list = [
            self.loss_total,
            
        ]
       

        return loss_func_list


################################################
################################################









