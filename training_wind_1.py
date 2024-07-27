#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:58:03 2024

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
root_dir = setting_directory(0)



from pathlib import Path
import torch
from scipy import linalg
import torch.nn as nn
import torch.nn.init as init
from siren_modules import Siren

import warnings
import time

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
from normalized_wind_data_final import loading_wind, combine_data

from wind_dataset_preparation import WindDataGen
from wind_deep_simulation_framework import WindDeepModel
from wind_loss import wind_loss_func
from wind_trainer import Trainer



scaled_unix_time_array, normalized_x_y, filtered_wind_power = loading_wind()


combined_array = combine_data(normalized_x_y, scaled_unix_time_array, filtered_wind_power)

# Check the shape of the combined array
print(combined_array.shape)

type(combined_array)


wind_dataset_instance = WindDataGen(combined_array[:,:3], combined_array[:,3:])

test_data_size = 0.2


x_train, u_train, train_test_loaders = wind_dataset_instance.prepare_data_random(0.9)




WindDeepModel_instance = WindDeepModel(
    in_features=3,
    out_features=1,
    hidden_features_str=128,
    hidden_layers=10,
    learning_rate_inr=1e-5,)

models_list, optim_adam, scheduler = WindDeepModel_instance.run()

model_str = models_list[0]


noise = 0
learning_rate_inr = 1e-5
hidden_features_str = 128
hidden_features = 64
hidden_layers = 3
num_epochs = 5000
prec = 1 - test_data_size


Train_inst = Trainer(
    model_str,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
    wind_loss_func = wind_loss_func
)


loss_func_list = Train_inst.train_func(
    train_test_loaders[0]
)

print(loss_func_list)
