#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:53:16 2024

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:23:59 2024

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


import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#####################################
#####################################


from wind_dataset_preparation_psr import (
    extract_pressure_for_germany,
    extract_wind_speed_for_germany,
    load_real_wind_csv,
    interpolate_wind_speed,
    loading_wind,
    interpolate_pressure,
    scale_interpolated_data,
    combine_data,
    repeat_target_points,
    scale_target_points
    )


from wind_dataset_preparation import (WindDataGen,
                                      RNNDataPreparation,
                                      LSTMDataPreparation,
                                      HybridDataPreparation)

from wind_deep_simulation_framework import (WindDeepModel,
                                            RNNDeepModel,
                                            LSTMDeepModel,
                                            HybridLSTMTransformerModel)
from wind_loss import wind_loss_func

from wind_trainer import (Trainer,
                          RNNTrainer,
                          LSTMTrainer,
                          HybridModelTrainer)


######################################
######################################

# Example usage
nc_file_path = 'nc_files/dataset-projections-2020/ps_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r3i1p1_GERICS-REMO2015_v1_3hr_202001010100-202012312200.nc'
csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'

# Extract pressure data
pressure_data, grid_lats, grid_lons = extract_pressure_for_germany(nc_file_path)



# Example usage
nc_file_path = 'nc_files/Klima_Daten_10m_3h_2020_RCP26.nc'
csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'

wind_speeds, grid_lats, grid_lons = extract_wind_speed_for_germany(nc_file_path)



print(f"Shape of extracted wind speed: {wind_speeds.shape}")
print(f"Sample of extracted wind speed (first 5 time steps, first 5 locations):")


target_points = load_real_wind_csv(csv_file_path)
interpolated_wind_speeds = interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points)

scaled_unix_time_array, filtered_x_y, filtered_wind_power = loading_wind()

interpolated_pressure = interpolate_pressure(pressure_data, grid_lats, grid_lons, target_points)



scaled_wind_speeds = scale_interpolated_data(interpolated_wind_speeds)


scaled_pressure = scale_interpolated_data(interpolated_pressure)

scaled_wind_power = scale_interpolated_data(filtered_wind_power)


scaled_target_points = scale_target_points(target_points)

# Number of time steps (from scaled_wind_speeds)
num_time_steps = scaled_wind_speeds.shape[0]
repeated_scaled_target_points = repeat_target_points(scaled_target_points, num_time_steps)

print(f"Shape of repeated_scaled_target_points: {repeated_scaled_target_points.shape}")



# Combine the data
combined_array = combine_data(scaled_target_points, scaled_unix_time_array,
                              scaled_wind_speeds,
                              scaled_pressure,
                              scaled_wind_power)


############################################################
############################################################


# Example usage
# Prepare your dataset using HybridDataPreparation
hybrid_data_prep = HybridDataPreparation(combined_array[:,:5], combined_array[:,5:], seq_length=1)

# Set the test data size
test_data_size = 0.05

# Prepare the training and test data loaders
x_train_seq, u_train_seq, train_loader, test_loader = hybrid_data_prep.prepare_data_random(test_data_size)


# Define the HybridLSTMTransformerModel
input_size = 5  # Number of input features
lstm_hidden_size = 64  # Number of LSTM hidden units, adjusted to be divisible by transformer_num_heads
lstm_num_layers = 6  # Number of LSTM layers
transformer_num_heads = 4   # Number of attention heads
transformer_hidden_size = 20  # Size of the feed-forward network after attention
transformer_num_layers = 2  # Number of transformer layers
output_size = 1  # Number of output features
learning_rate = 1e-3

hybrid_model_instance = HybridLSTMTransformerModel(
    input_size, lstm_hidden_size, lstm_num_layers,
    transformer_num_heads, transformer_hidden_size, transformer_num_layers,
    output_size, dropout=0.1, learning_rate=learning_rate
)


model_str, optim_adam, scheduler = hybrid_model_instance.run()






#####################################################
#####################################################




num_epochs = 10000


Train_inst = HybridModelTrainer(
    model_str,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
    
)

loss_func_list = Train_inst.train_func(train_loader, test_loader)






import numpy as np

# Save the NumPy array
np.save(f'model_repo/loss_func_list_{num_epochs}_{lstm_hidden_size}_{lstm_num_layers}_{transformer_hidden_size}_{transformer_num_layers}_hybrid.npy'
        , loss_func_list)

# To load it back later
loaded_loss = np.load(f'model_repo/loss_func_list_{num_epochs}_{lstm_hidden_size}_{lstm_num_layers}_{transformer_hidden_size}_{transformer_num_layers}_hybrid.npy')


##########################################
##########################################

import torch

# Define the paths to save the model
model_save_path_gpu = f'model_repo/wind_deep_model_{num_epochs}_{lstm_hidden_size}_{lstm_num_layers}_{transformer_hidden_size}_{transformer_num_layers}_hybrid_gpu.pth'
model_save_path_cpu = f'model_repo/wind_deep_model_{num_epochs}_{lstm_hidden_size}_{lstm_num_layers}_{transformer_hidden_size}_{transformer_num_layers}_hybrid_cpu.pth'

# Save the trained model for GPU
torch.save(model_str.state_dict(), model_save_path_gpu)
print(f"Model saved to {model_save_path_gpu}")

# Save the trained model for CPU
torch.save(model_str.state_dict(), model_save_path_cpu, _use_new_zipfile_serialization=False)
print(f"Model saved to {model_save_path_cpu}")

# Alternatively, save a CPU-compatible version directly by remapping the state dictionary
cpu_state_dict = {k: v.to('cpu') for k, v in model_str.state_dict().items()}
torch.save(cpu_state_dict, model_save_path_cpu)
print(f"CPU model saved to {model_save_path_cpu}")



