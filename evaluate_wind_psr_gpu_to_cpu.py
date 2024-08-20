#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:27:39 2024

@author: forootan
"""


###################################
###################################


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


from wind_dataset_preparation import WindDataGen
from wind_deep_simulation_framework import WindDeepModel
from wind_loss import wind_loss_func
from wind_trainer import Trainer


###################################
###################################


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
print(wind_speeds[:5, :5])

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



# Check the shape of the combined array
print(combined_array.shape)

type(combined_array)


wind_dataset_instance = WindDataGen(combined_array[:,:5], combined_array[:,5:])

test_data_size = 0.2


x_train, u_train, train_test_loaders = wind_dataset_instance.prepare_data_random(0.9)



# Access the test_loader from train_test_loaders
test_loader = train_test_loaders[1]

# Iterate through the test_loader to get batches of data
for x_test_batch, u_test_batch in test_loader:
    # Now x_test_batch contains the features and u_test_batch contains the targets
    print(x_test_batch)
    print(u_test_batch)
    break  # Remove this break if you want to iterate over all batches


# Access the test_loader from train_test_loaders
test_loader = train_test_loaders[1]

# Calculate the number of batches and print batch sizes
num_batches = len(test_loader)
print(f"Number of batches in test_loader: {num_batches}")

# Loop through the test_loader to see the size of each batch
#for i, (x_test_batch, u_test_batch) in enumerate(test_loader, 1):
#    print(f"Batch {i}: Size of x_test_batch = {x_test_batch.size()}, Size of u_test_batch = {u_test_batch.size()}")


##############################################################



##########################################
##########################################



##################################
##################################

num_epochs = 10000



hidden_layers = 8
hidden_features = 128



# Saving the model and using it!

# Define the path to save the model
model_save_path = f'model_repo/wind_deep_model_{num_epochs}_{hidden_features}_{hidden_layers}.pth'

# Save the trained model
#torch.save(model_str.state_dict(), model_save_path)
#print(f"Model saved to {model_save_path}")


# Define the path where the model is saved
model_load_path = f'model_repo/wind_deep_model_{num_epochs}_{hidden_features}_{hidden_layers}.pth'


###################################
###################################

# Create a model instance
loaded_model_instance = WindDeepModel(
    in_features=5,
    out_features=1,
    hidden_features_str = hidden_features,
    hidden_layers = hidden_layers,
    learning_rate_inr=1e-5,
)

# Initialize the internal model
loaded_model_list, _, _ = loaded_model_instance.run()
loaded_model = loaded_model_list[0]

# Load the saved state dictionary into the model
loaded_model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
print(f"Model loaded from {model_load_path}")


# Save the model's state dictionary for CPU usage
torch.save(loaded_model.state_dict(), f'model_repo/wind_deep_model_{num_epochs}_{hidden_features}_{hidden_layers}_cpu.pth')

