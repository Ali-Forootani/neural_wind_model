#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:41:07 2024

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

#########################################
#########################################



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


###################################################
###################################################

import torch
import numpy as np
from wind_deep_simulation_framework import HybridLSTMTransformerModel

# Define model parameters (same as used in training)
input_size = 5  # Number of input features
lstm_hidden_size = 20  # LSTM hidden units
lstm_num_layers = 4  # Number of LSTM layers
transformer_num_heads = 4  # Attention heads
transformer_hidden_size = 20  # Feed-forward network size after attention
transformer_num_layers = 4  # Transformer layers
output_size = 1  # Number of output features


num_epochs = 2000




"""
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
    output_size, dropout=0.1, learning_rate=learning_rate)

num_epochs = 5000

"""


"""
# Define model parameters (same as used in training)
input_size = 5  # Number of input features
lstm_hidden_size = 64  # LSTM hidden units
lstm_num_layers = 6  # Number of LSTM layers
transformer_num_heads = 4  # Attention heads
transformer_hidden_size = 20  # Feed-forward network size after attention
transformer_num_layers = 2  # Transformer layers
output_size = 1  # Number of output features


num_epochs = 10000
"""





# Define the paths
model_save_path_cpu = f'model_repo/wind_deep_model_{num_epochs}_{lstm_hidden_size}_{lstm_num_layers}_{transformer_hidden_size}_{transformer_num_layers}_hybrid_cpu.pth'



# Initialize model
model = HybridLSTMTransformerModel(
    input_size, lstm_hidden_size, lstm_num_layers,
    transformer_num_heads, transformer_hidden_size, transformer_num_layers,
    output_size
)

# Load model state dict
device = torch.device('cpu')  # Use CPU
model.load_state_dict(torch.load(model_save_path_cpu, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode




# Example usage
# Prepare your dataset using HybridDataPreparation
hybrid_data_prep = HybridDataPreparation(combined_array[:,:5], combined_array[:,5:], seq_length=10)

# Set the test data size
test_data_size = 0.5

# Prepare the training and test data loaders
x_train_seq, u_train_seq, train_loader, test_loader = hybrid_data_prep.prepare_data_random(test_data_size)





############################################3


import torch
from sklearn.metrics import mean_squared_error

# Assuming you have imported the `wind_loss_func` from your `wind_loss` module
# or you can use `mean_squared_error` from sklearn for evaluation

# Switch model to evaluation mode
model.eval()

# Initialize a list to store predicted and actual values
all_predictions = []
all_targets = []

# Loop through the test data loader to make predictions
with torch.no_grad():  # Disable gradient calculation for evaluation
    for x_test_batch, u_test_batch in test_loader:
        # Move tensors to the correct device
        x_test_batch = x_test_batch.to(device)
        u_test_batch = u_test_batch.to(device)

        # Forward pass: predict
        predictions = model(x_test_batch)

        # Store the predictions and targets for later analysis
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(u_test_batch.cpu().numpy())

# Convert lists to arrays for easier handling
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Calculate Mean Squared Error as an example metric
mse = mean_squared_error(all_targets, all_predictions)
print(f"Mean Squared Error on the test set: {mse}")

# Plotting actual vs predicted for visual evaluation
plt.figure(figsize=(10, 5))
plt.plot(all_targets[200:3000], label='Actual')
plt.plot(all_predictions[200:3000], label='Predicted', linestyle='dashed')
plt.title("Model Prediction vs Actual Values")
plt.xlabel("Time steps")
plt.ylabel("Wind Power")
plt.legend()
plt.grid(True)
plt.show()





########################################
########################################
########################################


max_wind_power = filtered_wind_power.max()
min_wind_power = filtered_wind_power.min()


# First, apply the inverse scaling to all_predictions and all_true_values

# Inverse scaling function
def inverse_scale(scaled_data, min_value, max_value):
    return (scaled_data + 1) / 2 * (max_value - min_value) + min_value

# Apply inverse scaling to the predictions and true values
all_predictions_original_scale = inverse_scale(all_predictions, min_wind_power, max_wind_power)
all_true_values_original_scale = inverse_scale(all_targets, min_wind_power, max_wind_power)

import matplotlib.pyplot as plt

# Plot for model predictions vs true values with a grid
plt.figure(figsize=(10, 6))

# Plot true values with a solid dark green line
plt.plot(all_true_values_original_scale[400:900], label='Measured Wind Power ', alpha=1.0,
         linewidth=3.5, linestyle='-', color='green')

# Plot predicted values with a dashed dark red line
plt.plot(all_predictions_original_scale[400:900], label='Transformer-LSTM-DNN Predicted Wind Power ', alpha=1.0,
         linewidth=3.5, linestyle='--', color='black')

# Set the title and labels with larger fonts
plt.title('Transformer-LSTM-DNN Model Predictions vs Measured Power (KWh)', fontsize=18)
plt.xlabel('Sample Index', fontsize=18)
plt.ylabel('Wind Power Generation', fontsize=18)

# Display the legend with a larger font
plt.legend(fontsize=16)

# Increase the font size of tick labels on both axes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Turn on the grid
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray')

# Display the plot

# Set the directory where images will be saved
output_dir = 'images_wind'

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(f'{output_dir}/Transformer_LSTM_vs_True.png', bbox_inches='tight')

plt.show()
#########################################################

# Optional: Scatter plot with a grid
plt.figure(figsize=(8, 8))
plt.scatter(all_true_values_original_scale, all_predictions_original_scale, alpha=0.8, s=60, color='black')

# Plot the reference line in dark green
plt.plot([min(all_true_values_original_scale), max(all_true_values_original_scale)], 
         [min(all_true_values_original_scale), max(all_true_values_original_scale)], color='blue', linestyle='--', linewidth=3.5)

# Set the title and labels with larger fonts
plt.title('Scatter Plot of Predictions vs True Values (Original Scale)', fontsize=18)
plt.xlabel('True Values (Original Scale)', fontsize=18)
plt.ylabel('Predicted Values (Original Scale)', fontsize=18)

# Increase the font size of tick labels on both axes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Turn on the grid
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray')

# Display the plot
plt.show()



#####################################################
#####################################################
#####################################################
#####################################################



import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

# Define font size for plot elements
font_size = 16

# Scatter plot of true vs predicted values
plt.figure(figsize=(12, 6))
plt.suptitle("Transformer-LSTM-DNN for Wind Power Forecast", fontsize=font_size + 4)



# Plot first feature vs target
plt.subplot(1, 2, 1)
plt.scatter(all_targets[:5000], all_predictions[:5000], alpha=0.5, color='royalblue')
plt.xlabel('True Values (Test Data)', fontsize=font_size)
plt.ylabel('Predictions', fontsize=font_size)
plt.title('True vs Predicted Values (Test Data)', fontsize=font_size)
plt.grid(True)

# Use scientific notation for axes tick labels
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_offset_text().set_fontsize(font_size)
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().yaxis.get_offset_text().set_fontsize(font_size)

# Plot histogram of prediction errors
plt.subplot(1, 2, 2)
test_errors = all_predictions[:, 0] - all_targets[:, 0]  # Assuming 1D target values for simplicity
plt.hist(test_errors, bins=30, edgecolor='k', color='lightcoral')
plt.xlabel('Prediction Error (Test Data)', fontsize=font_size)
plt.ylabel('Frequency', fontsize=font_size)
plt.title('Histogram of Prediction Errors (Test Data)', fontsize=font_size)
plt.grid(True)

# Use scientific notation for frequency axis
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
plt.gca().yaxis.get_offset_text().set_fontsize(font_size)

plt.tight_layout()


output_dir = 'images_wind'

plt.savefig(f'{output_dir}/histogram_prediction_Transformer_LSTM_error.png', bbox_inches='tight')

plt.show()


plt.show()




























