#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:29:29 2024

@author: forootan
"""

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


from wind_dataset_preparation import WindDataGen, RNNDataPreparation, LSTMDataPreparation
from wind_deep_simulation_framework import WindDeepModel, RNNDeepModel, LSTMDeepModel
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



wind_dataset_instance = LSTMDataPreparation(combined_array[:,:5], combined_array[:,5:])


test_data_size = 0.2


x_train_seq, u_train_seq, train_loader, test_loader = wind_dataset_instance.prepare_data_random(test_data_size)



##################################
##################################

num_epochs = 25000



# Dataset configuration
input_size = 5  # Number of input features
hidden_features = 128  # Number of RNN hidden units
hidden_layers = 6  # Number of RNN layers
output_size = 1  # Number of output features
learning_rate = 1e-3



###################################
###################################


# Saving the model and using it!

# Define the path to save the model
model_save_path = f'model_repo/wind_deep_model_{num_epochs}_{hidden_features}_{hidden_layers}_lstm_cpu.pth'

# Save the trained model
#torch.save(model_str.state_dict(), model_save_path)
#print(f"Model saved to {model_save_path}")


# Define the path where the model is saved
model_load_path = f'model_repo/wind_deep_model_{num_epochs}_{hidden_features}_{hidden_layers}_lstm_cpu.pth'


###################################
###################################


# Create a model instance
loaded_model_instance = LSTMDeepModel(input_size, hidden_features,
                                       hidden_layers, output_size,
                                       learning_rate, )

# Initialize the internal model
loaded_model_list, _, _ = loaded_model_instance.run()
loaded_model = loaded_model_list



# Load the saved state dictionary into the model
loaded_model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')), strict= False)
print(f"Model loaded from {model_load_path}")



# Set the model to evaluation mode (if you are going to use it for inference)
loaded_model.eval()

##################################
##################################



import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Ensure the model is in evaluation mode
loaded_model.eval()

# Initialize lists to store predictions and true values
all_predictions = []
all_true_values = []

# Iterate through the test_loader to get predictions
for x_test_batch, u_test_batch in test_loader:
    # Perform inference (forward pass) on the test batch
    with torch.no_grad():  # Disable gradient calculation
        predictions = loaded_model(x_test_batch)
    
    # Store the predictions and true values
    all_predictions.append(predictions.cpu().numpy())
    all_true_values.append(u_test_batch.cpu().numpy())

# Convert the lists to numpy arrays
all_predictions = np.concatenate(all_predictions, axis=0)
all_true_values = np.concatenate(all_true_values, axis=0)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(all_true_values, all_predictions)
print(f"Mean Squared Error on test set: {mse}")

# Plot the predicted vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(all_true_values[400:900], label='True Values', alpha=0.6)
plt.plot(all_predictions[400:900], label='Predicted Values', alpha=0.6)
plt.title('Model Predictions vs True Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# Optional: Plot a scatter plot of true vs predicted values
plt.figure(figsize=(8, 8))
plt.scatter(all_true_values, all_predictions, alpha=0.5)
plt.plot([min(all_true_values), max(all_true_values)], 
         [min(all_true_values), max(all_true_values)], color='red', linestyle='--')
plt.title('Scatter Plot of Predictions vs True Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()



###############################################

import torch
import matplotlib.pyplot as plt


# Ensure the model is in evaluation mode
loaded_model.eval()

# Get a single batch from the test_loader
for x_test_batch, u_test_batch in test_loader:
    # Move the data to the appropriate device (CPU or GPU)
    x_test_batch = x_test_batch.to(device)
    u_test_batch = u_test_batch.to(device)

    # Forward pass: get predictions from the model
    with torch.no_grad():  # Turn off gradients as we are in evaluation mode
        predictions = loaded_model(x_test_batch)

    # Move predictions and targets back to CPU and convert to numpy
    predictions = predictions.cpu().numpy()
    u_test_batch = u_test_batch.cpu().numpy()

    # Analyze and compare predictions with target values
    print("Sample Predictions:", predictions[:5])
    print("Sample Targets:", u_test_batch[:5])

    # Plot the predictions vs target values
    plt.figure(figsize=(12, 6))

    # Plot first feature vs target
    plt.subplot(1, 2, 1)
    plt.scatter(u_test_batch[:, 0], predictions[:, 0], alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')

    # Plot histogram of prediction errors
    plt.subplot(1, 2, 2)
    errors = predictions[:, 0] - u_test_batch[:, 0]
    plt.hist(errors, bins=30, edgecolor='k')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')

    plt.tight_layout()
    plt.show()

    # Exit the loop after one batch
    break

##############################################################
##############################################################

import torch

# Convert x_train and u_train to PyTorch tensors
x_train_tensor = torch.tensor(x_train_seq, dtype=torch.float32)
u_train_tensor = torch.tensor(u_train_seq, dtype=torch.float32)

# Ensure the model is in evaluation mode
loaded_model.eval()

# Set the batch size to something that fits into memory comfortably
batch_size = 1024  # Adjust this value based on your system's memory capacity

# Number of batches
num_batches = len(x_train_tensor) // batch_size + (1 if len(x_train_tensor) % batch_size != 0 else 0)

# Placeholder for all predictions
train_predictions = []

# Forward pass on the dataset in batches
with torch.no_grad():
    for i in range(num_batches):
        # Select the batch
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x_train_tensor))
        
        x_batch = x_train_tensor[start_idx:end_idx]
        
        # Forward pass for the batch
        batch_predictions = loaded_model(x_batch)
        
        # Store the predictions
        train_predictions.append(batch_predictions.cpu().numpy())

# Concatenate all batch predictions
train_predictions = np.concatenate(train_predictions, axis=0)

# Print the shape of the predictions to ensure they match
print(f"Shape of train_predictions: {train_predictions.shape}")

# Analyze and compare the predictions with the training targets (u_train_tensor)
print("Sample Train Predictions:", train_predictions[:5])
print("Sample Train Targets:", u_train_seq[:5])
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Increase font sizes
font_size = 16
plt.figure(figsize=(12, 6))
plt.suptitle("LSTM-DNN for Wind Power Forecast", fontsize=font_size + 4)


# Set scientific style for axes
plt.rc('axes', labelsize=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)

# Plot first feature vs target
plt.subplot(1, 2, 1)
plt.scatter(u_train_seq[:5000], train_predictions[:5000], alpha=0.5, color='royalblue')
plt.xlabel('True Values (Training Data)', fontsize=font_size)
plt.ylabel('Predictions', fontsize=font_size)
plt.title('True vs Predicted Values', fontsize=font_size)
plt.grid(True)

# Use scientific notation for axes tick labels
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_offset_text().set_fontsize(font_size)
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().yaxis.get_offset_text().set_fontsize(font_size)

# Plot histogram of prediction errors
plt.subplot(1, 2, 2)
train_errors = train_predictions[:, 0] - u_train_tensor[:, 0].numpy()
plt.hist(train_errors, bins=30, edgecolor='k', color='lightcoral')
plt.xlabel('Prediction Error (Training Data)', fontsize=font_size)
plt.ylabel('Frequency', fontsize=font_size)
plt.title('Histogram of Prediction Errors', fontsize=font_size)
plt.grid(True)

# Use scientific notation for frequency axis
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
plt.gca().yaxis.get_offset_text().set_fontsize(font_size)

plt.tight_layout()


output_dir = 'images_wind'

plt.savefig(f'{output_dir}/histogram_prediction_LSTM_error.png', bbox_inches='tight')

plt.show()



##############################################################
##############################################################


import numpy as np

# Load the array from a .npy file
training_loss = np.load(root_dir + f'/model_repo/loss_func_list_{num_epochs}_{hidden_features}_{hidden_layers}_lstm.npy')

print(training_loss)
plt.plot(training_loss)


####################################
####################################

scaled_data = train_predictions

#X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#X_scaled = X_std * (max - min) + min

max_wind_power = filtered_wind_power.max()

min_wind_power = filtered_wind_power.min()

inverse_scaled = (scaled_data + 1) /2

original_range_train = inverse_scaled * (filtered_wind_power.max() -  filtered_wind_power.min()) + filtered_wind_power.min()


time_step = 0
wind_power_prediction = []



###########################################
###########################################


import torch

time_step = 0
wind_power_prediction = []

max_wind_power = filtered_wind_power.max()
min_wind_power = filtered_wind_power.min()

wind_power_prediction = np.zeros((target_points.shape[0], scaled_unix_time_array.shape[0]))

#scaled_unix_time_array.shape[0]

for k in range(50):
    for i in range(target_points.shape[0]):
        for j in range(target_points.shape[1]):
        
            x = scaled_target_points[i, 0]
            y = scaled_target_points[i, 1]
            pr = scaled_pressure[k, i]
            wind_s = scaled_wind_speeds[k, i]
            t = scaled_unix_time_array[k][0]
        
            input_tensor = torch.tensor([x, y, t, pr, wind_s], dtype=torch.float32)
            
            # Unsqueeze to make the input 3D: (1, seq_length=1, input_size=5)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        
            with torch.no_grad():
                output_predictions = loaded_model(input_tensor)
                
                # Rescale the predictions back to the original range
                predictions = ((output_predictions + 1) / 2) * (max_wind_power - min_wind_power) + min_wind_power
                
                wind_power_prediction[i, k] = predictions.numpy()

# Optionally, you could now analyze or save the wind_power_prediction array



                
###########################################
###########################################            
    
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


def visualize_wind_power_map(filtered_x_y, filtered_wind_power, time_step=0):
    # Check the time_step range
    if time_step >= filtered_wind_power.shape[1]:
        raise ValueError(f"time_step ({time_step}) exceeds the number of available time steps ({filtered_wind_power.shape[1]})")

    # Get the wind powers for the specified time step
    wind_powers = filtered_wind_power[:, time_step]
    
    # Calculate mean wind power for each location across all time steps
    mean_wind_powers = np.mean(filtered_wind_power, axis=1)
    
    # Ensure filtered_x_y has the same number of rows as filtered_wind_power
    if filtered_x_y.shape[0] != filtered_wind_power.shape[0]:
        raise ValueError(f"Shape mismatch: filtered_x_y has {filtered_x_y.shape[0]} rows, but filtered_wind_power has {filtered_wind_power.shape[0]} rows")

    # Determine the min and max values for normalization
    min_power = np.min(wind_powers)
    max_power = np.max(wind_powers)
    
    fig, ax = plt.subplots(figsize=(18, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    ax.set_facecolor('lightgrey')

    # Plot target points
    ax.scatter(filtered_x_y[:, 1], filtered_x_y[:, 0], color='black', s=5, alpha=0.3, transform=ccrs.PlateCarree(), label='Measurement Points')

    # Scatter plot for mean wind powers with colormap
    sc = ax.scatter(filtered_x_y[:, 1], filtered_x_y[:, 0], 
                    c=wind_powers, cmap='viridis',  
                    s=50, alpha=0.7, transform=ccrs.PlateCarree(), 
                    vmin=min_power, vmax=max_power, label='Wind Power')

    # Add mean wind power annotation
    ax.text(0.05, 0.95, f'Overall Mean Wind Power: {np.mean(mean_wind_powers):.2f} kW', 
            fontsize=18, fontweight='bold', color='black',
            transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    # Customize axis labels and title
    ax.set_title(f'DNN Wind Power Map - Time Step: {time_step}\n'
                 f'DNN Wind Power Range: {min_power:.2f} - {max_power:.2f} kW',
                 fontsize=24, fontweight='bold')
    
    # Move the legend to a more visible location
    ax.legend(loc='upper right', fontsize=18, frameon=True, shadow=True)

    # Adjust colorbar placement and padding
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.025, pad=0.08)
    cbar.set_label('Wind Power (kW)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    # Customize gridlines
    gridlines = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.4)
    gridlines.xlabel_style = {'size': 20}
    gridlines.ylabel_style = {'size': 20}
    
    # Customize tick labels font size
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)  # Explicitly set font size for all tick labels
        
        
    # Set the directory where images will be saved
    output_dir = 'images_wind'

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(f'{output_dir}/DNN_wind_power_production_3h.png', bbox_inches='tight')
    
    
    plt.show()
    
    
    
#scaled_unix_time_array, filtered_x_y, filtered_wind_power = loading_wind()

# Visualize the wind power production map
visualize_wind_power_map(filtered_x_y, wind_power_prediction, time_step = 9)

    
        
# Visualize the wind power production map
visualize_wind_power_map(filtered_x_y, filtered_wind_power, time_step = 9)


#########################################
#########################################



# First, apply the inverse scaling to all_predictions and all_true_values

# Inverse scaling function
def inverse_scale(scaled_data, min_value, max_value):
    return (scaled_data + 1) / 2 * (max_value - min_value) + min_value

# Apply inverse scaling to the predictions and true values
all_predictions_original_scale = inverse_scale(all_predictions, min_wind_power, max_wind_power)
all_true_values_original_scale = inverse_scale(all_true_values, min_wind_power, max_wind_power)

import matplotlib.pyplot as plt

# Plot for model predictions vs true values with a grid
plt.figure(figsize=(10, 6))

# Plot true values with a solid dark green line
plt.plot(all_true_values_original_scale[400:900], label='Measured Wind Power ', alpha=1.0,
         linewidth=3.5, linestyle='-', color='green')

# Plot predicted values with a dashed dark red line
plt.plot(all_predictions_original_scale[400:900], label='LSTM-DNN Predicted Wind Power ', alpha=1.0,
         linewidth=3.5, linestyle='--', color='black')

# Set the title and labels with larger fonts
plt.title('LSTM-DNN Model Predictions vs Measured Power (KWh)', fontsize=18)
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

plt.savefig(f'{output_dir}/LSTM_vs_True.png', bbox_inches='tight')

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

















