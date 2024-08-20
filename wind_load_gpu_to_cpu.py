#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 08:28:17 2024

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


from wind_dataset_preparation import WindDataGen
from wind_deep_simulation_framework import WindDeepModel
from wind_loss import wind_loss_func
from wind_trainer import Trainer



##################################
##################################

num_epochs = 10000



hidden_layers = 6
hidden_features = 64



# Saving the model and using it!

# Define the path to save the model
#model_save_path = f'model_repo/wind_deep_model_{num_epochs}_{hidden_features}_{hidden_layers}_cpu.pth'

# Save the trained model
#torch.save(model_str.state_dict(), model_save_path)
#print(f"Model saved to {model_save_path}")


# Define the path where the model is saved
model_load_path = f'model_repo/wind_deep_model_{num_epochs}_{hidden_features}_{hidden_layers}_gpu.pth'


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
#loaded_model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
#print(f"Model loaded from {model_load_path}")


# Save the model's state dictionary for CPU usage
#torch.save(loaded_model.state_dict(), f'model_repo/wind_deep_model_{num_epochs}_{hidden_features}_{hidden_layers}_cpu.pth')
#print(f"Model loaded from {model_save_path}")


# Set the model to evaluation mode (if you are going to use it for inference)
#loaded_model.eval()


########################################

# Initialize the internal model
loaded_model_list, _, _ = loaded_model_instance.run()
loaded_model = loaded_model_list[0]

# Load the saved state dictionary into the model
loaded_model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
print(f"Model loaded from {model_load_path}")

# Define the new path for saving the model for CPU usage
model_save_path_cpu = f'model_repo/wind_deep_model_{num_epochs}_{hidden_features}_{hidden_layers}_cpu.pth'

# Save the model's state dictionary for CPU usage
torch.save(loaded_model.state_dict(), model_save_path_cpu)
print(f"Model saved to {model_save_path_cpu}")

# Set the model to evaluation mode (if you are going to use it for inference)
loaded_model.eval()









