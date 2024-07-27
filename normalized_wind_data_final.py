#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:26:33 2024

@author: forootan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from datetime import datetime


# Function to clean the date-time string by removing fractional seconds
def clean_date_time_string(date_time_str):
    # Split on '.' and keep the part before it
    return date_time_str.split('.')[0]

# Function to check if the date-time string is in the correct format
def is_valid_date_time(date_time_str, date_time_format):
    try:
        datetime.strptime(date_time_str, date_time_format)
        return True
    except ValueError:
        return False

# Function to convert to Unix time
def convert_to_unix_time(date_time_str, date_time_format):
    try:
        dt = datetime.strptime(date_time_str, date_time_format)
        return int(dt.timestamp())
    except ValueError:
        return None



def format_time_difference(start_timestamp, end_timestamp):
    """
    Calculate the time difference between two Unix timestamps and return it
    in a human-readable format (days, hours, minutes, and seconds).

    Parameters:
    - start_timestamp (int): The start Unix timestamp.
    - end_timestamp (int): The end Unix timestamp.

    Returns:
    - str: A human-readable string describing the time difference.
    """
    # Calculate the difference in seconds
    total_seconds = end_timestamp - start_timestamp

    # Convert seconds to days, hours, minutes, and seconds
    days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)      # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)        # 60 seconds in a minute

    # Format the result as a string
    return f"Time difference: {days} days, {hours} hours, {minutes} minutes, and {seconds} seconds"



def map_unix_time_to_range(unix_time_array, feature_range=(-1, 1)):
    """
    Map the unix_time_array to the specified range using MinMaxScaler.

    Parameters:
    - unix_time_array (np.array): The array of Unix timestamps.
    - feature_range (tuple): Desired range of transformed data.

    Returns:
    - np.array: Scaled array with values mapped to the specified range.
    """
    # Reshape the array to fit the scaler's expected input shape
    unix_time_array_reshaped = unix_time_array.reshape(-1, 1)
    
    # Initialize the scaler with the desired feature range
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Fit and transform the unix_time_array
    scaled_unix_time_array = scaler.fit_transform(unix_time_array_reshaped)
    
    # Flatten the array back to its original shape
    return scaled_unix_time_array.flatten()









def loading_wind():
    
    
    # Replace 'path_to_your_file.csv' with the actual path to your CSV file
    csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'

    # Load the CSV file into a DataFrame with low_memory=False to avoid DtypeWarning
    df = pd.read_csv(csv_file_path, low_memory=False)

    # Drop rows with any NaN values
    df.dropna(axis=0, how='any', inplace=True)

    # Convert the cleaned DataFrame to a NumPy array
    wind_data = df.to_numpy()

    # Extract x_y and the relevant part of the data for normalization
    x_y = wind_data[:, 1:3]
    data_to_normalize = wind_data[:, 6:wind_data.shape[1]-1]

    # Identify and count the rows where all values from column 5 onwards are zero
    num_all_zeros = np.sum(np.all(wind_data[:, 5:] == 0, axis=1))
    print("Number of rows where all values from column 5 onwards are zero:", num_all_zeros)

    # Create a mask to filter out rows where not all values from column 5 onwards are zero
    mask = np.any(wind_data[:, 5:] != 0, axis=1)

    # Apply the mask to filter wind_data and x_y
    filtered_wind_data = wind_data[mask]
    filtered_x_y = x_y[mask]

    # Normalize the filtered data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(filtered_wind_data[:, 6:wind_data.shape[1]-1])

    # Replace the original data with the normalized data
    filtered_wind_data[:, 6:wind_data.shape[1]-1] = normalized_data

    # Normalize the filtered_x_y data
    x_y_scaler = StandardScaler()
    normalized_x_y = x_y_scaler.fit_transform(filtered_x_y)

    # Replace the original x_y data with the normalized data
    # filtered_wind_data[:, 1:3] = normalized_x_y
    
    
    
    
    
    filtered_wind_power = filtered_wind_data[:,6:wind_data.shape[1]-1]
    
    
    ###########################################
    ###########################################
    
    # Define the date-time format
    date_time_format = '%d/%m/%y %H:%M'
    
    
    # Extract the headers (column names)
    headers = df.columns.tolist()

    # Print the headers
    #print(headers)

    # Identify date-time headers (from the 5th column onwards)
    date_time_headers = headers[6:-1]

    
    # Check each date-time header and clean or convert if necessary
    cleaned_date_time_headers = []
    invalid_indices = []

    for index, date_time_str in enumerate(date_time_headers):
        cleaned_str = clean_date_time_string(date_time_str)
        
        if not is_valid_date_time(cleaned_str, date_time_format):
            invalid_indices.append(index)
            print(f"Index {index} is invalid: '{date_time_str}' (cleaned: '{cleaned_str}')")
        else:
            cleaned_date_time_headers.append(cleaned_str)

    # Apply the conversion to cleaned date-time headers
    unix_times = [convert_to_unix_time(dt, date_time_format) for dt in cleaned_date_time_headers if convert_to_unix_time(dt, date_time_format) is not None]

    # Convert to NumPy array
    unix_time_array = np.array(unix_times)
    
    
    scaled_unix_time_array = map_unix_time_to_range(unix_time_array, feature_range=(-1, 1)).reshape(-1,1)
    
    ###########################################
    ###########################################
    
    
    
    return scaled_unix_time_array, normalized_x_y, filtered_wind_power

scaled_unix_time_array, normalized_x_y, filtered_wind_power = loading_wind()


# Example usage:
# start_time = unix_time_array[0]
# end_time = unix_time_array[-1]
# print(format_time_difference(start_time, end_time))

###############################################





def combine_data(normalized_x_y, scaled_unix_time_array, filtered_wind_power):
    """
    Combine normalized_x_y, scaled_unix_time_array, and flattened filtered_wind_power into a single array.

    Parameters:
    - normalized_x_y (np.array): The normalized x and y coordinates (shape: (232, 2)).
    - scaled_unix_time_array (np.array): The scaled Unix timestamps (shape: (8783, 1)).
    - filtered_wind_power (np.array): The filtered wind power data (shape: (232, 8783)).

    Returns:
    - np.array: Combined array with shape (232*8783, 3).
    """
    num_rows = normalized_x_y.shape[0]  # Number of rows (232)
    num_columns = scaled_unix_time_array.shape[0]  # Number of columns (8783)

    # Repeat normalized_x_y for each timestamp in scaled_unix_time_array
    repeated_x_y = np.repeat(normalized_x_y, num_columns, axis=0)

    # Repeat scaled_unix_time_array for each row in normalized_x_y
    repeated_unix_time = np.tile(scaled_unix_time_array, (num_rows, 1))

    # Flatten the filtered_wind_power while preserving the sequence
    flattened_wind_power = filtered_wind_power.flatten()

    # Combine all three arrays into one
    combined_array = np.column_stack((repeated_x_y, repeated_unix_time.flatten(), flattened_wind_power))

    return combined_array

# Combine the data
combined_array = combine_data(normalized_x_y, scaled_unix_time_array, filtered_wind_power)

# Check the shape of the combined array
print(combined_array.shape)

























