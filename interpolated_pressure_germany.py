#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:30:12 2024

@author: forootan
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

def rotate_coordinates(lat, lon, pole_lat, pole_lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    pole_lat, pole_lon = np.deg2rad(pole_lat), np.deg2rad(pole_lon)

    lon = lon - pole_lon

    rot_lat = np.arcsin(np.sin(lat) * np.sin(pole_lat) + 
                        np.cos(lat) * np.cos(pole_lat) * np.cos(lon))
    
    rot_lon = np.arctan2(np.cos(lat) * np.sin(lon),
                         np.sin(lat) * np.cos(pole_lat) - 
                         np.cos(lat) * np.sin(pole_lat) * np.cos(lon))

    return np.rad2deg(rot_lat), np.rad2deg(rot_lon)


def extract_pressure_for_germany(nc_file, resolution=0.05):
    germany_bbox = {
        'min_lat': 47.3,
        'max_lat': 55.1,
        'min_lon': 5.9,
        'max_lon': 15.0
    }

    lats = np.arange(germany_bbox['min_lat'], germany_bbox['max_lat'], resolution)
    lons = np.arange(germany_bbox['min_lon'], germany_bbox['max_lon'], resolution)
    target_lats, target_lons = np.meshgrid(lats, lons)
    target_lats, target_lons = target_lats.flatten(), target_lons.flatten()

    with Dataset(nc_file, 'r') as nc:
        rotated_lat_lon = nc.variables['rotated_latitude_longitude']
        
        # Extract projection parameters
        pole_lat = rotated_lat_lon.grid_north_pole_latitude
        pole_lon = rotated_lat_lon.grid_north_pole_longitude
        
        print(f"Pole Latitude: {pole_lat}")
        print(f"Pole Longitude: {pole_lon}")

        rlat = nc.variables['rlat'][:]
        rlon = nc.variables['rlon'][:]

        rlon_mesh, rlat_mesh = np.meshgrid(rlon, rlat)
        points = np.column_stack((rlat_mesh.ravel(), rlon_mesh.ravel()))

        tree = cKDTree(points)

        rot_target_lats, rot_target_lons = rotate_coordinates(target_lats, target_lons, pole_lat, pole_lon)
        distances, indices = tree.query(np.column_stack((rot_target_lats, rot_target_lons)))

        pressure = nc.variables['psl'][:]  # Assuming 'psl' is the pressure variable name
        extracted_pressure = pressure[:, indices // rlon.size, indices % rlon.size]

        if np.all(extracted_pressure == extracted_pressure[0, 0]):
            raise ValueError("Extracted pressure data is identical for all points. Check extraction process.")

        return extracted_pressure, lats, lons


def load_real_wind_csv(csv_file_path):
    df = pd.read_csv(csv_file_path, low_memory=False)
    df.dropna(axis=0, how='any', inplace=True)
    wind_data = df.to_numpy()
    x_y = wind_data[:, 1:3].astype(float)
    mask = np.any(wind_data[:, 5:] != 0, axis=1)
    filtered_x_y = x_y[mask]
    return filtered_x_y

def interpolate_data(data, grid_lats, grid_lons, target_points):
    grid_lats = np.sort(grid_lats)
    grid_lons = np.sort(grid_lons)
    interpolated_data = np.zeros((data.shape[0], target_points.shape[0]))

    for t in range(data.shape[0]):
        # Reshape data[t] to a 2D array
        data_t = data[t].reshape(len(grid_lats), len(grid_lons))
        
        interpolator = RegularGridInterpolator((grid_lats, grid_lons), data_t, method='linear', bounds_error=False, fill_value=np.nan)
        interpolated = interpolator(target_points)
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            nearest_interpolator = RegularGridInterpolator((grid_lats, grid_lons), data_t, method='nearest', bounds_error=False, fill_value=np.nan)
            interpolated[nan_mask] = nearest_interpolator(target_points[nan_mask])
        interpolated_data[t] = interpolated

    nan_count = np.isnan(interpolated_data).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values remain after interpolation.")
    
    return interpolated_data

# Example usage
nc_file_path = 'nc_files/dataset-projections-cordex-domains-single-levels-69ac4dd9-7e75-46a0-8eef-7be736876191/psl_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r3i1p1_GERICS-REMO2015_v1_3hr_202001010100-202012312200.nc'
csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'

# Extract pressure data
pressure_data, grid_lats, grid_lons = extract_pressure_for_germany(nc_file_path)

print(f"Shape of extracted pressure data: {pressure_data.shape}")
print(f"Sample of extracted pressure data (first 5 time steps, first 5 locations):")
print(pressure_data[:5, :5])

# Load real wind CSV
target_points = load_real_wind_csv(csv_file_path)

# Interpolate pressure data
interpolated_pressure = interpolate_data(pressure_data, grid_lats, grid_lons, target_points)

print(f"Shape of interpolated pressure data: {interpolated_pressure.shape}")
print(f"Sample of interpolated pressure data (first 5 time steps, first 5 locations):")
print(interpolated_pressure[:5, :5])

print(f"Number of NaN values: {np.isnan(interpolated_pressure).sum()}")
print(f"Number of infinite values: {np.isinf(interpolated_pressure).sum()}")

# Analyze pressure data
location_variation = np.any(np.diff(interpolated_pressure, axis=1) != 0, axis=0)
time_variation = np.any(np.diff(interpolated_pressure, axis=0) != 0, axis=1)

print(f"Number of locations with variation: {location_variation.sum()} out of {interpolated_pressure.shape[1]}")
print(f"Number of time steps with variation: {time_variation.sum()} out of {interpolated_pressure.shape[0]}")

location_stats = {
    'min': np.min(interpolated_pressure, axis=0),
    'max': np.max(interpolated_pressure, axis=0),
    'mean': np.mean(interpolated_pressure, axis=0),
    'std': np.std(interpolated_pressure, axis=0)
}

print("\nPressure statistics across locations:")
for stat, values in location_stats.items():
    print(f"{stat.capitalize()}: min = {values.min():.2f}, max = {values.max():.2f}")

min_loc = np.unravel_index(np.argmin(location_stats['min']), location_stats['min'].shape)
max_loc = np.unravel_index(np.argmax(location_stats['max']), location_stats['max'].shape)

print(f"\nLocation with lowest minimum pressure: Lat {target_points[min_loc[0], 1]:.2f}, Lon {target_points[min_loc[0], 0]:.2f}")
print(f"Location with highest maximum pressure: Lat {target_points[max_loc[0], 1]:.2f}, Lon {target_points[max_loc[0], 0]:.2f}")

min_pressure_index = np.unravel_index(np.argmin(interpolated_pressure, axis=None), interpolated_pressure.shape)
max_pressure_index = np.unravel_index(np.argmax(interpolated_pressure, axis=None), interpolated_pressure.shape)

print(f"\nShape of interpolated pressure data: {interpolated_pressure.shape}")

print(f"\nMinimum pressure of {interpolated_pressure[min_pressure_index]:.2f} hPa occurred at:")
print(f"Time step: {min_pressure_index[0]}, Location: Lat {target_points[min_pressure_index[1], 1]:.2f}, Lon {target_points[min_pressure_index[1], 0]:.2f}")

print(f"\nMaximum pressure of {interpolated_pressure[max_pressure_index]:.2f} hPa occurred at:")
print(f"Time step: {max_pressure_index[0]}, Location: Lat {target_points[max_pressure_index[1], 1]:.2f}, Lon {target_points[max_pressure_index[1], 0]:.2f}")

avg_pressure = np.mean(interpolated_pressure, axis=0)
min_avg_loc = np.argmin(avg_pressure)
max_avg_loc = np.argmax(avg_pressure)

print(f"\nLocation with lowest average pressure ({avg_pressure[min_avg_loc]:.2f} hPa):")
print(f"Lat {target_points[min_avg_loc, 1]:.2f}, Lon {target_points[min_avg_loc, 0]:.2f}")

print(f"\nLocation with highest average pressure ({avg_pressure[max_avg_loc]:.2f} hPa):")
print(f"Lat {target_points[max_avg_loc, 1]:.2f}, Lon {target_points[max_avg_loc, 0]:.2f}")

std_pressure = np.std(interpolated_pressure, axis=0)
min_std_loc = np.argmin(std_pressure)
max_std_loc = np.argmax(std_pressure)

print(f"\nLocation with lowest pressure variability (std dev: {std_pressure[min_std_loc]:.2f} hPa):")
print(f"Lat {target_points[min_std_loc, 1]:.2f}, Lon {target_points[min_std_loc, 0]:.2f}")

print(f"\nLocation with highest pressure variability (std dev: {std_pressure[max_std_loc]:.2f} hPa):")
print(f"Lat {target_points[max_std_loc, 1]:.2f}, Lon {target_points[max_std_loc, 0]:.2f}")





##############################################3333
##################################################



import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def debug_pressure_data(interpolated_pressures, target_points):
    """Debug the pressure data to understand why all points have the same values."""
    print(f"Shape of interpolated_pressures: {interpolated_pressures.shape}")
    print(f"Shape of target_points: {target_points.shape}")
    
    # Ensure target_points is a numeric array
    if target_points.dtype != 'float64':
        target_points = target_points.astype('float64')

    # Check if all values are the same for each time step
    same_values_per_timestep = np.all(interpolated_pressures == interpolated_pressures[:, [0]], axis=1)
    print(f"Number of time steps where all points have the same value: {np.sum(same_values_per_timestep)} out of {interpolated_pressures.shape[0]}")
    
    # Check if there's any variation across time steps
    variation_across_time = np.any(np.diff(interpolated_pressures, axis=0) != 0)
    print(f"Is there variation across time steps? {'Yes' if variation_across_time else 'No'}")
    
    # Print a few rows of data to inspect
    print("\nSample of interpolated pressures (first 5 time steps, first 5 locations):")
    print(interpolated_pressures[:5, :5])
    
    # Check for NaN or infinite values
    print(f"\nNumber of NaN values: {np.isnan(interpolated_pressures).sum()}")
    print(f"Number of infinite values: {np.isinf(interpolated_pressures).sum()}")
    
    # Print unique locations
    unique_locations = np.unique(target_points, axis=0)
    print(f"\nNumber of unique locations: {len(unique_locations)}")
    print("Sample of unique locations:")
    print(unique_locations[:5])


def comprehensive_pressure_analysis(interpolated_pressures, target_points):
    """Perform comprehensive statistical analysis on pressure data."""
    debug_pressure_data(interpolated_pressures, target_points)
    
    # Overall statistics
    min_pressure = np.min(interpolated_pressures)
    max_pressure = np.max(interpolated_pressures)
    mean_pressure = np.mean(interpolated_pressures)
    median_pressure = np.median(interpolated_pressures)
    std_pressure = np.std(interpolated_pressures)
    
    print("\nOverall Pressure Statistics:")
    print(f"Minimum: {min_pressure:.2f} hPa")
    print(f"Maximum: {max_pressure:.2f} hPa")
    print(f"Mean: {mean_pressure:.2f} hPa")
    print(f"Median: {median_pressure:.2f} hPa")
    print(f"Standard Deviation: {std_pressure:.2f} hPa")
    
    # Find locations with min and max pressures
    if len(np.unique(target_points, axis=0)) > 1:
        min_index = np.unravel_index(np.argmin(interpolated_pressures), interpolated_pressures.shape)
        max_index = np.unravel_index(np.argmax(interpolated_pressures), interpolated_pressures.shape)
        
        min_location = target_points[min_index[1]]
        max_location = target_points[max_index[1]]
        
        print(f"\nMinimum pressure of {min_pressure:.2f} hPa occurred at:")
        print(f"Time step: {min_index[0]}, Location: Lat {min_location[0]:.2f}, Lon {min_location[1]:.2f}")
        
        print(f"\nMaximum pressure of {max_pressure:.2f} hPa occurred at:")
        print(f"Time step: {max_index[0]}, Location: Lat {max_location[0]:.2f}, Lon {max_location[1]:.2f}")
        
        # Calculate average pressure for each location
        avg_pressures = np.mean(interpolated_pressures, axis=0)
        min_avg_index = np.argmin(avg_pressures)
        max_avg_index = np.argmax(avg_pressures)
        
        min_avg_location = target_points[min_avg_index]
        max_avg_location = target_points[max_avg_index]
        
        print(f"\nLocation with lowest average pressure ({avg_pressures[min_avg_index]:.2f} hPa):")
        print(f"Lat {min_avg_location[0]:.2f}, Lon {min_avg_location[1]:.2f}")
        
        print(f"\nLocation with highest average pressure ({avg_pressures[max_avg_index]:.2f} hPa):")
        print(f"Lat {max_avg_location[0]:.2f}, Lon {max_avg_location[1]:.2f}")
        
        # Calculate variability (standard deviation) for each location
        std_pressures = np.std(interpolated_pressures, axis=0)
        min_std_index = np.argmin(std_pressures)
        max_std_index = np.argmax(std_pressures)
        
        min_std_location = target_points[min_std_index]
        max_std_location = target_points[max_std_index]
        
        print(f"\nLocation with lowest pressure variability (std dev: {std_pressures[min_std_index]:.2f} hPa):")
        print(f"Lat {min_std_location[0]:.2f}, Lon {min_std_location[1]:.2f}")
        
        print(f"\nLocation with highest pressure variability (std dev: {std_pressures[max_std_index]:.2f} hPa):")
        print(f"Lat {max_std_location[0]:.2f}, Lon {max_std_location[1]:.2f}")
    else:
        print("\nWarning: All target points are identical. Cannot perform location-specific analysis.")
    
    return min_pressure, max_pressure


def visualize_pressure_map(grid_lats, grid_lons, target_points, interpolated_pressures, time_step=0):
    """Visualize pressure data on a map for a specific time step."""
    pressures = interpolated_pressures[time_step]
    
    # Determine the min and max values for normalization
    min_pressure = np.min(interpolated_pressures)
    max_pressure = np.max(interpolated_pressures)
    
    # Ensure grid_lats and grid_lons are 2D and have the same shape
    if grid_lats.ndim == 1 and grid_lons.ndim == 1:
        grid_lats, grid_lons = np.meshgrid(grid_lats, grid_lons)
    
    if grid_lats.shape != grid_lons.shape:
        raise ValueError(f"Shape mismatch: grid_lats shape = {grid_lats.shape}, grid_lons shape = {grid_lons.shape}")

    lons_flat = grid_lons.flatten()
    lats_flat = grid_lats.flatten()

    if lons_flat.size != lats_flat.size:
        raise ValueError(f"Flattened arrays size mismatch: lons_flat size = {lons_flat.size}, lats_flat size = {lats_flat.size}")

    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    ax.set_facecolor('lightgrey')

    # Plot grid points
    ax.scatter(lons_flat, lats_flat, color='black', s=5, alpha=0.3, transform=ccrs.PlateCarree(), label='Grid Points')

    # Scatter plot for interpolated pressures with colormap
    sc = ax.scatter(target_points[:, 1], target_points[:, 0], 
                    c=pressures, cmap='coolwarm',  
                    s=50, alpha=0.7, transform=ccrs.PlateCarree(), 
                    vmin=min_pressure, vmax=max_pressure, label='Interpolated Points')

    # Add text annotations for pressures
    for lon, lat, pressure in zip(target_points[:, 1], target_points[:, 0], pressures):
        ax.text(lon, lat, f'{pressure:.1f} hPa', fontsize=8, ha='center', transform=ccrs.PlateCarree(),
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5))

    # Customize axis labels and title
    ax.set_title(f'Pressure Map - Time Step: {time_step}\n'
                 f'Pressure Range: {min_pressure:.2f} - {max_pressure:.2f} hPa',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)

    # Customize tick labels font size
    ax.tick_params(labelsize=12)

    plt.show()


def visualize_mean_pressure_map(grid_lats, grid_lons, target_points, interpolated_pressures, time_step=0):
    """Visualize mean pressure data on a map."""
    # Calculate mean pressure for each location across all time steps
    mean_pressures = np.mean(interpolated_pressures, axis=0)
    
    # Determine the min and max values for normalization
    min_pressure = np.min(mean_pressures)
    max_pressure = np.max(mean_pressures)
    
    # Ensure grid_lats and grid_lons are 2D and have the same shape
    if grid_lats.ndim == 1 and grid_lons.ndim == 1:
        grid_lats, grid_lons = np.meshgrid(grid_lats, grid_lons)
    
    if grid_lats.shape != grid_lons.shape:
        raise ValueError(f"Shape mismatch: grid_lats shape = {grid_lats.shape}, grid_lons shape = {grid_lons.shape}")

    lons_flat = grid_lons.flatten()
    lats_flat = grid_lats.flatten()

    if lons_flat.size != lats_flat.size:
        raise ValueError(f"Flattened arrays size mismatch: lons_flat size = {lons_flat.size}, lats_flat size = {lats_flat.size}")

    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    ax.set_facecolor('lightgrey')

    # Plot grid points
    ax.scatter(lons_flat, lats_flat, color='black', s=5, alpha=0.3, transform=ccrs.PlateCarree(), label='Grid Points')

    # Scatter plot for mean pressures with colormap
    sc = ax.scatter(target_points[:, 1], target_points[:, 0], 
                    c=mean_pressures, cmap='coolwarm',  
                    s=50, alpha=0.7, transform=ccrs.PlateCarree(), 
                    vmin=min_pressure, vmax=max_pressure, label='Mean Pressure')

    # Add text annotations for mean pressures
    for lon, lat, mean_pressure in zip(target_points[:, 1], target_points[:, 0], mean_pressures):
        ax.text(lon, lat, f'{mean_pressure:.1f} hPa', fontsize=8, ha='center', transform=ccrs.PlateCarree(),
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

    # Add mean pressure annotation
    ax.text(0.05, 0.95, f'Overall Mean Pressure: {np.mean(mean_pressures):.2f} hPa', 
            fontsize=12, fontweight='bold', color='black',
            transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Customize axis labels and title
    ax.set_title(f'Pressure Map - Time Step: {time_step}\n'
                 f'Mean Pressure Range: {min_pressure:.2f} - {max_pressure:.2f} hPa',
                 fontsize=16, fontweight='bold')
    
    # Move the legend to a more visible location
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    # Add a colorbar to indicate pressure values
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Mean Pressure (hPa)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # Customize gridlines
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)

    # Customize tick labels font size
    ax.tick_params(labelsize=12)

    plt.show()



def visualize_mean_pressure_map(grid_lats, grid_lons, target_points, interpolated_pressures, time_step=0):
    """Visualize mean pressure data on a map."""
    # Calculate mean pressure for each location across all time steps
    mean_pressures = np.mean(interpolated_pressures, axis=0)
    
    # Determine the min and max values for normalization
    min_pressure = np.min(mean_pressures)
    max_pressure = np.max(mean_pressures)
    
    # Ensure grid_lats and grid_lons are 2D and have the same shape
    if grid_lats.ndim == 1 and grid_lons.ndim == 1:
        grid_lats, grid_lons = np.meshgrid(grid_lats, grid_lons)
    
    if grid_lats.shape != grid_lons.shape:
        raise ValueError(f"Shape mismatch: grid_lats shape = {grid_lats.shape}, grid_lons shape = {grid_lons.shape}")

    lons_flat = grid_lons.flatten()
    lats_flat = grid_lats.flatten()

    if lons_flat.size != lats_flat.size:
        raise ValueError(f"Flattened arrays size mismatch: lons_flat size = {lons_flat.size}, lats_flat size = {lats_flat.size}")

    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    ax.set_facecolor('lightgrey')

    # Plot grid points
    ax.scatter(lons_flat, lats_flat, color='black', s=5, alpha=0.3, transform=ccrs.PlateCarree(), label='Grid Points')

    # Scatter plot for mean pressures with colormap
    sc = ax.scatter(target_points[:, 1], target_points[:, 0], 
                    c=mean_pressures, cmap='coolwarm',  
                    s=50, alpha=0.7, transform=ccrs.PlateCarree(), 
                    vmin=min_pressure, vmax=max_pressure, label='Mean Pressure')

    # Add mean pressure annotation
    ax.text(0.05, 0.95, f'Overall Mean Pressure: {np.mean(mean_pressures):.2f} hPa', 
            fontsize=12, fontweight='bold', color='black',
            transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Customize axis labels and title
    ax.set_title(f'Pressure Map - Time Step: {time_step}\n'
                 f'Mean Pressure Range: {min_pressure:.2f} - {max_pressure:.2f} hPa',
                 fontsize=16, fontweight='bold')
    
    # Move the legend to a more visible location
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    # Add a colorbar to indicate pressure values
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Mean Pressure (hPa)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # Customize gridlines
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)

    # Customize tick labels font size
    ax.tick_params(labelsize=12)

    plt.show()





#####################################


#from Functions import plot_config_file

def plot_pressure_histogram(interpolated_pressures):
    """Plot a histogram of pressure values."""
    plt.figure(figsize=(10, 6))
    plt.hist(interpolated_pressures.flatten(), bins=50, edgecolor='black')
    plt.title('Distribution of Pressure')
    plt.xlabel('Pressure (hPa)')
    plt.ylabel('Frequency')
    plt.show()


def plot_pressure_time_series(interpolated_pressures):
    """Plot a time series of mean, min, and max pressures with natural colors and thicker lines."""
    mean_pressures = np.mean(interpolated_pressures, axis=1)
    min_pressures = np.min(interpolated_pressures, axis=1)
    max_pressures = np.max(interpolated_pressures, axis=1)
    
    plt.figure(figsize=(14, 7))
    
    # Plot with natural colors and thicker lines
    plt.plot(mean_pressures, linestyle='-', color='steelblue', linewidth=3.0, label='Mean')
    plt.plot(min_pressures, linestyle='--', color='darkgreen', linewidth=3.0, label='Min')
    plt.plot(max_pressures, linestyle=':', color='saddlebrown', linewidth=3.0, label='Max')
    
    # Adding title and labels with larger fonts
    plt.title('Pressure Over Time', fontsize=18, fontweight='bold')
    plt.xlabel('Time Step', fontsize=18)
    plt.ylabel('Pressure (hPa)', fontsize=18)
    
    # Adding grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adding legend with larger font
    plt.legend(fontsize=12)
    
    # Show the plot
    plt.tight_layout()
    plt.show()





# Example usage
min_pressure, max_pressure = comprehensive_pressure_analysis(interpolated_pressure, target_points)
visualize_pressure_map(grid_lats, grid_lons, target_points, interpolated_pressure, time_step=0)
visualize_mean_pressure_map(grid_lats, grid_lons, target_points, interpolated_pressure)
plot_pressure_histogram(interpolated_pressure)
plot_pressure_time_series(interpolated_pressure)
















