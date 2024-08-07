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

def extract_wind_speed_for_germany(nc_file, resolution=0.05):
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
        pole_lat = nc.variables['rotated_pole'].grid_north_pole_latitude
        pole_lon = nc.variables['rotated_pole'].grid_north_pole_longitude

        rlat = nc.variables['rlat'][:]
        rlon = nc.variables['rlon'][:]

        rlon_mesh, rlat_mesh = np.meshgrid(rlon, rlat)
        points = np.column_stack((rlat_mesh.ravel(), rlon_mesh.ravel()))

        tree = cKDTree(points)

        rot_target_lats, rot_target_lons = rotate_coordinates(target_lats, target_lons, pole_lat, pole_lon)
        distances, indices = tree.query(np.column_stack((rot_target_lats, rot_target_lons)))

        wind_speed = nc.variables['sfcWind'][:]
        extracted_wind_speed = wind_speed[:, indices // rlon.size, indices % rlon.size]

        if np.all(extracted_wind_speed == extracted_wind_speed[0, 0]):
            raise ValueError("Extracted wind speed data is identical for all points. Check extraction process.")

        return extracted_wind_speed, lats, lons

def load_real_wind_csv(csv_file_path):
    df = pd.read_csv(csv_file_path, low_memory=False)
    df.dropna(axis=0, how='any', inplace=True)
    wind_data = df.to_numpy()
    x_y = wind_data[:, 1:3].astype(float)
    mask = np.any(wind_data[:, 5:] != 0, axis=1)
    filtered_x_y = x_y[mask]
    return filtered_x_y

def interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points):
    grid_lats = np.sort(grid_lats)
    grid_lons = np.sort(grid_lons)
    interpolated_wind_speeds = np.zeros((wind_speeds.shape[0], target_points.shape[0]))

    for t in range(wind_speeds.shape[0]):
        # Reshape wind_speeds[t] to a 2D array
        wind_speed_t = wind_speeds[t].reshape(len(grid_lats), len(grid_lons))
        
        interpolator = RegularGridInterpolator((grid_lats, grid_lons), wind_speed_t, method='linear', bounds_error=False, fill_value=np.nan)
        interpolated = interpolator(target_points)
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            nearest_interpolator = RegularGridInterpolator((grid_lats, grid_lons), wind_speed_t, method='nearest', bounds_error=False, fill_value=np.nan)
            interpolated[nan_mask] = nearest_interpolator(target_points[nan_mask])
        interpolated_wind_speeds[t] = interpolated

    nan_count = np.isnan(interpolated_wind_speeds).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values remain after interpolation.")
    
    return interpolated_wind_speeds

# Example usage
nc_file_path = 'nc_files/Klima_Daten_10m_3h_2020_RCP26.nc'
csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'

wind_speeds, grid_lats, grid_lons = extract_wind_speed_for_germany(nc_file_path)

print(f"Shape of extracted wind speed: {wind_speeds.shape}")
print(f"Sample of extracted wind speed (first 5 time steps, first 5 locations):")
print(wind_speeds[:5, :5])

target_points = load_real_wind_csv(csv_file_path)
interpolated_wind_speeds = interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points)

print(f"Shape of interpolated wind speeds: {interpolated_wind_speeds.shape}")
print(f"Sample of interpolated wind speeds (first 5 time steps, first 5 locations):")
print(interpolated_wind_speeds[:5, :5])

print(f"Number of NaN values: {np.isnan(interpolated_wind_speeds).sum()}")
print(f"Number of infinite values: {np.isinf(interpolated_wind_speeds).sum()}")

location_variation = np.any(np.diff(interpolated_wind_speeds, axis=1) != 0, axis=0)
time_variation = np.any(np.diff(interpolated_wind_speeds, axis=0) != 0, axis=1)

print(f"Number of locations with variation: {location_variation.sum()} out of {interpolated_wind_speeds.shape[1]}")
print(f"Number of time steps with variation: {time_variation.sum()} out of {interpolated_wind_speeds.shape[0]}")

location_stats = {
    'min': np.min(interpolated_wind_speeds, axis=0),
    'max': np.max(interpolated_wind_speeds, axis=0),
    'mean': np.mean(interpolated_wind_speeds, axis=0),
    'std': np.std(interpolated_wind_speeds, axis=0)
}

print("\nWind speed statistics across locations:")
for stat, values in location_stats.items():
    print(f"{stat.capitalize()}: min = {values.min():.2f}, max = {values.max():.2f}")

min_loc = np.unravel_index(np.argmin(location_stats['min']), location_stats['min'].shape)
max_loc = np.unravel_index(np.argmax(location_stats['max']), location_stats['max'].shape)

print(f"\nLocation with lowest minimum wind speed: Lat {target_points[min_loc[0], 1]:.2f}, Lon {target_points[min_loc[0], 0]:.2f}")
print(f"Location with highest maximum wind speed: Lat {target_points[max_loc[0], 1]:.2f}, Lon {target_points[max_loc[0], 0]:.2f}")

min_wind_index = np.unravel_index(np.argmin(interpolated_wind_speeds, axis=None), interpolated_wind_speeds.shape)
max_wind_index = np.unravel_index(np.argmax(interpolated_wind_speeds, axis=None), interpolated_wind_speeds.shape)

print(f"\nShape of interpolated wind speeds: {interpolated_wind_speeds.shape}")

print(f"\nMinimum wind speed of {interpolated_wind_speeds[min_wind_index]:.2f} m/s occurred at:")
print(f"Time step: {min_wind_index[0]}, Location: Lat {target_points[min_wind_index[1], 1]:.2f}, Lon {target_points[min_wind_index[1], 0]:.2f}")

print(f"\nMaximum wind speed of {interpolated_wind_speeds[max_wind_index]:.2f} m/s occurred at:")
print(f"Time step: {max_wind_index[0]}, Location: Lat {target_points[max_wind_index[1], 1]:.2f}, Lon {target_points[max_wind_index[1], 0]:.2f}")

avg_wind_speeds = np.mean(interpolated_wind_speeds, axis=0)
min_avg_loc = np.argmin(avg_wind_speeds)
max_avg_loc = np.argmax(avg_wind_speeds)

print(f"\nLocation with lowest average wind speed ({avg_wind_speeds[min_avg_loc]:.2f} m/s):")
print(f"Lat {target_points[min_avg_loc, 1]:.2f}, Lon {target_points[min_avg_loc, 0]:.2f}")

print(f"\nLocation with highest average wind speed ({avg_wind_speeds[max_avg_loc]:.2f} m/s):")
print(f"Lat {target_points[max_avg_loc, 1]:.2f}, Lon {target_points[max_avg_loc, 0]:.2f}")

std_wind_speeds = np.std(interpolated_wind_speeds, axis=0)
min_std_loc = np.argmin(std_wind_speeds)
max_std_loc = np.argmax(std_wind_speeds)

print(f"\nLocation with lowest wind speed variability (std dev: {std_wind_speeds[min_std_loc]:.2f} m/s):")
print(f"Lat {target_points[min_std_loc, 1]:.2f}, Lon {target_points[min_std_loc, 0]:.2f}")

print(f"\nLocation with highest wind speed variability (std dev: {std_wind_speeds[max_std_loc]:.2f} m/s):")
print(f"Lat {target_points[max_std_loc, 1]:.2f}, Lon {target_points[max_std_loc, 0]:.2f}")



#############################################################
#############################################################

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def debug_wind_speed_data(interpolated_wind_speeds, target_points):
    """Debug the wind speed data to understand why all points have the same values."""
    print(f"Shape of interpolated_wind_speeds: {interpolated_wind_speeds.shape}")
    print(f"Shape of target_points: {target_points.shape}")
    
    # Ensure target_points is a numeric array
    if target_points.dtype != 'float64':
        target_points = target_points.astype('float64')

    # Check if all values are the same for each time step
    same_values_per_timestep = np.all(interpolated_wind_speeds == interpolated_wind_speeds[:, [0]], axis=1)
    print(f"Number of time steps where all points have the same value: {np.sum(same_values_per_timestep)} out of {interpolated_wind_speeds.shape[0]}")
    
    # Check if there's any variation across time steps
    variation_across_time = np.any(np.diff(interpolated_wind_speeds, axis=0) != 0)
    print(f"Is there variation across time steps? {'Yes' if variation_across_time else 'No'}")
    
    # Print a few rows of data to inspect
    print("\nSample of interpolated wind speeds (first 5 time steps, first 5 locations):")
    print(interpolated_wind_speeds[:5, :5])
    
    # Check for NaN or infinite values
    print(f"\nNumber of NaN values: {np.isnan(interpolated_wind_speeds).sum()}")
    print(f"Number of infinite values: {np.isinf(interpolated_wind_speeds).sum()}")
    
    # Print unique locations
    unique_locations = np.unique(target_points, axis=0)
    print(f"\nNumber of unique locations: {len(unique_locations)}")
    print("Sample of unique locations:")
    print(unique_locations[:5])


#############################################################
#############################################################


def comprehensive_wind_speed_analysis(interpolated_wind_speeds, target_points):
    """Perform comprehensive statistical analysis on wind speed data."""
    debug_wind_speed_data(interpolated_wind_speeds, target_points)
    
    # Overall statistics
    min_speed = np.min(interpolated_wind_speeds)
    max_speed = np.max(interpolated_wind_speeds)
    mean_speed = np.mean(interpolated_wind_speeds)
    median_speed = np.median(interpolated_wind_speeds)
    std_speed = np.std(interpolated_wind_speeds)
    
    print("\nOverall Wind Speed Statistics:")
    print(f"Minimum: {min_speed:.2f} m/s")
    print(f"Maximum: {max_speed:.2f} m/s")
    print(f"Mean: {mean_speed:.2f} m/s")
    print(f"Median: {median_speed:.2f} m/s")
    print(f"Standard Deviation: {std_speed:.2f} m/s")
    
    # Find locations with min and max wind speeds
    if len(np.unique(target_points, axis=0)) > 1:
        min_index = np.unravel_index(np.argmin(interpolated_wind_speeds), interpolated_wind_speeds.shape)
        max_index = np.unravel_index(np.argmax(interpolated_wind_speeds), interpolated_wind_speeds.shape)
        
        min_location = target_points[min_index[1]]
        max_location = target_points[max_index[1]]
        
        print(f"\nMinimum wind speed of {min_speed:.2f} m/s occurred at:")
        print(f"Time step: {min_index[0]}, Location: Lat {min_location[0]:.2f}, Lon {min_location[1]:.2f}")
        
        print(f"\nMaximum wind speed of {max_speed:.2f} m/s occurred at:")
        print(f"Time step: {max_index[0]}, Location: Lat {max_location[0]:.2f}, Lon {max_location[1]:.2f}")
        
        # Calculate average wind speed for each location
        avg_speeds = np.mean(interpolated_wind_speeds, axis=0)
        min_avg_index = np.argmin(avg_speeds)
        max_avg_index = np.argmax(avg_speeds)
        
        min_avg_location = target_points[min_avg_index]
        max_avg_location = target_points[max_avg_index]
        
        print(f"\nLocation with lowest average wind speed ({avg_speeds[min_avg_index]:.2f} m/s):")
        print(f"Lat {min_avg_location[0]:.2f}, Lon {min_avg_location[1]:.2f}")
        
        print(f"\nLocation with highest average wind speed ({avg_speeds[max_avg_index]:.2f} m/s):")
        print(f"Lat {max_avg_location[0]:.2f}, Lon {max_avg_location[1]:.2f}")
        
        # Calculate variability (standard deviation) for each location
        std_speeds = np.std(interpolated_wind_speeds, axis=0)
        min_std_index = np.argmin(std_speeds)
        max_std_index = np.argmax(std_speeds)
        
        min_std_location = target_points[min_std_index]
        max_std_location = target_points[max_std_index]
        
        print(f"\nLocation with lowest wind speed variability (std dev: {std_speeds[min_std_index]:.2f} m/s):")
        print(f"Lat {min_std_location[0]:.2f}, Lon {min_std_location[1]:.2f}")
        
        print(f"\nLocation with highest wind speed variability (std dev: {std_speeds[max_std_index]:.2f} m/s):")
        print(f"Lat {max_std_location[0]:.2f}, Lon {max_std_location[1]:.2f}")
    else:
        print("\nWarning: All target points are identical. Cannot perform location-specific analysis.")
    
    return min_speed, max_speed

# Example usage
comprehensive_wind_speed_analysis(interpolated_wind_speeds, target_points)


#####################################################
#####################################################



def visualize_wind_speed_map_4(grid_lats, grid_lons, target_points, interpolated_wind_speeds, time_step=0):
    min_speed, max_speed = comprehensive_wind_speed_analysis(interpolated_wind_speeds, target_points)
    
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

    # Plot grid points (optional, style them as needed)
    ax.scatter(lons_flat, lats_flat, color='black', s=5, alpha=0.3, transform=ccrs.PlateCarree(), label='Grid Points')

    # Scatter plot for interpolated wind speeds with new colormap
    sc = ax.scatter(target_points[:, 1], target_points[:, 0], 
                    c=interpolated_wind_speeds[time_step], cmap='plasma',  
                    s=50, alpha=0.7, transform=ccrs.PlateCarree(), 
                    label='Interpolated Points', vmin=min_speed, vmax=max_speed)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Wind Speed (m/s)', fontsize=12)

    # Customize axis labels and title
    ax.set_title(f'Wind Speed Map - Time Step: {time_step}\n'
                 f'Speed Range: {min_speed:.2f} - {max_speed:.2f} m/s',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)

    # Customize tick labels font size
    ax.tick_params(labelsize=12)

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def visualize_wind_speed_map_5(grid_lats, grid_lons, target_points, interpolated_wind_speeds, time_step=0):
    # Get the wind speeds for the specified time step
    wind_speeds = interpolated_wind_speeds[time_step]
    
    # Determine the min and max values for normalization
    min_speed = np.min(interpolated_wind_speeds)
    max_speed = np.max(interpolated_wind_speeds)
    
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

    # Scatter plot for interpolated wind speeds with colormap
    sc = ax.scatter(target_points[:, 1], target_points[:, 0], 
                    c=wind_speeds, cmap='plasma',  
                    s=50, alpha=0.7, transform=ccrs.PlateCarree(), 
                    vmin=min_speed, vmax=max_speed, label='Interpolated Points')

    # Add text annotations for wind speeds
    for lon, lat, wind_speed in zip(target_points[:, 1], target_points[:, 0], wind_speeds):
        ax.text(lon, lat, f'{wind_speed:.1f} m/s', fontsize=8, ha='center', transform=ccrs.PlateCarree(),
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5))

    # Customize axis labels and title
    ax.set_title(f'Wind Speed Map - Time Step: {time_step}\n'
                 f'Speed Range: {min_speed:.2f} - {max_speed:.2f} m/s',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)

    # Customize tick labels font size
    ax.tick_params(labelsize=12)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def visualize_wind_speed_map(grid_lats, grid_lons, target_points, interpolated_wind_speeds, time_step=0):
    # Get the wind speeds for the specified time step
    wind_speeds = interpolated_wind_speeds[time_step]
    
    # Calculate mean wind speed for each location across all time steps
    mean_wind_speeds = np.mean(interpolated_wind_speeds, axis=0)
    
    # Determine the min and max values for normalization
    min_speed = np.min(mean_wind_speeds)
    max_speed = np.max(mean_wind_speeds)
    
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

    # Scatter plot for mean wind speeds with colormap
    sc = ax.scatter(target_points[:, 1], target_points[:, 0], 
                    c=mean_wind_speeds, cmap='plasma',  
                    s=50, alpha=0.7, transform=ccrs.PlateCarree(), 
                    vmin=min_speed, vmax=max_speed, label='Mean Wind Speeds')

    # Add text annotations for mean wind speeds
    for lon, lat, mean_speed in zip(target_points[:, 1], target_points[:, 0], mean_wind_speeds):
        ax.text(lon, lat, f'{mean_speed:.1f} m/s', fontsize=8, ha='center', transform=ccrs.PlateCarree(),
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

    # Add mean wind speed annotation
    ax.text(0.05, 0.95, f'Overall Mean Wind Speed: {np.mean(mean_wind_speeds):.2f} m/s', 
            fontsize=12, fontweight='bold', color='black',
            transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Customize axis labels and title
    ax.set_title(f'Wind Speed Map - Time Step: {time_step}\n'
                 f'Mean Wind Speed Range: {min_speed:.2f} - {max_speed:.2f} m/s',
                 fontsize=16, fontweight='bold')
    
    # Move the legend to a more visible location
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    # Add a colorbar to indicate wind speed values
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Mean Wind Speed (m/s)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # Customize gridlines
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)

    # Customize tick labels font size
    ax.tick_params(labelsize=12)

    plt.show()



import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def visualize_wind_speed_map(grid_lats, grid_lons, target_points, interpolated_wind_speeds, time_step=0):
    """Visualize mean wind speed data on a map using color representation."""
    # Get the wind speeds for the specified time step
    wind_speeds = interpolated_wind_speeds[time_step]
    
    # Calculate mean wind speed for each location across all time steps
    mean_wind_speeds = np.mean(interpolated_wind_speeds, axis=0)
    
    # Determine the min and max values for normalization
    min_speed = np.min(mean_wind_speeds)
    max_speed = np.max(mean_wind_speeds)
    
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

    # Scatter plot for mean wind speeds with colormap
    sc = ax.scatter(target_points[:, 1], target_points[:, 0], 
                    c=mean_wind_speeds, cmap='plasma',  
                    s=50, alpha=0.7, transform=ccrs.PlateCarree(), 
                    vmin=min_speed, vmax=max_speed, label='Mean Wind Speed')

    # Add mean wind speed annotation
    ax.text(0.05, 0.95, f'Overall Mean Wind Speed: {np.mean(mean_wind_speeds):.2f} m/s', 
            fontsize=12, fontweight='bold', color='black',
            transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Customize axis labels and title
    ax.set_title(f'Wind Speed Map - Time Step: {time_step}\n'
                 f'Mean Wind Speed Range: {min_speed:.2f} - {max_speed:.2f} m/s',
                 fontsize=16, fontweight='bold')
    
    # Move the legend to a more visible location
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    # Add a colorbar to indicate wind speed values
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Mean Wind Speed (m/s)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # Customize gridlines
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)

    # Customize tick labels font size
    ax.tick_params(labelsize=12)

    plt.show()








##################################
##################################


def plot_wind_speed_histogram(interpolated_wind_speeds):
    """Plot a histogram of wind speeds."""
    plt.figure(figsize=(10, 6))
    plt.hist(interpolated_wind_speeds.flatten(), bins=50, edgecolor='black')
    plt.title('Distribution of Wind Speeds')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')
    plt.show()

def plot_wind_speed_time_series(interpolated_wind_speeds):
    """Plot a time series of mean, min, and max wind speeds."""
    mean_speeds = np.mean(interpolated_wind_speeds, axis=1)
    min_speeds = np.min(interpolated_wind_speeds, axis=1)
    max_speeds = np.max(interpolated_wind_speeds, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(mean_speeds, label='Mean')
    plt.plot(min_speeds, label='Min')
    plt.plot(max_speeds, label='Max')
    plt.title('Wind Speed Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.show()

# Print coordinate ranges for verification
print(f"Grid points range: Lat ({grid_lats.min():.2f}, {grid_lats.max():.2f}), Lon ({grid_lons.min():.2f}, {grid_lons.max():.2f})")
print(f"Target points range: Lat ({target_points[:, 0].min():.2f}, {target_points[:, 0].max():.2f}), Lon ({target_points[:, 1].min():.2f}, {target_points[:, 1].max():.2f})")

# Example usage
visualize_wind_speed_map(grid_lats, grid_lons, target_points, interpolated_wind_speeds)
plot_wind_speed_histogram(interpolated_wind_speeds)
plot_wind_speed_time_series(interpolated_wind_speeds)




























