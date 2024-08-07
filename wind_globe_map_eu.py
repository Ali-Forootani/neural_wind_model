#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:51:29 2024

@author: forootan

The provided Python code appears to be for analyzing and visualizing surface wind speed data from a NetCDF file. Let's break down what the code does:

**1. Function Definition:**

* `extract_metadata_and_analyze_data(nc_file_path)`: This function takes a NetCDF file path as input and extracts metadata, variable attributes, data arrays, and other relevant information.

**2. Data Loading and Processing:**

* The script defines a file path to a NetCDF file containing surface wind data (`nc_files/Klima_Daten_10m_3h_2020_RCP26.nc`).
* It uses the `netCDF4` library to open the file, extract metadata, variable attributes, wind speed data (`sfcWind`), time data (`time`), latitude (`lat`), and longitude (`lon`) data.
* The `num2date` function is used to convert the time data from numeric format to datetime objects.

**3. Printing Information:**

* The script prints relevant metadata like source, project, experiment obtained from the file.
* It then iterates through the attributes of the `sfcWind` variable and prints them.
* If a `height` variable exists, its value, units, and attributes are printed.
* The shapes of `sfcWind`, `lat`, and `lon` data are printed to understand their dimensions.
* The script calculates the time range covered by the data (from dates[0] to dates[-1]).
* Finally, it prints the minimum and maximum values for latitude and longitude.

**4. Visualization:**

* The code utilizes the `cartopy` library to create a map plot showing the geographical coverage of the surface wind speed data for the first time step (`sfcWind[0]`).
* A colorbar is added to represent the wind speed values.
* Another plot shows the time series of surface wind speed for a single point (center of the grid) over the entire time range.
* Finally, the average wind speed across the whole time period is calculated and visualized on a map using another cartopy plot.

**5. Warnings:**

* The script generates warnings related to how the latitude and longitude coordinates are interpreted by `pcolormesh`. This might be due to non-monotonic coordinates. You can potentially ignore these warnings if the plots seem visually correct.

Overall, the code effectively analyzes and visualizes surface wind speed data from a NetCDF file, providing informative plots and summaries.


https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/data_access/
https://www.renewables.ninja/

"""


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import datetime

def extract_metadata_and_iranian_data(nc_file_path):
    # Open the NetCDF file
    nc_file = Dataset(nc_file_path, 'r')

    # Extract metadata
    metadata = {}
    for attr in nc_file.ncattrs():
        metadata[attr] = getattr(nc_file, attr)

    # Check sfcWind variable attributes
    sfcWind_var = nc_file.variables['sfcWind']
    sfcWind_attrs = {attr: getattr(sfcWind_var, attr) for attr in sfcWind_var.ncattrs()}

    # Check if there's a 'height' variable
    if 'height' in nc_file.variables:
        height_var = nc_file.variables['height']
        height_value = height_var[:]
        height_attrs = {attr: getattr(height_var, attr) for attr in height_var.ncattrs()}
    else:
        height_value = None
        height_attrs = None

    # Get latitude and longitude data
    lat = nc_file.variables['lat'][:]
    lon = nc_file.variables['lon'][:]

    # Define approximate bounding box for Iran
    lat_min, lat_max = 25.0, 40.0
    lon_min, lon_max = 44.0, 63.5

    # Create a mask for Iran
    iran_mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)

    # Get the sfcWind data
    sfcWind = nc_file.variables['sfcWind'][:]

    # Apply the mask to sfcWind
    sfcWind_iran = sfcWind[:, iran_mask]

    # Get time data and convert to datetime objects
    time = nc_file.variables['time'][:]
    time_units = nc_file.variables['time'].units
    time_calendar = nc_file.variables['time'].calendar
    dates = num2date(time, units=time_units, calendar=time_calendar)
    
    # Convert cftime objects to standard Python datetime objects
    dates = [datetime.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dates]

    # Get masked lat and lon
    lat_iran = lat[iran_mask]
    lon_iran = lon[iran_mask]

    nc_file.close()

    return metadata, sfcWind_attrs, height_value, height_attrs, sfcWind_iran, dates, lat_iran, lon_iran

"""
# Usage
nc_file_path = 'nc_files/Klima_Daten_10m_3h_2020_RCP26.nc'
metadata, sfcWind_attrs, height_value, height_attrs, sfcWind_iran, dates, lat_iran, lon_iran = extract_metadata_and_iranian_data(nc_file_path)

# Print relevant metadata
print("Relevant metadata:")
print(f"Source: {metadata.get('source', 'Not specified')}")
print(f"Project: {metadata.get('project_id', 'Not specified')}")
print(f"Experiment: {metadata.get('experiment', 'Not specified')}")

print("\nsfcWind variable attributes:")
for key, value in sfcWind_attrs.items():
    print(f"{key}: {value}")

if height_value is not None:
    print(f"\nHeight value: {height_value} {height_attrs.get('units', '')}")
    print("\nHeight variable attributes:")
    for key, value in height_attrs.items():
        print(f"{key}: {value}")
else:
    print("\nNo separate height variable found.")

print(f"\nShape of Iranian sfcWind data: {sfcWind_iran.shape}")
print(f"Number of grid points in Iran: {len(lat_iran)}")
print(f"Time range: {dates[0]} to {dates[-1]}")
print(f"Latitude range: {lat_iran.min():.2f} to {lat_iran.max():.2f}")
print(f"Longitude range: {lon_iran.min():.2f} to {lon_iran.max():.2f}")

# Visualize a sample of the data
plt.figure(figsize=(12, 8))
plt.scatter(lon_iran, lat_iran, c=sfcWind_iran[0], cmap='viridis')
plt.colorbar(label='Surface Wind Speed (m/s)')
plt.title(f'Surface Wind Speed over Iran at {dates[0]}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Plot time series for a single point
point_index = 0  # You can change this to look at different points
plt.figure(figsize=(12, 6))
plt.plot(dates, sfcWind_iran[:, point_index])
plt.title(f'Surface Wind Speed Time Series at Lat: {lat_iran[point_index]:.2f}, Lon: {lon_iran[point_index]:.2f}')
plt.xlabel('Date')
plt.ylabel('Surface Wind Speed (m/s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""


##########################################################
##########################################################


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import datetime

def extract_metadata_and_analyze_data(nc_file_path):
    # Open the NetCDF file
    nc_file = Dataset(nc_file_path, 'r')

    # Extract metadata
    metadata = {}
    for attr in nc_file.ncattrs():
        metadata[attr] = getattr(nc_file, attr)

    # Check sfcWind variable attributes
    sfcWind_var = nc_file.variables['sfcWind']
    sfcWind_attrs = {attr: getattr(sfcWind_var, attr) for attr in sfcWind_var.ncattrs()}

    # Check if there's a 'height' variable
    if 'height' in nc_file.variables:
        height_var = nc_file.variables['height']
        height_value = height_var[:]
        height_attrs = {attr: getattr(height_var, attr) for attr in height_var.ncattrs()}
    else:
        height_value = None
        height_attrs = None

    # Get latitude and longitude data
    lat = nc_file.variables['lat'][:]
    lon = nc_file.variables['lon'][:]

    # Get the sfcWind data
    sfcWind = nc_file.variables['sfcWind'][:]

    # Get time data and convert to datetime objects
    time = nc_file.variables['time'][:]
    time_units = nc_file.variables['time'].units
    time_calendar = nc_file.variables['time'].calendar
    dates = num2date(time, units=time_units, calendar=time_calendar)
    
    # Convert cftime objects to standard Python datetime objects
    dates = [datetime.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dates]

    nc_file.close()

    return metadata, sfcWind_attrs, height_value, height_attrs, sfcWind, dates, lat, lon

# Usage
nc_file_path = 'nc_files/Klima_Daten_10m_3h_2020_RCP26.nc'
metadata, sfcWind_attrs, height_value, height_attrs, sfcWind, dates, lat, lon = extract_metadata_and_analyze_data(nc_file_path)

# Print relevant metadata
print("Relevant metadata:")
print(f"Source: {metadata.get('source', 'Not specified')}")
print(f"Project: {metadata.get('project_id', 'Not specified')}")
print(f"Experiment: {metadata.get('experiment', 'Not specified')}")

print("\nsfcWind variable attributes:")
for key, value in sfcWind_attrs.items():
    print(f"{key}: {value}")

if height_value is not None:
    print(f"\nHeight value: {height_value} {height_attrs.get('units', '')}")
    print("\nHeight variable attributes:")
    for key, value in height_attrs.items():
        print(f"{key}: {value}")
else:
    print("\nNo separate height variable found.")

print(f"\nShape of sfcWind data: {sfcWind.shape}")
print(f"Number of grid points: {len(lat) * len(lon)}")
print(f"Time range: {dates[0]} to {dates[-1]}")
print(f"Latitude range: {lat.min():.2f} to {lat.max():.2f}")
print(f"Longitude range: {lon.min():.2f} to {lon.max():.2f}")

# Visualize the geographical coverage
plt.figure(figsize=(12, 8))
plt.scatter(lon, lat, c=sfcWind[0], cmap='viridis', s=1)
plt.colorbar(label='Surface Wind Speed (m/s)')
plt.title(f'Geographical Coverage of Surface Wind Speed Data at {dates[0]}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


import numpy as np

# Plot time series for a single point
center_lat_index = lat.shape[0] // 2
center_lon_index = lat.shape[1] // 2

# Extract the center latitude and longitude
center_lat = lat[center_lat_index, center_lon_index]
center_lon = lon[center_lat_index, center_lon_index]

# Safely convert to float, handling potential masked values
center_lat_float = float(center_lat) if not np.ma.is_masked(center_lat) else np.nan
center_lon_float = float(center_lon) if not np.ma.is_masked(center_lon) else np.nan

plt.figure(figsize=(12, 6))
plt.plot(dates, sfcWind[:, center_lat_index, center_lon_index])

# Use a conditional statement to handle potential NaN values in the title
if np.isnan(center_lat_float) or np.isnan(center_lon_float):
    plt.title('Surface Wind Speed Time Series at Center Point')
else:
    plt.title(f'Surface Wind Speed Time Series at Lat: {center_lat_float:.2f}, Lon: {center_lon_float:.2f}')

plt.xlabel('Date')
plt.ylabel('Surface Wind Speed (m/s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





#####################################################
#####################################################


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def extract_metadata_and_analyze_data(nc_file_path):
    nc_file = Dataset(nc_file_path, 'r')
    
    metadata = {attr: getattr(nc_file, attr) for attr in nc_file.ncattrs()}
    sfcWind_attrs = {attr: getattr(nc_file.variables['sfcWind'], attr) for attr in nc_file.variables['sfcWind'].ncattrs()}
    
    height_value = nc_file.variables['height'][:] if 'height' in nc_file.variables else None
    height_attrs = {attr: getattr(nc_file.variables['height'], attr) for attr in nc_file.variables['height'].ncattrs()} if height_value is not None else None

    lat = nc_file.variables['lat'][:]
    lon = nc_file.variables['lon'][:]
    sfcWind = nc_file.variables['sfcWind'][:]

    time = nc_file.variables['time'][:]
    dates = num2date(time, units=nc_file.variables['time'].units, calendar=nc_file.variables['time'].calendar)
    dates = [datetime.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dates]

    nc_file.close()
    return metadata, sfcWind_attrs, height_value, height_attrs, sfcWind, dates, lat, lon

nc_file_path = 'nc_files/Klima_Daten_10m_3h_2020_RCP26.nc'
metadata, sfcWind_attrs, height_value, height_attrs, sfcWind, dates, lat, lon = extract_metadata_and_analyze_data(nc_file_path)

print("Relevant metadata:")
print(f"Source: {metadata.get('source', 'Not specified')}")
print(f"Project: {metadata.get('project_id', 'Not specified')}")
print(f"Experiment: {metadata.get('experiment', 'Not specified')}")

print("\nsfcWind variable attributes:")
for key, value in sfcWind_attrs.items():
    print(f"{key}: {value}")

if height_value is not None:
    print(f"\nHeight value: {height_value} {height_attrs.get('units', '')}")
    print("\nHeight variable attributes:")
    for key, value in height_attrs.items():
        print(f"{key}: {value}")

print(f"\nShape of sfcWind data: {sfcWind.shape}")
print(f"Number of grid points: {len(lat) * len(lon)}")
print(f"Time range: {dates[0]} to {dates[-1]}")
print(f"Latitude range: {np.min(lat):.2f} to {np.max(lat):.2f}")
print(f"Longitude range: {np.min(lon):.2f} to {np.max(lon):.2f}")

# Visualize the geographical coverage
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.scatter(lon, lat, c=sfcWind[0], cmap='viridis', s=1, transform=ccrs.PlateCarree())
plt.colorbar(label='Surface Wind Speed (m/s)')
plt.title(f'Geographical Coverage of Surface Wind Speed Data at {dates[0]}')
ax.set_global()
plt.show()


# Plot time series for a single point (center of the grid)
center_lat_index = len(lat) // 2
center_lon_index = len(lon) // 2

# Safely extract lat and lon values
center_lat = lat[center_lat_index, center_lon_index]
center_lon = lon[center_lat_index, center_lon_index]

# Convert to float if not masked, otherwise use a placeholder
center_lat_value = float(center_lat) if not np.ma.is_masked(center_lat) else None
center_lon_value = float(center_lon) if not np.ma.is_masked(center_lon) else None

plt.figure(figsize=(12, 6))
plt.plot(dates, sfcWind[:, center_lat_index, center_lon_index])

# Create title based on available information
if center_lat_value is not None and center_lon_value is not None:
    plt.title(f'Surface Wind Speed Time Series at Lat: {center_lat_value:.2f}, Lon: {center_lon_value:.2f}')
else:
    plt.title('Surface Wind Speed Time Series at Center Point')

plt.xlabel('Date')
plt.ylabel('Surface Wind Speed (m/s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



"""
# Plot time series for a single point (center of the grid)
center_lat_index = len(lat) // 2
center_lon_index = len(lon) // 2
plt.figure(figsize=(12, 6))
plt.plot(dates, sfcWind[:, center_lat_index, center_lon_index])
plt.title(f'Surface Wind Speed Time Series at Lat: {lat[center_lat_index]:.2f}, Lon: {lon[center_lon_index]:.2f}')
plt.xlabel('Date')
plt.ylabel('Surface Wind Speed (m/s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""








# Calculate and plot average wind speed
avg_wind_speed = np.mean(sfcWind, axis=0)
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.pcolormesh(lon, lat, avg_wind_speed, cmap='viridis', transform=ccrs.PlateCarree())
plt.colorbar(label='Average Surface Wind Speed (m/s)')
plt.title('Average Surface Wind Speed for 2020')
ax.set_global()
plt.show()

##########################################################
##########################################################

"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def extract_metadata_and_analyze_data(nc_file_path):
    nc_file = Dataset(nc_file_path, 'r')
    
    metadata = {attr: getattr(nc_file, attr) for attr in nc_file.ncattrs()}
    sfcWind_attrs = {attr: getattr(nc_file.variables['sfcWind'], attr) for attr in nc_file.variables['sfcWind'].ncattrs()}
    
    height_value = nc_file.variables['height'][:] if 'height' in nc_file.variables else None
    height_attrs = {attr: getattr(nc_file.variables['height'], attr) for attr in nc_file.variables['height'].ncattrs()} if height_value is not None else None
    lat = nc_file.variables['lat'][:]
    lon = nc_file.variables['lon'][:]
    sfcWind = nc_file.variables['sfcWind'][:]
    time = nc_file.variables['time'][:]
    dates = num2date(time, units=nc_file.variables['time'].units, calendar=nc_file.variables['time'].calendar)
    dates = [datetime.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dates]
    nc_file.close()
    return metadata, sfcWind_attrs, height_value, height_attrs, sfcWind, dates, lat, lon

nc_file_path = 'nc_files/Klima_Daten_10m_3h_2020_RCP26.nc'
metadata, sfcWind_attrs, height_value, height_attrs, sfcWind, dates, lat, lon = extract_metadata_and_analyze_data(nc_file_path)

print("Relevant metadata:")
print(f"Source: {metadata.get('source', 'Not specified')}")
print(f"Project: {metadata.get('project_id', 'Not specified')}")
print(f"Experiment: {metadata.get('experiment', 'Not specified')}")

print("\nsfcWind variable attributes:")
for key, value in sfcWind_attrs.items():
    print(f"{key}: {value}")

if height_value is not None:
    print(f"\nHeight value: {height_value.item()} {height_attrs.get('units', '')}")
    print("\nHeight variable attributes:")
    for key, value in height_attrs.items():
        print(f"{key}: {value}")

print(f"\nShape of sfcWind data: {sfcWind.shape}")
print(f"Number of grid points: {len(lat) * len(lon)}")
print(f"Time range: {dates[0]} to {dates[-1]}")
print(f"Latitude range: {np.min(lat):.2f} to {np.max(lat):.2f}")
print(f"Longitude range: {np.min(lon):.2f} to {np.max(lon):.2f}")

# Visualize the geographical coverage
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.scatter(lon, lat, c=sfcWind[0], cmap='viridis', s=1, transform=ccrs.PlateCarree())
plt.colorbar(label='Surface Wind Speed (m/s)')
plt.title(f'Geographical Coverage of Surface Wind Speed Data at {dates[0]}')
ax.set_global()
plt.show()

# Plot time series for a single point (center of the grid)
center_lat_index = len(lat) // 2
center_lon_index = len(lon) // 2
plt.figure(figsize=(12, 6))
plt.plot(dates, sfcWind[:, center_lat_index, center_lon_index])
plt.title(f'Surface Wind Speed Time Series at Lat: {lat[center_lat_index].item():.2f}, Lon: {lon[center_lon_index].item():.2f}')
plt.xlabel('Date')
plt.ylabel('Surface Wind Speed (m/s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate and plot average wind speed
avg_wind_speed = np.mean(sfcWind, axis=0)
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.pcolormesh(lon, lat, avg_wind_speed, cmap='viridis', transform=ccrs.PlateCarree())
plt.colorbar(label='Average Surface Wind Speed (m/s)')
plt.title('Average Surface Wind Speed for 2020')
ax.set_global()
plt.show()

"""

################################################################


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def extract_metadata_and_analyze_data(nc_file_path):
    nc_file = Dataset(nc_file_path, 'r')
    
    metadata = {attr: getattr(nc_file, attr) for attr in nc_file.ncattrs()}
    sfcWind_attrs = {attr: getattr(nc_file.variables['sfcWind'], attr) for attr in nc_file.variables['sfcWind'].ncattrs()}
    
    height_value = nc_file.variables['height'][:] if 'height' in nc_file.variables else None
    height_attrs = {attr: getattr(nc_file.variables['height'], attr) for attr in nc_file.variables['height'].ncattrs()} if height_value is not None else None
    lat = nc_file.variables['lat'][:]
    lon = nc_file.variables['lon'][:]
    sfcWind = nc_file.variables['sfcWind'][:]
    time = nc_file.variables['time'][:]
    dates = num2date(time, units=nc_file.variables['time'].units, calendar=nc_file.variables['time'].calendar)
    dates = [datetime.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dates]
    nc_file.close()
    return metadata, sfcWind_attrs, height_value, height_attrs, sfcWind, dates, lat, lon

nc_file_path = 'nc_files/Klima_Daten_10m_3h_2020_RCP26.nc'
metadata, sfcWind_attrs, height_value, height_attrs, sfcWind, dates, lat, lon = extract_metadata_and_analyze_data(nc_file_path)

print("Relevant metadata:")
print(f"Source: {metadata.get('source', 'Not specified')}")
print(f"Project: {metadata.get('project_id', 'Not specified')}")
print(f"Experiment: {metadata.get('experiment', 'Not specified')}")

print("\nsfcWind variable attributes:")
for key, value in sfcWind_attrs.items():
    print(f"{key}: {value}")

if height_value is not None:
    print(f"\nHeight value: {height_value.item()} {height_attrs.get('units', '')}")
    print("\nHeight variable attributes:")
    for key, value in height_attrs.items():
        print(f"{key}: {value}")

print(f"\nShape of sfcWind data: {sfcWind.shape}")
print(f"Shape of latitude data: {lat.shape}")
print(f"Shape of longitude data: {lon.shape}")
print(f"Number of grid points: {lat.size}")
print(f"Time range: {dates[0]} to {dates[-1]}")
print(f"Latitude range: {np.min(lat):.2f} to {np.max(lat):.2f}")
print(f"Longitude range: {np.min(lon):.2f} to {np.max(lon):.2f}")

# Visualize the geographical coverage
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.pcolormesh(lon, lat, sfcWind[0], cmap='viridis', transform=ccrs.PlateCarree())
plt.colorbar(label='Surface Wind Speed (m/s)')
plt.title(f'Geographical Coverage of Surface Wind Speed Data at {dates[0]}')
ax.set_global()
plt.show()

# Plot time series for a single point (center of the grid)
center_lat_index = lat.shape[0] // 2
center_lon_index = lat.shape[1] // 2
plt.figure(figsize=(12, 6))
plt.plot(dates, sfcWind[:, center_lat_index, center_lon_index])
plt.title(f'Surface Wind Speed Time Series at Lat: {lat[center_lat_index, center_lon_index]:.2f}, Lon: {lon[center_lat_index, center_lon_index]:.2f}')
plt.xlabel('Date')
plt.ylabel('Surface Wind Speed (m/s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate and plot average wind speed
avg_wind_speed = np.mean(sfcWind, axis=0)
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.pcolormesh(lon, lat, avg_wind_speed, cmap='viridis', transform=ccrs.PlateCarree())
plt.colorbar(label='Average Surface Wind Speed (m/s)')
plt.title('Average Surface Wind Speed for 2020')
ax.set_global()
plt.show()












