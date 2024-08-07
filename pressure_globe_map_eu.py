import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def extract_metadata_and_analyze_data(nc_file_path):
    """Extracts metadata, analyzes data, and returns relevant information.

    Args:
        nc_file_path (str): Path to the netCDF file.

    Returns:
        tuple: A tuple containing:
            - metadata (dict): Dictionary containing global attributes of the dataset.
            - variable_attrs (dict): Dictionary containing attributes of the 'psl' variable.
            - data (np.ndarray): The data array from the 'psl' variable.
            - dates (list): List of datetime objects corresponding to each time step.
            - lat (np.ndarray): Array containing latitude values.
            - lon (np.ndarray): Array containing longitude values.
    """

    nc_file = Dataset(nc_file_path, 'r')

    metadata = {attr: getattr(nc_file, attr) for attr in nc_file.ncattrs()}
    variable_attrs = {attr: getattr(nc_file.variables['psl'], attr) for attr in nc_file.variables['psl'].ncattrs()}

    data = nc_file.variables['psl'][:]

    time = nc_file.variables['time'][:]
    dates = num2date(time, units=nc_file.variables['time'].units, calendar=nc_file.variables['time'].calendar)
    dates = [datetime.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d  in dates]

    lat = nc_file.variables['lat'][:]
    lon = nc_file.variables['lon'][:]

    nc_file.close()

    return metadata, variable_attrs, data, dates, lat, lon


# Define the netCDF file path
nc_file_path = 'nc_files/dataset-projections-cordex-domains-single-levels-69ac4dd9-7e75-46a0-8eef-7be736876191/psl_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r3i1p1_GERICS-REMO2015_v1_3hr_202001010100-202012312200.nc'

# Extract data and metadata
metadata, variable_attrs, sfcPressure, dates, lat, lon = extract_metadata_and_analyze_data(nc_file_path)

print("Relevant metadata:")
print(f"Source: {metadata.get('source', 'Not specified')}")
print(f"Project: {metadata.get('project_id', 'Not specified')}")
print(f"Experiment: {metadata.get('experiment', 'Not specified')}")

print("\nsfcPressure variable attributes:")
for key, value in variable_attrs.items():
    print(f"{key}: {value}")

print(f"\nShape of sfcPressure data: {sfcPressure.shape}")
print(f"Number of grid points: {len(lat) * len(lon)}")
print(f"Time range: {dates[0]} to {dates[-1]}")
print(f"Latitude range: {np.min(lat):.2f} to {np.max(lat):.2f}")
print(f"Longitude range: {np.min(lon):.2f} to {np.max(lon):.2f}")

# Visualize the geographical coverage
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.scatter(lon, lat, c=sfcPressure[0], cmap='viridis', s=1, transform=ccrs.PlateCarree())
plt.colorbar(label='Surface Pressure (Pa)')
plt.title(f'Geographical Coverage of Surface Pressure Data at {dates[0]}')
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
plt.plot(dates, sfcPressure[:, center_lat_index, center_lon_index])

# Create title based on available information
if center_lat_value is not None and center_lon_value is not None:
    plt.title(f'Surface Pressure Time Series at Lat: {center_lat_value:.2f}, Lon: {center_lon_value:.2f}')
else:
    plt.title('Surface Pressure Time Series at Center Point')

plt.xlabel('Date')
plt.ylabel('Surface Pressure (Pa)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Calculate and plot average wind speed
avg_sfcPressure = np.mean(sfcPressure, axis=0)
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.pcolormesh(lon, lat, avg_sfcPressure, cmap='viridis', transform=ccrs.PlateCarree())
plt.colorbar(label='Average Surface Pressure (Pa)')
plt.title('Average Surface Pressure for 2020')
ax.set_global()
plt.show()



###################################################
###################################################


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def extract_metadata_and_analyze_data(nc_file_path):
    """Extracts metadata, analyzes data, and returns relevant information.

    Args:
        nc_file_path (str): Path to the netCDF file.

    Returns:
        tuple: A tuple containing:
            - metadata (dict): Dictionary containing global attributes of the dataset.
            - variable_attrs (dict): Dictionary containing attributes of the 'psl' variable.
            - data (np.ndarray): The data array from the 'psl' variable.
            - dates (list): List of datetime objects corresponding to each time step.
            - lat (np.ndarray): Array containing latitude values.
            - lon (np.ndarray): Array containing longitude values.
    """

    nc_file = Dataset(nc_file_path, 'r')

    metadata = {attr: getattr(nc_file, attr) for attr in nc_file.ncattrs()}
    variable_attrs = {attr: getattr(nc_file.variables['psl'], attr) for attr in nc_file.variables['psl'].ncattrs()}

    data = nc_file.variables['psl'][:]

    time = nc_file.variables['time'][:]
    dates = num2date(time, units=nc_file.variables['time'].units, calendar=nc_file.variables['time'].calendar)
    dates = [datetime.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d  in dates]

    lat = nc_file.variables['lat'][:]
    lon = nc_file.variables['lon'][:]

    nc_file.close()

    return metadata, variable_attrs, data, dates, lat, lon


# Define the netCDF file path
nc_file_path = 'nc_files/dataset-projections-cordex-domains-single-levels-69ac4dd9-7e75-46a0-8eef-7be736876191/psl_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r3i1p1_GERICS-REMO2015_v1_3hr_202001010100-202012312200.nc'

# Extract data and metadata
metadata, variable_attrs, sfcPressure, dates, lat, lon = extract_metadata_and_analyze_data(nc_file_path)

print("Relevant metadata:")
print(f"Source: {metadata.get('source', 'Not specified')}")
print(f"Project: {metadata.get('project_id', 'Not specified')}")
print(f"Experiment: {metadata.get('experiment', 'Not specified')}")

print("\nsfcPressure variable attributes:")
for key, value in variable_attrs.items():
    print(f"{key}: {value}")

print(f"\nShape of sfcPressure data: {sfcPressure.shape}")
print(f"Number of grid points: {len(lat) * len(lon)}")
print(f"Time range: {dates[0]} to {dates[-1]}")
print(f"Latitude range: {np.min(lat):.2f} to {np.max(lat):.2f}")
print(f"Longitude range: {np.min(lon):.2f} to {np.max(lon):.2f}")

# Visualize the geographical coverage
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.scatter(lon, lat, c=sfcPressure[0], cmap='viridis', s=1, transform=ccrs.PlateCarree())
plt.colorbar(label='Surface Pressure (Pa)')
plt.title(f'Geographical Coverage of Surface Pressure Data at {dates[0]}')
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
plt.plot(dates, sfcPressure[:, center_lat_index, center_lon_index])

# Create title based on available information
if center_lat_value is not None and center_lon_value is not None:
    plt.title(f'Surface Pressure Time Series at Lat: {center_lat_value:.2f}, Lon: {center_lon_value:.2f}')
else:
    plt.title('Surface Pressure Time Series at Center Point')

plt.xlabel('Date')
plt.ylabel('Surface Pressure (Pa)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Calculate and plot average wind speed
avg_sfcPressure = np.mean(sfcPressure, axis=0)
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.pcolormesh(lon, lat, avg_sfcPressure, cmap='viridis', transform=ccrs.PlateCarree())
plt.colorbar(label='Average Surface Pressure (Pa)')
plt.title('Average Surface Pressure for 2020')
ax.set_global()
plt.show()
























