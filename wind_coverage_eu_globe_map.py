import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

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

# Set the directory where images will be saved
output_dir = 'images_wind'

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Customize Matplotlib global settings for bigger fonts and labels
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'figure.figsize': (15, 10)
})

# Visualize the geographical coverage
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
pc = ax.pcolormesh(lon, lat, sfcWind[0], cmap='viridis', transform=ccrs.PlateCarree())

# Add colorbar
cbar = plt.colorbar(pc, ax=ax, orientation='vertical', pad=0.1, fraction=0.02)
cbar.set_label('Surface Wind Speed (m/s)')
cbar.ax.tick_params(labelsize=18)

plt.title(f'Geographical Coverage of Surface Wind Speed Data at {dates[0]}')
ax.set_global()
plt.savefig(f'{output_dir}/geographical_coverage_wind_speed.png', bbox_inches='tight')
plt.show()

# Plot time series for a single point (center of the grid)
center_lat_index = lat.shape[0] // 2
center_lon_index = lon.shape[1] // 2

# Extract and handle masked values
center_lat = lat[center_lat_index, center_lon_index]
center_lon = lon[center_lat_index, center_lon_index]

# Convert masked array values to float, replacing masked values with NaN
center_lat_value = float(center_lat) if not np.ma.is_masked(center_lat) else np.nan
center_lon_value = float(center_lon) if not np.ma.is_masked(center_lon) else np.nan

plt.figure()
plt.plot(dates, sfcWind[:, center_lat_index, center_lon_index], linewidth=4)
plt.title(f'Surface Wind Speed Time Series at Lat: {center_lat_value:.2f}, Lon: {center_lon_value:.2f}', fontsize=24)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Surface Wind Speed (m/s)', fontsize=20)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/time_series_wind_speed.png', bbox_inches='tight')
plt.show()

# Calculate and plot average wind speed
avg_wind_speed = np.mean(sfcWind, axis=0)

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
pc = ax.pcolormesh(lon, lat, avg_wind_speed, cmap='viridis', transform=ccrs.PlateCarree())

# Add colorbar
cbar = plt.colorbar(pc, ax=ax, orientation='vertical', pad=0.1, fraction=0.02)
cbar.set_label('Average Surface Wind Speed (m/s)')
cbar.ax.tick_params(labelsize=18)

plt.title(f'Average Surface Wind Speed for 2020 at {dates[0]}')
ax.set_global()
plt.savefig(f'{output_dir}/average_wind_speed_2020.png', bbox_inches='tight')
plt.show()
