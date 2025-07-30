

# Climate Aware Deep Neural Network (CADNN) for Wind Power Simulation

## Overview


This repository presents a framework for wind power forecasting using Deep Neural Networks (DNNs) that integrate climate datasets to improve prediction accuracy. The framework leverages data from the Coupled Model Intercomparison Project (CMIP), including wind speed, atmospheric pressure, temperature, and other meteorological variables, to train DNN models. The goal is to predict wind power generation at wind farms in Germany, considering the complex nonlinear relationships between climate data and wind energy output.

The study compares various DNN architectures, such as Multilayer Perceptron (MLP), Long Short-Term Memory (LSTM) networks, and Transformer-enhanced LSTM models, to identify the most effective approach for climate-aware wind power forecasting. The repository includes a Python package, *CADNN*, that supports tasks such as statistical analysis of climate data, data visualization, preprocessing, DNN training, and performance evaluation. The results demonstrate that integrating climate data with DNN models significantly enhances forecasting accuracy, offering valuable insights for wind power generation prediction and adaptability to other regions.



[Download PDF](./CADNN.pdf)

![My Image](./CADNN.jpg)

## Dependencies

The following libraries are required for running the code:

- `numpy`
- `torch`
- `scipy`
- `pandas`
- `matplotlib`
- `cartopy`
- `scikit-learn`
- `tqdm`
- `netCDF4`
- `torch_geometric`
  
You can install these dependencies using `pip`:

```bash
pip install numpy torch scipy pandas matplotlib cartopy scikit-learn tqdm netCDF4
```

## Installed Packages

```bash
pip list
Package                  Version
------------------------ ------------
aiohappyeyeballs         2.4.3
aiohttp                  3.10.10
aiosignal                1.3.1
async-timeout            4.0.3
attrs                    24.2.0
Cartopy                  0.24.1
certifi                  2024.8.30
cftime                   1.6.4
charset-normalizer       3.4.0
contourpy                1.3.0
cycler                   0.12.1
filelock                 3.13.1
fonttools                4.54.1
frozenlist               1.4.1
fsspec                   2024.2.0
idna                     3.10
Jinja2                   3.1.3
joblib                   1.4.2
kiwisolver               1.4.7
MarkupSafe               2.1.5
matplotlib               3.9.2
mpmath                   1.3.0
multidict                6.1.0
nest-asyncio             1.6.0
netCDF4                  1.7.1.post2
networkx                 3.2.1
numpy                    1.26.3
nvidia-cublas-cu11       11.11.3.6
nvidia-cuda-cupti-cu11   11.8.87
nvidia-cuda-nvrtc-cu11   11.8.89
nvidia-cuda-runtime-cu11 11.8.89
nvidia-cudnn-cu11        9.1.0.70
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.3.0.86
nvidia-cusolver-cu11     11.4.1.48
nvidia-cusparse-cu11     11.7.5.86
nvidia-nccl-cu11         2.20.5
nvidia-nvtx-cu11         11.8.86
packaging                24.1
pandas                   2.2.3
pillow                   10.2.0
pip                      24.3.1
propcache                0.2.0
protobuf                 5.28.2
psutil                   6.0.0
pyparsing                3.1.4
pyproj                   3.7.0
pyshp                    2.3.1
python-dateutil          2.9.0.post0
pytz                     2024.2
requests                 2.32.3
scikit-learn             1.5.2
scipy                    1.14.1
setuptools               65.5.0
shapely                  2.0.6
six                      1.16.0
sympy                    1.12
tensorboardX             2.6.2.2
threadpoolctl            3.5.0
torch                    2.4.1+cu118
torch-geometric          2.6.1
torchaudio               2.4.1+cu118
torchvision              0.19.1+cu118
tqdm                     4.66.5
triton                   3.0.0
typing_extensions        4.9.0
tzdata                   2024.2
urllib3                  2.2.3
yarl                     1.14.0
```






## File Structure

```
.
├── README.md                   # This file
├── nc_files/                   # NetCDF data files for weather projections
│   └── dataset-projections-2020/ # Example data files
├── Results_2020_REMix_ReSTEP_hourly_REF.csv  # Example wind power data file
├── wind_dataset_preparation_psr.py  # Functions to prepare the wind dataset
├── wind_dataset_preparation.py    # Additional data preparation functions
├── wind_deep_simulation_framework.py  # Model definitions for deep learning
├── wind_loss.py                # Loss function for the model
├── wind_trainer.py             # Training framework
└── model_repo/                 # Folder to save trained models
```

## Usage

### Step 1: Data Preparation

The first step is to load and prepare the data. The code extracts wind speed and pressure data from NetCDF files and combines it with real-world wind power data from a CSV file.

You need to provide:

- NetCDF files with weather data (`ps_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r3i1p1_GERICS-REMO2015_v1_3hr_202001010100-202012312200.nc`, etc.)
- A CSV file with real-world wind power data (`Results_2020_REMix_ReSTEP_hourly_REF.csv`)

You can extract the wind speed and pressure data from the NetCDF files using the following functions:

```python
pressure_data, grid_lats, grid_lons = extract_pressure_for_germany(nc_file_path)
wind_speeds, grid_lats, grid_lons = extract_wind_speed_for_germany(nc_file_path)
```

### Step 2: Data Scaling and Preprocessing

Once the data is loaded, it's scaled to prepare it for training. The wind speed, pressure, and wind power data are all normalized using MinMax scaling. This scaling ensures that the model can efficiently learn from the data.

### Step 3: Model Setup

The code uses an LSTM model to predict wind power generation. The model is defined in the `LSTMDeepModel` class. The following parameters are configured for the model:

- `input_size`: The number of input features (5 features: wind speed, pressure, and time data)
- `hidden_features`: The number of hidden units in the LSTM
- `hidden_layers`: The number of layers in the LSTM
- `output_size`: The number of output features (1, the wind power prediction)
- `learning_rate`: The learning rate for optimization

### Step 4: Training the Model

The training is performed using the `LSTMTrainer` class. The model is trained for a specified number of epochs, with loss functions tracked during the process.

```python
Train_inst = LSTMTrainer(
    model_str,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
)
loss_func_list = Train_inst.train_func(train_loader, test_loader)
```

### Step 5: Model Saving

After training, the model is saved to the `model_repo/` directory in both GPU and CPU formats for future use.

```python
torch.save(model_str.state_dict(), model_save_path_gpu)
torch.save(model_str.state_dict(), model_save_path_cpu)
```

The models are saved with filenames reflecting the number of epochs, hidden features, and layers used.

### Example Usage

```python
nc_file_path = 'nc_files/dataset-projections-2020/ps_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r3i1p1_GERICS-REMO2015_v1_3hr_202001010100-202012312200.nc'
csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'

# Extract data
pressure_data, grid_lats, grid_lons = extract_pressure_for_germany(nc_file_path)
wind_speeds, grid_lats, grid_lons = extract_wind_speed_for_germany(nc_file_path)

# Load real-world wind data
target_points = load_real_wind_csv(csv_file_path)

# Interpolate wind speeds and pressure
interpolated_wind_speeds = interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points)
interpolated_pressure = interpolate_pressure(pressure_data, grid_lats, grid_lons, target_points)

# Scale data
scaled_wind_speeds = scale_interpolated_data(interpolated_wind_speeds)
scaled_pressure = scale_interpolated_data(interpolated_pressure)
scaled_target_points = scale_target_points(target_points)

# Combine data
combined_array = combine_data(scaled_target_points, scaled_unix_time_array, scaled_wind_speeds, scaled_pressure, scaled_wind_power)

# Prepare the data for training
wind_dataset_instance = LSTMDataPreparation(combined_array[:,:5], combined_array[:,5:])
x_train_seq, u_train_seq, train_loader, test_loader = wind_dataset_instance.prepare_data_random(0.05)

# Initialize and train the model
lstm_deep_model_instance = LSTMDeepModel(input_size=5, hidden_features=64, hidden_layers=7, output_size=1, learning_rate=1e-3)
model_str, optim_adam, scheduler = lstm_deep_model_instance.run()
loss_func_list = LSTMTrainer(model_str, num_epochs=25000, optim_adam=optim_adam, scheduler=scheduler).train_func(train_loader, test_loader)

# Save the model
torch.save(model_str.state_dict(), 'model_repo/lstm_model.pth')
```

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





# To use the modules and simulate the `CADNN':

## **Pipeline Overview**

```
NetCDF (wind speed, pressure)  +  CSV (wind farms & power)
          │
          ▼
Spatial Interpolation (farm coordinates)
          │
          ▼
Scaling & Feature Combination
          │
          ▼
Sequence Preparation (LSTM input)
          │
          ▼
Multi-Layer (LSTM Training)
          │
          ▼
Saved Models & Loss Arrays
```

- The dataset used in this study is available on [Zenodo](https://zenodo.org/records/14979073) and [Zenodo](https://zenodo.org/records/15736940) and `Results_2020_REMix_ReSTEP_hourly_REF.csv`.         Create a folder and rename it `nc_files`.

- For LSTM framework you should run `training_wind_psr_lstm.py`
- For MLP framework you should run `training_wind_psr.npy`
- For Hybrid (LSTM+Transformers) you should run `training_wind_psr_lstm_transform.npy`
- When you finished with the training you can evaluate the models with files `evaluate_wind_psr_lstm_cpu.py`, `evaluate_wind_psr.py`, and `evaluate_transform_lstm_psr.py`
- The files such as `wind_globe_map_eu.py` is to visualize the wind statistics over EU map. The file `wind_load.sh` is to use High Performance Computing units SLRUM, but it is case specific adn     may differe from cluster to cluster.

## **An Example of Module Structures for LSTM Framework**

```
.
├── wind_main_lstm_simulation.py         # Main simulation script
├── wind_dataset_preparation_psr.py      # Data extraction & preprocessing
├── wind_dataset_preparation.py          # Dataset preparation for LSTM
├── wind_deep_simulation_framework.py    # Model definitions (LSTM)
├── wind_trainer.py                      # Training loop & optimizers
├── wind_loss.py                         # Custom loss functions
├── model_repo/                          # Saved models & loss arrays
└── nc_files/ & CSV files                # NetCDF and real wind data
```

---

## **1. Main Simulation Script**

**File:** `wind_main_lstm_simulation.py`

This is the **entry point** of the framework.
It integrates all components: **data extraction, preprocessing, model training, and saving results**.

**Main Steps:**

1. **Load & Interpolate Data**

   * `extract_pressure_for_germany()` → Load surface pressure from NetCDF.
   * `extract_wind_speed_for_germany()` → Load wind speed from NetCDF.
   * `load_real_wind_csv()` → Load wind farm coordinates & power data.
   * `interpolate_wind_speed()` / `interpolate_pressure()` → Interpolate gridded data to wind farm locations.
2. **Scale & Combine Features**

   * `scale_interpolated_data()`, `scale_target_points()` → Normalize features to $[-1,1]$.
   * `repeat_target_points()` → Expand wind farm coordinates across time.
   * `combine_data()` → Merge all features into one dataset.
3. **Prepare Dataset for Training**

   * `LSTMDataPreparation.prepare_data_random()` → Generate sequences for training/testing.
4. **Model Training**

   * `LSTMDeepModel()` → Define multi-layer LSTM.
   * `LSTMTrainer.train_func()` → Train the model with Adam optimizer.
5. **Save Outputs**

   * Trained model: `.pth` (GPU & CPU versions).
   * Training loss: `.npy`.

---

## **2. Data Extraction & Preprocessing**

**File:** `wind_dataset_preparation_psr.py`

Provides **functions for handling NetCDF & CSV data**.

### **Data Extraction**

* `extract_wind_speed_for_germany(nc_file)`
  Extract **10m wind speed** for Germany from rotated-pole CORDEX NetCDF files.
* `extract_pressure_for_germany(nc_file)`
  Extract **surface pressure** from CORDEX NetCDF files.
* `load_real_wind_csv(csv_file_path)`
  Load **wind farm coordinates & measured power output** from CSV.

### **Interpolation**

* `interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points)`
  Interpolate gridded wind speeds onto wind farm coordinates.
* `interpolate_pressure(data, grid_lats, grid_lons, target_points)`
  Interpolate gridded pressure values onto wind farm coordinates.

### **Temporal Processing**

* `loading_wind()`
  Load wind power CSV, **clean missing values**, and **resample to 3-hourly** to match NetCDF.
* `map_unix_time_to_range()`
  Normalize timestamps to $[-1,1]$.

### **Feature Scaling & Combination**

* `scale_target_points(target_points)` → Normalize wind farm coordinates.
* `scale_interpolated_data(data)` → Scale wind speed & pressure per time step.
* `repeat_target_points(scaled_target_points, num_time_steps)` → Duplicate farm coordinates across all time steps.
* `combine_data()` → Combine all features (coordinates, time, pressure, wind speed, power).

---

## **3. Model & Training**

* **`wind_deep_simulation_framework.py`** → Defines deep learning models:

  * `WindDeepModel`, `RNNDeepModel`, `LSTMDeepModel`.
* **`wind_trainer.py`** → Implements training loop:

  * `Trainer`, `RNNTrainer`, `LSTMTrainer`.
* **`wind_loss.py`** → Custom loss functions for wind power forecasting.

---

## **Input/Output**

### **Inputs**

* **NetCDF**:

  * Wind speed (10m) and surface pressure for Germany.
* **CSV**:

  * Wind farm coordinates (lat/lon).
  * Hourly wind power generation data.

### **Outputs**

* **Trained Model:**

  * `.pth` (GPU and CPU versions).
* **Training Loss:**

  * `.npy` loss progression during training.
* **Preprocessed Dataset:**

  * Combined spatiotemporal feature arrays.

---







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


## Notes

- The code uses GPU if available for faster training.
- Ensure that you have the necessary NetCDF data files for weather projections and real-world wind power data.
- Modify the hyperparameters (e.g., `hidden_features`, `hidden_layers`, `learning_rate`) as needed to experiment with different configurations.

## Conclusion

This repository provides a framework for forecasting wind power generation using deep learning techniques, specifically LSTM networks. It processes meteorological data and uses it to train the model, which can then be used for predictions in wind power generation applications.

---

Feel free to adjust any sections to match your specific needs, such as additional setup or details on how to run the code on specific environments!








## Contributing

Pull requests are welcome.

**Author**: **Dr. Ali Forootani**  
**Email**: **aliforootani@ieee.org/aliforootani@gmail.com/ali.forootani@ufz.de**

## License

[MIT](https://choosealicense.com/licenses/mit/)
