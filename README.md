# AWARE Narrator

## Overview
AWARE Narrator is a comprehensive Python toolkit that processes sensor data from mobile devices, performs DBSCAN clustering on location data, and generates detailed narrative descriptions of user mobility and activity patterns. The toolkit integrates multiple sensors including location, applications, keyboard input, screen usage, calls, messages, and more. It includes Google Maps API integration for reverse geocoding and uses a configuration file (`config.yaml`) to customize parameters.

## Manual Setup Requirements

### Google Maps Geocoding Update (Only if using Google Maps API)
If you plan to use Google Maps API for reverse geocoding (requires a valid API key in `config.yaml`), you may need to manually update the geocoding module due to recent changes in the Google Maps Services Python library:

1. Check for updates on the GitHub repository: https://github.com/googlemaps/google-maps-services-python.git
2. Find your local geocoding.py file path (typically in your conda/mamba environment)
   - Example: `/home/ubuntu/miniforge3/envs/your_envname/lib/python3.13/site-packages/googlemaps/geocoding.py`
3. Replace your local geocoding.py with the latest version from: https://github.com/googlemaps/google-maps-services-python/blob/master/googlemaps/geocoding.py

**Alternative:** A copy of the updated `geocoding.py` file has been included in this project for convenience. You can copy it directly to replace the installed package in your current environment:

```bash
# Make sure you're in the correct environment first
conda activate your_environment_name  # or mamba activate your_environment_name

# Find your googlemaps package location in the current environment
python -c "import googlemaps; print(googlemaps.__file__)"
# Copy the included geocoding.py to replace the installed version
cp geocoding.py $(python -c "import googlemaps; import os; print(os.path.dirname(googlemaps.__file__))")/geocoding.py
```

This manual update ensures compatibility with the address descriptor feature in geocoding.py

**Note:** This setup is only required if you plan to use Google Maps API for reverse geocoding. If you leave the `GOOGLE_MAP_KEY` empty in your `config.yaml`, the toolkit will work without this update.


## Installation

This project supports **Mamba** and **Conda** for managing and installing dependencies. We recommend using Mamba for faster package resolution and installation.

### Using Mamba (Recommended)

1. Ensure you have [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) installed.
2. Activate your Mamba environment (or create one if needed):

   ```bash
   mamba create -n my_env python=3.13
   mamba activate my_env
   ```
3. Create (or update) your environment from the file:

   ```bash
   # Create a new environment with the name defined in the file:
   mamba env create --file environment.yml

   # Or create under a custom name:
   mamba env create -n <your_env_name> --file environment.yml

   # To update an existing environment to match environment.yml:
   mamba env update -n <your_env_name> --file environment.yml --prune
   ```

### Using Conda

1. Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.
2. Activate your Conda environment (or create one if needed):

   ```bash
   conda create -n my_env python=3.13
   conda activate my_env
   ```
3. Create (or update) your environment from the file:

   ```bash
   # Create a new environment with the name defined in the file:
   conda env create --file environment.yml

   # Or create under a custom name:
   conda env create -n <your_env_name> --file environment.yml

   # To update an existing environment to match environment.yml:
   conda env update -n <your_env_name> --file environment.yml --prune
   ```

> **Note:** `environment.yml` was generated with:
>
> ```bash
> conda env export --from-history > environment.yml
> ```


## Project Structure
The project includes several Python scripts:

- **`aware_narrator.py`** - Main script that processes sensor data and generates narratives
- **`extract_sessions.py`** - Extracts session boundaries from screen on/off events
- **`json2jsonl.py`** - Utility to convert JSON files to JSONL format
- **`split_description.py`** - Splits output narratives by sensor type
- **`map_pid_deviceid.py`** - Generate the participant ID to device ID mapping file
- **`run_screentext_preprocess_pipeline.py`** - Master pipeline for screentext data preprocessing
- **`split_by_participant.py`** - Split JSONL files by participant ID based on device mapping


## Configuration File (`config.yaml`)
The scripts require a YAML configuration file with the following structure:

```yaml
# Replace by your own mapping csv. 
# Must contain device_id and pid columns. 
# Multiple device ids need to be splitted by ";"
# For example:
# Header: pid,device_id 
# Row 1:  1234,aaaa-bbbb-cccc;1111-082a-4a73-8ee3
# Row 2:  SS11,fa1da-3adrv-123a
pid_to_deviceid_map: "resources/pid_deviceid_mapping.csv" # generated by running map_pid_deviceid.py. See README for instructions.

# Replace by actual participant id(s)
P_IDs:
  - SS001

START_TIME: "2025-05-25 06:00:00"
END_TIME: "2025-06-04 23:59:00"
timezone: "Australia/Melbourne" # replace by actual timezone

input_directory: "participant_data/{P_ID}"
session_data_file: "step1_data/{P_ID}/sessions.jsonl" # required for applicaion and keyboard analysis; generated by running extract_sessions.py with screen.jsonl for the corresponding P_ID
cleaned_screentext_file: "step1_data/{P_ID}/clean_input.jsonl" # required for screen text description; generated by using screen text preprocessing pipeline from "Guardian Angel" project


reverse_geocoding_output_dir: "locations_query_results/{P_ID}"
output_file: "description/{P_ID}_output.txt"
daily_output_dir: "daily_description/{P_ID}"


sensor_integration_time_window: 60 # minutes
gate_time_window: 5 # minutes; required for wifi and bluetooth scan data integration.

sensors:
  - "applications_foreground"
  - "applications_notifications"
  - "battery"
  - "bluetooth"
  - "calls"
  - "installations"
  - "keyboard"
  - "messages"
  - "screen"
  - "screentext"
  - "wifi"
  - "sensor_wifi"
  - "locations"

DISCARD_SYSTEM_UI: true
GOOGLE_MAP_KEY: "" # Used for Google API Reverse geocoding
eps: 0.000047  # DBSCAN clustering parameter: ~300m radius (0.000047 radians × 6371000 m ≈ 300m)
min_samples: 3  # DBSCAN clustering parameter: mininum number of points to form a cluster

location_minimum_data_points: 3 # Minimum number of location data points to display a place in location description
location_minimum_stay_minutes: 3 # Minimum stay duration in minutes to display a place in location description

night_time_start: 22 # Start of nighttime in 24-hour format, used for determining home location  
night_time_end: 6 # End of nighttime in 24-hour format, used for determining home location

blacklist_apps:
  - AWARE-Light

whitelist_system_apps:
  - Phone
  - Dialer
  - Messages
  - Contacts
  - People
  - Camera
  - Gallery
  - Photos
  - Gboard
  - Android Keyboard
  - Calculator
  - Clock
  - Calendar
  - Files
  - File Manager
  - Email
  - Gmail
  - Settings
  - Browser
  - Chrome
  - Maps
  - YouTube
  - Play Store
  - Play Music
  - YouTube Music
  - Drive
  - Google Search
  - Assistant
  - Recorder
  - Notes
  - Keep
  - Weather
  # Optional pre-installed OEM apps
  - Samsung Notes
  - MIUI Security
  - Huawei Health
  - Samsung Internet Browser
  - My Files

```

### Key Configuration Parameters:

#### Data Source Parameters:
- **`pid_to_deviceid_map`**: CSV file mapping participant IDs to device IDs
- **`P_IDs`**: List of Participant IDs to process
- **`START_TIME` / `END_TIME`**: Time range for data processing (YYYY-MM-DD HH:MM:SS format)
- **`timezone`**: Timezone for timestamp conversion (e.g., "Australia/Melbourne")
- **`input_directory`**: Path to participant data folder (supports {P_ID} placeholder)
- **`session_data_file`**: Path to sessions.jsonl file for application and keyboard analysis

#### Processing Parameters:
- **`sensor_integration_time_window`**: Time window in minutes for sensor data integration
- **`gate_time_window`**: Time window in minutes for WiFi and Bluetooth scan integration
- **`sensors`**: List of sensors to include in analysis
- **`frequency_gps` / `frequency_network`**: Location sampling rates in seconds

#### Output Parameters:
- **`output_file`**: Path for the main narrative output (supports {P_ID} placeholder)
- **`daily_output_dir`**: Directory for daily output files (supports {P_ID} placeholder)
- **`DISCARD_SYSTEM_UI`**: Whether to filter out system UI applications

#### Location Clustering Parameters:
- **`GOOGLE_MAP_KEY`**: (Optional) API key for Google Maps reverse geocoding
- **`eps`**: DBSCAN epsilon parameter (distance threshold in radians, ~300m = 0.000047)
- **`min_samples`**: DBSCAN minimum samples to form a cluster
- **`night_time_start` / `night_time_end`**: Hours defining nighttime for home location identification
- **`location_minimum_data_points`**: Minimum number of location data points required to display a place in location description (default: 3)
- **`location_minimum_stay_minutes`**: Minimum stay duration in minutes required to display a place in location description (default: 3)

## How to Run the Scripts

### 1. PID to Device ID Mapping Generator

Generate the participant ID to device ID mapping file required by the main analysis script:

```sh
python map_pid_deviceid.py
```

This script processes a CSV file containing participant information and creates a mapping file used by `aware_narrator.py`.

**Input Requirements:**
- CSV file with at least 2 columns:
  - `pid`: Participant ID
  - `device_id`: Device ID (multiple device IDs can be separated by ";")

**Default behavior:**
- Input file: `resources/participants.csv`
- Output file: `resources/pid_deviceid_mapping.csv`

The output mapping file is referenced in `config.yaml` as `pid_to_deviceid_map` and is required for the main analysis script to function properly.

### 2. Split Data by Participant

Split JSONL files from the exported data into participant-specific directories:

```sh
# Split all JSONL files for all participants
python split_by_participant.py

# Split specific JSONL files only
python split_by_participant.py --jsonl-files locations applications_foreground

# Process only specific participants
python split_by_participant.py --pids SS001 SS002 SS003

# Run threshold analysis mode to analyze data distribution
python split_by_participant.py --threshold-analysis

# Custom input/output directories
python split_by_participant.py --input-dir exported_jsonl --output-dir participant_data
```

This script processes JSONL files from the `exported_jsonl` directory and splits them into the `participant_data/{P_ID}/` structure required by the main analysis.

**Features:**
- Parallel processing for large datasets
- Threshold analysis mode for data quality assessment
- Unknown device reporting
- Participant filtering options

### 3. Screentext Preprocessing Pipeline

Process screentext data through the complete preprocessing pipeline:

```sh
# Process a single participant
python run_screentext_preprocess_pipeline.py --participant SS001

# Process all participants (Step 1 sequential, Steps 2-5 parallel)
python run_screentext_preprocess_pipeline.py --all

# Process specific participants only
python run_screentext_preprocess_pipeline.py --all --include SS001 SS002 SS003

# Process all participants except specific ones
python run_screentext_preprocess_pipeline.py --all --exclude SS001 SS002

# Custom timezone and worker threads
python run_screentext_preprocess_pipeline.py --all --timezone "Australia/Melbourne" --workers 8
```

**Pipeline Steps:**
1. Generate app package pairs (participant-specific)
2. Clean screentext data
3. Generate filtered system app transition files
4. Add day IDs
5. Calculate session metrics

**Note:** The screentext preprocessing pipeline includes session extraction functionality, so you do **NOT** need to run `extract_sessions.py` separately if you're using the screentext pipeline. The pipeline generates the required `sessions.jsonl` file as part of Step 5.

### 4. Extract Sessions Script (Alternative)

If you're not using the screentext preprocessing pipeline, you can extract session boundaries separately:

```sh
# Process a single participant
python extract_sessions.py --participant SS001

# Process all participants in the input directory
python extract_sessions.py --all

# Custom session threshold (default: 45000ms = 45 seconds)
python extract_sessions.py --participant SS001 --threshold 45000

# Custom input/output directories
python extract_sessions.py --participant SS001 --input-dir custom_data --output-dir custom_output
```

**Important:** Only run this script if you're NOT using the screentext preprocessing pipeline, as the pipeline already includes session extraction.

### 5. JSON to JSONL Converter

Convert JSON files to JSONL format:

```sh
# Convert all JSON files in a folder
python json2jsonl.py /path/to/json/folder

# Specify output folder
python json2jsonl.py /path/to/json/folder -o /path/to/output/folder

# Search recursively in subdirectories
python json2jsonl.py /path/to/json/folder -r -o /path/to/output/folder
```

### 6. Main Analysis Script

```sh
python aware_narrator.py
```

This processes all sensor data according to the configuration and generates comprehensive narratives.

### 7. Split Description Script

Split output narratives by sensor type:

```sh
python split_description.py
```

This creates separate files for each sensor type in the `description_split/{PID}/` folder.

## Input Data Structure

The project expects the following directory structure:

```
exported_jsonl/                    # Raw exported JSONL files (for split_by_participant.py)
├── applications_foreground.jsonl
├── applications_notifications.jsonl
├── battery.jsonl
├── bluetooth.jsonl
├── calls.jsonl
├── installations.jsonl
├── keyboard.jsonl
├── locations.jsonl
├── messages.jsonl
├── screen.jsonl
├── screentext.jsonl              # Required for screentext analysis
├── wifi.jsonl
└── sensor_wifi.jsonl

participant_data/                  # After running split_by_participant.py
├── {P_ID}/
│   ├── applications_foreground.jsonl
│   ├── applications_notifications.jsonl
│   ├── battery.jsonl
│   ├── bluetooth.jsonl
│   ├── calls.jsonl
│   ├── installations.jsonl
│   ├── keyboard.jsonl
│   ├── locations.jsonl
│   ├── messages.jsonl
│   ├── screen.jsonl
│   ├── screentext.jsonl          # Required for screentext analysis
│   ├── wifi.jsonl
│   └── sensor_wifi.jsonl

step1_data/                        # After running screentext pipeline or extract_sessions.py
├── {P_ID}/
│   ├── sessions.jsonl            # Generated by screentext pipeline or extract_sessions.py
│   └── clean_input.jsonl         # Generated by screentext pipeline

resources/
├── pid_deviceid_mapping.csv      # Generated by map_pid_deviceid.py
└── app_package_pairs.jsonl       # Generated by screentext pipeline
```

## Output

The toolkit generates several types of output:

- **Main narrative file**: Comprehensive narrative saved to `output_file` path
- **Daily narratives**: Separate daily files in `daily_output_dir`
- **Session data**: Screen usage sessions in `step1_data/{P_ID}/sessions.jsonl`
- **Split narratives**: Sensor-specific files in `description_split/{P_ID}/`
- **Reverse geocoding**: Location data with address information (if Google Maps API is enabled)
- **Clustering analysis**: Location clusters with home detection

## Sensor Data Analysis

The toolkit analyzes the following sensor types:

- **Location**: GPS and network-based location with DBSCAN clustering
- **Applications**: Foreground app usage with session correlation
- **Keyboard**: Typing patterns and text input analysis with improved human-readable descriptions
- **Screen**: Screen on/off events and usage sessions
- **Screentext**: Text content displayed on screen during app usage
- **Battery**: Battery level and charging status
- **Calls**: Phone call events and durations
- **Messages**: SMS and messaging activity
- **WiFi/Bluetooth**: Network scanning and connectivity
- **Notifications**: Application notification events with proper text display handling

## Troubleshooting

- **JSONDecodeError**: Ensure `config.yaml` is properly formatted YAML (no JSON-style comments)
- **ModuleNotFoundError**: Install dependencies using `mamba env create --file environment.yml` or `conda env create --file environment.yml`
- **Google Maps API errors**: Ensure a valid API key is provided in `GOOGLE_MAP_KEY` (or leave empty if not using Google Maps API)
- **File not found errors**: Check that participant data follows the expected directory structure
- **Session data missing**: Run `extract_sessions.py` first to generate session boundaries
- **Empty sensor files**: Verify JSONL files contain valid JSON objects, one per line
- **Geocoding issues**: If you encounter geocoding errors and are using Google Maps API, follow the manual setup instructions above to update the geocoding.py file

## License
This project is for research purposes. Contact the developers for usage permissions.

## Contact
For questions, reach out to the maintainers of the AWARE Narrator project.