# AWARE Narrator

## Overview
AWARE Narrator is a comprehensive Python toolkit that processes sensor data from mobile devices, performs DBSCAN clustering on location data, and generates detailed narrative descriptions of user mobility and activity patterns. The toolkit integrates multiple sensors including location, applications, keyboard input, screen usage, calls, messages, and more. It includes Google Maps API integration for reverse geocoding and uses a configuration file (`config.yaml`) to customize parameters.

## Manual Setup Requirements

### Google Maps Geocoding Update (Only if using Google Maps API)
If you plan to use Google Maps API for reverse geocoding (requires `USE_GOOGLE_MAP: true` and a valid API key file referenced by `GOOGLE_MAP_KEY` in `config.yaml`), you may need to manually update the geocoding module due to recent changes in the Google Maps Services Python library:

1. Check for updates on the GitHub repository: https://github.com/googlemaps/google-maps-services-python.git
2. Find your local geocoding.py file path (typically in your mamba/conda environment)
   - Example: `/home/ubuntu/miniforge3/envs/mv_env/lib/python3.13/site-packages/googlemaps/geocoding.py`
3. Replace your local geocoding.py with the latest version from: https://github.com/googlemaps/google-maps-services-python/blob/master/googlemaps/geocoding.py

**Alternative:** A copy of the updated `geocoding.py` file has been included in this project for convenience. You can copy it directly to replace the installed package in your current environment:

```bash
# Make sure you're in the correct environment first
mamba activate my_env  # or conda activate my_env

# Find your googlemaps package location in the current environment
python -c "import googlemaps; print(googlemaps.__file__)"
# Copy the included geocoding.py to replace the installed version
cp geocoding.py $(python -c "import googlemaps; import os; print(os.path.dirname(googlemaps.__file__))")/geocoding.py
```

This manual update ensures compatibility with the address descriptor feature in geocoding.py

**Note:** This setup is only required if you plan to use Google Maps API for reverse geocoding. If you set `USE_GOOGLE_MAP: false` in your `config.yaml`, the toolkit will work without this update.


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
   mamba env create -n <my_env> --file environment.yml

   # To update an existing environment to match environment.yml:
   mamba env update -n <my_env> --file environment.yml --prune
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
   conda env create -n <my_env> --file environment.yml

   # To update an existing environment to match environment.yml:
   conda env update -n <mv_env> --file environment.yml --prune
   ```

> **Note:** `environment.yml` was generated with:
>
> ```bash
> conda env export --from-history > environment.yml
> ```


## Project Structure
The project includes several Python scripts:

- **`aware_narrator.py`** - Main script that processes sensor data and generates narratives (manual mode with a fixed time range, or auto mode driven by per-survey timestamps; supports parallel participant processing via `num_workers`)
- **`extract_sessions.py`** - Extracts session boundaries from screen on/off events
- **`json2jsonl.py`** - Utility to convert JSON files to JSONL format
- **`split_description.py`** - Splits output narratives by sensor type
- **`map_pid_deviceid.py`** - Generate the participant ID to device ID mapping file
- **`run_screentext_preprocess_pipeline.py`** - Master pipeline for screentext data preprocessing
- **`split_by_participant.py`** - Split JSONL files by participant ID based on device mapping


## Configuration File (`config.yaml`)
The scripts require a YAML configuration file with the following structure:

```yaml
# Configuration for Aware Narrator
# Two modes: manual or auto
# Manual mode: Uses P_IDs, START_TIME, and END_TIME, output_file
# Auto mode: Uses survey_time_file, time_ranges, output_dir

MODE: "manual"  # Options: "manual" or "auto"


# START of manual mode configuration (used when MODE: "manual")

P_IDs:
  - SS001

START_TIME: "2025-05-25 06:00:00"
END_TIME: "2025-06-04 23:59:00"
output_file: "description/{P_ID}_{START_TIME}_{END_TIME}.txt"

# END of manual mode configuration


# START of auto mode configuration (used when MODE: "auto")

# CSV File containing survey-specific answered timestamp, one pid might have multiple rows
# Main columns: pid,survey_id,survey_time_unix
survey_time_file: "resources/survey_time.csv"

# Direction for time range processing:
#   - "backward": survey_time is the END point, process data BEFORE survey (default, for weekly surveys)
#                 e.g., 7d range = [survey_time - 7d, survey_time]
#   - "forward":  survey_time is the START point, process data AFTER that time (for cumulative from baseline)
#                 e.g., 7d range = [survey_time, survey_time + 7d]
direction: "backward"  # Options: "backward" or "forward"

# Alignment for the reference timestamp:
#   - true:  Align to midnight (00:00:00) of the survey timestamp's day
#   - false: Use the exact survey timestamp as-is (default)
align_to_midnight: false

# Skip records from the survey date itself:
#   - true:  Exclude all data from the calendar day of the survey
#   - false: Include survey day data (default)
skip_survey_day: false

# Time ranges to save descriptions
# Option 1: Specify start and end to auto-generate all ranges in between (same unit required)
#   Order follows start→end (ascending if start < end, descending if start > end)
time_range_start: 7d
time_range_end: 1d
#
# Option 2: Explicit list (used if time_range_start/time_range_end are not set)
# time_ranges:
#   - 7d
#   - 3d
#   - 1d

output_dir: "description/{P_ID}"

# END of auto mode configuration


# Replace by your own mapping csv. 
# Must contain device_id and pid columns. 
# Multiple device ids need to be splitted by ";"
# For example:
# Header: pid,device_id 
# Row 1:  1234,aaaa-bbbb-cccc;1111-082a-4a73-8ee3
# Row 2:  SS11,fa1da-3adrv-123a
pid_to_deviceid_map: "resources/pid_deviceid_mapping.csv" # generated by running map_pid_deviceid.py. See README for instructions.

timezone: "Australia/Melbourne" # replace by actual timezone

input_directory: "participant_data/{P_ID}"
package_to_app_map: "resources/app_package_pairs.jsonl" # required; generated by the screentext preprocess pipeline
session_data_file: "step1_data/{P_ID}/sessions.jsonl" # required for applicaion and keyboard analysis; generated by running extract_sessions.py with screen.jsonl for the corresponding P_ID
cleaned_screentext_file: "step1_data/{P_ID}/clean_input.jsonl" # required for screen text description; generated by using screen text preprocessing pipeline


reverse_geocoding_output_dir: "locations_query_results/{P_ID}"
daily_output_dir: "daily_description/{P_ID}"


num_workers: 1 # Number of parallel workers for processing participants (1 = sequential, >1 = parallel via ProcessPoolExecutor)

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

DISCARD_SYSTEM_UI: true # Applied to 'applications_foreground', 'applications_notifications', 'installations' based on system_ui_apps
USE_GOOGLE_MAP: false # Set to true to enable Google Maps reverse geocoding
GOOGLE_MAP_KEY: "GOOGLE_MAP_API_KEY.txt" # Path to a text file containing your Google API key (only read when USE_GOOGLE_MAP is true)
eps: 0.000047  # DBSCAN clustering parameter: radians (0.000047 radians × 6371000 m ≈ 300m)
min_samples: 3  # DBSCAN clustering parameter: mininum number of points to form a cluster

location_minimum_data_points: 3 # Minimum number of location data points to display a place in location description
location_minimum_stay_minutes: 3 # Minimum stay duration in minutes to display a place in location description
night_time_start: 22 # Start of nighttime in 24-hour format, used for determining home location  
night_time_end: 6 # End of nighttime in 24-hour format, used for determining home location
merge_distance_threshold: 300 # Distance threshold in meters to merge home candidates and clusters with no night points

# Using package instead of app name in case of locales (different languages for the same app name)

blacklist_apps:
  - com.aware.phone # AWARE-Light

system_ui_apps:
  - com.android.systemui                         # System UI
  - com.sec.android.app.launcher                 # One UI Home / One UI 首頁 / One UI 主屏幕 / TouchWiz home / Samsung Experience Home / Écran d'accueil One UI
  - com.samsung.android.app.cocktailbarservice   # Edge panels
  - com.huawei.android.launcher                  # 华为桌面 / Huawei Home / Beranda Huawei
  - com.miui.home                                # System launcher / Peluncur sistem / 系统桌面
  - com.oppo.launcher                            # System Launcher / システムランチャー
  - com.google.android.apps.nexuslauncher        # Pixel Launcher / Peluncur Pixel / Lanceur d'applications Pixel
  - com.motorola.launcher3                       # Moto App Launcher
  - net.oneplus.launcher                         # OnePlus Launcher
  - jp.co.sharp.android.launcher3                # AQUOS Home
  - com.android.launcher3                        # Quickstep
  - com.android.launcher                         # System Launcher
  - com.vivo.hiboard                             # Jovi Home
  - com.mi.android.globallauncher                # POCO Launcher
  - com.bbk.launcher2                            # System launcher / 系统桌面
  - com.sec.android.app.desktoplauncher          # Samsung DeX home
  - com.sec.android.emergencylauncher            # Launcher
  - com.hihonor.android.launcher                 # 荣耀桌面 / HONOR Home
  - com.sonymobile.launcher                      # Xperia主屏幕 / Xperia Home
  - com.google.android.inputmethod.latin         # Gboard




```

### Key Configuration Parameters:

#### Mode Selection:
- **`MODE`**: `"manual"` (fixed time range for all participants) or `"auto"` (per-participant, driven by survey timestamps)

#### Manual Mode Parameters (used when `MODE: "manual"`):
- **`P_IDs`**: List of Participant IDs to process
- **`START_TIME` / `END_TIME`**: Time range for data processing (YYYY-MM-DD HH:MM:SS format)
- **`output_file`**: Path for the main narrative output (supports `{P_ID}`, `{START_TIME}`, `{END_TIME}` placeholders)

#### Auto Mode Parameters (used when `MODE: "auto"`):
- **`survey_time_file`**: CSV file with `pid`, `survey_id`, `survey_time_unix` columns; each row generates one set of narratives anchored to that survey's timestamp
- **`direction`**: `"backward"` (survey_time is the end of the range, e.g. for weekly recall surveys) or `"forward"` (survey_time is the start, e.g. for cumulative tracking from a baseline)
- **`align_to_midnight`**: If `true`, snaps the survey timestamp to midnight of that day before computing ranges
- **`skip_survey_day`**: If `true`, excludes data from the survey's own calendar day
- **`time_range_start` / `time_range_end`**: Auto-generates all time ranges between these two values (same unit, e.g. `7d` to `1d`); alternatively use an explicit **`time_ranges`** list (e.g. `["7d", "3d", "1d"]`). Only the smallest range is kept when multiple ranges produce identical window counts.
- **`output_dir`**: Base directory for narrative output, organized as `output_dir/{P_ID}/{time_range}/` (supports `{P_ID}` placeholder)

#### Data Source Parameters:
- **`pid_to_deviceid_map`**: CSV file mapping participant IDs to device IDs
- **`package_to_app_map`**: JSONL file mapping app package names to human-readable app names (required; generated by the screentext preprocessing pipeline)
- **`timezone`**: Timezone for timestamp conversion (e.g., "Australia/Melbourne")
- **`input_directory`**: Path to participant data folder (supports {P_ID} placeholder)
- **`session_data_file`**: Path to sessions.jsonl file for application and keyboard analysis

#### Processing Parameters:
- **`sensor_integration_time_window`**: Time window in minutes for sensor data integration
- **`gate_time_window`**: Time window in minutes for WiFi and Bluetooth scan integration
- **`sensors`**: List of sensors to include in analysis
- **`num_workers`**: Number of participants to process in parallel (1 = sequential; >1 uses a `ProcessPoolExecutor`)

#### Output Parameters:
- **`daily_output_dir`**: Directory for daily output files (supports {P_ID} placeholder)
- **`DISCARD_SYSTEM_UI`**: Whether to filter out system UI applications

#### Location Clustering Parameters:
- **`USE_GOOGLE_MAP`**: Whether to enable Google Maps reverse geocoding
- **`GOOGLE_MAP_KEY`**: Path to a text file containing your Google API key (only read when `USE_GOOGLE_MAP` is `true`)
- **`eps`**: DBSCAN epsilon parameter (distance threshold in radians, ~300m = 0.000047)
- **`min_samples`**: DBSCAN minimum samples to form a cluster
- **`night_time_start` / `night_time_end`**: Hours defining nighttime for home location identification
- **`merge_distance_threshold`**: Distance threshold in meters to merge home candidates and clusters with no night points
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

- **Manual mode** (`MODE: "manual"`): Processes each participant in `P_IDs` over the fixed `START_TIME`–`END_TIME` range and writes a single narrative to `output_file`.
- **Auto mode** (`MODE: "auto"`): Processes each row of `survey_time_file`, generating one narrative per participant per survey per time range (see `time_ranges`/`time_range_start`/`time_range_end`), written under `output_dir`.
- Set `num_workers` > 1 in `config.yaml` to process multiple participants in parallel.
- Detailed logs are written to `logs/processing.log` and `logs/{P_ID}/processing_{P_ID}.log`; console output is limited to high-level progress and errors. A `processing_summary.txt` is also written to the output directory at the end of a run.

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

- **Manual mode narrative**: Comprehensive narrative and matching JSON (`{P_ID}_manual.json`) saved alongside `output_file`
- **Auto mode narratives**: One narrative and JSON file per participant/survey/time range, under `output_dir/{P_ID}/{time_range}/{P_ID}_{survey_id}_{time_range}_output.txt` (and `.json`)
- **Daily narratives**: Separate daily `.txt`/`.json` files in `daily_output_dir`
- **Processing summary**: `processing_summary.txt` written to the output directory, and detailed logs in `logs/`
- **Session data**: Screen usage sessions in `step1_data/{P_ID}/sessions.jsonl`
- **Split narratives**: Sensor-specific files in `description_split/{P_ID}/`
- **Reverse geocoding**: Location data with address information (if `USE_GOOGLE_MAP` is enabled)
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
- **Google Maps API errors**: Ensure `USE_GOOGLE_MAP` is `true` and `GOOGLE_MAP_KEY` points to a text file containing a valid API key (set `USE_GOOGLE_MAP: false` if not using Google Maps API)
- **App package mapping errors**: The script exits at startup if `package_to_app_map` is missing or empty — generate it via the screentext preprocessing pipeline first
- **File not found errors**: Check that participant data follows the expected directory structure
- **Session data missing**: Run `extract_sessions.py` first to generate session boundaries
- **Empty sensor files**: Verify JSONL files contain valid JSON objects, one per line
- **Geocoding issues**: If you encounter geocoding errors and are using Google Maps API, follow the manual setup instructions above to update the geocoding.py file
- **Debugging a run**: Check `logs/processing.log` (and `logs/{P_ID}/processing_{P_ID}.log`) for detailed per-participant logs — the console only shows high-level summaries and errors

## License
This project is for research purposes. Contact the developers for usage permissions.

## Contact
For questions, reach out to the maintainers of the AWARE Narrator project.