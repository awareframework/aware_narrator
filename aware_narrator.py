import yaml
import pytz
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import googlemaps
from googlemaps import exceptions as gexceptions
import re
import logging
import unicodedata
from pathlib import Path
import ftfy
from concurrent.futures import ProcessPoolExecutor, as_completed

# Unicode categories to strip: Cc (control), Cf (format, e.g. bidi overrides,
# zero-width chars, soft hyphens).  We keep Zs (space separators) except for
# non-breaking space (U+00A0) which is normalised to a regular space by ftfy.
_STRIP_CATEGORIES = {'Cc', 'Cf'}


def clean_text(text: str) -> str:
    """Clean text by fixing encoding issues and removing invisible control characters."""
    if not text:
        return text
    # Fix mojibake / encoding errors and normalise whitespace
    text = ftfy.fix_text(text)
    # Remove characters whose Unicode category is control or format
    text = ''.join(ch for ch in text if unicodedata.category(ch) not in _STRIP_CATEGORIES or ch in '\n\r\t')
    return text


def _sanitize_unicode_recursive(obj):
    """Recursively clean Unicode in all strings in a data structure."""
    if isinstance(obj, str):
        return clean_text(obj)
    elif isinstance(obj, dict):
        return {k: _sanitize_unicode_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_unicode_recursive(item) for item in obj]
    return obj

# Set up logging
def setup_logging(pid=None, survey_id=None, console_level=logging.ERROR):
    """
    Set up logging to redirect detailed messages to log files while keeping only essential messages in console.
    
    Args:
        pid (str, optional): Participant ID for log file naming
        survey_id (str, optional): Survey ID for log file naming
        console_level: Logging level for console output (default: ERROR for minimal output)
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger()
    
    # Always ensure main logging is set up (don't check if handlers exist)
    main_log_file = log_dir / "processing.log"
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set up main logging manually since basicConfig won't work after handlers exist
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Main file handler for all messages
    file_handler = logging.FileHandler(main_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler for errors only (or specified level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # If participant-specific logging is requested, ADD a participant-specific handler
    # (don't remove the main handler)
    if pid:
        participant_log_dir = log_dir / pid
        participant_log_dir.mkdir(exist_ok=True)
        participant_log_file = participant_log_dir / f"processing_{pid}.log"
        
        # Check if we already have a handler for this participant file
        existing_participant_handler = None
        for handler in logger.handlers:
            if (isinstance(handler, logging.FileHandler) and 
                hasattr(handler, 'baseFilename') and 
                str(participant_log_file) in handler.baseFilename):
                existing_participant_handler = handler
                break
        
        # If no existing handler for this participant, add one
        if not existing_participant_handler:
            participant_handler = logging.FileHandler(participant_log_file, mode='w', encoding='utf-8')
            participant_handler.setLevel(logging.INFO)
            participant_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(participant_handler)
    
    return logger

def log_info(message, logger=None):
    """Log info message to file only (not console)"""
    if logger:
        logger.info(message)
    else:
        logging.info(message)

def log_warning(message, logger=None):
    """Log warning message to both file and console"""
    if logger:
        logger.warning(message)
    else:
        logging.warning(message)

def log_error(message, logger=None):
    """Log error message to both file and console"""
    if logger:
        logger.error(message)
    else:
        logging.error(message)

def log_summary(message, logger=None):
    """Log summary message to both file and console (for high-level progress)"""
    if logger:
        logger.info(message)
    else:
        logging.info(message)
    # Always print to console for summary information
    print(message)

# Load configuration from yaml file
CONFIG_FILE = "./config.yaml"
with open(CONFIG_FILE, "r", encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)

# Assign variables from yaml
pid_to_deviceid_map = CONFIG["pid_to_deviceid_map"]
package_to_app_map = CONFIG["package_to_app_map"]

# Device ID management functions
# Global cache for device ID mappings
_device_id_cache = None

def load_all_device_ids_from_csv(csv_file_path):
    """
    Load device IDs from CSV file for all participants at once.
    
    Args:
        csv_file_path (str): Path to the CSV file containing pid to device_id mapping
        
    Returns:
        dict: Dictionary mapping participant IDs to lists of device IDs
    """
    global _device_id_cache
    
    if _device_id_cache is not None:
        return _device_id_cache
    
    try:
        df = pd.read_csv(csv_file_path)
        device_mapping = {}
        
        for _, row in df.iterrows():
            pid = row['pid']
            device_id_str = row['device_id']
            device_ids = [device_id.strip() for device_id in device_id_str.split(';')]
            device_mapping[pid] = device_ids
        
        _device_id_cache = device_mapping
        log_info(f"Loaded device IDs for {len(device_mapping)} participants from {csv_file_path}")
        return device_mapping
        
    except FileNotFoundError:
        log_warning(f"Error: CSV file {csv_file_path} not found")
        return {}
    except Exception as e:
        log_error(f"Error reading CSV file {csv_file_path}: {e}")
        return {}

def get_device_ids_for_participant(participant_id, csv_file_path=None):
    """
    Convenient function to get device IDs for a participant using the global cache.
    
    Args:
        participant_id (str): Participant ID to lookup
        csv_file_path (str, optional): Path to CSV file, uses config default if not provided
        
    Returns:
        list: List of device IDs for the participant
    """
    if csv_file_path is None:
        csv_file_path = pid_to_deviceid_map
    
    all_device_mappings = load_all_device_ids_from_csv(csv_file_path)
    return all_device_mappings.get(participant_id, [])

# Global configuration variables
timezone = pytz.timezone(CONFIG["timezone"])
sensor_integration_time_window = CONFIG["sensor_integration_time_window"]
gate_time_window = CONFIG["gate_time_window"]
sensors = CONFIG["sensors"]

def load_google_maps_api_key(key_path):
    """
    Load Google Maps API key from a file path.
    
    Args:
        key_path (str): Path to the file containing the API key
        
    Returns:
        str: The API key, or None if file cannot be read
    """
    try:
        with open(key_path, 'r') as file:
            api_key = file.read().strip()
            if api_key:
                log_info(f"✓ Loaded Google Maps API key from {key_path}")
                return api_key
            else:
                log_warning(f"⚠️ Google Maps API key file {key_path} is empty")
                return None
    except FileNotFoundError:
        log_warning(f"⚠️ Google Maps API key file {key_path} not found")
        return None
    except Exception as e:
        log_error(f"❌ Error reading Google Maps API key from {key_path}: {e}")
        return None

# Load Google Maps API key from file
GOOGLE_MAP_KEY = load_google_maps_api_key(CONFIG["GOOGLE_MAP_KEY"])

# Google Maps configuration
USE_GOOGLE_MAP = CONFIG.get("USE_GOOGLE_MAP", False)

eps = CONFIG["eps"]
min_samples = CONFIG["min_samples"]
DISCARD_SYSTEM_UI = CONFIG["DISCARD_SYSTEM_UI"]
night_time_start = CONFIG["night_time_start"]
night_time_end = CONFIG["night_time_end"]
merge_distance_threshold = CONFIG.get("merge_distance_threshold", 300)  # Distance threshold in meters to merge home candidates
location_minimum_data_points = CONFIG.get("location_minimum_data_points", 3)  # Minimum location data points to display a location
location_minimum_stay_minutes = CONFIG.get("location_minimum_stay_minutes", 5)  # Minimum stay duration to display a location

blacklist_apps = CONFIG["blacklist_apps"]
system_ui_apps = CONFIG["system_ui_apps"]

# Global variables for app processing
# Load app package name to application name mapping from resources folder at module import
def _load_app_name_mapping_global():
    """Load app package mapping into global variable at module import time."""
    app_name_mapping = {}
    try:
        with open(package_to_app_map, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    app_data = json.loads(line)
                    package_name = app_data.get("package_name")
                    application_name = app_data.get("application_name")
                    if package_name and application_name:
                        app_name_mapping[package_name] = application_name
        
        if not app_name_mapping:
            log_error(f"❌ ERROR: App package pairs file {package_to_app_map} exists but contains no valid mappings")
            log_error("  Please ensure the file contains valid JSONL data with package_name and application_name fields")
            sys.exit(1)
        
        log_info(f"✓ Loaded {len(app_name_mapping)} app package mappings globally")
    except FileNotFoundError:
        log_error(f"❌ ERROR: App package pairs file {package_to_app_map} not found")
        log_error("  This file is required for processing. Please run generate_consolidated_app_package_pair.py first")
        sys.exit(1)
    except Exception as e:
        log_error(f"❌ ERROR: Failed to read app package pairs file {package_to_app_map}: {e}")
        sys.exit(1)
    return app_name_mapping

# Load the mapping globally at module import
application_name_list = _load_app_name_mapping_global()

def parse_time_range_duration(time_range):
    """
    Parse time range string to duration in milliseconds.
    
    Args:
        time_range (str): Time range string like '7d', '24h', '3h', '1h'
        
    Returns:
        int: Duration in milliseconds
    """
    if time_range.endswith('w'):
        # Week
        weeks = int(time_range[:-1])
        return weeks * 7 * 24 * 60 * 60 * 1000
    elif time_range.endswith('d'):
        # Day
        days = int(time_range[:-1])
        return days * 24 * 60 * 60 * 1000
    elif time_range.endswith('h'):
        # Hour
        hours = int(time_range[:-1])
        return hours * 60 * 60 * 1000
    else:
        raise ValueError(f"Unsupported time range format: {time_range}")

def expand_time_ranges(start, end):
    """
    Generate a list of time range strings from start to end (inclusive).
    Both must use the same unit (h, d, or w). Steps by 1 in that unit.
    Order follows start→end (ascending if start < end, descending if start > end).

    Args:
        start (str): Start time range (e.g., '1d')
        end (str): End time range (e.g., '119d')

    Returns:
        list: List of time range strings (e.g., ['1d', '2d', ..., '119d'])
    """
    start_unit = start[-1]
    end_unit = end[-1]
    if start_unit != end_unit:
        raise ValueError(
            f"time_range_start ('{start}') and time_range_end ('{end}') must use the same unit. "
            f"Got '{start_unit}' and '{end_unit}'."
        )
    if start_unit not in ('h', 'd', 'w'):
        raise ValueError(f"Unsupported time range unit: '{start_unit}'. Must be 'h', 'd', or 'w'.")

    start_val = int(start[:-1])
    end_val = int(end[:-1])
    step = 1 if start_val <= end_val else -1
    return [f"{v}{start_unit}" for v in range(start_val, end_val + step, step)]

def load_survey_time_data(survey_time_file):
    """
    Load survey time data from CSV file.
    
    Args:
        survey_time_file (str): Path to the survey time CSV file
        
    Returns:
        pandas.DataFrame: Survey time data with columns: pid, survey_id, survey_time_unix
    """
    try:
        df = pd.read_csv(survey_time_file)
        log_info(f"Loaded {len(df)} survey time records from {survey_time_file}")
        return df
    except FileNotFoundError:
        log_warning(f"Error: Survey time file {survey_time_file} not found")
        return pd.DataFrame()
    except Exception as e:
        log_error(f"Error reading survey time file {survey_time_file}: {e}")
        return pd.DataFrame()



def _process_single_participant_auto(pid, surveys, time_ranges, direction, config):
    """
    Process all surveys for a single participant in auto mode.
    Designed to be called either sequentially or in a ProcessPoolExecutor.

    Args:
        pid (str): Participant ID
        surveys (list): List of survey dicts with 'survey_id' and 'survey_time_unix'
        time_ranges (list): Time ranges to generate
        direction (str): "backward" or "forward"
        config (dict): Full CONFIG dictionary (passed explicitly for subprocess safety)

    Returns:
        tuple: (pid, participant_result, processed_surveys, skipped_surveys)
    """
    participant_logger = setup_logging(pid)

    log_summary(f"\n{'='*50}")
    log_summary(f"STARTING participant: {pid}")
    log_summary(f"{'='*50}")
    log_summary(f"Surveys to process: {[s['survey_id'] for s in surveys]}")

    log_info(f"\n{'='*50}")
    log_info(f"Processing participant: {pid}")
    log_info(f"{'='*50}")
    log_info(f"Surveys to process: {[s['survey_id'] for s in surveys]}")

    participant_output_dir = config["output_dir"].format(P_ID=pid)
    os.makedirs(participant_output_dir, exist_ok=True)

    participant_result = {
        'total_surveys': len(surveys),
        'processed_surveys': 0,
        'skipped_surveys': 0,
        'survey_details': [],
        'time_ranges_generated': set(),
        'total_files_created': 0
    }

    processed_surveys = 0
    skipped_surveys = 0

    for survey in surveys:
        survey_id = survey['survey_id']
        survey_time_unix = survey['survey_time_unix']

        survey_dt = pd.to_datetime(survey_time_unix, unit='ms', utc=True).tz_convert(config["timezone"])

        if config.get("align_to_midnight", False):
            original_time = survey_dt.strftime('%Y-%m-%d %H:%M:%S')
            midnight_dt = survey_dt.normalize()
            survey_time_unix = int(midnight_dt.value // 1_000_000)
            survey_dt = midnight_dt
            log_info(f"    Aligned survey time from {original_time} to midnight: {survey_dt.strftime('%Y-%m-%d %H:%M:%S')}", participant_logger)

        if config.get("skip_survey_day", False):
            if direction == "forward":
                original_time = survey_dt.strftime('%Y-%m-%d %H:%M:%S')
                next_day_dt = survey_dt.normalize() + pd.Timedelta(days=1)
                survey_time_unix = int(next_day_dt.value // 1_000_000)
                survey_dt = next_day_dt
                log_info(f"    Skipped survey day: adjusted start from {original_time} to {survey_dt.strftime('%Y-%m-%d %H:%M:%S')}", participant_logger)
            elif not config.get("align_to_midnight", False):
                original_time = survey_dt.strftime('%Y-%m-%d %H:%M:%S')
                midnight_dt = survey_dt.normalize()
                survey_time_unix = int(midnight_dt.value // 1_000_000)
                survey_dt = midnight_dt
                log_info(f"    Skipped survey day: adjusted end from {original_time} to {survey_dt.strftime('%Y-%m-%d %H:%M:%S')}", participant_logger)

        log_summary(f"  Processing {survey_id} (survey time: {survey_dt.strftime('%Y-%m-%d %H:%M:%S')})")
        log_info(f"  Processing {survey_id} (survey time: {survey_dt.strftime('%Y-%m-%d %H:%M:%S')})", participant_logger)

        survey_result = {
            'survey_id': survey_id,
            'survey_date': survey_dt.strftime('%Y-%m-%d'),
            'status': 'failed',
            'time_ranges': {},
            'files_created': 0
        }

        try:
            daily_output_dir = config["daily_output_dir"].format(P_ID=pid)

            log_info(f"    Generating all time ranges directly for survey {survey_id} ({direction})...", participant_logger)

            time_range_descriptions, daily_descriptions, time_range_windows, daily_windows = process_participant_auto(
                pid, survey_time_unix, time_ranges, daily_output_dir, direction, participant_logger
            )

            if not time_range_descriptions:
                log_warning(f"      ⚠️  Failed to generate any time ranges - skipping survey", participant_logger)
                skipped_surveys += 1
                participant_result['skipped_surveys'] += 1
                survey_result['status'] = 'processing_failed'
                participant_result['survey_details'].append(survey_result)
                continue

            result_files = {}
            for time_range, description in time_range_descriptions.items():
                if description.strip():
                    range_output_dir = f"{participant_output_dir}/{time_range}"
                    os.makedirs(range_output_dir, exist_ok=True)

                    output_file = f"{range_output_dir}/{pid}_{survey_id}_{time_range}_output.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(clean_text(description))

                    json_file = generate_json_output(
                        time_range_windows[time_range], pid, survey_id, time_range, range_output_dir
                    )

                    if json_file:
                        log_info(f"        ✓ Generated JSON: {os.path.basename(json_file)}", participant_logger)

                    window_count = description.count("Window ")
                    result_files[time_range] = (output_file, window_count)

            for day_date, day_content in daily_descriptions.items():
                daily_file = os.path.join(daily_output_dir, f"day_{day_date}.txt")
                with open(daily_file, 'w', encoding='utf-8') as f:
                    f.write(clean_text(day_content))

                if day_date in daily_windows and daily_windows[day_date]:
                    daily_json = generate_json_output(
                        daily_windows[day_date], pid, output_dir=daily_output_dir,
                        filename=f"day_{day_date}.json"
                    )
                    if daily_json:
                        log_info(f"        ✓ Generated daily JSON: day_{day_date}.json", participant_logger)

            if not result_files:
                log_warning(f"      ⚠️  Warning: No time ranges generated for {survey_id} - skipping survey", participant_logger)
                skipped_surveys += 1
                participant_result['skipped_surveys'] += 1
                survey_result['status'] = 'no_time_ranges'
                participant_result['survey_details'].append(survey_result)
                continue

            window_counts = {time_range: window_count for time_range, (_, window_count) in result_files.items()}

            if all(count == 0 for count in window_counts.values()):
                log_warning(f"      ⚠️  Warning: All time ranges for {survey_id} have zero windows - skipping survey", participant_logger)
                skipped_surveys += 1
                participant_result['skipped_surveys'] += 1
                survey_result['status'] = 'zero_windows'
                participant_result['survey_details'].append(survey_result)
                continue

            window_count_groups = {}
            for time_range, window_count in window_counts.items():
                if window_count not in window_count_groups:
                    window_count_groups[window_count] = []
                window_count_groups[window_count].append(time_range)

            removed_ranges = []
            for window_count, ranges_with_same_count in window_count_groups.items():
                if len(ranges_with_same_count) > 1:
                    ranges_with_same_count.sort(key=parse_time_range_duration)

                    kept_range = ranges_with_same_count[0]

                    for time_range in ranges_with_same_count[1:]:
                        output_file, _ = result_files[time_range]
                        if os.path.exists(output_file):
                            os.remove(output_file)
                            json_file = output_file.replace('_output.txt', '.json')
                            if os.path.exists(json_file):
                                os.remove(json_file)
                            output_dir = os.path.dirname(output_file)
                            if os.path.isdir(output_dir) and not os.listdir(output_dir):
                                os.rmdir(output_dir)
                            removed_ranges.append(time_range)

            files_created = 0
            for time_range, (output_file, window_count) in result_files.items():
                if time_range not in removed_ranges:
                    survey_result['time_ranges'][time_range] = window_count
                    participant_result['time_ranges_generated'].add(time_range)
                    files_created += 1
                    log_info(f"        ✓ {time_range}: {window_count} windows", participant_logger)
                else:
                    log_info(f"        - {time_range}: {window_count} windows (removed - duplicate content)", participant_logger)

            survey_result['files_created'] = files_created
            survey_result['status'] = 'success'
            participant_result['total_files_created'] += files_created
            participant_result['processed_surveys'] += 1
            processed_surveys += 1

        except Exception as e:
            log_warning(f"    ⚠️  Warning: Failed to process {survey_id}: {e} - skipping survey", participant_logger)
            skipped_surveys += 1
            participant_result['skipped_surveys'] += 1
            survey_result['status'] = f'exception: {str(e)}'

        participant_result['survey_details'].append(survey_result)

    log_info(f"  ✓ Completed participant {pid}", participant_logger)

    log_summary(f"FINISHED participant: {pid}")
    log_summary(f"{'='*50}")

    # Clean up empty participant directories if no files were created
    if participant_result['total_files_created'] == 0:
        if os.path.isdir(participant_output_dir):
            for subdir in os.listdir(participant_output_dir):
                subdir_path = os.path.join(participant_output_dir, subdir)
                if os.path.isdir(subdir_path) and not os.listdir(subdir_path):
                    os.rmdir(subdir_path)
            if not os.listdir(participant_output_dir):
                os.rmdir(participant_output_dir)
                log_info(f"  Removed empty output directory for {pid}", participant_logger)

        daily_output_dir = config["daily_output_dir"].format(P_ID=pid)
        if os.path.isdir(daily_output_dir) and not os.listdir(daily_output_dir):
            os.rmdir(daily_output_dir)
            log_info(f"  Removed empty daily output directory for {pid}", participant_logger)

    return (pid, participant_result, processed_surveys, skipped_surveys)


def process_auto_mode():
    """
    Optimized auto mode processing - processes each survey time range individually to avoid loading unnecessary data.
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Set up logging for auto mode
    logger = setup_logging()
    
    log_info(f"\n{'='*60}")
    log_info("PROCESSING IN AUTO MODE")
    log_info(f"{'='*60}")
    
    # Load survey time data
    survey_time_file = CONFIG["survey_time_file"]
    survey_df = load_survey_time_data(survey_time_file)
    
    if survey_df.empty:
        log_warning("No survey time data available. Skipping auto mode processing.")
        return False
    
    # Get time ranges and direction from config
    # Support either explicit list (time_ranges) or start/end shorthand (time_range_start + time_range_end)
    if "time_range_start" in CONFIG and "time_range_end" in CONFIG:
        time_ranges = expand_time_ranges(CONFIG["time_range_start"], CONFIG["time_range_end"])
        log_info(f"Expanded time_range_start='{CONFIG['time_range_start']}' to time_range_end='{CONFIG['time_range_end']}' → {len(time_ranges)} ranges")
    elif "time_ranges" in CONFIG:
        time_ranges = CONFIG["time_ranges"]
    else:
        log_error("Config must specify either 'time_ranges' or both 'time_range_start' and 'time_range_end'.")
        return False
    direction = CONFIG.get("direction", "backward")  # Default to backward for compatibility
    log_info(f"Time ranges to process: {time_ranges}")
    log_info(f"Direction: {direction}")
    
    # Group surveys by participant for efficient processing
    participant_surveys = {}
    for _, survey_row in survey_df.iterrows():
        pid = survey_row['pid']
        survey_id = survey_row['survey_id']
        survey_time_unix = int(survey_row['survey_time_unix'])
        
        if pid not in participant_surveys:
            participant_surveys[pid] = []
        participant_surveys[pid].append({
            'survey_id': survey_id,
            'survey_time_unix': survey_time_unix
        })
    
    log_info(f"Found {len(participant_surveys)} unique participants")
    for pid, surveys in participant_surveys.items():
        # Show survey times in local timezone for better readability
        survey_info = []
        for s in surveys:
            survey_dt = pd.to_datetime(s['survey_time_unix'], unit='ms', utc=True).tz_convert(CONFIG["timezone"])
            survey_info.append(f"{s['survey_id']}({survey_dt.strftime('%m-%d')})")
        log_info(f"  {pid}: {len(surveys)} surveys ({', '.join(survey_info)})")
    
    processed_participants = 0
    processed_surveys = 0
    skipped_surveys = 0

    # Collect participant-level results for summary
    participant_results = {}

    # Process each participant's surveys individually by time range
    num_workers = CONFIG.get("num_workers", 1)

    if num_workers > 1:
        log_summary(f"Processing {len(participant_surveys)} participants in parallel (num_workers={num_workers})")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_participant_auto, pid, surveys, time_ranges, direction, CONFIG
                ): pid
                for pid, surveys in participant_surveys.items()
            }
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    result_pid, result_data, p_processed, p_skipped = future.result()
                    participant_results[result_pid] = result_data
                    processed_participants += 1
                    processed_surveys += p_processed
                    skipped_surveys += p_skipped
                except Exception as e:
                    log_error(f"Participant {pid} failed with exception: {e}")
                    participant_results[pid] = {
                        'total_surveys': len(participant_surveys[pid]),
                        'processed_surveys': 0,
                        'skipped_surveys': len(participant_surveys[pid]),
                        'survey_details': [],
                        'time_ranges_generated': set(),
                        'total_files_created': 0
                    }
                    skipped_surveys += len(participant_surveys[pid])
    else:
        for pid, surveys in participant_surveys.items():
            result_pid, result_data, p_processed, p_skipped = _process_single_participant_auto(
                pid, surveys, time_ranges, direction, CONFIG
            )
            participant_results[result_pid] = result_data
            processed_participants += 1
            processed_surveys += p_processed
            skipped_surveys += p_skipped

    # Participant-focused summary - prepare content for both print and file
    summary_lines = []
    summary_lines.append("="*60)
    summary_lines.append("AUTO MODE PROCESSING SUMMARY BY PARTICIPANT")
    summary_lines.append("="*60)
    
    for pid, results in participant_results.items():
        summary_lines.append(f"\n{pid}:")
        summary_lines.append(f"  Total surveys: {results['total_surveys']}")
        summary_lines.append(f"  Successfully processed: {results['processed_surveys']}")
        summary_lines.append(f"  Skipped/failed: {results['skipped_surveys']}")
        summary_lines.append(f"  Time ranges generated: {', '.join(sorted(results['time_ranges_generated']))}")
        summary_lines.append(f"  Total output files: {results['total_files_created']}")
        
        # Show survey details
        successful_surveys = [s for s in results['survey_details'] if s['status'] == 'success']
        if successful_surveys:
            summary_lines.append(f"  Successful surveys:")
            for survey in successful_surveys:
                time_range_info = ', '.join([f"{tr}({count}w)" for tr, count in survey['time_ranges'].items()])
                summary_lines.append(f"    ✓ {survey['survey_id']} ({survey['survey_date']}): {time_range_info}")
        
        failed_surveys = [s for s in results['survey_details'] if s['status'] != 'success']
        if failed_surveys:
            summary_lines.append(f"  Failed surveys:")
            for survey in failed_surveys:
                summary_lines.append(f"    ✗ {survey['survey_id']} ({survey['survey_date']}): {survey['status']}")
    
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("OVERALL SUMMARY")
    summary_lines.append("="*60)
    summary_lines.append(f"Total participants: {len(participant_surveys)}")
    summary_lines.append(f"Participants processed: {processed_participants}")
    summary_lines.append(f"Total surveys processed: {processed_surveys}")
    summary_lines.append(f"Total surveys skipped: {skipped_surveys}")
    summary_lines.append(f"Directory structure: description/{{P_ID}}/{{time_range}}/{{P_ID}}_{{survey_id}}_{{time_range}}_output.txt")
    summary_lines.append(f"JSON output: description/{{P_ID}}/{{time_range}}/{{P_ID}}_{{survey_id}}_{{time_range}}.json")
    summary_lines.append(f"Optimization applied: Only smallest time range kept for duplicate window counts")
    
    # Print summary to console (keep this visible)
    log_summary(f"\n{chr(10).join(summary_lines)}")
    
    # Also log the summary
    log_info(f"\n{chr(10).join(summary_lines)}", logger)
    
    # Save summary to file
    try:
        # Use the base output directory (without P_ID formatting)
        base_output_dir = CONFIG["output_dir"].replace("/{P_ID}", "")
        summary_file = os.path.join(base_output_dir, "processing_summary.txt")
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Add timestamp to the summary
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Processing completed at: {timestamp}\n")
            f.write(f"Survey time file: {CONFIG['survey_time_file']}\n")
            if "time_range_start" in CONFIG and "time_range_end" in CONFIG:
                f.write(f"Time ranges: {CONFIG['time_range_start']} to {CONFIG['time_range_end']}\n\n")
            else:
                f.write(f"Time ranges: {CONFIG.get('time_ranges', [])}\n\n")
            f.write('\n'.join(summary_lines))
        
        log_info(f"\n✓ Summary saved to: {summary_file}", logger)
        
    except Exception as e:
        log_warning(f"\n⚠️  Warning: Could not save summary to file: {e}", logger)
    
    return processed_participants > 0

def process_participant_auto(pid, survey_time_unix, time_ranges, daily_output_dir, direction="backward", logger=None):
    """
    Process participant data for auto mode and generate all time ranges directly.
    This function handles sensor data processing and generates auto mode outputs with multiple time ranges.

    Args:
        pid (str): Participant ID to process
        survey_time_unix (int): Reference time in unix milliseconds
        time_ranges (list): List of time ranges to generate (e.g., ['7d', '3d', '24h'])
        daily_output_dir (str): Daily output directory
        direction (str): "backward" (survey_time is END) or "forward" (survey_time is START)
        logger: Logger instance for detailed logging

    Returns:
        tuple: (time_range_descriptions, daily_descriptions, time_range_windows, daily_windows)
               time_range_descriptions: dict mapping time_range -> description text
               daily_descriptions: dict mapping day -> description text
               time_range_windows: dict mapping time_range -> list of window data
               daily_windows: dict mapping day -> list of window data dicts
    """
    from collections import defaultdict

    # Calculate the longest time range to determine data loading boundaries
    longest_range_duration_ms = max(parse_time_range_duration(tr) for tr in time_ranges)

    # Calculate start/end timestamps based on direction
    if direction == "forward":
        # Forward: survey_time is START, load data until survey_time + longest_range
        start_timestamp = survey_time_unix
        end_timestamp = survey_time_unix + longest_range_duration_ms
    else:
        # Backward (default): survey_time is END, load data from survey_time - longest_range
        start_timestamp = survey_time_unix - longest_range_duration_ms
        end_timestamp = survey_time_unix

    log_info(f"Loading data for direct generation ({direction}) from {start_timestamp} to {end_timestamp}", logger)
    
    # Set up directories
    os.makedirs(daily_output_dir, exist_ok=True)
    
    # Load sensor data (similar to process_participant_manual_core but with direct generation)
    input_directory = CONFIG["input_directory"].format(P_ID=pid)
    
    # Load JSON files based on sensors defined in config
    sensors = CONFIG.get("sensors", [])
    jsonl_files = []
    for sensor in sensors:
        jsonl_file_path = os.path.join(input_directory, f"{sensor}.jsonl")
        jsonl_files.append((jsonl_file_path, sensor))
    
    # Load session data
    session_file_path = CONFIG.get("session_file", f"step1_data/{pid}/sessions.jsonl")
    sessions = load_session_data(session_file_path, logger)
    if not sessions:
        default_session_file = f"step1_data/{pid}/sessions.jsonl"
        sessions = load_session_data(default_session_file, logger)
    
    # Collect all narratives
    all_narratives = []
    location_data = []
    wifi_sensor_data = {}
    
    for jsonl_file, sensor_name in jsonl_files:
        sensor_data = get_sensor_data(sensor_name, start_timestamp, end_timestamp, input_directory, logger)
        if not sensor_data:
            continue
            
        # Convert to DataFrame for timestamp processing, then back to list of dictionaries
        df = pd.DataFrame(sensor_data)
        df = convert_timestamp_column(df, CONFIG["timezone"])
        sensor_data = df.to_dict('records')
        
        if sensor_name == "locations":
            location_data = sensor_data
            continue
            
        # Handle WiFi sensors specially for combined processing
        if sensor_name in ["wifi", "sensor_wifi"]:
            wifi_sensor_data[sensor_name] = sensor_data
            continue
            
        # Generate integrated descriptions for each sensor
        narratives = generate_integrated_description(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions, pid)
        if narratives:
            all_narratives.extend(narratives)
    
    # Process WiFi sensors together
    if wifi_sensor_data:
        sensor_wifi_data = wifi_sensor_data.get("sensor_wifi", [])
        wifi_data = wifi_sensor_data.get("wifi", [])
        combined_wifi_narratives = generate_wifi_combined_description(
            sensor_wifi_data, wifi_data, start_timestamp, end_timestamp, sessions
        )
        if combined_wifi_narratives:
            all_narratives.extend(combined_wifi_narratives)
    
    # Process locations with clustering (using the same logic as manual mode)
    if location_data:
        log_info(f"Processing location data with clustering for auto mode...", logger)
        location_narratives = process_location_data_with_clustering(
            location_data, start_timestamp, end_timestamp, sessions, pid, logger
        )
        if location_narratives:
            all_narratives.extend(location_narratives)
            log_info(f"Added {len(location_narratives)} location narratives to auto mode output", logger)
    
    # Use the new direct generation function
    time_range_descriptions, daily_descriptions, time_range_windows, daily_windows = generate_all_outputs_auto(
        all_narratives, survey_time_unix, time_ranges, CONFIG["timezone"], direction
    )

    log_info(f"Generated {len(time_range_descriptions)} time ranges and {len(daily_descriptions)} daily files ({direction})", logger)

    return time_range_descriptions, daily_descriptions, time_range_windows, daily_windows


def process_participant_manual(pid):
    """
    Process sensor data for a single participant in manual mode.
    Uses CONFIG values for all parameters.
    
    Args:
        pid (str): Participant ID to process
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    log_summary(f"\n{'='*60}")
    log_summary(f"Processing participant: {pid} (MANUAL MODE)")
    log_summary(f"{'='*60}")
    
    # Set up participant-specific logging (same as auto mode)
    participant_logger = setup_logging(pid)
    
    participant_start_time = CONFIG["START_TIME"]
    participant_end_time = CONFIG["END_TIME"]
    output_file = CONFIG["output_file"].format(
        P_ID=pid,
        START_TIME=CONFIG["START_TIME"].replace(" ", "_").replace(":", "-"),
        END_TIME=CONFIG["END_TIME"].replace(" ", "_").replace(":", "-")
    )
    daily_output_dir = CONFIG["daily_output_dir"].format(P_ID=pid)
    
    log_info(f"Manual mode: Using CONFIG time range: {participant_start_time} to {participant_end_time}", participant_logger)
    log_info(f"Manual mode: Using CONFIG output paths", participant_logger)
    
    return process_participant_manual_core(pid, participant_start_time, participant_end_time, output_file, daily_output_dir, participant_logger)

def load_app_name_mapping():
    """
    Load app package name to application name mapping from resources folder.
    
    NOTE: The mapping is now loaded globally at module import time in application_name_list.
    This function is kept for backward compatibility and testing purposes.
    
    PARALLEL PROCESSING SAFE: This function does not modify global state.
    
    Returns:
        dict: Package name to application name mapping
    """
    app_package_file_path = package_to_app_map
    app_name_mapping = {}
    
    try:
        with open(app_package_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    app_data = json.loads(line)
                    package_name = app_data.get("package_name")
                    application_name = app_data.get("application_name")
                    if package_name and application_name:
                        app_name_mapping[package_name] = application_name
        log_info(f"Loaded {len(app_name_mapping)} app package mappings")
    except FileNotFoundError:
        log_warning(f"Warning: App package pairs file {app_package_file_path} not found")
        app_name_mapping = {}
    except Exception as e:
        log_error(f"Error reading app package pairs file: {e}")
        app_name_mapping = {}
    
    return app_name_mapping

def process_location_data_with_clustering(location_data, start_timestamp, end_timestamp, sessions, pid, logger=None):
    """
    Process location data through clustering and reverse geocoding.
    Extracted from manual mode to be reused in auto mode.
    
    Args:
        location_data: Raw location sensor data
        start_timestamp: Start timestamp in milliseconds
        end_timestamp: End timestamp in milliseconds
        sessions: Session data
        pid: Participant ID
        logger: Logger instance
        
    Returns:
        list: Location narrative dictionaries
    """
    # Generate distance matrix from location_data
    log_info("Locations: Generate distance matrix...", logger)
    
    # Build points with original indices to maintain alignment
    points_with_indices = []
    for idx, record in enumerate(location_data):
        # Check if required keys exist in the dictionary
        if "double_latitude" in record and "double_longitude" in record and "double_speed" in record:
            points_with_indices.append((
                idx,  # Original index in location_data
                float(record["double_latitude"]), # Latitude
                float(record["double_longitude"]), # Longitude
                record["datetime"], # Parsed datetime object
                float(record["double_speed"]) # Speed
            ))
    
    if not points_with_indices:
        log_warning("No valid location points found! Skipping location processing.", logger)
        return []
    
    # Extract data for clustering (only coordinates)
    indices = [item[0] for item in points_with_indices]
    coordinates = np.array([[item[1], item[2]] for item in points_with_indices])  # Only lat, lon
    datetimes = [item[3] for item in points_with_indices]
    speeds = [item[4] for item in points_with_indices]
    
    # Only proceed with clustering if we have coordinates to process
    if len(coordinates) > 0:
        log_info(f"Data size is {len(coordinates)}", logger)
        # Determine if we should use daily clustering or all data
        start_dt = pd.to_datetime(start_timestamp, unit='ms')
        end_dt = pd.to_datetime(end_timestamp, unit='ms')
        time_span_hours = (end_dt - start_dt).total_seconds() / 3600
        
        log_info(f"Time span: {time_span_hours:.1f} hours", logger)
        
        if time_span_hours < 48:
            log_info("Time span is less than 48 hours - using all location data for clustering", logger)
            use_daily_clustering = False
        else:
            log_info("Time span is 48 hours or more - using daily clustering based on night_time_end", logger)
            use_daily_clustering = True
        
        # Perform DBSCAN clustering using haversine distance
        log_info("Locations: Performing DBSCAN clustering", logger)
        
        try:
            if len(coordinates) > 0: # Ensure there are points to process
                # Perform DBSCAN clustering using shared function
                cluster_labels, daily_clusters = perform_dbscan_clustering(
                    coordinates, datetimes, eps, min_samples, use_daily_clustering, 
                    start_dt, end_dt, night_time_end, logger
                )
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # Exclude noise
                log_info(f"DBSCAN found {n_clusters} clusters (excluding noise)", logger)
                log_info(f"Noise points: {list(cluster_labels).count(-1)} out of {len(cluster_labels)}", logger)
                
                # Process clustering results using shared function
                (cluster, clustered_coordinates, clustered_labels, clustered_datetimes, 
                 clustered_indices, clustered_speeds, home_group_center) = process_clustering_results(
                    coordinates, cluster_labels, datetimes, indices, speeds,
                    use_daily_clustering, daily_clusters, night_time_start, night_time_end, logger
                )
                
                # Update variables to use clustered data for the rest of the algorithm
                coordinates = clustered_coordinates
                cluster_labels = clustered_labels
                datetimes = clustered_datetimes
                indices = clustered_indices
                speeds = clustered_speeds

        except NameError as e:
            log_error(f"Error: Variable not initialized - {e}", logger)
            log_error("Skipping location clustering due to error.", logger)
            return []
        except ValueError as e:
            log_error(f"Error: {e}", logger)
            log_error("Skipping location clustering due to error.", logger)
            return []
        except Exception as e:
            log_error(f"Unexpected error during clustering: {e}", logger)
            log_error("Skipping location clustering due to error.", logger)
            return []
    else:
        log_info("No location data available for clustering", logger)
        return []

    # If Google Maps is enabled and API key is provided, perform reverse geocoding for all places
    reverse_geocoding_performed = False
    if USE_GOOGLE_MAP and GOOGLE_MAP_KEY and cluster:
        gmaps = googlemaps.Client(key=GOOGLE_MAP_KEY)
        # a list to store all reverse geocoding results
        reverse_geocode_results = []
        try:
            for idx, cluster_data in enumerate(cluster):
                # Extract cluster data (must have exactly 6 elements)
                if len(cluster_data) == 6:
                    cluster_id, center_lat, center_lon, num_points, place, distance_from_home = cluster_data
                else:
                    log_warning(f"Warning: Cluster data at index {idx} has {len(cluster_data)} elements, expected 6. Skipping.", logger)
                    continue
                # Perform reverse geocoding for all places (both home and unknown)
                reverse_geocode_data = None
                try:
                    reverse_geocode_data = gmaps.reverse_geocode((center_lat, center_lon), enable_address_descriptor=True)
                except (gexceptions.ApiError, gexceptions.HTTPError, gexceptions.Timeout, gexceptions.TransportError) as e:
                    log_error(f"Error during reverse geocoding for ({center_lat}, {center_lon}): {e}", logger)
                    continue
                except Exception as e:
                    log_error(f"Unexpected error during reverse geocoding for ({center_lat}, {center_lon}): {e}", logger)
                    continue

                if not reverse_geocode_data or reverse_geocode_data.get("status") != "OK":
                    log_warning(f"No valid address returned for ({center_lat}, {center_lon}).", logger)
                    continue
                
                # Process the reverse geocoding data
                reverse_geocode_results.append(reverse_geocode_data)
                data = reverse_geocode_data["results"][0]
                formatted_address = data.get("formatted_address", "")

                # enum→phrase maps
                rel_phrases = {
                    "NEAR": "near",
                    "WITHIN": "within",
                    "BESIDE": "beside",
                    "ACROSS_THE_ROAD": "across the road from",
                    "DOWN_THE_ROAD": "down the road from",
                    "AROUND_THE_CORNER": "around the corner from",
                    "BEHIND": "behind",
                }
                cont_phrases = {
                    "NEAR": "near",
                    "WITHIN": "within",
                    "OUTSKIRTS": "on the outskirts of",
                }

                desc = reverse_geocode_data.get("address_descriptor", {})
                landmarks = desc.get("landmarks", [])
                areas     = desc.get("areas", [])

                # build the "area" piece - display all areas in original order
                area_parts = []
                if areas:
                    for ar in areas:
                        name = ar["display_name"]["text"]
                        cont = cont_phrases.get(ar["containment"], ar["containment"].lower())
                        area_parts.append(f"{cont} {name}")
                    ar_part = "Areas: " + ", ".join(area_parts)
                else:
                    ar_part = ""

                # build the "landmark" piece (only the first one)
                if landmarks:
                    lm = landmarks[0]
                    name = lm["display_name"]["text"]
                    rel  = rel_phrases.get(lm["spatial_relationship"], lm["spatial_relationship"].lower())
                    dist = lm["straight_line_distance_meters"]
                    lm_part = f"Landmarks: {rel} {name} ({dist:.1f} m away)"
                else:
                    lm_part = ""

                # combine area and landmark descriptions, do not show empty
                desc_parts = []
                if ar_part:
                    desc_parts.append(ar_part)
                if lm_part:
                    desc_parts.append(lm_part)
                base_desc = ". ".join(desc_parts)

                # Use formatted address directly
                full_address = formatted_address

                # append the full address information
                if base_desc:
                    full_desc = f"{base_desc}. Address: {full_address}"
                else:
                    full_desc = full_address

                # prefix "home." instead of "home,"
                if place == "home":
                    updated_place = f"home. {full_desc}"
                else:
                    updated_place = full_desc

                # Update cluster data while preserving distance information
                cluster[idx] = (
                    cluster_id,
                    center_lat,
                    center_lon,
                    num_points,
                    updated_place,
                    distance_from_home
                )

        except Exception as e:
            log_error(f"Error in Google Maps API request: {e}")
        
        # Save reverse geocoding results to the configured directory
        # Parse the timestamps for file naming
        start_date = pd.to_datetime(start_timestamp, unit='ms').strftime('%Y%m%d%H%M%S')
        end_date = pd.to_datetime(end_timestamp, unit='ms').strftime('%Y%m%d%H%M%S')
        reverse_geocoding_output_dir = CONFIG["reverse_geocoding_output_dir"].format(P_ID=pid)
        os.makedirs(reverse_geocoding_output_dir, exist_ok=True)
        
        filename = os.path.join(reverse_geocoding_output_dir, f"{pid}_{start_date}_{end_date}_reverse_geocode_results.json")
        
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(reverse_geocode_results, file, indent=2, ensure_ascii=False, default=str)
        
        # Create cluster ID to query item mapping
        cluster_mapping = {}
        for idx, cluster_data in enumerate(cluster):
            if len(cluster_data) == 6:
                cluster_id, center_lat, center_lon, num_points, place, distance_from_home = cluster_data
                if idx < len(reverse_geocode_results):
                    # Convert numpy int64 to Python int for JSON serialization
                    cluster_id_int = int(cluster_id)
                    cluster_mapping[cluster_id_int] = {
                        "query_coordinates": (center_lat, center_lon),
                        "reverse_geocode_result": reverse_geocode_results[idx],
                        "place_name": place,
                        "num_points": int(num_points),  # Also convert to ensure compatibility
                        "distance_from_home_meters": float(distance_from_home)  # Convert to float
                    }
        
        # Save cluster mapping to JSON file
        mapping_filename = os.path.join(reverse_geocoding_output_dir, f"{pid}_{start_date}_{end_date}_cluster_mapping.json")
        with open(mapping_filename, "w", encoding='utf-8') as file:
            json.dump(cluster_mapping, file, indent=2, ensure_ascii=False, default=str)
        
        log_info(f"Saved reverse geocoding results to: {filename}", logger)
        log_info(f"Saved cluster mapping to: {mapping_filename}", logger)
        reverse_geocoding_performed = True

    # Display the updated cluster information only if reverse geocoding was performed successfully
    if len(coordinates) > 0 and cluster and reverse_geocoding_performed:
        log_info("Cluster Centers (Updated):", logger)
        for cluster_data in cluster:
            if len(cluster_data) == 6:  # Must have exactly 6 elements
                cluster_id, center_lat, center_lon, num_points, place, distance_from_home = cluster_data
                log_info(f"Cluster {cluster_id}: Center Lat = {center_lat:.6f}, Center Lon = {center_lon:.6f}, N = {num_points}, Place = {place}, Distance = {distance_from_home:.1f}m", logger)

    # Generate descriptions for locations based on clustering results
    # Reconstruct points data for describe_locations_integrated function
    # Create all_points structure: [latitude, longitude, datetime_string, speed]
    if len(coordinates) > 0 and len(cluster_labels) > 0 and cluster:
        # Reconstruct all_points with the correct structure for describe_locations_integrated
        reconstructed_points = []
        for i in range(len(coordinates)):
            lat, lon = coordinates[i]
            dt_str = datetimes[i] if isinstance(datetimes[i], str) else str(datetimes[i])
            speed = speeds[i]
            reconstructed_points.append([lat, lon, dt_str, speed])
        
        reconstructed_points = np.array(reconstructed_points, dtype=object)
        
        # Generate integrated location descriptions
        location_narratives = describe_locations_integrated(
            reconstructed_points, cluster_labels, cluster, 
            start_timestamp, end_timestamp, sessions
        )
        
        return location_narratives
    
    return []


def process_participant_manual_core(pid, participant_start_time, participant_end_time, output_file, daily_output_dir, logger=None):
    """
    Core processing logic for manual mode participant processing.
    This function handles sensor data processing and generates manual mode outputs.
    
    Args:
        pid (str): Participant ID to process
        participant_start_time (str): Start time string
        participant_end_time (str): End time string
        output_file (str): Output file path
        daily_output_dir (str): Daily output directory
        logger: Logger instance for detailed logging
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    
    # Load device IDs for this participant
    device_ids = get_device_ids_for_participant(pid)
    if not device_ids:
        log_warning(f"Warning: No device IDs found for participant {pid}. Skipping.", logger)
        return False
    
    # Set up participant-specific paths
    input_directory = CONFIG["input_directory"].format(P_ID=pid)
    reverse_geocoding_output_dir = CONFIG["reverse_geocoding_output_dir"].format(P_ID=pid)
    
    # Ensure output directories exist
    os.makedirs(daily_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(reverse_geocoding_output_dir, exist_ok=True)
    
    jsonl_files = [(f"{sensor}.jsonl", sensor) for sensor in sensors]
    
    # Initialize participant-specific variables
    sensor_narratives = {}
    
    # Convert start and end times to timestamps
    start_timestamp = convert_timestring_to_timestamp(participant_start_time, CONFIG["timezone"])
    end_timestamp = convert_timestring_to_timestamp(participant_end_time, CONFIG["timezone"])

    # Load session data for accurate active time calculation
    session_file_path = CONFIG.get("session_data_file", "").format(P_ID=pid)
    sessions = load_session_data(session_file_path, logger)
    
    # If config session file not found, try the default location
    if not sessions:
        default_session_file = f"step1_data/{pid}/sessions.jsonl"
        log_info(f"Trying default session file location: {default_session_file}", logger)
        sessions = load_session_data(default_session_file, logger)
    
    log_info(f"Loaded {len(sessions)} session records", logger)

    log_info(f"Keep data within time range: {participant_start_time} to {participant_end_time}", logger)
    
    #store location data in a list
    location_data = []
    
    # Store WiFi sensor data separately for combined processing
    wifi_sensor_data = {}

    for jsonl_file, sensor_name in jsonl_files:      
        sensor_data = get_sensor_data(sensor_name, start_timestamp, end_timestamp, input_directory, logger)
        #If sensor_data is not found, skip the sensor
        if not sensor_data:
            continue

        # Convert to DataFrame for timestamp processing, then back to list of dictionaries
        df = pd.DataFrame(sensor_data)
        df = convert_timestamp_column(df, CONFIG["timezone"])
        sensor_data = df.to_dict('records')

        if sensor_name == "locations":
            location_data = sensor_data
            continue
        
        # Handle WiFi and network sensors specially for combined processing
        if sensor_name in ["wifi", "sensor_wifi"]:
            wifi_sensor_data[sensor_name] = sensor_data
            log_info(f"Collected {sensor_name} data: {len(sensor_data)} records", logger)
            continue
        
        # Initialize sensor-specific narrative list
        if sensor_name not in sensor_narratives:
            sensor_narratives[sensor_name] = []

        # Generate integrated descriptions for each sensor
        narratives = generate_integrated_description(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions, pid)
        if narratives:
            sensor_narratives[sensor_name].extend(narratives)

    # Process WiFi sensors together if we have any type
    if wifi_sensor_data:
        sensor_wifi_data = wifi_sensor_data.get("sensor_wifi", [])
        wifi_data = wifi_sensor_data.get("wifi", [])
        
        log_info(f"Processing combined WiFi analysis:", logger)
        log_info(f"  - sensor_wifi: {len(sensor_wifi_data)} records", logger)
        log_info(f"  - wifi: {len(wifi_data)} records", logger)
        
        # Generate combined WiFi narratives
        combined_wifi_narratives = generate_wifi_combined_description(
            sensor_wifi_data, wifi_data, start_timestamp, end_timestamp, sessions
        )
        
        if combined_wifi_narratives:
            # Store combined WiFi narratives under a unified key
            sensor_narratives["wifi"] = combined_wifi_narratives
            log_info(f"Generated {len(combined_wifi_narratives)} combined WiFi narratives", logger)

    #summary len of each sensor narrative list
    for sensor_name, narrative_list in sensor_narratives.items():
        log_info(f"Sensor: {sensor_name}, Number of integrated descriptions: {len(narrative_list)}", logger)

    # Process locations with clustering using the shared function
    if location_data:
        log_info(f"Processing location data with clustering for manual mode...", logger)
        location_narratives = process_location_data_with_clustering(
            location_data, start_timestamp, end_timestamp, sessions, pid, logger
        )
        if location_narratives:
            sensor_narratives["locations"] = location_narratives
            log_info(f"Added {len(location_narratives)} location narratives to manual mode", logger)
    else:
        log_info("No location data available for processing", logger)

    all_narratives = []
    
    #load all narrative lists
    for sensor_name, sensor_narrative_list in sensor_narratives.items():
        for narrative_dict in sensor_narrative_list:
            all_narratives.append(narrative_dict)

    # Generate main description and daily descriptions directly (no temp files)
    log_info("Generating output descriptions directly...", logger)
    
    # Convert timestamps for manual mode boundaries
    start_timestamp = convert_timestring_to_timestamp(participant_start_time, CONFIG["timezone"])
    end_timestamp = convert_timestring_to_timestamp(participant_end_time, CONFIG["timezone"])
    
    # Use new direct generation approach
    main_description, daily_descriptions, all_windows, daily_windows = generate_all_outputs_manual(
        all_narratives, start_timestamp, end_timestamp, CONFIG["timezone"]
    )
    
    # Write main output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(main_description)
    
    # Generate JSON output for manual mode
    output_dir = os.path.dirname(output_file)
    json_file = generate_json_output(all_windows, pid, output_dir=output_dir)
    if json_file:
        log_info(f"Generated JSON output: {os.path.basename(json_file)}", logger)
    
    # Write daily output files
    output_files = []
    for day_date, day_content in daily_descriptions.items():
        daily_file = os.path.join(daily_output_dir, f"day_{day_date}.txt")
        with open(daily_file, 'w', encoding='utf-8') as f:
            f.write(day_content)
        output_files.append(daily_file)

        # Generate daily JSON output
        if day_date in daily_windows and daily_windows[day_date]:
            daily_json = generate_json_output(
                daily_windows[day_date], pid, output_dir=daily_output_dir,
                filename=f"day_{day_date}.json"
            )
            if daily_json:
                log_info(f"Generated daily JSON: day_{day_date}.json", logger)

    log_info(f"Generated main output, JSON output, and {len(output_files)} daily files in {daily_output_dir}", logger)
    
    log_info(f"Completed processing for participant {pid}", logger)
    return True

def convert_timestring_to_timestamp(timestring, timezone_str="Australia/Melbourne"):
    """
    Convert a timestring to a timestamp float based on the provided timezone.
    """
    try:
        tz = pytz.timezone(timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        log_warning(f"Unknown timezone '{timezone_str}'. Falling back to UTC.")
        tz = pytz.timezone("UTC")
    
    # Parse the timestring as a naive datetime, handling both with and without milliseconds
    try:
        # Try format with milliseconds first
        dt = datetime.strptime(timestring, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        # Fall back to format without milliseconds
        dt = datetime.strptime(timestring, '%Y-%m-%d %H:%M:%S')
    
    # Localize it to the specified timezone
    local_dt = tz.localize(dt)
    
    # Convert to timestamp (seconds since epoch) and then to milliseconds
    timestamp_ms = float(local_dt.timestamp() * 1000)
    
    return timestamp_ms

def format_timestamp_with_conditional_offset(unix_timestamp, timezone_str):
    """Format timestamp with conditional UTC offset for readability"""
    dt = pd.to_datetime(unix_timestamp, unit='ms', utc=True)
    local_time = dt.tz_convert(timezone_str)
    
    # Return both formatted time and offset for conditional display
    time_str = local_time.strftime('%H:%M:%S')
    offset_str = local_time.strftime('%z')
    
    # Format offset with colon (ISO 8601)
    if len(offset_str) == 5:
        offset_str = offset_str[:3] + ':' + offset_str[3:]
    
    return time_str, offset_str

def generate_window_header_conditional_offset(window_start_ts, window_end_ts, window_number, timezone_str):
    """Generate window header with conditional offset display"""
    start_dt = pd.to_datetime(window_start_ts, unit='ms', utc=True).tz_convert(timezone_str)
    end_dt = pd.to_datetime(window_end_ts, unit='ms', utc=True).tz_convert(timezone_str)
    
    # Get time strings and offsets
    start_time, start_offset = format_timestamp_with_conditional_offset(window_start_ts, timezone_str)
    end_time, end_offset = format_timestamp_with_conditional_offset(window_end_ts, timezone_str)
    
    # Generate day info
    if start_dt.date() == end_dt.date():
        day_name = start_dt.strftime('%A')
        day_info = f"Day {start_dt.strftime('%Y-%m-%d')} ({day_name})"
    else:
        start_day_name = start_dt.strftime('%A')
        end_day_name = end_dt.strftime('%A')
        day_info = f"Day {start_dt.strftime('%Y-%m-%d')} ({start_day_name}) to {end_dt.strftime('%Y-%m-%d')} ({end_day_name})"
    
    # Conditional offset display - only show when different
    if start_offset == end_offset:
        time_range = f"{start_time} - {end_time}"  # Clean display
    else:
        time_range = f"{start_time}{start_offset} - {end_time}{end_offset}"  # Show offsets
    
    return f"Window {window_number}\n{day_info}\n{time_range}"

def generate_all_outputs_auto(narratives, survey_time_unix, time_ranges, timezone_str, direction="backward"):
    """
    Generate multiple time ranges AND daily descriptions simultaneously.
    Ultimate efficiency - single pass through narratives.

    Args:
        narratives: List of narrative dictionaries with unix_timestamp
        survey_time_unix: Reference timestamp in milliseconds
        time_ranges: List of time range strings (e.g., ['7d', '3d', '24h'])
        timezone_str: Timezone string for formatting
        direction: "backward" (survey_time is END, default) or "forward" (survey_time is START)

    Returns:
        tuple: (time_range_descriptions, daily_descriptions, time_range_windows, daily_windows)
               time_range_descriptions: dict mapping time_range -> description text
               daily_descriptions: dict mapping day -> description text
               time_range_windows: dict mapping time_range -> list of window data
               daily_windows: dict mapping day -> list of window data dicts
    """
    from collections import defaultdict

    # Pre-calculate time range boundaries based on direction
    time_range_boundaries = {}
    for time_range in time_ranges:
        range_duration_ms = parse_time_range_duration(time_range)
        if direction == "forward":
            # Forward: survey_time is START, boundary is the END of each range
            time_range_boundaries[time_range] = survey_time_unix + range_duration_ms
        else:
            # Backward (default): survey_time is END, boundary is the START of each range
            time_range_boundaries[time_range] = survey_time_unix - range_duration_ms

    # Group narratives by time windows
    window_size_ms = sensor_integration_time_window * 60 * 1000

    # Determine the full processing range based on direction
    if direction == "forward":
        range_start = survey_time_unix
        range_end = max(time_range_boundaries.values())
    else:
        range_start = min(time_range_boundaries.values())
        range_end = survey_time_unix

    window_groups = defaultdict(list)
    for narrative in narratives:
        unix_ts = narrative['unix_timestamp']
        if range_start <= unix_ts <= range_end:
            window_start = ((unix_ts - range_start) // window_size_ms) * window_size_ms + range_start
            window_groups[window_start].append(narrative)

    # Prepare output collections - store window data instead of formatted content
    time_range_windows = {tr: [] for tr in time_ranges}
    daily_windows = defaultdict(list)

    # Process each window once, assign to all applicable outputs
    for window_start in sorted(window_groups.keys()):
        window_end = window_start + window_size_ms
        window_narratives = window_groups[window_start]

        # Generate window content without header (we'll add headers with correct numbers later)
        window_content = format_window_content_without_header(window_narratives)

        if window_content:  # Only process windows with content
            # Store window data (timing + content + narratives) instead of formatted string
            window_data = {
                'start': window_start,
                'end': window_end,
                'content': window_content,
                'narratives': window_narratives
            }

            # Determine day for this window
            window_dt = pd.to_datetime(window_start, unit='ms', utc=True).tz_convert(timezone_str)
            day_str = window_dt.strftime('%Y-%m-%d')
            daily_windows[day_str].append(window_data)

            # Assign to applicable time ranges based on direction
            for time_range, boundary in time_range_boundaries.items():
                if direction == "forward":
                    # Forward: include window if window_start < boundary (end of range)
                    if window_start < boundary:
                        time_range_windows[time_range].append(window_data)
                else:
                    # Backward: include window if window_start >= boundary (start of range)
                    if boundary <= window_start:
                        time_range_windows[time_range].append(window_data)
    
    # Format final outputs with relative window numbering
    time_range_descriptions = {}
    for time_range, window_data_list in time_range_windows.items():
        if window_data_list:
            # Generate complete windows with relative numbering (1, 2, 3, ...)
            formatted_windows = []
            for relative_window_num, window_data in enumerate(window_data_list, 1):
                # Generate header with relative window number
                header = generate_window_header_conditional_offset(
                    window_data['start'], window_data['end'], relative_window_num, timezone_str
                )
                
                # Combine header and content
                complete_window = f"{header}\n{window_data['content']}"
                formatted_windows.append(complete_window)
            
            time_range_descriptions[time_range] = "\n\n".join(formatted_windows)
        else:
            time_range_descriptions[time_range] = ""
    
    # Format daily descriptions with relative window numbering within each day
    daily_descriptions = {}
    for day, window_data_list in daily_windows.items():
        if window_data_list:
            # Generate complete windows with relative numbering within each day (1, 2, 3, ...)
            formatted_windows = []
            for relative_window_num, window_data in enumerate(window_data_list, 1):
                # Generate header with relative window number
                header = generate_window_header_conditional_offset(
                    window_data['start'], window_data['end'], relative_window_num, timezone_str
                )
                
                # Combine header and content
                complete_window = f"{header}\n{window_data['content']}"
                formatted_windows.append(complete_window)
            
            daily_descriptions[day] = "\n\n".join(formatted_windows)
        else:
            daily_descriptions[day] = ""
    
    return time_range_descriptions, daily_descriptions, time_range_windows, daily_windows

def generate_all_outputs_manual(narratives, start_timestamp, end_timestamp, timezone_str):
    """Generate main description AND daily descriptions for manual mode"""
    from collections import defaultdict
    
    # Group narratives by time windows
    window_size_ms = sensor_integration_time_window * 60 * 1000
    window_groups = defaultdict(list)
    
    for narrative in narratives:
        unix_ts = narrative['unix_timestamp']
        if start_timestamp <= unix_ts <= end_timestamp:
            window_start = ((unix_ts - start_timestamp) // window_size_ms) * window_size_ms + start_timestamp
            window_groups[window_start].append(narrative)
    
    all_windows = []
    daily_windows = defaultdict(list)
    
    for window_start in sorted(window_groups.keys()):
        window_end = window_start + window_size_ms
        window_narratives = window_groups[window_start]
        
        # Generate window content without header (we'll add headers with correct numbers later)
        window_content = format_window_content_without_header(window_narratives)
        
        if window_content:  # Only process windows with content
            # Store window data (timing + content + narratives) instead of formatted string
            window_data = {
                'start': window_start,
                'end': window_end,
                'content': window_content,
                'narratives': window_narratives
            }
            
            # Add window_id for manual mode
            window_data['window_id'] = len(all_windows) + 1
            all_windows.append(window_data)
            
            # Also assign to daily output
            window_dt = pd.to_datetime(window_start, unit='ms', utc=True).tz_convert(timezone_str)
            day_str = window_dt.strftime('%Y-%m-%d')
            daily_windows[day_str].append(window_data)
    
    # Format main description with sequential numbering
    if all_windows:
        formatted_windows = []
        for window_num, window_data in enumerate(all_windows, 1):
            # Generate header with sequential window number
            header = generate_window_header_conditional_offset(
                window_data['start'], window_data['end'], window_num, timezone_str
            )
            
            # Combine header and content
            complete_window = f"{header}\n{window_data['content']}"
            formatted_windows.append(complete_window)
        
        main_description = "\n\n".join(formatted_windows)
    else:
        main_description = ""
    
    # Format daily descriptions with relative window numbering within each day
    daily_descriptions = {}
    for day, window_data_list in daily_windows.items():
        if window_data_list:
            # Generate complete windows with relative numbering within each day (1, 2, 3, ...)
            formatted_windows = []
            for relative_window_num, window_data in enumerate(window_data_list, 1):
                # Generate header with relative window number
                header = generate_window_header_conditional_offset(
                    window_data['start'], window_data['end'], relative_window_num, timezone_str
                )
                
                # Combine header and content
                complete_window = f"{header}\n{window_data['content']}"
                formatted_windows.append(complete_window)
            
            daily_descriptions[day] = "\n\n".join(formatted_windows)
        else:
            daily_descriptions[day] = ""
    
    return main_description, daily_descriptions, all_windows, daily_windows


def format_window_content_without_header(window_narratives):
    """Format window content without header - just the narrative content"""
    if not window_narratives:
        return ""
    
    # Sort narratives by sensor category
    sorted_narratives = sort_narratives_by_sensor_category(window_narratives)
    
    content_lines = []
    current_category = None
    
    for narrative in sorted_narratives:
        # Add category headers
        category = get_sensor_category(narrative['sensor_type'])
        if category != current_category:
            if content_lines:  # Add empty line before new category (except for first)
                content_lines.append("")
            content_lines.append(get_category_display_name(category))
            current_category = category
        
        # Add narrative content
        formatted_description = f"- {narrative['description']}"
        content_lines.append(formatted_description)
    
    return "\n".join(content_lines)


def generate_json_output(windows_data, pid, survey_id=None, time_range=None, output_dir=None, filename=None):
    """
    Generate JSON output for both auto and manual modes.
    
    Args:
        windows_data: For auto mode: dict of window data; for manual mode: list of window dicts
        pid (str): Participant ID
        survey_id (str, optional): Survey ID (for auto mode)
        time_range (str, optional): Time range (for auto mode)
        output_dir (str): Output directory
        
    Returns:
        str: Path to the generated JSON file, or None if no windows
    """
    json_data = []
    
    # Handle both auto and manual modes (windows_data is a list of window data)
    if isinstance(windows_data, list):
        for window_idx, window_data in enumerate(windows_data, 1):
            # Determine window_id - use existing one if available (manual mode), otherwise use index (auto mode)
            window_id = window_data.get('window_id', window_idx)
            
            # Generate the complete window description (header + content) for auto mode
            if 'content' in window_data:
                # Auto mode - generate description from content
                header = generate_window_header_conditional_offset(
                    window_data['start'], window_data['end'], window_idx, timezone
                )
                complete_description = f"{header}\n{window_data['content']}"
            else:
                # Manual mode - use existing description
                complete_description = window_data['description']
            
            # Generate categorized description directly from narratives
            categorized_description = generate_categorized_description_from_narratives(window_data.get('narratives', []))
            
            # Generate window description header
            window_description = generate_window_header_conditional_offset(
                window_data['start'], 
                window_data.get('end', window_data['start'] + (sensor_integration_time_window * 60 * 1000)), 
                window_id, 
                timezone
            )
            
            json_element = {
                'window_id': window_id,
                'window_description': window_description,
                'description': complete_description,  # Keep original description for backward compatibility
                'categorized_description': categorized_description,
                'window_info': {
                    'start_timestamp': window_data['start'],
                    'end_timestamp': window_data.get('end', window_data['start'] + (sensor_integration_time_window * 60 * 1000)),
                    'duration_minutes': sensor_integration_time_window,
                    'narrative_count': len(window_data.get('narratives', [])),
                    'sensor_types': list(set(narrative['sensor_type'] for narrative in window_data.get('narratives', [])))
                }
            }
            json_data.append(json_element)
    
    if not json_data:
        return None
    
    # Determine filename based on mode
    if filename:
        json_filename = filename
    elif survey_id and time_range:
        # Auto mode
        json_filename = f"{pid}_{survey_id}_{time_range}.json"
    else:
        # Manual mode
        json_filename = f"{pid}_manual.json"

    json_filepath = os.path.join(output_dir, json_filename)
    
    # Strip invisible Unicode control characters from all strings in the output
    json_data = _sanitize_unicode_recursive(json_data)

    # Write JSON file with UTF-8 encoding and pretty printing
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return json_filepath


def generate_categorized_description_from_narratives(narratives):
    """
    Generate categorized description directly from narrative list.
    
    Args:
        narratives (list): List of narrative dictionaries with keys:
                          - unix_timestamp: Unix timestamp in milliseconds
                          - sensor_type: Sensor type string
                          - description: Human-readable narrative text
        
    Returns:
        dict: Dictionary with 4 category keys containing sensor names as keys and descriptions as values
    """
    # Initialize categories with empty dictionaries
    categories = {
        'environmental_context': {},
        'communication_events': {},
        'device_state': {},
        'engagement_signals': {}
    }
    
    if not narratives:
        return categories
    
    # Define sensor categories mapping
    sensor_categories = {
        # Environmental context
        'locations': 'environmental_context',
        'wifi': 'environmental_context',
        'bluetooth': 'environmental_context',
        # Communication events
        'notifications': 'communication_events',
        'applications_notifications': 'communication_events',  # Added missing sensor type
        'calls': 'communication_events',
        'messages': 'communication_events',
        # Device state
        'battery': 'device_state',
        'installations': 'device_state',
        # Engagement signals
        'screen': 'engagement_signals',
        'applications': 'engagement_signals',
        'keyboard': 'engagement_signals',
        'screentext': 'engagement_signals'
    }
    
    # Group narratives by sensor type
    sensor_groups = {}
    for narrative in narratives:
        sensor_type = narrative.get('sensor_type', '')
        if sensor_type and sensor_type != 'header':  # Skip header narratives
            if sensor_type not in sensor_groups:
                sensor_groups[sensor_type] = []
            sensor_groups[sensor_type].append(narrative)
    
    # Organize into categories
    for sensor_type, sensor_narratives in sensor_groups.items():
        category = sensor_categories.get(sensor_type, 'other')
        if category in categories:
            # Combine all descriptions for this sensor type
            descriptions = []
            for narrative in sensor_narratives:
                description = narrative.get('description', '').strip()
                if description:
                    # Remove the "- " prefix if present
                    if description.startswith("- "):
                        description = description[2:]
                    descriptions.append(description)
            
            if descriptions:
                categories[category][sensor_type] = '\n'.join(descriptions)
    
    return categories


def parse_description_to_categories(description):
    """
    Parse a window description into categorized sections.
    
    Args:
        description (str): The complete window description text
        
    Returns:
        dict: Dictionary with 4 category keys containing sensor names as keys and descriptions as values
    """
    # Initialize categories with empty dictionaries
    categories = {
        'environmental_context': {},
        'communication_events': {},
        'device_state': {},
        'engagement_signals': {}
    }
    
    if not description or not description.strip():
        return categories
    
    lines = description.split('\n')
    current_category = None
    current_sensor = None
    current_description = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check for category headers
        if line == "Environmental Context":
            # Save previous sensor content
            if current_category and current_sensor and current_description:
                categories[current_category][current_sensor] = '\n'.join(current_description)
            
            current_category = 'environmental_context'
            current_sensor = None
            current_description = []
        elif line == "Communication Events":
            # Save previous sensor content
            if current_category and current_sensor and current_description:
                categories[current_category][current_sensor] = '\n'.join(current_description)
            
            current_category = 'communication_events'
            current_sensor = None
            current_description = []
        elif line == "Device State":
            # Save previous sensor content
            if current_category and current_sensor and current_description:
                categories[current_category][current_sensor] = '\n'.join(current_description)
            
            current_category = 'device_state'
            current_sensor = None
            current_description = []
        elif line == "Engagement Signals":
            # Save previous sensor content
            if current_category and current_sensor and current_description:
                categories[current_category][current_sensor] = '\n'.join(current_description)
            
            current_category = 'engagement_signals'
            current_sensor = None
            current_description = []
        elif line.startswith("Window ") or line.startswith("Day ") or " - " in line:
            # Skip window headers and time ranges
            continue
        elif current_category and line.startswith("- "):
            # This is a sensor description line
            # Extract sensor name from the line (format: "- sensor_name | description")
            parts = line[2:].split(" | ", 1)  # Remove "- " and split on first " | "
            if len(parts) == 2:
                sensor_name = parts[0].strip()
                sensor_description = parts[1].strip()
                
                # Save previous sensor content
                if current_sensor and current_description:
                    categories[current_category][current_sensor] = '\n'.join(current_description)
                
                # Start new sensor
                current_sensor = sensor_name
                current_description = [sensor_description]
            else:
                # If no " | " found, treat as continuation of current sensor
                if current_sensor:
                    current_description.append(line[2:])  # Remove "- " prefix
        elif current_category and current_sensor and line.startswith("    "):
            # This is a continuation line for the current sensor (indented)
            current_description.append(line)
    
    # Save the last sensor content
    if current_category and current_sensor and current_description:
        categories[current_category][current_sensor] = '\n'.join(current_description)
    
    return categories

def get_sensor_category(sensor_type):
    """Get category for sensor type"""
    sensor_categories = {
        'locations': 'environmental_context',
        'wifi': 'environmental_context',
        'bluetooth': 'environmental_context',
        'notifications': 'communication_events', # FIXME: remove it
        'applications_notifications': 'communication_events',  # Added missing sensor type
        'calls': 'communication_events',
        'messages': 'communication_events',
        'battery': 'device_state',
        'installations': 'device_state',
        'screen': 'engagement_signals',
        'applications': 'engagement_signals',
        'keyboard': 'engagement_signals',
        'screentext': 'engagement_signals'
    }
    return sensor_categories.get(sensor_type, 'other')

def get_category_display_name(category):
    """Get display name for category"""
    display_names = {
        'environmental_context': 'Environmental Context',
        'communication_events': 'Communication Events',
        'device_state': 'Device State', 
        'engagement_signals': 'Engagement Signals'
    }
    return display_names.get(category, category)

def sort_narratives_by_sensor_category(narratives):
    """Sort narratives by sensor category priority"""
    category_order = [
        'environmental_context', 'communication_events', 
        'device_state', 'engagement_signals'
    ]
    
    def get_priority(narrative):
        category = get_sensor_category(narrative['sensor_type'])
        try:
            return category_order.index(category)
        except ValueError:
            return len(category_order)
    
    return sorted(narratives, key=get_priority)

def convert_timestamp_column(df, timezone_str="Australia/Melbourne"):
    """
    Convert timestamp columns to the provided timezone (accounting for DST) 
    and compute duration by adding a new column 'datetime'.
    
    Parameters:
        df (pd.DataFrame): DataFrame with timestamp columns.
        timezone_str (str): Timezone to convert to (default: Australia/Melbourne).
    
    Returns:
        pd.DataFrame: Updated DataFrame with datetime conversions by adding a new column 'datetime'.
    """
    try:
        tz = pytz.timezone(timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        log_warning(f"Unknown timezone '{timezone_str}'. Falling back to UTC.")
        tz = pytz.timezone("UTC")
    
    def convert_with_offset(ts):
        dt = pd.to_datetime(ts, unit='ms', utc=True)
        local_time = dt.tz_convert(tz)
        return local_time.strftime('%Y-%m-%d %H:%M:%S')

    df['datetime'] = df['timestamp'].apply(convert_with_offset)    
    
    return df

def should_filter_system_ui_app(record, sensor_name):
    """
    Check if a record should be filtered out based on system UI app settings.
    
    This function centralizes the system UI filtering logic to avoid duplication
    across different sensor processing functions. It checks:
    1. If DISCARD_SYSTEM_UI is enabled in config
    2. If the sensor is in the list of sensors that should be filtered
    3. If the record has is_system_app = 1
    4. If the package_name is in the system_ui_apps list
    
    Args:
        record (dict): The sensor record to check
        sensor_name (str): Name of the sensor
        
    Returns:
        bool: True if the record should be filtered out, False otherwise
    """
    # Define sensors that should have System UI filtering applied
    system_ui_filter_sensors = ['applications_foreground', 'applications_notifications', 'installations', 'screentext']
    
    return (DISCARD_SYSTEM_UI and 
            sensor_name in system_ui_filter_sensors and 
            record.get('is_system_app', 0) == 1 and
            record.get('package_name') in system_ui_apps)

def get_sensor_data(sensor_name, start_timestamp, end_timestamp, input_dir, logger=None):
    """
    Load sensor data from JSONL file and filter by timestamp range.
    
    Args:
        sensor_name (str): Name of the sensor (e.g., 'battery', 'applications_foreground')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        input_dir (str): Input directory path
        logger: Logger instance for detailed logging
        
    Returns:
        list: List of sensor records within the timestamp range
    """
    # Construct the file path
    file_path = os.path.join(input_dir, f"{sensor_name}.jsonl")
    
    # Check if file exists
    if not os.path.exists(file_path):
        log_warning(f"Warning: Sensor file {file_path} not found", logger)
        return []
    
    filtered_records = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    record = json.loads(line)
                    timestamp = record.get('timestamp')
                    
                    # Check if timestamp is within range
                    if timestamp and start_timestamp <= timestamp <= end_timestamp:
                        # Apply centralized System UI filtering for applications-related sensors
                        # This prevents system UI apps like Edge panels from appearing in the data
                        if should_filter_system_ui_app(record, sensor_name):
                            continue
                        
                        filtered_records.append(record)
                        
                except json.JSONDecodeError as e:
                    log_warning(f"Warning: Failed to parse JSON line in {file_path}: {e}", logger)
                    continue
    
        # Sort records by _id then by timestamp
        filtered_records = sorted(filtered_records, key=lambda x: (x.get('_id', ''), x.get('timestamp', 0)))
                        
    except FileNotFoundError:
        log_error(f"Error: File {file_path} not found", logger)
        return []
    except Exception as e:
        log_error(f"Error reading file {file_path}: {e}", logger)
        return []
    
    log_info(f"Loaded {len(filtered_records)} records from {sensor_name} sensor", logger)
    return filtered_records

def _create_fallback_app_sequence(window_apps):
    """
    Helper function to create app sequence without session awareness.
    Used as fallback when no sessions exist or session processing yields empty results.
    """
    app_sequence = []
    for app in window_apps:
        app_name = app.get('application_name', 'Unknown')
        package_name = app.get('package_name', 'Unknown')
        is_system_app = app.get('is_system_app', 0)
        
        # Apply blacklist filtering consistently
        if any(package_name.lower() == blacklist_app.lower() for blacklist_app in blacklist_apps):
            continue
        
        # Map package name to application name if available
        if package_name in application_name_list:
            app_name = application_name_list[package_name]
        
        app_sequence.append({
            'app_name': app_name,
            'timestamp': app['timestamp'],
            'session_id': 'no_session'
        })
    
    simplified_sequence = simplify_app_sequence_list(app_sequence)
    switches = len(simplified_sequence) - 1 if len(simplified_sequence) > 1 else 0
    revisit_counts = calculate_revisit_counts_from_sequence(simplified_sequence)
    return simplified_sequence, switches, [{'session_id': 'no_session', 'sequence': simplified_sequence, 'switches': switches, 'is_fallback': True}], revisit_counts

def calculate_session_aware_sequences_and_switches(window_apps, sessions, current_window_start, current_window_end):
    """
    Calculate app sequences, switches, and revisit counts respecting session boundaries.
    
    Args:
        window_apps: List of app records in the time window
        sessions: List of session records
        current_window_start: Window start timestamp
        current_window_end: Window end timestamp
        
    Returns:
        tuple: (combined_sequence, total_switches, session_sequences, app_revisit_counts)
    """
    session_sequences = []
    combined_sequence = []
    
    # Try session-aware processing if sessions exist
    if sessions:
        # Group apps by session active periods
        session_app_sequences = {}
        total_switches = 0
        
        # Create a mapping of apps to sessions based on active periods
        for session in sessions:
            if not (session['start_timestamp'] <= current_window_end and session['end_timestamp'] >= current_window_start):
                continue
                
            session_id = session['session_id']
            session_app_sequences[session_id] = []
            
            # For each active period in this session
            for active_period in session['active_periods']:
                period_start = max(active_period['start'], current_window_start)
                period_end = min(active_period['end'], current_window_end)
                
                if period_start >= period_end:
                    continue
                    
                # Find apps within this active period
                period_apps = []
                for app in window_apps:
                    if period_start <= app['timestamp'] < period_end:
                        app_name = app.get('application_name', 'Unknown')
                        package_name = app.get('package_name', 'Unknown')
                        is_system_app = app.get('is_system_app', 0)
                        
                        # Map package name to application name if available
                        if package_name in application_name_list:
                            app_name = application_name_list[package_name]
                        
                        period_apps.append({
                            'app_name': app_name,
                            'timestamp': app['timestamp'],
                            'session_id': session_id
                        })
                
                # Add period apps to session sequence
                session_app_sequences[session_id].extend(period_apps)
        
        # Calculate simplified sequences, switches, revisits, and durations per session
        total_revisit_counts = {}
        
        for session_id, apps in session_app_sequences.items():
            if not apps:
                continue
                
            # Sort apps by timestamp within session
            apps.sort(key=lambda x: x['timestamp'])
            
            # Simplify sequence within this session
            simplified = simplify_app_sequence_list(apps)
            switches = len(simplified) - 1 if len(simplified) > 1 else 0
            
            # Calculate revisits for this session
            session_revisits = calculate_revisit_counts_from_sequence(simplified)
            
            # Calculate session duration within this window
            session_duration_seconds = 0
            session_data = next((s for s in sessions if s['session_id'] == session_id), None)
            if session_data:
                # Calculate total active time for this session within the window
                for active_period in session_data['active_periods']:
                    period_start = max(active_period['start'], current_window_start)
                    period_end = min(active_period['end'], current_window_end)
                    
                    if period_start < period_end:
                        session_duration_seconds += (period_end - period_start) / 1000.0
            
            session_sequences.append({
                'session_id': session_id,
                'sequence': simplified,
                'switches': switches,
                'revisits': session_revisits,
                'duration_seconds': session_duration_seconds
            })
            
            total_switches += switches
            
            # Aggregate revisit counts across sessions
            for app_name, revisit_count in session_revisits.items():
                total_revisit_counts[app_name] = total_revisit_counts.get(app_name, 0) + revisit_count
            
            # Add session marker to combined sequence
            if combined_sequence and simplified:
                combined_sequence.append(f"[Session {session_id}]")
            combined_sequence.extend(simplified)
        
        # If session processing was successful, return results
        if session_sequences and combined_sequence:
            return combined_sequence, total_switches, session_sequences, total_revisit_counts
    
    # Fallback: no sessions or session processing resulted in empty sequences
    return _create_fallback_app_sequence(window_apps)

def simplify_app_sequence_list(app_sequence):
    """
    Simplify app sequence by combining consecutive same apps.
    
    Args:
        app_sequence: List of app dictionaries with 'app_name' key
        
    Returns:
        List of app names with consecutive duplicates removed
    """
    if not app_sequence:
        return []
        
    simplified_sequence = []
    current_app = app_sequence[0]['app_name']
    
    for item in app_sequence:
        if item['app_name'] != current_app:
            simplified_sequence.append(current_app)
            current_app = item['app_name']
    simplified_sequence.append(current_app)
    
    return simplified_sequence

def calculate_revisit_counts_from_sequence(simplified_sequence):
    """
    Calculate revisit counts from a simplified app sequence.
    A revisit occurs when an app appears again after a different app.
    
    Args:
        simplified_sequence: List of app names in order
        
    Returns:
        dict: App name -> revisit count
    """
    revisit_counts = {}
    seen_apps = set()
    
    for app_name in simplified_sequence:
        if app_name in seen_apps:
            # This is a revisit (app appeared before)
            revisit_counts[app_name] = revisit_counts.get(app_name, 0) + 1
        else:
            # First time seeing this app in the sequence
            seen_apps.add(app_name)
            revisit_counts[app_name] = 0
    
    return revisit_counts

def calculate_session_aware_app_durations(app_usage, sessions, current_window_start, current_window_end, total_active_seconds):
    """
    Calculate actual app durations based on app foreground timestamps and session active periods.
    
    This function calculates the actual time each app was in the foreground during active periods
    by analyzing the sequence of app foreground events and determining when each app started
    and stopped being active within session boundaries.
    
    Includes all sessions that start within the current window and tracks app durations
    that extend into subsequent windows.
    
    Args:
        app_usage: Dictionary of app usage statistics
        sessions: List of session records
        current_window_start: Window start timestamp
        current_window_end: Window end timestamp
        total_active_seconds: Total active time from sessions
        
    Returns:
        dict: Updated app_usage with actual calculated durations and extension info
    """
    if not sessions or total_active_seconds <= 0:
        return app_usage
    
    # Get all app foreground events within the window, sorted by timestamp
    all_app_events = []
    for app_name, stats in app_usage.items():
        for timestamp in stats['timestamps']:
            all_app_events.append({
                'timestamp': timestamp,
                'app_name': app_name,
                'package_name': stats['package_name']
            })
    
    # Sort all events by timestamp
    all_app_events.sort(key=lambda x: x['timestamp'])
    
    # Initialize duration tracking for each app
    app_durations = {app_name: 0.0 for app_name in app_usage.keys()}
    app_extensions = {app_name: {'extends_into_next': False, 'extension_duration': 0.0} for app_name in app_usage.keys()}
    
    # Process each session that starts within the current window
    for session in sessions:
        # Include sessions that start within the window (regardless of where they end)
        session_starts_in_window = current_window_start <= session['start_timestamp'] < current_window_end
        
        if not session_starts_in_window:
            continue
        
        # Get events that fall within this session's active periods
        session_events = []
        for event in all_app_events:
            for active_period in session['active_periods']:
                if active_period['start'] <= event['timestamp'] < active_period['end']:
                    session_events.append(event)
                    break  # Found a matching active period for this event, move to next event
        
        if not session_events:
            continue
        
        # Sort session events by timestamp
        session_events.sort(key=lambda x: x['timestamp'])
        
        # Calculate actual usage durations for this session
        current_app = None
        current_app_start = None
        
        for i, event in enumerate(session_events):
            # If this is a different app than the current one, end the previous app's duration
            if current_app is not None and event['app_name'] != current_app:
                # Calculate duration for the previous app
                if current_app_start is not None:
                    duration = (event['timestamp'] - current_app_start) / 1000.0  # Convert to seconds
                    app_durations[current_app] += duration
                
                # Start tracking the new app
                current_app = event['app_name']
                current_app_start = event['timestamp']
            elif current_app is None:
                # First app in this session
                current_app = event['app_name']
                current_app_start = event['timestamp']
            # If it's the same app, continue (no need to update start time)
        
        # Handle the last app in the session
        if current_app is not None and current_app_start is not None:
            # Find the end of the last active period for this session
            last_active_end = None
            for active_period in session['active_periods']:
                period_start = max(active_period['start'], current_window_start)
                period_end = min(active_period['end'], current_window_end)
                if last_active_end is None or period_end > last_active_end:
                    last_active_end = period_end
            
            if last_active_end is not None:
                # Calculate duration within this window
                window_duration = (last_active_end - current_app_start) / 1000.0  # Convert to seconds
                app_durations[current_app] += window_duration
                
                # Check if this app extends into the next window
                if last_active_end > current_window_end:
                    app_extensions[current_app]['extends_into_next'] = True
                    # Calculate extension duration (time beyond current window)
                    extension_duration = (last_active_end - current_window_end) / 1000.0
                    app_extensions[current_app]['extension_duration'] += extension_duration
    
    # Update app_usage with calculated durations and extension info, filtering out very short durations
    filtered_app_usage = {}
    
    for app_name, stats in app_usage.items():
        calculated_duration = app_durations.get(app_name, 0.0)
        extension_info = app_extensions.get(app_name, {'extends_into_next': False, 'extension_duration': 0.0})
        
        # Drop apps with very short durations (less than 1 second)
        if calculated_duration < 1.0:
            continue  # Skip this app entirely
        
        stats['duration_seconds'] = calculated_duration
        stats['duration_minutes'] = calculated_duration / 60.0
        stats['extends_into_next_window'] = extension_info['extends_into_next']
        stats['extension_duration_seconds'] = extension_info['extension_duration']
        stats['extension_duration_minutes'] = extension_info['extension_duration'] / 60.0
        
        filtered_app_usage[app_name] = stats
    
    return filtered_app_usage

def calculate_app_durations_from_sequence(window_apps, app_usage, current_window_start, current_window_end):
    """
    Calculate app durations from adjacent app foreground events when no session data is available.
    
    Args:
        window_apps: List of app records in the time window (sorted by timestamp)
        app_usage: Dictionary of app usage statistics to update
        current_window_start: Window start timestamp
        current_window_end: Window end timestamp
        
    Returns:
        dict: Updated app_usage with calculated durations (only apps with calculable durations)
    """
    if not window_apps:
        return {}
    
    # Sort window apps by timestamp to ensure proper sequence
    sorted_apps = sorted(window_apps, key=lambda x: x['timestamp'])
    
    # Filter out blacklisted apps from the sequence
    filtered_apps = []
    for app in sorted_apps:
        package_name = app.get('package_name', 'Unknown')
        if not any(package_name.lower() == blacklist_app.lower() for blacklist_app in blacklist_apps):
            filtered_apps.append(app)
    
    if not filtered_apps:
        return {}
    
    # Calculate durations based on adjacent app events
    filtered_app_usage = {}
    
    for i, app in enumerate(filtered_apps):
        app_name = app.get('application_name', 'Unknown')
        package_name = app.get('package_name', 'Unknown')
        
        # Map package name to application name if available
        if package_name in application_name_list:
            app_name = application_name_list[package_name]
        
        # Skip if this app was already filtered out from app_usage
        if app_name not in app_usage:
            continue
        
        # Calculate duration: time until next different app
        duration_seconds = None
        
        # Look for the next different app
        for j in range(i + 1, len(filtered_apps)):
            next_app = filtered_apps[j]
            next_app_name = next_app.get('application_name', 'Unknown')
            next_package_name = next_app.get('package_name', 'Unknown')
            
            # Map package name to application name if available
            if next_package_name in application_name_list:
                next_app_name = application_name_list[next_package_name]
            
            # If next app is different, calculate duration
            if next_app_name != app_name:
                duration_ms = next_app['timestamp'] - app['timestamp']
                duration_seconds = duration_ms / 1000.0
                break
        
        # Only include apps with calculable durations (not the last app or single apps)
        if duration_seconds is not None and duration_seconds > 0:
            # Update the existing app_usage entry with calculated duration
            app_usage[app_name]['duration_seconds'] = duration_seconds
            app_usage[app_name]['duration_minutes'] = duration_seconds / 60.0
            filtered_app_usage[app_name] = app_usage[app_name]
    
    return filtered_app_usage

def generate_app_usage_summary_by_timewindow(app_data, sensor_name, time_window_minutes, start_timestamp, end_timestamp, sessions=None):
    """
    Generate app usage summary based on foreground application records by time windows.
    
    Uses fixed time windows (e.g., 60-minute intervals) starting from start_timestamp.
    When session data is provided, calculates accurate active time from session boundaries.
    Note: Sessions may overlap time windows, resulting in partial session inclusion.
    
    Args:
        app_data (list): List of applications_foreground sensor records
        sensor_name (str): Name of the sensor
        time_window_minutes (int): Time window size in minutes
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records for accurate active time calculation
        
    Returns:
        list: List of app usage summaries by time window
    """
    if sensor_name != "applications_foreground" or not app_data:
        return []
    
    summaries = []
    
    # Convert time window to milliseconds
    time_window_ms = time_window_minutes * 60 * 1000
    
    # Sort app data by timestamp
    sorted_app_data = sorted(app_data, key=lambda x: x['timestamp'])
    
    # Use provided start and end timestamps
    start_ts = start_timestamp
    end_ts = end_timestamp
    
    # Generate time windows
    current_window_start = start_ts
    window_id = 1
    
    while current_window_start < end_ts:
        current_window_end = current_window_start + time_window_ms
        
        # Get apps used during this time window
        window_apps = [
            app for app in sorted_app_data
            if current_window_start <= app['timestamp'] < current_window_end
        ]
        
        if not window_apps:
            current_window_start = current_window_end
            continue

        # Calculate app usage statistics for this time window
        app_usage = {}
        
        for app in window_apps:
            package_name = app.get('package_name', 'Unknown')
            app_name = app.get('application_name', 'Unknown')
            is_system_app = app.get('is_system_app', 0)
            
            # Map package name to application name if available
            if package_name in application_name_list:
                app_name = application_name_list[package_name]

            # Check if app is in blacklist (skip if found) - compare package names
            if any(package_name.lower() == app.lower() for app in blacklist_apps):
                continue
                
            if app_name not in app_usage:
                app_usage[app_name] = {
                    'first_seen': app['datetime'],
                    'last_seen': app['datetime'],
                    'package_name': package_name,
                    'timestamps': []
                }
            app_usage[app_name]['last_seen'] = app['datetime']
            app_usage[app_name]['timestamps'].append(app['timestamp'])

        # Calculate session-aware sequences, switches, and revisit counts
        combined_sequence, total_app_switches, session_sequences, app_revisit_counts = calculate_session_aware_sequences_and_switches(
            window_apps, sessions, current_window_start, current_window_end
        )
        
        # Calculate active time from sessions if available, otherwise estimate
        if sessions:
            total_window_duration, overlap_info = calculate_active_time_from_sessions(
                sessions, current_window_start, current_window_end
            )
        else:
            total_window_duration = 0
        
        # Calculate duration for each app
        if sessions and total_window_duration > 0:
            # Use session-aware duration calculation
            app_usage = calculate_session_aware_app_durations(
                app_usage, sessions, current_window_start, current_window_end, total_window_duration
            )
        else:
            # Fallback method when no session data: calculate durations from adjacent app foreground events
            app_usage = calculate_app_durations_from_sequence(
                window_apps, app_usage, current_window_start, current_window_end
            )
            # Calculate total active time from calculable app durations only
            total_window_duration = sum(stats.get('duration_seconds', 0) for stats in app_usage.values())
        
        # Calculate percentages only for apps with durations
        for app_name, stats in app_usage.items():
            if 'duration_seconds' in stats and total_window_duration > 0:
                stats['percentage'] = (stats['duration_seconds'] / total_window_duration) * 100
            else:
                stats['percentage'] = 0
        
        # Sort apps by usage duration (only apps with calculable durations)
        sorted_apps = sorted(
            [(app_name, stats) for app_name, stats in app_usage.items() if 'duration_seconds' in stats],
            key=lambda x: x[1]['duration_seconds'], 
            reverse=True
        )
        
        # Create summary for this time window if there are any apps (even without durations)
        if window_apps:  # Changed from sorted_apps to window_apps to always show sequence
            window_summary = {
                'window_id': window_id,
                'window_start': current_window_start,
                'window_end': current_window_end,
                'window_duration_minutes': time_window_minutes,
                'total_apps': len(sorted_apps),  # Only count apps with calculable durations
                'total_active_minutes': total_window_duration / 60.0,
                'total_active_seconds': total_window_duration,
                'app_sequence': combined_sequence,
                'total_app_switches': total_app_switches,
                'session_sequences': session_sequences,  # Detailed per-session info
                'apps_used': [],
                'session_overlap_info': overlap_info if sessions else None
            }
            
            for app_name, stats in sorted_apps:
                duration_mins = int(stats['duration_minutes'])
                duration_secs = int((stats['duration_minutes'] - duration_mins) * 60)
                
                # Use revisit count instead of raw count
                revisit_count = app_revisit_counts.get(app_name, 0)
                
                app_info = {
                    'name': app_name,
                    'package_name': stats['package_name'],
                    'revisit_count': revisit_count,
                    'duration_minutes': round(stats['duration_minutes'], 1),
                    'duration_mins': duration_mins,
                    'duration_secs': duration_secs,
                    'percentage': round(stats['percentage'], 1),
                    'first_seen': stats['first_seen'],
                    'last_seen': stats['last_seen']
                }
                
                # Add extension information if available
                if 'extends_into_next_window' in stats:
                    app_info['extends_into_next_window'] = stats['extends_into_next_window']
                    app_info['extension_duration_minutes'] = stats.get('extension_duration_minutes', 0)
                    app_info['extension_duration_seconds'] = stats.get('extension_duration_seconds', 0)
                
                window_summary['apps_used'].append(app_info)
            
            summaries.append(window_summary)
        
        current_window_start = current_window_end
        window_id += 1
    
    return summaries

def format_duration_string(duration_seconds):
    """
    Format duration in seconds to human-readable string.
    
    Args:
        duration_seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if duration_seconds >= 60:
        duration_mins = int(duration_seconds // 60)
        duration_secs = int(duration_seconds % 60)
        if duration_secs > 0:
            return f"{duration_mins} min {duration_secs} sec"
        else:
            return f"{duration_mins} min"
    else:
        return f"{int(duration_seconds)} sec"

def format_app_usage_narratives(app_summaries):
    """
    Format app usage summaries into human-readable narratives.
    
    Args:
        app_summaries (list): List of app usage summaries from time windows
        
    Returns:
        list: List of formatted narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('applications')
              - description: Human-readable narrative text
    """
    narratives = []
    
    for summary in app_summaries:
        window_id = summary['window_id']
        window_duration = summary['window_duration_minutes']
        total_apps = summary['total_apps']
        apps_used = summary['apps_used']
        total_active_minutes = summary['total_active_minutes']
        total_active_seconds = summary['total_active_seconds']
        app_sequence = summary.get('app_sequence', [])
        total_app_switches = summary.get('total_app_switches', 0)
        
        # Convert window start timestamp to datetime string
        window_start_ts = summary['window_start']
        window_start_dt = pd.to_datetime(window_start_ts, unit='ms', utc=True)
        local_time = window_start_dt.tz_convert(timezone)
        datetime_str = local_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format window end time
        window_end_ts = summary['window_end']
        window_end_dt = pd.to_datetime(window_end_ts, unit='ms', utc=True)
        end_local_time = window_end_dt.tz_convert(timezone)
        end_time_str = end_local_time.strftime('%H:%M')
        
        # Create narrative in the requested format
        # Skip only if there are no apps with durations AND no app sequence to show
        app_sequence = summary.get('app_sequence', [])
        session_sequences = summary.get('session_sequences', [])
        has_sequence = (app_sequence or 
                       (session_sequences and any(s.get('sequence') for s in session_sequences)))
        
        if total_apps == 0 and not has_sequence:
            continue
        
        # Convert total active time to minutes and seconds
        active_mins = int(total_active_minutes)
        active_secs = int((total_active_minutes - active_mins) * 60)
        
        description_parts = [f"applications | App Usage"]
        
        # Add total active time
        description_parts.append(f"    - Total active time: {active_mins} min {active_secs} sec")
        
        # Add session overlap information if available
        overlap_info = summary.get('session_overlap_info', [])
        if overlap_info:
            overlapping_count = len(overlap_info)
            extends_before = len([s for s in overlap_info if s['overlap_type'] == 'extends_before'])
            extends_after = len([s for s in overlap_info if s['overlap_type'] == 'extends_after'])
            spans_window = len([s for s in overlap_info if s['overlap_type'] == 'spans_window'])
            
            if extends_before > 0 or extends_after > 0 or spans_window > 0:
                overlap_details = []
                if extends_before > 0:
                    overlap_details.append(f"{extends_before} extend from previous window")
                if extends_after > 0:
                    overlap_details.append(f"{extends_after} extend into next window")
                if spans_window > 0:
                    overlap_details.append(f"{spans_window} span entire window")
                
                description_parts.append(f"    - Sessions: {overlapping_count} total ({', '.join(overlap_details)})")
        
        # Add app sequence with session awareness
        session_sequences = summary.get('session_sequences', [])
        if session_sequences:
            if len(session_sequences) == 1:
                # Single session - show simplified sequence
                session = session_sequences[0]
                if session['sequence']:
                    sequence_str = " → ".join(session['sequence'])
                    # Only show duration for real sessions, not fallback sequences
                    if session.get('is_fallback', False):
                        description_parts.append(f"    - App sequence: {sequence_str}")
                    else:
                        duration_seconds = session.get('duration_seconds', 0)
                        duration_str = format_duration_string(duration_seconds)
                        description_parts.append(f"    - App sequence: {sequence_str} ({duration_str})")
            else:
                # Multiple sessions - show per-session sequences
                description_parts.append(f"    - App sequences by session:")
                for session in session_sequences:
                    if session['sequence']:
                        sequence_str = " → ".join(session['sequence'])
                        switches = session['switches']
                        
                        # Only show duration for real sessions, not fallback sequences
                        if session.get('is_fallback', False):
                            if switches > 0:
                                description_parts.append(f"         Session {session['session_id']}: {sequence_str} ({switches} switches)")
                            else:
                                description_parts.append(f"         Session {session['session_id']}: {sequence_str}")
                        else:
                            duration_seconds = session.get('duration_seconds', 0)
                            duration_str = format_duration_string(duration_seconds)
                            
                            if switches > 0:
                                description_parts.append(f"         Session {session['session_id']}: {sequence_str} ({switches} switches, {duration_str})")
                            else:
                                description_parts.append(f"         Session {session['session_id']}: {sequence_str} ({duration_str})")
        elif app_sequence:
            # Fallback for old format (no session data)
            sequence_str = " → ".join(app_sequence)
            description_parts.append(f"    - App sequence: {sequence_str}")
        
        # Add app switches info
        session_sequences = summary.get('session_sequences', [])
        if total_app_switches > 0:
            if len(session_sequences) > 1:
                total_sessions = len(session_sequences)
                description_parts.append(f"    - Total app switches: {total_app_switches} across {total_sessions} sessions")
            else:
                description_parts.append(f"    - App switches: {total_app_switches}")
        
        # Add primary app with detailed info
        if total_apps >= 1:
            primary_app = apps_used[0]
            duration_mins = primary_app['duration_mins']
            duration_secs = primary_app['duration_secs']
            percentage = primary_app['percentage']
            
            revisit_count = primary_app['revisit_count']
            extension_info = ""
            if primary_app.get('extends_into_next_window', False):
                ext_mins = int(primary_app.get('extension_duration_minutes', 0))
                ext_secs = int((primary_app.get('extension_duration_minutes', 0) - ext_mins) * 60)
                extension_info = f" (extends {ext_mins} min {ext_secs} sec into next window)"
            
            if revisit_count > 0:
                description_parts.append(f"    - Primary: {primary_app['name']} ({duration_mins} min {duration_secs} sec; {percentage}% of active periods; {revisit_count} revisits){extension_info}")
            else:
                description_parts.append(f"    - Primary: {primary_app['name']} ({duration_mins} min {duration_secs} sec; {percentage}% of active periods){extension_info}")
            
            # Add secondary apps with detailed info
            if len(apps_used) > 1:
                description_parts.append(f"    - Also used (sorted by usage time):")
                for app in apps_used[1:]:  # Show all secondary apps
                    duration_mins = app['duration_mins']
                    duration_secs = app['duration_secs']
                    percentage = app['percentage']
                    revisit_count = app['revisit_count']
                    extension_info = ""
                    if app.get('extends_into_next_window', False):
                        ext_mins = int(app.get('extension_duration_minutes', 0))
                        ext_secs = int((app.get('extension_duration_minutes', 0) - ext_mins) * 60)
                        extension_info = f" (extends {ext_mins} min {ext_secs} sec into next window)"
                    
                    if revisit_count > 0:
                        description_parts.append(f"         - {app['name']} ({duration_mins} min {duration_secs} sec; {percentage}% of active periods; {revisit_count} revisits){extension_info}")
                    else:
                        description_parts.append(f"         - {app['name']} ({duration_mins} min {duration_secs} sec; {percentage}% of active periods){extension_info}")
        
        description = '\n'.join(description_parts)
        narratives.append({
            'unix_timestamp': window_start_ts,  # Use actual unix timestamp from summary
            'sensor_type': 'applications',
            'description': description
        })
    
    return narratives

def describe_applications_foreground_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated foreground applications analysis by time windows.
    Generates app usage summaries and formats them into narrative descriptions.
    
    Args:
        sensor_data (list): List of application sensor records
        sensor_name (str): Name of the sensor (should be 'applications_foreground')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records for session correlation
        
    Returns:
        list: List of formatted application narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('applications')
              - description: Human-readable narrative text
    """
    log_info("Generating integrated description for applications_foreground")
    
    if not sensor_data:
        log_info("No application data available, skipping applications_foreground integration")
        return []
    
    # Generate app usage summaries by time window
    app_summaries = generate_app_usage_summary_by_timewindow(
        sensor_data, 
        sensor_name, 
        sensor_integration_time_window,
        start_timestamp,
        end_timestamp,
        sessions
    )
    
    # Format summaries into narratives
    narratives = format_app_usage_narratives(app_summaries)
    
    log_info(f"Generated {len(narratives)} app usage narratives for {len(app_summaries)} time windows (window size: {sensor_integration_time_window} minutes)")
    return narratives


def describe_locations_integrated(all_points, cluster_labels, cluster, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated location analysis by time windows based on clustering results.
    
    Uses timestamp-based calculations for accurate location stay times instead of proportional
    allocation based on data points. Properly handles multiple visits to the same location
    and provides detailed visit period information.
    
    Args:
        all_points: Array of location points [lat, lon, datetime_str, speed]
        cluster_labels: Labels assigned to each point by clustering
        cluster: Cluster information containing place names and distances
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records for session correlation
        
    Returns:
        list: List of formatted location narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('locations')
              - description: Human-readable narrative text
    
    Note: Speed data is not used for movement analysis due to unreliable sampling frequency.
    """
    log_info("Generating integrated description for locations")
    
    if len(all_points) == 0 or len(cluster_labels) == 0:
        log_info("No location data available, skipping location integration")
        return []
    
    # Build a richer cluster_id_to_info map
    cluster_id_to_info = {}
    log_info("DEBUG: Processing cluster data:")
    for cid, lat, lon, n_pts, raw_place, dist in cluster:
        log_info(f"  Cluster {cid}: {raw_place[:100]}...")
        # 1. split off a "home." prefix if present
        is_home = raw_place.startswith("home. ")
        if is_home:
            rest = raw_place[len("home. "):]
        else:
            rest = raw_place
        
        # 2. split "rest" into the base_desc and the address
        if ". Address: " in rest:
            base_desc, address = rest.split(". Address: ", 1)
            base_desc += "."  # re-append the period
        else:
            base_desc = rest
            address = ""
        
        # 3. Determine label for sequence
        if is_home:
            # If it's home, use "Home" as the label
            label = "Home"
        else:
            # For non-home places, preserve unique unknown labels when reverse geocoding failed
            # If raw_place starts with "unknown" followed by a number, it means reverse geocoding failed
            # and we should preserve the unique unknown label
            if raw_place.startswith("unknown") and raw_place[7:].isdigit():
                label = raw_place  # Preserve unique unknown label (e.g., "unknown1", "unknown2")
                landmark_name = None
                area_name = None
            else:
                # Extract label for sequence: first landmark name, else first area name, else formatted address
                label = "unknown"  # Default fallback
                landmark_name = None
                area_name = None
                
                                # Try to extract first landmark name
                if "Landmarks:" in rest:
                    lm_section = rest.split("Landmarks: ")[1].split(". Address:")[0]
                    if "(" in lm_section:
                        first_lm = lm_section.split("(")[0].strip()
                        # Remove the relationship word (near, within, etc.)
                        lm_parts = first_lm.split(" ")
                        if len(lm_parts) > 1:
                            landmark_name = " ".join(lm_parts[1:])
                        else:
                            landmark_name = first_lm
                
                # Try to extract first area name
                if "Areas:" in rest:
                    ar_section = rest.split("Areas: ")[1].split(". Landmarks:")[0]
                    if "," in ar_section:
                        area_name = ar_section.split(",")[0].strip()
                    else:
                        area_name = ar_section.strip()
                    # Remove the relationship word
                    ar_parts = area_name.split(" ")
                    if len(ar_parts) > 1:
                        area_name = " ".join(ar_parts[1:])
            
            if landmark_name:
                label = landmark_name
            elif area_name:
                label = area_name
            elif address:
                # If no landmark or area found, use the formatted address
                label = address
        
        cluster_id_to_info[cid] = {
            "label": label,
            "base_desc": base_desc,      # e.g. "Areas: ..., Landmarks: ..."
            "address": address,          # e.g. "27 Wreckyn St, North Melbourne…"
            "distance": dist
        }
    
    # Convert points to a more usable format with timestamps
    location_records = []
    
    for i, pt in enumerate(all_points):
        ts_str = pt[2]
        cid = cluster_labels[i]
        info = cluster_id_to_info.get(cid, {})
        label = info.get("label", "unknown")

        timestamp = convert_timestring_to_timestamp(ts_str, CONFIG["timezone"])
        location_records.append({
            'timestamp': timestamp,
            'datetime': ts_str,
            'latitude': pt[0],
            'longitude': pt[1],
            'speed': pt[3],
            'cluster_id': cid,
            'place_name': label,         # only the short label
            'place_base_desc': info.get("base_desc", ""),
            'place_address': info.get("address", ""),
            'distance_from_home': info.get("distance", 0),
        })
    
    def process_location_window(window_data, datetime_str, window_start, window_end):
        """Process location data for a single time window."""
        if not window_data:
            return None
        
        
        # Sort window data by timestamp
        sorted_data = sorted(window_data, key=lambda x: x['timestamp'])
        
        # Analyze locations visited in this window
        location_visits = {}
        location_sequence = []
        location_transitions = 0
        
        # Use timestamp-based calculation for more accurate location stay times
        location_visits = calculate_location_stay_times_from_timestamps(sorted_data, window_start, window_end)
        
        # Build sequence using only the filtered locations (not raw data)
        # First, get the set of valid location names that passed the threshold
        valid_location_names = set(location_visits.keys())
        
        # Build sequence from raw data but only include valid locations
        # Apply display name conversion for sequence (same logic as for location display)
        sequence_labels = []
        for record in sorted_data:
            lbl = record["place_name"]
            # Only include locations that passed the threshold
            if lbl in valid_location_names:
                # Apply same display name conversion as we do for location display
                display_lbl = lbl
                if lbl == "unknown":
                    # Find cluster info to check if we should display as "home"
                    cid = next((cid for cid, info in cluster_id_to_info.items() if info["label"] == lbl), None)
                    if cid is not None:
                        info = cluster_id_to_info[cid]
                        if info["base_desc"] == "home":
                            display_lbl = "home"
                
                if not sequence_labels or sequence_labels[-1] != display_lbl:
                    sequence_labels.append(display_lbl)
        
        # Generate description for this window
        # Show basic statistics
        total_locations = len(location_visits)

        # Skip windows with no significant locations
        if total_locations == 0:
            return None

        description_parts = [f"locations | Significant Location Analysis"]

        if total_locations == 1:
            description_parts.append(f"    - Visited {total_locations} location")
        else:
            description_parts.append(f"    - Visited {total_locations} locations")

        # Show location transitions
        location_transitions = len(sequence_labels) - 1 if len(sequence_labels) > 1 else 0
        if location_transitions > 0:
            description_parts.append(f"    - Location transitions: {location_transitions}")

        # Show location sequence if there are multiple locations
        if len(sequence_labels) > 1:
            sequence_str = " → ".join(sequence_labels)
            description_parts.append(f"    - Location sequence: {sequence_str}")

        # Show detailed location information
        # Sort locations by chronological order (first occurrence time)
        sorted_locations = sorted(location_visits.items(),
                                 key=lambda x: x[1]['visit_periods'][0]['start_time'] if x[1]['visit_periods'] else 0)
        
        description_parts.append(f"    - Time spent at locations:")
        
        for place_label, stats in sorted_locations:
            # 1) bullet with label & time
            total_seconds = stats["estimated_time_seconds"]
            if total_seconds >= 3600:  # 1 hour
                hrs = int(total_seconds // 3600)
                mins = int((total_seconds % 3600) // 60)
                time_str = f"{hrs}h {mins}m" if mins else f"{hrs}h"
            elif total_seconds >= 60:
                mins = int(total_seconds // 60)
                secs = int(total_seconds % 60)
                time_str = f"{mins}m {secs}s" if secs > 0 else f"{mins}m"
            else:
                time_str = f"{int(total_seconds)}s"

            visits = stats.get("visit_count", 1)
            visit_periods = stats.get('visit_periods', [])
            
            # For single visits, show the time period inline
            if visits == 1 and visit_periods:
                period = visit_periods[0]
                # Convert timestamps to datetime for display
                start_dt = pd.to_datetime(period['start_time'], unit='ms', utc=True).tz_convert(timezone)
                end_dt = pd.to_datetime(period['end_time'], unit='ms', utc=True).tz_convert(timezone)
                start_time_str = start_dt.strftime('%H:%M:%S')
                end_time_str = end_dt.strftime('%H:%M:%S')
                suffix = f" ({start_time_str} - {end_time_str})"
            else:
                suffix = f" ({visits} visits)" if visits > 1 else ""
            
            # For better readability, display "home" instead of "unknown" when appropriate
            display_label = place_label
            show_base_desc = True
            
            #    find the cluster_id for this label:
            cid = next((cid for cid, info in cluster_id_to_info.items() if info["label"] == place_label), None)
            if cid is not None:
                info = cluster_id_to_info[cid]
                
                # If label is "unknown" and description is simply "home", use "home" for better readability
                if place_label == "unknown" and info["base_desc"] == "home":
                    display_label = "Home"
                    show_base_desc = False  # Don't repeat "home" below
            
            description_parts.append(f"         - {display_label}: {time_str}{suffix}")

            # 2) now dump your saved human-sentence and address
            if cid is not None:
                info = cluster_id_to_info[cid]

                # Show the base description only if it provides meaningful additional information
                if info["base_desc"] and show_base_desc:
                    # Don't repeat the same label (e.g., don't show "unknown4" under "unknown4")
                    if info["base_desc"] != display_label:
                        description_parts.append(f"              {info['base_desc']}")
                if info["address"]:
                    description_parts.append(f"              Address: {info['address']}")
                # Show cluster distance from home (skip for Home location)
                if info["distance"] > 0:
                    description_parts.append(f"              Distance from home: {info['distance']:.1f}m")
            
            # Show individual visit periods only for multiple visits
            visit_periods = stats.get('visit_periods', [])
            if len(visit_periods) > 1:
                description_parts.append(f"              Visit periods:")
                for i, period in enumerate(visit_periods, 1):
                    period_duration = period['duration_seconds']
                    if period_duration >= 60:
                        period_mins = int(period_duration // 60)
                        period_secs = int(period_duration % 60)
                        if period_secs > 0:
                            period_time_str = f"{period_mins}m {period_secs}s"
                        else:
                            period_time_str = f"{period_mins}m" # use min to avoid confusion with meters
                    else:
                        period_time_str = f"{int(period_duration)}s"
                    
                    # Convert timestamps to datetime for display
                    start_dt = pd.to_datetime(period['start_time'], unit='ms', utc=True).tz_convert(timezone)
                    end_dt = pd.to_datetime(period['end_time'], unit='ms', utc=True).tz_convert(timezone)
                    
                    start_time_str = start_dt.strftime('%H:%M:%S')
                    end_time_str = end_dt.strftime('%H:%M:%S')
                    
                    description_parts.append(f"                 {i}. {start_time_str} - {end_time_str} ({period_time_str})")
        
        # Show session correlation if sessions are available
        if sessions:
            # Find sessions that overlap with this window
            overlapping_sessions = []
            for session in sessions:
                if (session['start_timestamp'] <= window_end and 
                    session['end_timestamp'] >= window_start):
                    overlapping_sessions.append(session['session_id'])
            
            if overlapping_sessions:
                if len(overlapping_sessions) == 1:
                    description_parts.append(f"    - Session activity: Session {overlapping_sessions[0]}")
                else:
                    description_parts.append(f"    - Session activity: Sessions {', '.join(map(str, overlapping_sessions))}")
        
        return '\n'.join(description_parts)
    
    # Process data using the helper function
    narratives = process_sensor_by_timewindow(
        location_records, "locations", start_timestamp, end_timestamp, process_location_window
    )
    
    log_info(f"Generated {len(narratives)} location narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives

def process_sensor_by_timewindow(sensor_data, sensor_name, start_timestamp, end_timestamp, process_window_func):
    """
    Shared helper function to process sensor data by time windows.
    
    Args:
        sensor_data (list): List of sensor records
        sensor_name (str): Name of the sensor
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        process_window_func (callable): Function to process each window's data
        
    Returns:
        list: List of formatted narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type string
              - description: Human-readable narrative text
    """
    if not sensor_data:
        return []
    
    narratives = []
    
    # Convert time window to milliseconds
    time_window_ms = sensor_integration_time_window * 60 * 1000
    
    # Sort sensor data by timestamp
    sorted_sensor_data = sorted(sensor_data, key=lambda x: x['timestamp'])
    
    # Generate time windows
    current_window_start = start_timestamp
    window_id = 1
    
    while current_window_start < end_timestamp:
        current_window_end = current_window_start + time_window_ms
        
        # Get sensor records during this time window
        window_data = [
            record for record in sorted_sensor_data
            if current_window_start <= record['timestamp'] < current_window_end
        ]
        

        if not window_data:
            current_window_start = current_window_end
            continue
        
        # Convert window start timestamp to datetime string
        window_start_dt = pd.to_datetime(current_window_start, unit='ms', utc=True)
        local_time = window_start_dt.tz_convert(timezone)
        datetime_str = local_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Process window data using the provided function
        description = process_window_func(window_data, datetime_str, current_window_start, current_window_end)
        
        if description:
            narratives.append({
                'unix_timestamp': current_window_start,  # Use actual unix timestamp
                'sensor_type': sensor_name,
                'description': description
            })
        
        current_window_start = current_window_end
        window_id += 1
    
    return narratives

def is_meaningful_bluetooth_name(bt_name):
    """
    Check if a bluetooth name is meaningful (human-readable) or just a default/manufacturer identifier.
    
    Args:
        bt_name (str): Bluetooth device name
        
    Returns:
        bool: True if the name is meaningful, False if it's a default identifier
    """
    if not bt_name or not bt_name.strip():
        return False
    
    bt_name = bt_name.strip()
    
    # Check for common default patterns
    # Pattern 1: Numbers with decimals (e.g., "46221610.00007702")
    if bt_name.replace('.', '').isdigit():
        return False
    
    # Pattern 2: Mostly numbers with minimal non-numeric characters
    numeric_chars = sum(c.isdigit() for c in bt_name)
    if len(bt_name) > 8 and numeric_chars / len(bt_name) > 0.7:
        return False
    
    # Pattern 3: Very long strings that look like IDs
    if len(bt_name) > 20 and not any(c.isalpha() and c.islower() for c in bt_name):
        return False
    
    # If none of the default patterns match, consider it meaningful
    return True

def describe_bluetooth_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated bluetooth device analysis by time windows.
    Uses configurable gate_time_window-minute gates to calculate average number of unique bluetooth devices within each time window.
    
    Args:
        sensor_data (list): List of bluetooth sensor records
        sensor_name (str): Name of the sensor (should be 'bluetooth')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records (unused for bluetooth)
        
    Returns:
        list: List of formatted bluetooth narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('bluetooth')
              - description: Human-readable narrative text
    """
    log_info("Generating integrated description for bluetooth")
    
    if sensor_name != "bluetooth" or not sensor_data:
        log_info("No bluetooth data available, skipping bluetooth integration")
        return []
    
    def process_bluetooth_gate(gate_data):
        """Process bluetooth data for a single gate_time_window-minute gate."""
        if not gate_data:
            return None
        
        # Group devices by bt_address and calculate statistics
        device_stats = {}
        
        for record in gate_data:
            bt_address = record.get('bt_address', 'Unknown')
            bt_name = record.get('bt_name', '')
            bt_rssi = record.get('bt_rssi', None)
            
            if bt_address not in device_stats:
                device_stats[bt_address] = {
                    'bt_name': bt_name,
                    'rssi_values': [],
                    'detection_count': 0
                }
            
            device_stats[bt_address]['detection_count'] += 1
            
            # Update bt_name if we get a valid one (non-empty)
            if bt_name and bt_name.strip():
                device_stats[bt_address]['bt_name'] = bt_name
            
            # Collect RSSI values for averaging
            if bt_rssi is not None:
                try:
                    rssi_value = float(bt_rssi)
                    device_stats[bt_address]['rssi_values'].append(rssi_value)
                except (ValueError, TypeError):
                    pass  # Skip invalid RSSI values
        
        if not device_stats:
            return None
        
        # Calculate total unique devices (including unnamed ones)
        total_unique_devices = len(device_stats)
        
        # Calculate average RSSI for each device and prepare sorted list
        # Only include devices with meaningful bt_name for display
        named_devices_with_stats = []
        for bt_address, stats in device_stats.items():
            # Only include devices with meaningful bt_name in the display list
            if not is_meaningful_bluetooth_name(stats['bt_name']):
                continue
                
            if stats['rssi_values']:
                avg_rssi = sum(stats['rssi_values']) / len(stats['rssi_values'])
            else:
                avg_rssi = None
            
            named_devices_with_stats.append({
                'bt_address': bt_address,
                'display_name': stats['bt_name'],  # Always use bt_name since we filtered for it
                'avg_rssi': avg_rssi,
                'detection_count': stats['detection_count']
            })
        
        # Sort by average RSSI (higher/closer to 0 means stronger signal, so reverse=True)
        named_devices_with_stats.sort(key=lambda x: x['avg_rssi'] if x['avg_rssi'] is not None else -999, reverse=True)
        
        return {
            'total_unique_devices': total_unique_devices,
            'named_devices': named_devices_with_stats
        }
    
    def process_bluetooth_window(window_data, datetime_str, window_start, window_end):
        """Process bluetooth data for a single time window, calculating gate statistics and averages."""
        if not window_data:
            return None
        
        # Define gate size in milliseconds from config
        gate_size_ms = gate_time_window * 60 * 1000  # Convert minutes to milliseconds
        
        # Collect statistics for each gate
        gate_stats = []
        window_device_appearances = {}  # Track device appearances across gates
        
        current_gate_start = window_start
        while current_gate_start < window_end:
            current_gate_end = min(current_gate_start + gate_size_ms, window_end)
            
            # Get data for this gate
            gate_data = [
                record for record in window_data
                if current_gate_start <= record['timestamp'] < current_gate_end
            ]
            
            if gate_data:
                gate_result = process_bluetooth_gate(gate_data)
                
                if gate_result:
                    # Calculate gate-level statistics
                    gate_unique_devices = gate_result['total_unique_devices']
                    gate_named_devices = len(gate_result['named_devices'])
                    
                    gate_stats.append({
                        'unique_devices': gate_unique_devices,
                        'named_devices': gate_named_devices,
                        'devices': gate_result['named_devices']
                    })
                    
                    # Track device appearances across gates for averaging
                    for device in gate_result['named_devices']:
                        bt_address = device['bt_address']
                        
                        # Double-check that this device has a meaningful name
                        if not is_meaningful_bluetooth_name(device['display_name']):
                            continue
                        
                        if bt_address not in window_device_appearances:
                            window_device_appearances[bt_address] = {
                                'display_name': device['display_name'],
                                'rssi_values': [],
                                'detection_counts': [],
                                'gate_count': 0
                            }
                        
                        # Record this gate's values for later averaging
                        if device['avg_rssi'] is not None:
                            window_device_appearances[bt_address]['rssi_values'].append(device['avg_rssi'])
                        window_device_appearances[bt_address]['detection_counts'].append(device['detection_count'])
                        window_device_appearances[bt_address]['gate_count'] += 1
            
            current_gate_start = current_gate_end
        
        if not gate_stats:
            return None
        
        # Calculate window-level statistics from gate statistics
        unique_device_counts = [gate['unique_devices'] for gate in gate_stats]
        named_device_counts = [gate['named_devices'] for gate in gate_stats]
        
        avg_unique_devices = sum(unique_device_counts) / len(unique_device_counts)
        min_unique_devices = min(unique_device_counts)
        max_unique_devices = max(unique_device_counts)
        
        avg_named_devices = sum(named_device_counts) / len(named_device_counts)
        min_named_devices = min(named_device_counts)
        max_named_devices = max(named_device_counts)
        
        # Skip windows with no meaningful activity (no unique devices or very low activity)
        if avg_unique_devices == 0 or max_unique_devices == 0:
            return None
        
        # Calculate average statistics for each device across gates
        # Group by device name to avoid counting same device multiple times
        # 
        # ASSUMPTION: Multiple MAC addresses with the same device name represent the same physical device
        # (e.g., device with multiple Bluetooth interfaces, or MAC address randomization)
        # 
        # LIMITATION: This method might be incorrect if there are genuinely different devices 
        # with identical names (e.g., multiple "LE_WH-1000XM5" headphones from different people).
        # In such cases, this would incorrectly merge statistics from separate devices.
        # 
        # CALCULATION: All detection counts from all MAC addresses are pooled together,
        # then averaged across all gate-appearances, giving more weight to MAC addresses
        # that appeared in more gates.
        device_name_aggregated = {}
        for bt_address, stats in window_device_appearances.items():
            device_name = stats['display_name']
            
            if device_name not in device_name_aggregated:
                device_name_aggregated[device_name] = {
                    'rssi_values': [],
                    'detection_counts': [],
                    'gate_appearances': 0
                }
            
            # Aggregate data for this device name
            device_name_aggregated[device_name]['rssi_values'].extend(stats['rssi_values'])
            device_name_aggregated[device_name]['detection_counts'].extend(stats['detection_counts'])
            device_name_aggregated[device_name]['gate_appearances'] += stats['gate_count']
        
        # Calculate final averages per unique device name
        averaged_devices = []
        for device_name, aggregated_stats in device_name_aggregated.items():
            # Calculate average RSSI across all appearances
            if aggregated_stats['rssi_values']:
                avg_rssi = sum(aggregated_stats['rssi_values']) / len(aggregated_stats['rssi_values'])
            else:
                avg_rssi = None
            
            # Calculate average detections per gate where device appeared
            avg_detections = sum(aggregated_stats['detection_counts']) / len(aggregated_stats['detection_counts'])
            
            averaged_devices.append({
                'display_name': device_name,
                'avg_rssi': avg_rssi,
                'avg_detections': avg_detections,
                'gate_appearances': aggregated_stats['gate_appearances'],
                'total_gates': len(gate_stats)
            })
        
        # Sort by average detections (descending)
        averaged_devices.sort(key=lambda x: x['avg_detections'], reverse=True)
        
        # Skip windows with no meaningful named devices if there are also very few unique devices
        if not averaged_devices and avg_unique_devices < 5:
            return None
        
        # Generate description for the window
        description_parts = [f"bluetooth | Bluetooth Devices Detected"]
        
        # Show average and range of unique devices (calculated from gate_time_window-min gate scans)
        if min_unique_devices == max_unique_devices:
            description_parts.append(f"    - Average unique devices: {avg_unique_devices:.1f} (from {gate_time_window}-min gate scans)")
        else:
            description_parts.append(f"    - Average unique devices: {avg_unique_devices:.1f} (range: {min_unique_devices}-{max_unique_devices}, from {gate_time_window}-min gate scans)")
        
        # Show average and range of named devices (calculated from gate_time_window-min gate scans)
        if min_named_devices == max_named_devices:
            description_parts.append(f"    - Average named devices: {avg_named_devices:.1f} (from {gate_time_window}-min gate scans)")
        else:
            description_parts.append(f"    - Average named devices: {avg_named_devices:.1f} (range: {min_named_devices}-{max_named_devices}, from {gate_time_window}-min gate scans)")
        
        if averaged_devices:
            # Limit to top 10 devices
            top_devices = averaged_devices[:10]
            total_devices = len(averaged_devices)
            
            if total_devices > 10:
                description_parts.append(f"    - Top 10 of {total_devices} named devices (by average detection frequency from {gate_time_window}-min gate scans):")
            else:
                description_parts.append(f"    - {total_devices} named devices (by average detection frequency from {gate_time_window}-min gate scans):")
            
            for device in top_devices:
                description_parts.append(
                    f"         - {device['display_name']} "
                    f"({device['avg_detections']:.1f} detections)"
                )
        else:
            description_parts.append(f"    - No named devices detected")
        
        return '\n'.join(description_parts)
    
    # Process data using the helper function
    narratives = process_sensor_by_timewindow(
        sensor_data, sensor_name, start_timestamp, end_timestamp, process_bluetooth_window
    )
    
    log_info(f"Generated {len(narratives)} bluetooth narratives with averaged {gate_time_window}-minute gate statistics (window size: {sensor_integration_time_window} minutes, filtering out default/manufacturer device names and grouping by device name)")
    return narratives

def describe_battery_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated battery status analysis by time windows.
    
    Args:
        sensor_data (list): List of battery sensor records
        sensor_name (str): Name of the sensor (should be 'battery')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records (unused for battery)
        
    Returns:
        list: List of formatted battery narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('battery')
              - description: Human-readable narrative text
    """
    log_info("Generating integrated description for battery")
    
    if sensor_name != "battery" or not sensor_data:
        log_info("No battery data available, skipping battery integration")
        return []
    
    # Battery status mapping
    statuses = {
        -2: "rebooted",
        -1: "shutdown",
        2: "charging",
        3: "discharging", 
        4: "not charging",
        5: "fully charged"
    }
    
    # First pass: build status periods from all data
    all_status_periods = []
    previous_status = None
    current_period = None
    
    # Sort sensor data by timestamp for sequential processing
    sorted_sensor_data = sorted(sensor_data, key=lambda x: x['timestamp'])
    
    for record in sorted_sensor_data:
        if 'battery_status' in record and record['battery_status'] in statuses:
            current_status = statuses[record['battery_status']]
            current_level = record.get('battery_level', 'Unknown')
            current_datetime = record['datetime']
            
            # Check if this continues the current status period or starts a new one
            if current_status != previous_status:
                # End previous period
                if current_period:
                    all_status_periods.append(current_period)
                
                # Start new period
                current_period = {
                    'status': current_status,
                    'start_datetime': current_datetime,
                    'end_datetime': current_datetime,
                    'start_level': current_level,
                    'end_level': current_level
                }
                previous_status = current_status
            else:
                # Continue current period - update end time and level
                if current_period:
                    current_period['end_datetime'] = current_datetime
                    current_period['end_level'] = current_level
    
    # Add the final period
    if current_period:
        all_status_periods.append(current_period)
    
    def process_battery_window(window_data, datetime_str, window_start, window_end):
        """Process battery data for a single time window."""
        if not window_data:
            return None
        
        # Find status periods that fall within this window
        window_periods = []
        for period in all_status_periods:
            # Convert datetime strings to timestamps for comparison
            period_start_ts = convert_timestring_to_timestamp(period['start_datetime'], CONFIG["timezone"])
            
            if window_start <= period_start_ts < window_end:
                window_periods.append(period)
        
        if not window_periods:
            return None
        
        # Generate description for this window
        description_parts = [f"battery | Status changes"]
        
        for period in window_periods:
            if period['start_datetime'] == period['end_datetime']:
                # Single status change
                description_parts.append(f"    - {period['start_datetime']} | {period['status']} ({period['start_level']}%)")
            else:
                # Calculate duration in minutes
                start_ts = convert_timestring_to_timestamp(period['start_datetime'], CONFIG["timezone"])
                end_ts = convert_timestring_to_timestamp(period['end_datetime'], CONFIG["timezone"])
                duration_ms = end_ts - start_ts
                duration_mins = int(duration_ms / (1000 * 60))  # Convert ms to minutes
                
                # Format duration with hours when necessary
                if duration_mins >= 60:
                    hours = duration_mins // 60
                    mins = duration_mins % 60
                    if mins > 0:
                        duration_str = f"{hours} hour{'s' if hours > 1 else ''} {mins} mins"
                    else:
                        duration_str = f"{hours} hour{'s' if hours > 1 else ''}"
                else:
                    duration_str = f"{duration_mins} mins"
                
                # Status period with level range
                if period['start_level'] != period['end_level']:
                    description_parts.append(f"    - {period['start_datetime']} | {period['status']} from {period['start_level']}% to {period['end_level']}% for {duration_str}")
                else:
                    # Same level throughout the period
                    description_parts.append(f"    - {period['start_datetime']} | {period['status']} ({period['start_level']}%) for {duration_str}")
        
        return '\n'.join(description_parts)
    
    # Process data using the shared helper function
    narratives = process_sensor_by_timewindow(
        sensor_data, sensor_name, start_timestamp, end_timestamp, process_battery_window
    )
    
    log_info(f"Generated {len(narratives)} battery narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives

def describe_applications_notifications_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated applications notifications analysis by time windows.
    Shows notification patterns, app sources, frequencies, and content summaries.
    
    Args:
        sensor_data (list): List of applications_notifications sensor records
        sensor_name (str): Name of the sensor (should be 'applications_notifications')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records for session correlation
        
    Returns:
        list: List of formatted notifications narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('notifications')
              - description: Human-readable narrative text
    """
    log_info("Generating integrated description for applications_notifications")
    
    if sensor_name != "applications_notifications" or not sensor_data:
        log_info("No applications_notifications data available, skipping notifications integration")
        return []
    
    def process_notifications_window(window_data, datetime_str, window_start, window_end):
        """Process notifications data for a single time window."""
        if not window_data:
            return None
        
        # Sort window data by timestamp
        sorted_data = sorted(window_data, key=lambda x: x['timestamp'])
        
        # Track notifications by app
        app_notifications = {}
        total_notifications = 0
        notifications_with_content = 0
        
        # Process each notification record
        for record in sorted_data:
            app_name = record.get('application_name', 'Unknown')
            text = record.get('text', '')
            package_name = record.get('package_name', '')
            record_datetime = record.get('datetime', datetime_str)
            
            # Initialize app tracking if not exists
            if app_name not in app_notifications:
                app_notifications[app_name] = {
                    'count': 0,
                    'package_name': package_name,
                    'notifications': [],
                    'has_content': 0,
                    'full_texts': []
                }
            
            # Update app statistics
            app_notifications[app_name]['count'] += 1
            app_notifications[app_name]['notifications'].append({
                'datetime': record_datetime,
                'timestamp': record.get('timestamp', 0),
                'text': text
            })
            
            # Track content statistics
            if text and text.strip() and text != "[]":
                app_notifications[app_name]['has_content'] += 1
                notifications_with_content += 1
                # Keep all texts instead of just samples
                if text not in app_notifications[app_name]['full_texts']:
                    app_notifications[app_name]['full_texts'].append(text)
            
            total_notifications += 1
        
        # Generate description for this window
        description_parts = [f"notifications | Notification Activity"]
        
        # Show total notifications summary
        if total_notifications > 0:
            description_parts.append(f"    - Total notifications: {total_notifications}")
            description_parts.append(f"    - Apps sending notifications: {len(app_notifications)}")
            
            # Show content statistics
            if notifications_with_content > 0:
                description_parts.append(f"    - Notifications with content: {notifications_with_content}")
            
            # Sort apps by notification count
            sorted_apps = sorted(app_notifications.items(), key=lambda x: x[1]['count'], reverse=True)
            
            # Show top active apps
            description_parts.append(f"    - Notification sources:")
            for app_name, app_data in sorted_apps[:5]:  # Show top 5 apps
                count = app_data['count']
                if count > 1:
                    description_parts.append(f"         - {app_name}: {count} notifications")
                else:
                    description_parts.append(f"         - {app_name}: {count} notification")
                
                # Show all notification content if available
                if app_data['full_texts']:
                    # Show each text as separate JSON
                    for i, text in enumerate(app_data['full_texts']):
                        # Remove brackets if they're just wrapping the content
                        clean_text = text.strip()
                        if clean_text.startswith('[') and clean_text.endswith(']'):
                            clean_text = clean_text[1:-1]
                        text_json = json.dumps(clean_text, ensure_ascii=False)
                        description_parts.append(f"           Text {i+1}: {text_json}")
            
            # Show timing patterns if multiple notifications
            if total_notifications > 1:
                # Calculate time span
                first_notification = sorted_data[0]
                last_notification = sorted_data[-1]
                
                first_time = first_notification.get('datetime', datetime_str).split(' ')[1]
                last_time = last_notification.get('datetime', datetime_str).split(' ')[1]
                
                if first_time != last_time:
                    description_parts.append(f"    - Time span: {first_time} to {last_time}")
                
                # Show notification frequency
                time_window_minutes = sensor_integration_time_window
                notifications_per_minute = total_notifications / time_window_minutes
                if notifications_per_minute > 1:
                    description_parts.append(f"    - Frequency: {notifications_per_minute:.1f} notifications/minute")
                else:
                    description_parts.append(f"    - Frequency: {total_notifications} notifications in {time_window_minutes} minutes")
            
            # Show session correlation if sessions are available
            if sessions:
                # Find sessions that overlap with this window
                overlapping_sessions = []
                for session in sessions:
                    if (session['start_timestamp'] <= window_end and 
                        session['end_timestamp'] >= window_start):
                        overlapping_sessions.append(session['session_id'])
                
                if overlapping_sessions:
                    if len(overlapping_sessions) == 1:
                        description_parts.append(f"    - Session activity: Session {overlapping_sessions[0]}")
                    else:
                        description_parts.append(f"    - Session activity: Sessions {', '.join(map(str, overlapping_sessions))}")
        
        return '\n'.join(description_parts)
    
    # Process data using the helper function
    narratives = process_sensor_by_timewindow(
        sensor_data, sensor_name, start_timestamp, end_timestamp, process_notifications_window
    )
    
    log_info(f"Generated {len(narratives)} applications_notifications narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives

def generate_integrated_description(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None, pid=None):
    """Generates an integrated description for the sensor."""
    if sensor_name == "applications_foreground":
        return describe_applications_foreground_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions)
    elif sensor_name == "battery":
        return describe_battery_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions)
    elif sensor_name == "bluetooth":
        return describe_bluetooth_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions)
    elif sensor_name == "keyboard":
        return describe_keyboard_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions)
    # WiFi sensors are now handled separately in the main loop with combined processing
    elif sensor_name == "screen":
        return describe_screen_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions)
    elif sensor_name == "screentext":
        return describe_screentext_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions, pid)
    elif sensor_name == "calls":
        return describe_calls_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions)
    elif sensor_name == "installations":
        return describe_installations_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions)
    elif sensor_name == "messages":
        return describe_messages_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions)
    elif sensor_name == "applications_notifications":
        return describe_applications_notifications_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions)
    elif sensor_name == "locations":
        # For locations, sensor_data should contain clustered location data
        # This will be handled specially in the main script
        return []
    else:
        # For other sensors, print sensor not supported
        log_info(f"Sensor {sensor_name} not supported")
        return []


def generate_wifi_combined_description(sensor_wifi_data, wifi_data, start_timestamp, end_timestamp, sessions=None):
    """
    Helper function to generate combined WiFi description from sensor types.
    This should be called instead of individual WiFi functions when data types are available.
    
    Args:
        sensor_wifi_data (list): List of sensor_wifi records (connection events)
        wifi_data (list): List of wifi records (network detections)
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records
        
    Returns:
        list: List of formatted wifi narrative tuples (datetime, description)
    """
    return describe_wifi_combined_integrated(sensor_wifi_data, wifi_data, start_timestamp, end_timestamp, sessions)

    
def load_session_data(session_file_path, logger=None):
    """
    Load session data from JSONL file.
    
    Args:
        session_file_path (str): Path to the sessions.jsonl file
        logger: Logger instance for detailed logging
        
    Returns:
        list: List of session records
    """
    sessions = []
    try:
        with open(session_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sessions.append(json.loads(line.strip()))
    except FileNotFoundError:
        log_warning(f"Warning: Session file {session_file_path} not found. Using estimated active time.", logger)
        return []
    except Exception as e:
        log_error(f"Error loading session data: {e}. Using estimated active time.", logger)
        return []
    
    log_info(f"Loaded {len(sessions)} session records", logger)
    return sessions

def calculate_active_time_from_sessions(sessions, start_ts, end_ts):
    """
    Calculate total active time from session data within a time window.
    Handles overlapping sessions by taking intersection with the time window.
    
    Args:
        sessions (list): List of session records
        start_ts (float): Window start timestamp
        end_ts (float): Window end timestamp
        
    Returns:
        tuple: (total_active_seconds, session_overlap_info)
    """
    total_active_seconds = 0
    overlapping_sessions = []
    
    for session in sessions:
        # Check if session overlaps with time window
        if (session['start_timestamp'] <= end_ts and session['end_timestamp'] >= start_ts):
            session_info = {
                'session_id': session['session_id'],
                'overlap_type': 'full',
                'active_seconds': 0
            }
            
            # Calculate active time within the window
            for active_period in session['active_periods']:
                period_start = max(active_period['start'], start_ts)
                period_end = min(active_period['end'], end_ts)
                
                if period_start < period_end:
                    period_duration = (period_end - period_start) / 1000.0
                    total_active_seconds += period_duration
                    session_info['active_seconds'] += period_duration
            
            # Determine overlap type
            if session['start_timestamp'] < start_ts and session['end_timestamp'] > end_ts:
                session_info['overlap_type'] = 'spans_window'
            elif session['start_timestamp'] < start_ts:
                session_info['overlap_type'] = 'extends_before'
            elif session['end_timestamp'] > end_ts:
                session_info['overlap_type'] = 'extends_after'
            
            if session_info['active_seconds'] > 0:
                overlapping_sessions.append(session_info)
    
    return total_active_seconds, overlapping_sessions

def describe_keyboard_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated keyboard typing analysis by time windows.
    Shows typing sessions with improved detection of actual character changes.
    
    Args:
        sensor_data (list): List of keyboard sensor records
        sensor_name (str): Name of the sensor (should be 'keyboard')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records for session info
        
    Returns:
        list: List of formatted keyboard narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('keyboard')
              - description: Human-readable narrative text
    """
    log_info("Generating integrated description for keyboard")
    
    if sensor_name != "keyboard" or not sensor_data:
        log_info("No keyboard data available, skipping keyboard integration")
        return []
    
    # Common placeholder texts to filter out
    PLACEHOLDER_TEXTS = {
        "Message", "message", "Search", "search", "Type a message", 
        "Enter text", "Compose", "compose", "Write something", 
        "What's on your mind?", "", "[]"
    }
    
    def is_meaningful_text(text):
        """Check if text is meaningful (not empty, not common placeholder)."""
        clean_text = text.strip('[]').strip()
        return clean_text and clean_text not in PLACEHOLDER_TEXTS
    
    def detect_typing_change(before_text, current_text):
        """
        Detect different types of typing changes.
        Returns: (change_type, is_significant, change_stats)
        """
        before_clean = before_text.strip('[]').strip()
        current_clean = current_text.strip('[]').strip()
        
        # Filter out placeholder text
        before_meaningful = is_meaningful_text(before_text)
        current_meaningful = is_meaningful_text(current_text)
        
        # Calculate detailed change statistics
        change_stats = {
            'chars_added': 0,
            'chars_deleted': 0,
            'words_added': 0,
            'words_deleted': 0,
            'before_word_count': len(before_clean.split()) if before_clean else 0,
            'current_word_count': len(current_clean.split()) if current_clean else 0,
            'before_char_count': len(before_clean),
            'current_char_count': len(current_clean)
        }
        
        if not before_meaningful and current_meaningful:
            change_stats['chars_added'] = len(current_clean)
            change_stats['words_added'] = len(current_clean.split())
            return ('typing_start', True, change_stats)
        elif before_meaningful and not current_meaningful:
            change_stats['chars_deleted'] = len(before_clean)
            change_stats['words_deleted'] = len(before_clean.split())
            return ('typing_end', True, change_stats)
        elif before_meaningful and current_meaningful:
            # Calculate character changes
            char_diff = len(current_clean) - len(before_clean)
            if char_diff > 0:
                change_stats['chars_added'] = char_diff
            elif char_diff < 0:
                change_stats['chars_deleted'] = abs(char_diff)
            
            # Calculate word changes
            word_diff = len(current_clean.split()) - len(before_clean.split())
            if word_diff > 0:
                change_stats['words_added'] = word_diff
            elif word_diff < 0:
                change_stats['words_deleted'] = abs(word_diff)
            
            if len(current_clean) > len(before_clean):
                return ('typing_continue_add', True, change_stats)
            elif len(current_clean) < len(before_clean):
                return ('typing_continue_delete', True, change_stats)
            elif current_clean != before_clean:
                return ('typing_continue_edit', True, change_stats)
            else:
                return ('no_change', False, change_stats)
        else:
            return ('no_change', False, change_stats)
    
    def find_matching_session(timestamp, sessions, window_start, window_end):
        """Find which session was active during a keyboard event."""
        session_id = None
        
        if sessions:
            for session in sessions:
                # Check if session overlaps with current window
                if not (session['start_timestamp'] <= window_end and session['end_timestamp'] >= window_start):
                    continue
                    
                # Check if timestamp falls within any active period of this session
                for period in session['active_periods']:
                    period_start = max(period['start'], window_start)
                    period_end = min(period['end'], window_end)
                    
                    if period_start <= timestamp < period_end:
                        session_id = session['session_id']
                        break
                
                if session_id:
                    break
        
        return session_id
    
    def detect_potential_typos(events):
        """
        Detect potential typos based on typing patterns.
        Returns: (typo_count, correction_patterns)
        """
        typo_count = 0
        correction_patterns = []
        
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            
            # Pattern 1: Deletion followed by addition (likely correction)
            if (current_event['change_type'] == 'typing_continue_delete' and 
                next_event['change_type'] == 'typing_continue_add' and 
                current_event['change_stats']['chars_deleted'] > 0 and
                next_event['change_stats']['chars_added'] > 0):
                
                # Check if this is a potential typo correction
                deleted_chars = current_event['change_stats']['chars_deleted']
                added_chars = next_event['change_stats']['chars_added']
                
                # If 1-3 characters were deleted and similar number added, likely a typo
                if 1 <= deleted_chars <= 3 and 1 <= added_chars <= 5:
                    # Extract actual text that was deleted and added
                    before_deletion = current_event['before_text'].strip('[]').strip()
                    after_deletion = current_event['current_text'].strip('[]').strip()
                    after_addition = next_event['current_text'].strip('[]').strip()
                    
                    # Try to identify what was deleted and what was added
                    deleted_text = ""
                    added_text = ""
                    
                    # More precise text extraction - find the difference at the end of strings
                    if len(before_deletion) > len(after_deletion):
                        if before_deletion.startswith(after_deletion):
                            deleted_text = before_deletion[len(after_deletion):]
                        elif after_deletion and before_deletion.endswith(after_deletion):
                            deleted_text = before_deletion[:-len(after_deletion)]
                        else:
                            # Find common prefix and suffix to isolate the change
                            common_prefix = 0
                            for i in range(min(len(before_deletion), len(after_deletion))):
                                if before_deletion[i] == after_deletion[i]:
                                    common_prefix += 1
                                else:
                                    break
                            
                            if common_prefix < len(before_deletion):
                                deleted_text = before_deletion[common_prefix:]
                            else:
                                deleted_text = f"{deleted_chars} chars"
                    
                    if len(after_addition) > len(after_deletion):
                        if after_addition.startswith(after_deletion):
                            added_text = after_addition[len(after_deletion):]
                        elif after_deletion and after_addition.endswith(after_deletion):
                            added_text = after_addition[:-len(after_deletion)]
                        else:
                            # Find common prefix and suffix to isolate the change
                            common_prefix = 0
                            for i in range(min(len(after_addition), len(after_deletion))):
                                if after_addition[i] == after_deletion[i]:
                                    common_prefix += 1
                                else:
                                    break
                            
                            if common_prefix < len(after_addition):
                                added_text = after_addition[common_prefix:]
                            else:
                                added_text = f"{added_chars} chars"
                    
                    # Only count as typo if the deleted and added text are actually different
                    if deleted_text and added_text and deleted_text.strip() != added_text.strip():
                        typo_count += 1
                        
                        correction_patterns.append({
                            'type': 'delete_then_add',
                            'chars_deleted': deleted_chars,
                            'chars_added': added_chars,
                            'deleted_text': deleted_text[:10] if deleted_text else f"{deleted_chars} chars",  # Limit length
                            'added_text': added_text[:10] if added_text else f"{added_chars} chars",  # Limit length
                            'timestamp': current_event['timestamp']
                        })
            
            # Pattern 2: Multiple consecutive deletions (likely backspacing due to error)
            elif (current_event['change_type'] == 'typing_continue_delete' and 
                  current_event['change_stats']['chars_deleted'] >= 3):
                
                # Check if followed by adding text back
                if (i + 1 < len(events) and 
                    events[i + 1]['change_type'] == 'typing_continue_add'):
                    
                    # Extract actual text that was deleted and added back
                    before_deletion = current_event['before_text'].strip('[]').strip()
                    after_deletion = current_event['current_text'].strip('[]').strip()
                    after_addition = events[i + 1]['current_text'].strip('[]').strip()
                    
                    # Try to identify what was deleted and what was added
                    deleted_text = ""
                    added_text = ""
                    
                    if len(before_deletion) > len(after_deletion):
                        if before_deletion.startswith(after_deletion):
                            deleted_text = before_deletion[len(after_deletion):]
                        elif after_deletion and before_deletion.endswith(after_deletion):
                            deleted_text = before_deletion[:-len(after_deletion)]
                        else:
                            # Find common prefix to isolate the change
                            common_prefix = 0
                            for j in range(min(len(before_deletion), len(after_deletion))):
                                if before_deletion[j] == after_deletion[j]:
                                    common_prefix += 1
                                else:
                                    break
                            
                            if common_prefix < len(before_deletion):
                                deleted_text = before_deletion[common_prefix:]
                            else:
                                deleted_text = f"{current_event['change_stats']['chars_deleted']} chars"
                    
                    if len(after_addition) > len(after_deletion):
                        if after_addition.startswith(after_deletion):
                            added_text = after_addition[len(after_deletion):]
                        elif after_deletion and after_addition.endswith(after_deletion):
                            added_text = after_addition[:-len(after_deletion)]
                        else:
                            # Find common prefix to isolate the change
                            common_prefix = 0
                            for j in range(min(len(after_addition), len(after_deletion))):
                                if after_addition[j] == after_deletion[j]:
                                    common_prefix += 1
                                else:
                                    break
                            
                            if common_prefix < len(after_addition):
                                added_text = after_addition[common_prefix:]
                            else:
                                added_text = f"{events[i + 1]['change_stats']['chars_added']} chars"
                    
                    # Only count as typo if the deleted and added text are meaningfully different
                    # Also check if it's a substantial correction (not just whitespace)
                    if (deleted_text and added_text and 
                        deleted_text.strip() != added_text.strip() and
                        len(deleted_text.strip()) > 0 and len(added_text.strip()) > 0):
                        
                        typo_count += 1
                        
                        correction_patterns.append({
                            'type': 'bulk_delete_correction',
                            'chars_deleted': current_event['change_stats']['chars_deleted'],
                            'chars_added': events[i + 1]['change_stats']['chars_added'],
                            'deleted_text': deleted_text[:15] if deleted_text else f"{current_event['change_stats']['chars_deleted']} chars",
                            'added_text': added_text[:15] if added_text else f"{events[i + 1]['change_stats']['chars_added']} chars",
                            'timestamp': current_event['timestamp']
                        })
        
        return typo_count, correction_patterns
    
    def calculate_typing_speed(events, total_duration_seconds):
        """
        Calculate typing speed metrics.
        Returns: (chars_per_minute, words_per_minute, net_chars_per_minute)
        """
        if total_duration_seconds <= 0:
            return 0, 0, 0
        
        # Calculate total characters typed (gross)
        total_chars_typed = sum(event['change_stats']['chars_added'] for event in events)
        
        # Calculate total words typed (gross)  
        total_words_typed = sum(event['change_stats']['words_added'] for event in events)
        
        # Calculate net characters (typed - deleted)
        total_chars_deleted = sum(event['change_stats']['chars_deleted'] for event in events)
        net_chars_typed = total_chars_typed - total_chars_deleted
        
        # Convert to per-minute rates
        duration_minutes = total_duration_seconds / 60.0
        
        chars_per_minute = total_chars_typed / duration_minutes if duration_minutes > 0 else 0
        words_per_minute = total_words_typed / duration_minutes if duration_minutes > 0 else 0
        net_chars_per_minute = net_chars_typed / duration_minutes if duration_minutes > 0 else 0
        
        return chars_per_minute, words_per_minute, net_chars_per_minute
    
    def process_keyboard_window(window_data, datetime_str, window_start, window_end, sessions_data):
        """Process keyboard data for a single time window with typing detection."""
        if not window_data:
            return None
        
        # Group typing sessions by package_name and track typing events
        app_typing_sessions = {}
        
        # Sort window data by timestamp to process in chronological order
        sorted_data = sorted(window_data, key=lambda x: x['timestamp'])
        
        for record in sorted_data:
            package_name = record.get('package_name', 'Unknown')
            before_text = record.get('before_text', '')
            current_text = record.get('current_text', '')
            is_password = record.get('is_password', 0)
            timestamp = record.get('timestamp', 0)
            record_datetime = record.get('datetime', datetime_str)
            
            # Skip password typing
            if is_password == 1:
                continue
            
            # Map package name to application name if available
            app_name = application_name_list.get(package_name, package_name)

            #check if app is blacklisted - compare package names
            if any(package_name.lower() == app.lower() for app in blacklist_apps):
                continue
            
            # Find matching session
            session_id = find_matching_session(timestamp, sessions_data, window_start, window_end)
            
            # Detect typing change type
            change_type, is_significant, change_stats = detect_typing_change(before_text, current_text)
            
            if not is_significant:
                continue
            
            if package_name not in app_typing_sessions:
                app_typing_sessions[package_name] = {
                    'app_name': app_name,
                    'events': [],
                    'session_id': session_id
                }
            
            app_typing_sessions[package_name]['events'].append({
                'timestamp': timestamp,
                'datetime': record_datetime,
                'before_text': before_text,
                'current_text': current_text,
                'change_type': change_type,
                'change_stats': change_stats,
                'session_id': session_id
            })
        
        # Process typing sessions to create meaningful descriptions
        keyboard_descriptions = []
        
        for package_name, session_data in app_typing_sessions.items():
            app_name = session_data['app_name']
            events = session_data['events']
            
            if not events:
                continue
            
            # Group events into typing periods based on change types
            typing_periods = []
            current_period = None
            
            for event in events:
                change_type = event['change_type']
                
                if change_type == 'typing_start':
                    # Start new typing period
                    if current_period and is_meaningful_text(current_period['final_text']):
                        typing_periods.append(current_period)
                    
                    current_period = {
                        'start_time': event['datetime'],
                        'end_time': event['datetime'],
                        'start_timestamp': event['timestamp'],
                        'end_timestamp': event['timestamp'],
                        'final_text': event['current_text'],
                        'session_id': event['session_id'],
                        'events_count': 1,
                        'events': [event]
                    }
                
                elif change_type in ['typing_continue_add', 'typing_continue_delete', 'typing_continue_edit']:
                    # Continue current typing period
                    if current_period:
                        current_period['end_time'] = event['datetime']
                        current_period['end_timestamp'] = event['timestamp']
                        current_period['final_text'] = event['current_text']
                        current_period['events_count'] += 1
                        current_period['events'].append(event)
                    else:
                        # Start new period if no current period (edge case)
                        current_period = {
                            'start_time': event['datetime'],
                            'end_time': event['datetime'],
                            'start_timestamp': event['timestamp'],
                            'end_timestamp': event['timestamp'],
                            'final_text': event['current_text'],
                            'session_id': event['session_id'],
                            'events_count': 1,
                            'events': [event]
                        }
                
                elif change_type == 'typing_end':
                    # End current typing period
                    if current_period and is_meaningful_text(current_period['final_text']):
                        typing_periods.append(current_period)
                        current_period = None
            
            # Add the last period if it exists and has meaningful content
            if current_period and is_meaningful_text(current_period['final_text']):
                typing_periods.append(current_period)
            
            # Create descriptions for each typing period
            for period in typing_periods:
                final_text = period['final_text'].strip('[]').strip()
                if not is_meaningful_text(final_text):
                    continue
                
                # Calculate typing duration
                duration_ms = period['end_timestamp'] - period['start_timestamp']
                duration_seconds = duration_ms / 1000.0
                
                # Filter out typing periods that are too short (less than 1 second) or suspiciously long (over 30 minutes)
                if duration_seconds < 1.0 or duration_seconds > 1800:
                    continue
                
                # Filter out single character typing that took too long (likely a timestamp bug)
                if len(final_text) <= 2 and duration_seconds > 60:
                    continue
                
                # Only include typing periods that START within this window
                if period['start_timestamp'] < window_start:
                    continue
                
                # Check if this typing session extends beyond the current window
                extends_note = ""
                if period['end_timestamp'] > window_end:
                    # Calculate how many windows this typing session extends into
                    window_size_ms = 60 * 60 * 1000  # 60 minutes in milliseconds
                    windows_after_current = int((period['end_timestamp'] - window_end) / window_size_ms) + 1
                    
                    # Format the note with correct pluralization
                    if windows_after_current == 1:
                        extends_note = f" → extends to the following window until {period['end_time'].split(' ')[1]}"
                    else:
                        extends_note = f" → extends to the following {windows_after_current} windows until {period['end_time'].split(' ')[1]}"
                
                if duration_seconds < 60:
                    duration_str = f"{int(duration_seconds)} seconds"
                else:
                    duration_mins = int(duration_seconds / 60)
                    remainder_secs = int(duration_seconds % 60)
                    if remainder_secs > 0:
                        duration_str = f"{duration_mins} minute{'s' if duration_mins > 1 else ''} {remainder_secs} seconds"
                    else:
                        duration_str = f"{duration_mins} minute{'s' if duration_mins > 1 else ''}"
                
                # Calculate advanced typing metrics
                period_events = period.get('events', [])
                
                # Calculate typing speed
                chars_per_minute, words_per_minute, net_chars_per_minute = calculate_typing_speed(period_events, duration_seconds)
                
                # Calculate words deleted
                total_words_deleted = sum(event['change_stats']['words_deleted'] for event in period_events)
                total_chars_deleted = sum(event['change_stats']['chars_deleted'] for event in period_events)
                
                # Detect potential typos
                typo_count, correction_patterns = detect_potential_typos(period_events)
                
                # Create description with session info if available
                if app_name != package_name:
                    app_desc = f" in {app_name}"
                else:
                    app_desc = f" in {package_name}"
                
                # Add session info if available
                session_info = ""
                if period['session_id']:
                    session_info = f" (Session {period['session_id']})"
                
                # Clean and format the text for display
                clean_text = final_text.replace('\n', ' ').strip()
                
                # Keep original text with newlines for JSON
                original_text = final_text.strip()
                
                # Truncate very long text for readability (keep full text in a separate field)
                display_text = clean_text
                if len(clean_text) > 100:
                    display_text = clean_text[:97] + "..."
                
                # Create enhanced description with typing metrics and extends note
                description = (
                    f"{period['start_time']} | Typed{app_desc}{session_info} "
                    f"for {duration_str}: \"{display_text}\"{extends_note}"
                )
                
                keyboard_descriptions.append({
                    'datetime': period['start_time'],
                    'description': description,
                    'app_name': app_name,
                    'package_name': package_name,
                    'duration_seconds': duration_seconds,
                    'start_timestamp': period['start_timestamp'],
                    'end_timestamp': period['end_timestamp'],
                    'session_id': period['session_id'],
                    'full_text': clean_text,           # For display (no newlines)
                    'original_text': original_text,    # For JSON (preserves newlines)
                    'events_count': period['events_count'],
                    # New enhanced metrics
                    'chars_per_minute': chars_per_minute,
                    'words_per_minute': words_per_minute,
                    'net_chars_per_minute': net_chars_per_minute,
                    'words_deleted': total_words_deleted,
                    'chars_deleted': total_chars_deleted,
                    'typo_count': typo_count,
                    'correction_patterns': correction_patterns
                })
        
        # Combine adjacent typing sessions in the same app that are close in time
        if keyboard_descriptions:
            # Sort by timestamp first
            keyboard_descriptions.sort(key=lambda x: x['start_timestamp'])
            
            # Combine adjacent sessions
            combined_descriptions = []
            for desc in keyboard_descriptions:
                if (combined_descriptions and 
                    combined_descriptions[-1]['app_name'] == desc['app_name'] and
                    combined_descriptions[-1]['session_id'] == desc['session_id'] and
                    desc['start_timestamp'] - combined_descriptions[-1]['end_timestamp'] <= 45000):  # 45 seconds (same as screen session threshold)
                    
                    # Combine with previous session
                    last_desc = combined_descriptions[-1]
                    
                    # Combine text with appropriate separator
                    last_text = last_desc['original_text'].strip()
                    current_text = desc['original_text'].strip()
                    
                    if last_text and current_text:
                        # Add space if neither text ends/starts with punctuation
                        if (not last_text[-1] in '.,!?;:\n' and 
                            not current_text[0] in '.,!?;:\n' and
                            not last_text.endswith(' ') and
                            not current_text.startswith(' ')):
                            separator = ' '
                        else:
                            separator = ''
                        last_desc['original_text'] = f"{last_text}{separator}{current_text}"
                        last_desc['full_text'] = last_desc['original_text'].replace('\n', ' ').strip()
                    elif current_text:
                        last_desc['original_text'] = current_text
                        last_desc['full_text'] = current_text.replace('\n', ' ').strip()
                    
                    # Update other properties
                    last_desc['end_timestamp'] = desc['end_timestamp']
                    last_desc['duration_seconds'] += desc['duration_seconds']
                    last_desc['events_count'] += desc['events_count']
                    
                    # Combine metrics (weighted average for rates, sum for counts)
                    total_duration = last_desc['duration_seconds']
                    last_weight = (total_duration - desc['duration_seconds']) / total_duration
                    current_weight = desc['duration_seconds'] / total_duration
                    
                    last_desc['chars_per_minute'] = (last_desc['chars_per_minute'] * last_weight + 
                                                    desc['chars_per_minute'] * current_weight)
                    last_desc['words_per_minute'] = (last_desc['words_per_minute'] * last_weight + 
                                                    desc['words_per_minute'] * current_weight)
                    last_desc['net_chars_per_minute'] = (last_desc['net_chars_per_minute'] * last_weight + 
                                                        desc['net_chars_per_minute'] * current_weight)
                    
                    # Sum the deletion counts and typos
                    last_desc['words_deleted'] += desc['words_deleted']
                    last_desc['chars_deleted'] += desc['chars_deleted']
                    last_desc['typo_count'] += desc['typo_count']
                    last_desc['correction_patterns'].extend(desc['correction_patterns'])
                    
                    # Update description
                    if last_desc['duration_seconds'] < 60:
                        duration_str = f"{int(last_desc['duration_seconds'])} seconds"
                    else:
                        duration_mins = int(last_desc['duration_seconds'] / 60)
                        remainder_secs = int(last_desc['duration_seconds'] % 60)
                        if remainder_secs > 0:
                            duration_str = f"{duration_mins} minute{'s' if duration_mins > 1 else ''} {remainder_secs} seconds"
                        else:
                            duration_str = f"{duration_mins} minute{'s' if duration_mins > 1 else ''}"
                    
                    display_text = last_desc['full_text']
                    if len(display_text) > 100:
                        display_text = display_text[:97] + "..."
                    
                    app_name = last_desc['app_name']
                    package_name = last_desc['package_name']
                    if app_name != package_name:
                        app_desc = f" in {app_name}"
                    else:
                        app_desc = f" in {package_name}"
                    
                    session_info = ""
                    if last_desc['session_id']:
                        session_info = f" (Session {last_desc['session_id']})"
                    
                    last_desc['description'] = (
                        f"{last_desc['datetime']} | Typed{app_desc}{session_info} "
                        f"for {duration_str}: \"{display_text}\""
                    )
                    
                else:
                    combined_descriptions.append(desc)
            
            keyboard_descriptions = combined_descriptions
        
        # If we have keyboard descriptions, create a summary
        if keyboard_descriptions:
            # Sort by timestamp
            keyboard_descriptions.sort(key=lambda x: x['start_timestamp'])
            
            # Create window summary
            total_typing_time = sum(desc['duration_seconds'] for desc in keyboard_descriptions)
            total_events = sum(desc['events_count'] for desc in keyboard_descriptions)
            
            # Calculate aggregated typing statistics
            total_words_deleted = sum(desc['words_deleted'] for desc in keyboard_descriptions)
            total_chars_deleted = sum(desc['chars_deleted'] for desc in keyboard_descriptions)
            total_typos = sum(desc['typo_count'] for desc in keyboard_descriptions)
            
            # Calculate average typing speed (weighted by duration)
            if total_typing_time > 0:
                avg_chars_per_minute = sum(desc['chars_per_minute'] * desc['duration_seconds'] for desc in keyboard_descriptions) / total_typing_time
                avg_words_per_minute = sum(desc['words_per_minute'] * desc['duration_seconds'] for desc in keyboard_descriptions) / total_typing_time
                avg_net_chars_per_minute = sum(desc['net_chars_per_minute'] * desc['duration_seconds'] for desc in keyboard_descriptions) / total_typing_time
            else:
                avg_chars_per_minute = 0
                avg_words_per_minute = 0
                avg_net_chars_per_minute = 0
            
            description_parts = [f"keyboard | Typing Activity"]
            description_parts.append(f"    - Total typing sessions: {len(keyboard_descriptions)}")
            description_parts.append(f"    - Total keyboard events: {total_events}")
            
            if total_typing_time >= 60:
                typing_mins = int(total_typing_time / 60)
                typing_secs = int(total_typing_time % 60)
                description_parts.append(f"    - Total typing time: {typing_mins} min {typing_secs} sec")
            else:
                description_parts.append(f"    - Total typing time: {int(total_typing_time)} seconds")
            
            # Add enhanced typing metrics
            if avg_chars_per_minute > 0:
                description_parts.append(f"    - Average typing speed: {avg_chars_per_minute:.1f} chars/min, {avg_words_per_minute:.1f} words/min")
                description_parts.append(f"    - Net typing speed: {avg_net_chars_per_minute:.1f} chars/min (after deletions)")
            
            if total_words_deleted > 0:
                description_parts.append(f"    - Words deleted: {total_words_deleted} ({total_chars_deleted} characters)")
            
            if total_typos > 0:
                description_parts.append(f"    - Potential typos detected: {total_typos}")
            
            # Group keyboard descriptions by app only
            app_groups = {}
            for desc in keyboard_descriptions:
                app_name = desc['app_name']
                
                if app_name not in app_groups:
                    app_groups[app_name] = {
                        'app_name': app_name,
                        'typing_sessions': []
                    }
                
                app_groups[app_name]['typing_sessions'].append(desc)
            
            # Add detailed typing sessions with mixed format (JSON for text, human-readable for metrics)
            description_parts.append(f"    - Typing sessions:")
            
            typing_session_counter = 1
            
            for app_name, app_data in app_groups.items():
                typing_sessions = app_data['typing_sessions']
                
                # Sort typing sessions by timestamp for consistent ordering
                typing_sessions.sort(key=lambda x: x['start_timestamp'])
                
                for desc in typing_sessions:
                    session_id = typing_session_counter
                    typing_session_counter += 1
                    
                    # Human-readable metrics
                    time_str = desc['datetime'].split(' ')[1]
                    duration_str = f"{round(desc['duration_seconds'], 1)} seconds"
                    chars_per_min = f"{round(desc['chars_per_minute'], 1)} chars/min"
                    words_per_min = f"{round(desc['words_per_minute'], 1)} words/min"
                    net_chars_per_min = f"{round(desc['net_chars_per_minute'], 1)} net chars/min"
                    chars_deleted = f"{desc['chars_deleted']} characters"
                    words_deleted = f"{desc['words_deleted']} words"
                    typo_count = f"{desc['typo_count']} typos"
                    
                    # Add session header with human-readable metrics
                    description_parts.append(f"        - Typing Session: {session_id}")
                    description_parts.append(f"            - App: {app_name}")
                    description_parts.append(f"            - Time: {time_str}")
                    description_parts.append(f"            - Duration: {duration_str}")
                    description_parts.append(f"            - Typing speed: {chars_per_min}, {words_per_min}")
                    description_parts.append(f"            - Net speed: {net_chars_per_min}")
                    description_parts.append(f"            - Deletions: {chars_deleted} ({words_deleted})")
                    description_parts.append(f"            - Typos detected: {typo_count}")
                    
                    # Add full text before typos section
                    description_parts.append(f"            - Full Text: {json.dumps(desc['original_text'], ensure_ascii=False)}")
                    
                    # Add typo details in human-readable format if any
                    if desc['correction_patterns']:
                        description_parts.append(f"            - Typos:")
                        for i, pattern in enumerate(desc['correction_patterns'], 1):
                            description_parts.append(f"                - Typo {i}: {pattern['type'].replace('_', ' ')}")
                            
                            # Add deleted text in JSON format
                            deleted_chars = pattern['chars_deleted']
                            deleted_text = f"{deleted_chars} char{'s' if deleted_chars != 1 else ''} deleted"
                            description_parts.append(f"                    - Deleted text: {json.dumps(pattern.get('deleted_text', ''), ensure_ascii=False)} ({deleted_text})")
                            
                            # Add added text in JSON format
                            added_chars = pattern['chars_added']
                            added_text = f"{added_chars} char{'s' if added_chars != 1 else ''} added"
                            description_parts.append(f"                    - Added text: {json.dumps(pattern.get('added_text', ''), ensure_ascii=False)} ({added_text})")
                    
                    description_parts.append("")  # Add empty line for readability
            
            return '\n'.join(description_parts)
        
        return None
    
    # Create a wrapper function that includes sessions in its closure
    def process_keyboard_window_with_sessions(window_data, datetime_str, window_start, window_end):
        return process_keyboard_window(window_data, datetime_str, window_start, window_end, sessions)
    
    # Process data using the helper function
    narratives = process_sensor_by_timewindow(
        sensor_data, sensor_name, start_timestamp, end_timestamp, process_keyboard_window_with_sessions
    )
    
    log_info(f"Generated {len(narratives)} keyboard narratives with typing detection (window size: {sensor_integration_time_window} minutes)")
    return narratives

def sort_narratives_by_time_window_and_sensor_order(all_narratives):
    """
    Sort narratives by time window and then by category-based sensor order within each window.
    
    Categories and order:
    1. Environmental context: locations, wifi, bluetooth
    2. Communication events: notifications, calls, messages  
    3. Device state: battery, installations
    4. Engagement signals: screen, applications, keyboard, screentext
    
    Args:
        all_narratives (list): List of narrative dictionaries with keys:
                              - unix_timestamp: Unix timestamp in milliseconds
                              - sensor_type: Sensor type string
                              - description: Human-readable narrative text
        
    Returns:
        list: Sorted list of narrative dictionaries with category headers
    """
    from collections import defaultdict
    
    # Define sensor categories and their order
    sensor_categories = {
        # Environmental context
        'locations': 'environmental_context',
        'wifi': 'environmental_context',
        'bluetooth': 'environmental_context',
        # Communication events
        'notifications': 'communication_events',
        'applications_notifications': 'communication_events',  # Added missing sensor type
        'calls': 'communication_events',
        'messages': 'communication_events',
        # Device state
        'battery': 'device_state',
        'installations': 'device_state',
        # Engagement signals
        'screen': 'engagement_signals',
        'applications': 'engagement_signals',
        'keyboard': 'engagement_signals',
        'screentext': 'engagement_signals'
    }
    
    # Define category order and display names
    category_order = [
        'environmental_context',
        'communication_events', 
        'device_state',
        'engagement_signals'
    ]
    
    category_display_names = {
        'environmental_context': 'Environmental Context',
        'communication_events': 'Communication Events',
        'device_state': 'Device State',
        'engagement_signals': 'Engagement Signals'
    }
    
    def get_sensor_type(narrative_dict):
        """Extract sensor type from narrative dictionary."""
        return narrative_dict.get('sensor_type', 'other')
    
    def get_sensor_category(narrative_dict):
        """Get category for sensor type."""
        sensor_type = get_sensor_type(narrative_dict)
        return sensor_categories.get(sensor_type, 'other')
    
    def get_category_priority(narrative_dict):
        """Get category priority for sorting."""
        category = get_sensor_category(narrative_dict)
        try:
            return category_order.index(category)
        except ValueError:
            return len(category_order)  # Put unknown categories at the end
    
    # Group narratives by time window using unix timestamp
    time_window_groups = defaultdict(list)
    
    for narrative_dict in all_narratives:
        # Convert unix timestamp to datetime string for grouping compatibility
        unix_ts = narrative_dict['unix_timestamp']
        dt = pd.to_datetime(unix_ts, unit='ms', utc=True)
        local_time = dt.tz_convert(timezone)
        datetime_str = local_time.strftime('%Y-%m-%d %H:%M:%S')
        time_window_groups[datetime_str].append(narrative_dict)
    
    # Sort within each time window by category priority and create formatted output
    sorted_narratives = []
    window_number = 1
    
    for time_window in sorted(time_window_groups.keys()):
        window_narratives = time_window_groups[time_window]
        
        # Sort this window's narratives by category priority
        window_narratives.sort(key=lambda x: get_category_priority(x))
        
        # Add newline before time window (except for the first one)
        if sorted_narratives:
            sorted_narratives.append({
                'unix_timestamp': window_narratives[0]['unix_timestamp'],
                'sensor_type': 'header',
                'description': ""
            })
        
        # Create time range header with window number and day information
        time_window_dt = pd.to_datetime(time_window)
        next_window_dt = time_window_dt + pd.Timedelta(minutes=sensor_integration_time_window)
        
        # Check if the window spans across days
        if time_window_dt.date() == next_window_dt.date():
            # Same day
            day_name = time_window_dt.strftime('%A')
            day_info = f"Day {time_window_dt.strftime('%Y-%m-%d')} ({day_name})"
            time_range = f"{time_window_dt.strftime('%H:%M:%S')} - {next_window_dt.strftime('%H:%M:%S')}"
        else:
            # Crosses midnight to next day
            start_day_name = time_window_dt.strftime('%A')
            end_day_name = next_window_dt.strftime('%A')
            day_info = f"Day {time_window_dt.strftime('%Y-%m-%d')} ({start_day_name}) to {next_window_dt.strftime('%Y-%m-%d')} ({end_day_name})"
            time_range = f"{time_window_dt.strftime('%H:%M:%S')} - {next_window_dt.strftime('%H:%M:%S')}"
        
        time_range_header = f"Window {window_number}\n{day_info}\n{time_range}"
        sorted_narratives.append({
            'unix_timestamp': window_narratives[0]['unix_timestamp'],
            'sensor_type': 'header',
            'description': time_range_header
        })
        
        window_number += 1
        
        # Group by category and add headers
        current_category = None
        for narrative_dict in window_narratives:
            category = get_sensor_category(narrative_dict)
            
            # Add category header if this is a new category
            if category != current_category and category in category_display_names:
                current_category = category
                # Add newline before category header
                sorted_narratives.append({
                    'unix_timestamp': narrative_dict['unix_timestamp'],
                    'sensor_type': 'header',
                    'description': ""
                })
                category_header = category_display_names[category]
                sorted_narratives.append({
                    'unix_timestamp': narrative_dict['unix_timestamp'],
                    'sensor_type': 'header',
                    'description': category_header
                })
            
            # Add dash prefix to description (description already contains sensor type)
            description = narrative_dict['description']
            formatted_description = f"- {description}"
            
            sorted_narratives.append({
                'unix_timestamp': narrative_dict['unix_timestamp'],
                'sensor_type': narrative_dict['sensor_type'],
                'description': formatted_description
            })
    
    return sorted_narratives

def describe_screen_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated screen status analysis by time windows.
    Focuses on screen sessions that start with activation (on/unlocked) and end with deactivation (off/locked).
    
    Args:
        sensor_data (list): List of screen sensor records
        sensor_name (str): Name of the sensor (should be 'screen')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records for session correlation
        
    Returns:
        list: List of formatted screen narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('screen')
              - description: Human-readable narrative text
    """
    log_info("Generating integrated description for screen")
    
    if sensor_name != "screen" or not sensor_data:
        log_info("No screen data available, skipping screen integration")
        return []
    
    # Screen status mapping
    screen_statuses = {
        0: "turned off",
        1: "turned on",
        2: "locked",
        3: "unlocked"
    }
    
    def process_screen_window(window_data, datetime_str, window_start, window_end):
        """Process screen data for a single time window focusing on screen status patterns."""
        if not window_data:
            return None
        
        # Sort window data by timestamp
        sorted_data = sorted(window_data, key=lambda x: x['timestamp'])
        
        # Analyze all screen status events (including locked/unlocked for counting)
        status_counts = {}
        screen_events = []
        all_events = []  # For complete event breakdown
        
        for record in sorted_data:
            if 'screen_status' in record and record['screen_status'] in screen_statuses:
                status = screen_statuses[record['screen_status']]
                raw_status = record['screen_status']
                
                # Count all status types for breakdown
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Store all events for breakdown
                all_events.append({
                    'datetime': record['datetime'],
                    'status': status,
                    'raw_status': raw_status,
                    'timestamp': record['timestamp']
                })
                
                # Only store on/off events for session processing
                if raw_status in [0, 1]:
                    screen_events.append({
                        'datetime': record['datetime'],
                        'status': status,
                        'raw_status': raw_status,
                        'timestamp': record['timestamp']
                    })
        
        # Create screen usage sessions (activation to deactivation)
        # Only process screen on (1) and screen off (0) events, ignore locked/unlocked
        # If screen turns off and on within 45 seconds, consider it the same session
        screen_sessions = []
        current_session = None
        pending_off_event = None
        
        # Check if there's a carryover session from previous window
        # This happens when the first event in this window is a screen "off" without a corresponding "on"
        if screen_events and screen_events[0]['raw_status'] == 0:
            # First event is screen off, so there must be a session from previous window
            carryover_session = {
                'start_datetime': "Previous window",
                'start_timestamp': window_start,
                'start_status': "carried over",
                'end_datetime': screen_events[0]['datetime'],
                'end_timestamp': screen_events[0]['timestamp'],
                'end_status': screen_events[0]['status'],
                'duration_seconds': (screen_events[0]['timestamp'] - window_start) / 1000.0,
                'is_carryover': True
            }
            screen_sessions.append(carryover_session)
            # Skip the first off event since we handled it as carryover end
            screen_events = screen_events[1:]
        
        for event in screen_events:
            if event['raw_status'] == 1:  # Screen turned on
                if pending_off_event is not None:
                    # Check if this "on" event is within 45 seconds of the pending "off" event
                    gap_seconds = (event['timestamp'] - pending_off_event['timestamp']) / 1000.0
                    
                    if gap_seconds <= 45:
                        # Continue the same session - ignore the brief off period
                        pending_off_event = None
                        continue
                    else:
                        # Gap is too long, finalize the previous session
                        if current_session is not None:
                            current_session['end_datetime'] = pending_off_event['datetime']
                            current_session['end_timestamp'] = pending_off_event['timestamp']
                            current_session['end_status'] = pending_off_event['status']
                            current_session['duration_seconds'] = (pending_off_event['timestamp'] - current_session['start_timestamp']) / 1000.0
                            screen_sessions.append(current_session)
                        pending_off_event = None
                
                # Start a new session
                current_session = {
                    'start_datetime': event['datetime'],
                    'start_timestamp': event['timestamp'],
                    'start_status': event['status'],
                    'end_datetime': None,
                    'end_timestamp': None,
                    'end_status': None,
                    'duration_seconds': 0,
                    'is_carryover': False
                }
                
            elif event['raw_status'] == 0:  # Screen turned off
                # Don't immediately end the session, wait to see if screen comes back on quickly
                if current_session is not None:
                    pending_off_event = event
            
            # Ignore locked (status 2) and unlocked (status 3) events
        
        # Handle any remaining pending off event
        if pending_off_event is not None and current_session is not None:
            current_session['end_datetime'] = pending_off_event['datetime']
            current_session['end_timestamp'] = pending_off_event['timestamp']
            current_session['end_status'] = pending_off_event['status']
            current_session['duration_seconds'] = (pending_off_event['timestamp'] - current_session['start_timestamp']) / 1000.0
            screen_sessions.append(current_session)
            current_session = None  # Mark as finalized
        
        # If there's an unclosed session, it's ongoing
        if current_session is not None:
            current_session['end_datetime'] = None
            current_session['end_timestamp'] = None
            current_session['end_status'] = "session ongoing"
            current_session['duration_seconds'] = 0
            current_session['is_carryover'] = False
            screen_sessions.append(current_session)
        
        # Calculate time between screen activations
        activation_intervals = []
        for i in range(1, len(screen_sessions)):
            if screen_sessions[i-1]['end_timestamp'] and screen_sessions[i]['start_timestamp']:
                interval = (screen_sessions[i]['start_timestamp'] - screen_sessions[i-1]['end_timestamp']) / 1000.0
                activation_intervals.append(interval)
        
        # Generate description
        description_parts = [f"screen | Screen Status Analysis"]
        
        # Show event breakdown (all event types including locked/unlocked)
        if status_counts:
            event_breakdown = []
            for status, count in status_counts.items():
                event_breakdown.append(f"{count} {status}")
            description_parts.append(f"    - Event breakdown: {', '.join(event_breakdown)}")
        
        # Screen session information (count only sessions that started in this window)
        new_activations = len([s for s in screen_sessions if not s.get('is_carryover', False) and s['start_timestamp'] >= window_start])
        if screen_sessions:
            description_parts.append(f"    - Screen activations: {new_activations}")
            
            # Average time between activations (only if there are multiple new activations)
            if new_activations > 1 and len(activation_intervals) > 0:
                avg_interval = sum(activation_intervals) / len(activation_intervals)
                if avg_interval >= 60:
                    interval_mins = int(avg_interval // 60)
                    interval_secs = int(avg_interval % 60)
                    interval_str = f"{interval_mins} min {interval_secs} sec" if interval_secs > 0 else f"{interval_mins} min"
                else:
                    interval_str = f"{int(avg_interval)} sec"
                description_parts.append(f"    - Average time between activations: {interval_str}")
        
        # Show recent screen sessions as timeline
        if len(screen_sessions) > 0:
            description_parts.append(f"    - Screen timelines:")
            # Show all sessions
            for session in screen_sessions:
                if session['end_status'] == "session ongoing":
                    # Check if session started in previous window
                    if session.get('is_carryover', False) or session['start_timestamp'] < window_start:
                        # Session carried over from previous window
                        # Calculate duration from current window start to window end
                        ongoing_duration = (window_end - window_start) / 1000.0
                        if ongoing_duration >= 60:
                            ongoing_duration_mins = int(ongoing_duration // 60)
                            ongoing_duration_secs = int(ongoing_duration % 60)
                            if ongoing_duration_secs > 0:
                                ongoing_duration_str = f"{ongoing_duration_mins} min {ongoing_duration_secs} sec"
                            else:
                                ongoing_duration_str = f"{ongoing_duration_mins} min"
                        else:
                            ongoing_duration_str = f"{int(ongoing_duration)} sec"
                        
                        description_parts.append(f"         - Previous window → ongoing | Screen session (duration this window: {ongoing_duration_str})")
                    else:
                        # Session started in current window
                        # Calculate duration from session start to end of current window
                        ongoing_duration = (window_end - session['start_timestamp']) / 1000.0
                        if ongoing_duration >= 60:
                            ongoing_duration_mins = int(ongoing_duration // 60)
                            ongoing_duration_secs = int(ongoing_duration % 60)
                            if ongoing_duration_secs > 0:
                                ongoing_duration_str = f"{ongoing_duration_mins} min {ongoing_duration_secs} sec"
                            else:
                                ongoing_duration_str = f"{ongoing_duration_mins} min"
                        else:
                            ongoing_duration_str = f"{int(ongoing_duration)} sec"
                        
                        # Format ongoing session
                        start_time = session['start_datetime'].split(' ')[1]  # Get time part
                        start_date = session['start_datetime'].split(' ')[0]  # Get date part
                        description_parts.append(f"         - {start_date} {start_time} → ongoing | Screen session (duration: {ongoing_duration_str})")
                else:
                    # Show session as timeline: start → end (duration) on one line
                    duration = session['duration_seconds']
                    if duration >= 60:
                        duration_mins = int(duration // 60)
                        duration_secs = int(duration % 60)
                        if duration_secs > 0:
                            duration_str = f"{duration_mins} min {duration_secs} sec"
                        else:
                            duration_str = f"{duration_mins} min"
                    else:
                        duration_str = f"{int(duration)} sec"
                    
                    # Handle carryover sessions that ended in this window
                    if session.get('is_carryover', False):
                        description_parts.append(f"         - Previous window → {session['end_datetime'].split(' ')[1]} | Screen session (duration this window: {duration_str})")
                    else:
                        # Format start and end times (only show time, not date if same day)
                        start_time = session['start_datetime'].split(' ')[1]  # Get time part
                        end_time = session['end_datetime'].split(' ')[1]  # Get time part
                        start_date = session['start_datetime'].split(' ')[0]  # Get date part
                        end_date = session['end_datetime'].split(' ')[0]  # Get date part
                        
                        if start_date == end_date:
                            # Same day - show date once and time range
                            description_parts.append(f"         - {start_date} {start_time} → {end_time} | Screen session (duration: {duration_str})")
                        else:
                            # Different days - show full timestamps
                            description_parts.append(f"         - {session['start_datetime']} → {session['end_datetime']} | Screen session (duration: {duration_str})")
        
        return '\n'.join(description_parts)
    
    # Process data using the helper function
    narratives = process_sensor_by_timewindow(
        sensor_data, sensor_name, start_timestamp, end_timestamp, process_screen_window
    )
    
    log_info(f"Generated {len(narratives)} screen narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives

# Do not remove this function, it is used for post-clustering home candidate analysis
def identify_and_merge_daily_home_candidates(daily_clusters, night_time_start, night_time_end, merge_distance_threshold, logger=None):
    """
    Identify home cluster candidates for each day and merge nearby candidates.
    
    Args:
        daily_clusters: Dictionary containing daily clustering results
        night_time_start: Start hour of nighttime (e.g., 22 for 10 PM)
        night_time_end: End hour of nighttime (e.g., 6 for 6 AM)
        merge_distance_threshold: Distance threshold in meters to merge candidates
        logger: Logger instance for detailed logging
        
    Returns:
        tuple: (home_cluster_index, daily_home_analysis, merged_home_center, clusters_to_merge)
    """
    daily_home_candidates = {}
    days_without_nighttime = []
    
    log_info("\n=== Daily Home Candidate Analysis ===", logger)
    
    # Step 1: Identify home cluster candidate for each day
    for day_id, day_data in daily_clusters.items():
        day_labels = day_data['labels']
        day_datetimes = day_data['datetimes']
        day_coordinates = day_data['coordinates']
        
        # Find nighttime points for this day
        day_night_labels = []
        for i, label in enumerate(day_labels):
            if label == -1:  # Skip noise
                continue
                
            dt_str = day_datetimes[i]
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') if isinstance(dt_str, str) else dt_str
            hour = dt.hour
            
            # Check if hour is in nighttime range
            if night_time_start <= hour or hour <= night_time_end:
                day_night_labels.append(label)
        
        if day_night_labels:
            # Find most frequent nighttime cluster for this day
            day_night_labels = np.array(day_night_labels)
            day_home_candidate = np.bincount(day_night_labels).argmax()
            
            # Calculate center of this candidate cluster
            candidate_mask = np.array(day_labels) == day_home_candidate
            candidate_coords = day_coordinates[candidate_mask]
            candidate_center = np.mean(candidate_coords, axis=0)
            
            daily_home_candidates[day_id] = {
                'cluster_id': day_home_candidate,
                'center': candidate_center,
                'nighttime_points': len(day_night_labels),
                'nighttime_points_in_candidate': np.sum(day_night_labels == day_home_candidate),
                'total_points_in_candidate': np.sum(candidate_mask)
            }
            
            log_info(f"Day {day_id}: Home candidate cluster {day_home_candidate} at ({candidate_center[0]:.6f}, {candidate_center[1]:.6f})", logger)
            log_info(f"  - {len(day_night_labels)} nighttime points, {np.sum(day_night_labels == day_home_candidate)} in candidate cluster", logger)
        else:
            log_info(f"Day {day_id}: No nighttime points found - will check for home proximity after finding merged home", logger)
            days_without_nighttime.append(day_id)
    
    if not daily_home_candidates:
        raise ValueError("No daily home candidates found")
    
    # Step 2: Calculate distances between daily home candidates
    log_info("\n=== Merging Nearby Home Candidates ===", logger)
    candidate_centers = []
    candidate_days = []
    
    for day_id, candidate_data in daily_home_candidates.items():
        candidate_centers.append(candidate_data['center'])
        candidate_days.append(day_id)
    
    # Group candidates by proximity
    candidate_groups = []
    used_candidates = set()
    
    for i, center_i in enumerate(candidate_centers):
        if i in used_candidates:
            continue
            
        # Start new group with this candidate
        group = {
            'days': [candidate_days[i]],
            'centers': [center_i],
            'candidates': [daily_home_candidates[candidate_days[i]]]
        }
        used_candidates.add(i)
        
        # Find nearby candidates to merge
        for j, center_j in enumerate(candidate_centers):
            if j in used_candidates:
                continue
                
            distance = geodesic(center_i, center_j).meters
            if distance <= merge_distance_threshold:
                group['days'].append(candidate_days[j])
                group['centers'].append(center_j)
                group['candidates'].append(daily_home_candidates[candidate_days[j]])
                used_candidates.add(j)
                log_info(f"Merging Day {candidate_days[j]} candidate with Day {candidate_days[i]} (distance: {distance:.1f}m)", logger)
        
        candidate_groups.append(group)
    
    # Step 3: Identify the primary home group (most days)
    primary_group = max(candidate_groups, key=lambda g: len(g['days']))
    
    # Calculate merged center (weighted by nighttime points)
    total_nighttime_points = sum(c['nighttime_points_in_candidate'] for c in primary_group['candidates'])
    if total_nighttime_points > 0:
        weighted_lat = sum(c['center'][0] * c['nighttime_points_in_candidate'] for c in primary_group['candidates']) / total_nighttime_points
        weighted_lon = sum(c['center'][1] * c['nighttime_points_in_candidate'] for c in primary_group['candidates']) / total_nighttime_points
        merged_home_center = np.array([weighted_lat, weighted_lon])
    else:
        merged_home_center = np.mean(primary_group['centers'], axis=0)
    
    log_info(f"\nPrimary home group: {len(primary_group['days'])} days", logger)
    log_info(f"Days with home activity: {sorted(primary_group['days'])}", logger)
    log_info(f"Merged home center: ({merged_home_center[0]:.6f}, {merged_home_center[1]:.6f})", logger)
    
    # Step 4: Collect all cluster IDs that should be merged
    clusters_to_merge = set()
    for candidate in primary_group['candidates']:
        clusters_to_merge.add(candidate['cluster_id'])
    
    log_info(f"Clusters to merge: {sorted(list(clusters_to_merge))}", logger)
    
    # Step 5: Check days without nighttime points for clusters that could be merged to home
    log_info("\n=== Checking Days Without Nighttime Points ===", logger)
    for day_id in days_without_nighttime:
        day_data = daily_clusters[day_id]
        day_labels = day_data['labels']
        day_coordinates = day_data['coordinates']
        
        # Find all unique clusters for this day (excluding noise)
        unique_clusters = set(label for label in day_labels if label != -1)
        
        for cluster_id in unique_clusters:
            # Calculate center of this cluster
            cluster_mask = np.array(day_labels) == cluster_id
            cluster_coords = day_coordinates[cluster_mask]
            cluster_center = np.mean(cluster_coords, axis=0)
            
            # Check distance to merged home center
            distance_to_home = geodesic(cluster_center, merged_home_center).meters
            
            if distance_to_home <= merge_distance_threshold:
                log_info(f"Day {day_id}: Cluster {cluster_id} at ({cluster_center[0]:.6f}, {cluster_center[1]:.6f}) is {distance_to_home:.1f}m from home - merging", logger)
                clusters_to_merge.add(cluster_id)
            else:
                log_info(f"Day {day_id}: Cluster {cluster_id} at ({cluster_center[0]:.6f}, {cluster_center[1]:.6f}) is {distance_to_home:.1f}m from home - not merging", logger)
    
    # Step 6: Find the best representative cluster ID (most nighttime points)
    best_candidate = max(primary_group['candidates'], key=lambda c: c['nighttime_points_in_candidate'])
    home_cluster_index = best_candidate['cluster_id']
    
    # Step 7: Analyze daily home presence
    daily_home_analysis = {}
    for day_id in daily_home_candidates.keys():
        if day_id in primary_group['days']:
            candidate = daily_home_candidates[day_id]
            daily_home_analysis[day_id] = {
                'was_home': True,
                'home_cluster_id': candidate['cluster_id'],
                'nighttime_points': candidate['nighttime_points'],
                'nighttime_points_at_home': candidate['nighttime_points_in_candidate'],
                'home_percentage': (candidate['nighttime_points_in_candidate'] / candidate['nighttime_points'] * 100) if candidate['nighttime_points'] > 0 else 0
            }
        else:
            # Check if this day had a home candidate but it wasn't merged
            if day_id in daily_home_candidates:
                candidate = daily_home_candidates[day_id]
                # Calculate distance to merged home
                distance_to_home = geodesic(candidate['center'], merged_home_center).meters
                daily_home_analysis[day_id] = {
                    'was_home': False,
                    'alternative_location': True,
                    'distance_from_home': distance_to_home,
                    'nighttime_points': candidate['nighttime_points'],
                    'alternative_cluster_id': candidate['cluster_id']
                }
            else:
                daily_home_analysis[day_id] = {
                    'was_home': False,
                    'alternative_location': False,
                    'nighttime_points': 0
                }
    
    # Also analyze days without nighttime points
    for day_id in days_without_nighttime:
        day_data = daily_clusters[day_id]
        day_labels = day_data['labels']
        day_coordinates = day_data['coordinates']
        
        # Find all unique clusters for this day (excluding noise)
        unique_clusters = set(label for label in day_labels if label != -1)
        
        # Check if any cluster was merged to home
        merged_clusters = [cluster_id for cluster_id in unique_clusters if cluster_id in clusters_to_merge]
        
        if merged_clusters:
            # Find the cluster closest to home
            closest_cluster = None
            min_distance = float('inf')
            
            for cluster_id in merged_clusters:
                cluster_mask = np.array(day_labels) == cluster_id
                cluster_coords = day_coordinates[cluster_mask]
                cluster_center = np.mean(cluster_coords, axis=0)
                distance = geodesic(cluster_center, merged_home_center).meters
                
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_id
            
            daily_home_analysis[day_id] = {
                'was_home': True,
                'home_cluster_id': closest_cluster,
                'nighttime_points': 0,
                'nighttime_points_at_home': 0,
                'home_percentage': 0,
                'merged_to_home': True,
                'distance_from_home': min_distance
            }
        else:
            # Find the closest cluster to home
            closest_cluster = None
            min_distance = float('inf')
            
            for cluster_id in unique_clusters:
                cluster_mask = np.array(day_labels) == cluster_id
                cluster_coords = day_coordinates[cluster_mask]
                cluster_center = np.mean(cluster_coords, axis=0)
                distance = geodesic(cluster_center, merged_home_center).meters
                
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_id
            
            daily_home_analysis[day_id] = {
                'was_home': False,
                'alternative_location': True,
                'distance_from_home': min_distance,
                'nighttime_points': 0,
                'alternative_cluster_id': closest_cluster
            }
    
    # Print daily analysis
    log_info("\n=== Daily Home Presence Analysis ===", logger)
    for day_id in sorted(daily_home_analysis.keys()):
        analysis = daily_home_analysis[day_id]
        if analysis['was_home']:
            if analysis.get('merged_to_home', False):
                log_info(f"Day {day_id}: At home (merged to home cluster, {analysis['distance_from_home']:.1f}m from home center)", logger)
            else:
                log_info(f"Day {day_id}: At home ({analysis['nighttime_points_at_home']}/{analysis['nighttime_points']} nighttime points, {analysis['home_percentage']:.1f}%)", logger)
        elif analysis.get('alternative_location', False):
            log_info(f"Day {day_id}: Away from home ({analysis['distance_from_home']:.1f}m from home, {analysis['nighttime_points']} nighttime points)", logger)
        else:
            log_info(f"Day {day_id}: No clear nighttime location", logger)
    
    return home_cluster_index, daily_home_analysis, merged_home_center, clusters_to_merge

def process_clustering_results(coordinates, cluster_labels, datetimes, indices, speeds, 
                               use_daily_clustering, daily_clusters, night_time_start, night_time_end, logger=None):
    """
    Process clustering results to identify home cluster, build cluster data structure,
    and calculate distances. This function handles common post-clustering steps for both
    single-day and multi-day clustering approaches.
    
    Args:
        coordinates: Array of coordinate points [lat, lon]
        cluster_labels: Array of cluster labels for each point
        datetimes: List of datetime strings for each point
        indices: List of original indices for each point
        speeds: List of speed values for each point
        use_daily_clustering: Boolean indicating if daily clustering was used
        daily_clusters: Dictionary of daily clustering results (for multi-day)
        night_time_start: Start hour of nighttime period
        night_time_end: End hour of nighttime period
        logger: Logger instance for detailed logging
        
    Returns:
        tuple: (cluster, clustered_coordinates, clustered_labels, clustered_datetimes, 
                clustered_indices, clustered_speeds, home_group_center)
    """
    
    # Filter out noise points (label == -1)
    mask = cluster_labels >= 0
    if not np.any(mask):
        raise ValueError("No valid clusters found - all points classified as noise")
    
    clustered_coordinates = coordinates[mask]
    clustered_labels = cluster_labels[mask]
    clustered_datetimes = [datetimes[i] for i in range(len(datetimes)) if mask[i]]
    clustered_indices = [indices[i] for i in range(len(indices)) if mask[i]]
    clustered_speeds = [speeds[i] for i in range(len(speeds)) if mask[i]]
    
    log_info(f"After filtering noise: {len(clustered_labels)} clustered points", logger)
    
    # Identify home cluster using appropriate method
    if use_daily_clustering and daily_clusters:
        # Use daily home candidate merging approach
        home_cluster_index, daily_home_analysis, merged_home_center, clusters_to_merge = identify_and_merge_daily_home_candidates(
            daily_clusters, night_time_start, night_time_end, merge_distance_threshold, logger
        )

            
    else:
        # Use original method for single day or short periods
        night_labels = []
        for i, cluster_label in enumerate(clustered_labels):
            dt = clustered_datetimes[i]  # Use already parsed datetime object
            hour = dt.hour if hasattr(dt, 'hour') else datetime.strptime(str(dt), '%Y-%m-%d %H:%M:%S').hour
            
            # Check if hour is in nighttime range (night_time_start PM to night_time_end AM)
            if night_time_start <= hour or hour <= night_time_end:
                night_labels.append(cluster_label)
        
        night_labels = np.array(night_labels)
        if len(night_labels) == 0:
            raise ValueError("No nighttime data available for home cluster identification")
        
        log_info(f"Found {len(night_labels)} nighttime location points in valid clusters", logger)
        home_cluster_index = np.bincount(night_labels).argmax() # Identify home cluster
        log_info(f"Home cluster identified as cluster {home_cluster_index}", logger)
        merged_home_center = None
        clusters_to_merge = set()
        
        # Find home cluster center for distance calculation
        home_cluster_mask = clustered_labels == home_cluster_index
        home_cluster_points = clustered_coordinates[home_cluster_mask]
        if len(home_cluster_points) > 0:
            home_group_center = np.mean(home_cluster_points, axis=0)
    
    # Set home_group_center before processing clusters
    if merged_home_center is not None:
        home_group_center = merged_home_center
        log_info(f"Using merged home center: ({merged_home_center[0]:.6f}, {merged_home_center[1]:.6f})", logger)
    else:
        # For single-day clustering, we'll set this when we find the home cluster
        home_group_center = None
        log_info("Will determine home center from home cluster centroid", logger)
    
    # COMBINED MERGING AND RENUMBERING IN SINGLE OPERATION
    # This efficiently merges home candidates and assigns final consecutive IDs in one pass
    log_info("\n=== Combined Merging and Renumbering Operation ===", logger)
    unique_cluster_ids = sorted(set(clustered_labels))
    
    # Create direct mapping from original cluster IDs to final consecutive IDs
    old_to_new_mapping = {}
    
    if use_daily_clustering and daily_clusters and 'clusters_to_merge' in locals():
        # Multi-day clustering: merge home candidates to ID 0, others get consecutive IDs
        log_info(f"Merging home candidates {sorted(list(clusters_to_merge))} → ID 0", logger)
        
        # All home candidate clusters map to ID 0
        for cluster_id in clusters_to_merge:
            old_to_new_mapping[cluster_id] = 0
            log_info(f"  - Home candidate cluster {cluster_id} → 0", logger)
        
        # Assign consecutive IDs to non-home clusters
        new_cluster_id = 1
        for old_cluster_id in unique_cluster_ids:
            if old_cluster_id not in clusters_to_merge:
                old_to_new_mapping[old_cluster_id] = new_cluster_id
                log_info(f"  - Unknown cluster {old_cluster_id} → {new_cluster_id}", logger)
                new_cluster_id += 1
        
        home_cluster_index = 0  # Merged home cluster is always ID 0
        
    else:
        # Single-day clustering: home cluster to ID 0, others get consecutive IDs
        log_info(f"Single home cluster {home_cluster_index} → ID 0", logger)
        
        # Home cluster gets ID 0
        old_to_new_mapping[home_cluster_index] = 0
        log_info(f"  - Home cluster {home_cluster_index} → 0", logger)
        
        # Assign consecutive IDs to other clusters
        new_cluster_id = 1
        for old_cluster_id in unique_cluster_ids:
            if old_cluster_id != home_cluster_index:
                old_to_new_mapping[old_cluster_id] = new_cluster_id
                log_info(f"  - Unknown cluster {old_cluster_id} → {new_cluster_id}", logger)
                new_cluster_id += 1
        
        home_cluster_index = 0  # Home cluster is always ID 0
    
    # Apply the mapping in a single pass through all points
    log_info(f"Applying final cluster assignments to {len(clustered_labels)} points...", logger)
    final_labels = clustered_labels.copy()
    merged_point_count = 0
    
    for i, old_id in enumerate(clustered_labels):
        final_labels[i] = old_to_new_mapping[old_id]
        if old_to_new_mapping[old_id] == 0:  # Count points in merged home cluster
            merged_point_count += 1
    
    # Update cluster labels with final assignments
    clustered_labels = final_labels
    
    final_cluster_count = len(set(clustered_labels))
    log_info(f"✓ Final result: {final_cluster_count} clusters (Home: {merged_point_count} points, Unknown: {len(clustered_labels) - merged_point_count} points)", logger)
    
    # Set merged_home_center for distance calculations
    if use_daily_clustering and daily_clusters and 'clusters_to_merge' in locals() and len(clusters_to_merge) > 1:
        merged_home_center = merged_home_center  # Use the calculated merged center
    else:
        merged_home_center = None  # Will use cluster center
    
    cluster = []
    unknown_place_counter = 0
    
    # Process clusters in order of their final IDs to ensure consistent unknown numbering
    for cluster_id in sorted(set(clustered_labels)):
        # Filter data by cluster ID, extract latitude and longitude for each cluster
        cluster_mask = clustered_labels == cluster_id
        cluster_points = clustered_coordinates[cluster_mask]  # Extract only latitude and longitude
        
        # Skip if there is no data (this handles merged clusters automatically)
        if len(cluster_points) == 0:
            continue
            
        cluster_center = np.mean(cluster_points, axis=0)
        
        if cluster_id == home_cluster_index:  # Home cluster (always ID 0)
            place = "home"
            # For home cluster, use merged home center if available, otherwise use cluster center
            if merged_home_center is not None:
                actual_center = merged_home_center
                home_group_center = merged_home_center  # Ensure it's set for distance calculations
                log_info(f"Home cluster {cluster_id}: Using merged center ({actual_center[0]:.6f}, {actual_center[1]:.6f}) vs cluster center ({cluster_center[0]:.6f}, {cluster_center[1]:.6f})", logger)
            else:
                actual_center = cluster_center
                home_group_center = cluster_center  # Set for single-day clustering
                log_info(f"Home cluster {cluster_id}: Using cluster center ({actual_center[0]:.6f}, {actual_center[1]:.6f})", logger)
            
            cluster.append((int(cluster_id), float(actual_center[0]), float(actual_center[1]), int(len(cluster_points)), place))
        else:
            # Generate unique unknown place labels: unknown1, unknown2, unknown3, etc.
            unknown_place_counter += 1
            place = f"unknown{unknown_place_counter}"
            log_info(f"Unknown cluster {cluster_id}: Labeled as '{place}' at ({cluster_center[0]:.6f}, {cluster_center[1]:.6f})", logger)
            
            # For non-home clusters, use the calculated cluster center
            cluster.append((int(cluster_id), float(cluster_center[0]), float(cluster_center[1]), int(len(cluster_points)), place))

    # Calculate distances of each cluster from home
    if home_group_center is None:
        raise ValueError("home_group_center not properly set - this should not happen")
    
    log_info(f"Calculating distances from home center: ({home_group_center[0]:.6f}, {home_group_center[1]:.6f})", logger)
    
    for i, cluster_entry in enumerate(cluster):
        cluster_id, center_lat, center_lon, num_points, place = cluster_entry
        if place == "home":
            # Manually assign 0 distance for home cluster
            distance_from_home = 0.0
        else:
            # Calculate distance for non-home clusters
            distance_from_home = geodesic(home_group_center, (center_lat, center_lon)).meters
        cluster[i] = cluster_entry + (float(distance_from_home),)  # Update the tuple in the cluster list
    
    # Validation: Show home cluster distance (should be 0 or very small)  
    for cluster_data in cluster:
        if len(cluster_data) == 6:  # Must have exactly 6 elements
            cluster_id, center_lat, center_lon, num_points, place, cluster_distance_from_home = cluster_data
            if place == "home":
                log_info(f"VALIDATION: Home cluster {cluster_id} distance from home: {cluster_distance_from_home:.1f}m (should be ~0 for merged center)", logger)
                break

    # Print cluster details
    if len(clustered_coordinates) > 0:
        log_info("Cluster Centers:", logger)
        for cluster_data in cluster:
            if len(cluster_data) == 6:  # Must have exactly 6 elements
                cluster_id, center_lat, center_lon, num_points, place, distance_from_home = cluster_data
                log_info(f"Cluster {cluster_id}: Center Lat = {center_lat:.6f}, Center Lon = {center_lon:.6f}, N = {num_points}, Place = {place}, Distance = {distance_from_home:.1f}m", logger)

    return (cluster, clustered_coordinates, clustered_labels, clustered_datetimes, 
            clustered_indices, clustered_speeds, home_group_center)

def perform_dbscan_clustering(coordinates, datetimes, eps, min_samples, use_daily_clustering, 
                               start_dt, end_dt, night_time_end, logger=None):
    """
    Perform DBSCAN clustering using either single-day or multi-day approach.
    
    Args:
        coordinates: Array of coordinate points [lat, lon]
        datetimes: List of datetime strings for each point
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        use_daily_clustering: Boolean indicating if daily clustering should be used
        start_dt: Start datetime object
        end_dt: End datetime object  
        night_time_end: Hour when daily periods end
        logger: Logger instance for detailed logging
        
    Returns:
        tuple: (cluster_labels, daily_clusters)
    """
    
    if use_daily_clustering:
        log_info("Performing multi-day clustering...", logger)
        
        # Group data by daily periods starting from night_time_end
        daily_clusters = {}
        all_cluster_labels = []
        global_cluster_counter = 0
        
        # Create daily periods starting from night_time_end
        current_day_start = start_dt.replace(hour=night_time_end, minute=0, second=0, microsecond=0)
        if start_dt.hour < night_time_end:
            # If start time is before night_time_end, use the same day
            current_day_start = current_day_start
        else:
            # If start time is after night_time_end, use the next day
            current_day_start = current_day_start + timedelta(days=1)
        
        # First, collect all daily periods
        daily_periods = []
        temp_day_start = current_day_start
        while temp_day_start < end_dt:
            temp_day_end = min(temp_day_start + timedelta(days=1), end_dt)
            duration_hours = (temp_day_end - temp_day_start).total_seconds() / 3600
            
            daily_periods.append({
                'start': temp_day_start,
                'end': temp_day_end,
                'duration_hours': duration_hours
            })
            
            temp_day_start = temp_day_start + timedelta(days=1)
        
        # Check if last day is less than 24 hours and combine with previous day if needed
        if len(daily_periods) > 1 and daily_periods[-1]['duration_hours'] < 24:
            log_info(f"Last day has {daily_periods[-1]['duration_hours']:.1f} hours - combining with previous day", logger)
            # Extend the previous day to include the last day
            daily_periods[-2]['end'] = daily_periods[-1]['end']
            daily_periods[-2]['duration_hours'] = (daily_periods[-2]['end'] - daily_periods[-2]['start']).total_seconds() / 3600
            # Remove the last day
            daily_periods.pop()
        
        # Process each daily period
        day_counter = 0
        for period in daily_periods:
            period_start = period['start']
            period_end = period['end']
            duration_hours = period['duration_hours']
            
            # Find points within this daily period
            daily_indices = []
            daily_coordinates = []
            daily_datetimes = []
            
            for i, dt_str in enumerate(datetimes):
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') if isinstance(dt_str, str) else dt_str
                if period_start <= dt < period_end:
                    daily_indices.append(i)
                    daily_coordinates.append([coordinates[i][0], coordinates[i][1]])
                    daily_datetimes.append(dt_str)
            
            if len(daily_coordinates) >= min_samples:
                log_info(f"Day {day_counter}: {period_start.strftime('%Y-%m-%d %H:%M:%S')} to {period_end.strftime('%Y-%m-%d %H:%M:%S')} ({duration_hours:.1f}h) - {len(daily_coordinates)} points", logger)
                
                # Perform clustering for this day
                daily_coordinates = np.array(daily_coordinates)
                daily_coordinates_radians = np.radians(daily_coordinates)
                
                db = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric='haversine'
                ).fit(daily_coordinates_radians)
                
                daily_labels = db.labels_
                
                # Adjust cluster labels to be globally unique
                adjusted_labels = []
                for label in daily_labels:
                    if label == -1:
                        adjusted_labels.append(-1)  # Keep noise as -1
                    else:
                        adjusted_labels.append(label + global_cluster_counter)
                
                # Update global cluster counter
                if len(daily_labels) > 0:
                    max_label = max([l for l in daily_labels if l != -1]) if any(l != -1 for l in daily_labels) else -1
                    if max_label != -1:
                        global_cluster_counter += max_label + 1
                
                # Store daily results
                daily_clusters[day_counter] = {
                    'indices': daily_indices,
                    'coordinates': daily_coordinates,
                    'labels': adjusted_labels,
                    'datetimes': daily_datetimes
                }
                
                # Map back to global indices
                for i, global_idx in enumerate(daily_indices):
                    if global_idx < len(all_cluster_labels):
                        all_cluster_labels[global_idx] = adjusted_labels[i]
                    else:
                        # Extend the list if needed
                        while len(all_cluster_labels) <= global_idx:
                            all_cluster_labels.append(-1)
                        all_cluster_labels[global_idx] = adjusted_labels[i]
            else:
                log_info(f"Day {day_counter}: {period_start.strftime('%Y-%m-%d %H:%M:%S')} to {period_end.strftime('%Y-%m-%d %H:%M:%S')} ({duration_hours:.1f}h) - {len(daily_coordinates)} points (insufficient for clustering)", logger)
            
            day_counter += 1
        
        # Convert to numpy array and ensure it matches the length of coordinates
        while len(all_cluster_labels) < len(coordinates):
            all_cluster_labels.append(-1)
        cluster_labels = np.array(all_cluster_labels[:len(coordinates)])
        
    else:
        log_info("Performing single-day clustering...", logger)
        
        # Use all data for clustering (original behavior)
        # Convert coordinates to radians for haversine metric
        coordinates_radians = np.radians(coordinates)
        
        # Apply DBSCAN clustering
        db = DBSCAN(
            eps=eps,  # radians (e.g., 0.000047 radians × 6371000 m ≈ 300m)
            min_samples=min_samples,  # require at least min_samples points to form a cluster
            metric='haversine'
        ).fit(coordinates_radians)
        
        cluster_labels = db.labels_  # -1 are noise, 0,1,2... are clusters
        daily_clusters = {}  # Empty for single-day clustering
    
    return cluster_labels, daily_clusters

def calculate_location_stay_times_from_timestamps(sorted_data, window_start, window_end):
    """
    Calculate location stay times using actual timestamps instead of proportional allocation.
    Handles multiple visits to the same location properly.
    
    Args:
        sorted_data: List of location records sorted by timestamp
        window_start: Window start timestamp (milliseconds)
        window_end: Window end timestamp (milliseconds)
        
    Returns:
        dict: Location stats with accurate time calculations
    """
    location_visits = {}
    location_periods = []  # Track continuous periods at each location
    
    # Group consecutive records by location to create visit periods
    current_location = None
    current_period_start = None
    
    for i, record in enumerate(sorted_data):
        place_name = record['place_name']
        timestamp = record['timestamp']
        
        # Initialize location stats if not exists
        if place_name not in location_visits:
            location_visits[place_name] = {
                'cluster_id': record['cluster_id'],
                'distance_from_home': record['distance_from_home'],
                'visit_count': 0,
                'first_seen': record['datetime'],
                'last_seen': record['datetime'],
                'data_points': 0,
                'visit_periods': [],  # Track separate visit periods
                'total_time_seconds': 0
            }
        
        # Update basic stats
        location_visits[place_name]['data_points'] += 1
        location_visits[place_name]['last_seen'] = record['datetime']
        
        # Track continuous periods at each location
        if current_location != place_name:
            # End previous period if exists
            if current_location is not None and current_period_start is not None:
                period_end = sorted_data[i-1]['timestamp'] if i > 0 else timestamp
                location_periods.append({
                    'location': current_location,
                    'start_time': current_period_start,
                    'end_time': period_end,
                    'duration_seconds': (period_end - current_period_start) / 1000.0
                })
            
            # Start new period
            current_location = place_name
            current_period_start = timestamp
            location_visits[place_name]['visit_count'] += 1
    
    # Handle the last period
    if current_location is not None and current_period_start is not None:
        # If the last period extends beyond window, cap it at window end
        last_timestamp = sorted_data[-1]['timestamp']
        period_end = min(last_timestamp, window_end)
        location_periods.append({
            'location': current_location,
            'start_time': current_period_start,
            'end_time': period_end,
            'duration_seconds': (period_end - current_period_start) / 1000.0
        })
    
    # Calculate total time spent at each location
    for period in location_periods:
        location = period['location']
        duration = period['duration_seconds']
        
        # Store individual visit periods
        location_visits[location]['visit_periods'].append({
            'start_time': period['start_time'],
            'end_time': period['end_time'],
            'duration_seconds': duration
        })
        
        # Add to total time
        location_visits[location]['total_time_seconds'] += duration
    
    # Identify locations that will be filtered out
    locations_to_filter = set()
    for place_name, stats in location_visits.items():
        # Filter based on minimum data points
        if stats['data_points'] < location_minimum_data_points:
            locations_to_filter.add(place_name)
        # Also filter based on minimum stay duration
        elif stats['total_time_seconds'] < (location_minimum_stay_minutes * 60):
            locations_to_filter.add(place_name)
    
    # Merge periods for locations that will remain, accounting for filtered locations
    if locations_to_filter:
        # For each location that will remain, merge its periods if there are filtered locations between them
        for place_name, stats in location_visits.items():
            if place_name not in locations_to_filter and len(stats['visit_periods']) > 1:
                # Sort periods by start time
                sorted_periods = sorted(stats['visit_periods'], key=lambda x: x['start_time'])
                
                # Merge all periods into one continuous period (since filtered locations are between them)
                earliest_start = sorted_periods[0]['start_time']
                latest_end = max(period['end_time'] for period in sorted_periods)
                
                # Create a single merged period
                merged_period = {
                    'start_time': earliest_start,
                    'end_time': latest_end,
                    'duration_seconds': (latest_end - earliest_start) / 1000.0
                }
                
                # Update stats
                stats['visit_periods'] = [merged_period]
                stats['visit_count'] = 1  # Now it's one continuous visit
                
                # Recalculate total time from the merged period
                total_time = merged_period['duration_seconds']
                stats['total_time_seconds'] = total_time
    
    # Now filter out locations with insufficient data points
    filtered_location_visits = {}
    filtered_count = 0
    for place_name, stats in location_visits.items():
        if place_name not in locations_to_filter:
            filtered_location_visits[place_name] = stats
        else:
            filtered_count += 1
    
    # Convert to minutes for remaining locations and filter out data quality issues
    final_location_visits = {}
    for place_name, stats in filtered_location_visits.items():
        total_seconds = stats['total_time_seconds']
        
        # With multiple data points, we should always have a valid time span
        if total_seconds <= 0:
            log_warning(f"Warning: Location '{place_name}' has {stats['data_points']} data points but zero time span - possible data quality issue")
            log_warning(f"  Visit periods: {stats.get('visit_periods', [])}")
            log_warning(f"  Skipping location due to zero time span")
            continue
        else:
            stats['estimated_time_seconds'] = total_seconds
            stats['estimated_time_minutes'] = total_seconds / 60.0
            final_location_visits[place_name] = stats
    
    return final_location_visits

# Dictionary to store message traces and keep track of unique IDs
message_traces = {}
message_trace_n = 0
def get_message_number(trace):
    """Assigns a unique number to each message trace."""
    global message_trace_n, message_traces
    if trace not in message_traces:
        message_trace_n += 1
        message_traces[trace] = message_trace_n
    return message_traces[trace]  

def describe_messages_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated messages analysis by time windows.
    Shows messaging patterns, types, frequencies, and communication sequences.
    
    Args:
        sensor_data (list): List of messages sensor records
        sensor_name (str): Name of the sensor (should be 'messages')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records for session correlation
        
    Returns:
        list: List of formatted messages narrative tuples (datetime, description)
    """
    log_info("Generating integrated description for messages")
    
    if sensor_name != "messages" or not sensor_data:
        log_info("No messages data available, skipping messages integration")
        return []
    
    # Message type mapping
    message_types = {
        1: "received",
        2: "sent"
    }
    
    def process_messages_window(window_data, datetime_str, window_start, window_end):
        """Process messages data for a single time window."""
        if not window_data:
            return None
        
        # Sort window data by timestamp
        sorted_data = sorted(window_data, key=lambda x: x['timestamp'])
        
        # Track messages by type and person
        message_stats = {
            'received': {'count': 0, 'people': {}},
            'sent': {'count': 0, 'people': {}}
        }
        
        message_sequence = []
        total_messages = 0
        unique_people = set()
        
        # Process each message record
        for record in sorted_data:
            message_type_id = record.get('message_type', 0)
            trace = record.get('trace', 'Unknown')
            record_datetime = record.get('datetime', datetime_str)
            
            if message_type_id not in message_types:
                continue
                
            message_type = message_types[message_type_id]
            
            # Get person number from trace (using existing message numbering system)
            person_number = get_message_number(trace)
            unique_people.add(person_number)
            
            # Update message statistics
            message_stats[message_type]['count'] += 1
            
            # Track per-person statistics
            if person_number not in message_stats[message_type]['people']:
                message_stats[message_type]['people'][person_number] = {
                    'count': 0,
                    'messages': []
                }
            
            message_stats[message_type]['people'][person_number]['count'] += 1
            message_stats[message_type]['people'][person_number]['messages'].append({
                'datetime': record_datetime,
                'timestamp': record.get('timestamp', 0)
            })
            
            # Add to sequence
            message_sequence.append({
                'datetime': record_datetime,
                'timestamp': record.get('timestamp', 0),
                'type': message_type,
                'person': person_number
            })
            
            total_messages += 1
        
        # Generate description for this window
        description_parts = [f"messages | Messaging Activity"]
        
        # Show total messages summary
        if total_messages > 0:
            description_parts.append(f"    - Total messages: {total_messages}")
            description_parts.append(f"    - People involved: {len(unique_people)}")
            
            # Show breakdown by type
            message_breakdown = []
            for message_type, stats in message_stats.items():
                if stats['count'] > 0:
                    message_breakdown.append(f"{stats['count']} {message_type}")
            
            if message_breakdown:
                description_parts.append(f"    - Message breakdown: {', '.join(message_breakdown)}")
            
            # Show message sequence if multiple messages
            if len(message_sequence) > 1:
                # Group consecutive messages by person for cleaner display
                grouped_sequence = group_consecutive_messages(message_sequence)
                
                if len(grouped_sequence) > 1:
                    description_parts.append(f"    - Message sequence:")
                    for group in grouped_sequence:
                        time_part = group['start_time'].split(' ')[1]  # Get time part only
                        if group['count'] > 1:
                            description_parts.append(f"         - {time_part} {group['type']} {group['count']} messages to/from person {group['person']}")
                        else:
                            description_parts.append(f"         - {time_part} {group['type']} message to/from person {group['person']}")
            
            # Show detailed statistics by message type
            description_parts.append(f"    - Message details:")
            
            for message_type, stats in message_stats.items():
                if stats['count'] > 0:
                    # Show per-person breakdown for this message type
                    if stats['people']:
                        people_list = []
                        for person_num, person_stats in stats['people'].items():
                            if person_stats['count'] > 1:
                                people_list.append(f"person {person_num} ({person_stats['count']} messages)")
                            else:
                                people_list.append(f"person {person_num}")
                        
                        description_parts.append(f"         - {message_type.title()}: {', '.join(people_list)}")
            
            # Show session correlation if sessions are available
            if sessions:
                # Find sessions that overlap with this window
                overlapping_sessions = []
                for session in sessions:
                    if (session['start_timestamp'] <= window_end and 
                        session['end_timestamp'] >= window_start):
                        overlapping_sessions.append(session['session_id'])
                
                if overlapping_sessions:
                    if len(overlapping_sessions) == 1:
                        description_parts.append(f"    - Session activity: Session {overlapping_sessions[0]}")
                    else:
                        description_parts.append(f"    - Session activity: Sessions {', '.join(map(str, overlapping_sessions))}")
        
        return '\n'.join(description_parts)
    
    # Process data using the helper function
    narratives = process_sensor_by_timewindow(
        sensor_data, sensor_name, start_timestamp, end_timestamp, process_messages_window
    )
    
    log_info(f"Generated {len(narratives)} messages narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives



def group_consecutive_messages(message_sequence):
    """
    Group consecutive messages of the same type and person for cleaner display.
    
    Args:
        message_sequence (list): List of message records in chronological order
        
    Returns:
        list: List of grouped message records
    """
    if not message_sequence:
        return []
    
    grouped = []
    current_group = None
    
    for message in message_sequence:
        message_type = message['type']
        person = message['person']
        datetime_str = message['datetime']
        
        # Start new group if different type or person
        if (current_group is None or 
            current_group['type'] != message_type or 
            current_group['person'] != person):
            
            if current_group is not None:
                grouped.append(current_group)
            
            current_group = {
                'type': message_type,
                'person': person,
                'start_time': datetime_str,
                'end_time': datetime_str,
                'count': 1
            }
        else:
            # Add to current group
            current_group['count'] += 1
            current_group['end_time'] = datetime_str
    
    # Don't forget the last group
    if current_group is not None:
        grouped.append(current_group)
    
    return grouped

def describe_calls_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated calls analysis by time windows.
    Shows call patterns, types, durations, and call sequences.
    
    Args:
        sensor_data (list): List of calls sensor records
        sensor_name (str): Name of the sensor (should be 'calls')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records for session correlation
        
    Returns:
        list: List of formatted calls narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('calls')
              - description: Human-readable narrative text
    """
    log_info("Generating integrated description for calls")
    
    if sensor_name != "calls" or not sensor_data:
        log_info("No calls data available, skipping calls integration")
        return []
    
    # Call type mapping
    call_types = {
        1: "received",
        2: "made", 
        3: "missed"
    }
    
    def process_calls_window(window_data, datetime_str, window_start, window_end):
        """Process calls data for a single time window."""
        if not window_data:
            return None
        
        # Sort window data by timestamp
        sorted_data = sorted(window_data, key=lambda x: x['timestamp'])
        
        # Track calls by type and person
        call_stats = {
            'received': {'count': 0, 'total_duration': 0, 'people': {}},
            'made': {'count': 0, 'total_duration': 0, 'people': {}},
            'missed': {'count': 0, 'total_duration': 0, 'people': {}}
        }
        
        call_sequence = []
        total_calls = 0
        total_duration = 0
        
        # Process each call record
        for record in sorted_data:
            call_type_id = record.get('call_type', 0)
            call_duration = record.get('call_duration', 0)
            trace = record.get('trace', 'Unknown')
            record_datetime = record.get('datetime', datetime_str)
            
            if call_type_id not in call_types:
                continue
                
            call_type = call_types[call_type_id]
            
            # Get person number from trace (using existing message numbering system)
            person_number = get_message_number(trace)
            
            # Update call statistics
            call_stats[call_type]['count'] += 1
            call_stats[call_type]['total_duration'] += call_duration
            
            # Track per-person statistics
            if person_number not in call_stats[call_type]['people']:
                call_stats[call_type]['people'][person_number] = {
                    'count': 0,
                    'total_duration': 0,
                    'calls': []
                }
            
            call_stats[call_type]['people'][person_number]['count'] += 1
            call_stats[call_type]['people'][person_number]['total_duration'] += call_duration
            call_stats[call_type]['people'][person_number]['calls'].append({
                'datetime': record_datetime,
                'duration': call_duration
            })
            
            # Add to sequence
            call_sequence.append({
                'datetime': record_datetime,
                'type': call_type,
                'person': person_number,
                'duration': call_duration
            })
            
            total_calls += 1
            total_duration += call_duration
        
        # Generate description for this window
        description_parts = [f"calls | Call Activity"]
        
        # Show total calls summary
        if total_calls > 0:
            description_parts.append(f"    - Total calls: {total_calls}")
            
            # Show breakdown by type
            call_breakdown = []
            for call_type, stats in call_stats.items():
                if stats['count'] > 0:
                    call_breakdown.append(f"{stats['count']} {call_type}")
            
            if call_breakdown:
                description_parts.append(f"    - Call breakdown: {', '.join(call_breakdown)}")
            
            # Show total call duration
            if total_duration > 0:
                if total_duration >= 3600:  # 1 hour or more
                    hours = total_duration // 3600
                    minutes = (total_duration % 3600) // 60
                    seconds = total_duration % 60
                    if minutes > 0 or seconds > 0:
                        duration_str = f"{hours}h {minutes}m {seconds}s"
                    else:
                        duration_str = f"{hours}h"
                elif total_duration >= 60:  # 1 minute or more
                    minutes = total_duration // 60
                    seconds = total_duration % 60
                    if seconds > 0:
                        duration_str = f"{minutes}m {seconds}s"
                    else:
                        duration_str = f"{minutes}m"
                else:
                    duration_str = f"{total_duration}s"
                
                description_parts.append(f"    - Total talk time: {duration_str}")
            
            # Show call sequence if multiple calls
            if len(call_sequence) > 1:
                sequence_str = []
                for call in call_sequence:
                    time_part = call['datetime'].split(' ')[1]  # Get time part only
                    if call['duration'] > 0:
                        duration_str = f"{call['duration']}s"
                        sequence_str.append(f"{time_part} {call['type']} person {call['person']} ({duration_str})")
                    else:
                        sequence_str.append(f"{time_part} {call['type']} person {call['person']}")
                
                description_parts.append(f"    - Call sequence:")
                for seq in sequence_str:
                    description_parts.append(f"         - {seq}")
            
            # Show detailed statistics by call type
            description_parts.append(f"    - Call details:")
            
            for call_type, stats in call_stats.items():
                if stats['count'] > 0:
                    # Show per-person breakdown for this call type
                    if stats['people']:
                        people_list = []
                        for person_num, person_stats in stats['people'].items():
                            if person_stats['count'] > 1:
                                if person_stats['total_duration'] > 0:
                                    avg_duration = person_stats['total_duration'] / person_stats['count']
                                    people_list.append(f"person {person_num} ({person_stats['count']} calls, avg {avg_duration:.0f}s)")
                                else:
                                    people_list.append(f"person {person_num} ({person_stats['count']} calls)")
                            else:
                                if person_stats['total_duration'] > 0:
                                    people_list.append(f"person {person_num} ({person_stats['total_duration']}s)")
                                else:
                                    people_list.append(f"person {person_num}")
                        
                        description_parts.append(f"         - {call_type.title()}: {', '.join(people_list)}")
            
            # Show session correlation if sessions are available
            if sessions:
                # Find sessions that overlap with this window
                overlapping_sessions = []
                for session in sessions:
                    if (session['start_timestamp'] <= window_end and 
                        session['end_timestamp'] >= window_start):
                        overlapping_sessions.append(session['session_id'])
                
                if overlapping_sessions:
                    if len(overlapping_sessions) == 1:
                        description_parts.append(f"    - Session activity: Session {overlapping_sessions[0]}")
                    else:
                        description_parts.append(f"    - Session activity: Sessions {', '.join(map(str, overlapping_sessions))}")
        
        return '\n'.join(description_parts)
    
    # Process data using the helper function
    narratives = process_sensor_by_timewindow(
        sensor_data, sensor_name, start_timestamp, end_timestamp, process_calls_window
    )
    
    log_info(f"Generated {len(narratives)} calls narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives

def describe_wifi_combined_integrated(sensor_wifi_data, wifi_data, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated WiFi analysis combining connection activity and network detection.
    Shows networks connected to and networks detected in the area.
    
    Args:
        sensor_wifi_data (list): List of sensor_wifi sensor records (connection events)
        wifi_data (list): List of wifi sensor records (detection data)
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records (unused for wifi)
        
    Returns:
        list: List of formatted wifi narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('wifi')
              - description: Human-readable narrative text
    """
    log_info("Generating combined integrated description for WiFi (connections + detections)")
    
    if not sensor_wifi_data and not wifi_data:
        log_info("No WiFi data available, skipping WiFi integration")
        return []
    
    def process_wifi_combined_window(window_data_tuple, datetime_str, window_start, window_end):
        """Process combined wifi data for a single time window."""
        sensor_wifi_window, wifi_window = window_data_tuple
        
        # Initialize results
        connection_info = None
        detection_info = None
        
        # Process sensor_wifi data (connections)
        if sensor_wifi_window:
            connection_info = process_sensor_wifi_window(
                sensor_wifi_window, datetime_str, window_start, window_end
            )
        
        # Process wifi data (detections) - only for nearby networks information
        if wifi_window:
            detection_info = process_wifi_detection_window(wifi_window, datetime_str, window_start, window_end)
        
        # Combine the results
        if not connection_info and not detection_info:
            return None
        
        # Build combined description
        description_parts = [f"wifi | WiFi Activity Analysis"]
        
        # Add connection information
        if connection_info:
            connection_lines = connection_info.split('\n')[1:]  # Skip the header
            for line in connection_lines:
                if line.strip():
                    description_parts.append(line)
        
        # Add detection information if available (nearby networks)
        if detection_info:
            detection_lines = detection_info.split('\n')[1:]  # Skip the header
            for line in detection_lines:
                if line.strip():
                    description_parts.append(line)
        
        return '\n'.join(description_parts)
    

    
    def process_sensor_wifi_window(window_data, datetime_str, window_start, window_end):
        """Process sensor_wifi data for connections (extracted from original function)."""
        if not window_data:
            return None
        
        # Sort window data by timestamp
        sorted_data = sorted(window_data, key=lambda x: x['timestamp'])
        
        # Track wifi networks and connections
        network_connections = {}
        connection_sequence = []
        connection_switches = 0
        
        # Process each wifi connection record
        previous_ssid = None
        current_connection = None
        
        for i, record in enumerate(sorted_data):
            ssid = record.get('ssid', '')
            timestamp = record.get('timestamp', 0)
            record_datetime = record.get('datetime', datetime_str)
            
            # Clean up SSID display
            if ssid == '':
                display_ssid = '<unknown ssid>'
            else:
                # Strip quotes from SSID for display
                display_ssid = ssid.strip('"')
            
            # Track network statistics
            if ssid not in network_connections:
                network_connections[ssid] = {
                    'display_name': display_ssid,
                    'connection_count': 0,
                    'first_seen': record_datetime,
                    'last_seen': record_datetime,
                    'connection_times': []
                }
            
            network_connections[ssid]['connection_count'] += 1
            network_connections[ssid]['last_seen'] = record_datetime
            network_connections[ssid]['connection_times'].append(timestamp)
            
            # Track connection sequence and switches
            if previous_ssid is None:
                # First connection in window
                current_connection = {
                    'ssid': ssid,
                    'display_name': display_ssid,
                    'start_time': record_datetime,
                    'start_timestamp': timestamp
                }
                connection_sequence.append(current_connection)
            elif ssid != previous_ssid:
                # Network switch detected
                connection_switches += 1
                
                # End previous connection
                if current_connection:
                    current_connection['end_time'] = record_datetime
                    current_connection['end_timestamp'] = timestamp
                    current_connection['duration_ms'] = timestamp - current_connection['start_timestamp']
                
                # Start new connection
                current_connection = {
                    'ssid': ssid,
                    'display_name': display_ssid,
                    'start_time': record_datetime,
                    'start_timestamp': timestamp
                }
                connection_sequence.append(current_connection)
            
            previous_ssid = ssid
        
        # End the last connection if it exists
        if current_connection and 'end_time' not in current_connection:
            current_connection['end_time'] = current_connection['start_time']
            current_connection['end_timestamp'] = current_connection['start_timestamp']
            current_connection['duration_ms'] = 0
        
        # Calculate durations for connections
        for connection in connection_sequence:
            if 'duration_ms' not in connection:
                connection['duration_ms'] = 0
        
        # Determine primary network (most connected to)
        if network_connections:
            primary_network = max(network_connections.items(), key=lambda x: x[1]['connection_count'])
            primary_ssid, primary_stats = primary_network
        else:
            primary_ssid, primary_stats = None, None
        
        # Generate description for this window
        description_parts = [f"wifi | WiFi Connection Activity"]
        
        # Show total networks and connections
        total_networks = len(network_connections)
        total_connections = sum(stats['connection_count'] for stats in network_connections.values())
        
        if total_networks == 1:
            description_parts.append(f"    - Connected to {total_networks} network ({total_connections} connection events)")
        else:
            description_parts.append(f"    - Connected to {total_networks} networks ({total_connections} connection events)")
        
        # Show network switches
        if connection_switches > 0:
            description_parts.append(f"    - Network switches: {connection_switches}")
        
        # Show connection sequence if there are multiple connections or switches
        if len(connection_sequence) > 1 or connection_switches > 0:
            sequence_names = []
            for connection in connection_sequence:
                sequence_names.append(connection['display_name'])
            
            # Remove consecutive duplicates for cleaner display
            simplified_sequence = []
            for name in sequence_names:
                if not simplified_sequence or simplified_sequence[-1] != name:
                    simplified_sequence.append(name)
            
            if len(simplified_sequence) > 1:
                sequence_str = " → ".join(simplified_sequence)
                description_parts.append(f"    - Connection sequence: {sequence_str}")
        
        # Show primary network details
        if primary_stats:
            display_name = primary_stats['display_name']
            description_parts.append(f"    - Primary network: {display_name}")
        
        # Show all networks if multiple networks were used
        if total_networks > 1:
            description_parts.append(f"    - Networks used:")
            
            # Sort networks by connection count (descending)
            sorted_networks = sorted(network_connections.items(), 
                                   key=lambda x: x[1]['connection_count'], 
                                   reverse=True)
            
            for ssid, stats in sorted_networks:
                display_name = stats['display_name']
                description_parts.append(f"         - {display_name}")
        
        return '\n'.join(description_parts)
    
    def process_wifi_detection_window(window_data, datetime_str, window_start, window_end):
        """Process wifi data for detections (extracted from original function)."""
        if not window_data:
            return None
        
        # Define gate size in milliseconds from config
        gate_size_ms = gate_time_window * 60 * 1000  # Convert minutes to milliseconds
        
        # Collect statistics for each gate
        gate_stats = []
        window_network_appearances = {}  # Track network appearances across gates
        
        current_gate_start = window_start
        while current_gate_start < window_end:
            current_gate_end = min(current_gate_start + gate_size_ms, window_end)
            
            # Get data for this gate
            gate_data = [
                record for record in window_data
                if current_gate_start <= record['timestamp'] < current_gate_end
            ]
            
            if gate_data:
                gate_result = process_wifi_gate(gate_data)
                
                if gate_result:
                    # Calculate gate-level statistics
                    gate_unique_networks = gate_result['total_unique_networks']
                    gate_named_networks = len(gate_result['named_networks'])
                    
                    gate_stats.append({
                        'unique_networks': gate_unique_networks,
                        'named_networks': gate_named_networks,
                        'networks': gate_result['named_networks']
                    })
                    
                    # Track network appearances across gates for averaging
                    for network in gate_result['named_networks']:
                        ssid = network['ssid']
                        
                        if ssid not in window_network_appearances:
                            window_network_appearances[ssid] = {
                                'display_name': network['display_name'],
                                'detection_counts': [],
                                'gate_count': 0
                            }
                        
                        # Record this gate's values for later averaging
                        window_network_appearances[ssid]['detection_counts'].append(network['detection_count'])
                        window_network_appearances[ssid]['gate_count'] += 1
            
            current_gate_start = current_gate_end
        
        if not gate_stats:
            return None
        
        # Calculate window-level statistics from gate statistics
        unique_network_counts = [gate['unique_networks'] for gate in gate_stats]
        named_network_counts = [gate['named_networks'] for gate in gate_stats]
        
        avg_unique_networks = sum(unique_network_counts) / len(unique_network_counts)
        min_unique_networks = min(unique_network_counts)
        max_unique_networks = max(unique_network_counts)
        
        avg_named_networks = sum(named_network_counts) / len(named_network_counts)
        min_named_networks = min(named_network_counts)
        max_named_networks = max(named_network_counts)
        
        # Calculate average statistics for each network across gates
        averaged_networks = []
        for ssid, stats in window_network_appearances.items():
            # Calculate average detections per gate where network appeared
            avg_detections = sum(stats['detection_counts']) / len(stats['detection_counts'])
            
            averaged_networks.append({
                'display_name': stats['display_name'],
                'avg_detections': avg_detections,
                'gate_appearances': stats['gate_count'],
                'total_gates': len(gate_stats)
            })
        
        # Sort by average detections (descending)
        averaged_networks.sort(key=lambda x: x['avg_detections'], reverse=True)
        
        # Generate description for the window
        description_parts = [f"wifi | WiFi Networks Detected"]
        
        # Show average and range of unique networks (calculated from gate_time_window-min gate scans)
        if min_unique_networks == max_unique_networks:
            description_parts.append(f"    - Average unique networks: {avg_unique_networks:.1f} (from {gate_time_window}-min gate scans)")
        else:
            description_parts.append(f"    - Average unique networks: {avg_unique_networks:.1f} (range: {min_unique_networks}-{max_unique_networks}, from {gate_time_window}-min gate scans)")
        
        # Show average and range of named networks (calculated from gate_time_window-min gate scans)
        if min_named_networks == max_named_networks:
            description_parts.append(f"    - Average named networks: {avg_named_networks:.1f} (from {gate_time_window}-min gate scans)")
        else:
            description_parts.append(f"    - Average named networks: {avg_named_networks:.1f} (range: {min_named_networks}-{max_named_networks}, from {gate_time_window}-min gate scans)")
        
        if averaged_networks:
            # Limit to top 10 networks
            top_networks = averaged_networks[:10]
            total_networks = len(averaged_networks)
            
            if total_networks > 10:
                description_parts.append(f"    - Top 10 of {total_networks} named networks (by average detection frequency from {gate_time_window}-min gate scans):")
            else:
                description_parts.append(f"    - {total_networks} named networks (by average detection frequency from {gate_time_window}-min gate scans):")
            
            for network in top_networks:
                description_parts.append(
                    f"         - {network['display_name']} "
                    f"({network['avg_detections']:.1f} detections)"
                )
        else:
            description_parts.append(f"    - No named networks detected")
        
        return '\n'.join(description_parts)
    
    def process_wifi_gate(gate_data):
        """Process wifi data for a single gate_time_window-minute gate."""
        if not gate_data:
            return None
        
        # Group networks by SSID and calculate statistics
        network_stats = {}
        
        for record in gate_data:
            ssid = record.get('ssid', '')
            
            if ssid not in network_stats:
                network_stats[ssid] = {
                    'detection_count': 0,
                    'display_name': ssid if ssid else '<unknown network>'
                }
            
            network_stats[ssid]['detection_count'] += 1
        
        if not network_stats:
            return None
        
        # Calculate total unique networks (including unnamed ones)
        total_unique_networks = len(network_stats)
        
        # Calculate named networks (networks with valid SSID)
        named_networks_with_stats = []
        for ssid, stats in network_stats.items():
            if ssid and ssid.strip():  # Has valid SSID
                named_networks_with_stats.append({
                    'ssid': ssid,
                    'display_name': stats['display_name'],
                    'detection_count': stats['detection_count']
                })
        
        # Sort by detection count (descending)
        named_networks_with_stats.sort(key=lambda x: x['detection_count'], reverse=True)
        
        return {
            'total_unique_networks': total_unique_networks,
            'named_networks': named_networks_with_stats
        }
    
    # Create combined data structure with sensor type tagging
    combined_data = []
    
    # Add sensor_wifi records with type tag
    for record in sensor_wifi_data or []:
        combined_record = dict(record)
        combined_record['_sensor_type'] = 'sensor_wifi'
        combined_data.append(combined_record)
    
    # Add wifi records with type tag
    for record in wifi_data or []:
        combined_record = dict(record)
        combined_record['_sensor_type'] = 'wifi'
        combined_data.append(combined_record)
    
    def process_wifi_combined_window_refactored(window_data, datetime_str, window_start, window_end):
        """Process combined wifi data for a single time window."""
        if not window_data:
            return None
        
        # Separate the data by sensor type
        sensor_wifi_window = [r for r in window_data if r.get('_sensor_type') == 'sensor_wifi']
        wifi_window = [r for r in window_data if r.get('_sensor_type') == 'wifi']
        
        # Call the original combined window processing function
        window_data_tuple = (sensor_wifi_window, wifi_window)
        return process_wifi_combined_window(window_data_tuple, datetime_str, window_start, window_end)
    
    # Process combined data using the shared helper function
    narratives = process_sensor_by_timewindow(
        combined_data, "wifi", start_timestamp, end_timestamp, process_wifi_combined_window_refactored
    )
    
    sensor_types = []
    if sensor_wifi_data:
        sensor_types.append("connections")
    if wifi_data:
        sensor_types.append("detections")
    
    log_info(f"Generated {len(narratives)} combined WiFi narratives ({', '.join(sensor_types)}) (window size: {sensor_integration_time_window} minutes)")
    return narratives

def describe_screentext_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None, pid=None):
    """
    Generate integrated screentext analysis by time windows.
    Loads screentext data from the config file and processes screen text logs and app usage durations.
    
    Important: The session_id in screentext data has been renumbered (to avoid gaps) and does NOT 
    correspond to the original session_id in sessions.jsonl. Session correlation is performed 
    using timestamp overlap between screentext records and original sessions only.
    
    Args:
        sensor_data (list): Unused - we load data from config file
        sensor_name (str): Name of the sensor (should be 'screentext')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of original session records for session correlation
        
    Returns:
        list: List of formatted screentext narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('screentext')
              - description: Human-readable narrative text
    """
    log_info("Generating integrated description for screentext")
    
    if sensor_name != "screentext":
        log_info("Invalid sensor name for screentext integration")
        return []
    
    # Load screentext data from config file
    cleaned_screentext_file = CONFIG.get("cleaned_screentext_file", "").format(P_ID=pid)
    if not cleaned_screentext_file or not os.path.exists(cleaned_screentext_file):
        log_info(f"Screentext file not found: {cleaned_screentext_file}")
        return []
    
    # Load and parse screentext data
    screentext_records = []
    try:
        with open(cleaned_screentext_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        session_record = json.loads(line)
                        # Extract screentext logs from the session record
                        if 'screen_text_logs' in session_record:
                            for text_log in session_record['screen_text_logs']:
                                # Add session_id to each text log for reference
                                text_log['session_id'] = session_record.get('session_id', 'Unknown')
                                screentext_records.append(text_log)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        log_error(f"Error loading screentext data: {e}")
        return []
    
    if not screentext_records:
        log_info("No screentext records found")
        return []
    
    # Filter records within the time range and add timestamp field
    filtered_records = []
    for record in screentext_records:
        try:
            start_timestamp_record = convert_timestring_to_timestamp(record.get('start_datetime', ''), CONFIG["timezone"])
            end_timestamp_record = convert_timestring_to_timestamp(record.get('end_datetime', ''), CONFIG["timezone"])
            
            # Include records that:
            # 1. Start within the time range, OR
            # 2. End within the time range, OR  
            # 3. Span across the time range (start before and end after)
            if ((start_timestamp <= start_timestamp_record < end_timestamp) or
                (start_timestamp <= end_timestamp_record < end_timestamp) or
                (start_timestamp_record < start_timestamp and end_timestamp_record > end_timestamp)):
                
                # Add timestamp field for compatibility with process_sensor_by_timewindow
                record['timestamp'] = start_timestamp_record
                filtered_records.append(record)
        except:
            continue
    
    if not filtered_records:
        log_info("No screentext records in time range")
        return []
    
    def process_screentext_window(window_data, datetime_str, window_start, window_end):
        """Process screentext data for a single time window."""
        if not window_data:
            return None
        
        # Sort window data by start time
        sorted_data = sorted(window_data, key=lambda x: x.get('start_datetime', ''))
        
        # Group by active period and app
        active_period_groups = {}
        total_screen_time = 0
        
        for record in sorted_data:
            package_name = record.get('package_name', 'Unknown')
            app_name = record.get('application_name', 'Unknown')
            duration = record.get('duration_seconds', 0)
            text = record.get('text', '')
            is_system_app = record.get('is_system_app', 0)
            start_datetime = record.get('start_datetime', '')
            end_datetime = record.get('end_datetime', '')
            active_period_id = record.get('active_period_id', '')

            # Skip blacklisted apps - compare package names
            if any(package_name.lower() == app.lower() for app in blacklist_apps):
                continue

            # This is based on cleaned_input.jsonl, so we do not using centralized should_filter_system_ui_app function
            if DISCARD_SYSTEM_UI and is_system_app == 1:
                # Skip system UI apps - compare package names
                if any(package_name.lower() == app.lower() for app in system_ui_apps):
                    continue
            
            if not text.strip():
                continue
            
            # Create unique key for active period + app combination
            period_key = f"{active_period_id}_{app_name}"
            
            # Initialize period group if not exists
            if period_key not in active_period_groups:
                active_period_groups[period_key] = {
                    'app_name': app_name,
                    'start_datetime': start_datetime,
                    'end_datetime': end_datetime,
                    'duration_seconds': duration,
                    'texts': []
                }
            else:
                # Update end time and duration if this record extends the period
                if end_datetime > active_period_groups[period_key]['end_datetime']:
                    active_period_groups[period_key]['end_datetime'] = end_datetime
                    active_period_groups[period_key]['duration_seconds'] += duration
                else:
                    active_period_groups[period_key]['duration_seconds'] += duration
            
            # Add to total screen time
            total_screen_time += duration
            
            # Add text to the period
            active_period_groups[period_key]['texts'].append(text.strip())
        
        # Return None if no content after filtering (don't display empty windows)
        if total_screen_time == 0:
            return None
        
        # Generate description
        description_parts = [f"screentext | Tracked Logs"]
        
        # Format total screen time
        if total_screen_time >= 3600:
            hours = int(total_screen_time // 3600)
            minutes = int((total_screen_time % 3600) // 60)
            seconds = int(total_screen_time % 60)
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif total_screen_time >= 60:
            minutes = int(total_screen_time // 60)
            seconds = int(total_screen_time % 60)
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{int(total_screen_time)}s"
        
        # uncomment to show screen text tracked time
        # description_parts.append(f"    - Screen text tracked time: {time_str}")
        
        # Count unique apps
        unique_apps = set(period['app_name'] for period in active_period_groups.values())
        description_parts.append(f"    - Apps tracked: {len(unique_apps)}")
        
        # Screen text logs in new format
        description_parts.append("    - Screen text logs breakdown:")
        
        # Sort periods by start time
        sorted_periods = sorted(active_period_groups.values(), key=lambda x: x['start_datetime'])
        
        for period in sorted_periods:
            app_name = period['app_name']
            start_dt = period['start_datetime']
            end_dt = period['end_datetime']
            duration = period['duration_seconds']
            
            # Check if this period extends beyond the current window
            period_start_ts = convert_timestring_to_timestamp(start_dt, CONFIG["timezone"])
            period_end_ts = convert_timestring_to_timestamp(end_dt, CONFIG["timezone"])
            extends_beyond_window = period_end_ts > window_end
            
            # Format duration
            if duration >= 3600:
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                duration_str = f"{hours}h {minutes}m {seconds}s"
            elif duration >= 60:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{int(duration)}s"
            
            # Add note if period extends beyond window
            period_line = f"        - {app_name} ({start_dt} - {end_dt}, {duration_str})"
            if extends_beyond_window:
                period_line += " (extends to following windows)"
            description_parts.append(period_line)
            
            # Add all texts for this period
            for text in period['texts']:
                # Create JSON format for the text
                text_json = json.dumps(text, ensure_ascii=False)
                description_parts.append(f"            - Text: {text_json}")
        
        # Show session correlation if sessions are available
        if sessions:
            # Find sessions that overlap with this window
            overlapping_sessions = []
            for session in sessions:
                if (session['start_timestamp'] <= window_end and 
                    session['end_timestamp'] >= window_start):
                    overlapping_sessions.append(session['session_id'])

        
        return '\n'.join(description_parts)
    
    # Process data using the shared helper function
    narratives = process_sensor_by_timewindow(
        filtered_records, sensor_name, start_timestamp, end_timestamp, process_screentext_window
    )
    
    log_info(f"Generated {len(narratives)} screentext narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives

def describe_installations_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
    """
    Generate integrated app installation analysis by time windows.
    Shows installation, removal, and update activities with app names and timing.
    
    Args:
        sensor_data (list): List of installations sensor records
        sensor_name (str): Name of the sensor (should be 'installations')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        sessions (list, optional): List of session records for session correlation
        
    Returns:
        list: List of formatted installations narrative dictionaries with keys:
              - unix_timestamp: Unix timestamp in milliseconds
              - sensor_type: Sensor type ('installations')
              - description: Human-readable narrative text
    """
    log_info("Generating integrated description for installations")
    
    if sensor_name != "installations" or not sensor_data:
        log_info("No installations data available, skipping installations integration")
        return []
    
    # Installation status mapping
    statuses = {
        0: "was removed",
        1: "was added", 
        2: "was updated"
    }
    
    def process_installations_window(window_data, datetime_str, window_start, window_end):
        """Process installations data for a single time window."""
        if not window_data:
            return None
        
        # Sort window data by timestamp
        sorted_data = sorted(window_data, key=lambda x: x['timestamp'])
        
        # Track installations by status
        installation_activities = {
            'added': [],
            'removed': [],
            'updated': []
        }
        
        total_activities = 0
        
        # Process each installation record
        for record in sorted_data:
            app_name = record.get('application_name', 'Unknown')
            installation_status = record.get('installation_status', -1)
            package_name = record.get('package_name', '')
            record_datetime = record.get('datetime', datetime_str)
            
            # Map status to readable format
            if installation_status == 0:
                status = "removed"
                installation_activities['removed'].append({
                    'app_name': app_name,
                    'package_name': package_name,
                    'datetime': record_datetime
                })
            elif installation_status == 1:
                status = "added"
                installation_activities['added'].append({
                    'app_name': app_name,
                    'package_name': package_name,
                    'datetime': record_datetime
                })
            elif installation_status == 2:
                status = "updated"
                installation_activities['updated'].append({
                    'app_name': app_name,
                    'package_name': package_name,
                    'datetime': record_datetime
                })
            else:
                # Unknown status, skip
                continue
            
            total_activities += 1
        
        if total_activities == 0:
            return None
        
        # Generate description for this window
        description_parts = [f"installations | App Installation Activity"]
        
        # Show total activities summary
        description_parts.append(f"    - Total activities: {total_activities}")
        
        # Show breakdown by type
        activity_breakdown = []
        for status, activities in installation_activities.items():
            if activities:
                activity_breakdown.append(f"{len(activities)} {status}")
        
        if activity_breakdown:
            description_parts.append(f"    - Activity breakdown: {', '.join(activity_breakdown)}")
        
        # Show detailed activities by type
        for status, activities in installation_activities.items():
            if activities:
                description_parts.append(f"    - Apps {status}:")
                
                # Sort by datetime for chronological order
                sorted_activities = sorted(activities, key=lambda x: x['datetime'])
                
                for activity in sorted_activities:
                    app_name = activity['app_name']
                    package_name = activity['package_name']
                    activity_time = activity['datetime'].split(' ')[1] if ' ' in activity['datetime'] else activity['datetime']
                    
                    # Handle empty or missing application names
                    if not app_name or app_name.strip() == '':
                        if package_name:
                            display_name = f"Unknown App ({package_name})"
                        else:
                            display_name = "Unknown App"
                    else:
                        display_name = app_name
                    
                    if package_name and package_name != app_name and app_name.strip():
                        description_parts.append(f"         - {display_name} ({package_name}) at {activity_time}")
                    else:
                        description_parts.append(f"         - {display_name} at {activity_time}")
        
        # Show timing patterns if multiple activities
        if total_activities > 1:
            # Calculate time span
            first_activity = sorted_data[0]
            last_activity = sorted_data[-1]
            
            first_time = first_activity.get('datetime', datetime_str).split(' ')[1]
            last_time = last_activity.get('datetime', datetime_str).split(' ')[1]
            
            if first_time != last_time:
                description_parts.append(f"    - Time span: {first_time} to {last_time}")
            
            # Show activity frequency
            time_window_minutes = sensor_integration_time_window
            activities_per_minute = total_activities / time_window_minutes
            if activities_per_minute > 1:
                description_parts.append(f"    - Frequency: {activities_per_minute:.1f} activities/minute")
            else:
                description_parts.append(f"    - Frequency: {total_activities} activities in {time_window_minutes} minutes")
        
        # Show session correlation if sessions are available
        if sessions:
            # Find sessions that overlap with this window
            overlapping_sessions = []
            for session in sessions:
                if (session['start_timestamp'] <= window_end and 
                    session['end_timestamp'] >= window_start):
                    overlapping_sessions.append(session['session_id'])
            
            if overlapping_sessions:
                if len(overlapping_sessions) == 1:
                    description_parts.append(f"    - Session activity: Session {overlapping_sessions[0]}")
                else:
                    description_parts.append(f"    - Session activity: Sessions {', '.join(map(str, overlapping_sessions))}")
        
        return '\n'.join(description_parts)
    
    # Process data using the shared helper function
    narratives = process_sensor_by_timewindow(
        sensor_data, sensor_name, start_timestamp, end_timestamp, process_installations_window
    )
    
    log_info(f"Generated {len(narratives)} installations narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives



if __name__ == "__main__":
    # Set up logging with minimal console output (only errors and summaries)
    main_logger = setup_logging(console_level=logging.ERROR)
    
    # Check processing mode
    mode = CONFIG.get("MODE", "manual").lower()
    log_summary(f"Start narrating sensor data in {mode.upper()} mode...")
    
    # App package mapping is now loaded globally at module import
    # Using the global application_name_list variable
    
    # Process based on mode
    if mode == "auto":
        # Auto mode: Process participants using survey time file and time ranges
        direction = CONFIG.get("direction", "backward")
        log_summary(f"Running in AUTO mode with survey time file: {CONFIG['survey_time_file']}")
        if "time_range_start" in CONFIG and "time_range_end" in CONFIG:
            time_ranges_display = f"{CONFIG['time_range_start']} to {CONFIG['time_range_end']}"
        else:
            time_ranges_display = str(CONFIG.get('time_ranges', []))
        log_summary(f"Time ranges: {time_ranges_display}, Direction: {direction}")
        
        success = process_auto_mode()
        
        # Auto mode already prints its own detailed summary
        if not success:
            log_summary(f"\n{'='*60}")
            log_summary("AUTO MODE PROCESSING FAILED")
            log_summary(f"{'='*60}")
            log_summary("✗ No participants were successfully processed")
        
    else:
        # Manual mode: Process each participant with fixed time range
        P_IDs = CONFIG["P_IDs"]  # Only load participant IDs in manual mode
        log_summary(f"Running in MANUAL mode with participants: {P_IDs}")
        log_summary(f"Time range: {CONFIG['START_TIME']} to {CONFIG['END_TIME']}")
        
        successful_participants = []
        failed_participants = []

        num_workers = CONFIG.get("num_workers", 1)

        if num_workers > 1:
            log_summary(f"Processing {len(P_IDs)} participants in parallel (num_workers={num_workers})")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(process_participant_manual, pid): pid
                    for pid in P_IDs
                }
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        success = future.result()
                        if success:
                            successful_participants.append(pid)
                            log_summary(f"FINISHED participant: {pid}")
                            log_summary(f"{'='*50}")
                        else:
                            failed_participants.append(pid)
                    except Exception as e:
                        log_error(f"Error processing participant {pid}: {e}")
                        failed_participants.append(pid)
        else:
            for pid in P_IDs:
                try:
                    success = process_participant_manual(pid)
                    if success:
                        successful_participants.append(pid)
                        log_summary(f"FINISHED participant: {pid}")
                        log_summary(f"{'='*50}")
                    else:
                        failed_participants.append(pid)
                except Exception as e:
                    log_error(f"Error processing participant {pid}: {e}")
                    failed_participants.append(pid)
        
        # Print summary for manual mode
        log_summary(f"\n{'='*60}")
        log_summary("MANUAL MODE PROCESSING SUMMARY")
        log_summary(f"{'='*60}")
        log_summary(f"Successfully processed: {len(successful_participants)} participants")
        if successful_participants:
            log_summary(f"  - {', '.join(successful_participants)}")
        
        log_summary(f"Failed to process: {len(failed_participants)} participants")
        if failed_participants:
            log_summary(f"  - {', '.join(failed_participants)}")
        
        log_summary(f"\nTotal participants: {len(P_IDs)}")
        
        # Save summary to file for manual mode
        try:
            # Use the base output directory (without P_ID formatting)
            base_output_dir = CONFIG["output_dir"].replace("/{P_ID}", "")
            summary_file = os.path.join(base_output_dir, "processing_summary.txt")
            os.makedirs(base_output_dir, exist_ok=True)
            
            # Add timestamp to the summary
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Build summary lines
            summary_lines = []
            summary_lines.append("MANUAL MODE PROCESSING SUMMARY")
            summary_lines.append("="*60)
            summary_lines.append(f"Successfully processed: {len(successful_participants)} participants")
            if successful_participants:
                summary_lines.append(f"  - {', '.join(successful_participants)}")
            
            summary_lines.append(f"Failed to process: {len(failed_participants)} participants")
            if failed_participants:
                summary_lines.append(f"  - {', '.join(failed_participants)}")
            
            summary_lines.append(f"\nTotal participants: {len(P_IDs)}")
            summary_lines.append(f"Time range: {CONFIG['START_TIME']} to {CONFIG['END_TIME']}")
            summary_lines.append(f"JSON output: description/{{P_ID}}/{{P_ID}}_manual.json")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Processing completed at: {timestamp}\n")
                f.write(f"Mode: Manual\n")
                f.write(f"Time range: {CONFIG['START_TIME']} to {CONFIG['END_TIME']}\n\n")
                f.write('\n'.join(summary_lines))
            
            log_info(f"\n✓ Summary saved to: {summary_file}", main_logger)
            
        except Exception as e:
            log_warning(f"\n⚠️  Warning: Could not save summary to file: {e}", main_logger)
    
    log_summary("Sensor narration completed!")