"""
Clean Screentext JSONL Script

This script processes screen text data by performing the following operations:
  - Loads JSONL data from a screentext file.
  - Filters and groups text content based on patterns (e.g., Rect(...) segments).
  - Extracts screen on/off events to create session boundaries.
  - Assigns session and active period IDs to each text record.
  - Enhances each record with application information and converts timestamps.
  - Removes duplicate entries within sessions.
  - Saves the cleaned data to a JSONL file.

Usage:
    Process a single participant:
       python clean_screentext_jsonl.py --participant 1234

    Process all participants:
       python clean_screentext_jsonl.py --all

The script uses the following directory structure by default:
  - Input directory: participant_data
  - Output directory: step1_data
"""

import os
import json
import re
import pandas as pd
from collections import defaultdict
import pytz
from datetime import datetime
import threading
from typing import Dict
import argparse

# Create a global lock for unknown_packages file access
unknown_packages_lock = threading.Lock()

class DataPaths:
    def __init__(self, base_input="participant_data", base_output="step1_data", participant_prefix="", p_id=None):
        """
        Initialize data paths configuration
        
        Parameters:
            base_input (str): Base directory for input data.
            base_output (str): Base directory for output data.
            participant_prefix (str): Prefix for participant folders.
            p_id (str or None): Participant ID.
        """
        self.base_input = base_input
        self.base_output = base_output
        self.participant_prefix = participant_prefix
        self.p_id = p_id
        
    def get_input_dir(self):
        """Get input directory path"""
        if self.p_id:
            return os.path.join(self.base_input, f"{self.participant_prefix}{self.p_id}")
        return self.base_input
    
    def get_output_dir(self):
        """Get output directory path"""
        if self.p_id:
            return os.path.join(self.base_output, f"{self.participant_prefix}{self.p_id}")
        return self.base_output
    
    def get_root_resources_dir(self):
        """Get root resources directory path"""
        return "resources"

def load_jsonl(file_path):
    """
    Load a JSONL file into a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the JSONL file.
    
    Returns:
        pd.DataFrame: The loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"File is empty: {file_path}")
    
    return pd.read_json(file_path, lines=True)

def load_screentext_data(paths):
    """Load screentext data from the defined input directory and filter out com.aware.phone records."""
    json_file = os.path.join(paths.get_input_dir(), "screentext.jsonl")
    df = load_jsonl(json_file)
    
    # Filter out records with package_name "com.aware.phone"
    initial_count = len(df)
    df = df[~df['package_name'].isin(['com.aware.phone'])]
    filtered_count = len(df)
    
    print(f"Filtered out {initial_count - filtered_count} com.aware.phone records from screentext data")
    
    return df

def filter_and_group_by_rect(text):
    """Filter and group text content based on 'Rect' segments.
    
    Args:
        text (str): Input text to filter.
    
    Returns:
        str: Joined string that groups content by detected Rect segments.
    """
    pattern = r'([\s\S]*?)Rect\(([-]?\d+),\s*([-]?\d+)\s*-\s*([-]?\d+),\s*([-]?\d+)\)\|\|'
    matches = re.findall(pattern, text)

    grouped_parts = defaultdict(list)
    for match in matches:
        content, x1, y1, x2, y2 = match
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        
        if 0 <= x1 < x2 and 0 <= y1 < y2:
            clean_content = re.sub(r'\s*\*{3}\s*', ' ', content).strip()
            key = (y1, y2)
            grouped_parts[key].append(clean_content)
    
    result = []
    for parts in grouped_parts.values():
        joined_content = "||".join(str(part) for part in parts)
        result.append(joined_content)
    
    one_string_result = "||".join(result)
    
    return one_string_result

def get_sessions(paths, threshold=45000):
    """Extract session data from screen on/off events."""
    json_file = os.path.join(paths.get_input_dir(), "screen.jsonl")
    df = load_jsonl(json_file)
    
    print(f"Loaded {len(df)} records from screen.jsonl")
    
    # Convert timestamp from nanoseconds to milliseconds
    df['timestamp'] = (pd.to_numeric(df['timestamp']) / 1_000_000).astype(int)
    
    # Only keep screen on/off events and sort by timestamp
    df = df[df['screen_status'].isin([0, 1])].sort_values('timestamp')
    
    print(f"Found {len(df)} screen on/off events")
    
    sessions = []
    session_start = None
    last_timestamp = None
    active_period_start = None
    active_periods = []
    active_seconds = 0
    
    for i in range(len(df)):
        current_status = df.iloc[i]['screen_status']
        current_timestamp = df.iloc[i]['timestamp']

        if current_status == 1:  # Screen turning on
            active_period_start = current_timestamp
            if last_timestamp is None or (current_timestamp - last_timestamp > threshold):
                if session_start is not None:
                    sessions.append({
                        'session_id': len(sessions) + 1,
                        'start_timestamp': int(session_start),
                        'end_timestamp': int(last_timestamp),
                        'active_periods': active_periods,
                        'duration_seconds': round((last_timestamp - session_start) / 1000, 2),
                        'active_seconds': round(active_seconds / 1000, 2)
                    })
                session_start = current_timestamp
                active_periods = []
                active_seconds = 0
                
        elif current_status == 0 and active_period_start is not None:
            period_duration = current_timestamp - active_period_start
            active_seconds += period_duration
            active_periods.append({
                'start': int(active_period_start),
                'end': int(current_timestamp)
            })
            active_period_start = None
            
        last_timestamp = current_timestamp
    
    # Add final session if present
    if session_start is not None:
        sessions.append({
            'session_id': len(sessions) + 1,
            'start_timestamp': int(session_start),
            'end_timestamp': int(last_timestamp),
            'active_periods': active_periods,
            'duration_seconds': round((last_timestamp - session_start) / 1000, 2),
            'active_seconds': round(active_seconds / 1000, 2)
        })
    
    result_df = pd.DataFrame(sessions)
    
    output_dir = paths.get_output_dir()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "sessions.jsonl")
    result_df.to_json(output_file, orient='records', lines=True)
    
    return result_df

def add_session_id_to_texts(text_df, session_df, paths):
    """Assign session and active period IDs to each text record.
    
    Args:
        text_df (pd.DataFrame): DataFrame containing screentext records.
        session_df (pd.DataFrame): DataFrame containing session details.
        paths (DataPaths): Object holding directory paths.
    
    Returns:
        pd.DataFrame: Text DataFrame updated with session_id and active_period_id.
    """
    print(f"\nInitial text records: {len(text_df)}")
    print(f"Number of sessions: {len(session_df)}")
    
    text_df['timestamp'] = (pd.to_numeric(text_df['timestamp']) / 1_000_000).astype(int)
    text_df['session_id'] = None
    text_df['active_period_id'] = None

    valid_sessions = session_df[session_df['active_periods'].apply(len) > 0].copy()
    print(f"Number of valid sessions (with active periods): {len(valid_sessions)}")

    print(f"\nTimestamp ranges:")
    print(f"Text data: {text_df['timestamp'].min()} to {text_df['timestamp'].max()}")
    print(f"Session data: {valid_sessions['start_timestamp'].min()} to {valid_sessions['end_timestamp'].max()}")
    
    for _, session in valid_sessions.iterrows():
        session_id = session['session_id']
        active_periods = session['active_periods']
        
        for idx, period in enumerate(active_periods):
            mask = (text_df['timestamp'] >= period['start']) & (text_df['timestamp'] < period['end'])
            text_df.loc[mask, 'session_id'] = int(session_id)
            text_df.loc[mask, 'active_period_id'] = f"{session_id}_{idx + 1}"
        
    # Analyze records that couldn't be assigned to any session (data quality monitoring)
    # These are screentext records that occurred outside of detected active periods
    dropped_df = text_df[text_df['session_id'].isnull()].copy()
    
    if len(dropped_df) > 0:
        # Focus on regular app records that were lost, as these might indicate data loss
        non_ui_packages = dropped_df[dropped_df['package_name'] != 'com.android.systemui']
        if len(non_ui_packages) > 0:
            print("\nFound non UI package names in dropped records:")
            for pkg in non_ui_packages['package_name'].unique():
                count = len(non_ui_packages[non_ui_packages['package_name'] == pkg])
                print(f"- {pkg}: {count} records")
        
        output_dir = paths.get_output_dir()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dropped_df.to_json(os.path.join(output_dir, 'dropped_text_df.jsonl'), orient='records', lines=True)
    
    return text_df

def add_application_info(df, paths):
    """Enhance records with application information.
    
    Args:
        df (pd.DataFrame): DataFrame with package_name field.
        paths (DataPaths): Object holding directory paths.
    
    Returns:
        pd.DataFrame: Updated DataFrame containing application_name and is_system_app.
    
    Raises:
        FileNotFoundError: If the app_package_pairs.jsonl file does not exist.
    """
    root_resources_dir = paths.get_root_resources_dir()
    app_pairs_file = os.path.join(root_resources_dir, "app_package_pairs.jsonl")
    
    if not os.path.exists(app_pairs_file):
        raise FileNotFoundError(f"app_package_pairs.jsonl not found in {root_resources_dir}. Please run generate_app_pairs.py first.")
    
    package_mappings = load_jsonl(app_pairs_file)
    
    app_name_map = dict(zip(package_mappings['package_name'], package_mappings['application_name']))
    system_app_map = dict(zip(package_mappings['package_name'], package_mappings['is_system_app']))
    
    unknown_packages_file = os.path.join(root_resources_dir, "unknown_packages.jsonl")
    package_counts = df['package_name'].value_counts().to_dict()
    
    with unknown_packages_lock:
        existing_unknown_dict: Dict[str, int] = {}
        if os.path.exists(unknown_packages_file):
            try:
                with open(unknown_packages_file, 'r') as f:
                    for line in f:
                        pkg_data = json.loads(line)
                        existing_unknown_dict[pkg_data['package_name']] = pkg_data['count']
            except:
                existing_unknown_dict = {}

        updated_unknown_dict = existing_unknown_dict.copy()
        for pkg, count in package_counts.items():
            if pkg not in app_name_map:
                updated_unknown_dict[pkg] = updated_unknown_dict.get(pkg, 0) + count

        if updated_unknown_dict:
            unknown_packages = [
                {"package_name": pkg, "count": count}
                for pkg, count in updated_unknown_dict.items()
            ]
            with open(unknown_packages_file, 'w') as f:
                for pkg_data in unknown_packages:
                    f.write(json.dumps(pkg_data) + '\n')

    df['application_name'] = df['package_name'].map(app_name_map).combine_first(df['package_name'])
    df['is_system_app'] = df['package_name'].map(system_app_map).fillna(1)
    df = df.drop(columns=['package_name'])

    return df

def process_all_sessions(df, paths):
    """Merge screentext records with session details.
    
    Args:
        df (pd.DataFrame): DataFrame containing screentext data with session IDs.
        paths (DataPaths): Object holding directory paths.
    
    Returns:
        pd.DataFrame: DataFrame after consolidating all session data.
    """
    sessions_file = os.path.join(paths.get_output_dir(), "sessions.jsonl")
    sessions_df = load_jsonl(sessions_file)
    
    valid_session_ids = df['session_id'].unique()
    sessions_df = sessions_df[sessions_df['session_id'].isin(valid_session_ids)].copy()
    
    df = df[df['session_id'].isin(sessions_df['session_id'])].copy()
    
    if len(df) == 0:
        print("No matching screen text and session data found")
        return pd.DataFrame()
        
    grouped = df.groupby(['session_id', 'active_period_id'])
    all_results = []
    
    for (session_id, active_period_id), group_df in grouped:
        session_info = sessions_df[sessions_df['session_id'] == session_id].iloc[0]
        period_index = int(active_period_id.split('_')[1]) - 1
        active_period = session_info['active_periods'][period_index]
        
        clean_session_df = remove_duplicates_within_session_with_active_period(
            group_df, 
            active_period,
            original_active_period_id=active_period_id
        )
        all_results.append(clean_session_df)

    if len(all_results) > 0:
        result_df = pd.concat(all_results, ignore_index=True)
    else:
        result_df = pd.DataFrame()

    return result_df

def convert_timestamp_column(df, timezone_str="Australia/Melbourne"):
    """
    Convert timestamp columns to the provided timezone (accounting for DST) 
    and compute duration.
    
    Parameters:
        df (pd.DataFrame): DataFrame with timestamp columns.
        timezone_str (str): Timezone to convert to (default: Australia/Melbourne).
    
    Returns:
        pd.DataFrame: Updated DataFrame with datetime conversions.
    """
    try:
        tz = pytz.timezone(timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Unknown timezone '{timezone_str}'. Falling back to UTC.")
        tz = pytz.timezone("UTC")
    
    def convert_with_offset(ts):
        dt = pd.to_datetime(ts, unit='ms', utc=True)
        local_time = dt.tz_convert(tz)
        return local_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    df['start_datetime'] = df['start_timestamp'].apply(convert_with_offset)
    df['end_datetime'] = df['end_timestamp'].apply(convert_with_offset)
    df['duration_seconds'] = ((df['end_timestamp'] - df['start_timestamp']) / 1000).round(2)
    df['utc_offset'] = df['start_timestamp'].apply(lambda x: tz.utcoffset(pd.to_datetime(x, unit='ms')).total_seconds() / 3600)
    
    df = df.drop(columns=['start_timestamp', 'end_timestamp'])
    
    return df

def remove_duplicates_within_session_with_active_period(df, active_period, original_active_period_id):
    """
    Remove duplicate screentext entries within a session active period while keeping the original active_period_id.
    """
    df['text'] = df['text'].apply(filter_and_group_by_rect)
    result = []
    
    period_df = df[
        (df['timestamp'] >= active_period['start']) & 
        (df['timestamp'] <= active_period['end'])
    ].copy()
    
    if len(period_df) == 0:
        return pd.DataFrame()
        
    period_df = period_df.sort_values(['timestamp', 'package_name'])
    
    current_group = []
    current_package = None
    
    for _, row in period_df.iterrows():
        if current_package != row['package_name']:
            if current_group:
                start_time = current_group[0]['timestamp']
                end_time = current_group[-1]['timestamp']
                all_texts = set()
                for item in current_group:
                    all_texts.update(item['text'].split("||"))
                
                result.append({
                    'start_timestamp': start_time,
                    'end_timestamp': end_time,
                    'package_name': current_package,
                    'text': "||".join(all_texts),
                    'session_id': row['session_id'],
                    'active_period_id': original_active_period_id
                })
            
            current_package = row['package_name']
            current_group = [row]
        else:
            current_group.append(row)
    
    if current_group:
        start_time = current_group[0]['timestamp']
        end_time = min(active_period['end'], current_group[-1]['timestamp'])
        all_texts = set()
        for item in current_group:
            all_texts.update(item['text'].split("||"))
        
        result.append({
            'start_timestamp': start_time,
            'end_timestamp': end_time,
            'package_name': current_package,
            'text': "||".join(all_texts),
            'session_id': current_group[0]['session_id'],
            'active_period_id': original_active_period_id
        })
    
    return pd.DataFrame(result)

def renumber_sessions(df):
    """
    Renumber sessions sequentially while preserving original session info.
    """
    df['orig_period_num'] = df['active_period_id'].str.split('_').str[1].astype(int)
    df = df.sort_values(['session_id', 'orig_period_num'])
    df['original_session_id'] = df['session_id']
    df['original_active_period_id'] = df['active_period_id']
    df['orig_period_num'] = df['active_period_id'].str.split('_').str[1].astype(int)
    df = df.sort_values(['session_id', 'orig_period_num'])
    
    unique_sessions = df['session_id'].unique()
    session_map = {old: new + 1 for new, old in enumerate(sorted(unique_sessions))}
    
    for orig_session_id in unique_sessions:
        session_mask = df['session_id'] == orig_session_id
        unique_periods = df.loc[session_mask, 'active_period_id'].unique()
        new_session_id = session_map[orig_session_id]
        period_map = {old: f"{new_session_id}_{idx+1}" 
                     for idx, old in enumerate(sorted(unique_periods))}
        
        df.loc[session_mask, 'session_id'] = new_session_id
        df.loc[session_mask, 'active_period_id'] = df.loc[session_mask, 'active_period_id'].map(period_map)
    
    df = df.drop(columns=['orig_period_num'])
    df = df.drop(columns=['original_session_id', 'original_active_period_id'])
    
    return df

def clean_screentext(paths, timezone="Australia/Melbourne"):
    """
    Clean and process screentext data, merging session details, application info, and formatting timestamps.
    
    Parameters:
        paths (DataPaths): Object holding path configuration.
        timezone (str): Timezone for timestamp conversion, default is Australia/Melbourne.
    
    Returns:
        pd.DataFrame: The final cleaned screentext DataFrame.
    """
    output_dir = paths.get_output_dir()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    screentext_data = load_screentext_data(paths)
    df = pd.DataFrame(screentext_data)

    sessions_df = get_sessions(paths, threshold=45000)
    df = add_session_id_to_texts(df, sessions_df, paths)
    df = process_all_sessions(df, paths)

    if len(df) > 0:
        df = add_application_info(df, paths)
        df = convert_timestamp_column(df, timezone_str=timezone)
        df = renumber_sessions(df)
        
        output_file = os.path.join(output_dir, "clean_df.jsonl")
        df.to_json(output_file, orient='records', lines=True)
        print(f"Cleaned data saved to {output_file}")
    else:
        print("No records to process after session matching")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Clean screentext data for a single participant or all participants."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--participant', type=str, help="Participant ID to process (e.g., 1234)")
    group.add_argument('--all', action='store_true', help="Process all participants")
    parser.add_argument('--timezone', type=str, default="Australia/Melbourne", 
                        help="Timezone for timestamp conversion (default: Australia/Melbourne)")
    # New flag to override timezone with UTC
    parser.add_argument('--utc', action='store_true', help="If set, overrides timezone with UTC")
    args = parser.parse_args()
    
    if args.utc:
        args.timezone = "UTC"

    # Set default base directories.
    base_input = "participant_data"
    base_output = "step1_data"

    if args.participant:
        print(f"Processing participant {args.participant} ...")
        paths = DataPaths(base_input=base_input, base_output=base_output, p_id=args.participant)
        try:
            clean_screentext(paths, timezone=args.timezone)
        except Exception as e:
            print(f"Error processing participant {args.participant}: {str(e)}")
    else:
        print("Processing all participants...")
        participant_dirs = [d for d in os.listdir(base_input) if os.path.isdir(os.path.join(base_input, d))]
        if not participant_dirs:
            print(f"No participant directories found in {base_input}")
            return
        for p in participant_dirs:
            print(f"\nProcessing participant {p} ...")
            paths = DataPaths(base_input=base_input, base_output=base_output, p_id=p)
            try:
                clean_screentext(paths, timezone=args.timezone)
            except Exception as e:
                print(f"Error processing participant {p}: {str(e)}")

if __name__ == "__main__":
    main()