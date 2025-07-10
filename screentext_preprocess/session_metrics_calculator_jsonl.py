"""
Session Metrics Calculator JSONL Script

This script processes system app transition data that already has day identifiers and computes 
session-level metrics such as total session duration, main app usage duration, and app usage patterns.
It also extracts additional temporal features (e.g., time of day, day of week) and produces a cleaned 
JSONL output combining session summaries with individual screen text records.

Usage:
    Process a specific participant:
       python session_metrics_calculator_jsonl.py --participant 1234

    Process all participants:
       python session_metrics_calculator_jsonl.py --all
"""

from datetime import datetime
import json
from collections import Counter
from typing import List, Dict, Any
import sys
import os
import re
import argparse

INVISIBLE_CHARS = {
    '\u200b': 'zero-width space',
    '\u200c': 'zero-width non-joiner',
    '\u200d': 'zero-width joiner',
    '\u200e': 'left-to-right mark',
    '\u200f': 'right-to-left mark',
    '\ufeff': 'zero-width no-break space (BOM)',
    '\u2060': 'word joiner',
    '\u2061': 'function application',
    '\u2062': 'invisible times',
    '\u2063': 'invisible separator',
    '\u2064': 'invisible plus'
}

def format_duration(total_seconds):
    """Format a duration in seconds into a human-readable string.
    
    Args:
        total_seconds (float): Duration in seconds.
    
    Returns:
        str: Duration formatted as "1h30m15s" or similar.
    """
    total_seconds = round(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    duration_parts = []
    if hours > 0:
        duration_parts.append(f"{hours}h")
    if minutes > 0 or (hours > 0 and seconds > 0):
        duration_parts.append(f"{minutes}m")
    duration_parts.append(f"{seconds}s")
    return "".join(duration_parts)

def format_datetime_for_filename(datetime_str):
    """Format a datetime string for use in filenames.
    
    Args:
        datetime_str (str): Date/time string in the format "%Y-%m-%d %H:%M:%S.%f".
    
    Returns:
        str: Formatted string, e.g., "20210101_123456".
    """
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.strftime("%Y%m%d_%H%M%S")

def get_dates_from_data(data):
    """Extract the start and end dates from session data.
    
    Args:
        data (list): List of session records.
    
    Returns:
        tuple: (start_date, end_date) as strings.
    """
    sorted_data = sorted(data, key=lambda x: x["start_datetime"])
    start_date = sorted_data[0]["start_datetime"]
    end_date = sorted_data[-1]["end_datetime"]
    return start_date, end_date

def get_time_of_day_category(datetime_str):
    """Categorize the time of day based on the hour.
    
    Args:
        datetime_str (str): Datetime string in "%Y-%m-%d %H:%M:%S.%f" format.
    
    Returns:
        str: Time of day category ("morning", "afternoon", "evening", or "late_night").
    """
    hour = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f").hour
    if 5 <= hour < 11:
        return "morning"
    elif 11 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 23:
        return "evening"
    else:
        return "late_night"

def get_day_of_week_category(datetime_str):
    """Return the day of the week for a given datetime string.
    
    Args:
        datetime_str (str): Datetime string in "%Y-%m-%d %H:%M:%S.%f" format.
    
    Returns:
        str: Day of the week (e.g., "monday").
    """
    day = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f").weekday()
    days = {0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday", 4: "friday", 5: "saturday", 6: "sunday"}
    return days[day]

def is_weekend(datetime_str):
    """Determine whether the datetime corresponds to a weekend.
    
    Args:
        datetime_str (str): Datetime string in "%Y-%m-%d %H:%M:%S.%f" format.
    
    Returns:
        bool: True if weekend, False otherwise.
    """
    day = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f").weekday()
    return day >= 5

def calculate_session_metrics(session_data: List[dict]) -> Dict[str, Any]:
    """Calculate metrics for a session, including duration and app usage.
    
    Args:
        session_data (List[dict]): List of records in a session.
    
    Returns:
        dict: Dictionary containing session metrics and app metrics.
    """
    sorted_data = sorted(session_data, key=lambda x: x["start_datetime"])
    session_start = sorted_data[0]["start_datetime"]
    session_end = sorted_data[-1]["end_datetime"]
    app_sequences = []
    main_app_durations = {}
    for entry in session_data:
        app = entry["application_name"]
        duration = entry["duration_seconds"]
        if not entry["is_system_app"]:
            app_sequences.append({"app": app, "duration": duration})
            main_app_durations[app] = main_app_durations.get(app, 0) + duration
    main_usage_duration = sum(main_app_durations.values())
    top_3_apps = sorted(main_app_durations.items(), key=lambda x: x[1], reverse=True)[:3]
    total_duration = sum(entry["duration_seconds"] for entry in session_data)
    return {
        "session_metrics": {
            "start_time": session_start,
            "end_time": session_end,
            "total_duration": total_duration,
            "main_usage_duration": main_usage_duration,
            "main_usage_ratio": main_usage_duration / total_duration if total_duration > 0 else 0
        },
        "app_metrics": {
            "primary_apps": [
                {"name": app, "duration": duration, "usage_ratio": duration / main_usage_duration if main_usage_duration > 0 else 0}
                for app, duration in top_3_apps
            ],
            "app_sequence": app_sequences
        }
    }

def process_sessions(data: List[dict]) -> Dict[str, Any]:
    """Process session data to compute metrics for each session.
    
    Args:
        data (List[dict]): List of session records.
    
    Returns:
        dict: Mapping of session identifiers to their computed metrics.
    """
    sessions = {}
    for entry in data:
        session_id = entry["session_id"]
        sessions.setdefault(session_id, []).append(entry)
    output = {}
    for session_id, session_data in sessions.items():
        session_data.sort(key=lambda x: x["start_datetime"])
        metrics = calculate_session_metrics(session_data)
        output[f"session_{session_id}"] = {
            "time_range": {"start": metrics["session_metrics"]["start_time"], "end": metrics["session_metrics"]["end_time"]},
            "duration": {
                "total": format_duration(metrics["session_metrics"]["total_duration"]),
                "main_usage": format_duration(metrics["session_metrics"]["main_usage_duration"]),
                "main_usage_ratio": round(metrics["session_metrics"]["main_usage_ratio"], 2)
            },
            "apps": {
                "primary_apps": [
                    {"name": app["name"], "duration": format_duration(app["duration"]), "usage_ratio": round(app["usage_ratio"], 2)}
                    for app in metrics["app_metrics"]["primary_apps"][:3]
                ]
            }
        }
    return output

def simplify_app_sequence(app_sequences):
    """Simplify an app usage sequence by combining consecutive duplicates.
    
    Args:
        app_sequences (list): List of dictionaries representing app usage.
    
    Returns:
        list: Simplified app sequence.
    """
    if not app_sequences:
        return []
    simplified = []
    current_app = app_sequences[0]
    current_duration = 0
    for seq in app_sequences:
        if seq["app"] == current_app["app"]:
            current_duration += seq["duration"]
        else:
            simplified.append({"app": current_app["app"], "duration": current_duration})
            current_app = seq
            current_duration = seq["duration"]
    simplified.append({"app": current_app["app"], "duration": current_duration})
    return simplified

def combine_metrics_with_records(data: List[dict]) -> List[dict]:
    """Combine computed session metrics with original records.
    
    Args:
        data (List[dict]): List of records.
    
    Returns:
        List[dict]: Combined data entries, each including session metrics and logs.
    """
    metrics_by_session = process_sessions(data)
    sessions = {}
    session_day_ids = {}
    for entry in data:
        session_id = entry["session_id"]
        sessions.setdefault(session_id, [])
        if session_id not in session_day_ids:
            session_day_ids[session_id] = entry.get('day_id')
        record = entry.copy()
        record.pop('session_id', None)
        record.pop('utc_offset', None)
        record.pop('day_id', None)
        sessions[session_id].append(record)
    
    combined_data = []
    for session_id, records in sessions.items():
        metrics_key = f"session_{session_id}"
        if metrics_key in metrics_by_session:
            combined_data.append({
                "session_id": session_id,
                "day_id": str(session_day_ids[session_id]),
                "overview": metrics_by_session[metrics_key],
                "screen_text_logs": records
            })
    return combined_data

def process_participant_data(participant_id: str) -> None:
    """Process data for a specific participant by combining session metrics with records.
    
    Args:
        participant_id (str): The participant's ID.
    """
    input_path = f"step1_data/{participant_id}/system_app_transition_removed_2sec_with_day_id.jsonl"
    output_path = f"step1_data/{participant_id}/clean_input.jsonl"
    try:
        data = []
        pattern = '[' + ''.join(INVISIBLE_CHARS.keys()) + ']'
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if 'text' in record:
                        record['text'] = re.sub(pattern, '', record['text'])
                    data.append(record)
        combined_data = combine_metrics_with_records(data)
        with open(output_path, 'w') as f:
            for entry in combined_data:
                f.write(json.dumps(entry) + '\n')
        print(f"Processed participant {participant_id}")
    except FileNotFoundError:
        print(f"Error: Could not find data file for participant {participant_id}")
    except Exception as e:
        print(f"Error processing participant {participant_id}: {str(e)}")

def main():
    """Main function to calculate session metrics.
    
    Parses command-line arguments to process a specific participant or all participants in 'step1_data'.
    """
    parser = argparse.ArgumentParser(
        description="Calculate session metrics for a specific participant or all participants"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--participant', type=str, help="Participant ID to process")
    group.add_argument('--all', action='store_true', help="Process all participants in step1_data")
    args = parser.parse_args()
    
    base_dir = "step1_data"
    if args.participant:
        process_participant_data(args.participant)
    else:
        try:
            participants = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if not participants:
                print(f"Error: Could not find any participant directories in {base_dir}")
                return
            for participant_id in participants:
                process_participant_data(participant_id)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

# python session_metrics_calculator_jsonl.py --participant 1234
# python session_metrics_calculator_jsonl.py --all