"""
Generate App Package Pair Script

This script generates a mapping between package names and their corresponding application names 
from the 'applications_foreground.jsonl' file found in a specified raw data directory. It also ensures 
that an 'is_system_app' column is present. The output is saved as 'app_package_pairs.jsonl' in the designated 
resources directory.

Usage:
    Process a specific participant:
       python generate_app_package_pair.py --participant 1234 [--resources_dir resources]

    You can also override the raw data directory:
       python generate_app_package_pair.py --raw_data_dir <path/to/raw_data> --resources_dir <path/to/resources>
"""

import os
import pandas as pd
import argparse

def load_jsonl(file_path):
    """
    Load a JSONL file into a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the JSONL file.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"File is empty: {file_path}")
    
    return pd.read_json(file_path, lines=True)

def generate_app_pairs(raw_data_dir, resources_dir):
    """
    Generate app_package_pairs.jsonl from the applications_foreground.jsonl file.
    
    Parameters:
        raw_data_dir (str): Directory containing the source file.
        resources_dir (str): Directory to save the resulting file.
    """
    input_file = os.path.join(raw_data_dir, "applications_foreground.jsonl")
    
    try:
        applications_data = load_jsonl(input_file)
        print(f"Processing: {input_file}")
        
        if 'is_system_app' not in applications_data.columns:
            applications_data['is_system_app'] = 1
        
        final_mappings = applications_data[['package_name', 'application_name', 'is_system_app']].drop_duplicates(subset=['package_name'])
        
        os.makedirs(resources_dir, exist_ok=True)
        
        output_file = os.path.join(resources_dir, "app_package_pairs.jsonl")
        final_mappings.to_json(output_file, orient='records', lines=True)
        print(f"\nSaved {len(final_mappings)} unique package mappings to {output_file}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate app package pairs from applications_foreground.jsonl"
    )
    parser.add_argument('--participant', '-p', type=str, help="Participant ID to process (overrides raw_data_dir)")
    parser.add_argument('--raw_data_dir', type=str, default=None, help="Directory with raw data (if not using participant ID)")
    parser.add_argument('--resources_dir', type=str, default="resources", help="Directory to save app package pairs")
    args = parser.parse_args()

    # Determine raw_data_dir: if participant is provided and raw_data_dir not explicitly set,
    # assume raw_data_dir = "participant_data/<participant>"
    if args.participant and not args.raw_data_dir:
        raw_data_dir = os.path.join("participant_data", args.participant)
    elif args.raw_data_dir:
        raw_data_dir = args.raw_data_dir
    else:
        # Default value if nothing is provided
        raw_data_dir = os.path.join("participant_data", "1234")
    
    generate_app_pairs(raw_data_dir, args.resources_dir)

if __name__ == "__main__":
    main()