"""
Add Day ID Script

This script processes JSONL files containing system app transition data for 
each participant and assigns a day identifier (day_id) based on the date extracted 
from the 'start_datetime' field. It can process either a specific participant or all participants.

Usage:
    For a specific participant:
        python add_day_id.py <base_dir> -p <participant_id>
    For all participants:
        python add_day_id.py <base_dir>
"""

import json
import os
from datetime import datetime
import argparse

def process_participant_jsonl(input_file, output_file):
    """Process a participant JSONL file to add day IDs.
    
    It reads each line from the input file, extracts the date from the 'start_datetime' field,
    assigns a sequential day_id to each unique date, and writes the updated records to the output file.
    
    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to the output JSONL file.
    
    Returns:
        int: Number of unique days found.
    """
    date_to_day_id = {}
    current_day_id = 1
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            data = json.loads(line.strip())
            date_str = data['start_datetime'].split()[0]
            if date_str not in date_to_day_id:
                print(f"New date found: {date_str} -> day_id: {current_day_id}")
                date_to_day_id[date_str] = current_day_id
                current_day_id += 1
            data['day_id'] = date_to_day_id[date_str]
            fout.write(json.dumps(data) + '\n')
    
    return len(date_to_day_id)

def process_all_participants(base_dir):
    """Process all participant folders to add day IDs.
    
    Iterates through each participant folder in the base directory, processes the corresponding
    JSONL file to add day IDs, and prints a summary of the results.
    
    Args:
        base_dir (str): Base directory containing participant folders.
    """
    participant_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    total_stats = {}
    
    for participant_id in participant_dirs:
        input_file = os.path.join(base_dir, participant_id, 'system_app_transition_removed_2sec.jsonl')
        output_file = os.path.join(base_dir, participant_id, 'system_app_transition_removed_2sec_with_day_id.jsonl')
        
        if os.path.exists(input_file):
            print(f"\nProcessing participant: {participant_id}")
            print("-" * 50)
            num_days = process_participant_jsonl(input_file, output_file)
            total_stats[participant_id] = num_days
    
    print("\nOverall Summary")
    print("=" * 50)
    print("Participant ID | Number of Days")
    print("-" * 50)
    for participant_id, num_days in sorted(total_stats.items()):
        print(f"{participant_id:<14}| {num_days}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process app transition data to add day IDs')
    parser.add_argument('base_dir', help='Base directory containing participant folders')
    parser.add_argument('--participant', '-p', help='Specific participant ID to process (optional)')
    
    args = parser.parse_args()
    
    if args.participant:
        input_file = os.path.join(args.base_dir, args.participant, 'system_app_transition_removed_2sec.jsonl')
        output_file = os.path.join(args.base_dir, args.participant, 'system_app_transition_removed_2sec_with_day_id.jsonl')
        if os.path.exists(input_file):
            print(f"\nProcessing participant: {args.participant}")
            print("-" * 50)
            num_days = process_participant_jsonl(input_file, output_file)
            print(f"\nProcessed {args.participant}: {num_days} days found")
        else:
            print(f"Error: Input file not found for participant {args.participant}")
    else:
        process_all_participants(args.base_dir)

# python add_day_id.py step1_data -p 1234