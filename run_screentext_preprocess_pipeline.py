#!/usr/bin/env python
"""Master Pipeline Script for Data Preprocessing.

This script orchestrates the data preprocessing pipeline for either a single participant
or all participants. The steps for a participant-specific processing are:

    1. Generate app package pairs (participant-specific).
    2. Clean screentext data.
    3. Generate filtered system app transition files.
    4. Add day IDs.
    5. Calculate session metrics.

Processing Modes:
    Single participant mode: All steps run sequentially for the specified participant.
    All participants mode: Step 1 runs sequentially for all participants first (to avoid
    thread safety issues with shared file writes), then Steps 2-5 run in parallel.

Usage:
    For one participant:
        $ python run_screentext_preprocess_pipeline.py --participant <participant_id> [--timezone <timezone>] [--utc]

    For processing all participants (Step 1 sequential, Steps 2-5 parallel):
        $ python run_screentext_preprocess_pipeline.py --all [--timezone <timezone>] [--utc] [--workers <num>]

    For processing specific participants only:
        $ python run_screentext_preprocess_pipeline.py --all --include 1234 5678 9012

    For processing all participants except specific ones:
        $ python run_screentext_preprocess_pipeline.py --all --exclude 1234 5678

Notes:
    Ensure that directories (e.g., participant_data, step1_data) exist in the parent directory.
    Also ensure that all required scripts support participant-specific operation (i.e. using -p).
    If a script (e.g., generate_system_app_transition_filtered_files.py) does not currently 
    support a participant-specific flag, you will need to modify it accordingly.
"""

import subprocess
import sys
import os
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_command(cmd_list, exit_on_error=True):
    """Run a system command as a subprocess.
    
    Prints the command, executes it, and either exits on error
    or raises the exception if exit_on_error is False.
    """
    print("Running command: " + " ".join(cmd_list))
    try:
        subprocess.run(cmd_list, check=True)
        print("Finished successfully: " + " ".join(cmd_list) + "\n")
    except subprocess.CalledProcessError as e:
        print("Error while executing: " + " ".join(cmd_list))
        if exit_on_error:
            sys.exit(e.returncode)
        else:
            raise e


def process_remaining_participant_pipeline(participant, timezone):
    """
    Process steps 2-5 of the preprocessing pipeline for a single participant.
    
    Step 1 (generate app package pairs) is handled sequentially before 
    calling this function to avoid thread safety issues with shared file writes.
    
    Steps processed:
      2. Clean screentext data for the participant.
      3. Generate filtered system app transition files.
      4. Add day IDs.
      5. Calculate session metrics.
    """
    print(f"Started processing participant {participant} (steps 2-5)")
    try:
        # Step 2: Clean screentext data for the participant.
        run_command(["python", "screentext_preprocess/clean_screentext_jsonl.py", "-p", participant,
                     "--timezone", timezone], exit_on_error=False)
        # Step 3: Generate filtered system app transition files.
        # (Ensure this script supports participant-specific processing using a flag like -p)
        run_command(["python", "screentext_preprocess/generate_system_app_transition_filtered_files.py", "-p", participant,
                     "--base_dir", "step1_data", "--mode", "generate", "--threshold", "2.0"],
                    exit_on_error=False)
        # Step 4: Add day IDs.
        run_command(["python", "screentext_preprocess/add_day_id.py", "step1_data", "-p", participant],
                    exit_on_error=False)
        # Step 5: Calculate session metrics.
        run_command(["python", "screentext_preprocess/session_metrics_calculator_jsonl.py", "-p", participant],
                    exit_on_error=False)
        print(f"Completed processing participant {participant} (steps 2-5)")
        return True, participant
    except Exception as e:
        print(f"Error processing participant {participant}: {e}")
        return False, participant


def process_full_participant_pipeline(participant, timezone):
    """
    Process the entire preprocessing pipeline for a single participant.
    
    For full parallelization, each participant's pipeline is executed sequentially
    within a separate thread:
      1. Generate app package pairs for the participant.
      2. Clean screentext data for the participant.
      3. Generate filtered system app transition files 
         (assumed to support a participant flag; modify if needed).
      4. Add day IDs.
      5. Calculate session metrics.
    """
    print(f"Started processing participant {participant}")
    try:
        # Step 1: Generate app package pairs for the participant.
        run_command(["python", "screentext_preprocess/generate_app_package_pair.py", "-p", participant],
                    exit_on_error=False)
        # Step 2: Clean screentext data for the participant.
        run_command(["python", "screentext_preprocess/clean_screentext_jsonl.py", "-p", participant,
                     "--timezone", timezone], exit_on_error=False)
        # Step 3: Generate filtered system app transition files.
        # (Ensure this script supports participant-specific processing using a flag like -p)
        run_command(["python", "screentext_preprocess/generate_system_app_transition_filtered_files.py", "-p", participant,
                     "--base_dir", "step1_data", "--mode", "generate", "--threshold", "2.0"],
                    exit_on_error=False)
        # Step 4: Add day IDs.
        run_command(["python", "screentext_preprocess/add_day_id.py", "step1_data", "-p", participant],
                    exit_on_error=False)
        # Step 5: Calculate session metrics.
        run_command(["python", "screentext_preprocess/session_metrics_calculator_jsonl.py", "-p", participant],
                    exit_on_error=False)
        print(f"Completed processing participant {participant}")
        return True, participant
    except Exception as e:
        print(f"Error processing participant {participant}: {e}")
        return False, participant


def main():
    """Execute the screentext preprocess pipeline."""
    # Calculate default workers as 75% of available CPU cores
    default_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    parser = argparse.ArgumentParser(
        description="Run screentext preprocess pipeline for one or all participants."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--participant', '-p', type=str,
                        help="Participant ID to process (e.g., 1234)")
    group.add_argument('--all', action='store_true',
                        help="Process data for all participants with hybrid sequential/parallel processing")
    
    parser.add_argument('--include', type=str, nargs='+', 
                        help="Include only these participant IDs (space-separated). Use with --all.")
    parser.add_argument('--exclude', type=str, nargs='+',
                        help="Exclude these participant IDs (space-separated). Use with --all.")
    parser.add_argument('--timezone', type=str, default="Australia/Melbourne",
                        help="Timezone for timestamp conversion (default: Australia/Melbourne)")
    parser.add_argument('--utc', action='store_true',
                        help="If set, overrides timezone with UTC")
    parser.add_argument('--workers', type=int, default=default_workers,
                        help=f"Number of worker threads for processing participants in parallel (default: {default_workers}, 75% of CPU cores)")
    args = parser.parse_args()

    # Validate include/exclude arguments
    if (args.include or args.exclude) and not args.all:
        parser.error("--include and --exclude can only be used with --all")
    
    if args.include and args.exclude:
        parser.error("Cannot use both --include and --exclude at the same time")

    if args.utc:
        args.timezone = "UTC"

    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}\n")

    if args.participant:
        # In single participant mode, run the pipeline sequentially.
        run_command(["python", "screentext_preprocess/generate_app_package_pair.py", "-p", args.participant])
        run_command(["python", "screentext_preprocess/clean_screentext_jsonl.py", "-p", args.participant,
                     "--timezone", args.timezone])
        run_command(["python", "screentext_preprocess/generate_system_app_transition_filtered_files.py", "-p", args.participant,
                     "--base_dir", "step1_data", "--mode", "generate", "--threshold", "2.0"])
        run_command(["python", "screentext_preprocess/add_day_id.py", "step1_data", "-p", args.participant])
        run_command(["python", "screentext_preprocess/session_metrics_calculator_jsonl.py", "-p", args.participant])
    else:
        # Sequential Step 1 followed by parallel Steps 2-5 processing
        participant_dir = "participant_data"
        if not os.path.exists(participant_dir):
            print("Error: participant_data directory does not exist.")
            sys.exit(1)

        participants = [d for d in os.listdir(participant_dir)
                        if os.path.isdir(os.path.join(participant_dir, d))]

        if not participants:
            print("No participants found in the participant_data directory.")
            sys.exit(1)

        # Filter participants based on include/exclude arguments
        original_count = len(participants)
        if args.include:
            participants = [p for p in participants if p in args.include]
            missing = [p for p in args.include if p not in participants]
            if missing:
                print(f"Warning: The following included participants were not found: {missing}")
            print(f"Including {len(participants)} participants: {participants}")
        elif args.exclude:
            excluded_found = [p for p in args.exclude if p in participants]
            participants = [p for p in participants if p not in args.exclude]
            print(f"Excluding {len(excluded_found)} participants: {excluded_found}")
            print(f"Processing {len(participants)} participants: {participants}")

        if not participants:
            if args.include:
                print("Error: None of the included participants were found.")
            else:
                print("Error: All participants were excluded.")
            sys.exit(1)

        print(f"\nFound {original_count} total participants, processing {len(participants)} participants.")
        
        # Step 1: Generate app package pairs sequentially for all participants
        print(f"\nStep 1: Generating app package pairs sequentially for {len(participants)} participants...")
        step1_failures = 0
        for i, participant in enumerate(participants, 1):
            print(f"Processing participant {participant} - Step 1 ({i}/{len(participants)})")
            try:
                run_command(["python", "screentext_preprocess/generate_app_package_pair.py", "-p", participant],
                           exit_on_error=False)
                print(f"Completed Step 1 for participant {participant}")
            except Exception as e:
                step1_failures += 1
                print(f"Failed Step 1 for participant {participant}: {e}")
        
        print(f"\nStep 1 completed. Failures: {step1_failures}/{len(participants)}")
        
        # Steps 2-5: Process remaining pipeline in parallel
        print(f"\nSteps 2-5: Processing remaining pipeline in parallel using {args.workers} worker threads...")
        
        successes = 0
        failures = 0
        total = len(participants)
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_participant = {
                executor.submit(process_remaining_participant_pipeline, participant, args.timezone): participant
                for participant in participants
            }
            for future in as_completed(future_to_participant):
                participant = future_to_participant[future]
                try:
                    success, pid = future.result()
                    if success:
                        successes += 1
                        print(f"Completed participant {pid} - Steps 2-5 ({successes + failures}/{total})")
                    else:
                        failures += 1
                        print(f"Failed participant {pid} - Steps 2-5 ({successes + failures}/{total})")
                except Exception as exc:
                    failures += 1
                    print(f"Exception processing participant {participant} - Steps 2-5: {exc} ({successes + failures}/{total})")
        
        print("\nProcessing Complete!")
        print(f"Step 1 failures: {step1_failures}/{len(participants)} participants")
        print(f"Steps 2-5 - Successfully processed: {successes}/{len(participants)} participants")
        print(f"Steps 2-5 - Failed to process: {failures}/{len(participants)} participants")
        total_failures = step1_failures + failures
        total_successes = len(participants) - total_failures
        print(f"Overall - Successfully processed: {total_successes}/{len(participants)} participants")
        print(f"Overall - Failed to process: {total_failures}/{len(participants)} participants")


if __name__ == '__main__':
    main()
