"""
Generate System App Transition Filtered Files Script

This script filters the cleaned data ('clean_df.jsonl') to remove entries where:
  - The record represents a system app (is_system_app = 1) and
  - The duration is below a specified threshold.
  
For each participant (or a specified participant), it outputs filtered JSONL files.
It also generates an analysis plot and a CSV file showing how different thresholds impact 
the percentage of records and sessions removed.

Usage:
    To generate filtered files:
        python generate_system_app_transition_filtered_files.py --base_dir step1_data --mode generate --threshold <value>
    To analyze threshold impact:
        python generate_system_app_transition_filtered_files.py --base_dir step1_data --mode plot
    (Optionally, a specific participant can be specified using --participant.)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from clean_screentext_jsonl import load_jsonl, renumber_sessions
import argparse

def renumber_filtered_sessions(df):
    """
    Renumber sessions chronologically while preserving period structure
    """
    # Store original order
    df['original_order'] = df.index
    
    # Get first timestamp for each session
    session_starts = (df.groupby('session_id')
                     .agg({'start_datetime': 'min'})
                     .reset_index())
    
    # Create chronological session mapping
    session_starts = session_starts.sort_values('start_datetime')
    session_mapping = {old: str(i+1) 
                      for i, old in enumerate(session_starts['session_id'])}
    
    # Renumber sessions and periods
    for orig_session_id in session_starts['session_id']:
        session_mask = df['session_id'] == orig_session_id
        unique_periods = df.loc[session_mask, 'active_period_id'].unique()
        new_session_id = session_mapping[orig_session_id]
        
        # Create new period IDs
        period_map = {old: f"{new_session_id}_{idx+1}" 
                     for idx, old in enumerate(sorted(unique_periods))}
        
        # Apply new IDs
        df.loc[session_mask, 'session_id'] = new_session_id
        df.loc[session_mask, 'active_period_id'] = \
            df.loc[session_mask, 'active_period_id'].map(period_map)
    
    # Restore original order
    df = df.sort_values('original_order').drop('original_order', axis=1)
    
    return df

def remove_system_app_transitions(df, threshold=3):
    """
    Remove records where is_system_app = 1 and duration_seconds < threshold,
    then update active_period_id and session_id while preserving relationships
    """
    # Create mask for records to keep
    mask = ~((df['is_system_app'] == 1) & (df['duration_seconds'] < threshold))
    filtered_df = df[mask].copy()
    
    # Ensure we have no NaN values
    filtered_df = filtered_df.dropna(subset=['active_period_id', 'session_id'])
    
    # Convert IDs to string type
    filtered_df['active_period_id'] = filtered_df['active_period_id'].astype(str)
    filtered_df['session_id'] = filtered_df['session_id'].astype(str)
    
    # Use custom renumbering function
    filtered_df = renumber_filtered_sessions(filtered_df)
    
    return filtered_df

def format_threshold_for_filename(threshold):
    if threshold % 1 == 0:
        return f"{int(threshold)}sec"
    elif threshold % 1 == 0.5:
        return f"{int(threshold)}andhalf_sec"
    else:
        threshold_str = str(threshold).rstrip('0').rstrip('.')
        threshold_str = threshold_str.replace('.', 'point')
        return f"{threshold_str}sec"

def analyze_filtered_results(original_df, filtered_results):
    analysis = []
    original_records = len(original_df)
    original_sessions = len(original_df['session_id'].unique())
    
    # Generate analysis data based on actual filtered results
    for threshold in sorted(filtered_results.keys()):
        df = filtered_results[threshold]
        analysis.append({
            'threshold': threshold,
            'records_retained': len(df),
            'records_removed': original_records - len(df),
            'records_removal_percentage': int(((original_records - len(df)) / original_records) * 100),  # Convert to int
            'sessions_retained': len(df['session_id'].unique()),
            'sessions_removed': original_sessions - len(df['session_id'].unique()),
            'sessions_removal_percentage': int(((original_sessions - len(df['session_id'].unique())) / original_sessions) * 100)  # Convert to int
        })
    
    analysis_df = pd.DataFrame(analysis)
    analysis_df = analysis_df.sort_values('threshold')
    
    # Create plot with actual threshold values
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot using actual threshold values
    ax.plot(analysis_df['threshold'], analysis_df['records_removal_percentage'], 
            marker='o', label='Records Removed (%)', linestyle='-')
    ax.plot(analysis_df['threshold'], analysis_df['sessions_removal_percentage'], 
            marker='o', color='orange', label='Sessions Removed (%)', linestyle='-')
    
    # Set x-axis ticks to show all threshold values
    ax.set_xticks(analysis_df['threshold'])
    ax.tick_params(axis='x', rotation=45, labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    # Add grid and labels with larger font size
    ax.set_xlabel('Threshold (seconds)', fontsize=20)
    ax.set_ylabel('Percentage Removed', fontsize=20)
    ax.set_title('Impact Analysis of System App Transition Removal', fontsize=22)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=18)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, analysis_df

def generate_system_app_transition_filtered_files(base_dir, p_id, thresholds):
    participant_dir = os.path.join(base_dir, f"{p_id}")
    clean_df_path = os.path.join(participant_dir, "clean_df.jsonl")
    
    if not os.path.exists(clean_df_path):
        raise FileNotFoundError(f"clean_df.jsonl not found in {participant_dir}")
    
    print("Loading clean_df.jsonl...")
    df = load_jsonl(clean_df_path)
    
    filtered_results = {}
    
    for threshold in thresholds:
        threshold_str = format_threshold_for_filename(threshold)
        
        filtered_output_file = os.path.join(
            participant_dir, 
            f"system_app_transition_removed_{threshold_str}.jsonl"
        )
        
        print(f"\nRemoving system app transitions with threshold: {threshold} seconds...")
        filtered_df = remove_system_app_transitions(df.copy(), threshold)
        filtered_df.to_json(filtered_output_file, orient='records', lines=True)
        
        filtered_results[threshold] = filtered_df
        
        # filtered_df['date'] = pd.to_datetime(filtered_df['start_datetime'])
        # min_date = filtered_df['date'].min()
        # one_day_cutoff = min_date + pd.Timedelta(days=1)
        # first_day_df = filtered_df[filtered_df['date'] < one_day_cutoff].copy()
        # first_day_df = first_day_df.drop('date', axis=1)
        
        # first_day_output_file = os.path.join(
        #     participant_dir, 
        #     f"system_app_transition_removed_{threshold_str}_first_day.jsonl"
        # )
        # first_day_df.to_json(first_day_output_file, orient='records', lines=True)
    
    fig, analysis_df = analyze_filtered_results(df, filtered_results)
    
    plot_file = os.path.join(participant_dir, "threshold_impact_analysis.png")
    fig.savefig(plot_file)
    plt.close(fig)
    
    analysis_file = os.path.join(participant_dir, "threshold_impact_analysis.csv")
    analysis_df.to_csv(analysis_file, index=False)
    
    print(f"\nAnalysis saved to {analysis_file}")
    print(f"Plot saved to {plot_file}")
    
    return analysis_df

def generate_thresholds(start, end, gap):
    """
    Generate a list of thresholds from start to end with specified gap
    
    Parameters:
    start: float, starting threshold value
    end: float, ending threshold value (inclusive)
    gap: float, increment between threshold values
    
    Returns:
    list of float values
    """
    thresholds = []
    current = start
    while current <= end:
        thresholds.append(round(current, 1))  # Round to 1 decimal place
        current = round(current + gap, 1)
    return thresholds

def analyze_filtered_results_multi_participant(all_dfs):
    """Analyze threshold impact across multiple participants"""
    thresholds = generate_thresholds(0.0, 10.0, 0.5)
    combined_analysis = []
    
    # Process each participant once per threshold
    for p_id, original_df in all_dfs.items():
        # Calculate these once per participant
        original_records = len(original_df)
        original_sessions = len(original_df['session_id'].unique())
        
        # Pre-process the DataFrame once
        df_copy = original_df.copy()
        df_copy['active_period_id'] = df_copy['active_period_id'].astype(str)
        df_copy['session_id'] = df_copy['session_id'].astype(str)
        if not df_copy['active_period_id'].str.contains('_').any():
            df_copy['active_period_id'] = df_copy['session_id'] + '_' + df_copy['active_period_id']
        
        for threshold in thresholds:
            # Apply threshold filter
            mask = ~((df_copy['is_system_app'] == 1) & (df_copy['duration_seconds'] < threshold))
            filtered_df = df_copy[mask]
            
            # Calculate metrics
            combined_analysis.append({
                'participant_id': p_id,
                'threshold': threshold,
                'records_removal_percentage': ((original_records - len(filtered_df)) / original_records) * 100,
                'sessions_removal_percentage': ((original_sessions - len(filtered_df['session_id'].unique())) / original_sessions) * 100
            })
    
    analysis_df = pd.DataFrame(combined_analysis)
    
    # Calculate summary statistics
    summary_df = analysis_df.groupby('threshold').agg({
        'records_removal_percentage': ['mean', 'std'],
        'sessions_removal_percentage': ['mean', 'std']
    }).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot mean with error bands
    ax.fill_between(summary_df['threshold'], 
                   summary_df[('records_removal_percentage', 'mean')] - summary_df[('records_removal_percentage', 'std')],
                   summary_df[('records_removal_percentage', 'mean')] + summary_df[('records_removal_percentage', 'std')],
                   alpha=0.2, color='blue')
    ax.plot(summary_df['threshold'], summary_df[('records_removal_percentage', 'mean')], 
            marker='o', label='Records Removed (%)', color='blue')
    
    ax.fill_between(summary_df['threshold'],
                   summary_df[('sessions_removal_percentage', 'mean')] - summary_df[('sessions_removal_percentage', 'std')],
                   summary_df[('sessions_removal_percentage', 'mean')] + summary_df[('sessions_removal_percentage', 'std')],
                   alpha=0.2, color='orange')
    ax.plot(summary_df['threshold'], summary_df[('sessions_removal_percentage', 'mean')], 
            marker='o', color='orange', label='Sessions Removed (%)')
    
    # Increase font size for labels, title, and ticks
    ax.set_xlabel('Threshold (seconds)', fontsize=20)
    ax.set_ylabel('Mean Percentage Removed (with std)', fontsize=20)
    ax.set_title('Impact Analysis of System App Transition Removal', fontsize=22)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=18)
    
    plt.tight_layout()
    return fig, analysis_df

def main():
    parser = argparse.ArgumentParser(
        description="Generate filtered system app transition files."
    )
    # Accept both -p and --participant flags
    parser.add_argument(
        '-p', '--participant', type=str,
        help="Participant ID to process (e.g., 1234). Leave empty to process all participants."
    )
    parser.add_argument(
        '--base_dir', type=str, default='step1_data',
        help="Base directory containing participant folders (default: step1_data)"
    )
    parser.add_argument(
        '--mode', type=str, default="generate",
        help="Mode of operation (default: generate)"
    )
    parser.add_argument(
        '--threshold', type=float, default=2.0,
        help="Threshold value (default: 2.0)"
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    all_dfs = {}

    if args.participant:
        # Load single participant
        clean_df_path = os.path.join(base_dir, args.participant, "clean_df.jsonl")
        if os.path.exists(clean_df_path):
            print(f"Loading data for participant {args.participant}...")
            all_dfs[args.participant] = load_jsonl(clean_df_path)
        else:
            raise FileNotFoundError(f"No data found for participant {args.participant}")
    else:
        # Load all participants
        for participant_folder in os.listdir(base_dir):
            participant_path = os.path.join(base_dir, participant_folder)
            if os.path.isdir(participant_path):
                clean_df_path = os.path.join(participant_path, "clean_df.jsonl")
                if os.path.exists(clean_df_path):
                    print(f"Loading data for participant {participant_folder}...")
                    all_dfs[participant_folder] = load_jsonl(clean_df_path)

    if args.mode == 'plot':
        print(f"\nAnalyzing {len(all_dfs)} participants...")
        fig, analysis_df = analyze_filtered_results_multi_participant(all_dfs)
        
        # Modify plot save location based on whether processing single participant
        if args.participant:
            plot_file = os.path.join(base_dir, args.participant, "threshold_impact_analysis.png")
        else:
            plot_file = os.path.join(base_dir, "threshold_impact_analysis_all_participants.png")
            
        fig.savefig(plot_file)
        print(f"Plot saved to {plot_file}")
        plt.show()
        
    elif args.mode == 'generate':
        if args.threshold is None:
            raise ValueError("Threshold must be specified when using generate mode")
            
        print(f"\nGenerating filtered files for {len(all_dfs)} participants using threshold {args.threshold}...")
        for p_id, df in all_dfs.items():
            threshold_str = format_threshold_for_filename(args.threshold)
            filtered_output_file = os.path.join(
                base_dir,
                p_id,
                f"system_app_transition_removed_{threshold_str}.jsonl"
            )
            
            print(f"\nProcessing participant {p_id}...")
            filtered_df = remove_system_app_transitions(df.copy(), args.threshold)
            filtered_df.to_json(filtered_output_file, orient='records', lines=True)
            print(f"Saved to {filtered_output_file}")

if __name__ == '__main__':
    main()