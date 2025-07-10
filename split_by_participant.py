#!/usr/bin/env python3
"""
Script to split JSONL files by participant ID based on device_id mapping.
Reads from exported_jsonl folder and splits files into participant_data/PID/ structure.
Also supports threshold analysis mode for analyzing record distributions.
"""

import os
import json
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_device_mapping(mapping_file: str) -> Dict[str, str]:
    """
    Load device_id to participant_id mapping from CSV file.
    
    Args:
        mapping_file: Path to the CSV file containing pid,device_id mapping
        
    Returns:
        Dictionary mapping device_id -> participant_id
    """
    device_to_pid = {}
    
    with open(mapping_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['pid']
            device_ids = row['device_id']
            
            # Split device_ids by semicolon and strip whitespace
            for device_id in device_ids.split(';'):
                device_id = device_id.strip()
                if device_id:  # Only add non-empty device_ids
                    device_to_pid[device_id] = pid
    
    return device_to_pid


def create_participant_directories(participant_data_dir: str, pids: Set[str]) -> None:
    """
    Create directory structure for each participant.
    
    Args:
        participant_data_dir: Base directory for participant data
        pids: Set of participant IDs
    """
    for pid in pids:
        participant_dir = os.path.join(participant_data_dir, pid)
        os.makedirs(participant_dir, exist_ok=True)


def split_jsonl_file(input_file: str, output_dir: str, device_mapping: Dict[str, str]) -> Tuple[Dict[str, int], Set[str], int]:
    """
    Split a JSONL file by participant based on device_id.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Base output directory
        device_mapping: Dictionary mapping device_id -> participant_id
        
    Returns:
        Tuple of (participant_counts, unknown_device_ids, skipped_count)
    """
    filename = os.path.basename(input_file)
    participant_counts = defaultdict(int)
    participant_files = {}
    unknown_device_ids = set()
    skipped_count = 0
    
    try:
        with open(input_file, 'r') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    device_id = record.get('device_id')
                    
                    if not device_id:
                        #print(f"Warning: No device_id found in line {line_num} of {filename}")
                        skipped_count += 1
                        continue
                    
                    pid = device_mapping.get(device_id)
                    if not pid:
                        unknown_device_ids.add(device_id)
                        skipped_count += 1
                        continue
                    
                    # Create output file handle if not exists
                    if pid not in participant_files:
                        output_file = os.path.join(output_dir, pid, filename)
                        participant_files[pid] = open(output_file, 'w')
                    
                    # Write record to participant's file
                    participant_files[pid].write(line + '\n')
                    participant_counts[pid] += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in line {line_num} of {filename}: {e}")
                    skipped_count += 1
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return participant_counts, unknown_device_ids, skipped_count
    
    finally:
        # Close all output files
        for f in participant_files.values():
            f.close()
    
    return participant_counts, unknown_device_ids, skipped_count


def save_unknown_device_report(unknown_devices_data: Dict[str, Tuple[Set[str], int]], output_dir: str) -> None:
    """
    Save unknown device IDs report to CSV file.
    
    Args:
        unknown_devices_data: Dictionary mapping filename -> (unknown_device_ids_set, skipped_count)
        output_dir: Output directory to save the report
    """
    report_file = os.path.join(output_dir, 'unknown_devices_report.csv')
    
    with open(report_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['table_name', 'unknown_device_ids', 'number_of_skipped_records'])
        
        for filename, (unknown_ids, skipped_count) in sorted(unknown_devices_data.items()):
            if unknown_ids or skipped_count > 0:  # Only write rows with unknown devices or skipped records
                unknown_ids_str = ';'.join(sorted(unknown_ids)) if unknown_ids else ''
                writer.writerow([filename, unknown_ids_str, skipped_count])
    
    print(f"Unknown devices report saved to: {report_file}")


def process_single_file(args_tuple):
    """
    Wrapper function for parallel processing of a single JSONL file.
    
    Args:
        args_tuple: Tuple of (input_path, output_dir, device_mapping, jsonl_file)
        
    Returns:
        Tuple of (jsonl_file, participant_counts, unknown_device_ids, skipped_count, file_total)
    """
    input_path, output_dir, device_mapping, jsonl_file = args_tuple
    
    print(f"Processing {jsonl_file}...")
    participant_counts, unknown_device_ids, skipped_count = split_jsonl_file(input_path, output_dir, device_mapping)
    
    file_total = sum(participant_counts.values())
    print(f"  {jsonl_file}: Processed {file_total} records")
    
    return jsonl_file, participant_counts, unknown_device_ids, skipped_count, file_total


def count_records_per_participant(input_file: str, device_mapping: Dict[str, str]) -> Dict[str, int]:
    """
    Count records per participant for a JSONL file without splitting.
    
    Args:
        input_file: Path to input JSONL file
        device_mapping: Dictionary mapping device_id -> participant_id
        
    Returns:
        Dictionary mapping participant_id -> record_count
    """
    participant_counts = defaultdict(int)
    
    try:
        with open(input_file, 'r') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    device_id = record.get('device_id')
                    
                    if not device_id:
                        continue
                    
                    pid = device_mapping.get(device_id)
                    if not pid:
                        continue
                    
                    participant_counts[pid] += 1
                    
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return {}
    
    return dict(participant_counts)


def perform_threshold_analysis(jsonl_files: List[str], input_dir: str, device_mapping: Dict[str, str], output_dir: str) -> None:
    """
    Perform threshold analysis on JSONL files and generate visualizations.
    
    Args:
        jsonl_files: List of JSONL filenames to analyze
        input_dir: Input directory containing JSONL files
        device_mapping: Dictionary mapping device_id -> participant_id
        output_dir: Output directory for results
    """
    print(f"\n=== Threshold Analysis Mode ===")
    print(f"Analyzing {len(jsonl_files)} JSONL files...")
    
    # Create output directory for threshold analysis
    analysis_dir = os.path.join('analysis_results', 'threshold_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Collect summary statistics for all files
    summary_stats = []
    
    for jsonl_file in jsonl_files:
        input_path = os.path.join(input_dir, jsonl_file)
        print(f"Counting records in {jsonl_file}...")
        
        participant_counts = count_records_per_participant(input_path, device_mapping)
        
        if not participant_counts:
            print(f"Warning: No records found for {jsonl_file}")
            continue
        

        
        # Calculate statistics
        counts = list(participant_counts.values())
        median_count = np.median(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        min_count = min(counts)
        max_count = max(counts)
        q25_count = np.percentile(counts, 25)
        q75_count = np.percentile(counts, 75)
        q90_count = np.percentile(counts, 90)
        
        # Calculate outlier information (beyond 90th percentile)
        outliers = [c for c in counts if c > q90_count]
        outlier_min = min(outliers) if outliers else None
        outlier_max = max(outliers) if outliers else None
        outlier_count = len(outliers)
        
        print(f"  {jsonl_file}: {len(participant_counts)} participants")
        print(f"    Records - Mean: {mean_count:.1f}, Median: {median_count:.1f}, Std: {std_count:.1f}")
        print(f"    Range: {min_count} - {max_count}")
        
        # Store summary statistics
        file_summary = {
            'filename': jsonl_file,
            'total_participants': len(participant_counts),
            'mean_records': mean_count,
            'median_records': median_count,
            'std_records': std_count,
            'min_records': min_count,
            'max_records': max_count,
            'q25_records': q25_count,
            'q75_records': q75_count,
            'q90_records': q90_count,
            'outlier_count': outlier_count,
            'outlier_min': outlier_min,
            'outlier_max': outlier_max,
            'threshold_increment': None,
            'thresholds_analyzed': None
        }
        
        # Generate thresholds using logarithmic scaling
        # Handle edge case where min_count is 0
        min_nonzero = min([c for c in counts if c > 0]) if any(c > 0 for c in counts) else 1
        log_min = max(1, min_nonzero)  # Start from at least 1
        log_max = max_count
        
        if log_max > log_min:
            # Generate logarithmically spaced thresholds
            num_thresholds = min(50, max(20, int(np.log10(log_max/log_min) * 10)))  # 10 points per order of magnitude
            log_thresholds = np.logspace(np.log10(log_min), np.log10(log_max), num=num_thresholds)
            # Add 0 at the beginning and ensure unique integer values
            thresholds = [0] + sorted(list(set([int(t) for t in log_thresholds])))
        else:
            # Fallback for edge cases
            thresholds = [0, max_count]
        
        # Update summary with threshold info
        file_summary['threshold_increment'] = f"Logarithmic ({len(thresholds)} points)"
        file_summary['thresholds_analyzed'] = len(thresholds)
        
        # Calculate threshold statistics
        threshold_stats = []
        for threshold in thresholds:
            participants_above = sum(1 for count in counts if count >= threshold)
            percentage_above = (participants_above / len(counts)) * 100
            threshold_stats.append({
                'threshold': threshold,
                'participants_above_threshold': participants_above,
                'percentage_above_threshold': percentage_above,
                'total_participants': len(counts)
            })
        
        # Save threshold statistics to CSV
        stats_file = os.path.join(analysis_dir, f'{jsonl_file.replace(".jsonl", "")}_threshold_stats.csv')
        with open(stats_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['threshold', 'participants_above_threshold', 'percentage_above_threshold', 'total_participants'])
            writer.writeheader()
            writer.writerows(threshold_stats)
        
        print(f"    Threshold stats saved to: {stats_file}")
        
        # Generate box plot
        generate_box_plot(counts, jsonl_file, analysis_dir)
        
        # Generate threshold trend plot
        generate_threshold_plot(threshold_stats, jsonl_file, analysis_dir)
        
        # Save participant counts to CSV
        participant_file = os.path.join(analysis_dir, f'{jsonl_file.replace(".jsonl", "")}_participant_counts.csv')
        with open(participant_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['participant_id', 'record_count'])
            for pid, count in sorted(participant_counts.items()):
                writer.writerow([pid, count])
        
        print(f"    Participant counts saved to: {participant_file}")
        
        # Add to summary stats
        summary_stats.append(file_summary)
    
    # Generate comprehensive text summary
    generate_text_summary(summary_stats, analysis_dir)
    
    print(f"\n✓ Threshold analysis complete! Results saved to: {analysis_dir}")


def generate_box_plot(counts: List[int], filename: str, output_dir: str) -> None:
    """
    Generate box plot for record counts distribution with outlier handling.
    
    Args:
        counts: List of record counts per participant
        filename: Name of the JSONL file
        output_dir: Output directory for plots
    """
    # Calculate statistics using all data
    mean_val = np.mean(counts)
    median_val = np.median(counts)
    std_val = np.std(counts)
    q90 = np.percentile(counts, 90)
    
    # Identify outliers beyond 90th percentile
    outliers = [c for c in counts if c > q90]
    regular_data = [c for c in counts if c <= q90]
    
    plt.figure(figsize=(12, 7))
    
    # Create box plot with outlier control
    box_plot = plt.boxplot(regular_data, vert=True, patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          showfliers=True)  # Show outliers within the 95% range
    
    plt.title(f'Record Count Distribution - {filename}', fontsize=18, fontweight='bold')
    plt.ylabel('Number of Records per Participant', fontsize=16)
    plt.xlabel('Distribution (90th percentile view)', fontsize=16)
    
    # Increase tick label font sizes
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Statistics text using ALL data
    stats_text = f'All Data Statistics:\nMean: {mean_val:.1f}\nMedian: {median_val:.1f}\nStd: {std_val:.1f}\nN: {len(counts)}'
    
    # Outlier information
    if outliers:
        outlier_text = f'\nOutliers (>{q90:.0f}):\nCount: {len(outliers)}\nRange: {min(outliers):.0f} - {max(outliers):.0f}'
        stats_text += outlier_text
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=14)
    
    # Add note about visualization range
    note_text = f'Note: Plot shows 90% of data (≤{q90:.0f} records)\nfor better readability'
    plt.text(0.98, 0.98, note_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             fontsize=14, style='italic')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f'{filename.replace(".jsonl", "")}_boxplot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print outlier summary
    if outliers:
        print(f"    Box plot saved to: {plot_file}")
        print(f"      └─ {len(outliers)} outliers (>{q90:.0f} records) excluded from plot for readability")
        print(f"         Outlier range: {min(outliers):,.0f} - {max(outliers):,.0f} records")
    else:
        print(f"    Box plot saved to: {plot_file} (no extreme outliers detected)")


def generate_threshold_plot(threshold_stats: List[Dict], filename: str, output_dir: str) -> None:
    """
    Generate threshold trend plot with dual y-axes.
    
    Args:
        threshold_stats: List of threshold statistics dictionaries
        filename: Name of the JSONL file
        output_dir: Output directory for plots
    """
    df = pd.DataFrame(threshold_stats)
    
    # Filter out threshold=0 for log scale (log(0) is undefined)
    df_nonzero = df[df['threshold'] > 0].copy()
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot 1: Participants above threshold (absolute numbers) - left y-axis
    line1 = ax1.semilogx(df_nonzero['threshold'], df_nonzero['participants_above_threshold'], 
                        'b-o', linewidth=2, markersize=5, label='Participants (count)')
    ax1.set_xlabel('Threshold (Minimum Records) - Log Scale', fontsize=16)
    ax1.set_ylabel('Participants Above Threshold', fontsize=16, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for percentage
    ax2 = ax1.twinx()
    line2 = ax2.semilogx(df_nonzero['threshold'], df_nonzero['percentage_above_threshold'], 
                        'r-s', linewidth=2, markersize=5, label='Participants (%)')
    ax2.set_ylabel('Percentage Above Threshold (%)', fontsize=16, color='red')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=14)
    ax2.set_ylim(0, 100)
    
    # Title and legend
    ax1.set_title(f'Threshold Analysis - {filename}', fontsize=18, fontweight='bold')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', framealpha=0.9, fontsize=14)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f'{filename.replace(".jsonl", "")}_threshold_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Threshold plot saved to: {plot_file}")



def generate_text_summary(summary_stats: List[Dict], output_dir: str) -> None:
    """
    Generate a comprehensive text summary of all threshold analysis results.
    
    Args:
        summary_stats: List of summary statistics dictionaries for each file
        output_dir: Output directory for the summary file
    """
    summary_file = os.path.join(output_dir, 'threshold_analysis_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("THRESHOLD ANALYSIS SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of JSONL files analyzed: {len(summary_stats)}\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-" * 40 + "\n")
        total_participants = sum(stat['total_participants'] for stat in summary_stats)
        total_records = sum(stat['mean_records'] * stat['total_participants'] for stat in summary_stats)
        
        f.write(f"Total unique participants across all files: {total_participants}\n")
        f.write(f"Total records across all files: {total_records:.0f}\n")
        f.write(f"Average records per participant (overall): {total_records/total_participants:.1f}\n\n")
        
        # Individual file statistics
        f.write("INDIVIDUAL FILE STATISTICS\n")
        f.write("-" * 40 + "\n\n")
        
        for i, stat in enumerate(summary_stats, 1):
            f.write(f"{i}. {stat['filename']}\n")
            f.write("   " + "="*50 + "\n")
            f.write(f"   Participants: {stat['total_participants']}\n")
            f.write(f"   Records per participant:\n")
            f.write(f"     Mean:   {stat['mean_records']:.1f}\n")
            f.write(f"     Median: {stat['median_records']:.1f}\n")
            f.write(f"     Std:    {stat['std_records']:.1f}\n")
            f.write(f"     Min:    {stat['min_records']}\n")
            f.write(f"     Q25:    {stat['q25_records']:.1f}\n")
            f.write(f"     Q75:    {stat['q75_records']:.1f}\n")
            f.write(f"     Q90:    {stat['q90_records']:.1f}\n")
            f.write(f"     Max:    {stat['max_records']}\n")
            f.write(f"     Range:  {stat['min_records']} - {stat['max_records']}\n")
            if stat['outlier_count'] > 0:
                f.write(f"     Outliers (>{stat['q90_records']:.0f}): {stat['outlier_count']} participants\n")
                f.write(f"     Outlier range: {stat['outlier_min']} - {stat['outlier_max']}\n")
            else:
                f.write(f"     Outliers (>{stat['q90_records']:.0f}): 0 participants\n")
            f.write(f"   Threshold analysis:\n")
            f.write(f"     Scaling method: {stat['threshold_increment']}\n")
            f.write(f"     Number of thresholds: {stat['thresholds_analyzed']}\n")
            f.write(f"   Data quality:\n")
            
            # Calculate some data quality metrics
            cv = stat['std_records'] / stat['mean_records'] if stat['mean_records'] > 0 else 0
            iqr = stat['q75_records'] - stat['q25_records']
            f.write(f"     Coefficient of variation: {cv:.3f}\n")
            f.write(f"     Interquartile range: {iqr:.1f}\n")
            
            # Data completeness assessment
            if stat['min_records'] == 0:
                f.write(f"     ⚠️  Warning: Some participants have 0 records\n")
            
            low_data_threshold = stat['median_records'] * 0.1
            if stat['min_records'] < low_data_threshold:
                f.write(f"     ⚠️  Warning: Some participants have very few records (< 10% of median)\n")
            
            f.write("\n")
        
        # Comparison table if multiple files
        if len(summary_stats) > 1:
            f.write("COMPARISON TABLE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'File':<25} {'Participants':<12} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<6} {'Max':<8}\n")
            f.write("-" * 80 + "\n")
            
            for stat in summary_stats:
                filename = stat['filename'].replace('.jsonl', '')[:24]
                f.write(f"{filename:<25} {stat['total_participants']:<12} {stat['mean_records']:<8.1f} {stat['median_records']:<8.1f} {stat['std_records']:<8.1f} {stat['min_records']:<6} {stat['max_records']:<8.0f}\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        # Find files with concerning data quality
        problematic_files = []
        for stat in summary_stats:
            if stat['min_records'] == 0:
                problematic_files.append(f"{stat['filename']} (participants with 0 records)")
            elif stat['min_records'] < stat['median_records'] * 0.1:
                problematic_files.append(f"{stat['filename']} (very low minimum record count)")
        
        if problematic_files:
            f.write("Data quality concerns identified:\n")
            for issue in problematic_files:
                f.write(f"  • {issue}\n")
            f.write("\n")
        
        # General recommendations
        f.write("General recommendations:\n")
        f.write("  • Review threshold plots to set minimum data requirements\n")
        f.write("  • Consider excluding participants below certain thresholds\n")
        f.write("  • Use box plots to identify and investigate outliers\n")
        f.write("  • Check participant_counts.csv files for detailed breakdowns\n\n")
        
        f.write("FILES GENERATED:\n")
        f.write("-" * 40 + "\n")
        for stat in summary_stats:
            clean_name = stat['filename'].replace('.jsonl', '')
            f.write(f"  {clean_name}:\n")
            f.write(f"    • {clean_name}_threshold_stats.csv\n")
            f.write(f"    • {clean_name}_participant_counts.csv\n")
            f.write(f"    • {clean_name}_boxplot.png\n")
            f.write(f"    • {clean_name}_threshold_plot.png\n")
        
        f.write(f"  Summary:\n")
        f.write(f"    • threshold_analysis_summary.txt (this file)\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"  Comprehensive summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Split JSONL files by participant ID based on device mapping')
    parser.add_argument('--input-dir', default='exported_jsonl', 
                       help='Input directory containing JSONL files (default: exported_jsonl)')
    parser.add_argument('--output-dir', default='participant_data',
                       help='Output directory for participant data (default: participant_data)')
    parser.add_argument('--mapping-file', default='resources/pid_deviceid_mapping.csv',
                       help='CSV file containing pid,device_id mapping (default: resources/pid_deviceid_mapping.csv)')
    parser.add_argument('--skip-files', nargs='*', default=[],
                       help='List of JSONL files to skip (default: empty)')
    parser.add_argument('--max-workers', type=int, default=12,
                       help='Maximum number of worker processes (default: number of CPU cores)')
    parser.add_argument('--pids', nargs='*', default=[],
                       help='List of specific participant IDs to process (default: process all)')
    parser.add_argument('--threshold-analysis', action='store_true',
                       help='Run threshold analysis mode instead of splitting files')
    parser.add_argument('--jsonl-files', nargs='*', default=[],
                       help='List of specific JSONL files to analyze (default: all .jsonl files in input directory)')

    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # Check if mapping file exists
    if not os.path.exists(args.mapping_file):
        print(f"Error: Mapping file '{args.mapping_file}' does not exist")
        return 1
    
    print(f"Loading device mapping from {args.mapping_file}...")
    device_mapping = load_device_mapping(args.mapping_file)
    print(f"Loaded mapping for {len(device_mapping)} devices")
    
    # Filter device mapping by specified PIDs if provided
    if args.pids:
        original_device_count = len(device_mapping)
        filtered_mapping = {device_id: pid for device_id, pid in device_mapping.items() if pid in args.pids}
        device_mapping = filtered_mapping
        filtered_device_count = len(device_mapping)
        print(f"Filtered to {filtered_device_count} devices (from {original_device_count}) for specified PIDs: {', '.join(args.pids)}")
        
        # Check if any specified PIDs were not found
        found_pids = set(device_mapping.values())
        missing_pids = set(args.pids) - found_pids
        if missing_pids:
            print(f"Warning: No devices found for PIDs: {', '.join(missing_pids)}")
    
    # Get unique participant IDs
    unique_pids = set(device_mapping.values())
    print(f"Found {len(unique_pids)} unique participants to process")
    
    # Get all JSONL files in input directory
    all_jsonl_files = [f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
    
    # Determine which files to process
    if args.jsonl_files:
        # Use specified files, but check they exist
        specified_files = []
        for filename in args.jsonl_files:
            if not filename.endswith('.jsonl'):
                filename += '.jsonl'
            if filename in all_jsonl_files:
                specified_files.append(filename)
            else:
                print(f"Warning: Specified file '{filename}' not found in {args.input_dir}")
        jsonl_files = [f for f in specified_files if f not in args.skip_files]
    else:
        # Use all files, filter out skipped ones
        jsonl_files = [f for f in all_jsonl_files if f not in args.skip_files]
    
    skipped_files = [f for f in all_jsonl_files if f in args.skip_files]
    
    print(f"Found {len(all_jsonl_files)} JSONL files total")
    if args.jsonl_files:
        print(f"Processing specified files: {', '.join(args.jsonl_files)}")
    if skipped_files:
        print(f"Skipping {len(skipped_files)} files: {', '.join(skipped_files)}")
    print(f"Processing {len(jsonl_files)} JSONL files")
    
    # Check if running threshold analysis mode
    if args.threshold_analysis:
        if not jsonl_files:
            print("Error: No JSONL files to analyze")
            return 1
        perform_threshold_analysis(jsonl_files, args.input_dir, device_mapping, args.output_dir)
        return 0
    
    # Create output directory structure for splitting mode
    print(f"Creating participant directories in {args.output_dir}...")
    create_participant_directories(args.output_dir, unique_pids)
    
    # Prepare arguments for parallel processing
    process_args = []
    for jsonl_file in sorted(jsonl_files):
        input_path = os.path.join(args.input_dir, jsonl_file)
        process_args.append((input_path, args.output_dir, device_mapping, jsonl_file))
    
    # Process files in parallel
    total_records_processed = 0
    unknown_devices_data = {}
    locations_participant_counts = {}
    
    print(f"\nStarting parallel processing with max {args.max_workers or 'CPU count'} workers...")
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        future_to_file = {executor.submit(process_single_file, arg): arg[3] for arg in process_args}
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            try:
                jsonl_file, participant_counts, unknown_device_ids, skipped_count, file_total = future.result()
                total_records_processed += file_total
                unknown_devices_data[jsonl_file] = (unknown_device_ids, skipped_count)
                
                # Store participant counts for locations.jsonl
                if jsonl_file == 'locations.jsonl':
                    locations_participant_counts = participant_counts
                
            except Exception as exc:
                filename = future_to_file[future]
                print(f"Error processing {filename}: {exc}")
    
    print(f"\n✓ Processing complete!")
    print(f"  Total records processed: {total_records_processed}")
    print(f"  Output saved to: {args.output_dir}")
    
    # Print PIDs with less than 20000 records in locations.jsonl (given 5 min of data collection of 60 days, it is about 17,280 records)
    if locations_participant_counts:
        pids_with_few_locations = [pid for pid, count in locations_participant_counts.items() if count < 20000]
        if pids_with_few_locations:
            print(f"\nParticipants with less than 20000 location records:")
            for pid in sorted(pids_with_few_locations):
                record_count = locations_participant_counts[pid]
                print(f"  PID {pid}: {record_count} records")
        else:
            print(f"\nAll participants have 1000 or more location records.")
    
    # Save unknown device report
    save_unknown_device_report(unknown_devices_data, args.output_dir)


if __name__ == "__main__":
    exit(main())
