#!/usr/bin/env python3
"""
Script to split sensor data by sensor type from SS010_output.txt
Each sensor type gets its own file in the splitted_output folder
Updated to work with new format including window headers and category headers
"""

import os
import re
from collections import defaultdict

def split_sensor_data(input_file, output_folder):
    """
    Split sensor data by sensor type and save to separate files
    
    Args:
        input_file (str): Path to the input file
        output_folder (str): Path to the output folder
    """
    
    # Extract PID from input filename (assuming format: {pid}_output.txt)
    input_basename = os.path.basename(input_file)
    if '_output.txt' in input_basename:
        pid = input_basename.replace('_output.txt', '')
    else:
        # Fallback: use filename without extension as PID
        pid = os.path.splitext(input_basename)[0]
    
    # Create output folder with PID subfolder
    pid_output_folder = os.path.join(output_folder, pid)
    os.makedirs(pid_output_folder, exist_ok=True)
    
    # Dictionary to store lines for each sensor type
    sensor_data = defaultdict(list)
    
    # Regular expressions to match different line types
    window_pattern = re.compile(r'^Window \d+')
    day_pattern = re.compile(r'^Day \d{4}-\d{2}-\d{2}')
    time_range_pattern = re.compile(r'^\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2}')
    category_pattern = re.compile(r'^(Environmental Context|Communication Events|Device State|Engagement Signals)$')
    sensor_pattern = re.compile(r'^- (\w+) \| (.+)')
    
    # Variables to track current context
    current_window_info = None
    current_category = None
    current_sensor_type = None
    current_line_parts = []
    
    # Read the input file and group lines by sensor type
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip('\n\r')  # Remove newline but keep other whitespace
                
                if not line.strip():  # Skip empty lines
                    continue
                
                # Check for window header
                if window_pattern.match(line):
                    # Save previous sensor entry if it exists
                    if current_sensor_type and current_line_parts:
                        complete_line = '\n'.join(current_line_parts)
                        sensor_data[current_sensor_type].append(complete_line)
                        current_line_parts = []
                    
                    current_window_info = line
                    current_category = None
                    current_sensor_type = None
                    continue
                
                # Check for day information
                if day_pattern.match(line):
                    if current_window_info:
                        current_window_info += '\n' + line
                    continue
                
                # Check for time range
                if time_range_pattern.match(line):
                    if current_window_info:
                        current_window_info += '\n' + line
                    continue
                
                # Check for category header
                if category_pattern.match(line):
                    current_category = line
                    # Save previous sensor entry if it exists
                    if current_sensor_type and current_line_parts:
                        complete_line = '\n'.join(current_line_parts)
                        sensor_data[current_sensor_type].append(complete_line)
                        current_line_parts = []
                    current_sensor_type = None
                    continue
                
                # Check for sensor description (starts with dash)
                sensor_match = sensor_pattern.match(line)
                if sensor_match:
                    # Save previous sensor entry if it exists
                    if current_sensor_type and current_line_parts:
                        complete_line = '\n'.join(current_line_parts)
                        sensor_data[current_sensor_type].append(complete_line)
                        current_line_parts = []
                    
                    # Parse the new sensor entry
                    sensor_type = sensor_match.group(1)
                    content = sensor_match.group(2)
                    
                    # Create the formatted line with window and category context
                    formatted_line_parts = []
                    if current_window_info:
                        formatted_line_parts.append(current_window_info)
                    if current_category:
                        formatted_line_parts.append(current_category)
                    
                    # Add the sensor description in the old format for compatibility
                    formatted_line = f"Window Context | {sensor_type} | {content}"
                    if formatted_line_parts:
                        formatted_line = '\n'.join(formatted_line_parts) + '\n' + formatted_line
                    
                    current_sensor_type = sensor_type
                    current_line_parts = [formatted_line]
                else:
                    # This is a continuation line for the current sensor
                    if current_sensor_type and current_line_parts:
                        current_line_parts.append(line)
                    else:
                        print(f"Warning: Line {line_num} appears to be a continuation but no current sensor entry: {line}")
            
            # Don't forget to save the last entry
            if current_sensor_type and current_line_parts:
                complete_line = '\n'.join(current_line_parts)
                sensor_data[current_sensor_type].append(complete_line)
                
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Write each sensor's data to a separate file
    for sensor_type, lines in sensor_data.items():
        # Create a safe filename by replacing spaces and special characters
        safe_filename = sensor_type.replace(' ', '_').replace('/', '_')
        output_file = os.path.join(pid_output_folder, f"{safe_filename}.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')
            
            print(f"Created {output_file} with {len(lines)} entries")
        except Exception as e:
            print(f"Error writing to {output_file}: {e}")

def main():
    input_file = "description/SS001_output.txt"
    output_folder = "description_split"
    
    # Extract PID for display purposes
    input_basename = os.path.basename(input_file)
    if '_output.txt' in input_basename:
        pid = input_basename.replace('_output.txt', '')
    else:
        pid = os.path.splitext(input_basename)[0]
    
    print(f"Splitting sensor data from {input_file}")
    print(f"Output will be saved to {output_folder}/{pid}/")
    print("-" * 50)
    
    split_sensor_data(input_file, output_folder)
    
    print("-" * 50)
    print("Splitting complete!")

if __name__ == "__main__":
    main()
