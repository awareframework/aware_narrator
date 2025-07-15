import yaml
import pytz
import os
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import googlemaps
from googlemaps import exceptions as gexceptions
import re

# Load configuration from yaml file
CONFIG_FILE = "./config.yaml"
with open(CONFIG_FILE, "r", encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)

# Assign variables from yaml
pid_to_deviceid_map = CONFIG["pid_to_deviceid_map"]

# Load DEVICE_IDs from CSV file based on P_ID
def load_device_ids_from_csv(csv_file_path, participant_id):
    """
    Load device IDs from CSV file for a given participant ID.
    
    Args:
        csv_file_path (str): Path to the CSV file containing pid to device_id mapping
        participant_id (str): Participant ID to lookup
        
    Returns:
        list: List of device IDs for the participant
    """
    try:
        df = pd.read_csv(csv_file_path)
        # Find the row for the given participant ID
        participant_row = df[df['pid'] == participant_id]
        
        if participant_row.empty:
            print(f"Warning: Participant ID '{participant_id}' not found in {csv_file_path}")
            return []
        
        # Get the device_id value and split by semicolon
        device_id_str = participant_row['device_id'].iloc[0]
        device_ids = [device_id.strip() for device_id in device_id_str.split(';')]
        
        print(f"Loaded {len(device_ids)} device IDs for participant {participant_id}: {device_ids}")
        return device_ids
        
    except FileNotFoundError:
        print(f"Error: CSV file {csv_file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return []
    
# Get participant IDs from config
P_IDs = CONFIG["P_IDs"]

# Global configuration variables
START_TIME = CONFIG["START_TIME"]
END_TIME = CONFIG["END_TIME"]
timezone = pytz.timezone(CONFIG["timezone"])
sensor_integration_time_window = CONFIG["sensor_integration_time_window"]
gate_time_window = CONFIG["gate_time_window"]
sensors = CONFIG["sensors"]
GOOGLE_MAP_KEY = CONFIG["GOOGLE_MAP_KEY"]
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
application_name_list = {}

def process_participant(P_ID):
    """
    Process sensor data for a single participant.
    
    Args:
        P_ID (str): Participant ID to process
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing participant: {P_ID}")
    print(f"{'='*60}")
    
    # Load device IDs for this participant
    DEVICE_IDs = load_device_ids_from_csv(pid_to_deviceid_map, P_ID)
    if not DEVICE_IDs:
        print(f"Warning: No device IDs found for participant {P_ID}. Skipping.")
        return False
    
    # Set up participant-specific paths
    input_directory = CONFIG["input_directory"].format(P_ID=P_ID)
    output_file = CONFIG["output_file"].format(P_ID=P_ID)
    daily_output_dir = CONFIG["daily_output_dir"].format(P_ID=P_ID)
    reverse_geocoding_output_dir = CONFIG["reverse_geocoding_output_dir"].format(P_ID=P_ID)
    
    # Ensure output directories exist
    os.makedirs(daily_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(reverse_geocoding_output_dir, exist_ok=True)
    
    jsonl_files = [(f"{sensor}.jsonl", sensor) for sensor in sensors]
    
    # Initialize participant-specific variables
    sensor_narratives = {}
    
    # Convert start and end times to timestamps
    START_TIMESTAMP = convert_timestring_to_timestamp(START_TIME, CONFIG["timezone"])
    END_TIMESTAMP = convert_timestring_to_timestamp(END_TIME, CONFIG["timezone"])

    # Load session data for accurate active time calculation
    session_file_path = CONFIG.get("session_data_file", "").format(P_ID=P_ID)
    sessions = load_session_data(session_file_path)
    
    # If config session file not found, try the default location
    if not sessions:
        default_session_file = f"step1_data/{P_ID}/sessions.jsonl"
        print(f"Trying default session file location: {default_session_file}")
        sessions = load_session_data(default_session_file)
    
    print(f"Loaded {len(sessions)} session records")

    print("Keep data within time range: ", START_TIME, "to", END_TIME)
    
    #store location data in a list
    location_data = []
    
    # Store WiFi sensor data separately for combined processing
    wifi_sensor_data = {}

    for jsonl_file, sensor_name in jsonl_files:      
        sensor_data = get_sensor_data(sensor_name, START_TIMESTAMP, END_TIMESTAMP, input_directory)
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
            print(f"Collected {sensor_name} data: {len(sensor_data)} records")
            continue
        
        # Initialize sensor-specific narrative list
        if sensor_name not in sensor_narratives:
            sensor_narratives[sensor_name] = []

        # Generate integrated descriptions for each sensor
        narratives = generate_integrated_description(sensor_data, sensor_name, START_TIMESTAMP, END_TIMESTAMP, sessions)
        if narratives:
            sensor_narratives[sensor_name].extend(narratives)

    # Process WiFi sensors together if we have any type
    if wifi_sensor_data:
        sensor_wifi_data = wifi_sensor_data.get("sensor_wifi", [])
        wifi_data = wifi_sensor_data.get("wifi", [])
        
        print(f"Processing combined WiFi analysis:")
        print(f"  - sensor_wifi: {len(sensor_wifi_data)} records")
        print(f"  - wifi: {len(wifi_data)} records")
        
        # Generate combined WiFi narratives
        combined_wifi_narratives = generate_wifi_combined_description(
            sensor_wifi_data, wifi_data, START_TIMESTAMP, END_TIMESTAMP, sessions
        )
        
        if combined_wifi_narratives:
            # Store combined WiFi narratives under a unified key
            sensor_narratives["wifi_combined"] = combined_wifi_narratives
            print(f"Generated {len(combined_wifi_narratives)} combined WiFi narratives")

    #summary len of each sensor narrative list
    for sensor_name, narrative_list in sensor_narratives.items():
        print(f"Sensor: {sensor_name}, Number of integrated descriptions: {len(narrative_list)}")

    # Generate distance matrix from location_data
    print("Locations: Generate distance matrix...")
    
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
        print("No valid location points found! Skipping location processing.")
        # Initialize empty variables for location processing
        indices = []
        coordinates = np.array([])
        datetimes = []
        speeds = []
        cluster = []
        cluster_labels = np.array([])
        daily_clusters = {}
    else:
        # Extract data for clustering (only coordinates)
        indices = [item[0] for item in points_with_indices]
        coordinates = np.array([[item[1], item[2]] for item in points_with_indices])  # Only lat, lon
        datetimes = [item[3] for item in points_with_indices]
        speeds = [item[4] for item in points_with_indices]
    
    # Only proceed with clustering if we have coordinates to process
    if len(coordinates) > 0:
        print(f"Data size is {len(coordinates)}")
        # Determine if we should use daily clustering or all data
        start_dt = datetime.strptime(START_TIME, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(END_TIME, '%Y-%m-%d %H:%M:%S')
        time_span_hours = (end_dt - start_dt).total_seconds() / 3600
        
        print(f"Time span: {time_span_hours:.1f} hours")
        
        if time_span_hours < 48:
            print("Time span is less than 48 hours - using all location data for clustering")
            use_daily_clustering = False
        else:
            print("Time span is 48 hours or more - using daily clustering based on night_time_end")
            use_daily_clustering = True
        
        # Perform DBSCAN clustering using haversine distance
        print("Locations: Performing DBSCAN clustering")
        
        try:
            if len(coordinates) > 0: # Ensure there are points to process
                # Perform DBSCAN clustering using shared function
                cluster_labels, daily_clusters = perform_dbscan_clustering(
                    coordinates, datetimes, eps, min_samples, use_daily_clustering, 
                    start_dt, end_dt, night_time_end
                )
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # Exclude noise
                print(f"DBSCAN found {n_clusters} clusters (excluding noise)")
                print(f"Noise points: {list(cluster_labels).count(-1)} out of {len(cluster_labels)}")
                
                # Process clustering results using shared function
                (cluster, clustered_coordinates, clustered_labels, clustered_datetimes, 
                 clustered_indices, clustered_speeds, home_group_center) = process_clustering_results(
                    coordinates, cluster_labels, datetimes, indices, speeds,
                    use_daily_clustering, daily_clusters, night_time_start, night_time_end
                )
                
                # Update variables to use clustered data for the rest of the algorithm
                coordinates = clustered_coordinates
                cluster_labels = clustered_labels
                datetimes = clustered_datetimes
                indices = clustered_indices
                speeds = clustered_speeds

        except NameError as e:
            print(f"Error: Variable not initialized - {e}")
            print("Skipping location clustering due to error.")
        except ValueError as e:
            print(f"Error: {e}")
            print("Skipping location clustering due to error.")
        except Exception as e:
            print(f"Unexpected error during clustering: {e}")
            print("Skipping location clustering due to error.")
    else:
        print("No location data available for clustering")

    # If a Google Maps API key is provided, perform reverse geocoding for all places
    reverse_geocoding_performed = False
    if GOOGLE_MAP_KEY and cluster:
        gmaps = googlemaps.Client(key=GOOGLE_MAP_KEY)
        # a list to store all reverse geocoding results
        reverse_geocode_results = []
        try:
            for idx, cluster_data in enumerate(cluster):
                # Extract cluster data (must have exactly 6 elements)
                if len(cluster_data) == 6:
                    cluster_id, center_lat, center_lon, num_points, place, distance_from_home = cluster_data
                else:
                    print(f"Warning: Cluster data at index {idx} has {len(cluster_data)} elements, expected 6. Skipping.")
                    continue
                # Perform reverse geocoding for all places (both home and unknown)
                reverse_geocode_data = None
                try:
                    reverse_geocode_data = gmaps.reverse_geocode((center_lat, center_lon), enable_address_descriptor=True)
                except (gexceptions.ApiError, gexceptions.HTTPError, gexceptions.Timeout, gexceptions.TransportError) as e:
                    print(f"Error during reverse geocoding for ({center_lat}, {center_lon}): {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error during reverse geocoding for ({center_lat}, {center_lon}): {e}")
                    continue

                if not reverse_geocode_data or reverse_geocode_data.get("status") != "OK":
                    print(f"No valid address returned for ({center_lat}, {center_lon}).")
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
            print(f"Error in Google Maps API request: {e}")
        
        # Save reverse geocoding results to the configured directory
        # Parse the full END_TIME
        start_date = datetime.strptime(START_TIME, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')
        end_date = datetime.strptime(END_TIME, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')
        filename = os.path.join(reverse_geocoding_output_dir, f"{P_ID}_{start_date}_{end_date}_reverse_geocode_results.json")
        
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
        mapping_filename = os.path.join(reverse_geocoding_output_dir, f"{P_ID}_{start_date}_{end_date}_cluster_mapping.json")
        with open(mapping_filename, "w", encoding='utf-8') as file:
            json.dump(cluster_mapping, file, indent=2, ensure_ascii=False, default=str)
        
        print(f"Saved reverse geocoding results to: {filename}")
        print(f"Saved cluster mapping to: {mapping_filename}")
        reverse_geocoding_performed = True

    # Display the updated cluster information only if reverse geocoding was performed successfully
    if len(coordinates) > 0 and cluster and reverse_geocoding_performed:
        print("Cluster Centers (Updated):")
        for cluster_data in cluster:
            if len(cluster_data) == 6:  # Must have exactly 6 elements
                cluster_id, center_lat, center_lon, num_points, place, distance_from_home = cluster_data
                print(f"Cluster {cluster_id}: Center Lat = {center_lat:.6f}, Center Lon = {center_lon:.6f}, N = {num_points}, Place = {place}, Distance = {distance_from_home:.1f}m")

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
        
        # Initialize locations narrative list
        if "locations" not in sensor_narratives:
            sensor_narratives["locations"] = []
        
        # Generate integrated location descriptions
        location_narratives = describe_locations_integrated(
            reconstructed_points, cluster_labels, cluster, 
            START_TIMESTAMP, END_TIMESTAMP, sessions
        )
        
        if location_narratives:
            sensor_narratives["locations"].extend(location_narratives)

    all_narratives = []
    
    #load all narrative lists
    for sensor_name, sensor_narrative_list in sensor_narratives.items():
        for item in sensor_narrative_list:
            all_narratives.append(item)

    # Write output description to a text file
    print("Output description into text file...")
    # Remove duplicates first, then sort by time window and sensor order
    unique_narratives = list(set(all_narratives))
    all_narratives = sort_narratives_by_time_window_and_sensor_order(unique_narratives)
    text = "\n".join([str(item[1]) for item in all_narratives])

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)
        
    # Split description by days and save each day's content to separate files
    output_files = split_description_by_days(output_file, daily_output_dir)
    print(f"Split description into {len(output_files)} daily files in {daily_output_dir}")
    
    print(f"Completed processing for participant {P_ID}")
    return True

def convert_timestring_to_timestamp(timestring, timezone_str="Australia/Melbourne"):
    """
    Convert a timestring to a timestamp float based on the provided timezone.
    """
    try:
        tz = pytz.timezone(timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Unknown timezone '{timezone_str}'. Falling back to UTC.")
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
        print(f"Unknown timezone '{timezone_str}'. Falling back to UTC.")
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

def get_sensor_data(sensor_name, start_timestamp, end_timestamp, input_dir):
    """
    Load sensor data from JSONL file and filter by timestamp range.
    
    Args:
        sensor_name (str): Name of the sensor (e.g., 'battery', 'applications_foreground')
        start_timestamp (float): Start timestamp in milliseconds
        end_timestamp (float): End timestamp in milliseconds
        input_dir (str): Input directory path
    
    Returns:
        list: List of sensor records within the timestamp range
    """
    # Construct the file path
    file_path = os.path.join(input_dir, f"{sensor_name}.jsonl")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: Sensor file {file_path} not found")
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
                    print(f"Warning: Failed to parse JSON line in {file_path}: {e}")
                    continue
    
        # Sort records by _id then by timestamp
        filtered_records = sorted(filtered_records, key=lambda x: (x.get('_id', ''), x.get('timestamp', 0)))
                        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    print(f"Loaded {len(filtered_records)} records from {sensor_name} sensor")
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
        list: List of formatted narrative tuples (datetime, description)
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
        
        description_parts = [f"{datetime_str} | applications | App Usage"]
        
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
        narratives.append((datetime_str, description))
    
    return narratives

def describe_applications_foreground_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
    """Generates a description for foreground applications detected."""
    print("Generating integrated description for applications_foreground")
    
    if not sensor_data:
        print("No application data available, skipping applications_foreground integration")
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
    
    print(f"Generated {len(narratives)} app usage narratives for {len(app_summaries)} time windows (window size: {sensor_integration_time_window} minutes)")
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
        list: List of formatted location narrative tuples (datetime, description)
    
    Note: Speed data is not used for movement analysis due to unreliable sampling frequency.
    """
    print("Generating integrated description for locations")
    
    if len(all_points) == 0 or len(cluster_labels) == 0:
        print("No location data available, skipping location integration")
        return []
    
    # Build a richer cluster_id_to_info map
    cluster_id_to_info = {}
    print("DEBUG: Processing cluster data:")
    for cid, lat, lon, n_pts, raw_place, dist in cluster:
        print(f"  Cluster {cid}: {raw_place[:100]}...")
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
        description_parts = [f"{datetime_str} | locations | Location Analysis"]
        
        # Show basic statistics
        total_locations = len(location_visits)
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
        
        if not sorted_locations:
            description_parts.append(f"    - No significant locations detected (insufficient data points)")
            return '\n'.join(description_parts)
        
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

            # 2) now dump your saved human‐sentence and address
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
    
    print(f"Generated {len(narratives)} location narratives (window size: {sensor_integration_time_window} minutes)")
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
        list: List of formatted narrative tuples (datetime, description)
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
            narratives.append((datetime_str, description))
        
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
        list: List of formatted bluetooth narrative tuples (datetime, description)
    """
    print("Generating integrated description for bluetooth")
    
    if sensor_name != "bluetooth" or not sensor_data:
        print("No bluetooth data available, skipping bluetooth integration")
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
        description_parts = [f"{datetime_str} | bluetooth | Bluetooth Devices Detected"]
        
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
    
    print(f"Generated {len(narratives)} bluetooth narratives with averaged {gate_time_window}-minute gate statistics (window size: {sensor_integration_time_window} minutes, filtering out default/manufacturer device names and grouping by device name)")
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
        list: List of formatted battery narrative tuples (datetime, description)
    """
    print("Generating integrated description for battery")
    
    if sensor_name != "battery" or not sensor_data:
        print("No battery data available, skipping battery integration")
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
        description_parts = [f"{datetime_str} | battery | Status changes"]
        
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
    
    print(f"Generated {len(narratives)} battery narratives (window size: {sensor_integration_time_window} minutes)")
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
        list: List of formatted notifications narrative tuples (datetime, description)
    """
    print("Generating integrated description for applications_notifications")
    
    if sensor_name != "applications_notifications" or not sensor_data:
        print("No applications_notifications data available, skipping notifications integration")
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
        description_parts = [f"{datetime_str} | notifications | Notification Activity"]
        
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
    
    print(f"Generated {len(narratives)} applications_notifications narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives

def generate_integrated_description(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
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
        return describe_screentext_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions)
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
        print(f"Sensor {sensor_name} not supported")
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

    
def load_session_data(session_file_path):
    """
    Load session data from JSONL file.
    
    Args:
        session_file_path (str): Path to the sessions.jsonl file
        
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
        print(f"Warning: Session file {session_file_path} not found. Using estimated active time.")
        return []
    except Exception as e:
        print(f"Error loading session data: {e}. Using estimated active time.")
        return []
    
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
        list: List of formatted keyboard narrative tuples (datetime, description)
    """
    print("Generating integrated description for keyboard")
    
    if sensor_name != "keyboard" or not sensor_data:
        print("No keyboard data available, skipping keyboard integration")
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
            
            description_parts = [f"{datetime_str} | keyboard | Typing Activity"]
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
    
    print(f"Generated {len(narratives)} keyboard narratives with typing detection (window size: {sensor_integration_time_window} minutes)")
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
        all_narratives (list): List of (datetime_str, description) tuples
        
    Returns:
        list: Sorted list of (datetime_str, description) tuples with category headers
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
    
    def get_sensor_type(description):
        """Extract sensor type from description."""
        if ' | locations | ' in description:
            return 'locations'
        elif ' | wifi | ' in description:
            return 'wifi'
        elif ' | bluetooth | ' in description:
            return 'bluetooth'
        elif ' | notifications | ' in description:
            return 'notifications'
        elif ' | calls | ' in description:
            return 'calls'
        elif ' | messages | ' in description:
            return 'messages'
        elif ' | battery | ' in description:
            return 'battery'
        elif ' | installations | ' in description:
            return 'installations'
        elif ' | screen | ' in description:
            return 'screen'
        elif ' | applications | ' in description:
            return 'applications'
        elif ' | keyboard | ' in description:
            return 'keyboard'
        elif ' | screentext | ' in description:
            return 'screentext'
        else:
            # For other sensors, assign a default high priority to appear after main sensors
            return 'other'
    
    def get_sensor_category(description):
        """Get category for sensor type."""
        sensor_type = get_sensor_type(description)
        return sensor_categories.get(sensor_type, 'other')
    
    def get_category_priority(description):
        """Get category priority for sorting."""
        category = get_sensor_category(description)
        try:
            return category_order.index(category)
        except ValueError:
            return len(category_order)  # Put unknown categories at the end
    
    # Group narratives by time window (datetime string)
    time_window_groups = defaultdict(list)
    
    for datetime_str, description in all_narratives:
        time_window_groups[datetime_str].append((datetime_str, description))
    
    # Sort within each time window by category and create formatted output
    sorted_narratives = []
    window_number = 1
    
    for time_window in sorted(time_window_groups.keys()):
        window_narratives = time_window_groups[time_window]
        
        # Sort this window's narratives by category priority
        window_narratives.sort(key=lambda x: get_category_priority(x[1]))
        
        # Add newline before time window (except for the first one)
        if sorted_narratives:
            sorted_narratives.append((time_window, ""))
        
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
        sorted_narratives.append((time_window, time_range_header))
        
        window_number += 1
        
        # Group by category and add headers
        current_category = None
        for datetime_str, description in window_narratives:
            category = get_sensor_category(description)
            
            # Add category header if this is a new category
            if category != current_category and category in category_display_names:
                current_category = category
                # Add newline before category header
                sorted_narratives.append((time_window, ""))
                category_header = category_display_names[category]
                sorted_narratives.append((time_window, category_header))
            
            # Remove timestamp prefix from description and add dash prefix
            if ' | ' in description:
                # Extract the part after the timestamp
                description_parts = description.split(' | ', 2)
                if len(description_parts) >= 3:
                    sensor_type = description_parts[1]
                    content = description_parts[2]
                    formatted_description = f"- {sensor_type} | {content}"
                else:
                    formatted_description = f"- {description}"
            else:
                formatted_description = f"- {description}"
            
            sorted_narratives.append((time_window, formatted_description))
    
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
        list: List of formatted screen narrative tuples (datetime, description)
    """
    print("Generating integrated description for screen")
    
    if sensor_name != "screen" or not sensor_data:
        print("No screen data available, skipping screen integration")
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
        description_parts = [f"{datetime_str} | screen | Screen Status Analysis"]
        
        # Show event breakdown (all event types including locked/unlocked)
        if status_counts:
            event_breakdown = []
            for status, count in status_counts.items():
                event_breakdown.append(f"{count} {status}")
            description_parts.append(f"    - Event breakdown: {', '.join(event_breakdown)}")
        
        # Screen session information (exclude carryover sessions from activation count)
        new_activations = len([s for s in screen_sessions if not s.get('is_carryover', False)])
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
    
    print(f"Generated {len(narratives)} screen narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives

"""
IMPROVED MULTI-DAY HOME DETECTION APPROACH

This implementation addresses the limitations of the original approach by:

1. **Daily Home Candidate Identification**: For each day, identifies the most frequent nighttime cluster
2. **Proximity-Based Merging**: Merges daily candidates that are within merge_distance_threshold of each other
3. **Robust Home Detection**: Uses the merged location as the true home
4. **Daily Analysis**: Provides insights into which days the user was actually home

Example scenario:
- Day 1: User's home GPS reads (lat: -37.8136, lon: 144.9631) - Cluster 0
- Day 2: User's home GPS reads (lat: -37.8138, lon: 144.9629) - Cluster 3  
- Day 3: User stays at friend's house (lat: -37.8200, lon: 144.9500) - Cluster 7
- Day 4: User's home GPS reads (lat: -37.8135, lon: 144.9633) - Cluster 10

Results:
- Day 1, 2, 4 candidates are within merge_distance_threshold → merged as "home"
- Day 3 candidate is >500m away → identified as "away from home"
- Final home location: weighted average of Days 1, 2, 4 coordinates

Benefits:
- Handles GPS drift and daily clustering variations
- Detects when user wasn't home on specific days
- Provides daily home presence analysis
- More robust than single-pass nighttime frequency analysis
"""

def identify_and_merge_daily_home_candidates(daily_clusters, coordinates, cluster_labels, datetimes, night_time_start, night_time_end, merge_distance_threshold=50):
    """
    Identify home cluster candidates for each day and merge nearby candidates.
    
    Args:
        daily_clusters: Dictionary containing daily clustering results
        coordinates: All clustered coordinates
        cluster_labels: All cluster labels
        datetimes: All datetime strings
        night_time_start: Start hour of nighttime (e.g., 22 for 10 PM)
        night_time_end: End hour of nighttime (e.g., 6 for 6 AM)
        merge_distance_threshold: Distance threshold in meters to merge candidates
        
    Returns:
        tuple: (home_cluster_index, daily_home_analysis, merged_home_center, clusters_to_merge)
    """
    daily_home_candidates = {}
    days_without_nighttime = []
    
    print("\n=== Daily Home Candidate Analysis ===")
    
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
            
            print(f"Day {day_id}: Home candidate cluster {day_home_candidate} at ({candidate_center[0]:.6f}, {candidate_center[1]:.6f})")
            print(f"  - {len(day_night_labels)} nighttime points, {np.sum(day_night_labels == day_home_candidate)} in candidate cluster")
        else:
            print(f"Day {day_id}: No nighttime points found - will check for home proximity after finding merged home")
            days_without_nighttime.append(day_id)
    
    if not daily_home_candidates:
        raise ValueError("No daily home candidates found")
    
    # Step 2: Calculate distances between daily home candidates
    print("\n=== Merging Nearby Home Candidates ===")
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
                print(f"Merging Day {candidate_days[j]} candidate with Day {candidate_days[i]} (distance: {distance:.1f}m)")
        
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
    
    print(f"\nPrimary home group: {len(primary_group['days'])} days")
    print(f"Days with home activity: {sorted(primary_group['days'])}")
    print(f"Merged home center: ({merged_home_center[0]:.6f}, {merged_home_center[1]:.6f})")
    
    # Step 4: Collect all cluster IDs that should be merged
    clusters_to_merge = set()
    for candidate in primary_group['candidates']:
        clusters_to_merge.add(candidate['cluster_id'])
    
    print(f"Clusters to merge: {sorted(list(clusters_to_merge))}")
    
    # Step 5: Check days without nighttime points for clusters that could be merged to home
    print("\n=== Checking Days Without Nighttime Points ===")
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
                print(f"Day {day_id}: Cluster {cluster_id} at ({cluster_center[0]:.6f}, {cluster_center[1]:.6f}) is {distance_to_home:.1f}m from home - merging")
                clusters_to_merge.add(cluster_id)
            else:
                print(f"Day {day_id}: Cluster {cluster_id} at ({cluster_center[0]:.6f}, {cluster_center[1]:.6f}) is {distance_to_home:.1f}m from home - not merging")
    
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
    print("\n=== Daily Home Presence Analysis ===")
    for day_id in sorted(daily_home_analysis.keys()):
        analysis = daily_home_analysis[day_id]
        if analysis['was_home']:
            if analysis.get('merged_to_home', False):
                print(f"Day {day_id}: At home (merged to home cluster, {analysis['distance_from_home']:.1f}m from home center)")
            else:
                print(f"Day {day_id}: At home ({analysis['nighttime_points_at_home']}/{analysis['nighttime_points']} nighttime points, {analysis['home_percentage']:.1f}%)")
        elif analysis.get('alternative_location', False):
            print(f"Day {day_id}: Away from home ({analysis['distance_from_home']:.1f}m from home, {analysis['nighttime_points']} nighttime points)")
        else:
            print(f"Day {day_id}: No clear nighttime location")
    
    return home_cluster_index, daily_home_analysis, merged_home_center, clusters_to_merge

def process_clustering_results(coordinates, cluster_labels, datetimes, indices, speeds, 
                               use_daily_clustering, daily_clusters, night_time_start, night_time_end):
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
    
    print(f"After filtering noise: {len(clustered_labels)} clustered points")
    
    # Identify home cluster using appropriate method
    if use_daily_clustering and daily_clusters:
        # Use daily home candidate merging approach
        home_cluster_index, daily_home_analysis, merged_home_center, clusters_to_merge = identify_and_merge_daily_home_candidates(
            daily_clusters, clustered_coordinates, clustered_labels, clustered_datetimes,
            night_time_start, night_time_end, merge_distance_threshold
        )
        
        # Store daily analysis for potential use in narratives
        globals()['daily_home_analysis'] = daily_home_analysis
        
        # Note: Merging and renumbering will be done together in the next step for efficiency
            
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
        
        print(f"Found {len(night_labels)} nighttime location points in valid clusters")
        home_cluster_index = np.bincount(night_labels).argmax() # Identify home cluster
        print(f"Home cluster identified as cluster {home_cluster_index}")
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
        print(f"Using merged home center: ({merged_home_center[0]:.6f}, {merged_home_center[1]:.6f})")
    else:
        # For single-day clustering, we'll set this when we find the home cluster
        home_group_center = None
        print("Will determine home center from home cluster centroid")
    
    # COMBINED MERGING AND RENUMBERING IN SINGLE OPERATION
    # This efficiently merges home candidates and assigns final consecutive IDs in one pass
    print("\n=== Combined Merging and Renumbering Operation ===")
    unique_cluster_ids = sorted(set(clustered_labels))
    
    # Create direct mapping from original cluster IDs to final consecutive IDs
    old_to_new_mapping = {}
    
    if use_daily_clustering and daily_clusters and 'clusters_to_merge' in locals():
        # Multi-day clustering: merge home candidates to ID 0, others get consecutive IDs
        print(f"Merging home candidates {sorted(list(clusters_to_merge))} → ID 0")
        
        # All home candidate clusters map to ID 0
        for cluster_id in clusters_to_merge:
            old_to_new_mapping[cluster_id] = 0
            print(f"  - Home candidate cluster {cluster_id} → 0")
        
        # Assign consecutive IDs to non-home clusters
        new_cluster_id = 1
        for old_cluster_id in unique_cluster_ids:
            if old_cluster_id not in clusters_to_merge:
                old_to_new_mapping[old_cluster_id] = new_cluster_id
                print(f"  - Unknown cluster {old_cluster_id} → {new_cluster_id}")
                new_cluster_id += 1
        
        home_cluster_index = 0  # Merged home cluster is always ID 0
        
    else:
        # Single-day clustering: home cluster to ID 0, others get consecutive IDs
        print(f"Single home cluster {home_cluster_index} → ID 0")
        
        # Home cluster gets ID 0
        old_to_new_mapping[home_cluster_index] = 0
        print(f"  - Home cluster {home_cluster_index} → 0")
        
        # Assign consecutive IDs to other clusters
        new_cluster_id = 1
        for old_cluster_id in unique_cluster_ids:
            if old_cluster_id != home_cluster_index:
                old_to_new_mapping[old_cluster_id] = new_cluster_id
                print(f"  - Unknown cluster {old_cluster_id} → {new_cluster_id}")
                new_cluster_id += 1
        
        home_cluster_index = 0  # Home cluster is always ID 0
    
    # Apply the mapping in a single pass through all points
    print(f"Applying final cluster assignments to {len(clustered_labels)} points...")
    final_labels = clustered_labels.copy()
    merged_point_count = 0
    
    for i, old_id in enumerate(clustered_labels):
        final_labels[i] = old_to_new_mapping[old_id]
        if old_to_new_mapping[old_id] == 0:  # Count points in merged home cluster
            merged_point_count += 1
    
    # Update cluster labels with final assignments
    clustered_labels = final_labels
    
    final_cluster_count = len(set(clustered_labels))
    print(f"✓ Final result: {final_cluster_count} clusters (Home: {merged_point_count} points, Unknown: {len(clustered_labels) - merged_point_count} points)")
    
    # Set merged_home_center for distance calculations
    if use_daily_clustering and daily_clusters and 'clusters_to_merge' in locals() and len(clusters_to_merge) > 1:
        merged_home_center = merged_home_center  # Use the calculated merged center
    else:
        merged_home_center = None  # Will use cluster center

    """
    CLUSTER DISTANCE CALCULATION FIX WITH UNIQUE UNKNOWN PLACE LABELING:
    
    Issue: All non-home clusters were labeled as "unknown", making them indistinguishable.
    
    Fix:
    1. Set home_group_center properly before cluster processing
    2. For home cluster: Use "home" label and store merged/cluster center coordinates
    3. For other clusters: Use unique labels "unknown1", "unknown2", etc. based on their new IDs
    4. Calculate ALL distances using the same home_group_center reference point
    
    Result:
    - Home cluster: labeled as "home", distance = 0m
    - Other clusters: labeled as "unknown1", "unknown2", etc. with accurate distances
    - Each unknown place gets a unique, meaningful identifier
    """
    
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
                print(f"Home cluster {cluster_id}: Using merged center ({actual_center[0]:.6f}, {actual_center[1]:.6f}) vs cluster center ({cluster_center[0]:.6f}, {cluster_center[1]:.6f})")
            else:
                actual_center = cluster_center
                home_group_center = cluster_center  # Set for single-day clustering
                print(f"Home cluster {cluster_id}: Using cluster center ({actual_center[0]:.6f}, {actual_center[1]:.6f})")
            
            cluster.append((int(cluster_id), float(actual_center[0]), float(actual_center[1]), int(len(cluster_points)), place))
        else:
            # Generate unique unknown place labels: unknown1, unknown2, unknown3, etc.
            unknown_place_counter += 1
            place = f"unknown{unknown_place_counter}"
            print(f"Unknown cluster {cluster_id}: Labeled as '{place}' at ({cluster_center[0]:.6f}, {cluster_center[1]:.6f})")
            
            # For non-home clusters, use the calculated cluster center
            cluster.append((int(cluster_id), float(cluster_center[0]), float(cluster_center[1]), int(len(cluster_points)), place))

    # Calculate distances of each cluster from home
    if home_group_center is None:
        raise ValueError("home_group_center not properly set - this should not happen")
    
    print(f"Calculating distances from home center: ({home_group_center[0]:.6f}, {home_group_center[1]:.6f})")
    
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
                print(f"VALIDATION: Home cluster {cluster_id} distance from home: {cluster_distance_from_home:.1f}m (should be ~0 for merged center)")
                break

    # Print cluster details
    if len(clustered_coordinates) > 0:
        print("Cluster Centers:")
        for cluster_data in cluster:
            if len(cluster_data) == 6:  # Must have exactly 6 elements
                cluster_id, center_lat, center_lon, num_points, place, distance_from_home = cluster_data
                print(f"Cluster {cluster_id}: Center Lat = {center_lat:.6f}, Center Lon = {center_lon:.6f}, N = {num_points}, Place = {place}, Distance = {distance_from_home:.1f}m")

    return (cluster, clustered_coordinates, clustered_labels, clustered_datetimes, 
            clustered_indices, clustered_speeds, home_group_center)

def perform_dbscan_clustering(coordinates, datetimes, eps, min_samples, use_daily_clustering, 
                               start_dt, end_dt, night_time_end):
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
        
    Returns:
        tuple: (cluster_labels, daily_clusters)
    """
    
    if use_daily_clustering:
        print("Performing multi-day clustering...")
        
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
            print(f"Last day has {daily_periods[-1]['duration_hours']:.1f} hours - combining with previous day")
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
                print(f"Day {day_counter}: {period_start.strftime('%Y-%m-%d %H:%M:%S')} to {period_end.strftime('%Y-%m-%d %H:%M:%S')} ({duration_hours:.1f}h) - {len(daily_coordinates)} points")
                
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
                print(f"Day {day_counter}: {period_start.strftime('%Y-%m-%d %H:%M:%S')} to {period_end.strftime('%Y-%m-%d %H:%M:%S')} ({duration_hours:.1f}h) - {len(daily_coordinates)} points (insufficient for clustering)")
            
            day_counter += 1
        
        # Convert to numpy array and ensure it matches the length of coordinates
        while len(all_cluster_labels) < len(coordinates):
            all_cluster_labels.append(-1)
        cluster_labels = np.array(all_cluster_labels[:len(coordinates)])
        
    else:
        print("Performing single-day clustering...")
        
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
        #print(f"Will filter out locations: {', '.join(locations_to_filter)}") #Uncomment to see the locations that are being filtered out
        
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

    #Uncomment to see the locations that are being filtered out
            # if stats['data_points'] < location_minimum_data_points:
            #     print(f"Filtering out location '{place_name}' with only {stats['data_points']} data points (threshold: {location_minimum_data_points})")
            # elif stats['total_time_seconds'] < (location_minimum_stay_minutes * 60):
            #     print(f"Filtering out location '{place_name}' with only {stats['total_time_seconds']:.1f}s stay time (minimum: {location_minimum_stay_minutes} minutes)")
    
    #Uncomment to see the number of locations that are being filtered out
    # if filtered_count > 0:
    #     print(f"Filtered out {filtered_count} locations with insufficient data points or stay time")
    
    # Convert to minutes for remaining locations and filter out data quality issues
    final_location_visits = {}
    for place_name, stats in filtered_location_visits.items():
        total_seconds = stats['total_time_seconds']
        
        # With multiple data points, we should always have a valid time span
        # If total_seconds is 0, it indicates a data quality issue (same timestamps)
        if total_seconds <= 0:
            print(f"Warning: Location '{place_name}' has {stats['data_points']} data points but zero time span - possible data quality issue")
            print(f"  Visit periods: {stats.get('visit_periods', [])}")
            print(f"  Skipping location due to zero time span")
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
    print("Generating integrated description for messages")
    
    if sensor_name != "messages" or not sensor_data:
        print("No messages data available, skipping messages integration")
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
        description_parts = [f"{datetime_str} | messages | Messaging Activity"]
        
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
    
    print(f"Generated {len(narratives)} messages narratives (window size: {sensor_integration_time_window} minutes)")
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
        list: List of formatted calls narrative tuples (datetime, description)
    """
    print("Generating integrated description for calls")
    
    if sensor_name != "calls" or not sensor_data:
        print("No calls data available, skipping calls integration")
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
        description_parts = [f"{datetime_str} | calls | Call Activity"]
        
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
    
    print(f"Generated {len(narratives)} calls narratives (window size: {sensor_integration_time_window} minutes)")
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
        list: List of formatted wifi narrative tuples (datetime, description)
    """
    print("Generating combined integrated description for WiFi (connections + detections)")
    
    if not sensor_wifi_data and not wifi_data:
        print("No WiFi data available, skipping WiFi integration")
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
        description_parts = [f"{datetime_str} | wifi | WiFi Activity Analysis"]
        
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
        description_parts = [f"{datetime_str} | wifi | WiFi Connection Activity"]
        
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
        description_parts = [f"{datetime_str} | wifi | WiFi Networks Detected"]
        
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
        combined_data, "wifi_combined", start_timestamp, end_timestamp, process_wifi_combined_window_refactored
    )
    
    sensor_types = []
    if sensor_wifi_data:
        sensor_types.append("connections")
    if wifi_data:
        sensor_types.append("detections")
    
    print(f"Generated {len(narratives)} combined WiFi narratives ({', '.join(sensor_types)}) (window size: {sensor_integration_time_window} minutes)")
    return narratives

def process_network_state_window(network_window_data, datetime_str, window_start, window_end):
    """Process network data to analyze WiFi and Mobile network connectivity patterns with airplane mode detection."""
    if not network_window_data:
        return None
    
    # Define network types with airplane mode detection
    relevant_network_types = {
        -1: "Airplane mode",
        1: "WiFi network", 
        4: "Mobile network"
    }
    
    # Group by network type and process each type
    network_types = {}
    airplane_mode_active = False
    
    for record in network_window_data:
        network_type = record.get('network_type')
        network_subtype = record.get('network_subtype', 'UNKNOWN')
        
        # Only process relevant network types
        if network_type not in relevant_network_types:
            continue
            
        if network_type not in network_types:
            network_types[network_type] = {
                'subtype': network_subtype,
                'display_name': relevant_network_types[network_type],
                'records': []
            }
        network_types[network_type]['records'].append(record)
        
        # Check for airplane mode
        if network_type == -1 and record.get('network_state') == 1:
            airplane_mode_active = True
    
    # Process each network type and analyze patterns
    all_results = []
    final_network_candidates = []
    network_periods = {}  # Store period information for pattern analysis
    
    for network_type, type_data in network_types.items():
        subtype = type_data['subtype']
        display_name = type_data['display_name']
        records = type_data['records']
        
        if not records:
            continue
    
        # Sort by timestamp
        sorted_data = sorted(records, key=lambda x: x['timestamp'])
        
        # Analyze state transitions and create periods
        state_periods = []
        current_state = None
        current_start = None
        
        for record in sorted_data:
            network_state = record.get('network_state')
            timestamp = record.get('timestamp')
            record_datetime = record.get('datetime', datetime_str)
            
            if current_state is None:
                # First record
                current_state = network_state
                current_start = timestamp
                current_start_datetime = record_datetime
            elif network_state != current_state:
                # State change
                state_periods.append({
                    'state': current_state,
                    'start_timestamp': current_start,
                    'start_datetime': current_start_datetime,
                    'end_timestamp': timestamp,
                    'end_datetime': record_datetime,
                    'duration_ms': timestamp - current_start
                })
                
                current_state = network_state
                current_start = timestamp
                current_start_datetime = record_datetime
        
        # Add final period
        if current_state is not None:
            state_periods.append({
                'state': current_state,
                'start_timestamp': current_start,
                'start_datetime': current_start_datetime,
                'end_timestamp': window_end,
                'end_datetime': datetime_str,
                'duration_ms': window_end - current_start
            })
        
        # Store periods for pattern analysis
        network_periods[network_type] = state_periods
        
        # Generate description for this network type
        if not state_periods:
            continue
    
        # Calculate total times
        total_on_time = sum(period['duration_ms'] for period in state_periods if period['state'] == 1)
        total_off_time = sum(period['duration_ms'] for period in state_periods if period['state'] == 0)
        window_duration = window_end - window_start
        
        on_percentage = (total_on_time / window_duration) * 100 if window_duration > 0 else 0
        off_percentage = (total_off_time / window_duration) * 100 if window_duration > 0 else 0
        
        # Format durations
        def format_duration(ms):
            if ms > 60000:
                mins = int(ms / 60000)
                secs = int((ms % 60000) / 1000)
                if mins > 0 and secs > 0:
                    return f"{mins} min {secs} sec"
                elif mins > 0:
                    return f"{mins} min"
                else:
                    return f"{secs} sec"
            else:
                return f"{int(ms / 1000)} sec"
        
        # Build description for this network type
        type_description = []
        
        # Only show detailed information for WiFi network (type 1)
        # Mobile network info will only appear in the pattern sequence
        if network_type == 1:  # WiFi network only
            # Show overall connection summary
            if total_on_time > 0:
                on_str = format_duration(total_on_time)
                type_description.append(f"    - {display_name} ON time: {on_str} ({on_percentage:.1f}% of window)")
            
            if total_off_time > 0:
                off_str = format_duration(total_off_time)
                type_description.append(f"    - {display_name} OFF time: {off_str} ({off_percentage:.1f}% of window)")
            
            # Show state transitions if any
            state_changes = len([p for p in state_periods if len(state_periods) > 1])
            if state_changes > 1:
                type_description.append(f"    - State changes: {state_changes - 1} transitions")
                
                # Show sequence of states with periods
                if len(state_periods) <= 6:  # Only show sequence if not too long
                    sequence = []
                    for period in state_periods:
                        state_label = "ON" if period['state'] == 1 else "OFF"
                        duration = format_duration(period['duration_ms'])
                        sequence.append(f"{state_label}({duration})")
                    
                    type_description.append(f"    - State sequence: {' → '.join(sequence)}")
            
            # Show WiFi usage periods with start/end times
            wifi_on_periods = [p for p in state_periods if p['state'] == 1]
            if wifi_on_periods:
                type_description.append(f"    - WiFi usage periods:")
                for i, period in enumerate(wifi_on_periods):
                    start_time = pd.to_datetime(period['start_timestamp'], unit='ms', utc=True).tz_convert('Australia/Melbourne').strftime('%H:%M:%S')
                    end_time = pd.to_datetime(period['end_timestamp'], unit='ms', utc=True).tz_convert('Australia/Melbourne').strftime('%H:%M:%S')
                    duration = format_duration(period['duration_ms'])
                    type_description.append(f"         - Period {i+1}: {start_time} to {end_time} ({duration})")
        
        # Track final state candidates
        final_state = state_periods[-1]['state'] if state_periods else None
        if final_state is not None:
            final_state_label = "ON" if final_state == 1 else "OFF"
            last_timestamp = sorted_data[-1].get('timestamp', 0)
            
            # Priority: Airplane mode highest, then WiFi, then Mobile
            if network_type == -1:
                priority = 3  # Airplane mode highest priority
            elif network_type == 1:
                priority = 2  # WiFi
            else:
                priority = 1  # Mobile
            
            final_network_candidates.append({
                'network_type': network_type,
                'subtype': subtype,
                'display_name': display_name,
                'final_state_label': final_state_label,
                'last_timestamp': last_timestamp,
                'priority': priority
            })
        
        # Store result for this network type
        all_results.extend(type_description)
    
    # Analyze disconnection patterns and airplane mode correlation
    if airplane_mode_active:
        # Check if WiFi/Mobile disconnections correlate with airplane mode
        airplane_periods = network_periods.get(-1, [])
        wifi_periods = network_periods.get(1, [])
        mobile_periods = network_periods.get(4, [])
        
        # Find airplane mode ON periods
        airplane_on_periods = [p for p in airplane_periods if p['state'] == 1]
        
        if airplane_on_periods:
            all_results.append(f"    - Airplane mode detected during {len(airplane_on_periods)} period(s)")
            
            # Check if disconnections happen during airplane mode
            disconnections_during_airplane = []
            for airplane_period in airplane_on_periods:
                airplane_start = airplane_period['start_timestamp']
                airplane_end = airplane_period['end_timestamp']
                
                # Check WiFi disconnections during airplane mode
                for wifi_period in wifi_periods:
                    if (wifi_period['state'] == 0 and 
                        wifi_period['start_timestamp'] >= airplane_start and 
                        wifi_period['end_timestamp'] <= airplane_end):
                        disconnections_during_airplane.append("WiFi")
                
                # Check Mobile disconnections during airplane mode
                for mobile_period in mobile_periods:
                    if (mobile_period['state'] == 0 and 
                        mobile_period['start_timestamp'] >= airplane_start and 
                        mobile_period['end_timestamp'] <= airplane_end):
                        disconnections_during_airplane.append("Mobile")
            
            if disconnections_during_airplane:
                unique_disconnections = list(set(disconnections_during_airplane))
                all_results.append(f"    - Disconnections caused by airplane mode: {', '.join(unique_disconnections)}")
    
    # Generate network pattern summary (WiFi takes precedence over Mobile when both are ON)
    if len(network_periods) > 1:
        # Create combined pattern sequence based on actual network usage
        wifi_periods = network_periods.get(1, [])
        mobile_periods = network_periods.get(4, [])
        airplane_periods = network_periods.get(-1, [])
        
        # Create timeline of network usage events
        usage_events = []
        
        # Add airplane mode events (highest priority)
        for period in airplane_periods:
            if period['state'] == 1:  # Airplane mode ON
                usage_events.append({
                    'timestamp': period['start_timestamp'],
                    'end_timestamp': period['end_timestamp'],
                    'display_name': 'Airplane mode',
                    'duration_ms': period['duration_ms'],
                    'priority': 3
                })
        
        # Determine network usage periods (WiFi takes precedence over Mobile)
        # Create a timeline of all network state changes
        all_state_changes = []
        
        for period in wifi_periods:
            all_state_changes.append({
                'timestamp': period['start_timestamp'],
                'network_type': 1,
                'state': period['state'],
                'duration_ms': period['duration_ms']
            })
        
        for period in mobile_periods:
            all_state_changes.append({
                'timestamp': period['start_timestamp'],
                'network_type': 4,
                'state': period['state'],
                'duration_ms': period['duration_ms']
            })
        
        # Sort by timestamp
        all_state_changes.sort(key=lambda x: x['timestamp'])
        
        # Track current state of each network
        current_wifi_state = 0
        current_mobile_state = 0
        
        # Determine actual network usage at each time point
        for i, change in enumerate(all_state_changes):
            if change['network_type'] == 1:  # WiFi
                current_wifi_state = change['state']
            elif change['network_type'] == 4:  # Mobile
                current_mobile_state = change['state']
            
            # Determine which network is actually being used
            if current_wifi_state == 1:
                # WiFi is ON - use WiFi (takes precedence)
                active_network = 'WiFi network'
                priority = 2
            elif current_mobile_state == 1:
                # Only Mobile is ON - use Mobile
                active_network = 'Mobile network'
                priority = 1
            else:
                # Both are OFF - no network
                active_network = None
                priority = 0
            
            # Add usage event if network is active
            if active_network:
                # Calculate duration until next state change or end of window
                if i + 1 < len(all_state_changes):
                    next_change_time = all_state_changes[i + 1]['timestamp']
                    duration = next_change_time - change['timestamp']
                else:
                    duration = window_end - change['timestamp']
                
                # Only add if this represents a meaningful change in network usage
                if not usage_events or usage_events[-1]['display_name'] != active_network:
                    usage_events.append({
                        'timestamp': change['timestamp'],
                        'end_timestamp': change['timestamp'] + duration,
                        'display_name': active_network,
                        'duration_ms': duration,
                        'priority': priority
                    })
                else:
                    # Extend the duration of the current usage period
                    usage_events[-1]['duration_ms'] += duration
                    usage_events[-1]['end_timestamp'] = change['timestamp'] + duration
        
        # Sort usage events by timestamp
        usage_events.sort(key=lambda x: x['timestamp'])
        
        # Create pattern summary with start times
        if len(usage_events) > 0:
            pattern_summary = []
            for event in usage_events[:5]:  # Show up to 5 events
                start_time = pd.to_datetime(event['timestamp'], unit='ms', utc=True).tz_convert('Australia/Melbourne').strftime('%H:%M:%S')
                duration = format_duration(event['duration_ms'])
                pattern_summary.append(f"{event['display_name']} ON({start_time}, {duration})")
            
            if len(usage_events) > 5:
                pattern_summary.append("...")
            
            all_results.append(f"    - Network pattern sequence: {' → '.join(pattern_summary)}")

    
    # Generate final description
    if not all_results:
        return None
    
    description_parts = [f"{datetime_str} | wifi | WiFi Activity Analysis"]
    description_parts.extend(all_results)
    return '\n'.join(description_parts)



def get_sensor_wifi_basic_analysis(window_data, datetime_str, window_start, window_end, has_network_state=False):
    """Standalone function to analyze sensor_wifi data for a window - extracted from nested function."""
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
    description_parts = [f"{datetime_str} | wifi | WiFi Connection Activity"]
    
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
    
    # Connection pattern analysis - be honest about what we know vs. infer
    window_duration_ms = window_end - window_start
    
    # What we can reliably measure: time between network switches
    network_switch_durations = [conn for conn in connection_sequence if conn.get('duration_ms', 0) > 0]
    
    if len(connection_sequence) == 1:
        # Single network connection event in this window
        single_network = connection_sequence[0]['display_name']
        description_parts.append(f"    - Connection pattern: {single_network}")
        
    elif len(network_switch_durations) > 0:
        # Multiple networks - we can measure time between switches
        total_switch_time_ms = sum(conn['duration_ms'] for conn in network_switch_durations)
        
        # Format switch time
        if total_switch_time_ms > 60000:
            switch_mins = int(total_switch_time_ms / 60000)
            switch_secs = int((total_switch_time_ms % 60000) / 1000)
            
            if switch_mins > 0:
                if switch_secs > 0:
                    switch_str = f"{switch_mins} min {switch_secs} sec"
                else:
                    switch_str = f"{switch_mins} min"
            else:
                switch_str = f"{switch_secs} sec"
        else:
            switch_str = f"{int(total_switch_time_ms / 1000)} sec"
        
        description_parts.append(f"    - Time between network switches: {switch_str}")
        
        # Switch intervals and switching pattern removed as requested
        
    else:
        # Multiple connection events but no measurable durations
        description_parts.append(f"    - Connection pattern: Multiple events with rapid switches (durations too short to measure)")
     
    return '\n'.join(description_parts)

def describe_screentext_integrated(sensor_data, sensor_name, start_timestamp, end_timestamp, sessions=None):
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
        list: List of formatted screentext narrative tuples (datetime, description)
    """
    print("Generating integrated description for screentext")
    
    if sensor_name != "screentext":
        print("Invalid sensor name for screentext integration")
        return []
    
    # Load screentext data from config file
    cleaned_screentext_file = CONFIG.get("cleaned_screentext_file", "").format(P_ID=P_ID)
    if not cleaned_screentext_file or not os.path.exists(cleaned_screentext_file):
        print(f"Screentext file not found: {cleaned_screentext_file}")
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
        print(f"Error loading screentext data: {e}")
        return []
    
    if not screentext_records:
        print("No screentext records found")
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
        print("No screentext records in time range")
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
        description_parts = [f"{datetime_str} | screentext | Tracked Logs"]
        
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
            
            if overlapping_sessions:
                if len(overlapping_sessions) == 1:
                    description_parts.append(f"    - Session activity: Session {overlapping_sessions[0]}")
                else:
                    description_parts.append(f"    - Session activity: Sessions {', '.join(map(str, overlapping_sessions))}")
        
        return '\n'.join(description_parts)
    
    # Process data using the shared helper function
    narratives = process_sensor_by_timewindow(
        filtered_records, sensor_name, start_timestamp, end_timestamp, process_screentext_window
    )
    
    print(f"Generated {len(narratives)} screentext narratives (window size: {sensor_integration_time_window} minutes)")
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
        list: List of formatted installations narrative tuples (datetime, description)
    """
    print("Generating integrated description for installations")
    
    if sensor_name != "installations" or not sensor_data:
        print("No installations data available, skipping installations integration")
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
        description_parts = [f"{datetime_str} | installations | App Installation Activity"]
        
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
    
    print(f"Generated {len(narratives)} installations narratives (window size: {sensor_integration_time_window} minutes)")
    return narratives

def split_description_by_days(description_file_path, daily_output_dir):
    """
    Split description file by days and save each day's content to separate files.
    
    Args:
        description_file_path (str): Path to the input description file
        daily_output_dir (str): Directory to save daily output files
    
    Returns:
        dict: Dictionary mapping dates to their output file paths
    """
    import os
    import re
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(daily_output_dir, exist_ok=True)
    
    # Read the entire description file
    with open(description_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by windows - look for window headers
    window_pattern = r'(Window \d+\nDay \d{4}-\d{2}-\d{2} \(.*?\)\n\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2})'
    windows = re.split(window_pattern, content)
    
    # Group windows by day
    daily_windows = {}
    current_day = None
    current_content = ""
    
    for i, window in enumerate(windows):
        if window.startswith('Window '):
            # Extract date from the window header
            date_match = re.search(r'Day (\d{4}-\d{2}-\d{2})', window)
            if date_match:
                day_date = date_match.group(1)
                
                # If we have content from a previous day, save it
                if current_day and current_content.strip():
                    if current_day not in daily_windows:
                        daily_windows[current_day] = ""
                    daily_windows[current_day] += current_content.strip() + "\n\n"
                
                # Start new day
                current_day = day_date
                current_content = window
            else:
                # If no date found, append to current content
                current_content += window
        else:
            # This is the content between windows
            current_content += window
    
    # Save the last day's content
    if current_day and current_content.strip():
        if current_day not in daily_windows:
            daily_windows[current_day] = ""
        daily_windows[current_day] += current_content.strip()
    
    # Write each day's content to a separate file
    output_files = {}
    for day_date, day_content in daily_windows.items():
        # Create filename with date
        filename = f"day_{day_date}.txt"
        file_path = os.path.join(daily_output_dir, filename)
        
        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(day_content)
        
        output_files[day_date] = file_path
    
    return output_files

if __name__ == "__main__":
    print("Start narrating sensor data for multiple participants...")
    
    # Load app_package_pairs.jsonl from resources folder for future package name to application name mapping
    # This is loaded once and shared across all participants
    app_package_file_path = os.path.join("resources", "app_package_pairs.jsonl")
    
    try:
        with open(app_package_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    app_data = json.loads(line)
                    package_name = app_data.get("package_name")
                    application_name = app_data.get("application_name")
                    if package_name and application_name:
                        application_name_list[package_name] = application_name
        print(f"Loaded {len(application_name_list)} app package mappings")
    except FileNotFoundError:
        print(f"Warning: App package pairs file {app_package_file_path} not found")
        application_name_list = {}
    except Exception as e:
        print(f"Error reading app package pairs file: {e}")
        application_name_list = {}
    
    # Process each participant
    successful_participants = []
    failed_participants = []
    
    for P_ID in P_IDs:
        try:
            success = process_participant(P_ID)
            if success:
                successful_participants.append(P_ID)
            else:
                failed_participants.append(P_ID)
        except Exception as e:
            print(f"Error processing participant {P_ID}: {e}")
            failed_participants.append(P_ID)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(successful_participants)} participants")
    if successful_participants:
        print(f"  - {', '.join(successful_participants)}")
    
    print(f"Failed to process: {len(failed_participants)} participants")
    if failed_participants:
        print(f"  - {', '.join(failed_participants)}")
    
    print(f"\nTotal participants: {len(P_IDs)}")
    print("Sensor narration completed!")