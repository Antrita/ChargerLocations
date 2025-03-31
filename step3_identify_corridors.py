"""
Step 3. Identifying existing corridors where slow/fast chargers are located.
"""

import pandas as pd
from shapely import wkt
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import requests
from tqdm import tqdm
import pickle
import os
import sys
from tabulate import tabulate
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("corridor_identification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("corridor-identifier")


def load_data(csv_path):
    """Load the CSV file with proper handling of geometry column"""
    try:
        # Load as a regular CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {csv_path}")
        logger.info(f"CSV has {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Column names: {', '.join(df.columns)}")

        # Check if geometry column exists
        if 'geometry' not in df.columns:
            logger.error("No 'geometry' column found in the CSV file.")
            return None

        # Convert geometry strings to shapely objects
        try:
            df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
            return df
        except Exception as e:
            logger.error(f"Error converting geometry: {e}")
            return None

    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None


def find_roads_in_polygon(polygon, attempt=1, max_attempts=3):
    """Query Overpass API to find major roads within polygon bounds"""
    if attempt > max_attempts:
        return []

    try:
        # Get polygon bounds
        minx, miny, maxx, maxy = polygon.bounds

        # Overpass API query for major roads within the bounding box
        # We focus on motorways, trunks, primary and secondary roads
        overpass_url = "https://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        (
          way["highway"="motorway"]({miny},{minx},{maxy},{maxx});
          way["highway"="trunk"]({miny},{minx},{maxy},{maxx});
          way["highway"="primary"]({miny},{minx},{maxy},{maxx});
          way["highway"="secondary"]({miny},{minx},{maxy},{maxx});
        );
        out body;
        >;
        out skel qt;
        """

        # Send the request
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=30)

        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            roads = []

            # Extract road information
            for element in data['elements']:
                if element['type'] == 'way' and 'tags' in element and 'highway' in element['tags']:
                    road_type = element['tags']['highway']
                    road_name = element['tags'].get('name', '')
                    road_ref = element['tags'].get('ref', '')

                    # Use ref number if name is not available (common for highways)
                    if not road_name and road_ref:
                        road_name = road_ref

                    # Assign priority based on road type
                    priority = {
                        'motorway': 1,
                        'trunk': 2,
                        'primary': 3,
                        'secondary': 4
                    }.get(road_type, 5)

                    if road_name:  # Only include roads with names
                        roads.append({
                            'name': road_name,
                            'type': road_type,
                            'priority': priority
                        })

            # Sort roads by priority
            roads.sort(key=lambda x: x['priority'])
            return roads

        else:
            logger.warning(f"Overpass API request failed: {response.status_code}")
            time.sleep(5)  # Longer wait on error
            return find_roads_in_polygon(polygon, attempt + 1, max_attempts)

    except Exception as e:
        logger.warning(f"Error querying roads: {e}")
        time.sleep(3)  # Wait before retry
        return find_roads_in_polygon(polygon, attempt + 1, max_attempts)


def get_settlement_name(polygon, geocoder, attempt=1, max_attempts=2):
    """Get settlement name for a polygon"""
    if attempt > max_attempts:
        return "Unknown Area"

    try:
        # Get the centroid of the polygon
        centroid = polygon.centroid

        # Query Nominatim for location data
        location = geocoder.reverse((centroid.y, centroid.x), language='en', timeout=10)

        if location:
            address = location.raw.get('address', {})

            # Try to get the most relevant settlement name
            settlement = (address.get('city') or
                          address.get('town') or
                          address.get('village') or
                          address.get('county') or
                          address.get('state'))

            return settlement if settlement else "Unknown Area"

        return "Unknown Area"

    except Exception as e:
        logger.warning(f"Error in geocoding: {e}")
        time.sleep(1)  # Wait before retry
        return get_settlement_name(polygon, geocoder, attempt + 1, max_attempts)


def display_corridors(df, recent_indices, total=10):
    """Display recently identified corridors"""
    # Get the recent corridors
    recent_corridors = df.loc[recent_indices, ['corridor_name']].copy()

    # Add index as ID
    recent_corridors.insert(0, 'ID', recent_indices)

    # Add centroid coordinates for reference
    recent_corridors['Coordinates'] = df.loc[recent_indices, 'geometry'].apply(
        lambda g: f"{g.centroid.y:.4f}, {g.centroid.x:.4f}"
    )

    # Display the corridors
    print("\n=== Recently Identified Corridors ===")
    print(tabulate(recent_corridors.head(total), headers='keys', tablefmt='grid', showindex=False))
    print()


def calculate_distance(point1, point2):
    """Calculate distance between two points (in km)"""
    return geodesic(
        (point1.y, point1.x),
        (point2.y, point2.x)
    ).kilometers


def evaluate_charger_importance(row):
    """
    Calculate importance score for a charging station based on key variables.
    Higher score = more important to keep in the filtered dataset.
    """
    score = 0

    # Number of electrified trucks in 2030 (higher is more important)
    # Note: Column name is MainDTN in your data
    if 'MainDTN' in row and pd.notna(row['MainDTN']):
        score += min(row['MainDTN'] / 100, 50)  # Cap at 50 points

    # Total number of chargers (higher is more important)
    if 'TotCha' in row and pd.notna(row['TotCha']):
        score += min(row['TotCha'] * 2, 30)  # Cap at 30 points

    # Balance of fast and slow chargers (balanced mix is more important)
    if 'NFCh30m' in row and 'NSCh2pD' in row and pd.notna(row['NFCh30m']) and pd.notna(row['NSCh2pD']):
        if row['NFCh30m'] > 0 and row['NSCh2pD'] > 0:
            fast_ratio = row['NFCh30m'] / (row['NFCh30m'] + row['NSCh2pD'])
            # Score is highest when there's a balanced mix (around 0.5 ratio)
            balance_score = 20 * (1 - abs(fast_ratio - 0.5) * 2)
            score += balance_score

    # Charged energy (higher is more important)
    if 'ChE30' in row and pd.notna(row['ChE30']):
        score += min(row['ChE30'] / 50, 20)  # Cap at 20 points

    return score


def identify_transport_corridors(min_distance=500, input_file=None):
    """Main function to identify corridors"""
    try:
        print("=== Transport Corridor Identification with Overpass API ===")
        logger.info(f"Starting corridor identification with min_distance={min_distance}km")

        # Determine input file path
        if input_file is None:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            input_file = os.path.join(script_dir, 'ChargerLocations.csv')

        # Check if the input file exists
        if not os.path.exists(input_file):
            error_msg = f"Input file not found: {input_file}"
            logger.error(error_msg)
            print(error_msg)
            return False

        # Check for cache file to resume processing
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_file = os.path.join(script_dir, 'corridor_cache.pkl')
        corridor_cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    corridor_cache = pickle.load(f)
                logger.info(f"Loaded {len(corridor_cache)} cached corridors from previous run.")
            except Exception as e:
                logger.warning(f"Could not load cache file: {e}. Starting fresh.")

        # Load the data
        print(f"Loading {input_file}...")
        df = load_data(input_file)

        if df is None:
            logger.error("Could not load data. Please check your CSV file.")
            return False

        total_rows = len(df)
        logger.info(f"Loaded {total_rows} charging location polygons.")

        # Initialize OpenStreetMap geocoder
        geocoder = Nominatim(user_agent="transport_corridor_identifier")
        logger.info("Initialized OpenStreetMap geocoder.")

        # Initialize corridor names column
        if 'corridor_name' not in df.columns:
            df['corridor_name'] = None

        # Create a hash function for polygons
        def hash_polygon(poly):
            return str(hash(poly.wkt))

        # Add a hash column for caching
        df['poly_hash'] = df['geometry'].apply(hash_polygon)

        # Add centroid column for faster distance calculations
        df['centroid'] = df['geometry'].apply(lambda x: x.centroid)

        # Calculate importance score for each charging station
        print("Calculating importance scores for charging stations...")
        try:
            # Adjust column names to match your data
            df['importance_score'] = df.apply(evaluate_charger_importance, axis=1)
            # Sort by importance score (descending) to prioritize more important stations
            df = df.sort_values('importance_score', ascending=False).reset_index(drop=True)
            logger.info("Sorted charging stations by importance score")
        except Exception as e:
            logger.warning(f"Could not calculate importance scores: {e}")
            # Add a default importance score if calculation fails
            df['importance_score'] = 50

        # Set checkpoint frequency
        checkpoint_frequency = 50

        # Create a dataframe to store selected locations
        selected_df = df.iloc[0:0].copy()

        # Dictionary to track corridors and their representative locations
        corridor_representatives = {}

        # List to store centroids of selected locations for distance checks
        selected_centroids = []

        # Process rows
        print(f"Processing {total_rows} polygons to identify transport corridors...")
        print(f"Checkpoints will be saved every {checkpoint_frequency} records")

        # Keep track of recently processed indices
        recent_indices = []

        # Use tqdm for progress bar
        for i, row in tqdm(df.iterrows(), total=total_rows, desc="Processing polygons"):
            # Skip if already processed
            if row['poly_hash'] in corridor_cache:
                df.at[i, 'corridor_name'] = corridor_cache[row['poly_hash']]
            else:
                polygon = row['geometry']

                # Find roads in this polygon
                roads = find_roads_in_polygon(polygon)

                # Get settlement name
                settlement = get_settlement_name(polygon, geocoder)

                # Format corridor name
                if roads:
                    top_road = roads[0]  # Get the highest priority road
                    road_name = top_road['name']
                    road_type = top_road['type']

                    # Format based on road type
                    if road_type == 'motorway':
                        corridor_name = f"Motorway {road_name} Corridor ({settlement})"
                    elif road_type == 'trunk':
                        corridor_name = f"Trunk Road {road_name} Corridor ({settlement})"
                    else:
                        corridor_name = f"{road_name} Corridor ({settlement})"
                else:
                    # No roads found
                    corridor_name = f"{settlement} Area"

                # Save to dataframe and cache
                df.at[i, 'corridor_name'] = corridor_name
                corridor_cache[row['poly_hash']] = corridor_name

                # Add to recent indices
                recent_indices.append(i)

                # Add a delay to respect API usage policies
                time.sleep(1.1)

            # Apply filtering logic on the fly
            corridor = df.at[i, 'corridor_name']
            centroid = df.at[i, 'centroid']
            importance_score = df.at[i, 'importance_score']

            # Check if this location should be included
            include_location = False

            # For high importance stations (top 10%), lower the distance threshold
            adjusted_min_distance = min_distance
            if i < len(df) * 0.1:  # Top 10% by importance
                adjusted_min_distance = min_distance * 0.8  # 20% shorter distance requirement

            # Include if it's the first representative of its corridor
            if corridor not in corridor_representatives:
                # Get the highest importance score for this corridor
                same_corridor_indices = df[df['corridor_name'] == corridor].index
                highest_score_idx = df.loc[same_corridor_indices, 'importance_score'].idxmax()

                # If this is the most important station in the corridor, add it
                if i == highest_score_idx:
                    corridor_representatives[corridor] = i
                    include_location = True

            # Or include if it's at least min_distance away from all selected locations
            if not include_location:
                far_enough = True
                for selected_centroid in selected_centroids:
                    distance = calculate_distance(centroid, selected_centroid)
                    if distance < adjusted_min_distance:
                        far_enough = False
                        break

                if far_enough:
                    include_location = True

            # Additional criteria based on truck data:
            # If this is a high-traffic location (high MainDTN) but wasn't included by other criteria,
            # check if we can make an exception
            if not include_location and 'MainDTN' in df.columns:
                truck_count = df.at[i, 'MainDTN'] if pd.notna(df.at[i, 'MainDTN']) else 0
                if truck_count > 1000:  # High truck traffic threshold
                    # Try with a more relaxed distance requirement
                    relaxed_distance = min_distance * 0.7  # 30% shorter
                    far_enough = True
                    for selected_centroid in selected_centroids:
                       distance = calculate_distance(centroid, selected_centroid)
                       if distance < relaxed_distance:
                          far_enough = False
                          break

                    if far_enough:
                        include_location = True
                # If we're including this location, add it to our selected dataframe
                if include_location:
                    selected_df = pd.concat([selected_df, df.iloc[[i]]])
                    selected_centroids.append(centroid)

                # Save progress at checkpoints
                if (i + 1) % checkpoint_frequency == 0 or i == len(df) - 1:
                    # Display recently identified corridors
                    if recent_indices:
                        display_corridors(df, recent_indices)
                        recent_indices = []  # Reset after display

                    # Output directory is the same as script directory
                    output_dir = os.path.dirname(os.path.abspath(__file__))
                    # Remove temporary columns and save
                    save_df = df.drop(['poly_hash', 'centroid', 'importance_score'], axis=1)
                    save_path = os.path.join(output_dir, 'ChargerLocationsRefined.csv')
                    save_df.to_csv(save_path, index=False)

                    # Save filtered data
                    filtered_save_df = selected_df.drop(['poly_hash', 'centroid', 'importance_score'], axis=1)
                    filtered_save_path = os.path.join(output_dir, 'ChargerLocationsFiltered.csv')
                    filtered_save_df.to_csv(filtered_save_path, index=False)
                    # Save cache
                    with open(cache_file, 'wb') as f:
                        pickle.dump(corridor_cache, f)

                    logger.info(f"Saved checkpoint: {i + 1}/{total_rows} polygons processed")
                    logger.info(f"Selected {len(selected_df)}/{i + 1} locations so far")
                    print(f"Saved checkpoint: {i + 1}/{total_rows} polygons processed")
                    print(f"Selected {len(selected_df)}/{i + 1} locations so far")
                    print(f"Refined data saved to {save_path}")
                    print(f"Filtered data saved to {filtered_save_path}")

            # Remove temporary columns
            df = df.drop(['poly_hash', 'centroid', 'importance_score'], axis=1)
            selected_df = selected_df.drop(['poly_hash', 'centroid', 'importance_score'], axis=1)

            # Save final results
            output_dir = os.path.dirname(os.path.abspath(__file__))
            final_path = os.path.join(output_dir, 'ChargerLocationsRefined.csv')
            filtered_final_path = os.path.join(output_dir, 'ChargerLocationsFiltered.csv')

            df.to_csv(final_path, index=False)
            selected_df.to_csv(filtered_final_path, index=False)
            # Print summary
            print("\n=== Processing Complete ===")
            logger.info("Processing complete!")
            print(f"Total locations (before filtering): {len(df)}")
            print(f"Total locations (after filtering): {len(selected_df)}")
            print(f"Unique corridors identified: {len(set(selected_df['corridor_name']))}")

            # Display summary of corridor types
            corridor_types = []
            for name in selected_df['corridor_name']:
                if "Motorway" in str(name):
                    corridor_types.append("Motorway")
                elif "Trunk Road" in str(name):
                    corridor_types.append("Trunk Road")
                elif "Corridor" in str(name):
                    corridor_types.append("Other Road")
                else:
                    corridor_types.append("Area Only")
            type_counts = pd.Series(corridor_types).value_counts()
            print("\nCorridor Type Distribution:")
            for type_name, count in type_counts.items():
             print(f"  {type_name}: {count} ({count / len(selected_df):.1%})")

            print(f"\nRefined data saved to {final_path}")
            print(f"Filtered data saved to {filtered_final_path}")
            return True
    except Exception as e:
        # Catch any unhandled exceptions to avoid silent exit
        logger.error(f"Unhandled exception in corridor identification: {e}")
        logger.error(f"Stack trace: {sys.exc_info()}")
        print(f"ERROR: {e}")
        return False


if __name__ == "__main__":
    # Allow for command-line override of the input file
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else None

    success = identify_transport_corridors(min_distance=500, input_file=input_file)

    if not success:
        print("Script did not complete successfully. Check the log file.")
        sys.exit(1)  # Exit with non-zero status to indicate failure
    else:
        print("Script completed successfully.")
        sys.exit(0)  # Explicitly exit with 0 to indicate success

