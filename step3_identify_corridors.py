"""
Step 3. Identifying existing corridors where slow/fast chargers are located.
Enhanced with Google Maps API integration for more accurate corridor identification in EU countries.
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
        logging.FileHandler("corridor_identification_eu.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("corridor-identifier-eu")

# Try to import Google Maps utilities
try:
    # Try to import the EU-optimized version first
    try:
        from google_maps_utils_eu import batch_process_polygons, get_corridor_name, load_cache, save_cache

        GOOGLE_MAPS_AVAILABLE = True
        EU_OPTIMIZED = True
        logger.info("EU-optimized Google Maps utilities available for corridor identification")
    except ImportError:
        # Fall back to standard version
        from google_maps_utils import batch_process_polygons, get_corridor_name, load_cache, save_cache

        GOOGLE_MAPS_AVAILABLE = True
        EU_OPTIMIZED = False
        logger.info("Standard Google Maps utilities available for corridor identification")
except ImportError:
    GOOGLE_MAPS_AVAILABLE = False
    EU_OPTIMIZED = False
    logger.info("Google Maps utilities not available. Will use OpenStreetMap for corridor identification.")


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
        # Include European motorway designations
        overpass_url = "https://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        (
          way["highway"="motorway"]({miny},{minx},{maxy},{maxx});
          way["highway"="trunk"]({miny},{minx},{maxy},{maxx});
          way["highway"="primary"]({miny},{minx},{maxy},{maxx});
          way["highway"="secondary"]({miny},{minx},{maxy},{maxx});
          way["highway"="motorway_link"]({miny},{minx},{maxy},{maxx});
          way["highway"="trunk_link"]({miny},{minx},{maxy},{maxx});
          way["route"="road"]({miny},{minx},{maxy},{maxx});
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
                    # Check for European designation (e.g., "E55")
                    int_ref = element['tags'].get('int_ref', '')

                    # Use ref number if name is not available (common for highways)
                    road_name_final = road_name
                    if not road_name_final and road_ref:
                        road_name_final = road_ref
                    if not road_name_final and int_ref:
                        road_name_final = int_ref

                    # Assign priority based on road type
                    priority = {
                        'motorway': 1,
                        'trunk': 2,
                        'primary': 3,
                        'secondary': 4,
                        'motorway_link': 5,
                        'trunk_link': 6
                    }.get(road_type, 7)

                    # Boost priority for European routes (starting with E)
                    if int_ref and int_ref.startswith('E'):
                        priority -= 1

                    if road_name_final:  # Only include roads with names
                        roads.append({
                            'name': road_name_final,
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


def get_corridor_name_openstreetmap(polygon, geocoder):
    """
    Get corridor name using OpenStreetMap (legacy method).
    Used as a fallback when Google Maps API is not available.
    Optimized for European road naming conventions.
    """
    # Find roads in this polygon
    roads = find_roads_in_polygon(polygon)

    # Get settlement name
    settlement = get_settlement_name(polygon, geocoder)

    # Format corridor name
    if roads:
        top_road = roads[0]  # Get the highest priority road
        road_name = top_road['name']
        road_type = top_road['type']

        # Format based on road type and European conventions
        if road_type == 'motorway' or road_type == 'motorway_link':
            # Check for European motorway designation (e.g., "A1", "E55")
            if road_name.startswith('A') or road_name.startswith('E') or road_name.startswith('M'):
                corridor_name = f"Motorway {road_name} Corridor ({settlement})"
            else:
                corridor_name = f"{road_name} Highway Corridor ({settlement})"
        elif road_type == 'trunk' or road_type == 'trunk_link':
            corridor_name = f"{road_name} Route ({settlement})"
        else:
            corridor_name = f"{road_name} Corridor ({settlement})"
    else:
        # No roads found
        corridor_name = f"{settlement} Area"

    return corridor_name


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


def identify_transport_corridors(min_distance=500, input_file=None, use_google_maps=False, google_maps_api_key=None,
                                 batch_size=50, save_interval=50):
    """Main function to identify corridors with EU optimization"""
    try:
        print("=== Transport Corridor Identification (EU Optimized) ===")
        logger.info(f"Starting corridor identification with min_distance={min_distance}km")

        # Check if we should use Google Maps API
        if use_google_maps:
            if not GOOGLE_MAPS_AVAILABLE:
                logger.warning("Google Maps utilities not available despite request to use them.")
                print("Google Maps utilities not available. Falling back to OpenStreetMap.")
                use_google_maps = False
            elif not google_maps_api_key:
                logger.warning("No Google Maps API key provided. Using OpenStreetMap instead.")
                print("No Google Maps API key provided. Using OpenStreetMap instead.")
                use_google_maps = False
            else:
                if EU_OPTIMIZED:
                    logger.info("Using EU-optimized Google Maps API for corridor identification")
                    print("Using EU-optimized Google Maps API for corridor identification")
                else:
                    logger.info("Using Google Maps API for corridor identification")
                    print("Using Google Maps API for corridor identification")
                # Set the API key
                if EU_OPTIMIZED:
                    import google_maps_utils_eu
                    google_maps_utils_eu.GOOGLE_MAPS_API_KEY = google_maps_api_key
                else:
                    import google_maps_utils
                    google_maps_utils.GOOGLE_MAPS_API_KEY = google_maps_api_key

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
        cache_file = os.path.join(script_dir, 'corridor_cache_eu.pkl')
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

        # If using Google Maps API, process all polygons in a batch
        if use_google_maps:
            # Load the API cache first
            try:
                load_cache()
            except:
                pass

            print(f"Batch processing {total_rows} polygons with Google Maps API...")
            logger.info(f"Batch processing {total_rows} polygons with Google Maps API...")

            # Process all polygons at once with regular progress updates
            df = batch_process_polygons(df, geometry_col='geometry', batch_size=batch_size,
                                        api_key=google_maps_api_key, save_interval=save_interval)

            # Save the processed data immediately to avoid losing API calls
            save_path = os.path.join(script_dir, 'ChargerLocationsRefined.csv')
            df.to_csv(save_path, index=False)
            logger.info(f"Saved data with Google Maps corridors to {save_path}")
        else:
            # Initialize OpenStreetMap geocoder
            geocoder = Nominatim(user_agent="transport_corridor_identifier_eu")
            logger.info("Initialized OpenStreetMap geocoder.")

            # Initialize corridor names column
            if 'corridor_name' not in df.columns:
                df['corridor_name'] = None

            # Create a hash function for polygons
            def hash_polygon(poly):
                return str(hash(poly.wkt))

            # Add a hash column for caching
            df['poly_hash'] = df['geometry'].apply(hash_polygon)

            # Process rows
            print(f"Processing {total_rows} polygons to identify transport corridors...")
            print(f"Using OpenStreetMap to identify road corridors (EU optimized)...")

            # Keep track of recently processed indices
            recent_indices = []

            # Use tqdm for progress bar
            for i, row in tqdm(df.iterrows(), total=total_rows, desc="Processing polygons"):
                # Skip if already processed
                if row['poly_hash'] in corridor_cache:
                    df.at[i, 'corridor_name'] = corridor_cache[row['poly_hash']]
                else:
                    polygon = row['geometry']

                    # Get corridor name using OpenStreetMap
                    corridor_name = get_corridor_name_openstreetmap(polygon, geocoder)

                    # Save to dataframe and cache
                    df.at[i, 'corridor_name'] = corridor_name
                    corridor_cache[row['poly_hash']] = corridor_name

                    # Add to recent indices
                    recent_indices.append(i)

                    # Add a delay to respect API usage policies
                    time.sleep(1.1)

                # Save progress at specified intervals
                if (i + 1) % save_interval == 0 or i == len(df) - 1:
                    # Display recently identified corridors
                    if recent_indices:
                        display_corridors(df, recent_indices)
                        recent_indices = []  # Reset after display

                    # Save cache
                    with open(cache_file, 'wb') as f:
                        pickle.dump(corridor_cache, f)

                    # Output directory is the same as script directory
                    output_dir = os.path.dirname(os.path.abspath(__file__))

                    # Remove temporary columns and save
                    if 'poly_hash' in df.columns:
                        save_df = df.drop(['poly_hash'], axis=1)
                    else:
                        save_df = df

                    save_path = os.path.join(output_dir, 'ChargerLocationsRefined.csv')
                    save_df.to_csv(save_path, index=False)

                    logger.info(f"Saved checkpoint: {i + 1}/{total_rows} polygons processed")
                    print(f"Saved checkpoint: {i + 1}/{total_rows} polygons processed")
                    print(f"Refined data saved to {save_path}")

        # Now that we have corridor names, calculate centroids for all locations
        print("Calculating centroids for distance filtering...")
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

        # Create a dataframe to store selected locations
        selected_df = df.iloc[0:0].copy()

        # Dictionary to track corridors and their representative locations
        corridor_representatives = {}

        # List to store centroids of selected locations for distance checks
        selected_centroids = []

        # Filter locations based on distance and importance
        print("Filtering locations to select representative charging stations...")
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Filtering locations"):
            corridor = row['corridor_name']
            centroid = row['centroid']
            importance_score = row['importance_score']

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
            # If this is a high-traffic location but wasn't included by other criteria,
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

        # Remove temporary columns
        if 'centroid' in df.columns:
            df = df.drop(['centroid'], axis=1)
        if 'importance_score' in df.columns:
            df = df.drop(['importance_score'], axis=1)

        if 'centroid' in selected_df.columns:
            selected_df = selected_df.drop(['centroid'], axis=1)
        if 'importance_score' in selected_df.columns:
            selected_df = selected_df.drop(['importance_score'], axis=1)

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
            elif "Highway" in str(name):
                corridor_types.append("Highway")
            elif "Route" in str(name):
                corridor_types.append("Route")
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
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        print(f"ERROR: {e}")
        return False


if __name__ == "__main__":
    # Allow for command-line override of the input file
    import argparse

    parser = argparse.ArgumentParser(description='Identify transport corridors in charging station data (EU optimized)')
    parser.add_argument('--csv', type=str, help='Path to CSV file with charger locations')
    parser.add_argument('--min-distance', type=float, default=500,
                        help='Minimum distance between charging stations (km)')
    parser.add_argument('--use-google-maps', action='store_true',
                        help='Use Google Maps API for corridor identification')
    parser.add_argument('--api-key', type=str,
                        help='Google Maps API key (required if --use-google-maps is specified)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for Google Maps API requests (default: 50)')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Save after processing this many locations (default: 50)')

    args = parser.parse_args()

    # Check if API key is provided when Google Maps is requested
    if args.use_google_maps and not args.api_key:
        print("ERROR: --use-google-maps specified but no API key provided.")
        print("Please provide an API key with --api-key YOUR_API_KEY")
        sys.exit(1)

    success = identify_transport_corridors(
        min_distance=args.min_distance,
        input_file=args.csv,
        use_google_maps=args.use_google_maps,
        google_maps_api_key=args.api_key,
        batch_size=args.batch_size,
        save_interval=args.save_interval
    )

    if not success:
        print("Script did not complete successfully. Check the log file.")
        sys.exit(1)  # Exit with non-zero status to indicate failure
    else:
        print("Script completed successfully.")
        sys.exit(0)  # Explicitly exit with 0 to indicate success