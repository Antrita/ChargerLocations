"""
Utility functions for Google Maps API integration - EU version.
Used to convert polygon coordinates to corridor names and locations for European data.
"""

import requests
import time
import os
import logging
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import pickle
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("google_maps_api_eu.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("google-maps-utils-eu")

# Google Maps API configuration
GOOGLE_MAPS_API_KEY = ""  # Add your API key here

# EU country code (change to specific country if known)
EU_COUNTRY_CODE = "EU"  # Change to specific country code like "DE", "FR", "IT", etc.

# Cache for API responses to avoid duplicate requests
CACHE_FILE = "google_maps_cache_eu.pkl"
api_cache = {}


def load_cache():
    """Load the API response cache from file if it exists."""
    global api_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                api_cache = pickle.load(f)
            logger.info(f"Loaded {len(api_cache)} cached responses")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            api_cache = {}


def save_cache():
    """Save the API response cache to file."""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(api_cache, f)
        logger.info(f"Saved {len(api_cache)} responses to cache")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def get_polygon_centroid(polygon):
    """
    Calculate the centroid of a polygon.

    Args:
        polygon: A shapely Polygon object

    Returns:
        (lat, lng) tuple of centroid coordinates
    """
    try:
        centroid = polygon.centroid
        return (centroid.y, centroid.x)  # Return as (lat, lng)
    except Exception as e:
        logger.error(f"Error calculating centroid: {e}")
        return None


def reverse_geocode(lat, lng, api_key=None):
    """
    Reverse geocode a location using Google Maps API - EU optimized.

    Args:
        lat: Latitude
        lng: Longitude
        api_key: Google Maps API key (optional)

    Returns:
        Dictionary with location information including:
        - formatted_address: Full address
        - road_name: Name of the nearest road
        - locality: City or town
        - administrative_area: State or province
        - country: Country name
    """
    global EU_COUNTRY_CODE  # Add global declaration at the beginning of the function

    if api_key is None:
        api_key = GOOGLE_MAPS_API_KEY

    if not api_key:
        logger.warning("No Google Maps API key provided")
        return {
            "formatted_address": "Unknown Location",
            "road_name": None,
            "locality": None,
            "administrative_area": None,
            "country": None
        }

    # Check if we have a cached response for these coordinates
    cache_key = f"{lat:.6f},{lng:.6f}"
    if cache_key in api_cache:
        logger.debug(f"Using cached response for {cache_key}")
        return api_cache[cache_key]

    # Base URL for Google Maps Geocoding API
    url = "https://maps.googleapis.com/maps/api/geocode/json"

    # Parameters for the API request
    params = {
        "latlng": f"{lat},{lng}",
        "key": api_key,
        "region": EU_COUNTRY_CODE.lower(),  # Use EU region to prioritize European results
        "language": "en",  # Request results in English for consistency
        "result_type": "route|locality|administrative_area_level_1|country"
    }

    try:
        # Make the API request
        response = requests.get(url, params=params, timeout=30)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            if data["status"] == "OK":
                # Initialize result dictionary
                result = {
                    "formatted_address": "Unknown Location",
                    "road_name": None,
                    "locality": None,
                    "administrative_area": None,
                    "country": None
                }

                # Get the first result (most relevant)
                if data["results"]:
                    result["formatted_address"] = data["results"][0]["formatted_address"]

                    # Extract components
                    for component in data["results"][0]["address_components"]:
                        types = component["types"]

                        if "route" in types:
                            result["road_name"] = component["long_name"]
                        elif "locality" in types:
                            result["locality"] = component["long_name"]
                        elif "administrative_area_level_1" in types:
                            result["administrative_area"] = component["long_name"]
                        elif "country" in types:
                            result["country"] = component["long_name"]
                            # Set EU_COUNTRY_CODE if we detect the country
                            if result["country"] and len(result["country"]) >= 2:
                                for country_entry in [
                                    {"name": "Germany", "code": "DE"},
                                    {"name": "France", "code": "FR"},
                                    {"name": "Italy", "code": "IT"},
                                    {"name": "Spain", "code": "ES"},
                                    {"name": "Netherlands", "code": "NL"},
                                    {"name": "Belgium", "code": "BE"},
                                    {"name": "Sweden", "code": "SE"},
                                    {"name": "Poland", "code": "PL"},
                                    {"name": "Austria", "code": "AT"},
                                    {"name": "Denmark", "code": "DK"},
                                    {"name": "Finland", "code": "FI"},
                                    {"name": "Portugal", "code": "PT"},
                                    {"name": "Ireland", "code": "IE"},
                                    {"name": "Greece", "code": "GR"},
                                    {"name": "Czech Republic", "code": "CZ"},
                                    {"name": "Romania", "code": "RO"},
                                    {"name": "Hungary", "code": "HU"}
                                ]:
                                    if country_entry["name"] == result["country"]:
                                        EU_COUNTRY_CODE = country_entry["code"]
                                        logger.info(f"Detected country: {result['country']} (code: {EU_COUNTRY_CODE})")
                                        break

                # Cache the result
                api_cache[cache_key] = result

                # Save cache every 50 new entries
                if len(api_cache) % 50 == 0:
                    save_cache()

                return result
            else:
                logger.warning(f"API returned status: {data['status']}")
                return None
        else:
            logger.error(f"API request failed with status code {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"Error making API request: {e}")
        return None


def find_nearby_roads(lat, lng, radius=1000, api_key=None):
    """
    Find nearby roads using Google Maps Roads API - EU optimized.

    Args:
        lat: Latitude
        lng: Longitude
        radius: Search radius in meters (default: 1000)
        api_key: Google Maps API key (optional)

    Returns:
        List of road names, ordered by importance
    """
    if api_key is None:
        api_key = GOOGLE_MAPS_API_KEY

    if not api_key:
        logger.warning("No Google Maps API key provided")
        return []

    # Use reverse geocoding for now since Roads API has limitations
    location_info = reverse_geocode(lat, lng, api_key)

    if location_info and location_info["road_name"]:
        return [location_info["road_name"]]
    else:
        return []


def get_corridor_name(polygon, api_key=None):
    """
    Get a meaningful corridor name for a polygon using Google Maps API - EU optimized.

    Args:
        polygon: A shapely Polygon object
        api_key: Google Maps API key (optional)

    Returns:
        String with the corridor name
    """
    # Get centroid of the polygon
    centroid = get_polygon_centroid(polygon)

    if not centroid:
        return "Unknown Corridor"

    lat, lng = centroid

    # Get location information via reverse geocoding
    location_info = reverse_geocode(lat, lng, api_key)

    if not location_info:
        return "Unknown Corridor"

    # Find nearby roads
    roads = find_nearby_roads(lat, lng, api_key=api_key)

    # Format the corridor name - EU format
    if roads:
        road_name = roads[0]

        # Check for European motorway designation (e.g., "A1", "E55")
        is_motorway = any(char in road_name for char in ["A", "E", "M"]) and any(char.isdigit() for char in road_name)

        # If we have locality information, include it
        if location_info["locality"]:
            if is_motorway:
                corridor_name = f"Motorway {road_name} Corridor ({location_info['locality']})"
            else:
                corridor_name = f"{road_name} Corridor ({location_info['locality']})"
        elif location_info["administrative_area"]:
            if is_motorway:
                corridor_name = f"Motorway {road_name} Corridor ({location_info['administrative_area']})"
            else:
                corridor_name = f"{road_name} Corridor ({location_info['administrative_area']})"
        else:
            if is_motorway:
                corridor_name = f"Motorway {road_name} Corridor"
            else:
                corridor_name = f"{road_name} Corridor"
    else:
        # No road found, use locality or administrative area
        if location_info["locality"]:
            corridor_name = f"{location_info['locality']} Area"
        elif location_info["administrative_area"]:
            corridor_name = f"{location_info['administrative_area']} Area"
        else:
            corridor_name = "Unknown Corridor"

    return corridor_name


def batch_process_polygons(df, geometry_col='geometry', batch_size=50, api_key=None, save_interval=50):
    """
    Batch process polygons to get corridor names with EU optimization.

    Args:
        df: DataFrame with polygon geometries
        geometry_col: Name of the column containing polygon geometries
        batch_size: Number of requests to process before pausing
        api_key: Google Maps API key (optional)
        save_interval: Number of processed items before saving (default: 50)

    Returns:
        DataFrame with added 'corridor_name' column
    """
    # Load existing cache
    load_cache()

    # Create a copy of the DataFrame
    result_df = df.copy()

    # Create output path for intermediate saves
    output_file = 'ChargerLocationsRefined.csv'

    # Ensure the geometry column exists
    if geometry_col not in result_df.columns:
        logger.error(f"Geometry column '{geometry_col}' not found in DataFrame")
        return result_df

    # Add a corridor_name column if it doesn't exist
    if 'corridor_name' not in result_df.columns:
        result_df['corridor_name'] = None

    # Add a centroid column for caching
    result_df['_centroid'] = result_df[geometry_col].apply(get_polygon_centroid)

    # Process in batches
    total_rows = len(result_df)
    logger.info(f"Processing {total_rows} polygons to identify corridors")

    for i in tqdm(range(0, total_rows), desc="Geocoding polygons"):
        row = result_df.iloc[i]

        # Skip if already processed
        if pd.notna(row['corridor_name']):
            continue

        # Skip if the centroid is not valid
        if row['_centroid'] is None:
            result_df.at[i, 'corridor_name'] = "Unknown Corridor"
            continue

        # Get corridor name for this polygon
        polygon = row[geometry_col]
        corridor_name = get_corridor_name(polygon, api_key)

        # Save to dataframe
        result_df.at[i, 'corridor_name'] = corridor_name

        # Save progress at specified intervals
        if (i + 1) % save_interval == 0:
            logger.info(f"Processed {i + 1}/{total_rows} polygons")

            # Create a temporary copy without the centroid column for saving
            save_df = result_df.copy()
            if '_centroid' in save_df.columns:
                save_df = save_df.drop('_centroid', axis=1)

            # Save the intermediate results
            save_df.to_csv(output_file, index=False)
            logger.info(f"Saved intermediate results to {output_file} after processing {i + 1} rows")

            # Save cache
            save_cache()

            # Add delay after each batch to avoid hitting API rate limits
            if (i + 1) % batch_size == 0:
                time.sleep(2)  # 2-second delay between batches

    # Drop temporary centroid column
    result_df = result_df.drop('_centroid', axis=1)

    # Save cache one last time
    save_cache()

    # Save final results
    if '_centroid' in result_df.columns:
        result_df = result_df.drop('_centroid', axis=1)
    result_df.to_csv(output_file, index=False)
    logger.info(f"Saved final results to {output_file}")

    return result_df


def export_corridor_data(df, output_file='corridor_data_eu.json'):
    """
    Export corridor data to a JSON file for visualization.

    Args:
        df: DataFrame with polygon geometries and corridor names
        output_file: Path to output JSON file

    Returns:
        None
    """
    try:
        # Create a list of corridors with their centroids
        corridors = []

        for _, row in df.iterrows():
            if 'corridor_name' in row and pd.notna(row['corridor_name']):
                # Get centroid coordinates
                if 'geometry' in row and row['geometry'] is not None:
                    centroid = get_polygon_centroid(row['geometry'])

                    if centroid:
                        # Create corridor object
                        corridor = {
                            'name': row['corridor_name'],
                            'lat': centroid[0],
                            'lng': centroid[1]
                        }

                        # Add any other important information
                        for key in ['priority_score', 'truck_per_charger', 'energy_per_charger']:
                            if key in row and pd.notna(row[key]):
                                corridor[key] = float(row[key])

                        # Add region info for EU data
                        if 'corridor_name' in row and pd.notna(row['corridor_name']):
                            corridor_parts = row['corridor_name'].split('(')
                            if len(corridor_parts) > 1:
                                region = corridor_parts[1].replace(')', '').strip()
                                corridor['region'] = region

                        corridors.append(corridor)

        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(corridors, f, indent=2)

        logger.info(f"Exported {len(corridors)} corridors to {output_file}")

    except Exception as e:
        logger.error(f"Error exporting corridor data: {e}")


if __name__ == "__main__":
    # Example usage
    import geopandas as gpd

    # Set your API key here
    GOOGLE_MAPS_API_KEY = ""  # Add your API key

    if not GOOGLE_MAPS_API_KEY:
        print("Please add your Google Maps API key to use this script")
        exit(1)

    # Load test data
    try:
        print("Loading test data...")
        test_df = gpd.read_file("ChargerLocations.csv")
        print(f"Loaded {len(test_df)} polygons")

        # Process the polygons with EU optimization
        print("Processing polygons...")
        result_df = batch_process_polygons(test_df, api_key=GOOGLE_MAPS_API_KEY, save_interval=50)

        # Save results
        result_df.to_csv("ChargerLocationsWithCorridors.csv", index=False)
        print("Results saved to ChargerLocationsWithCorridors.csv")

        # Export for visualization
        export_corridor_data(result_df)
        print("Corridor data exported to corridor_data_eu.json")

    except Exception as e:
        print(f"Error: {e}")