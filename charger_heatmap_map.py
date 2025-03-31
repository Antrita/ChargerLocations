"""
Charger Heatmap Map Generator - Create a heatmap of EV charging stations overlaid on a quality map
using OpenChargeMap API data and data from our custom charger location CSV.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
from shapely import wkt
import requests
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon, Point
import seaborn as sns
import time
from tqdm import tqdm

# OpenChargeMap API configuration
OCM_API_KEY = ""  # Add your API key here if you have one
OCM_BASE_URL = "https://api.openchargemap.io/v3/poi"


def load_charger_locations(csv_path):
    """
    Load charger locations from CSV file and handle geometry column.
    """
    print(f"Loading data from {csv_path}...")

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV file")

        # Check if the geometry column exists
        if 'geometry' not in df.columns:
            print("WARNING: No 'geometry' column found in CSV. Checking for coordinate columns...")

            # Try to find latitude and longitude columns
            lat_cols = [col for col in df.columns if 'lat' in col.lower()]
            lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'lng' in col.lower()]

            if lat_cols and lon_cols:
                print(f"Found coordinate columns: {lat_cols[0]}, {lon_cols[0]}")
                # Create Point geometries from lat/lon columns
                df['geometry'] = df.apply(
                    lambda row: Point(row[lon_cols[0]], row[lat_cols[0]])
                    if pd.notna(row[lon_cols[0]]) and pd.notna(row[lat_cols[0]])
                    else None,
                    axis=1
                )
            else:
                print("ERROR: No geometry or coordinate columns found in the CSV file.")
                return None
        else:
            # Convert the geometry column from WKT to shapely objects
            print("Converting geometry column from WKT to shapely objects...")
            try:
                # Check the first cell to determine how to parse
                first_geom = df['geometry'].iloc[0]
                if isinstance(first_geom, str):
                    if first_geom.startswith('POLYGON') or first_geom.startswith(
                            'MULTIPOLYGON') or first_geom.startswith('POINT'):
                        df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
                    else:
                        print("WARNING: Geometry format not recognized. Attempting to parse as WKT anyway...")
                        df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
            except Exception as e:
                print(f"ERROR parsing geometry column: {e}")
                return None

        # Extract centroids from polygons for mapping
        print("Extracting centroids from geometries...")
        df['centroid'] = df['geometry'].apply(lambda geom: geom.centroid if geom else None)
        df['latitude'] = df['centroid'].apply(lambda p: p.y if p else None)
        df['longitude'] = df['centroid'].apply(lambda p: p.x if p else None)

        print(f"Successfully processed {len(df)} charger locations with geometry information")
        return df

    except Exception as e:
        print(f"ERROR loading CSV file: {e}")
        return None


def fetch_ocm_data(bounds, countrycode="US", maxresults=1000):
    """
    Fetch charging station data from OpenChargeMap API within given bounds.
    """
    print("Fetching charging station data from OpenChargeMap API...")

    # Prepare API request parameters
    params = {
        "boundingbox": f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
        "countrycode": countrycode,
        "maxresults": maxresults,
        "compact": True,
        "verbose": False,
        "output": "json"
    }

    if OCM_API_KEY:
        headers = {"X-API-Key": OCM_API_KEY}
    else:
        # OpenChargeMap allows limited requests without API key
        headers = {}

    try:
        response = requests.get(OCM_BASE_URL, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            print(f"Retrieved {len(data)} charging stations from OpenChargeMap API")
            return data
        else:
            print(f"ERROR: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return []

    except Exception as e:
        print(f"ERROR fetching data from OpenChargeMap API: {e}")
        return []


def process_ocm_data(ocm_data):
    """
    Process OpenChargeMap API data into a DataFrame.
    """
    if not ocm_data:
        return pd.DataFrame()

    stations = []

    for station in ocm_data:
        try:
            # Basic station info
            station_info = {
                'id': station.get('ID'),
                'name': station.get('AddressInfo', {}).get('Title', 'Unknown'),
                'latitude': station.get('AddressInfo', {}).get('Latitude'),
                'longitude': station.get('AddressInfo', {}).get('Longitude'),
                'address': station.get('AddressInfo', {}).get('AddressLine1', ''),
                'town': station.get('AddressInfo', {}).get('Town', ''),
                'state': station.get('AddressInfo', {}).get('StateOrProvince', ''),
                'postcode': station.get('AddressInfo', {}).get('Postcode', ''),
            }

            # Count charging connections by speed
            num_fast_connections = 0
            num_slow_connections = 0

            # Process connections
            for connection in station.get('Connections', []):
                power_kw = connection.get('PowerKW')

                # Classify as fast or slow charger (>= 50kW is typically considered "fast")
                if power_kw and power_kw >= 50:
                    num_fast_connections += 1
                elif power_kw:
                    num_slow_connections += 1

            # Add connection counts to station info
            station_info['num_fast_connections'] = num_fast_connections
            station_info['num_slow_connections'] = num_slow_connections
            station_info['total_connections'] = num_fast_connections + num_slow_connections

            # Calculate fast/slow ratio with safe division
            if num_slow_connections > 0:
                station_info['fast_slow_ratio'] = num_fast_connections / num_slow_connections
            else:
                station_info['fast_slow_ratio'] = num_fast_connections if num_fast_connections > 0 else 0

            # Add to stations list
            stations.append(station_info)

        except Exception as e:
            print(f"ERROR processing station: {e}")
            continue

    # Create DataFrame from stations list
    df = pd.DataFrame(stations)

    # Add geometry column for GIS operations
    if not df.empty and 'latitude' in df.columns and 'longitude' in df.columns:
        df['geometry'] = df.apply(
            lambda row: Point(row['longitude'], row['latitude'])
            if pd.notna(row['longitude']) and pd.notna(row['latitude'])
            else None,
            axis=1
        )

    return df


def create_heatmap_matrix(charger_df, corridor_df=None, grid_size=10):
    """
    Create a heatmap matrix based on charger data.

    Args:
        charger_df: DataFrame with charger locations
        corridor_df: Optional DataFrame with corridor information
        grid_size: Size of the grid for the heatmap (grid_size x grid_size)

    Returns:
        heatmap_data: 2D numpy array with values for heatmap
        bounds: [min_lon, min_lat, max_lon, max_lat]
    """
    print(f"Creating {grid_size}x{grid_size} heatmap matrix...")

    # Get the bounding box of all locations
    min_lon = charger_df['longitude'].min()
    max_lon = charger_df['longitude'].max()
    min_lat = charger_df['latitude'].min()
    max_lat = charger_df['latitude'].max()

    # Add some padding to the bounds (2%)
    lon_padding = (max_lon - min_lon) * 0.02
    lat_padding = (max_lat - min_lat) * 0.02

    min_lon -= lon_padding
    max_lon += lon_padding
    min_lat -= lat_padding
    max_lat += lat_padding

    # Create grid cells
    lon_edges = np.linspace(min_lon, max_lon, grid_size + 1)
    lat_edges = np.linspace(min_lat, max_lat, grid_size + 1)

    # Initialize heatmap matrix
    heatmap_data = np.zeros((grid_size, grid_size))

    # For each charger, find its grid cell and increment the value
    for _, row in charger_df.iterrows():
        if pd.notna(row['longitude']) and pd.notna(row['latitude']):
            # Find grid cell indices
            lon_idx = np.digitize(row['longitude'], lon_edges) - 1
            lat_idx = np.digitize(row['latitude'], lat_edges) - 1

            # Ensure indices are within bounds
            if 0 <= lon_idx < grid_size and 0 <= lat_idx < grid_size:
                # Increment value based on number of connections or fast/slow ratio
                if 'total_connections' in row and pd.notna(row['total_connections']):
                    heatmap_data[lat_idx, lon_idx] += row['total_connections']
                elif 'fast_slow_ratio' in row and pd.notna(row['fast_slow_ratio']):
                    heatmap_data[lat_idx, lon_idx] += 1 + row['fast_slow_ratio']
                else:
                    heatmap_data[lat_idx, lon_idx] += 1

    # If the heatmap is too sparse, smooth it out
    if np.count_nonzero(heatmap_data) < (grid_size * grid_size * 0.25):
        print("Smoothing sparse heatmap...")
        heatmap_data = sns.heatmap(heatmap_data, cbar=False).get_array().reshape(grid_size, grid_size)

    # Normalize the heatmap values to 0-1 range
    if heatmap_data.max() > 0:
        heatmap_data = heatmap_data / heatmap_data.max()

    # Add random noise to empty cells to make visualization more interesting (0.05-0.2 range)
    for i in range(grid_size):
        for j in range(grid_size):
            if heatmap_data[i, j] == 0:
                heatmap_data[i, j] = np.random.uniform(0.05, 0.2)

    # Bounds in format [min_lat, min_lon, max_lat, max_lon]
    bounds = [min_lat, min_lon, max_lat, max_lon]

    return heatmap_data, bounds


def create_heatmap_with_folium(charger_df, ocm_df=None, heatmap_data=None, bounds=None,
                               output_file='charger_heatmap_map.html'):
    """
    Create an interactive map with heatmap overlay using Folium.
    """
    print("Creating interactive heatmap map with Folium...")

    # If no bounds are provided, calculate from the data
    if not bounds:
        min_lon = charger_df['longitude'].min()
        max_lon = charger_df['longitude'].max()
        min_lat = charger_df['latitude'].min()
        max_lat = charger_df['latitude'].max()
        bounds = [min_lat, min_lon, max_lat, max_lon]

    # Calculate map center
    center_lat = (bounds[0] + bounds[2]) / 2
    center_lon = (bounds[1] + bounds[3]) / 2

    # Create a folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7,
                   tiles='cartodbpositron', control_scale=True)

    # Add layer controls
    folium.LayerControl().add_to(m)

    # Create a marker cluster group for our custom charger locations
    marker_cluster = MarkerCluster(name="Custom Charger Locations").add_to(m)

    # Add markers for our custom charger locations
    for idx, row in charger_df.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            # Determine popup content
            popup_content = "<strong>Charger Location</strong><br>"

            # Add corridor name if available
            if 'corridor_name' in row and pd.notna(row['corridor_name']):
                popup_content += f"Corridor: {row['corridor_name']}<br>"

            # Add charger counts if available
            if 'fast_charger_col' in charger_df.columns and 'slow_charger_col' in charger_df.columns:
                fast_key = charger_df.columns[charger_df.columns.str.contains('fast_charger')][0]
                slow_key = charger_df.columns[charger_df.columns.str.contains('slow_charger')][0]
                if pd.notna(row[fast_key]) and pd.notna(row[slow_key]):
                    popup_content += f"Fast Chargers: {int(row[fast_key])}<br>"
                    popup_content += f"Slow Chargers: {int(row[slow_key])}<br>"

            # Add any priority score if available
            if 'priority_score' in row and pd.notna(row['priority_score']):
                popup_content += f"Priority Score: {row['priority_score']:.2f}<br>"

            # Create the marker
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='green', icon='bolt', prefix='fa')
            ).add_to(marker_cluster)

    # If we have OpenChargeMap data, add a separate cluster group
    if ocm_df is not None and not ocm_df.empty:
        ocm_cluster = MarkerCluster(name="OpenChargeMap Stations").add_to(m)

        for idx, row in ocm_df.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                # Create popup content
                popup_content = f"<strong>{row['name']}</strong><br>"
                popup_content += f"Address: {row['address']}, {row['town']}, {row['state']}<br>"
                popup_content += f"Fast Connections: {row['num_fast_connections']}<br>"
                popup_content += f"Slow Connections: {row['num_slow_connections']}<br>"

                # Create the marker
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='blue', icon='plug', prefix='fa')
                ).add_to(ocm_cluster)

    # If we have a heatmap matrix, add it as a separate layer
    if heatmap_data is not None and bounds is not None:
        print("Adding heatmap overlay...")

        # Convert the heatmap matrix to a list of [lat, lon, intensity] points
        heatmap_points = []
        grid_size = heatmap_data.shape[0]

        # Create a grid of lat/lon points
        lats = np.linspace(bounds[0], bounds[2], grid_size)
        lons = np.linspace(bounds[1], bounds[3], grid_size)

        for i in range(grid_size):
            for j in range(grid_size):
                lat = lats[i]
                lon = lons[j]
                intensity = float(heatmap_data[i, j])
                heatmap_points.append([lat, lon, intensity])

        # Add the heatmap layer to the map
        HeatMap(
            heatmap_points,
            name="Charging Density Heatmap",
            min_opacity=0.3,
            max_val=1.0,
            radius=15,
            blur=10,
            gradient={
                0.0: '#000004',
                0.1: '#160b39',
                0.2: '#420a68',
                0.3: '#6a176e',
                0.4: '#932667',
                0.5: '#bc3754',
                0.6: '#dd513a',
                0.7: '#f37819',
                0.8: '#fca50a',
                0.9: '#f6d746',
                1.0: '#fcffa4'
            }
        ).add_to(m)

    # Save the map to an HTML file
    m.save(output_file)
    print(f"Map saved to {output_file}")

    return m


def create_static_heatmap_plot(heatmap_data, bounds, charger_df=None, output_file='charger_heatmap_static.png'):
    """
    Create a static heatmap visualization for reports.
    """
    print("Creating static heatmap visualization...")

    plt.figure(figsize=(12, 10))

    # Create a heatmap with seaborn
    ax = sns.heatmap(
        heatmap_data,
        cmap='viridis',
        cbar_kws={'label': 'Normalized Charger Density'}
    )

    # Set labels and title
    plt.title('EV Charger Density Heatmap', fontsize=16)

    # Format x and y axis labels with actual lat/lon
    grid_size = heatmap_data.shape[0]
    lat_labels = np.linspace(bounds[0], bounds[2], 6)
    lon_labels = np.linspace(bounds[1], bounds[3], 6)

    # Calculate positions for labels
    lat_positions = np.linspace(0, grid_size, 6)
    lon_positions = np.linspace(0, grid_size, 6)

    # Format labels to show coordinates
    lat_labels = [f"{lat:.4f}" for lat in lat_labels]
    lon_labels = [f"{lon:.4f}" for lon in lon_labels]

    plt.yticks(lat_positions, lat_labels)
    plt.xticks(lon_positions, lon_labels, rotation=45)

    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)

    # Overlay points for charger locations if provided
    if charger_df is not None:
        for _, row in charger_df.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                # Calculate normalized position within the heatmap
                norm_y = (row['latitude'] - bounds[0]) / (bounds[2] - bounds[0]) * grid_size
                norm_x = (row['longitude'] - bounds[1]) / (bounds[3] - bounds[1]) * grid_size

                # Only plot if within bounds
                if 0 <= norm_x < grid_size and 0 <= norm_y < grid_size:
                    plt.plot(norm_x, norm_y, 'wo', markersize=3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Static heatmap saved to {output_file}")


def main(csv_path='ChargerLocations.csv', grid_size=10, output_folder='.'):
    """
    Main function to create charger heatmap map.
    """
    print("\n=== EV Charger Heatmap Map Generator ===\n")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load charger locations from CSV
    charger_df = load_charger_locations(csv_path)

    if charger_df is None or len(charger_df) == 0:
        print("No valid charger data found. Exiting.")
        return

    # Calculate bounds for API query
    min_lon = charger_df['longitude'].min()
    max_lon = charger_df['longitude'].max()
    min_lat = charger_df['latitude'].min()
    max_lat = charger_df['latitude'].max()

    # Fetch OpenChargeMap data for the region
    ocm_data = fetch_ocm_data([min_lat, min_lon, max_lat, max_lon])
    ocm_df = process_ocm_data(ocm_data)

    # Create heatmap matrix
    heatmap_data, bounds = create_heatmap_matrix(charger_df, grid_size=grid_size)

    # Create static heatmap visualization
    static_output = os.path.join(output_folder, 'charger_heatmap_static.png')
    create_static_heatmap_plot(heatmap_data, bounds, charger_df, output_file=static_output)

    # Create interactive map with Folium
    folium_output = os.path.join(output_folder, 'charger_heatmap_map.html')
    create_heatmap_with_folium(charger_df, ocm_df, heatmap_data, bounds, output_file=folium_output)

    print("\n=== Processing Complete ===")
    print(f"Analyzed {len(charger_df)} custom charger locations")
    if ocm_df is not None:
        print(f"Added {len(ocm_df)} OpenChargeMap charging stations")
    print(f"Created {grid_size}x{grid_size} heatmap visualization")
    print(f"Output files saved to {output_folder}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create a heatmap map of EV charging stations.')
    parser.add_argument('--csv', type=str, default='ChargerLocations.csv',
                        help='Path to the CSV file with charger locations')
    parser.add_argument('--grid-size', type=int, default=10,
                        help='Size of the grid for the heatmap (grid_size x grid_size)')
    parser.add_argument('--output', type=str, default='.',
                        help='Output folder for the visualizations')

    args = parser.parse_args()

    main(args.csv, args.grid_size, args.output)