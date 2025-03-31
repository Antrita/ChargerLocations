"""
Step 1. Preparing the data we need by converting .shp to csv as our first step.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

print("Libraries imported successfully")


def process_shapefile(shapefile_path):
    # Read the shapefile with geopandas
    try:
        gdf = gpd.read_file(shapefile_path)
        print("\nSuccessfully read shapefile")
        print(f"Shapefile contains {len(gdf)} features")

        # Display the column names
        print("\nColumn names in the shapefile:")
        print(gdf.columns.tolist())

        # Display the first few rows
        print("\nPreview of the data:")
        print(gdf.head())

        # Convert to CSV
        csv_path = 'ChargerLocations.csv'
        gdf.to_csv(csv_path, index=False)
        print(f"\nSuccessfully converted shapefile to CSV: {csv_path}")

        # Show basic statistics
        numeric_columns = gdf.select_dtypes(include=['number']).columns
        print("\nBasic statistics for numeric columns:")
        print(gdf[numeric_columns].describe())

        # Create a simple visualization - map of the features
        try:
            plt.figure(figsize=(12, 8))
            gdf.plot()
            plt.title('Map of Charger Locations')
            plt.savefig('charger_map.png')
            print("\nCreated map visualization and saved to charger_map.png")
        except Exception as e:
            print(f"Error creating map: {e}")

        return gdf, csv_path

    except Exception as e:
        print(f"Error reading shapefile: {e}")
        print("\nTroubleshooting tips:")
        print("Make sure you have all required shapefile components (.shp, .dbf, .shx, .prj)")
        return None, None


if __name__ == "__main__":
    # Get shapefile path from user
    shapefile_path = input("Enter the path to your .shp file: ")

    if not os.path.exists(shapefile_path):
        print(f"File not found: {shapefile_path}")
    elif not shapefile_path.endswith('.shp'):
        print("Please specify a .shp file")
    else:
        gdf, csv_path = process_shapefile(shapefile_path)