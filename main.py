"""Main controller script to run all three steps in sequence"""

import os
import argparse
from step1_shapefile_to_csv import process_shapefile
from step2_analyze_corridors import analyze_corridors
from step3_identify_corridors import identify_transport_corridors


def main():
    parser = argparse.ArgumentParser(description='EV Charging Corridor Analysis Pipeline')
    parser.add_argument('--shapefile', type=str, help='Path to the shapefile')
    parser.add_argument('--min-distance', type=float, default=500,
                        help='Minimum distance between charging stations (km)')
    parser.add_argument('--steps', type=str, default='123',
                        help='Steps to run (e.g., "12" runs steps 1 and 2 only)')
    args = parser.parse_args()

    csv_path = 'ChargerLocations.csv'

    if '1' in args.steps:
        print("\n=== STEP 1: CONVERTING SHAPEFILE TO CSV ===\n")
        if not args.shapefile:
            shapefile_path = input("Enter the path to your .shp file: ")
        else:
            shapefile_path = args.shapefile

        if not os.path.exists(shapefile_path):
            print(f"File not found: {shapefile_path}")
            return

        gdf, csv_path = process_shapefile(shapefile_path)
        if gdf is None:
            print("Failed to process shapefile. Stopping.")
            return

    if '2' in args.steps:
        print("\n=== STEP 2: ANALYZING CORRIDORS ===\n")
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return

        df = analyze_corridors(csv_path)
        if df is None:
            print("Failed to analyze corridors. Stopping.")
            return

    if '3' in args.steps:
        print("\n=== STEP 3: IDENTIFYING TRANSPORT CORRIDORS ===\n")
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return

        identify_transport_corridors(min_distance=args.min_distance)

    print("\n=== ALL STEPS COMPLETED ===\n")


if __name__ == "__main__":
    main()