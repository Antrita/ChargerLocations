"""
Step 2. Sample data visualization of:
  1) Truck-to-charger ratio
  2) Fast/slow charger ratio
  3) Energy per charger
  4) Fast to slow charger ratio
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def analyze_corridors(csv_path='ChargerLocations.csv'):
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Analyzing {len(df)} locations for corridor prioritization")

    # Function to identify column names based on the field descriptions
    def identify_columns(dataframe):
        col_mapping = {
            'truck_col': None,  # DTN30/MainDTN - number of electrified trucks in 2030
            'fast_truck_col': None,  # MDTN_B - number of trucks using fast chargers
            'slow_truck_col': None,  # MDTN_R - number of trucks using slow chargers
            'fast_charger_col': None,  # NFCh30m - number of fast chargers
            'slow_charger_col': None,  # NSCh2pD - number of slow chargers
            'total_charger_col': None,  # TotCha - total number of chargers
            'fast_energy_col': None,  # ChEBM - charged energy with fast charging
            'slow_energy_col': None,  # ChERM - charged energy with slow charging
            'total_energy_col': None  # ChE30 - total charged energy
        }

        # Define patterns to look for in column names
        patterns = {
            'truck_col': ['dtn30', 'maindtn', 'electrified truck', 'total truck'],
            'fast_truck_col': ['mdtn_b', 'mdtnb', 'fast truck', 'break'],
            'slow_truck_col': ['mdtn_r', 'mdtnr', 'slow truck', 'rest'],
            'fast_charger_col': ['nfch', 'fast charger'],
            'slow_charger_col': ['nsch', 'slow charger'],
            'total_charger_col': ['totcha', 'total charger'],
            'fast_energy_col': ['chebm', 'fast energy', 'break energy'],
            'slow_energy_col': ['cherm', 'slow energy', 'rest energy'],
            'total_energy_col': ['che30', 'total energy']
        }

        # Match column names
        for col_type, search_patterns in patterns.items():
            for col in dataframe.columns:
                col_lower = col.lower()
                for pattern in search_patterns:
                    if pattern in col_lower:
                        col_mapping[col_type] = col
                        break
                if col_mapping[col_type]:
                    break

        # Print results
        print("\nIdentified columns:")
        for col_type, col_name in col_mapping.items():
            print(f"- {col_type}: {col_name}")

        return col_mapping

    # Identify the relevant columns
    col_mapping = identify_columns(df)

    # Calculate key metrics for each corridor
    print("\nCalculating corridor metrics...")

    # Initialize new columns for analysis (with error handling)
    try:
        # Truck-to-charger ratio
        if col_mapping['truck_col'] and col_mapping['total_charger_col']:
            df['truck_per_charger'] = df[col_mapping['truck_col']] / df[col_mapping['total_charger_col']]
            print("- Calculated truck-to-charger ratio")

        # Fast/slow charger ratio
        if col_mapping['fast_charger_col'] and col_mapping['slow_charger_col']:
            df['fast_slow_ratio'] = df[col_mapping['fast_charger_col']] / df[col_mapping['slow_charger_col']]
            print("- Calculated fast/slow charger ratio")

        # Energy per charger
        if col_mapping['total_energy_col'] and col_mapping['total_charger_col']:
            df['energy_per_charger'] = df[col_mapping['total_energy_col']] / df[col_mapping['total_charger_col']]
            print("- Calculated energy per charger")

        # Create priority score (if we have enough metrics)
        score_components = []

        if 'truck_per_charger' in df.columns:
            df['truck_per_charger_norm'] = (df['truck_per_charger'] - df['truck_per_charger'].min()) / \
                                           (df['truck_per_charger'].max() - df['truck_per_charger'].min())
            score_components.append(df['truck_per_charger_norm'] * 0.5)  # 50% weight

        if 'energy_per_charger' in df.columns:
            df['energy_per_charger_norm'] = (df['energy_per_charger'] - df['energy_per_charger'].min()) / \
                                            (df['energy_per_charger'].max() - df['energy_per_charger'].min())
            score_components.append(df['energy_per_charger_norm'] * 0.3)  # 30% weight

        if 'fast_slow_ratio' in df.columns:
            df['fast_slow_ratio_norm'] = (df['fast_slow_ratio'] - df['fast_slow_ratio'].min()) / \
                                         (df['fast_slow_ratio'].max() - df['fast_slow_ratio'].min())
            score_components.append(df['fast_slow_ratio_norm'] * 0.2)  # 20% weight

        if score_components:
            df['priority_score'] = sum(score_components)
            print("- Created priority score based on available metrics")

            # Sort by priority score
            prioritized = df.sort_values('priority_score', ascending=False)

            # Display top corridors
            print("\n=== Top 5 Priority Corridors for New Charging Infrastructure ===")
            top_corridors = prioritized.head(5)

            for i, (idx, row) in enumerate(top_corridors.iterrows(), 1):
                print(f"\nPriority #{i} - Score: {row.get('priority_score', 'N/A'):.2f}")

                # Display available metrics
                if col_mapping['truck_col']:
                    print(f"  E-Trucks: {int(row[col_mapping['truck_col']])}")

                if 'truck_per_charger' in df.columns:
                    print(f"  Truck/Charger Ratio: {row['truck_per_charger']:.2f}")

                if col_mapping['fast_charger_col']:
                    print(f"  Fast Chargers: {int(row[col_mapping['fast_charger_col']])}")

                if col_mapping['slow_charger_col']:
                    print(f"  Slow Chargers: {int(row[col_mapping['slow_charger_col']])}")

                if col_mapping['total_energy_col']:
                    print(f"  Energy Consumption: {row[col_mapping['total_energy_col']]:.2f} MWh")

            # Create visualizations
            print("\nGenerating visualizations...")

            # Visualization 1: Truck-to-Charger Ratio
            if 'truck_per_charger' in df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(prioritized)), prioritized['truck_per_charger'], alpha=0.7)
                plt.axhline(y=prioritized['truck_per_charger'].mean(), color='r', linestyle='--', label='Average')
                plt.title('Truck-to-Charger Ratio by Location (Higher = More Infrastructure Needed)')
                plt.ylabel('Trucks per Charger')
                plt.xlabel('Location (sorted by priority)')
                plt.legend()
                plt.tight_layout()
                plt.savefig('truck_per_charger.png')
                print("- Created truck-per-charger visualization")

            # Visualization 2: Fast vs Slow Charger Distribution in Top Corridors
            if col_mapping['fast_charger_col'] and col_mapping['slow_charger_col']:
                plt.figure(figsize=(10, 5))

                top5 = prioritized.head(5)
                x = np.arange(len(top5))
                width = 0.35

                plt.bar(x - width / 2, top5[col_mapping['fast_charger_col']], width, label='Fast Chargers')
                plt.bar(x + width / 2, top5[col_mapping['slow_charger_col']], width, label='Slow Chargers')

                plt.xlabel('Priority Corridors')
                plt.ylabel('Number of Chargers')
                plt.title('Fast vs Slow Charger Distribution in Top Priority Corridors')
                plt.xticks(x, [f'Corridor {i + 1}' for i in range(len(top5))])
                plt.legend()
                plt.tight_layout()
                plt.savefig('charger_distribution.png')
                print("- Created charger distribution visualization")

            # Save recommendations to CSV
            recommendations = prioritized.head(10)
            recommendations.to_csv('corridor_recommendations.csv', index=False)
            print("\nSaved top 10 corridor recommendations to CSV")

    except Exception as e:
        print(f"\nError in analysis: {e}")
        print("Check your data to ensure it contains the expected columns and formats.")

    return df


if __name__ == "__main__":
    analyze_corridors()