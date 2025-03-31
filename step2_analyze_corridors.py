"""
Step 2. Enhanced data visualization of:
  1) Truck-to-charger ratio
  2) Fast/slow charger ratio
  3) Energy per charger
  4) Fast to slow charger ratio
  5) Heatmap of slow vs fast charger traffic
  6) Google Maps API integration for corridor identification
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import sys
import subprocess
import time
from shapely import wkt
from tqdm import tqdm

# Try to import our custom Google Maps utilities
try:
    from google_maps_utils import batch_process_polygons, export_corridor_data

    GOOGLE_MAPS_AVAILABLE = True
    print("Google Maps utilities available for corridor identification")
except ImportError:
    GOOGLE_MAPS_AVAILABLE = False
    print("Google Maps utilities not available. Will use basic corridor identification.")

# Check for plotly and install if missing
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Plotly is already installed. Will use interactive visualizations.")
except ImportError:
    print("Plotly is not installed. Attempting to install it now...")

    try:
        # Install plotly package
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        print("Plotly installed successfully!")

        # Import plotly modules after installation
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        print("Plotly imported successfully. Will use interactive visualizations.")
    except Exception as e:
        print(f"Failed to install Plotly: {e}")
        print("Please install Plotly manually with: pip install plotly")
        sys.exit(1)  # Exit with error code


def analyze_corridors(csv_path='ChargerLocations.csv', use_google_maps=False, google_maps_api_key=None):
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Analyzing {len(df)} locations for corridor prioritization")

        # Process geometry column if it exists
        if 'geometry' in df.columns:
            print("Processing geometry column...")
            try:
                # Try to convert WKT strings to geometry objects
                df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
                print("Geometry column processed successfully")

                # Use Google Maps API for corridor identification if requested
                if use_google_maps and GOOGLE_MAPS_AVAILABLE:
                    print("\n=== Using Google Maps API for corridor identification ===")
                    if google_maps_api_key:
                        # Update the API key in the google_maps_utils module
                        import google_maps_utils
                        google_maps_utils.GOOGLE_MAPS_API_KEY = google_maps_api_key

                    # Process polygons to get corridor names
                    print("Processing polygons to identify corridors...")
                    df = batch_process_polygons(df, geometry_col='geometry', batch_size=10)
                    print(f"Identified {df['corridor_name'].nunique()} unique corridors")

                    # Export corridor data for visualization
                    export_corridor_data(df, output_file='corridor_data.json')
            except Exception as geo_error:
                print(f"Warning: Could not process geometry column: {geo_error}")
                print("Continuing without geometry processing")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

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

    # Initialize new columns for analysis (with robust error handling)
    try:
        # Truck-to-charger ratio
        if col_mapping['truck_col'] and col_mapping['total_charger_col']:
            # Ensure we don't divide by zero
            df['truck_per_charger'] = df.apply(
                lambda row: row[col_mapping['truck_col']] / row[col_mapping['total_charger_col']]
                if row[col_mapping['total_charger_col']] > 0 else 0,
                axis=1
            )
            print("- Calculated truck-to-charger ratio")

        # Fast/slow charger ratio
        if col_mapping['fast_charger_col'] and col_mapping['slow_charger_col']:
            # Ensure we don't divide by zero
            df['fast_slow_ratio'] = df.apply(
                lambda row: row[col_mapping['fast_charger_col']] / row[col_mapping['slow_charger_col']]
                if row[col_mapping['slow_charger_col']] > 0 else 0,
                axis=1
            )
            print("- Calculated fast/slow charger ratio")

        # Energy per charger
        if col_mapping['total_energy_col'] and col_mapping['total_charger_col']:
            # Ensure we don't divide by zero
            df['energy_per_charger'] = df.apply(
                lambda row: row[col_mapping['total_energy_col']] / row[col_mapping['total_charger_col']]
                if row[col_mapping['total_charger_col']] > 0 else 0,
                axis=1
            )
            print("- Calculated energy per charger")

        # Multiple approaches to calculate truck traffic metrics - using safer methods

        # SOLUTION 1: Calculate truck traffic directly using available columns
        # Calculate this only if both columns exist
        if col_mapping['fast_truck_col'] and col_mapping['fast_charger_col']:
            # Ensure we don't divide by zero
            df['fast_traffic'] = df.apply(
                lambda row: row[col_mapping['fast_truck_col']]
                if pd.notna(row[col_mapping['fast_truck_col']]) else 0,
                axis=1
            )
            # If charger column exists, calculate truck per charger ratio
            df['fast_truck_per_charger'] = df.apply(
                lambda row: row[col_mapping['fast_truck_col']] / row[col_mapping['fast_charger_col']]
                if row[col_mapping['fast_charger_col']] > 0 and pd.notna(row[col_mapping['fast_truck_col']]) else 0,
                axis=1
            )
            print("- Calculated fast truck traffic metrics")

        if col_mapping['slow_truck_col'] and col_mapping['slow_charger_col']:
            # Ensure we don't divide by zero
            df['slow_traffic'] = df.apply(
                lambda row: row[col_mapping['slow_truck_col']]
                if pd.notna(row[col_mapping['slow_truck_col']]) else 0,
                axis=1
            )
            # If charger column exists, calculate truck per charger ratio
            df['slow_truck_per_charger'] = df.apply(
                lambda row: row[col_mapping['slow_truck_col']] / row[col_mapping['slow_charger_col']]
                if row[col_mapping['slow_charger_col']] > 0 and pd.notna(row[col_mapping['slow_truck_col']]) else 0,
                axis=1
            )
            print("- Calculated slow truck traffic metrics")

        # SOLUTION 2: If direct truck columns don't exist, try deriving from other metrics
        if (not col_mapping['fast_truck_col'] or not col_mapping['slow_truck_col']) and col_mapping['truck_col']:
            # Check if we can estimate the split using energy consumption
            if col_mapping['fast_energy_col'] and col_mapping['slow_energy_col'] and col_mapping['total_energy_col']:
                # Calculate the proportion of energy used by fast/slow charging
                df['fast_energy_ratio'] = df.apply(
                    lambda row: row[col_mapping['fast_energy_col']] / row[col_mapping['total_energy_col']]
                    if row[col_mapping['total_energy_col']] > 0 else 0.5,
                    axis=1
                )
                df['slow_energy_ratio'] = df.apply(
                    lambda row: row[col_mapping['slow_energy_col']] / row[col_mapping['total_energy_col']]
                    if row[col_mapping['total_energy_col']] > 0 else 0.5,
                    axis=1
                )

                # Estimate truck counts using energy ratios
                df['fast_traffic_est'] = df[col_mapping['truck_col']] * df['fast_energy_ratio']
                df['slow_traffic_est'] = df[col_mapping['truck_col']] * df['slow_energy_ratio']

                # Calculate per-charger metrics if available
                if col_mapping['fast_charger_col']:
                    df['fast_truck_per_charger'] = df.apply(
                        lambda row: row['fast_traffic_est'] / row[col_mapping['fast_charger_col']]
                        if row[col_mapping['fast_charger_col']] > 0 else 0,
                        axis=1
                    )

                if col_mapping['slow_charger_col']:
                    df['slow_truck_per_charger'] = df.apply(
                        lambda row: row['slow_traffic_est'] / row[col_mapping['slow_charger_col']]
                        if row[col_mapping['slow_charger_col']] > 0 else 0,
                        axis=1
                    )

                print("- Estimated fast/slow truck traffic using energy consumption ratios")

        # SOLUTION 3: If all else fails, create placeholder columns with zeros for visualization
        if 'fast_traffic' not in df.columns and 'fast_traffic_est' not in df.columns:
            df['fast_traffic'] = 0
            if col_mapping['fast_charger_col']:
                df['fast_truck_per_charger'] = 0
            print("- Created placeholder for fast truck traffic (no data available)")

        if 'slow_traffic' not in df.columns and 'slow_traffic_est' not in df.columns:
            df['slow_traffic'] = 0
            if col_mapping['slow_charger_col']:
                df['slow_truck_per_charger'] = 0
            print("- Created placeholder for slow truck traffic (no data available)")

        # Create priority score (if we have enough metrics)
        score_components = []

        if 'truck_per_charger' in df.columns:
            # Normalize safely with handling for min==max edge case
            min_val = df['truck_per_charger'].min()
            max_val = df['truck_per_charger'].max()
            if min_val == max_val or max_val - min_val < 0.001:
                df['truck_per_charger_norm'] = 0.5  # Set to middle value if all values are the same
            else:
                df['truck_per_charger_norm'] = (df['truck_per_charger'] - min_val) / (max_val - min_val)
            score_components.append(df['truck_per_charger_norm'] * 0.5)  # 50% weight

        if 'energy_per_charger' in df.columns:
            # Normalize safely with handling for min==max edge case
            min_val = df['energy_per_charger'].min()
            max_val = df['energy_per_charger'].max()
            if min_val == max_val or max_val - min_val < 0.001:
                df['energy_per_charger_norm'] = 0.5  # Set to middle value if all values are the same
            else:
                df['energy_per_charger_norm'] = (df['energy_per_charger'] - min_val) / (max_val - min_val)
            score_components.append(df['energy_per_charger_norm'] * 0.3)  # 30% weight

        if 'fast_slow_ratio' in df.columns:
            # Normalize safely with handling for min==max edge case
            min_val = df['fast_slow_ratio'].min()
            max_val = df['fast_slow_ratio'].max()
            if min_val == max_val or max_val - min_val < 0.001:
                df['fast_slow_ratio_norm'] = 0.5  # Set to middle value if all values are the same
            else:
                df['fast_slow_ratio_norm'] = (df['fast_slow_ratio'] - min_val) / (max_val - min_val)
            score_components.append(df['fast_slow_ratio_norm'] * 0.2)  # 20% weight

        if score_components:
            df['priority_score'] = sum(score_components)
            print("- Created priority score based on available metrics")

            # Sort by priority score
            prioritized = df.sort_values('priority_score', ascending=False).reset_index(drop=True)

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


fig.write_html('truck_per_charger_interactive.html')

# Also save as static image for reports
fig.write_image('truck_per_charger.png', scale=2)

print("- Created truck-per-charger visualization (interactive HTML and static PNG)")
except Exception as vis_error:
print(f"Error creating truck-per-charger visualization: {vis_error}")

# Visualization 2: Fast vs Slow Charger Distribution in Top Corridors using Plotly
if col_mapping['fast_charger_col'] and col_mapping['slow_charger_col']:
    try:
        # Get top 10 corridors
        top10 = prioritized.head(10)

        # Create labels for corridors
        if 'corridor_name' in top10.columns:
            labels = [f"{i + 1}. {name}" if isinstance(name, str) else f"Corridor {i + 1}"
                      for i, name in enumerate(top10['corridor_name'])]
        else:
            labels = [f"Corridor {i + 1}" for i in range(len(top10))]

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add fast chargers bar
        fig.add_trace(
            go.Bar(
                x=labels,
                y=top10[col_mapping['fast_charger_col']],
                name="Fast Chargers",
                marker_color='rgba(46, 134, 193, 0.8)',
                marker_line_color='rgba(46, 134, 193, 1.0)',
                marker_line_width=1.5
            ),
            secondary_y=False,
        )

        # Add slow chargers bar
        fig.add_trace(
            go.Bar(
                x=labels,
                y=top10[col_mapping['slow_charger_col']],
                name="Slow Chargers",
                marker_color='rgba(241, 196, 15, 0.8)',
                marker_line_color='rgba(241, 196, 15, 1.0)',
                marker_line_width=1.5
            ),
            secondary_y=False,
        )

        # Add ratio line if we have fast/slow ratio
        if 'fast_slow_ratio' in top10.columns:
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=top10['fast_slow_ratio'],
                    name="Fast/Slow Ratio",
                    mode='lines+markers',
                    marker=dict(color='rgba(231, 76, 60, 1.0)', size=8),
                    line=dict(color='rgba(231, 76, 60, 0.8)', width=2)
                ),
                secondary_y=True,
            )

        # Update layout
        fig.update_layout(
            title='Fast vs Slow Charger Distribution in Top Priority Corridors',
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=600,
            width=1000,
            template="plotly_white",
        )

        # Set axes titles
        fig.update_yaxes(title_text="Number of Chargers", secondary_y=False)
        if 'fast_slow_ratio' in top10.columns:
            fig.update_yaxes(title_text="Fast/Slow Ratio", secondary_y=True)

        # Save as HTML for interactive viewing
        fig.write_html('charger_distribution_interactive.html')

        # Also save as static image for reports
        fig.write_image('charger_distribution.png', scale=2)

        print("- Created charger distribution visualization (interactive HTML and static PNG)")
    except Exception as vis_error:
        print(f"Error creating charger distribution visualization: {vis_error}")

    # Visualization 3: NEW HEATMAP showing Fast vs Slow Charger Traffic
print("- Generating Slow vs Fast Charger Traffic Heatmap...")
try:
    # Prepare data for heatmap - top 10 corridors
    top10 = prioritized.head(10).copy()

    # Create corridor labels
    if 'corridor_name' in top10.columns:
        corridor_labels = top10['corridor_name'].values
    else:
        corridor_labels = [f'Corridor {i + 1}' for i in range(len(top10))]

    # Create short labels for better display
    short_labels = []
    for i, label in enumerate(corridor_labels):
        if isinstance(label, str) and len(label) > 20:
            parts = label.split()
            if len(parts) > 2:
                # Keep first and last parts only
                short_labels.append(f"{i + 1}. {parts[0]}...{parts[-1]}")
            else:
                short_labels.append(f"{i + 1}. {label[:20]}...")
        else:
            short_labels.append(f"{i + 1}. {label}")

    # Initialize the heatmap data
    heatmap_data = pd.DataFrame(index=short_labels)

    # SOLUTION A: Use direct truck traffic if available
    if all(col in top10.columns for col in ['fast_traffic', 'slow_traffic']):
        heatmap_data['Fast Traffic'] = top10['fast_traffic'].values
        heatmap_data['Slow Traffic'] = top10['slow_traffic'].values
        metric_type = "Truck Traffic"

    # SOLUTION B: Use estimated truck traffic if available
    elif all(col in top10.columns for col in ['fast_traffic_est', 'slow_traffic_est']):
        heatmap_data['Fast Traffic'] = top10['fast_traffic_est'].values
        heatmap_data['Slow Traffic'] = top10['slow_traffic_est'].values
        metric_type = "Estimated Truck Traffic"

    # SOLUTION C: Use truck-per-charger ratio if available
    elif all(col in top10.columns for col in ['fast_truck_per_charger', 'slow_truck_per_charger']):
        heatmap_data['Fast Ratio'] = top10['fast_truck_per_charger'].values
        heatmap_data['Slow Ratio'] = top10['slow_truck_per_charger'].values
        metric_type = "Trucks per Charger Ratio"

    # SOLUTION D: Use energy consumption if available
    elif col_mapping['fast_energy_col'] and col_mapping['slow_energy_col']:
        heatmap_data['Fast Energy'] = top10[col_mapping['fast_energy_col']].values
        heatmap_data['Slow Energy'] = top10[col_mapping['slow_energy_col']].values
        metric_type = "Energy Consumption (MWh)"

    # SOLUTION E: Use charger counts if nothing else is available
    elif col_mapping['fast_charger_col'] and col_mapping['slow_charger_col']:
        heatmap_data['Fast Chargers'] = top10[col_mapping['fast_charger_col']].values
        heatmap_data['Slow Chargers'] = top10[col_mapping['slow_charger_col']].values
        metric_type = "Number of Chargers"

    # If we have data, create the heatmap using Plotly
    if len(heatmap_data.columns) >= 2:
        # Transpose for better visualization
        heatmap_data_t = heatmap_data.T

        # Create heatmap with Plotly
        fig = px.imshow(
            heatmap_data_t,
            labels=dict(x="Corridor", y="Charger Type", color=metric_type),
            x=heatmap_data_t.columns,
            y=heatmap_data_t.index,
            color_continuous_scale="Blues",
            aspect="auto",
            text_auto='.1f'
        )

        fig.update_layout(
            title=f'Fast vs Slow Charger {metric_type} by Priority Corridor',
            height=600,
            width=1000,
            xaxis=dict(
                tickangle=-45,
                title_font=dict(size=14),
            ),
            yaxis=dict(
                title_font=dict(size=14),
            ),
            coloraxis_colorbar=dict(
                title=metric_type,
                title_font=dict(size=12)
            ),
            template="plotly_white",
        )

        # Save as HTML for interactive viewing
        fig.write_html('charger_traffic_heatmap_interactive.html')

        # Also save as static image for reports
        fig.write_image('charger_traffic_heatmap.png', scale=2)

        print(f"- Created slow vs fast charger traffic heatmap using {metric_type} data")
    else:
        print("- Couldn't create heatmap: insufficient data for fast vs slow comparison")

        # Create fallback visualization using bar chart
        if col_mapping['fast_charger_col'] and col_mapping['slow_charger_col']:
            print("- Creating alternative visualization using charger counts instead")

            fast_counts = top10[col_mapping['fast_charger_col']].values
            slow_counts = top10[col_mapping['slow_charger_col']].values

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=short_labels,
                y=fast_counts,
                name='Fast Chargers',
                marker_color='rgba(65, 105, 225, 0.8)'
            ))
            fig.add_trace(go.Bar(
                x=short_labels,
                y=slow_counts,
                name='Slow Chargers',
                marker_color='rgba(220, 170, 50, 0.8)'
            ))

            fig.update_layout(
                title='Fast vs Slow Charger Distribution (Alternative to Heatmap)',
                xaxis_tickangle=-45,
                barmode='group',
                height=600,
                width=1000,
                template="plotly_white",
            )

            fig.write_html('charger_counts_interactive.html')
            fig.write_image('charger_counts.png', scale=2)

            print("- Created alternative visualization for charger counts")
except Exception as heatmap_error:
    print(f"Error creating the heatmap visualization: {heatmap_error}")
    print("- Attempting simpler visualization as fallback...")

    # Create simple fallback visualization
    try:
        # Use matplotlib for a simpler visualization that's more likely to succeed
        plt.figure(figsize=(12, 6))

        # Get data that's most likely to be available
        if col_mapping['fast_charger_col'] and col_mapping['slow_charger_col']:
            top5 = prioritized.head(5)
            fast_counts = top5[col_mapping['fast_charger_col']].values
            slow_counts = top5[col_mapping['slow_charger_col']].values

            x = range(len(top5))
            plt.bar([i - 0.2 for i in x], fast_counts, width=0.4, label='Fast Chargers', color='royalblue')
            plt.bar([i + 0.2 for i in x], slow_counts, width=0.4, label='Slow Chargers', color='orange')

            plt.title('Fast vs Slow Charger Distribution (Fallback Visualization)')
            plt.xlabel('Priority Corridor')
            plt.ylabel('Number of Chargers')
            plt.xticks(x, [f'Corridor {i + 1}' for i in range(len(top5))])
            plt.legend()
            plt.tight_layout()
            plt.savefig('charger_distribution_fallback.png')
            print("- Created fallback visualization using matplotlib")
    except Exception as fallback_error:
        print(f"Error creating fallback visualization: {fallback_error}")

# Save recommendations to CSV
try:
    recommendations = prioritized.head(10)
    recommendations.to_csv('corridor_recommendations.csv', index=False)
    print("\nSaved top 10 corridor recommendations to CSV")
except Exception as csv_error:
    print(f"Error saving recommendations to CSV: {csv_error}")

except Exception as e:
    print(f"\nError in analysis: {e}")
print("Check your data to ensure it contains the expected columns and formats.")
# Continue execution with what we have instead of returning None

return df


def count_processed_files(output_pattern='charger_*.png'):
    """Count the number of visualization files that have been processed"""
    files = glob.glob(output_pattern)
    return len(files)


if __name__ == "__main__":
    try:
        # Parse command line arguments
        import argparse

        parser = argparse.ArgumentParser(description='Analyze charger locations and identify corridors')
        parser.add_argument('--csv', type=str, default='ChargerLocations.csv',
                            help='Path to CSV file with charger locations')
        parser.add_argument('--use-google-maps', action='store_true',
                            help='Use Google Maps API for corridor identification')
        parser.add_argument('--api-key', type=str,
                            help='Google Maps API key (required if --use-google-maps is specified)')

        args = parser.parse_args()

        # Check if API key is provided when Google Maps is requested
        if args.use_google_maps and not args.api_key:
            print("Warning: --use-google-maps specified but no API key provided.")
            print("You can provide an API key with --api-key YOUR_API_KEY")
            # Continue without Google Maps
            args.use_google_maps = False

        # Clear previous visualization files to get accurate count
        for old_file in glob.glob('charger_*.png') + glob.glob('*.html'):
            try:
                os.remove(old_file)
            except:
                pass  # Ignore errors on file deletion

        # Process the data
        print("Starting analysis...")
        df_result = analyze_corridors(
            csv_path=args.csv,
            use_google_maps=args.use_google_maps,
            google_maps_api_key=args.api_key
        )

        # Count processed files
        num_visualizations = count_processed_files()

        if df_result is not None:
            print("\nAnalysis completed successfully!")
            print(f"Total locations analyzed: {len(df_result)}")
            print(f"Number of visualizations created: {num_visualizations}")

            # Report on identified corridors if available
            if 'corridor_name' in df_result.columns:
                corridors = df_result['corridor_name'].dropna().unique()
                print(f"Number of unique corridors identified: {len(corridors)}")

                # Display top corridors by frequency
                corridor_counts = df_result['corridor_name'].value_counts().head(5)
                print("\nTop 5 corridors by frequency:")
                for corridor, count in corridor_counts.items():
                    print(f"- {corridor}: {count} locations")

            print("\nVisualization files created:")
            for viz_file in glob.glob('*.html') + glob.glob('*.png'):
                print(f"- {viz_file}")
        else:
            print("\nAnalysis could not be completed. Please check the errors above.")
    except Exception as main_error:
        print(f"\nUnexpected error in main execution: {main_error}")
        # Don't exit with a non-zero code, as that would interrupt batch processing