import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from shapely import wkt
import os


# Load the data
def load_data(file_path='ChargerLocationsRefined.csv'):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data: {file_path}")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Process the data to prepare for visualization
def process_data(df):
    # Group by corridor
    corridor_data = df.groupby('corridor_name').agg({
        'MainDTN': 'sum',  # Sum of electrified trucks
        'NFCh30m': 'sum',  # Sum of fast chargers
        'NSCh2pD': 'sum',  # Sum of slow chargers
        'TotCha': 'sum',  # Sum of total chargers
        'ChEBM': 'sum',  # Fast charging energy
        'ChERM': 'sum',  # Slow charging energy
        'ChE30': 'sum',  # Total energy
        'MDTN_B': 'sum',  # Trucks using fast chargers
        'MDTN_R': 'sum'  # Trucks using slow chargers
    }).reset_index()

    # Calculate additional metrics
    corridor_data['fast_charger_ratio'] = corridor_data['NFCh30m'] / corridor_data['TotCha']
    corridor_data['slow_charger_ratio'] = corridor_data['NSCh2pD'] / corridor_data['TotCha']
    corridor_data['trucks_per_fast_charger'] = corridor_data['MDTN_B'] / corridor_data['NFCh30m'].replace(0, np.nan)
    corridor_data['trucks_per_slow_charger'] = corridor_data['MDTN_R'] / corridor_data['NSCh2pD'].replace(0, np.nan)
    corridor_data['energy_per_fast_charger'] = corridor_data['ChEBM'] / corridor_data['NFCh30m'].replace(0, np.nan)
    corridor_data['energy_per_slow_charger'] = corridor_data['ChERM'] / corridor_data['NSCh2pD'].replace(0, np.nan)

    # Sort by total trucks (descending)
    corridor_data = corridor_data.sort_values('MainDTN', ascending=False)

    # Take top 20 corridors for better visualization
    top_corridors = corridor_data.head(20)

    return top_corridors


# Create a heatmap visualization
def create_heatmap(data):
    # Prepare data for heatmap
    # Get top corridors by total truck traffic
    corridors = data['corridor_name'].tolist()

    # Create matrix for heatmap
    # Each row is a corridor, columns are fast and slow chargers
    fast_chargers = data['NFCh30m'].tolist()
    slow_chargers = data['NSCh2pD'].tolist()

    # Create custom hover text
    hover_texts = []
    for _, row in data.iterrows():
        hover_text = f"<b>{row['corridor_name']}</b><br>" + \
                     f"Total Trucks: {row['MainDTN']:.0f}<br>" + \
                     f"Fast Chargers: {row['NFCh30m']:.0f}<br>" + \
                     f"Slow Chargers: {row['NSCh2pD']:.0f}<br>" + \
                     f"Total Chargers: {row['TotCha']:.0f}<br>" + \
                     f"Trucks per Fast Charger: {row['trucks_per_fast_charger']:.1f}<br>" + \
                     f"Trucks per Slow Charger: {row['trucks_per_slow_charger']:.1f}<br>" + \
                     f"Fast Charging Energy (MWh): {row['ChEBM']:.1f}<br>" + \
                     f"Slow Charging Energy (MWh): {row['ChERM']:.1f}<br>" + \
                     f"Total Energy (MWh): {row['ChE30']:.1f}"
        hover_texts.append(hover_text)

    # Create a figure with two subplots
    fig = go.Figure()

    # Add heatmap for fast chargers
    fig.add_trace(go.Heatmap(
        z=[fast_chargers],
        y=['Fast Chargers'],
        x=corridors,
        colorscale='Blues',
        hoverinfo='text',
        hovertext=[hover_texts],
        showscale=False
    ))

    # Add heatmap for slow chargers
    fig.add_trace(go.Heatmap(
        z=[slow_chargers],
        y=['Slow Chargers'],
        x=corridors,
        colorscale='Greens',
        hoverinfo='text',
        hovertext=[hover_texts],
        showscale=True,
        colorbar=dict(title="Number of Chargers")
    ))

    # Update layout
    fig.update_layout(
        title='Fast vs Slow Charger Distribution Across Top Corridors',
        height=600,
        width=1200,
        xaxis=dict(
            title='Corridor',
            tickangle=45
        ),
        yaxis=dict(
            title='Charger Type'
        )
    )

    return fig


# Create stacked bar chart comparing fast vs slow chargers
def create_charger_comparison(data):
    fig = go.Figure()

    # Add fast chargers
    fig.add_trace(go.Bar(
        x=data['corridor_name'],
        y=data['NFCh30m'],
        name='Fast Chargers',
        hovertemplate='<b>%{x}</b><br>Fast Chargers: %{y}<br>Total Trucks: %{customdata[0]:.0f}<br>Trucks using Fast Chargers: %{customdata[1]:.0f}<extra></extra>',
        customdata=np.stack((data['MainDTN'], data['MDTN_B']), axis=-1),
        marker_color='royalblue'
    ))

    # Add slow chargers
    fig.add_trace(go.Bar(
        x=data['corridor_name'],
        y=data['NSCh2pD'],
        name='Slow Chargers',
        hovertemplate='<b>%{x}</b><br>Slow Chargers: %{y}<br>Total Trucks: %{customdata[0]:.0f}<br>Trucks using Slow Chargers: %{customdata[1]:.0f}<extra></extra>',
        customdata=np.stack((data['MainDTN'], data['MDTN_R']), axis=-1),
        marker_color='seagreen'
    ))

    # Add total trucks (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=data['corridor_name'],
        y=data['MainDTN'],
        name='Total Trucks',
        hovertemplate='<b>%{x}</b><br>Total Trucks: %{y:.0f}<extra></extra>',
        mode='lines+markers',
        marker=dict(color='firebrick', size=10),
        line=dict(width=3, dash='dot'),
        yaxis='y2'
    ))

    # Update layout with secondary y-axis
    fig.update_layout(
        title='Fast vs Slow Chargers and Truck Traffic by Corridor',
        xaxis=dict(
            title='Corridor',
            tickangle=45
        ),
        yaxis=dict(
            title='Number of Chargers',
            side='left'
        ),
        yaxis2=dict(
            title='Number of Trucks',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        barmode='stack',
        height=600,
        width=1200,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    return fig


# Create a bubble chart for truck traffic and charger distribution
def create_bubble_chart(data):
    fig = px.scatter(
        data,
        x='NFCh30m',  # Fast chargers
        y='NSCh2pD',  # Slow chargers
        size='MainDTN',  # Bubble size based on truck traffic
        color='ChE30',  # Color based on total energy
        hover_name='corridor_name',
        text='corridor_name',
        size_max=50,
        color_continuous_scale='Viridis',
        labels={
            'NFCh30m': 'Number of Fast Chargers',
            'NSCh2pD': 'Number of Slow Chargers',
            'MainDTN': 'Total Trucks',
            'ChE30': 'Total Energy (MWh)'
        },
        title='Corridor Charger Distribution vs Truck Traffic',
        hover_data={
            'corridor_name': False,  # Already in hover_name
            'NFCh30m': True,
            'NSCh2pD': True,
            'MainDTN': True,
            'TotCha': True,
            'ChEBM': ':.1f',
            'ChERM': ':.1f',
            'fast_charger_ratio': ':.2f',
            'slow_charger_ratio': ':.2f'
        }
    )

    # Update layout
    fig.update_layout(
        height=700,
        width=1000,
        xaxis=dict(title='Number of Fast Chargers'),
        yaxis=dict(title='Number of Slow Chargers'),
        coloraxis_colorbar=dict(title='Total Energy (MWh)')
    )

    return fig


# Main function
def main():
    # Load data
    df = load_data()
    if df is None:
        print("Error: Could not load data. Exiting.")
        return

    # Process data
    processed_data = process_data(df)
    print(f"Processed data for {len(processed_data)} corridors")

    # Create visualizations
    heatmap_fig = create_heatmap(processed_data)
    print("Created heatmap visualization")

    comparison_fig = create_charger_comparison(processed_data)
    print("Created charger comparison visualization")

    bubble_fig = create_bubble_chart(processed_data)
    print("Created bubble chart visualization")

    # Save visualizations
    heatmap_fig.write_html("corridor_charger_heatmap.html")
    comparison_fig.write_html("corridor_charger_comparison.html")
    bubble_fig.write_html("corridor_charger_bubble.html")
    print("Saved visualizations as HTML files")

    # Show first figures
    heatmap_fig.show()
    comparison_fig.show()
    bubble_fig.show()


if __name__ == "__main__":
    main()