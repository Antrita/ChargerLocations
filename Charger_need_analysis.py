import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_charger_distribution_csv(input_file='ChargerLocationsRefined.csv', output_file='Charger_Distr_viz.csv'):
    """
    Process the input data and generate a CSV file with charger distribution metrics
    focusing on identifying corridors that need more fast chargers.
    """
    print(f"Loading data from {input_file}...")
    try:
        # Load the data
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} records from {input_file}")

        # Group by corridor
        print("Aggregating data by corridor...")
        corridor_data = df.groupby('corridor_name').agg({
            'MainDTN': 'sum',  # Total electrified trucks
            'NFCh30m': 'sum',  # Fast chargers
            'NSCh2pD': 'sum',  # Slow chargers
            'TotCha': 'sum',  # Total chargers
            'ChEBM': 'sum',  # Fast charging energy
            'ChERM': 'sum',  # Slow charging energy
            'ChE30': 'sum',  # Total energy
            'MDTN_B': 'sum',  # Trucks using fast chargers
            'MDTN_R': 'sum'  # Trucks using slow chargers
        }).reset_index()

        print(f"Generated aggregated data for {len(corridor_data)} corridors")

        # Calculate key metrics for charger planning
        print("Calculating corridor metrics...")

        # Truck to charger ratios
        corridor_data['trucks_per_fast_charger'] = corridor_data['MDTN_B'] / corridor_data['NFCh30m'].replace(0, np.nan)
        corridor_data['trucks_per_slow_charger'] = corridor_data['MDTN_R'] / corridor_data['NSCh2pD'].replace(0, np.nan)
        corridor_data['total_trucks_per_charger'] = corridor_data['MainDTN'] / corridor_data['TotCha']

        # Charger type distributions
        corridor_data['fast_slow_ratio'] = corridor_data['NFCh30m'] / corridor_data['NSCh2pD'].replace(0, np.nan)
        corridor_data['fast_charger_percent'] = 100 * corridor_data['NFCh30m'] / corridor_data['TotCha']

        # Energy metrics
        corridor_data['energy_per_fast_charger'] = corridor_data['ChEBM'] / corridor_data['NFCh30m'].replace(0, np.nan)
        corridor_data['energy_per_slow_charger'] = corridor_data['ChERM'] / corridor_data['NSCh2pD'].replace(0, np.nan)
        corridor_data['energy_per_truck'] = corridor_data['ChE30'] / corridor_data['MainDTN'].replace(0, np.nan)

        # Calculate the ideal number of fast chargers (using industry standard of 50 trucks per charger)
        ideal_trucks_per_charger = 50
        corridor_data['recommended_fast_chargers'] = np.ceil(corridor_data['MDTN_B'] / ideal_trucks_per_charger)
        corridor_data['fast_charger_deficit'] = corridor_data['recommended_fast_chargers'] - corridor_data['NFCh30m']
        corridor_data['fast_charger_deficit'] = corridor_data['fast_charger_deficit'].apply(lambda x: max(0, x))

        # Calculate charger need score (composite metric)
        corridor_data['charger_need_score'] = (
            # Traffic weight (40%)
                0.4 * (corridor_data['MainDTN'] / corridor_data['MainDTN'].max()) +
                # Trucks per fast charger weight (40%)
                0.4 * (corridor_data['trucks_per_fast_charger'] / corridor_data['trucks_per_fast_charger'].max()) +
                # Energy per charger weight (20%)
                0.2 * (corridor_data['energy_per_fast_charger'] / corridor_data['energy_per_fast_charger'].max())
        )

        # Sort by charger need score (descending)
        corridor_data = corridor_data.sort_values('charger_need_score', ascending=False)

        # Save to CSV
        print(f"Saving processed data to {output_file}...")
        corridor_data.to_csv(output_file, index=False)
        print(f"Successfully saved data to {output_file}")

        return corridor_data

    except Exception as e:
        print(f"Error processing data: {e}")
        return None


def create_fast_charger_need_visualization(data=None, input_file='Charger_Distr_viz.csv',
                                           output_file='Fast_Charger_Need_Analysis.html'):
    """
    Create comprehensive visualizations to identify corridors needing more fast chargers.
    """
    try:
        # Load data if not provided
        if data is None:
            print(f"Loading data from {input_file}...")
            data = pd.read_csv(input_file)
            print(f"Loaded data with {len(data)} corridors")

        # Create a subplot with 2 rows and 2 columns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Top Corridors Needing Fast Chargers",
                "Fast Charger Distribution vs. Truck Traffic",
                "Fast vs. Slow Charger Analysis",
                "Investment Priority Matrix"
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        # 1. Bar chart for top corridors needing fast chargers (top left)
        top_need = data.sort_values('fast_charger_deficit', ascending=False).head(10)

        fig.add_trace(
            go.Bar(
                x=top_need['corridor_name'],
                y=top_need['fast_charger_deficit'],
                marker_color='firebrick',
                name='Additional Fast Chargers Needed',
                hovertemplate='<b>%{x}</b><br>Additional Chargers Needed: %{y}<br>Current Fast Chargers: %{customdata[0]}<br>Truck Traffic: %{customdata[1]:,.0f}<extra></extra>',
                customdata=np.stack((top_need['NFCh30m'], top_need['MainDTN']), axis=-1)
            ),
            row=1, col=1
        )

        # 2. Scatter plot for fast charger distribution vs truck traffic (top right)
        fig.add_trace(
            go.Scatter(
                x=data['NFCh30m'],
                y=data['MDTN_B'],
                mode='markers',
                marker=dict(
                    size=data['fast_charger_deficit'] * 2 + 5,
                    color=data['trucks_per_fast_charger'],
                    colorscale='YlOrRd',
                    showscale=True,
                    colorbar=dict(
                        title='Trucks per<br>Fast Charger',
                        x=0.98,
                        y=0.84,
                        len=0.3
                    )
                ),
                name='Corridors',
                text=data['corridor_name'],
                hovertemplate='<b>%{text}</b><br>Fast Chargers: %{x}<br>Trucks Using Fast Chargers: %{y:,.0f}<br>Additional Chargers Needed: %{marker.size:.0f}<br>Trucks per Fast Charger: %{marker.color:.1f}<extra></extra>'
            ),
            row=1, col=2
        )

        # Add reference line for ideal ratio (50 trucks per charger)
        x_vals = np.linspace(0, data['NFCh30m'].max() * 1.1, 100)
        y_vals = x_vals * 50  # 50 trucks per charger

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name='Ideal Ratio (50 trucks/charger)',
                hoverinfo='skip'
            ),
            row=1, col=2
        )

        # 3. Stacked bar chart for fast vs slow charger analysis (bottom left)
        top_corridors = data.sort_values('MainDTN', ascending=False).head(10)

        fig.add_trace(
            go.Bar(
                x=top_corridors['corridor_name'],
                y=top_corridors['NFCh30m'],
                name='Fast Chargers',
                marker_color='royalblue',
                hovertemplate='<b>%{x}</b><br>Fast Chargers: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(
                x=top_corridors['corridor_name'],
                y=top_corridors['NSCh2pD'],
                name='Slow Chargers',
                marker_color='seagreen',
                hovertemplate='<b>%{x}</b><br>Slow Chargers: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add truck traffic line to the stacked bar chart
        fig.add_trace(
            go.Scatter(
                x=top_corridors['corridor_name'],
                y=top_corridors['MainDTN'],
                name='Total Trucks',
                mode='lines+markers',
                marker=dict(color='orange', size=8),
                line=dict(width=2, dash='dot'),
                yaxis='y3',
                hovertemplate='<b>%{x}</b><br>Total Trucks: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )

        # 4. Investment priority matrix (bottom right)
        fig.add_trace(
            go.Scatter(
                x=data['fast_charger_percent'],
                y=data['trucks_per_fast_charger'],
                mode='markers',
                marker=dict(
                    size=data['MainDTN'] / data['MainDTN'].max() * 25 + 5,
                    color=data['fast_charger_deficit'],
                    colorscale='YlOrRd',
                    showscale=True,
                    colorbar=dict(
                        title='Fast Charger<br>Deficit',
                        x=1.0,
                        y=0.34,
                        len=0.3
                    )
                ),
                name='Corridors',
                text=data['corridor_name'],
                hovertemplate='<b>%{text}</b><br>Fast Charger %: %{x:.1f}%<br>Trucks per Fast Charger: %{y:.1f}<br>Total Trucks: %{marker.size:,.0f}<br>Deficit: %{marker.color:.0f}<extra></extra>'
            ),
            row=2, col=2
        )

        # Add quadrant lines to investment priority matrix
        fig.add_shape(
            type="line",
            x0=30, y0=0,
            x1=30, y1=data['trucks_per_fast_charger'].max() * 1.1,
            line=dict(color="black", width=1, dash="dash"),
            row=2, col=2
        )

        fig.add_shape(
            type="line",
            x0=0, y0=100,
            x1=100, y1=100,
            line=dict(color="black", width=1, dash="dash"),
            row=2, col=2
        )

        # Add quadrant annotations
        fig.add_annotation(
            x=15, y=data['trucks_per_fast_charger'].max() * 0.75,
            text="HIGH PRIORITY",
            showarrow=False,
            font=dict(size=10, color="red"),
            row=2, col=2
        )

        fig.add_annotation(
            x=65, y=data['trucks_per_fast_charger'].max() * 0.75,
            text="MEDIUM PRIORITY",
            showarrow=False,
            font=dict(size=10, color="orange"),
            row=2, col=2
        )

        fig.add_annotation(
            x=15, y=50,
            text="MEDIUM PRIORITY",
            showarrow=False,
            font=dict(size=10, color="blue"),
            row=2, col=2
        )

        fig.add_annotation(
            x=65, y=50,
            text="LOW PRIORITY",
            showarrow=False,
            font=dict(size=10, color="green"),
            row=2, col=2
        )

        # Update layout for each subplot
        fig.update_xaxes(title_text="Corridor", tickangle=45, row=1, col=1)
        fig.update_yaxes(title_text="Additional Fast Chargers Needed", row=1, col=1)

        fig.update_xaxes(title_text="Current Fast Chargers", row=1, col=2)
        fig.update_yaxes(title_text="Trucks Using Fast Chargers", row=1, col=2)

        fig.update_xaxes(title_text="Corridor", tickangle=45, row=2, col=1)
        fig.update_yaxes(title_text="Number of Chargers", row=2, col=1)

        fig.update_xaxes(title_text="Fast Charger Percentage", range=[0, 100], row=2, col=2)
        fig.update_yaxes(title_text="Trucks per Fast Charger", range=[0, data['trucks_per_fast_charger'].max() * 1.1],
                         row=2, col=2)

        # Add a third y-axis for the truck traffic line in the stacked bar chart
        fig.update_layout(
            yaxis3=dict(
                title="Number of Trucks",
                titlefont=dict(color="orange"),
                tickfont=dict(color="orange"),
                anchor="x",
                overlaying="y2",
                side="right"
            ),
            barmode='stack',
            height=900,
            width=1200,
            title={
                'text': 'Fast Charger Infrastructure Needs Analysis',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )

        # Save the figure
        print(f"Saving visualization to {output_file}...")
        fig.write_html(output_file)
        print(f"Successfully saved visualization to {output_file}")

        # Create a summary for printing
        print("\n===== FAST CHARGER NEEDS ANALYSIS SUMMARY =====")

        # Total stats
        total_corridors = len(data)
        total_fast_chargers = int(data['NFCh30m'].sum())
        total_deficit = int(data['fast_charger_deficit'].sum())

        print(f"Total corridors analyzed: {total_corridors}")
        print(f"Current total fast chargers: {total_fast_chargers}")
        print(f"Total additional fast chargers needed: {total_deficit}")

        # Critical corridors
        critical_corridors = data[(data['trucks_per_fast_charger'] > 100) & (data['fast_charger_deficit'] > 5)]
        print(f"\nCritical priority corridors: {len(critical_corridors)}")

        # Top 5 corridors by deficit
        print("\nTop 5 corridors needing fast chargers:")
        top5 = data.sort_values('fast_charger_deficit', ascending=False).head(5)
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"{i}. {row['corridor_name']}: {int(row['fast_charger_deficit'])} additional fast chargers needed " +
                  f"(Current: {int(row['NFCh30m'])}, Trucks: {int(row['MDTN_B']):,}, " +
                  f"Trucks per Charger: {row['trucks_per_fast_charger']:.1f})")

        # Conclusions based on 500km optimal distance
        print("\nBased on 500km optimal distance requirement:")

        # Identify major corridors that span approximately 500km
        # This is an approximation since we don't have actual distance data
        major_corridors = data[data['corridor_name'].str.contains('Motorway|Highway')].copy()

        # For demonstration, we'll use the truck count as a proxy for corridor length
        # Higher truck count often correlates with longer/major corridors
        major_corridors['approx_importance'] = major_corridors['MainDTN'] * major_corridors['fast_charger_deficit']
        top_distance_corridors = major_corridors.sort_values('approx_importance', ascending=False).head(3)

        for i, (_, row) in enumerate(top_distance_corridors.iterrows(), 1):
            print(f"{i}. {row['corridor_name']}: Strategic corridor for 500km optimal distance planning")
            print(f"   Current fast chargers: {int(row['NFCh30m'])}")
            print(f"   Additional needed: {int(row['fast_charger_deficit'])}")
            print(f"   Total truck traffic: {int(row['MainDTN']):,}")

        return fig

    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None


def main():
    # Generate the CSV
    data = generate_charger_distribution_csv()

    if data is not None:
        # Create visualizations
        fig = create_fast_charger_need_visualization(data)

        print("\nAnalysis complete. Open Fast_Charger_Need_Analysis.html to view the interactive visualization.")
        print(
            "The visualization identifies regions that need more fast chargers based on truck traffic and existing infrastructure.")


if __name__ == "__main__":
    main()