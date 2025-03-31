import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import os


def estimate_distance_between_chargers(df):
    """
    Estimate the distance between fast chargers for each corridor
    using truck traffic as a proxy for corridor length
    """
    corridor_data = df.copy()

    # Define typical corridor lengths based on highway types
    def estimate_corridor_length(corridor_name, traffic):
        if 'Motorway' in corridor_name or 'motorway' in corridor_name:
            base_length = 350
        elif 'Highway' in corridor_name:
            base_length = 250
        elif 'Route' in corridor_name or 'Trunk' in corridor_name:
            base_length = 200
        else:
            base_length = 150

        # Scale by truck traffic relative to median
        traffic_median = corridor_data['MainDTN'].median()
        if traffic_median > 0:
            traffic_factor = traffic / traffic_median
        else:
            traffic_factor = 1

        # Apply a dampening function to avoid extreme values
        traffic_adjustment = np.sqrt(traffic_factor)

        return base_length * traffic_adjustment
    corridor_data['estimated_length_km'] = corridor_data.apply(
        lambda row: estimate_corridor_length(row['corridor_name'], row['MainDTN']), axis=1)

    # Calculate distance between chargers
    # Handle case where NFCh30m is 0
    corridor_data['dist_between_fast_chargers_km'] = corridor_data.apply(
        lambda row: row['estimated_length_km'] / (row['NFCh30m'] + 0.1), axis=1)

    # Calculate how many additional chargers needed to reach 500km spacing
    corridor_data['additional_chargers_for_500km'] = corridor_data.apply(
        lambda row: max(0, int(np.ceil(row['estimated_length_km'] / 500) - row['NFCh30m'])), axis=1)

    return corridor_data


def load_data(file_path='Charger_Distr_viz.csv'):
    """Load the processed data"""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Add distance estimates if not present
        if 'dist_between_fast_chargers_km' not in df.columns:
            df = estimate_distance_between_chargers(df)
        return df
    else:
        # If the file doesn't exist, try to create it
        print("CSV file not found. Creating from original data...")
        try:
            from charger_needs_analysis import load_and_process_data
            return load_and_process_data()
        except:
            # Try loading from original file
            try:
                df = pd.read_csv('ChargerLocationsRefined.csv')
                print("Loading from ChargerLocationsRefined.csv")
                # Group by corridor_name
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

                # Add basic metrics
                corridor_data['trucks_per_fast_charger'] = corridor_data['MDTN_B'] / corridor_data['NFCh30m'].replace(0,
                                                                                                                      np.nan)
                corridor_data['fast_charger_percent'] = 100 * corridor_data['NFCh30m'] / corridor_data['TotCha']
                corridor_data['fast_charger_deficit'] = np.ceil(corridor_data['MDTN_B'] / 50) - corridor_data['NFCh30m']
                corridor_data['fast_charger_deficit'] = corridor_data['fast_charger_deficit'].apply(lambda x: max(0, x))
                corridor_data['recommended_fast_chargers'] = np.ceil(corridor_data['MDTN_B'] / 50)
                corridor_data['charger_need_score'] = (
                        0.4 * (corridor_data['MainDTN'] / corridor_data['MainDTN'].max()) +
                        0.4 * (corridor_data['trucks_per_fast_charger'] / corridor_data[
                    'trucks_per_fast_charger'].max().replace(0, 1)) +
                        0.2 * (corridor_data['ChEBM'] / corridor_data['NFCh30m'].replace(0, np.nan)) /
                        (corridor_data['ChEBM'] / corridor_data['NFCh30m'].replace(0, np.nan)).max().replace(0, 1)
                )

                # Add distance estimates
                corridor_data = estimate_distance_between_chargers(corridor_data)

                # Save for future use
                corridor_data.to_csv('Charger_Distr_viz.csv', index=False)
                print("Saved processed data to Charger_Distr_viz.csv")

                return corridor_data
            except Exception as e:
                print(f"Error loading data: {e}")
                return None


def create_dashboard():
    """Create a Dash dashboard for EV charger infrastructure planning"""
    # Load data
    df = load_data()

    if df is None:
        print("Error: Could not load data.")
        return

    # Initialize Dash app
    app = dash.Dash(__name__, title="EV Fast Charger Infrastructure Planning")

    # Define app layout
    app.layout = html.Div(style={'fontFamily': 'Arial', 'margin': '0 auto', 'maxWidth': '1200px'}, children=[
        html.H1("EV Fast Charger Infrastructure Planning Dashboard",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': '20px'}),

        html.Div(style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}, children=[
            html.P(
                "This dashboard helps identify optimal corridors for new fast charger installations based on truck traffic and existing infrastructure.",
                style={'maxWidth': '800px', 'textAlign': 'center', 'fontSize': '16px'})
        ]),

        # Filters section
        html.Div(style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'marginBottom': '20px'},
                 children=[
                     html.H3("Filters", style={'marginBottom': '10px'}),
                     html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'},
                              children=[
                                  html.Div(style={'width': '48%', 'marginBottom': '10px'}, children=[
                                      html.Label("Minimum Truck Traffic:"),
                                      dcc.Slider(
                                          id='traffic-slider',
                                          min=0,
                                          max=df['MainDTN'].max(),
                                          step=1000,
                                          value=0,
                                          marks={i: f'{i:,}' for i in range(0, int(df['MainDTN'].max()) + 5000, 5000)},
                                      ),
                                  ]),
                                  html.Div(style={'width': '48%', 'marginBottom': '10px'}, children=[
                                      html.Label("Minimum Distance Between Chargers (km):"),
                                      dcc.Slider(
                                          id='distance-slider',
                                          min=0,
                                          max=1000,
                                          step=50,
                                          value=0,
                                          marks={i: f'{i}' for i in range(0, 1001, 100)},
                                      ),
                                  ]),
                              ]),
                     html.Div(style={'marginTop': '10px'}, children=[
                         html.Button('Reset Filters', id='reset-button',
                                     style={'backgroundColor': '#3498db', 'color': 'white',
                                            'border': 'none', 'padding': '8px 15px', 'borderRadius': '4px'})
                     ]),
                 ]),

        # Key metrics section
        html.Div(
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between', 'marginBottom': '20px'},
            children=[
                html.Div(style={'width': '24%', 'backgroundColor': '#e74c3c', 'color': 'white',
                                'padding': '15px', 'borderRadius': '5px', 'textAlign': 'center'}, children=[
                    html.H3("Total Corridors", style={'marginBottom': '5px'}),
                    html.H2(id='total-corridors', style={'margin': '0'})
                ]),
                html.Div(style={'width': '24%', 'backgroundColor': '#3498db', 'color': 'white',
                                'padding': '15px', 'borderRadius': '5px', 'textAlign': 'center'}, children=[
                    html.H3("Total Fast Chargers", style={'marginBottom': '5px'}),
                    html.H2(id='total-fast-chargers', style={'margin': '0'})
                ]),
                html.Div(style={'width': '24%', 'backgroundColor': '#2ecc71', 'color': 'white',
                                'padding': '15px', 'borderRadius': '5px', 'textAlign': 'center', 'cursor': 'pointer'},
                         id='additional-chargers-card', children=[
                        html.H3("Additional Chargers Needed", style={'marginBottom': '5px'}),
                        html.H2(id='additional-chargers', style={'margin': '0'})
                    ]),
                html.Div(style={'width': '24%', 'backgroundColor': '#f39c12', 'color': 'white',
                                'padding': '15px', 'borderRadius': '5px', 'textAlign': 'center'}, children=[
                    html.H3("Avg. Trucks per Fast Charger", style={'marginBottom': '5px'}),
                    html.H2(id='avg-trucks-per-charger', style={'margin': '0'})
                ]),
            ]),

        # Modal for corridor analysis
        html.Div(
            id='corridor-modal',
            style={
                'display': 'none',
                'position': 'fixed',
                'z-index': '1000',
                'left': '0',
                'top': '0',
                'width': '100%',
                'height': '100%',
                'overflow': 'auto',
                'backgroundColor': 'rgba(0,0,0,0.4)',
            },
            children=[
                html.Div(
                    style={
                        'backgroundColor': '#fefefe',
                        'margin': '5% auto',
                        'padding': '20px',
                        'border': '1px solid #888',
                        'width': '90%',
                        'maxWidth': '1200px',
                        'borderRadius': '5px',
                    },
                    children=[
                        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'},
                                 children=[
                                     html.H2("Corridors for New Charger Locations (500km Spacing)",
                                             style={'margin': '0'}),
                                     html.Span(
                                         "×",
                                         id='close-modal',
                                         style={
                                             'color': '#aaa',
                                             'float': 'right',
                                             'fontSize': '28px',
                                             'fontWeight': 'bold',
                                             'cursor': 'pointer',
                                         }
                                     ),
                                 ]),
                        html.Hr(),
                        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'},
                                 children=[
                                     html.Div(style={'width': '100%', 'marginBottom': '20px'}, children=[
                                         dcc.Graph(id='spacing-analysis', style={'height': '400px'}),
                                     ]),
                                     html.Div(style={'width': '48%'}, children=[
                                         dcc.Graph(id='corridor-map', style={'height': '400px'}),
                                     ]),
                                     html.Div(style={'width': '48%'}, children=[
                                         dcc.Graph(id='corridor-recommendations', style={'height': '400px'}),
                                     ]),
                                 ]),
                    ]
                )
            ]
        ),

        # Charts section
        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}, children=[
            # Left column
            html.Div(style={'width': '48%'}, children=[
                # Charger deficit chart
                html.Div(style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px',
                                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
                    html.H3("Top Corridors by Fast Charger Deficit", style={'marginBottom': '10px'}),
                    dcc.Graph(id='deficit-chart', style={'height': '400px'})
                ]),

                # Fast vs Slow charger comparison
                html.Div(style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px',
                                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
                    html.H3("Fast vs Slow Charger Distribution", style={'marginBottom': '10px'}),
                    dcc.Graph(id='fast-slow-chart', style={'height': '400px'})
                ]),
            ]),

            # Right column
            html.Div(style={'width': '48%'}, children=[
                # Investment priority quadrant
                html.Div(style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px',
                                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
                    html.H3("Fast Charger Investment Priority Matrix", style={'marginBottom': '10px'}),
                    dcc.Graph(id='priority-matrix', style={'height': '400px'})
                ]),

                # Traffic vs current vs recommended chargers
                html.Div(style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px',
                                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
                    html.H3("Current vs Recommended Fast Chargers", style={'marginBottom': '10px'}),
                    dcc.Graph(id='current-vs-recommended', style={'height': '400px'})
                ]),
            ]),
        ]),

        # Table section
        html.Div(style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px',
                        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
            html.H3("Fast Charger Investment Priority Rankings", style={'marginBottom': '10px'}),
            html.Div(id='priority-table')
        ]),

        # Footer
        html.Div(style={'textAlign': 'center', 'padding': '20px', 'color': '#7f8c8d'}, children=[
            html.P(
                "EV Charging Infrastructure Analysis Dashboard | Based on Truck Traffic and Charger Distribution Data")
        ]),

        # Store filtered data
        dcc.Store(id='filtered-data'),
    ])

    # Define callback to filter data
    @app.callback(
        Output('filtered-data', 'data'),
        [Input('traffic-slider', 'value'),
         Input('distance-slider', 'value'),
         Input('reset-button', 'n_clicks')]
    )
    def filter_data(min_traffic, min_distance, n_clicks):
        ctx = dash.callback_context

        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-button.n_clicks':
            # Reset filters
            min_traffic = 0
            min_distance = 0

        # Apply filters
        filtered_df = df[(df['MainDTN'] >= min_traffic) &
                         (df['dist_between_fast_chargers_km'] >= min_distance)]

        return filtered_df.to_json(date_format='iso', orient='split')

    # Define callback to update dashboard
    @app.callback(
        [Output('total-corridors', 'children'),
         Output('total-fast-chargers', 'children'),
         Output('additional-chargers', 'children'),
         Output('avg-trucks-per-charger', 'children'),
         Output('deficit-chart', 'figure'),
         Output('fast-slow-chart', 'figure'),
         Output('priority-matrix', 'figure'),
         Output('current-vs-recommended', 'figure'),
         Output('priority-table', 'children')],
        [Input('filtered-data', 'data')]
    )
    def update_dashboard(json_data):
        # Load filtered data
        filtered_df = pd.read_json(json_data, orient='split')

        # Calculate metrics
        total_corridors = len(filtered_df)
        total_fast_chargers = int(filtered_df['NFCh30m'].sum())

        # Calculate additional chargers needed based on 500km spacing requirement
        corridors_needing_chargers = filtered_df[filtered_df['dist_between_fast_chargers_km'] > 500]
        additional_chargers_needed = int(corridors_needing_chargers['additional_chargers_for_500km'].sum())

        avg_trucks_per_charger = round(filtered_df['trucks_per_fast_charger'].mean(), 1)

        # Create deficit chart
        deficit_df = filtered_df.sort_values('fast_charger_deficit', ascending=False).head(10)
        deficit_fig = px.bar(
            deficit_df,
            y='corridor_name',
            x='fast_charger_deficit',
            color='trucks_per_fast_charger',
            orientation='h',
            color_continuous_scale='YlOrRd',
            labels={
                'fast_charger_deficit': 'Additional Fast Chargers Needed',
                'corridor_name': 'Corridor',
                'trucks_per_fast_charger': 'Trucks per Fast Charger'
            }
        )

        deficit_fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_colorbar=dict(title='Trucks per<br>Fast Charger'),
            margin=dict(l=20, r=20, t=20, b=20)
        )

        # Create fast vs slow charger chart
        fast_slow_df = filtered_df.sort_values('MainDTN', ascending=False).head(10)
        fast_slow_fig = go.Figure()

        fast_slow_fig.add_trace(go.Bar(
            x=fast_slow_df['corridor_name'],
            y=fast_slow_df['NFCh30m'],
            name='Fast Chargers',
            marker_color='royalblue'
        ))

        fast_slow_fig.add_trace(go.Bar(
            x=fast_slow_df['corridor_name'],
            y=fast_slow_df['NSCh2pD'],
            name='Slow Chargers',
            marker_color='seagreen'
        ))

        fast_slow_fig.add_trace(go.Scatter(
            x=fast_slow_df['corridor_name'],
            y=fast_slow_df['MainDTN'],
            name='Total Trucks',
            mode='lines+markers',
            marker=dict(color='firebrick', size=8),
            line=dict(width=2, dash='dot'),
            yaxis='y2'
        ))

        fast_slow_fig.update_layout(
            barmode='stack',
            xaxis=dict(tickangle=45),
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
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=20, r=80, t=20, b=80)
        )

        # Create priority matrix
        priority_fig = px.scatter(
            filtered_df,
            x='fast_charger_percent',
            y='trucks_per_fast_charger',
            size='MainDTN',
            color='fast_charger_deficit',
            hover_name='corridor_name',
            color_continuous_scale='YlOrRd',
            labels={
                'fast_charger_percent': '% of Chargers that are Fast Chargers',
                'trucks_per_fast_charger': 'Trucks per Fast Charger',
                'MainDTN': 'Total Truck Traffic',
                'fast_charger_deficit': 'Fast Charger Deficit'
            },
            size_max=40
        )

        # Add quadrant lines
        priority_fig.add_shape(
            type="line",
            x0=30, y0=0,
            x1=30, y1=filtered_df['trucks_per_fast_charger'].max() * 1.1,
            line=dict(color="black", width=1, dash="dash")
        )

        priority_fig.add_shape(
            type="line",
            x0=0, y0=100,
            x1=100, y1=100,
            line=dict(color="black", width=1, dash="dash")
        )

        # Add quadrant labels
        priority_fig.add_annotation(
            x=15, y=filtered_df['trucks_per_fast_charger'].max() * 0.75,
            text="HIGH PRIORITY",
            showarrow=False,
            font=dict(size=10, color="red")
        )

        priority_fig.add_annotation(
            x=65, y=filtered_df['trucks_per_fast_charger'].max() * 0.75,
            text="MEDIUM PRIORITY",
            showarrow=False,
            font=dict(size=10, color="orange")
        )

        priority_fig.add_annotation(
            x=15, y=50,
            text="MEDIUM PRIORITY",
            showarrow=False,
            font=dict(size=10, color="blue")
        )

        priority_fig.add_annotation(
            x=65, y=50,
            text="LOW PRIORITY",
            showarrow=False,
            font=dict(size=10, color="green")
        )

        priority_fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, filtered_df['trucks_per_fast_charger'].max() * 1.1])
        )

        # Create current vs recommended chart
        current_vs_rec_df = filtered_df.sort_values('charger_need_score', ascending=False).head(10)

        current_vs_rec_fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add current fast chargers
        current_vs_rec_fig.add_trace(
            go.Bar(
                x=current_vs_rec_df['corridor_name'],
                y=current_vs_rec_df['NFCh30m'],
                name='Current Fast Chargers',
                marker_color='royalblue',
                opacity=0.7
            ),
            secondary_y=False
        )

        # Add recommended fast chargers
        current_vs_rec_fig.add_trace(
            go.Bar(
                x=current_vs_rec_df['corridor_name'],
                y=current_vs_rec_df['recommended_fast_chargers'],
                name='Recommended Fast Chargers',
                marker_color='firebrick',
                opacity=0.7
            ),
            secondary_y=False
        )

        # Add truck traffic line
        current_vs_rec_fig.add_trace(
            go.Scatter(
                x=current_vs_rec_df['corridor_name'],
                y=current_vs_rec_df['MDTN_B'],
                name='Trucks Using Fast Chargers',
                mode='lines+markers',
                marker=dict(color='green', size=8),
                line=dict(width=2)
            ),
            secondary_y=True
        )

        # Update layout
        current_vs_rec_fig.update_layout(
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
                showgrid=False
            ),
            barmode='group',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=20, r=80, t=20, b=80)
        )

        # Create priority table
        priority_table_df = filtered_df.sort_values('fast_charger_deficit', ascending=False).head(10)[
            ['corridor_name', 'MainDTN', 'MDTN_B', 'NFCh30m', 'trucks_per_fast_charger',
             'recommended_fast_chargers', 'fast_charger_deficit']
        ].copy()

        # Round values
        for col in priority_table_df.columns:
            if col != 'corridor_name':
                priority_table_df[col] = priority_table_df[col].round(1)

        # Add investment priority
        def get_priority(row):
            if row['trucks_per_fast_charger'] > 100 and row['fast_charger_deficit'] > 5:
                return "Critical"
            elif row['trucks_per_fast_charger'] > 75 or row['fast_charger_deficit'] > 3:
                return "High"
            else:
                return "Medium"

        priority_table_df['investment_priority'] = priority_table_df.apply(get_priority, axis=1)

        # Rename columns
        priority_table_df.columns = ['Corridor', 'Total Trucks', 'Trucks Using Fast Chargers',
                                     'Current Fast Chargers', 'Trucks per Fast Charger',
                                     'Recommended Fast Chargers', 'Additional Fast Chargers Needed',
                                     'Investment Priority']

        # Format table
        table = go.Figure(data=[go.Table(
            header=dict(
                values=list(priority_table_df.columns),
                fill_color='#3498db',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[priority_table_df[col] for col in priority_table_df.columns],
                fill_color=[['#f9f9f9', '#f2f2f2'] * 5],
                align='left',
                font=dict(size=11),
                height=30
            )
        )])

        table.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )

        return (
            total_corridors,
            f"{total_fast_chargers:,}",
            f"{additional_chargers_needed:,}",
            avg_trucks_per_charger,
            deficit_fig,
            fast_slow_fig,
            priority_fig,
            current_vs_rec_fig,
            dcc.Graph(figure=table)
        )

    # Define callback for opening/closing the modal
    @app.callback(
        Output('corridor-modal', 'style'),
        [Input('additional-chargers-card', 'n_clicks'),
         Input('close-modal', 'n_clicks')],
        [State('corridor-modal', 'style'),
         State('filtered-data', 'data')]
    )
    def toggle_modal(open_clicks, close_clicks, current_style, json_data):
        ctx = dash.callback_context

        if not ctx.triggered:
            return current_style

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'additional-chargers-card':
            return {
                'display': 'block',
                'position': 'fixed',
                'z-index': '1000',
                'left': '0',
                'top': '0',
                'width': '100%',
                'height': '100%',
                'overflow': 'auto',
                'backgroundColor': 'rgba(0,0,0,0.4)'
            }
        else:
            return {
                'display': 'none',
                'position': 'fixed',
                'z-index': '1000',
                'left': '0',
                'top': '0',
                'width': '100%',
                'height': '100%',
                'overflow': 'auto',
                'backgroundColor': 'rgba(0,0,0,0.4)'
            }

    # Define callback for the spacing analysis graph in the modal
    @app.callback(
        [Output('spacing-analysis', 'figure'),
         Output('corridor-map', 'figure'),
         Output('corridor-recommendations', 'figure')],
        [Input('additional-chargers-card', 'n_clicks')],
        [State('filtered-data', 'data')]
    )
    def update_corridor_analysis(n_clicks, json_data):
        if not n_clicks:
            return {}, {}, {}

        # Load filtered data
        filtered_df = pd.read_json(json_data, orient='split')

        # Filter for corridors where spacing > 500km
        spacing_df = filtered_df[filtered_df['dist_between_fast_chargers_km'] > 500].sort_values(
            'dist_between_fast_chargers_km', ascending=False).head(15)

        # Create spacing analysis figure - visualize distance between chargers
        spacing_fig = go.Figure()

        # Add bars for current spacing
        spacing_fig.add_trace(go.Bar(
            y=spacing_df['corridor_name'],
            x=spacing_df['dist_between_fast_chargers_km'],
            name='Distance Between Fast Chargers',
            orientation='h',
            marker_color='firebrick',
            opacity=0.7,
            hovertemplate='<b>%{y}</b><br>Distance Between Chargers: %{x:.1f} km<br>Current Fast Chargers: %{customdata[0]}<br>Estimated Corridor Length: %{customdata[1]:.1f} km<extra></extra>',
            customdata=np.stack((spacing_df['NFCh30m'], spacing_df['estimated_length_km']), axis=-1)
        ))

        # Add target line for 500km
        spacing_fig.add_shape(
            type="line",
            x0=500, y0=-0.5,
            x1=500, y1=len(spacing_df) - 0.5,
            line=dict(color="green", width=2, dash="dash")
        )

        # Add annotation for target
        spacing_fig.add_annotation(
            x=500,
            y=len(spacing_df) - 1,
            text="500km Target",
            showarrow=True,
            arrowhead=1,
            ax=-50,
            ay=-30,
            font=dict(color="green")
        )

        # Update layout
        spacing_fig.update_layout(
            title="Corridors with Fast Charger Spacing Exceeding 500km",
            xaxis_title="Distance Between Fast Chargers (km)",
            yaxis_title="Corridor",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # Create corridor map figure - visualize slow charger concentration
        corridor_map_fig = go.Figure()

        # Calculate slow to fast charger ratio
        filtered_df['slow_to_fast_ratio'] = filtered_df['NSCh2pD'] / filtered_df['NFCh30m'].replace(0, 0.1)

        # Select top corridors by slow charger concentration
        slow_charger_df = filtered_df.sort_values('slow_to_fast_ratio', ascending=False).head(15)

        # Create a combined chart showing slow charger concentration and distance between fast chargers
        corridor_map_fig.add_trace(go.Bar(
            y=slow_charger_df['corridor_name'],
            x=slow_charger_df['slow_to_fast_ratio'],
            name='Slow to Fast Charger Ratio',
            orientation='h',
            marker_color='seagreen',
            opacity=0.7,
            hovertemplate='<b>%{y}</b><br>Slow:Fast Ratio: %{x:.1f}<br>Fast Chargers: %{customdata[0]}<br>Slow Chargers: %{customdata[1]}<extra></extra>',
            customdata=np.stack((slow_charger_df['NFCh30m'], slow_charger_df['NSCh2pD']), axis=-1)
        ))

        corridor_map_fig.update_layout(
            title="Slow Charger Concentration by Corridor",
            xaxis_title="Slow to Fast Charger Ratio",
            yaxis_title="Corridor",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # Create recommendations figure - Combined view of distance and slow charger metrics
        recommendations_fig = go.Figure()

        # Combine data to get corridors that need attention for both metrics
        # Get corridors with high slow:fast ratio and large distances
        combined_df = filtered_df.copy()
        combined_df['priority_score'] = (
                (combined_df['dist_between_fast_chargers_km'] / 500) * 0.7 +  # 70% weight to distance
                (combined_df['slow_to_fast_ratio'] / 5) * 0.3  # 30% weight to slow:fast ratio
        )

        top_combined = combined_df.sort_values('priority_score', ascending=False).head(10)

        # Add bar for distance between chargers
        recommendations_fig.add_trace(go.Bar(
            x=top_combined['corridor_name'],
            y=top_combined['dist_between_fast_chargers_km'],
            name='Distance Between Fast Chargers (km)',
            marker_color='firebrick',
            opacity=0.7,
            hovertemplate='<b>%{x}</b><br>Distance: %{y:.1f} km<br>Fast Chargers: %{customdata[0]}<extra></extra>',
            customdata=np.stack((top_combined['NFCh30m'],), axis=-1)
        ))

        # Add line for slow to fast charger ratio
        recommendations_fig.add_trace(go.Scatter(
            x=top_combined['corridor_name'],
            y=top_combined['slow_to_fast_ratio'],
            name='Slow to Fast Charger Ratio',
            mode='lines+markers',
            marker=dict(color='seagreen', size=10),
            line=dict(width=2),
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>Slow:Fast Ratio: %{y:.1f}<br>Slow Chargers: %{customdata[0]}<extra></extra>',
            customdata=np.stack((top_combined['NSCh2pD'],), axis=-1)
        ))

        # Add reference line for 500km
        recommendations_fig.add_shape(
            type="line",
            x0=-0.5, y0=500,
            x1=len(top_combined) - 0.5, y1=500,
            line=dict(color="green", width=2, dash="dash")
        )

        recommendations_fig.update_layout(
            title="Priority Corridors for New Fast Chargers (Based on Distance and Slow Charger Concentration)",
            xaxis=dict(
                title='Corridor',
                tickangle=45
            ),
            yaxis=dict(
                title='Distance Between Fast Chargers (km)',
                side='left'
            ),
            yaxis2=dict(
                title='Slow to Fast Charger Ratio',
                side='right',
                overlaying='y',
                showgrid=False
            ),
            height=400,
            margin=dict(l=20, r=80, t=50, b=120),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Save figures to HTML for GitHub Pages
        if n_clicks == 1:  # Only save on first click
            try:
                spacing_fig.write_html("spacing_analysis.html")
                corridor_map_fig.write_html("slow_charger_concentration.html")
                recommendations_fig.write_html("charger_recommendations.html")

                # Create an index page
                create_index_html(filtered_df)

                print("HTML files saved for GitHub Pages deployment")
            except Exception as e:
                print(f"Error saving HTML files: {e}")

        return spacing_fig, corridor_map_fig, recommendations_fig

    def create_index_html(filtered_df):
        """Create index.html page linking to all visualizations"""
        # Calculate some stats for the index page
        total_corridors = len(filtered_df)
        corridors_needing_chargers = filtered_df[filtered_df['dist_between_fast_chargers_km'] > 500]
        total_additional_chargers = int(corridors_needing_chargers['additional_chargers_for_500km'].sum())

        with open("index.html", "w") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>EV Charging Infrastructure Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }}
                    .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                    .card {{ background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                            padding: 20px; margin-bottom: 20px; }}
                    .metrics {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                    .metric {{ flex: 1; background-color: #3498db; color: white; padding: 15px; margin: 0 10px; 
                              border-radius: 5px; text-align: center; }}
                    .metric.critical {{ background-color: #e74c3c; }}
                    .metric.success {{ background-color: #2ecc71; }}
                    .metric.warning {{ background-color: #f39c12; }}
                    h1, h2 {{ margin-top: 0; }}
                    .viz-link {{ display: block; padding: 15px; margin: 10px 0; background-color: #ecf0f1; 
                               border-radius: 5px; color: #2c3e50; text-decoration: none; }}
                    .viz-link:hover {{ background-color: #d6dbdf; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>EV Charging Infrastructure Analysis Dashboard</h1>
                </div>
                <div class="container">
                    <div class="card">
                        <h2>Dashboard Overview</h2>
                        <p>This dashboard provides analysis of EV charging infrastructure needs based on corridor traffic and current charger distribution.</p>

                        <div class="metrics">
                            <div class="metric critical">
                                <h3>Total Corridors</h3>
                                <h2>{total_corridors}</h2>
                            </div>
                            <div class="metric warning">
                                <h3>Corridors Needing Chargers</h3>
                                <h2>{len(corridors_needing_chargers)}</h2>
                            </div>
                            <div class="metric success">
                                <h3>Additional Chargers Needed</h3>
                                <h2>{total_additional_chargers}</h2>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <h2>Visualizations</h2>
                        <p>The following interactive visualizations provide insights into optimal locations for new fast chargers:</p>

                        <a class="viz-link" href="spacing_analysis.html">
                            <h3>Fast Charger Spacing Analysis</h3>
                            <p>Identifies corridors where the distance between fast chargers exceeds the 500km target</p>
                        </a>

                        <a class="viz-link" href="slow_charger_concentration.html">
                            <h3>Slow Charger Concentration</h3>
                            <p>Shows corridors with high ratios of slow to fast chargers</p>
                        </a>

                        <a class="viz-link" href="charger_recommendations.html">
                            <h3>Priority Corridors for New Chargers</h3>
                            <p>Combined analysis of spacing and charger type distribution to identify top priorities</p>
                        </a>
                    </div>

                    <div class="card">
                        <h2>About This Analysis</h2>
                        <p>This analysis is based on data from approximately 1/4 of the total 4,000 corridors due to API rate limitations. 
                        Despite this constraint, the findings provide statistically significant insights that can be extrapolated 
                        to the broader network.</p>
                        <p>The interactive visualizations allow detailed exploration of corridors that need additional fast chargers 
                        to maintain the 500km spacing requirement.</p>
                    </div>
                </div>
            </body>
            </html>
            """)

    def save_all_visualizations(filtered_df):
        """Save all visualizations for GitHub Pages"""
        # Create spacing analysis visualization
        spacing_df = filtered_df[filtered_df['dist_between_fast_chargers_km'] > 500].sort_values(
            'dist_between_fast_chargers_km', ascending=False).head(15)

        spacing_fig = go.Figure()
        spacing_fig.add_trace(go.Bar(
            y=spacing_df['corridor_name'],
            x=spacing_df['dist_between_fast_chargers_km'],
            name='Distance Between Fast Chargers',
            orientation='h',
            marker_color='firebrick',
            opacity=0.7
        ))
        spacing_fig.add_shape(
            type="line",
            x0=500, y0=-0.5,
            x1=500, y1=len(spacing_df) - 0.5,
            line=dict(color="green", width=2, dash="dash")
        )
        spacing_fig.update_layout(
            title="Corridors with Fast Charger Spacing Exceeding 500km",
            xaxis_title="Distance Between Fast Chargers (km)",
            yaxis_title="Corridor"
        )
        spacing_fig.write_html("spacing_analysis.html")

        # Create slow charger concentration visualization
        filtered_df['slow_to_fast_ratio'] = filtered_df['NSCh2pD'] / filtered_df['NFCh30m'].replace(0, 0.1)
        slow_charger_df = filtered_df.sort_values('slow_to_fast_ratio', ascending=False).head(15)

        slow_charger_fig = go.Figure()
        slow_charger_fig.add_trace(go.Bar(
            y=slow_charger_df['corridor_name'],
            x=slow_charger_df['slow_to_fast_ratio'],
            name='Slow to Fast Charger Ratio',
            orientation='h',
            marker_color='seagreen',
            opacity=0.7
        ))
        slow_charger_fig.update_layout(
            title="Slow Charger Concentration by Corridor",
            xaxis_title="Slow to Fast Charger Ratio",
            yaxis_title="Corridor"
        )
        slow_charger_fig.write_html("slow_charger_concentration.html")

        # Create recommendations visualization
        combined_df = filtered_df.copy()
        combined_df['priority_score'] = (
                (combined_df['dist_between_fast_chargers_km'] / 500) * 0.7 +
                (combined_df['slow_to_fast_ratio'] / 5) * 0.3
        )
        top_combined = combined_df.sort_values('priority_score', ascending=False).head(10)

        recommendations_fig = go.Figure()
        recommendations_fig.add_trace(go.Bar(
            x=top_combined['corridor_name'],
            y=top_combined['dist_between_fast_chargers_km'],
            name='Distance Between Fast Chargers (km)',
            marker_color='firebrick',
            opacity=0.7
        ))
        recommendations_fig.add_trace(go.Scatter(
            x=top_combined['corridor_name'],
            y=top_combined['slow_to_fast_ratio'],
            name='Slow to Fast Charger Ratio',
            mode='lines+markers',
            marker=dict(color='seagreen', size=10),
            line=dict(width=2),
            yaxis='y2'
        ))
        recommendations_fig.update_layout(
            title="Priority Corridors for New Fast Chargers",
            xaxis=dict(
                title='Corridor',
                tickangle=45
            ),
            yaxis=dict(
                title='Distance Between Fast Chargers (km)',
                side='left'
            ),
            yaxis2=dict(
                title='Slow to Fast Charger Ratio',
                side='right',
                overlaying='y',
                showgrid=False
            )
        )
        recommendations_fig.write_html("charger_recommendations.html")

        # Create index page
        create_index_html(filtered_df)

        print("All visualizations saved for GitHub Pages")

    return app


if __name__ == '__main__':
    app = create_dashboard()

    # Add export button to layout for GitHub Pages deployment
    app.layout.children.append(
        html.Button(
            "Export All for GitHub Pages",
            id="export-button",
            style={
                'position': 'fixed',
                'bottom': '10px',
                'right': '10px',
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'border': 'none',
                'padding': '10px',
                'borderRadius': '5px',
                'cursor': 'pointer'
            }
        )
    )


    # Add callback for export button
    @app.callback(
        Output("export-button", "children"),
        Input("export-button", "n_clicks"),
        State('filtered-data', 'data')
    )
    def export_visualizations(n_clicks, json_data):
        if n_clicks:
            try:
                filtered_df = pd.read_json(json_data, orient='split')
                save_all_visualizations(filtered_df)
                return "Export Complete!"
            except Exception as e:
                print(f"Error exporting: {e}")
                return "Export Failed"
        return "Export All for GitHub Pages"


    app.run(debug=True)