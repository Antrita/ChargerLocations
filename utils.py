"""Utility functions for memory management and large file processing"""

import os
import pandas as pd
from shapely import wkt
import geopandas as gpd


def read_shapefile_in_chunks(shapefile_path, chunk_size=10000):
    """Read a large shapefile in chunks to reduce memory usage"""
    gdf_chunks = []
    for i, gdf_chunk in enumerate(gpd.read_file(shapefile_path, rows=chunk_size)):
        print(f"Processing chunk {i + 1} ({len(gdf_chunk)} features)")
        gdf_chunks.append(gdf_chunk)

    return pd.concat(gdf_chunks)


def load_csv_with_geometry(csv_path, chunk_size=None):
    """Load a CSV with geometry column, optionally in chunks"""
    if chunk_size:
        # Process in chunks
        chunks = []
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            if 'geometry' in chunk.columns:
                chunk['geometry'] = chunk['geometry'].apply(
                    lambda x: wkt.loads(x) if isinstance(x, str) else x
                )
            chunks.append(chunk)
        return pd.concat(chunks)
    else:
        # Process in one go
        df = pd.read_csv(csv_path)
        if 'geometry' in df.columns:
            df['geometry'] = df['geometry'].apply(
                lambda x: wkt.loads(x) if isinstance(x, str) else x
            )
        return df