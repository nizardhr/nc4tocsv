"""
============================================================================
DATA PROCESSOR MODULE: NASA GLDAS NC4 to CSV Converter
============================================================================

MODULE PURPOSE:
Handles flattening of 3D NetCDF data arrays into 2D tabular format suitable
for CSV export. Creates Cartesian product of coordinates and extracts all
variable values.

KEY FUNCTIONS:
- flatten_dataset: Main function to convert xarray Dataset to pandas DataFrame
- create_coordinate_mesh: Generate all coordinate combinations
- append_variable_data: Extract and append variable values

============================================================================
"""

import xarray as xr
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from tqdm import tqdm


def create_coordinate_mesh(
    coords: Dict[str, np.ndarray],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Create Cartesian product of all coordinate combinations.
    
    INPUTS:
    - coords (dict): Dictionary with 'time', 'lat', 'lon' numpy arrays
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - pd.DataFrame: DataFrame with columns [time, lat, lon]
    
    FUNCTIONALITY:
    Generates all possible combinations of (time, lat, lon) coordinates.
    This creates the base structure for the flattened data.
    """
    logger.debug("Creating coordinate mesh...")
    
    # Get coordinate arrays
    time_array = coords['time']
    lat_array = coords['lat']
    lon_array = coords['lon']
    
    # Calculate total number of points
    n_time = len(time_array)
    n_lat = len(lat_array)
    n_lon = len(lon_array)
    total_points = n_time * n_lat * n_lon
    
    logger.debug(f"Generating {total_points:,} coordinate combinations")
    
    # Create meshgrid for all coordinate combinations
    # Using numpy's meshgrid for efficiency
    time_mesh, lat_mesh, lon_mesh = np.meshgrid(
        time_array, lat_array, lon_array,
        indexing='ij'  # 'ij' indexing matches (time, lat, lon) order
    )
    
    # Flatten meshgrids to 1D arrays
    time_flat = time_mesh.ravel()
    lat_flat = lat_mesh.ravel()
    lon_flat = lon_mesh.ravel()
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time_flat,
        'lat': lat_flat,
        'lon': lon_flat
    })
    
    logger.debug(f"Created coordinate mesh with {len(df):,} rows")
    
    return df


def extract_variable_values(
    dataset: xr.Dataset,
    variable_name: str,
    coord_names: Dict[str, str],
    logger: logging.Logger
) -> np.ndarray:
    """
    Extract flattened values for a single variable.
    
    INPUTS:
    - dataset (xr.Dataset): Opened xarray Dataset
    - variable_name (str): Name of variable to extract
    - coord_names (dict): Mapping of standard to actual coordinate names
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - np.ndarray: Flattened 1D array of variable values
    
    FUNCTIONALITY:
    Extracts variable data, ensures proper dimension ordering (time, lat, lon),
    and flattens to 1D array matching coordinate mesh order.
    """
    try:
        # Get variable data
        var_data = dataset[variable_name]
        
        # Get actual coordinate names
        time_name = coord_names.get('time', 'time')
        lat_name = coord_names.get('lat', 'lat')
        lon_name = coord_names.get('lon', 'lon')
        
        # Check if variable has the required dimensions
        var_dims = list(var_data.dims)
        
        if not all(coord in var_dims for coord in [time_name, lat_name, lon_name]):
            logger.warning(f"Variable '{variable_name}' missing required dimensions. Skipping.")
            return None
        
        # Transpose to ensure (time, lat, lon) order
        var_data = var_data.transpose(time_name, lat_name, lon_name)
        
        # Convert to numpy array and flatten
        values = var_data.values.ravel()
        
        return values
        
    except Exception as e:
        logger.warning(f"Error extracting variable '{variable_name}': {e}")
        return None


def append_variable_data(
    df: pd.DataFrame,
    dataset: xr.Dataset,
    variable_names: List[str],
    coord_names: Dict[str, str],
    logger: logging.Logger,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Append all variable data columns to coordinate DataFrame.
    
    INPUTS:
    - df (pd.DataFrame): Base DataFrame with time, lat, lon columns
    - dataset (xr.Dataset): Opened xarray Dataset
    - variable_names (list): List of variable names to extract
    - coord_names (dict): Mapping of coordinate names
    - logger (logging.Logger): Logger instance
    - show_progress (bool): Show progress bar if True
    
    OUTPUTS:
    - pd.DataFrame: DataFrame with all variables appended
    
    FUNCTIONALITY:
    Iterates through all variables, extracts their values, and appends
    as new columns to the DataFrame. Shows progress bar for tracking.
    """
    logger.info(f"Extracting {len(variable_names)} variables...")
    
    # Create progress bar if requested
    iterator = tqdm(variable_names, desc="Extracting variables") if show_progress else variable_names
    
    successful_vars = 0
    failed_vars = []
    
    for var_name in iterator:
        # Extract variable values
        values = extract_variable_values(dataset, var_name, coord_names, logger)
        
        if values is not None:
            # Verify length matches DataFrame
            if len(values) == len(df):
                df[var_name] = values
                successful_vars += 1
            else:
                logger.warning(f"Length mismatch for '{var_name}': "
                             f"expected {len(df)}, got {len(values)}. Skipping.")
                failed_vars.append(var_name)
        else:
            failed_vars.append(var_name)
    
    logger.info(f"Successfully extracted {successful_vars}/{len(variable_names)} variables")
    
    if failed_vars:
        logger.warning(f"Failed to extract {len(failed_vars)} variables: {', '.join(failed_vars[:5])}")
    
    return df


def flatten_dataset(
    dataset: xr.Dataset,
    coords: Dict[str, np.ndarray],
    variable_names: List[str],
    coord_names: Dict[str, str],
    logger: logging.Logger,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Main function to flatten 3D NetCDF dataset to 2D DataFrame.
    
    INPUTS:
    - dataset (xr.Dataset): Opened xarray Dataset
    - coords (dict): Dictionary with time, lat, lon arrays
    - variable_names (list): List of variables to extract
    - coord_names (dict): Mapping of coordinate names
    - logger (logging.Logger): Logger instance
    - show_progress (bool): Show progress bars if True
    
    OUTPUTS:
    - pd.DataFrame: Flattened DataFrame with structure:
                    [time, lat, lon, var1, var2, ..., varN]
    
    FUNCTIONALITY:
    Orchestrates the complete flattening process:
    1. Creates coordinate mesh
    2. Extracts all variable values
    3. Combines into single DataFrame
    """
    logger.info("Starting dataset flattening process...")
    
    # Step 1: Create base coordinate mesh
    df = create_coordinate_mesh(coords, logger)
    
    # Step 2: Append all variable data
    df = append_variable_data(
        df, dataset, variable_names, coord_names, logger, show_progress
    )
    
    # Step 3: Ensure column order (time, lat, lon, then variables)
    coord_cols = ['time', 'lat', 'lon']
    var_cols = [col for col in df.columns if col not in coord_cols]
    ordered_cols = coord_cols + sorted(var_cols)  # Sort variables alphabetically
    df = df[ordered_cols]
    
    logger.info(f"Flattening complete. DataFrame shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)} (3 coordinates + {len(var_cols)} variables)")
    
    return df


def optimize_dataframe_memory(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting data types.
    
    INPUTS:
    - df (pd.DataFrame): DataFrame to optimize
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - pd.DataFrame: Optimized DataFrame
    
    FUNCTIONALITY:
    Converts float64 to float32 where appropriate to reduce memory usage.
    Useful for very large datasets.
    """
    logger.debug("Optimizing DataFrame memory usage...")
    
    memory_before = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    # Convert float64 columns to float32 (except coordinates)
    for col in df.columns:
        if col not in ['time', 'lat', 'lon'] and df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    memory_after = df.memory_usage(deep=True).sum() / 1024**2  # MB
    reduction = ((memory_before - memory_after) / memory_before) * 100
    
    logger.debug(f"Memory reduced from {memory_before:.1f} MB to {memory_after:.1f} MB "
                f"({reduction:.1f}% reduction)")
    
    return df
