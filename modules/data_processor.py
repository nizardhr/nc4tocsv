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
- remove_empty_variable_rows: Remove rows with any empty variables

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


def remove_empty_variable_rows(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Remove rows where ANY variable column is empty/NaN.
    
    INPUTS:
    - df (pd.DataFrame): DataFrame to clean
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - pd.DataFrame: Cleaned DataFrame with incomplete rows removed
    
    FUNCTIONALITY:
    Identifies rows where ANY data variable (excluding time, lat, lon)
    is NaN or missing, and removes them from the DataFrame.
    Only keeps rows where ALL variables have valid values.
    """
    logger.info("Removing rows with any empty variables...")
    
    rows_before = len(df)
    
    # Get variable columns (exclude coordinates)
    coord_cols = ['time', 'lat', 'lon']
    var_cols = [col for col in df.columns if col not in coord_cols]
    
    if not var_cols:
        logger.warning("No variable columns found to check for empty rows")
        return df
    
    # Remove rows where ANY variable column is NaN (UPDATED)
    # Keep row only if all variables have values
    df_cleaned = df.dropna(subset=var_cols, how='any')
    
    rows_after = len(df_cleaned)
    rows_removed = rows_before - rows_after
    
    if rows_removed > 0:
        logger.info(f"Removed {rows_removed:,} rows with any empty variables "
                   f"({(rows_removed/rows_before*100):.1f}% of data)")
        logger.info(f"Remaining rows: {rows_after:,}")
    else:
        logger.info("No incomplete rows found - all data retained")
    
    return df_cleaned


def process_date_features(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Convert time column to date and hour features with cyclical encoding.
    
    INPUTS:
    - df (pd.DataFrame): DataFrame with 'time' column
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - pd.DataFrame: DataFrame with date, hour, and cyclical hour encoding
    
    FUNCTIONALITY:
    Creates the following features from 'time' column:
    - date: Date only (YYYY-MM-DD format, no time)
    - hour: Hour in format 00, 03, 06, 09, 12, 15, 18, or 21
    - sin_hour: Sine component of hour (cyclical encoding)
    - cos_hour: Cosine component of hour (cyclical encoding)
    - sin_day_of_year: Sine component of day of year (cyclical encoding)
    - cos_day_of_year: Cosine component of day of year (cyclical encoding)
    - sin_month: Sine component of month (cyclical encoding)
    - cos_month: Cosine component of month (cyclical encoding)
    - sin_day_of_week: Sine component of day of week (cyclical encoding)
    - cos_day_of_week: Cosine component of day of week (cyclical encoding)
    
    Then removes the original 'time' column.
    """
    logger.info("Processing date features from time column...")
    
    # Check if time column exists
    if 'time' not in df.columns:
        logger.warning("No 'time' column found - skipping date processing")
        return df
    
    # Convert time column to datetime if not already
    df['time'] = pd.to_datetime(df['time'])
    
    # Extract date only (no time component) - format as string YYYY-MM-DD
    df['date'] = df['time'].dt.strftime('%Y-%m-%d')
    # Extract hour and format as 00, 03, 06, 09, 12, 15, 18, 21
    df['hour'] = df['time'].dt.hour.astype(str).str.zfill(2)
    
    # Get numeric hour for cyclical encoding
    hour_numeric = df['time'].dt.hour
    
    # Cyclical encoding for hour (24-hour cycle)
    df['sin_hour'] = np.sin(2 * np.pi * hour_numeric / 24)
    df['cos_hour'] = np.cos(2 * np.pi * hour_numeric / 24)
    
    # Extract temporal features for cyclical encoding
    day_of_year = df['time'].dt.dayofyear
    month = df['time'].dt.month
    day_of_week = df['time'].dt.dayofweek
    
    # Cyclical encoding for day of year (handles leap years)
    df['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 366)
    df['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 366)
    
    # Cyclical encoding for month
    df['sin_month'] = np.sin(2 * np.pi * month / 12)
    df['cos_month'] = np.cos(2 * np.pi * month / 12)
    
    # Cyclical encoding for day of week
    df['sin_day_of_week'] = np.sin(2 * np.pi * day_of_week / 7)
    df['cos_day_of_week'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Reorder columns: lat, lon, date, hour, cyclical encodings, then variable columns
    coord_cols = ['lat', 'lon']
    date_cols = [
        'date', 'hour',
        'sin_hour', 'cos_hour',
        'sin_day_of_year', 'cos_day_of_year',
        'sin_month', 'cos_month',
        'sin_day_of_week', 'cos_day_of_week'
    ]
    
    # Get variable columns (all columns except time, lat, lon, and new date columns)
    all_cols = set(df.columns)
    exclude_cols = set(['time'] + coord_cols + date_cols)
    var_cols = sorted(list(all_cols - exclude_cols))
    
    # Reorder: coordinates, date features, then variables
    ordered_cols = coord_cols + date_cols + var_cols
    df = df[ordered_cols]
    
    logger.info(f"Date features created: {len(date_cols)} new columns added, 'time' column removed")
    logger.debug(f"New date columns: {', '.join(date_cols)}")
    logger.info(f"Final column order: lat, lon, date features ({len(date_cols)}), " +
               f"variables ({len(var_cols)})")
    
    




    return df