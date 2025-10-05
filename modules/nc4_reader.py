"""
============================================================================
NC4 READER MODULE: NASA GLDAS NC4 to CSV Converter
============================================================================

MODULE PURPOSE:
Handles loading and validation of NetCDF4 files, extraction of coordinates,
and identification of data variables.

KEY FUNCTIONS:
- load_nc4_file: Open and validate NC4 file
- extract_coordinates: Get time, lat, lon arrays
- extract_variables: Identify all data variables
- validate_nc4_structure: Verify file has required dimensions

============================================================================
"""

import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging


def load_nc4_file(filepath: str, logger: logging.Logger) -> xr.Dataset:
    """
    Load NetCDF4 file and return xarray Dataset.
    
    INPUTS:
    - filepath (str): Path to NC4 file
    - logger (logging.Logger): Logger instance for status messages
    
    OUTPUTS:
    - xr.Dataset: Opened xarray Dataset object
    
    RAISES:
    - FileNotFoundError: If file doesn't exist
    - ValueError: If file is not valid NetCDF format
    
    FUNCTIONALITY:
    Opens NC4 file using xarray with optimal settings for GLDAS data.
    Validates file can be read before returning.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.debug(f"Opening NC4 file: {filepath.name}")
    
    try:
        # Open with decode_times for automatic datetime conversion
        dataset = xr.open_dataset(
            filepath,
            decode_times=True,
            mask_and_scale=True
        )
        
        logger.debug(f"Successfully opened file: {filepath.name}")
        return dataset
        
    except Exception as e:
        raise ValueError(f"Failed to open NC4 file: {e}")


def validate_nc4_structure(
    dataset: xr.Dataset,
    logger: logging.Logger
) -> Tuple[bool, str]:
    """
    Validate that NC4 file has required coordinate dimensions.
    
    INPUTS:
    - dataset (xr.Dataset): Opened xarray Dataset
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - Tuple[bool, str]: (is_valid, error_message)
    
    FUNCTIONALITY:
    Checks for presence of time, lat/latitude, lon/longitude dimensions.
    Returns True if valid, False with error message if invalid.
    """
    # Common coordinate name variations in GLDAS files
    time_names = ['time', 'Time', 'TIME']
    lat_names = ['lat', 'latitude', 'Latitude', 'LAT', 'LATITUDE']
    lon_names = ['lon', 'longitude', 'Longitude', 'LON', 'LONGITUDE']
    
    dims = list(dataset.dims.keys())
    coords = list(dataset.coords.keys())
    all_names = dims + coords
    
    # Check for time dimension
    has_time = any(name in all_names for name in time_names)
    if not has_time:
        return False, f"Missing time dimension. Found: {dims}"
    
    # Check for latitude dimension
    has_lat = any(name in all_names for name in lat_names)
    if not has_lat:
        return False, f"Missing latitude dimension. Found: {dims}"
    
    # Check for longitude dimension
    has_lon = any(name in all_names for name in lon_names)
    if not has_lon:
        return False, f"Missing longitude dimension. Found: {dims}"
    
    logger.debug("NC4 structure validation passed")
    return True, ""


def extract_coordinates(
    dataset: xr.Dataset,
    logger: logging.Logger
) -> Dict[str, np.ndarray]:
    """
    Extract time, latitude, and longitude coordinate arrays.
    
    INPUTS:
    - dataset (xr.Dataset): Opened xarray Dataset
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - dict: Dictionary with keys 'time', 'lat', 'lon' containing numpy arrays
    
    FUNCTIONALITY:
    Identifies coordinate variables by common naming patterns and extracts
    their values. Handles various naming conventions in GLDAS files.
    """
    # Define possible coordinate names (in priority order)
    time_names = ['time', 'Time', 'TIME']
    lat_names = ['lat', 'latitude', 'Latitude', 'LAT', 'LATITUDE']
    lon_names = ['lon', 'longitude', 'Longitude', 'LON', 'LONGITUDE']
    
    coords = {}
    all_names = list(dataset.coords.keys()) + list(dataset.dims.keys())
    
    # Extract time coordinate
    for name in time_names:
        if name in dataset.coords:
            coords['time'] = dataset.coords[name].values
            logger.debug(f"Found time coordinate: '{name}'")
            break
        elif name in dataset.dims:
            coords['time'] = np.arange(dataset.dims[name])
            logger.debug(f"Using time index from dimension: '{name}'")
            break
    
    # Extract latitude coordinate
    for name in lat_names:
        if name in dataset.coords:
            coords['lat'] = dataset.coords[name].values
            logger.debug(f"Found latitude coordinate: '{name}'")
            break
    
    # Extract longitude coordinate
    for name in lon_names:
        if name in dataset.coords:
            coords['lon'] = dataset.coords[name].values
            logger.debug(f"Found longitude coordinate: '{name}'")
            break
    
    # Verify all coordinates were found
    if len(coords) != 3:
        missing = []
        if 'time' not in coords:
            missing.append('time')
        if 'lat' not in coords:
            missing.append('latitude')
        if 'lon' not in coords:
            missing.append('longitude')
        raise ValueError(f"Could not extract coordinates: {', '.join(missing)}")
    
    logger.debug(f"Extracted coordinates: time({len(coords['time'])}), "
                f"lat({len(coords['lat'])}), lon({len(coords['lon'])})")
    
    return coords


def extract_variables(
    dataset: xr.Dataset,
    logger: logging.Logger,
    exclude_coords: bool = True
) -> List[str]:
    """
    Extract list of data variable names from dataset.
    
    INPUTS:
    - dataset (xr.Dataset): Opened xarray Dataset
    - logger (logging.Logger): Logger instance
    - exclude_coords (bool): If True, exclude coordinate variables
    
    OUTPUTS:
    - list: List of data variable names
    
    FUNCTIONALITY:
    Identifies all data variables in the dataset, excluding coordinate
    variables and dimension variables if requested.
    """
    # Get all data variables
    all_vars = list(dataset.data_vars.keys())
    
    if exclude_coords:
        # Coordinate names to exclude
        coord_names = ['time', 'Time', 'TIME',
                      'lat', 'latitude', 'Latitude', 'LAT', 'LATITUDE',
                      'lon', 'longitude', 'Longitude', 'LON', 'LONGITUDE']
        
        # Filter out coordinate variables
        data_vars = [var for var in all_vars if var not in coord_names]
    else:
        data_vars = all_vars
    
    logger.debug(f"Found {len(data_vars)} data variables")
    
    # Log first few variable names for verification
    if data_vars:
        preview = data_vars[:5]
        logger.debug(f"First variables: {', '.join(preview)}")
        if len(data_vars) > 5:
            logger.debug(f"... and {len(data_vars) - 5} more")
    
    return data_vars


def get_coordinate_names(dataset: xr.Dataset) -> Dict[str, str]:
    """
    Get actual coordinate names used in the dataset.
    
    INPUTS:
    - dataset (xr.Dataset): Opened xarray Dataset
    
    OUTPUTS:
    - dict: Dictionary mapping standard names to actual names
            e.g., {'time': 'time', 'lat': 'latitude', 'lon': 'lon'}
    
    FUNCTIONALITY:
    Identifies which naming convention is used for coordinates in this
    specific file. Useful for dimension indexing.
    """
    time_names = ['time', 'Time', 'TIME']
    lat_names = ['lat', 'latitude', 'Latitude', 'LAT', 'LATITUDE']
    lon_names = ['lon', 'longitude', 'Longitude', 'LON', 'LONGITUDE']
    
    coord_map = {}
    all_names = list(dataset.coords.keys()) + list(dataset.dims.keys())
    
    for name in time_names:
        if name in all_names:
            coord_map['time'] = name
            break
    
    for name in lat_names:
        if name in all_names:
            coord_map['lat'] = name
            break
    
    for name in lon_names:
        if name in all_names:
            coord_map['lon'] = name
            break
    
    return coord_map
