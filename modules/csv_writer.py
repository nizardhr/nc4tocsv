"""
============================================================================
CSV WRITER MODULE: NASA GLDAS NC4 to CSV Converter
============================================================================

MODULE PURPOSE:
Handles writing pandas DataFrames to CSV files with proper formatting,
encoding, and optimization for large datasets.

KEY FUNCTIONS:
- write_csv: Main function to write DataFrame to CSV
- format_datetime_column: Convert datetime objects to ISO format strings
- get_output_path: Generate appropriate output file path

============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging


def format_datetime_column(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Format datetime columns to ISO 8601 string format.
    
    INPUTS:
    - df (pd.DataFrame): DataFrame with potential datetime columns
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - pd.DataFrame: DataFrame with formatted datetime strings
    
    FUNCTIONALITY:
    Converts numpy datetime64 or pandas datetime objects in 'time' column
    to ISO 8601 formatted strings (YYYY-MM-DD HH:MM:SS).
    """
    if 'time' in df.columns:
        try:
            # Convert to pandas datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])
            
            # Format as ISO 8601 string
            df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            logger.debug("Formatted time column to ISO 8601 strings")
            
        except Exception as e:
            logger.warning(f"Could not format time column: {e}")
    
    return df


def get_output_path(
    input_filepath: str,
    output_dir: str,
    logger: logging.Logger
) -> Path:
    """
    Generate output CSV file path based on input NC4 filename.
    
    INPUTS:
    - input_filepath (str): Path to input NC4 file
    - output_dir (str): Directory for output CSV files
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - Path: Full path to output CSV file
    
    FUNCTIONALITY:
    Creates output filename by replacing .nc4 extension with .csv
    and placing in specified output directory.
    """
    input_path = Path(input_filepath)
    output_directory = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Replace .nc4 extension with .csv
    output_filename = input_path.stem + '.csv'
    output_path = output_directory / output_filename
    
    logger.debug(f"Output path: {output_path}")
    
    return output_path


def write_csv(
    df: pd.DataFrame,
    output_path: str,
    logger: logging.Logger,
    float_precision: int = 6,
    chunksize: Optional[int] = None
) -> None:
    """
    Write DataFrame to CSV file with optimal settings.
    
    INPUTS:
    - df (pd.DataFrame): DataFrame to write
    - output_path (str): Path to output CSV file
    - logger (logging.Logger): Logger instance
    - float_precision (int): Number of decimal places for floats
    - chunksize (int, optional): Write in chunks if specified (for large files)
    
    OUTPUTS:
    - None (writes file to disk)
    
    FUNCTIONALITY:
    Writes DataFrame to CSV with UTF-8 encoding, proper formatting,
    and optional chunked writing for large datasets.
    """
    output_path = Path(output_path)
    
    logger.info(f"Writing CSV to: {output_path.name}")
    logger.debug(f"DataFrame shape: {df.shape}")
    
    # Format datetime column if present
    df = format_datetime_column(df, logger)
    
    try:
        # Calculate file size estimate
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        logger.debug(f"Estimated memory usage: {memory_mb:.1f} MB")
        
        # Write to CSV
        df.to_csv(
            output_path,
            index=False,
            encoding='utf-8',
            float_format=f'%.{float_precision}f',
            chunksize=chunksize
        )
        
        # Report file size
        file_size_mb = output_path.stat().st_size / 1024**2
        logger.info(f"CSV file written successfully ({file_size_mb:.1f} MB)")
        
    except Exception as e:
        logger.error(f"Failed to write CSV file: {e}")
        raise


def validate_csv_output(
    output_path: str,
    expected_rows: int,
    expected_cols: int,
    logger: logging.Logger
) -> bool:
    """
    Validate that CSV file was written correctly.
    
    INPUTS:
    - output_path (str): Path to CSV file
    - expected_rows (int): Expected number of rows
    - expected_cols (int): Expected number of columns
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - bool: True if validation passes, False otherwise
    
    FUNCTIONALITY:
    Reads first few rows of CSV to verify structure and dimensions.
    """
    try:
        output_path = Path(output_path)
        
        if not output_path.exists():
            logger.error("CSV file does not exist")
            return False
        
        # Read first few rows to validate
        sample = pd.read_csv(output_path, nrows=5)
        
        # Check column count (accounting for header)
        if len(sample.columns) != expected_cols:
            logger.warning(f"Column count mismatch: expected {expected_cols}, "
                          f"got {len(sample.columns)}")
            return False
        
        logger.debug("CSV validation passed")
        return True
        
    except Exception as e:
        logger.error(f"CSV validation failed: {e}")
        return False


def estimate_csv_size(
    df: pd.DataFrame,
    logger: logging.Logger
) -> float:
    """
    Estimate final CSV file size in MB.
    
    INPUTS:
    - df (pd.DataFrame): DataFrame to estimate
    - logger (logging.Logger): Logger instance
    
    OUTPUTS:
    - float: Estimated file size in MB
    
    FUNCTIONALITY:
    Provides rough estimate of CSV file size based on DataFrame memory usage.
    CSV files are typically 2-3x larger than in-memory size due to text formatting.
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    
    # CSV files are typically 2.5x larger than memory due to text representation
    estimated_size = memory_mb * 2.5
    
    logger.debug(f"Estimated CSV size: {estimated_size:.1f} MB "
                f"(DataFrame memory: {memory_mb:.1f} MB)")
    
    return estimated_size


def merge_csv_files(
    csv_files: list,
    output_path: str,
    logger: logging.Logger,
    remove_source_files: bool = False
) -> None:
    """
    Merge multiple CSV files into a single combined file.
    
    INPUTS:
    - csv_files (list): List of paths to CSV files to merge
    - output_path (str): Path to combined output CSV file
    - logger (logging.Logger): Logger instance
    - remove_source_files (bool): If True, delete source CSVs after merging
    
    OUTPUTS:
    - None (writes merged file to disk)
    
    FUNCTIONALITY:
    Reads all CSV files and combines them into a single DataFrame,
    then writes to output. Optionally removes source files to save space.
    """
    if not csv_files:
        logger.warning("No CSV files to merge")
        return
    
    logger.info(f"Merging {len(csv_files)} CSV files into single output...")
    
    dataframes = []
    total_rows = 0
    
    # Read each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        try:
            logger.debug(f"Reading file {i}/{len(csv_files)}: {Path(csv_file).name}")
            df = pd.read_csv(csv_file)
            dataframes.append(df)
            total_rows += len(df)
            logger.debug(f"  Loaded {len(df):,} rows")
            
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
            continue
    
    if not dataframes:
        logger.error("No CSV files could be read successfully")
        return
    
    # Concatenate all dataframes
    logger.info(f"Combining {len(dataframes)} dataframes ({total_rows:,} total rows)...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    logger.info(f"Combined DataFrame shape: {combined_df.shape}")
    
    # Write combined CSV
    logger.info(f"Writing merged CSV to: {Path(output_path).name}")
    write_csv(combined_df, output_path, logger)
    
    # Optionally remove source files
    if remove_source_files:
        logger.info("Removing source CSV files...")
        for csv_file in csv_files:
            try:
                Path(csv_file).unlink()
                logger.debug(f"Deleted: {Path(csv_file).name}")
            except Exception as e:
                logger.warning(f"Could not delete {csv_file}: {e}")
    
    logger.info(f"Merge complete! Final file contains {len(combined_df):,} rows")