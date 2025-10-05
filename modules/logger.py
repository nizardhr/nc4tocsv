"""
============================================================================
LOGGER MODULE: NASA GLDAS NC4 to CSV Converter
============================================================================

MODULE PURPOSE:
Provides centralized logging functionality for tracking conversion progress,
errors, and batch processing statistics.

KEY FUNCTIONS:
- setup_logger: Initialize logger with file and console output
- log_file_processing: Track individual file processing status
- log_batch_summary: Report final batch statistics

============================================================================
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    log_file: Optional[str] = None,
    verbose: bool = False
) -> logging.Logger:
    """
    Setup and configure logger with both file and console handlers.
    
    INPUTS:
    - log_file (str, optional): Path to log file. If None, only console logging
    - verbose (bool): If True, set log level to DEBUG, otherwise INFO
    
    OUTPUTS:
    - logging.Logger: Configured logger instance
    
    FUNCTIONALITY:
    Creates logger with formatted output to both console and file (if specified).
    Console shows INFO+ messages, file shows DEBUG+ messages.
    """
    # Create logger instance
    logger = logging.getLogger('nc4_to_csv_converter')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter with timestamp and level
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler - INFO level or DEBUG if verbose
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - always DEBUG level
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def log_file_processing(
    logger: logging.Logger,
    filename: str,
    status: str,
    details: Optional[str] = None
) -> None:
    """
    Log the processing status of a single file.
    
    INPUTS:
    - logger (logging.Logger): Logger instance
    - filename (str): Name of file being processed
    - status (str): Processing status ('start', 'success', 'error', 'skip')
    - details (str, optional): Additional details or error message
    
    FUNCTIONALITY:
    Logs file processing events with appropriate level and formatting.
    """
    status_messages = {
        'start': f"Processing file: {filename}",
        'success': f"✓ Successfully converted: {filename}",
        'error': f"✗ Error processing: {filename}",
        'skip': f"⊗ Skipping file: {filename}"
    }
    
    message = status_messages.get(status, f"Unknown status for {filename}")
    
    if details:
        message += f" - {details}"
    
    if status == 'error':
        logger.error(message)
    elif status == 'skip':
        logger.warning(message)
    else:
        logger.info(message)


def log_batch_summary(
    logger: logging.Logger,
    total_files: int,
    successful: int,
    failed: int,
    skipped: int
) -> None:
    """
    Log final summary statistics for batch processing.
    
    INPUTS:
    - logger (logging.Logger): Logger instance
    - total_files (int): Total number of files found
    - successful (int): Number of successfully converted files
    - failed (int): Number of failed conversions
    - skipped (int): Number of skipped files
    
    FUNCTIONALITY:
    Outputs formatted summary of batch processing results.
    """
    logger.info("=" * 70)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total files found:       {total_files}")
    logger.info(f"Successfully converted:  {successful}")
    logger.info(f"Failed conversions:      {failed}")
    logger.info(f"Skipped files:           {skipped}")
    logger.info(f"Success rate:            {(successful/total_files*100):.1f}%" if total_files > 0 else "N/A")
    logger.info("=" * 70)


def log_variable_info(
    logger: logging.Logger,
    variable_count: int,
    variable_names: list,
    max_display: int = 10
) -> None:
    """
    Log information about variables found in NC4 file.
    
    INPUTS:
    - logger (logging.Logger): Logger instance
    - variable_count (int): Total number of variables
    - variable_names (list): List of variable names
    - max_display (int): Maximum number of variable names to display
    
    FUNCTIONALITY:
    Logs variable discovery information with truncation for readability.
    """
    logger.info(f"Found {variable_count} data variables")
    
    if variable_count <= max_display:
        logger.debug(f"Variables: {', '.join(variable_names)}")
    else:
        displayed = ', '.join(variable_names[:max_display])
        logger.debug(f"Variables (first {max_display}): {displayed}...")
        logger.debug(f"... and {variable_count - max_display} more")


def log_dimension_info(
    logger: logging.Logger,
    time_steps: int,
    lat_points: int,
    lon_points: int
) -> None:
    """
    Log dimension information from NC4 file.
    
    INPUTS:
    - logger (logging.Logger): Logger instance
    - time_steps (int): Number of time steps
    - lat_points (int): Number of latitude points
    - lon_points (int): Number of longitude points
    
    FUNCTIONALITY:
    Logs coordinate dimension sizes and estimates total data points.
    """
    total_points = time_steps * lat_points * lon_points
    
    logger.info(f"Dimensions: {time_steps} time steps × {lat_points} lat × {lon_points} lon")
    logger.info(f"Total data points to generate: {total_points:,}")
    
    # Warn if very large dataset
    if total_points > 10_000_000:
        logger.warning(f"Large dataset detected ({total_points:,} points). Processing may take time.")