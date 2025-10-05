#!/usr/bin/env python3
"""
============================================================================
NASA GLDAS NC4 to CSV BATCH CONVERTER
============================================================================

SCRIPT PURPOSE:
Command-line tool for batch conversion of NASA GLDAS NetCDF4 files to
CSV format. Flattens 3D data arrays (time × lat × lon) into tabular format
with one row per coordinate combination.

USAGE:
    python nc4_to_csv.py --input-dir /path/to/nc4/files --output-dir /path/to/output

REQUIRED ARGUMENTS:
    --input-dir     Directory containing NC4 files to convert
    --output-dir    Directory for output CSV files

OPTIONAL ARGUMENTS:
    --pattern       File pattern to match (default: *.nc4)
    --log-file      Log file path (default: conversion.log)
    --verbose       Enable detailed debug logging
    --no-progress   Disable progress bars

OUTPUT FORMAT:
    CSV files with structure: time, lat, lon, var1, var2, ..., varN
    One CSV file generated per input NC4 file
    Merged CSV file (merged_output.csv) if multiple files processed

============================================================================
"""

import argparse
import sys
from pathlib import Path
from typing import List
import traceback

# Import custom modules
from modules import (
    setup_logger,
    log_file_processing,
    log_batch_summary,
    log_variable_info,
    log_dimension_info,
    load_nc4_file,
    validate_nc4_structure,
    extract_coordinates,
    extract_variables,
    get_coordinate_names,
    flatten_dataset,
    remove_empty_variable_rows,
    write_csv,
    get_output_path,
    merge_csv_files
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    OUTPUTS:
    - argparse.Namespace: Parsed arguments
    
    FUNCTIONALITY:
    Defines and parses all command-line arguments for the converter.
    """
    parser = argparse.ArgumentParser(
        description='Convert NASA GLDAS NC4 files to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all NC4 files in a directory
  python nc4_to_csv.py --input-dir ./data --output-dir ./output
  
  # Convert single file with verbose logging
  python nc4_to_csv.py --input-dir ./data/file.nc4 --output-dir ./output --verbose
  
  # Convert with custom pattern
  python nc4_to_csv.py --input-dir ./data --output-dir ./output --pattern "*.nc"
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing NC4 files OR path to single NC4 file'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory for output CSV files'
    )
    
    # Optional arguments
    parser.add_argument(
        '--pattern',
        default='*.nc4',
        help='File pattern to match (default: *.nc4)'
    )
    
    parser.add_argument(
        '--log-file',
        default='conversion.log',
        help='Log file path (default: conversion.log)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable detailed debug logging'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )
    
    return parser.parse_args()


def find_nc4_files(
    input_dir: str,
    pattern: str,
    logger
) -> List[Path]:
    """
    Find all NC4 files matching the pattern.
    
    INPUTS:
    - input_dir (str): Directory to search OR path to single file
    - pattern (str): File pattern to match (ignored if input_dir is a file)
    - logger: Logger instance
    
    OUTPUTS:
    - List[Path]: List of matching file paths
    
    FUNCTIONALITY:
    If input_dir is a file, returns that file in a list.
    If input_dir is a directory, searches for files matching the pattern.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_dir}")
        return []
    
    # Check if input is a single file
    if input_path.is_file():
        logger.info(f"Processing single file: {input_path.name}")
        return [input_path]
    
    # Input is a directory - search for matching files
    if input_path.is_dir():
        files = list(input_path.glob(pattern))
        logger.info(f"Found {len(files)} files matching pattern '{pattern}'")
        return files
    
    logger.error(f"Input path is neither a file nor directory: {input_dir}")
    return []


def process_single_file(
    nc4_file: Path,
    output_dir: str,
    logger,
    show_progress: bool = True
) -> bool:
    """
    Process a single NC4 file and convert to CSV.
    
    INPUTS:
    - nc4_file (Path): Path to NC4 file
    - output_dir (str): Output directory for CSV
    - logger: Logger instance
    - show_progress (bool): Show progress bars
    
    OUTPUTS:
    - bool: True if successful, False otherwise
    
    FUNCTIONALITY:
    Complete pipeline for single file conversion:
    1. Load NC4 file
    2. Validate structure
    3. Extract coordinates and variables
    4. Flatten data
    5. Remove empty rows
    6. Write CSV
    """
    try:
        # Log processing start
        log_file_processing(logger, nc4_file.name, 'start')
        
        # Step 1: Load NC4 file
        dataset = load_nc4_file(str(nc4_file), logger)
        
        # Step 2: Validate structure
        is_valid, error_msg = validate_nc4_structure(dataset, logger)
        if not is_valid:
            log_file_processing(logger, nc4_file.name, 'error', error_msg)
            dataset.close()
            return False
        
        # Step 3: Extract coordinates
        coords = extract_coordinates(dataset, logger)
        coord_names = get_coordinate_names(dataset)
        
        # Log dimension information
        log_dimension_info(
            logger,
            len(coords['time']),
            len(coords['lat']),
            len(coords['lon'])
        )
        
        # Step 4: Extract variables
        variables = extract_variables(dataset, logger)
        
        if not variables:
            log_file_processing(logger, nc4_file.name, 'error', 
                              'No data variables found in file')
            dataset.close()
            return False
        
        # Log variable information
        log_variable_info(logger, len(variables), variables)
        
        # Step 5: Flatten dataset
        df = flatten_dataset(
            dataset,
            coords,
            variables,
            coord_names,
            logger,
            show_progress=show_progress
        )
        
        # Close dataset to free memory
        dataset.close()
        
        # Step 5.5: Remove empty rows (ADDED)
        df = remove_empty_variable_rows(df, logger)
        
        # Step 6: Write CSV
        output_path = get_output_path(str(nc4_file), output_dir, logger)
        write_csv(df, str(output_path), logger)
        
        # Log success
        log_file_processing(logger, nc4_file.name, 'success', 
                          f'Generated {len(df):,} rows, {len(df.columns)} columns')
        
        return True
        
    except Exception as e:
        # Log detailed error
        error_msg = f"{type(e).__name__}: {str(e)}"
        log_file_processing(logger, nc4_file.name, 'error', error_msg)
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        
        # Clean up
        try:
            dataset.close()
        except:
            pass
        
        return False


def main():
    """
    Main execution function for batch NC4 to CSV conversion.
    
    FUNCTIONALITY:
    1. Parse command-line arguments
    2. Setup logging
    3. Find all NC4 files
    4. Process each file
    5. Merge CSV files if multiple files processed
    6. Report summary statistics
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(args.log_file, args.verbose)
    
    logger.info("=" * 70)
    logger.info("NASA GLDAS NC4 to CSV Batch Converter")
    logger.info("=" * 70)
    logger.info(f"Input directory:  {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"File pattern:     {args.pattern}")
    logger.info("=" * 70)
    
    # Find all NC4 files
    nc4_files = find_nc4_files(args.input_dir, args.pattern, logger)
    
    if not nc4_files:
        logger.error("No NC4 files found. Exiting.")
        sys.exit(1)
    
    # Process each file
    total_files = len(nc4_files)
    successful = 0
    failed = 0
    output_csv_files = []  # Track generated CSV files (ADDED)
    
    for i, nc4_file in enumerate(nc4_files, 1):
        logger.info(f"\n[{i}/{total_files}] Processing: {nc4_file.name}")
        logger.info("-" * 70)
        
        success = process_single_file(
            nc4_file,
            args.output_dir,
            logger,
            show_progress=not args.no_progress
        )
        
        if success:
            successful += 1
            # Track the output CSV file path (ADDED)
            output_path = get_output_path(str(nc4_file), args.output_dir, logger)
            output_csv_files.append(str(output_path))
        else:
            failed += 1
    
    # Merge CSV files if multiple files were processed successfully (ADDED)
    if successful > 1:
        logger.info("\n" + "=" * 70)
        logger.info("MERGING CSV FILES")
        logger.info("=" * 70)
        
        merged_output_path = Path(args.output_dir) / "merged_output.csv"
        merge_csv_files(
            output_csv_files,
            str(merged_output_path),
            logger,
            remove_source_files=False  # Keep individual files by default
        )
    elif successful == 1:
        logger.info("\nOnly one file processed successfully - no merge needed")
    
    # Log final summary
    log_batch_summary(logger, total_files, successful, failed, 0)
    
    # Exit with appropriate code
    if failed > 0:
        logger.warning(f"Completed with {failed} failures")
        sys.exit(1)
    else:
        logger.info("All files processed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()