"""
============================================================================
MODULES PACKAGE: NASA GLDAS NC4 to CSV Converter
============================================================================

MODULE PURPOSE:
Package initialization file exposing core functionality from submodules.

AVAILABLE MODULES:
- logger: Logging and progress tracking
- nc4_reader: NetCDF4 file reading and validation
- data_processor: Data flattening and transformation
- csv_writer: CSV file output generation

============================================================================
"""

from .logger import (
    setup_logger,
    log_file_processing,
    log_batch_summary,
    log_variable_info,
    log_dimension_info
)

from .nc4_reader import (
    load_nc4_file,
    validate_nc4_structure,
    extract_coordinates,
    extract_variables,
    get_coordinate_names
)

from .data_processor import (
    flatten_dataset,
    create_coordinate_mesh,
    append_variable_data,
    optimize_dataframe_memory
)

from .csv_writer import (
    write_csv,
    get_output_path,
    validate_csv_output,
    estimate_csv_size
)

__all__ = [
    # Logger functions
    'setup_logger',
    'log_file_processing',
    'log_batch_summary',
    'log_variable_info',
    'log_dimension_info',
    
    # NC4 Reader functions
    'load_nc4_file',
    'validate_nc4_structure',
    'extract_coordinates',
    'extract_variables',
    'get_coordinate_names',
    
    # Data Processor functions
    'flatten_dataset',
    'create_coordinate_mesh',
    'append_variable_data',
    'optimize_dataframe_memory',
    
    # CSV Writer functions
    'write_csv',
    'get_output_path',
    'validate_csv_output',
    'estimate_csv_size'
]

__version__ = '1.0.0'
__author__ = 'NASA GLDAS Data Processing Team'
