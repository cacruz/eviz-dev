"""
Autoviz Command-Line Interface

This module provides the command-line interface for the eViz automatic visualization system.
It handles command-line argument parsing, logging setup, and serves as the main entry point
for the autoviz application.

The module integrates with the metadump.py tool for extracting metadata from files,
automatically invoking it when the --file option is used.
"""
import sys
import time
import subprocess
import argparse

from eviz.lib.utils import logger_setup, timer
from eviz.lib.autoviz.base import Autoviz


def parse_command_line() -> argparse.Namespace:
    """
    Parse command line arguments for the autoviz application.
    
    Returns:
        argparse.Namespace: Object containing parsed command-line arguments.
    
    This function defines and processes the command-line interface for autoviz,
    supporting various options for configuring the visualization process:
    
    Arguments:
        -s, --sources: Sources of data input (e.g., 'gridded', 'wrf')
        --compare, -m: Enable comparison mode for specified experiment names
        --file: Path to specific file(s) to be processed
        --vars: List of variables to be processed from the specified file(s)
        --configfile, -f: Path to specific configuration file
        --config, -c: Directory containing YAML configuration files
        --verbose, -v: Set logging verbosity (0=ERROR, 1=INFO, 2=DEBUG)
        --log, -l: Enable logging to file (Eviz.LOG)
        --integrate: Enable data integration from multiple files
        --composite: Create composite fields from multiple variables
    
    Example:
        >>> python autoviz.py -s gridded
        >>> python autoviz.py -s gridded -c /path/to/config
        >>> python autoviz.py -s gridded -f /path/to/config/my_config.yaml
        >>> python autoviz.py -h
    
    Note:
        If neither --file nor --sources is provided, the function will raise an error.
        When using the first example, the EVIZ_CONFIG_PATH environment variable must be defined.
    """
    parser = argparse.ArgumentParser(description='Arguments being passed')

    parser.add_argument('-s', '--sources', type=str, nargs='+', required=False,
                        help='Sources of data input')
    parser.add_argument('--compare', '-m', action='store_true',
                        help='Perform comparison for specified exp_name(s)')
    parser.add_argument('--file', nargs='+', required=False, default=None,
                        help='Enter the path of file to be processed, default=None')
    parser.add_argument('--vars', nargs='+', required=False, default=None,
                        help='Enter a space-separated list of variables to be processed, default=None')
    parser.add_argument('--configfile', '-f', nargs='+', required=False, default=None,
                        help='Enter the full config file path, default=None')
    parser.add_argument('--config', '-c', nargs='+', required=False, default=None,
                        help='Enter the directory wherein YAML specifications can be found, default=None')
    parser.add_argument('--verbose', '-v', nargs='+', required=False, default=1,
                        help='Set logging verbosity to DEBUG (2) or ERROR(0), default=1 (INFO)')
    parser.add_argument('--log', '-l', nargs='+', required=False, default=1,
                        help='Create LOG file (Eviz.LOG)')
    parser.add_argument('--integrate', action='store_true',
                        help='Integrate data from multiple files')
    parser.add_argument('--composite', nargs='+', required=False, default=None,
                        help='Create a composite field from multiple variables (format: field1,field2,operation)')

    args = parser.parse_args()

    if not args.file and not args.sources:
        parser.error("The --sources argument is required unless --file is specified.")

    return args


def main():
    """
    Main driver for the autoviz plotting tool.
    
    This function:
    
    1. Parses command-line arguments
    2. Handles metadata extraction if --file option is used
    3. Sets up logging with appropriate verbosity
    4. Initializes the Autoviz instance with specified sources
    5. Executes the visualization process
    6. Reports the total execution time
    
    The function supports two main execution paths:
    
    - Metadata extraction: When --file is specified, it invokes metadump.py to extract
      metadata from the file, optionally focusing on specific variables if --vars is provided
    - Visualization generation: Otherwise, it creates an Autoviz instance with the
      specified sources and runs the visualization process
    
    Execution time is measured and reported at the end of the process.
    
    Example::

        # Generate visualizations for gridded data
        >>> python autoviz.py -s gridded
        >>> python autoviz.py -s gridded -c /path/to/config
        >>> python autoviz.py -s gridded -f /path/to/config/my_config.yaml
        # Extract metadata from a file
        >>> python autoviz.py --file data.nc
        # Extract metadata for specific variables
        >>> python autoviz.py --file data.nc --vars temperature humidity
    """
    start_time = time.time()
    args = parse_command_line()

    if args.file and args.vars:
        subprocess.run(['python',
                        'metadump.py', args.file[0],
                        '--vars', *args.vars])
        sys.exit()
    else:
        if args.file:
            subprocess.run(['python',
                            'metadump.py', args.file[0]])
            sys.exit()

    verbose = int(args.verbose[0] if isinstance(args.verbose, list) else '1')
    log = int(args.log[0] if isinstance(args.log, list) else '1')
    logger_setup('autoviz', log=log, verbose=verbose)

    input_sources = [s.strip() for s in args.sources[0].split(',')]

    autoviz = Autoviz(input_sources, args=args)
    autoviz.run()
    print(f"Time taken = {timer(start_time, time.time())}")


if __name__ == "__main__":
    main()
