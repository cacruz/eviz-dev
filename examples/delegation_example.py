#!/usr/bin/env python
"""
Example script demonstrating how to use the delegation pattern in DataSource.

This script shows how to:
1. Load data from a NetCDF file
2. Use xarray methods directly on the DataSource object
3. Chain operations without accessing the .dataset attribute
"""

import os
import sys
import argparse
import logging
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eviz.lib.data import DataPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DataSource delegation example')
    
    parser.add_argument('--file', type=str, required=True,
                        help='Path to a NetCDF file')
    parser.add_argument('--variable', type=str, required=True,
                        help='Variable to analyze')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for the plot')
    
    return parser.parse_args()


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    
    # Create a data pipeline
    pipeline = DataPipeline()
    
    try:
        # Process the file
        print(f"Processing file: {args.file}")
        data_source = pipeline.process_file(args.file)
        
        print("\n=== Traditional approach (accessing .dataset) ===")
        # Traditional approach: access the dataset attribute
        if args.variable in data_source.dataset.data_vars:
            # Get basic statistics
            var_data = data_source.dataset[args.variable]
            min_val = var_data.min().values
            max_val = var_data.max().values
            mean_val = var_data.mean().values
            
            print(f"Variable: {args.variable}")
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")
            print(f"  Mean: {mean_val}")
            
            # Compute monthly means (if time dimension exists)
            if 'time' in var_data.dims:
                monthly_means = var_data.groupby('time.month').mean()
                print(f"  Monthly means: {monthly_means.values}")
        
        print("\n=== Delegation approach (direct method access) ===")
        # New approach with delegation: access methods directly
        if args.variable in data_source.data_vars:
            print(args.variable)
            # Get the same statistics but with direct method access
            min_val = data_source[args.variable].min().values
            max_val = data_source[args.variable].max().values
            mean_val = data_source[args.variable].mean().values
            
            print(f"Variable: {args.variable}")
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")
            print(f"  Mean: {mean_val}")
            
            # Method chaining example
            if 'time' in data_source[args.variable].dims:
                # Chain operations directly on the data_source
                monthly_means = data_source[args.variable].groupby('time.month').mean()
                print(f"  Monthly means: {monthly_means.values}")
                
                # More complex chaining example
                if 'lat' in data_source.dims and 'lon' in data_source.dims:
                    # Subset to a region, compute seasonal means, and take zonal average
                    result = (
                        data_source
                        .sel(lat=slice(-30, 30), lon=slice(0, 360))  # Subset to tropics
                        [args.variable]  # Select variable
                        .groupby('time.season')  # Group by season
                        .mean()  # Compute seasonal mean
                        .mean(dim='lon')  # Compute zonal mean
                    )
                    print(f"  Tropical seasonal zonal means: {result.values}")
        
        # Create a plot using the delegation approach
        if args.variable in data_source.data_vars and args.output:
            plt.figure(figsize=(10, 6))
            
            # Direct plotting using the data_source
            if 'lat' in data_source.dims and 'lon' in data_source.dims:
                # Plot a global map
                data_source[args.variable].mean(dim='time', skipna=True).plot()
                plt.title(f"Mean {args.variable}")
                plt.savefig(args.output)
                print(f"\nPlot saved to: {args.output}")
            else:
                # Plot a time series
                data_source[args.variable].plot()
                plt.title(f"{args.variable} Time Series")
                plt.savefig(args.output)
                print(f"\nPlot saved to: {args.output}")
        
    finally:
        # Clean up
        pipeline.close()


if __name__ == '__main__':
    main()
