#!/usr/bin/env python
"""
Example script demonstrating how to use the eviz data source architecture.

This script shows how to:
1. Load data from different file formats
2. Process and transform the data
3. Integrate data from multiple sources
4. Access metadata and variables
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
    parser = argparse.ArgumentParser(description='Data source example')
    
    parser.add_argument('--files', nargs='+', required=True,
                        help='Paths to data files')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name')
    parser.add_argument('--integrate', action='store_true',
                        help='Integrate data from multiple files')
    parser.add_argument('--variable', type=str, default=None,
                        help='Variable to plot')
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
        # Process the files
        print(f"Processing {len(args.files)} files...")
        data_sources = pipeline.process_files(args.files, model_name=args.model)
        
        # Print information about the data sources
        for file_path, data_source in data_sources.items():
            print(f"\nData source: {file_path}")
            print(f"  Model: {data_source.model_name}")
            print(f"  Variables: {list(data_source.dataset.data_vars.keys())}")
            print(f"  Dimensions: {data_source.dataset.dims}")
            print(f"  Coordinates: {list(data_source.dataset.coords.keys())}")
        
        # Integrate data sources if requested
        if args.integrate and len(args.files) > 1:
            print("\nIntegrating data sources...")
            dataset = pipeline.integrate_data_sources()
            print(f"  Integrated dataset dimensions: {dataset.dims}")
            print(f"  Integrated dataset variables: {list(dataset.data_vars.keys())}")
        else:
            # Use the first data source's dataset
            dataset = data_sources[args.files[0]].dataset
        
        # Plot a variable if specified
        if args.variable:
            if args.variable in dataset.data_vars:
                print(f"\nPlotting variable: {args.variable}")
                
                # Get the variable
                var = dataset[args.variable]
                
                # Create a simple plot based on the variable's dimensions
                plt.figure(figsize=(10, 6))
                
                if len(var.dims) == 1:
                    # 1D variable
                    plt.plot(var)
                    plt.xlabel(var.dims[0])
                    plt.ylabel(args.variable)
                    
                elif len(var.dims) == 2:
                    # 2D variable
                    if 'lat' in var.dims and 'lon' in var.dims:
                        # Geospatial data
                        plt.contourf(var.lon, var.lat, var.values)
                        plt.colorbar(label=f"{args.variable}")
                        plt.xlabel('Longitude')
                        plt.ylabel('Latitude')
                    else:
                        # Gridded 2D data
                        plt.pcolormesh(var.values)
                        plt.colorbar(label=f"{args.variable}")
                        plt.xlabel(var.dims[1])
                        plt.ylabel(var.dims[0])
                
                elif len(var.dims) >= 3:
                    # 3D+ variable, plot a slice
                    if 'time' in var.dims:
                        # Time series data, plot the first time step
                        time_idx = 0
                        time_value = var.time.values[time_idx]
                        slice_data = var.isel(time=time_idx)
                        
                        if len(slice_data.dims) == 2:
                            plt.pcolormesh(slice_data.values)
                            plt.colorbar(label=f"{args.variable}")
                            plt.xlabel(slice_data.dims[1])
                            plt.ylabel(slice_data.dims[0])
                            plt.title(f"{args.variable} at time {time_value}")
                    else:
                        # Gridded 3D data, plot the first slice
                        slice_idx = 0
                        slice_dim = var.dims[0]
                        slice_value = var[slice_dim].values[slice_idx]
                        slice_data = var.isel({slice_dim: slice_idx})
                        
                        plt.pcolormesh(slice_data.values)
                        plt.colorbar(label=f"{args.variable}")
                        plt.xlabel(slice_data.dims[1])
                        plt.ylabel(slice_data.dims[0])
                        plt.title(f"{args.variable} at {slice_dim}={slice_value}")
                
                # Save or show the plot
                if args.output:
                    plt.savefig(args.output)
                    print(f"  Plot saved to: {args.output}")
                else:
                    plt.show()
            else:
                print(f"  Variable '{args.variable}' not found in the dataset")
        
    finally:
        # Clean up
        pipeline.close()


if __name__ == '__main__':
    main()
