# Eviz Data Source Architecture

This document provides an overview of the data source architecture in the Eviz project.

## Overview

The data source architecture is designed to handle various file formats commonly used in earth-system model data, including:
- NetCDF
- HDF5
- CSV
- GRIB

The architecture follows several design patterns:
1. **Factory Pattern**: For creating appropriate data source instances based on file extensions
2. **Strategy Pattern**: For handling different file formats with a common interface
3. **Pipeline Pattern**: For processing data through multiple stages

## Directory Structure

```
eviz/lib/data/
├── __init__.py           # Main package exports
├── sources/              # Data source implementations
│   ├── __init__.py
│   ├── base.py           # Base DataSource class
│   ├── netcdf_source.py  # NetCDF implementation
│   ├── hdf5_source.py    # HDF5 implementation
│   ├── csv_source.py     # CSV implementation
│   └── grib_source.py    # GRIB implementation
├── factory/              # Factory pattern implementation
│   ├── __init__.py
│   ├── registry.py       # Registry for data source types
│   └── source_factory.py # Factory for creating data sources
└── pipeline/             # Data processing pipeline
    ├── __init__.py
    ├── reader.py         # Data reading stage
    ├── processor.py      # Data processing stage
    ├── transformer.py    # Data transformation stage
    ├── integrator.py     # Data integration stage
    └── pipeline.py       # Complete pipeline
```

## Key Components

### DataSource

The `DataSource` class is the base class for all data source implementations. It provides a common interface for working with different file formats.

Key features:
- Loading data from files
- Accessing metadata
- Validating data
- Closing resources

### DataSourceFactory

The `DataSourceFactory` class creates appropriate data source instances based on file extensions.

Key features:
- Automatic detection of file formats
- Registration of custom data source types
- Support for multiple file extensions per data source type

### DataPipeline

The `DataPipeline` class provides a complete pipeline for reading, processing, transforming, and integrating data from various sources.

Key features:
- Reading data from files
- Processing data (standardizing coordinates, handling missing values, etc.)
- Transforming data (regridding, subsetting, averaging, etc.)
- Integrating data from multiple sources
- Integrating variables within a dataset

## Usage Examples

### Basic Usage

```python
from eviz.lib.data import DataPipeline

# Create a pipeline
pipeline = DataPipeline()

# Process a file
data_source = pipeline.process_file('path/to/file.nc')

# Access the dataset
dataset = data_source.dataset

# Access variables
variable = dataset['variable_name']

# Clean up
pipeline.close()
```

### Advanced Usage

```python
from eviz.lib.data import DataPipeline

# Create a pipeline
pipeline = DataPipeline()

# Process multiple files
data_sources = pipeline.process_files(['file1.nc', 'file2.nc'])

# Transform a data source
transform_params = {
    'regrid': True,
    'target_grid': {
        'lat_min': -90,
        'lat_max': 90,
        'lon_min': -180,
        'lon_max': 180,
        'lat_res': 1.0,
        'lon_res': 1.0
    }
}
data_source = pipeline.process_file('file.nc', transform=True, transform_params=transform_params)

# Integrate data sources
integration_params = {
    'method': 'merge',
    'join': 'outer'
}
dataset = pipeline.integrate_data_sources(integration_params=integration_params)

# Integrate variables
dataset = pipeline.integrate_variables(['var1', 'var2'], 'add', 'var_sum')

# Clean up
pipeline.close()
```

## Extending the Architecture

### Adding a New Data Source Type

1. Create a new class that inherits from `DataSource`
2. Implement the required methods (`load_data`, etc.)
3. Register the new class with the `DataSourceFactory`

Example:
```python
from eviz.lib.data.sources import DataSource
from eviz.lib.data.factory import DataSourceFactory

# Create a new data source class
class MyDataSource(DataSource):
    def load_data(self, file_path):
        # Implementation here
        pass

# Register the new class
factory = DataSourceFactory()
factory.register_data_source(['myext'], MyDataSource)
```

## Best Practices

1. Always close data sources when done to free resources
2. Use the pipeline for consistent data processing
3. Handle exceptions appropriately
4. Validate data before processing
5. Use appropriate data structures for large datasets (e.g., dask arrays)
