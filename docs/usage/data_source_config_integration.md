# Data Source Architecture and Configuration Integration

This document explains how the data source architecture integrates with the YAML configuration system in Eviz.

## Overview

The Eviz data source architecture provides a unified interface for working with different file formats (NetCDF, HDF5, CSV, GRIB) through a common API. The YAML configuration system allows users to specify data sources, variables, and visualization options in a declarative way.

The integration between these two systems is handled by the `ConfigurationAdapter` class, which bridges the gap between the YAML configuration and the data source architecture.

## Key Components

### 1. Configuration Adapter

The `ConfigurationAdapter` class is responsible for:

- Processing the YAML configuration
- Creating appropriate data sources using the data source architecture
- Applying processing and transformation options
- Handling integration and composite fields
- Managing resources

```python
from eviz.lib.config.configuration_adapter import ConfigurationAdapter

# Create a configuration adapter
adapter = ConfigurationAdapter(config_manager)

# Process the configuration
adapter.process_configuration()

# Get a data source
data_source = adapter.get_data_source(file_path)

# Get the integrated dataset
dataset = adapter.get_dataset()

# Clean up resources
adapter.close()
```

### 2. Model Extensions

Model-specific extensions provide a way to apply model-specific processing to data sources:

- `CCMExtension`: For CCM and GEOS models
- `GenericExtension`: For generic NetCDF models
- `ModelExtensionFactory`: Factory for creating model-specific extensions

```python
from eviz.models.extensions.factory import ModelExtensionFactory

# Create a model extension
extension = ModelExtensionFactory.create_extension(model_name, config_manager)

# Apply the extension to a data source
extension.process_data_source(data_source)
```

The model extensions are organized in the following directory structure:

```
eviz/
└── models/
    ├── extensions/
    │   ├── base.py             # Base ModelExtension class
    │   └── factory.py          # ModelExtensionFactory
    └── esm/
        ├── ccm_extension.py    # CCM-specific extension
        └── generic_extension.py # Generic-specific extension
```

This structure keeps the library code (eviz/lib) model-independent, while allowing for model-specific processing in the models directory.

## Configuration Structure

The YAML configuration structure has been extended to support the new data source architecture. Here's an example:

```yaml
inputs:
   - name: example.nc
     location: /path/to/data
     exp_name: Example
     format: netcdf  # Optional, can be auto-detected
     processing:
       standardize_coordinates: true
       handle_missing_values: true
       unit_conversions: true
     transformations:
       regrid:
         enabled: false
         target_grid:
           lat_res: 1.0
           lon_res: 1.0
       subset:
         enabled: false
         lat_range: [-90, 90]
         lon_range: [-180, 180]
     variables:
       temperature:
         plot_type: xy
         units: K
```

### Processing Options

The `processing` section allows you to specify how the data should be processed:

- `standardize_coordinates`: Standardize coordinate names
- `handle_missing_values`: Handle missing values
- `unit_conversions`: Apply unit conversions

### Transformation Options

The `transformations` section allows you to specify how the data should be transformed:

- `regrid`: Regrid the data to a different grid
- `subset`: Subset the data to a specific region
- `average`: Average the data over a dimension

### Integration Options

The `integration` section in `for_inputs` allows you to specify how multiple data sources should be integrated:

```yaml
for_inputs:
  integration:
    enabled: true
    method: merge
    join: outer
    align: time
```

### Composite Fields

The `composite` section in `for_inputs` allows you to create composite fields from multiple variables:

```yaml
for_inputs:
  composite:
    o3_plus_co:
      variables: [o3, co]
      operation: add
      units: ppbv
```

## Backward Compatibility

The new system is designed to be backward compatible with existing YAML configurations. If a configuration doesn't include the new options, the system will use default values.

For example, if the `processing` section is not specified, the system will use default processing options based on the model type.

## Example Usage

Here's an example of how to use the new system:

1. Create a YAML configuration file with the new options
2. Run Eviz with the configuration file:

```bash
python autoviz.py -s ccm -f config/ccm/ccm_new.yaml
```

3. The system will:
   - Parse the YAML configuration
   - Create data sources using the data source architecture
   - Apply processing and transformation options
   - Handle integration and composite fields
   - Generate visualizations

## Benefits

The integration between the YAML configuration system and the data source architecture provides several benefits:

1. **Unified Interface**: A common interface for working with different file formats
2. **Declarative Configuration**: Specify data sources, variables, and visualization options in a declarative way
3. **Model-Specific Processing**: Apply model-specific processing to data sources
4. **Advanced Data Manipulation**: Regrid, subset, and integrate data sources
5. **Composite Fields**: Create composite fields from multiple variables
6. **Resource Management**: Properly manage resources and clean up when done

## Future Enhancements

Future enhancements to the integration include:

1. **Parallel Processing**: Add support for parallel processing of large datasets
2. **Caching**: Implement caching mechanisms to improve performance
3. **Additional File Formats**: Add support for more file formats
4. **Advanced Transformations**: Implement more advanced data transformation operations
5. **Visualization Integration**: Integrate with visualization components for seamless data visualization
