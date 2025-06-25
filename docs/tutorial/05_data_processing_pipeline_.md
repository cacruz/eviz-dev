# Chapter 5: Data Processing Pipeline

Welcome back! In the previous chapters, we've built up eViz's understanding:
*   [Chapter 1: Autoviz Application](01_autoviz_application_.md): The main director program.
*   [Chapter 2: Configuration Management](02_configuration_management_.md): The director's detailed script (YAML files).
*   [Chapter 3: Data Source Abstraction](03_data_source_abstraction_.md): How eViz works with any data format using the standard `xarray.Dataset`.
*   [Chapter 4: Data Source Factory](04_data_source_factory_.md): How eViz automatically picks the right reader for a given file.

So now eViz knows *what* data you want to visualize (from the config) and *how* to open and read it (using the right `DataSource` created by the Factory).

But what happens right *after* the data is read? Raw data isn't always perfectly ready for plotting. You might need to:

*   **Standardize** dimension names (e.g., ensure 'latitude' is always called 'lat').
*   **Clean** the data (e.g., handle missing values).
*   **Convert units** (e.g., convert temperature from Kelvin to Celsius).
*   **Calculate new values** based on existing ones (e.g., calculate relative humidity from temperature and specific humidity).
*   **Combine** data from multiple files (e.g., merge data from different time steps or different models).

Doing all these steps manually for every variable and every file would be repetitive! This is where the **Data Processing Pipeline** comes in.

## What is the Data Processing Pipeline?

Think of the **Data Processing Pipeline** as an **assembly line for your data**. Once the raw data comes off the "reading" station (handled by the [Data Source Abstraction](03_data_source_abstraction_.md) and [Data Source Factory](04_data_source_factory_.md)), it doesn't go straight to the plotting station. Instead, it moves through a series of processing stations.

Each station on the assembly line performs a specific task:

1.  **Reading:** (Already covered!) Loading the data into a standard format (`xarray.Dataset`).
2.  **Processing:** Standardizing names, cleaning, applying basic transformations.
3.  **Transforming:** Performing more complex calculations or reshaping.
4.  **Integrating:** Combining multiple datasets or variables into a single, ready-to-use dataset.

This assembly line ensures that by the time your data reaches the end of the pipeline, it's clean, standardized, and in the correct format needed for visualization.

## Our Use Case: Standardizing and Calculating Data

Let's stick with our 'gridded' data example. Suppose we have a NetCDF file where temperature is in Kelvin and the latitude dimension is called `y_coord`. Before plotting, we want to:

1.  Ensure the latitude dimension is renamed to the standard `lat`.
2.  Convert the temperature variable from Kelvin (`K`) to Celsius (`C`).
3.  Calculate a new variable called `temp_diff` by subtracting temperature from a different file (comparison).

The Data Processing Pipeline is responsible for these steps. You don't need to write code to loop through variables and apply these changes; you specify them in your configuration (or eViz uses defaults), and the Pipeline handles the execution.

## How the Data Processing Pipeline Works (High-Level)

The `DataPipeline` class is the orchestrator of this assembly line. It holds references to the different "stations" (Reader, Processor, Transformer, Integrator) and moves the data through them in the correct order.

Here's a simplified look at the flow when the `Autoviz` application tells the `ConfigManager` to get data ready for plotting:

```{mermaid}
sequenceDiagram
    participant AutovizApp as Autoviz
    participant ConfigMgr as ConfigManager
    participant DataPipeline as DataPipeline
    participant Reader as DataReader
    participant Processor as DataProcessor
    participant Integrator as DataIntegrator

    AutovizApp->>ConfigMgr: "Get data ready for plotting variable X..."
    ConfigMgr->>DataPipeline: "Process this file path..."
    DataPipeline->>Reader: read_file(file_path)
    Reader->>Reader: Use DataSourceFactory to get reader
    Reader->>Reader: Load data into xarray.Dataset
    Reader-->>DataPipeline: Return DataSource object with loaded data
    DataPipeline->>Processor: process_data_source(data_source)
    Processor->>Processor: Standardize dims, handle missing, convert units...
    Processor-->>DataPipeline: Return processed DataSource
    DataPipeline->>Integrator: integrate_data_sources([DataSource]) (if combining files)
    Integrator->>Integrator: Merge/Concatenate Datasets
    Integrator-->>DataPipeline: Return integrated Dataset
    DataPipeline->>Integrator: integrate_variables(variables, operation) (if calculating new variable)
    Integrator->>Integrator: Perform calculation on Dataset
    Integrator-->>DataPipeline: Return Dataset with new variable
    DataPipeline-->>ConfigMgr: Data is ready (either DataSource or integrated Dataset)
    ConfigMgr-->>AutovizApp: Data ready

```

In this flow:

1.  The `Autoviz` application, guided by the `ConfigManager` (which holds the configuration plan), initiates a data processing task.
2.  The `ConfigManager` tells the `DataPipeline` (often instantiated and managed by the `ConfigManager` itself) which file(s) to process.
3.  The `DataPipeline` starts by calling its `DataReader` component.
4.  The `DataReader` (as we saw in [Chapter 4](04_data_source_factory_.md)) uses the `DataSourceFactory` to get the right `DataSource` for the file and calls its `load_data` method to get an `xarray.Dataset`.
5.  The `DataPipeline` then passes the resulting `DataSource` to the `DataProcessor`.
6.  The `DataProcessor` applies standard processing steps (like renaming dimensions, handling NaNs, basic unit conversions).
7.  (If configured) The `DataPipeline` might then pass multiple processed `DataSource` objects (if you're comparing or combining files) to the `DataIntegrator`.
8.  (If configured) The `DataPipeline` might then use the `DataIntegrator` again to perform calculations that create new variables based on existing ones within a dataset.
9.  Finally, the `DataPipeline` holds onto the processed data (either in the `DataSource` objects or a newly integrated `xarray.Dataset`) making it available for plotting.

## Inside the Code: The DataPipeline Orchestrator

The central class is `DataPipeline`, found in `eviz/lib/data/pipeline/pipeline.py`.

```python
# --- File: eviz/lib/data/pipeline/pipeline.py (simplified init) ---
import logging
from eviz.lib.data.pipeline.reader import DataReader
from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.lib.data.pipeline.transformer import DataTransformer
from eviz.lib.data.pipeline.integrator import DataIntegrator

class DataPipeline:
    def __init__(self, config_manager=None):
        self.logger = logging.getLogger(__name__)
        # Initialize each component (station) of the pipeline
        self.reader = DataReader(config_manager)
        self.processor = DataProcessor(config_manager)
        self.transformer = DataTransformer() # Note: often minimal for now (TODO)
        self.integrator = DataIntegrator()
        self.data_sources = {} # Stores processed individual sources
        self.dataset = None # Stores the final integrated dataset
        self.config_manager = config_manager

    # ... other methods ...
```

The `__init__` method shows that the `DataPipeline` class is primarily a container for instances of its component classes: `DataReader`, `DataProcessor`, `DataTransformer`, and `DataIntegrator`. It also keeps track of the data sources it has processed and the final integrated dataset.

The `process_file` method is a key entry point for handling a single input file entry from the configuration:

```python
# --- File: eviz/lib/data/pipeline/pipeline.py (simplified process_file) ---
# ... imports ...
from eviz.lib.data.sources import DataSource
from typing import Optional, Dict, Any

class DataPipeline:
    # ... __init__ and other methods ...

    def process_file(self, file_path: str, model_name: Optional[str] = None,
                    process: bool = True, transform: bool = False,
                    transform_params: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    file_format: Optional[str] = None) -> DataSource:
        """Process a single file through the pipeline."""
        self.logger.debug(f"Processing file: {file_path}")

        # 1. Reading Stage
        data_source = self.reader.read_file(file_path, model_name, file_format=file_format)

        # Attach any provided metadata
        if metadata and hasattr(data_source, 'metadata'):
            data_source.metadata.update(metadata)

        # 2. Processing Stage (if enabled)
        if process:
            data_source = self.processor.process_data_source(data_source)

        # 3. Transformation Stage (if enabled and parameters provided)
        if transform and transform_params:
            data_source = self.transformer.transform_data_source(data_source, **transform_params)

        # Store the result (for potential later integration)
        self.data_sources[file_path] = data_source

        return data_source
```

This method clearly shows the sequence: it calls the `reader`, then the `processor`, then the `transformer` (if requested), passing the `DataSource` object along the line.

Methods like `integrate_data_sources` and `integrate_variables` then use the `DataIntegrator` component to combine data *after* individual sources have been processed:

```python
# --- File: eviz/lib/data/pipeline/pipeline.py (simplified integration methods) ---
# ... imports ...
from typing import List
import xarray as xr

class DataPipeline:
    # ... __init__, process_file, etc. ...

    def integrate_data_sources(self, file_paths: Optional[List[str]] = None,
                              integration_params: Optional[Dict[str, Any]] = None) -> xr.Dataset:
        """Integrate data sources into a single dataset."""
        self.logger.debug("Integrating data sources")
        # Get the specific DataSource objects needed
        if file_paths:
            sources_to_integrate = [self.data_sources[fp] for fp in file_paths if fp in self.data_sources]
        else: # Integrate all if no specific paths given
            sources_to_integrate = list(self.data_sources.values())

        if not sources_to_integrate:
            self.logger.warning("No data sources to integrate")
            return None

        integration_params = integration_params or {}
        # Call the Integrator component
        self.dataset = self.integrator.integrate_data_sources(sources_to_integrate, **integration_params)

        return self.dataset

    def integrate_variables(self, variables: List[str], operation: str, output_name: str) -> xr.Dataset:
        """Integrate multiple variables *within* the dataset (calculate new variable)."""
        self.logger.debug(f"Integrating variables {variables} with operation '{operation}'")

        if self.dataset is None:
            self.logger.warning("No dataset available for variable integration")
            return None

        # Call the Integrator component on the current integrated dataset
        self.dataset = self.integrator.integrate_variables(self.dataset, variables, operation, output_name)

        return self.dataset
```

These methods show how the `DataPipeline` directs the `DataIntegrator` to perform tasks like merging multiple `xarray.Dataset`s (from the `DataSource` objects) or calculating new variables within the current dataset.

## Inside the Code: The Pipeline Components

Let's briefly look at snippets from the component classes (`DataReader`, `DataProcessor`, `DataTransformer`, `DataIntegrator`) to see what happens *inside* each station on the assembly line.

### DataReader

As we discussed, the `DataReader` uses the [Data Source Factory](04_data_source_factory_.md) to get the correct `DataSource` instance and then calls its `load_data` method.

```python
# --- File: eviz/lib/data/pipeline/reader.py (simplified read_file core) ---
# ... imports and DataReader class definition ...

    def read_file(self, file_path: str, model_name: Optional[str] = None, file_format: Optional[str] = None) -> DataSource:
        # ... wildcard and existence checks ...

        try:
            # Use the factory to get the specific DataSource object
            data_source = self.factory.create_data_source(file_path, model_name, file_format=file_format)
            # Tell the DataSource to load its data into its internal xarray.Dataset
            data_source.load_data(file_path)
            # Store the DataSource instance for potential later use
            self.data_sources[file_path] = data_source
            return data_source

        except Exception as e:
            self.logger.error(f"Error reading file: {file_path}. Exception: {e}")
            raise # Re-raise the error
```

This snippet from `DataReader.read_file` confirms it calls the `DataSourceFactory` and then the `load_data` method on the resulting `data_source` object.

### DataProcessor

The `DataProcessor` contains the logic for standardizing and cleaning the `xarray.Dataset` stored inside the `DataSource`.

```python
# --- File: eviz/lib/data/pipeline/processor.py (simplified process_data_source) ---
# ... imports and DataProcessor class definition ...

    def process_data_source(self, data_source: DataSource) -> DataSource:
        """Process a data source."""
        # Optional: Data validation (not shown here)
        # if not data_source.validate_data(): ... error ...

        # Process the core dataset
        # Note: the processor modifies the dataset *inside* the DataSource
        data_source.dataset = self._process_dataset(data_source.dataset, data_source.model_name)

        # Optional: Extract metadata after processing
        # self._extract_metadata(data_source.dataset, data_source)

        # Apply GEOS-specific processing (as examples from the original code)
        data_source = self._apply_geos_processing(data_source)

        return data_source

    def _process_dataset(self, dataset: xr.Dataset, model_name: str = None) -> Optional[xr.Dataset]:
        """Apply core processing steps to the xarray Dataset."""
        if dataset is None:
            return None

        # Call methods for specific processing steps
        dataset = self._standardize_coordinates(dataset, model_name)
        dataset = self._handle_missing_values(dataset)
        dataset = self._apply_unit_conversions(dataset)

        return dataset
```

The `DataProcessor.process_data_source` calls `_process_dataset` which in turn calls helper methods like `_standardize_coordinates`, `_handle_missing_values`, and `_apply_unit_conversions`. These methods contain the actual logic for renaming dimensions (using mappings often loaded via [Configuration Management](02_configuration_management_.md)), filling missing values, and converting units for common variables like temperature or pressure.

```python
# --- File: eviz/lib/data/pipeline/processor.py (simplified unit conversion example) ---
# ... imports and DataProcessor class definition ...

    def _apply_unit_conversions(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply unit conversions to the dataset."""
        for var_name, var in dataset.data_vars.items():
            if 'units' not in var.attrs:
                continue

            units = var.attrs['units'].lower()

            # Example: Convert temperature from Kelvin to Celsius
            if units == 'k' and var_name.lower() in ['temp', 'temperature', 'air_temperature']:
                var_data = var.values - 273.15
                # Create a new DataArray with the converted data and update attributes
                dataset[var_name] = xr.DataArray(var_data, dims=var.dims,
                                                 coords=var.coords, attrs=var.attrs)
                dataset[var_name].attrs['units'] = 'C' # Update units attribute
                self.logger.debug(f"Converted {var_name} from K to C")

            # Example: Convert pressure from hPa to Pa
            elif units == 'hpa' and var_name.lower() in ['pressure', 'air_pressure']:
                var_data = var.values * 100
                dataset[var_name] = xr.DataArray(var_data, dims=var.dims,
                                                 coords=var.coords, attrs=var.attrs)
                dataset[var_name].attrs['units'] = 'Pa' # Update units attribute
                self.logger.debug(f"Converted {var_name} from hPa to Pa")

            # ... add more unit conversions here ...

        return dataset
```

This snippet shows a simple example of the unit conversion logic within `_apply_unit_conversions`. It loops through variables, checks their units, and if a known conversion is needed (like K to C), it performs the calculation and updates the variable's data and its `units` attribute.

### DataTransformer

Based on the provided code, the `DataTransformer` is currently a placeholder for more complex or custom transformations.

```python
# --- File: eviz/lib/data/pipeline/transformer.py (simplified) ---
# ... imports and DataTransformer class definition ...

    def transform_data_source(self, data_source: DataSource, **kwargs) -> DataSource:
        """Transform a data source (currently a placeholder)."""
        self.logger.debug("Applying placeholder data transformation")

        # The actual transformation logic would go in _transform_dataset
        data_source.dataset = self._transform_dataset(data_source.dataset)

        return data_source

    @staticmethod
    def _transform_dataset(dataset: xr.Dataset) -> xr.Dataset:
        """Transform a Xarray dataset (currently does nothing)."""
        # TODO: Implement data transformation logic
        return dataset
```

While minimal now, this component is designed to be the place for future complex transformations, keeping them separate from the more basic processing steps.

### DataIntegrator

The `DataIntegrator` handles combining multiple datasets or performing calculations *between* variables to create new ones.

```python
# --- File: eviz/lib/data/pipeline/integrator.py (simplified integrate_variables) ---
# ... imports and DataIntegrator class definition ...

    def integrate_variables(self, dataset: xr.Dataset, variables: List[str], operation: str, output_name: str) -> xr.Dataset:
        """Integrate multiple variables within a dataset (calculate new variable)."""
        self.logger.debug(f"Integrating variables {variables} with operation '{operation}'")

        # Basic checks
        if not dataset or not variables: return dataset
        for var in variables:
            if var not in dataset.data_vars:
                self.logger.warning(f"Variable '{var}' not found for integration")
                return dataset

        try:
            # Perform the specified operation
            if operation == 'add':
                result = sum(dataset[var] for var in variables)
            elif operation == 'subtract':
                 # Ensure correct number of variables for subtraction
                 if len(variables) != 2:
                     self.logger.error("Subtract requires exactly 2 variables")
                     return dataset
                 result = dataset[variables[0]] - dataset[variables[1]]
            # ... other operations like multiply, divide, mean, max, min ...
            else:
                self.logger.error(f"Unknown operation: {operation}")
                return dataset

            # Add the calculated result as a new variable to the dataset
            dataset[output_name] = result
            # Update attributes for the new variable (helpful for plotting)
            dataset[output_name].attrs['long_name'] = f"{operation.capitalize()} of {', '.join(variables)}"
            dataset[output_name].attrs['operation'] = operation
            dataset[output_name].attrs['source_variables'] = variables

            self.logger.info(f"Successfully integrated variables '{output_name}'")
            return dataset

        except Exception as e:
            self.logger.error(f"Error integrating variables: {e}")
            return dataset

    def integrate_data_sources(self, data_sources: List[DataSource], **kwargs) -> xr.Dataset:
         """Integrate multiple data sources (merge or concatenate their datasets)."""
         self.logger.debug(f"Integrating {len(data_sources)} sources")

         if not data_sources: return None

         method = kwargs.get('method', 'merge')
         datasets_to_combine = [ds.dataset for ds in data_sources if ds.dataset is not None]

         if method == 'merge':
             # Use xarray.merge
             return self._merge_datasets(datasets_to_combine, **kwargs)
         elif method == 'concatenate':
             # Use xarray.concat
             return self._concatenate_datasets(datasets_to_combine, **kwargs)
         else:
             self.logger.error(f"Unknown integration method: {method}")
             return None

    # ... _merge_datasets and _concatenate_datasets methods use xr.merge/xr.concat ...
```

This shows how `DataIntegrator.integrate_variables` performs simple arithmetic or statistical operations (`add`, `subtract`, etc.) on existing variables and adds the result back into the `xarray.Dataset` as a new variable. The `integrate_data_sources` method takes a list of `DataSource` objects, extracts their internal `xarray.Dataset`s, and uses `xarray.merge` or `xarray.concat` (via helper methods `_merge_datasets`/`_concatenate_datasets`) to combine them into a single dataset.

## Summary

In this chapter, we learned about the **Data Processing Pipeline** in eViz, which acts as an assembly line to prepare data for visualization.

*   The Pipeline consists of modular components: `DataReader`, `DataProcessor`, `DataTransformer`, and `DataIntegrator`.
*   The `DataPipeline` class orchestrates the flow, passing the `DataSource` object (containing the `xarray.Dataset`) through these stages.
*   The `DataReader` loads the data (using the [Data Source Factory](04_data_source_factory_.md)).
*   The `DataProcessor` standardizes coordinates, handles missing values, and performs basic unit conversions.
*   The `DataTransformer` is a placeholder for more complex transformations.
*   The `DataIntegrator` can combine multiple datasets (from different files) or perform calculations to create new variables within a dataset.
*   By using the pipeline, eViz ensures that data is prepared consistently and correctly before it's handed off for plotting, regardless of the original file format or necessary preprocessing steps.

With data now loaded and prepared by the Data Processing Pipeline, it's finally ready to be turned into plots! The next crucial step is deciding *how* to visualize this data. This is where the concept of **Plotter Abstraction** comes into play.

[Plotter Abstraction](06_plotter_abstraction_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)