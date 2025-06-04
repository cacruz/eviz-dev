# Chapter 6: Data Processing Pipeline

Welcome back to the eViz tutorial! In our journey so far, we've learned how the [Configuration System](01_configuration_system_.md) acts as eViz's control panel, how the [Autoviz Application Core](02_autoviz_application_core_.md) is the engine that reads those instructions and orchestrates the process, how the [Metadata Generator (metadump)](03_metadata_generator__metadump__.md) helps create configurations, how the [Plotting Components](04_plotting_components_.md) are the artists that actually draw the visualizations, and how [Source Models](05_source_models_.md) are the specialized workers that handle data based on its specific type.

Now, let's talk about the **Data Processing Pipeline**. Imagine you're on an assembly line. Raw materials come in one end, and they go through several stations – maybe cutting, shaping, painting, and quality checking – before the finished product comes out the other end. The Data Processing Pipeline is like this assembly line for your data. It's responsible for taking raw data *from* a file and getting it ready for the [Source Models](05_source_models_.md) to use and the [Plotting Components](04_plotting_components_.md) to visualize.

## What Problem Does the Data Processing Pipeline Solve?

When a [Source Model](05_source_models_.md) (like our `GriddedSource` from the last chapter) needs data for a variable (like `Temperature`) from a file (`my_weather_data.nc`), it can't just directly grab a number. The data often needs preparation:

1.  **Reading:** How do we open different file formats (NetCDF, CSV, GRIB, etc.)?
2.  **Standardization:** Variable or dimension names might be different (`lat` vs `latitude` vs `XLAT`). Units might need converting (Kelvin to Celsius). Missing values might need consistent handling.
3.  **Filtering/Selecting:** We might only need data for a specific time, level, or region.
4.  **Integration:** We might need to combine data from multiple files or calculate new variables from existing ones (like the difference between two model runs).

Doing all these steps manually within each [Source Model](05_source_models_.md) would lead to a lot of duplicated code and make the models overly complicated. The Data Processing Pipeline provides a structured way to handle these common data tasks.

## Your Sixth Task: Getting Processed Data for Plotting

Let's return to our `GriddedSource` model needing the `Temperature` variable from `my_weather_data.nc`. The `GriddedSource` model's job is to figure out *what* data it needs based on the [Configuration System](01_configuration_system_.md) (e.g., `Temperature` for an `xy` plot) and then ask the Data Processing Pipeline to *get* that data.

The pipeline will then perform the necessary steps:
1.  Identify the correct reader for `my_weather_data.nc`.
2.  Read the raw data for `Temperature`.
3.  Standardize the data (e.g., rename dimensions, convert units from K to C).
4.  Provide the standardized data back to the `GriddedSource` model.

The `GriddedSource` model can then confidently work with the standardized data, knowing it's in a predictable format, regardless of the original file's quirks.

## Key Concepts in the Data Processing Pipeline

The pipeline is built from several components, each responsible for a specific step in the assembly line:

1.  **`DataReader`:** This component is the entry point. Its job is to open the specified file(s), figure out the file type, and load the raw data. It uses the [Data Source Abstraction](07_data_source_abstraction_.md) and [Data Source Factory](08_data_source_factory_.md) (coming in the next chapters!) to find the right tool to read the specific file format.
2.  **`DataProcessor`:** This component takes the raw data loaded by the Reader and applies standard processing steps like renaming dimensions, handling missing values, and performing unit conversions. It ensures the data is consistent.
3.  **`DataTransformer`:** (Less commonly used in basic workflows) This component is intended for more complex transformations, like changing data representations (e.g., from a grid to scattered points), although much of this logic is currently handled elsewhere or is a placeholder for future development.
4.  **`DataIntegrator`:** This component is used when you need to combine data. It can merge datasets from different files or perform calculations to create new variables from existing ones within a dataset (like `var1 - var2`).
5.  **`DataPipeline`:** This is the main class (`eviz/lib/data/pipeline/pipeline.py`) that orchestrates the flow. It contains instances of the Reader, Processor, Transformer, and Integrator and manages the sequence of steps when you ask it to process a file or a set of files.

Think of it like this:

```{mermaid}
graph LR
    A[Raw File Data] --> B(DataReader);
    B --> C(DataProcessor);
    C --> D(DataTransformer);
    D --> E(DataIntegrator);
    E --> F[Processed Data (Ready for Models)];

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#9cf,stroke:#333,stroke-width:2px
```
The `DataPipeline` object controls this flow. It tells the Reader to read, then passes the result to the Processor, and so on.

## How the Data Processing Pipeline Gets Data

When a [Source Model](05_source_models_.md) needs data, it typically interacts with the `DataPipeline` instance that is available to it (often stored within the `ConfigManager`).

Here's a simplified flow for getting data for one file:

```{mermaid}
sequenceDiagram
    participant SourceModel as Source Model
    participant ConfigMgr as ConfigManager
    participant DataPipeline as Data Pipeline
    participant DataReader as Reader
    participant DataProcessor as Processor
    participant DataSource as Data Source (Abstraction)
    participant RawFile as Data File

    SourceModel->>ConfigMgr: "Give me the Pipeline instance!"
    ConfigMgr-->>SourceModel: Returns DataPipeline instance
    SourceModel->>DataPipeline: "Process file 'my_weather_data.nc' and get 'Temperature'!"
    DataPipeline->>DataReader: "Read file 'my_weather_data.nc'!"
    DataReader->>RawFile: Opens and reads raw data structure
    RawFile-->>DataReader: Raw data loaded into DataSource object
    DataReader-->>DataPipeline: Returns DataSource object (with raw data)
    DataPipeline->>DataProcessor: "Process this DataSource!"
    DataProcessor->>DataSource: Standardizes dims, converts units, etc.
    DataSource-->>DataProcessor: Data is now processed within DataSource
    DataProcessor-->>DataPipeline: Returns processed DataSource
    DataPipeline->>DataPipeline: Stores processed DataSource
    DataPipeline-->>SourceModel: "Here's the DataSource for 'my_weather_data.nc'!"
    SourceModel->>DataSource: Gets the 'Temperature' variable DataArray
```
The key takeaway is that the [Source Model](05_source_models_.md) doesn't handle the low-level reading or standardizing; it delegates that responsibility to the `DataPipeline`.

## Diving Deeper into the Code

Let's look at snippets of the pipeline components, keeping them very simple.

### The Orchestrator: `DataPipeline` (`eviz/lib/data/pipeline/pipeline.py`)

This class ties everything together.

```python
# --- Simplified eviz/lib/data/pipeline/pipeline.py ---
import logging
from typing import Dict, List, Optional, Any
import xarray as xr

# Import the pipeline components
from eviz.lib.data.pipeline.reader import DataReader
from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.lib.data.pipeline.transformer import DataTransformer
from eviz.lib.data.pipeline.integrator import DataIntegrator
# Import DataSource Abstraction (from next chapter)
from eviz.lib.data.sources import DataSource # Used internally

class DataPipeline:
    """
    Orchestrates the data processing workflow.
    """
    def __init__(self, config_manager=None):
        """Initialize a new DataPipeline."""
        self.logger = logging.getLogger(__name__)
        # Create instances of the pipeline stages
        self.reader = DataReader(config_manager)
        self.processor = DataProcessor(config_manager)
        self.transformer = DataTransformer() # Currently simple
        self.integrator = DataIntegrator() # For combining datasets
        self.data_sources = {} # Keep track of processed data
        self.dataset = None # For integrated dataset

    def process_file(self, file_path: str, model_name: Optional[str] = None,
                    process: bool = True, transform: bool = False,
                    transform_params: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    file_format: Optional[str] = None) -> DataSource:
        """Process a single file through the pipeline."""
        self.logger.debug(f"Processing file: {file_path}")

        # 1. Use the Reader to get the initial data
        data_source = self.reader.read_file(file_path, model_name, file_format=file_format)

        # 2. Apply Processor (if requested)
        if process:
            data_source = self.processor.process_data_source(data_source)

        # 3. Apply Transformer (if requested)
        if transform and transform_params:
            data_source = self.transformer.transform_data_source(data_source, **transform_params)

        # Store the result
        self.data_sources[file_path] = data_source

        return data_source

    # ... methods like process_files, integrate_data_sources, get_data_source ...
```

**Explanation:**

*   The `DataPipeline` constructor creates instances of the `DataReader`, `DataProcessor`, etc.
*   The key method `process_file` takes a file path and some optional parameters.
*   It calls `self.reader.read_file` first to load the data.
*   It then conditionally calls `self.processor.process_data_source` and `self.transformer.transform_data_source` if those steps are needed.
*   Finally, it stores the resulting `DataSource` object (which now contains the processed data) in `self.data_sources` and returns it.

### The Reader: `DataReader` (`eviz/lib/data/pipeline/reader.py`)

This component handles opening files.

```python
# --- Simplified eviz/lib/data/pipeline/reader.py ---
import glob
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

# Import the Data Source Factory (from next chapters)
from eviz.lib.data.factory import DataSourceFactory
# Import DataSource Abstraction (from next chapter)
from eviz.lib.data.sources import DataSource

@dataclass
class DataReader:
    """Data reading stage of the pipeline."""
    config_manager: Optional[object] = None
    data_sources: Dict = field(default_factory=dict, init=False)
    factory: DataSourceFactory = field(init=False) # Uses the Factory

    def __post_init__(self):
        """Post-initialization to set up factory."""
        # The Reader needs a Factory to create different kinds of DataSources
        self.factory = DataSourceFactory(self.config_manager)

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def read_file(self, file_path: str, model_name: Optional[str] = None, file_format: Optional[str] = None) -> DataSource:
        """Read data from a file or URL, supporting wildcards."""
        self.logger.debug(f"Reading file: {file_path}")

        # Check if already read (caching)
        if file_path in self.data_sources:
            self.logger.debug(f"Using cached data source for {file_path}")
            return self.data_sources[file_path]

        try:
            # *** Use the Factory to get the correct DataSource type ***
            # The Factory knows how to create a NetCDFDataSource, CSVDataSource, etc.
            data_source = self.factory.create_data_source(file_path, model_name, file_format=file_format)

            # *** Ask the created DataSource object to load its data ***
            # How load_data works depends on the specific DataSource subclass
            data_source.load_data(file_path)

            # Store and return
            self.data_sources[file_path] = data_source
            return data_source

        except Exception as e:
            self.logger.error(f"Error reading file: {file_path}. Exception: {e}")
            raise # Re-raise the exception

    # ... methods like read_files, get_data_source, close ...
```

**Explanation:**

*   The `DataReader` holds a `DataSourceFactory` instance.
*   The `read_file` method first checks if the data for this file has already been loaded (`self.data_sources` acts as a simple cache).
*   If not cached, it calls `self.factory.create_data_source(file_path, ...)` This is where the [Data Source Factory](08_data_source_factory_.md) magic happens – it returns an instance of the correct [Data Source Abstraction](07_data_source_abstraction_.md) class (like `NetCDFDataSource`) based on the file extension or format hint.
*   Then, it calls `data_source.load_data(file_path)`. This method is defined on the [Data Source Abstraction](07_data_source_abstraction_.md) object returned by the factory. The specific implementation (e.g., how `NetCDFDataSource.load_data` opens a `.nc` file using `xarray`) is hidden from the `DataReader`.
*   The loaded `DataSource` object is stored and returned.

### The Processor: `DataProcessor` (`eviz/lib/data/pipeline/processor.py`)

This component cleans and standardizes the data.

```python
# --- Simplified eviz/lib/data/pipeline/processor.py ---
import logging
from dataclasses import dataclass
import xarray as xr
import numpy as np # For array operations

# Import DataSource Abstraction (from next chapter)
from eviz.lib.data.sources import DataSource

@dataclass
class DataProcessor:
    """Data processing stage of the pipeline."""
    config_manager: Optional[object] = None

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")

    def process_data_source(self, data_source: DataSource) -> DataSource:
        """Process a data source."""
        self.logger.debug("Processing data source")

        if not data_source or not hasattr(data_source, 'dataset'):
            self.logger.error("Invalid data source provided")
            return data_source # Return as is if invalid

        # Get the xarray Dataset from the DataSource object
        dataset = data_source.dataset

        # Apply processing steps
        dataset = self._process_dataset(dataset, data_source.model_name)

        # Update the dataset within the DataSource object
        data_source.dataset = dataset

        return data_source

    def _process_dataset(self, dataset: xr.Dataset, model_name: str = None) -> Optional[xr.Dataset]:
        """Apply core processing steps to an xarray Dataset."""
        if dataset is None:
            return None

        # 1. Standardize coordinate names (e.g., 'latitude' -> 'lat')
        dataset = self._standardize_coordinates(dataset, model_name)

        # 2. Handle missing values
        dataset = self._handle_missing_values(dataset)

        # 3. Apply unit conversions (e.g., K -> C)
        dataset = self._apply_unit_conversions(dataset)

        # ... potentially other processing steps ...

        return dataset

    def _standardize_coordinates(self, dataset: xr.Dataset, model_name: str = None) -> xr.Dataset:
        """Standardize dimension names in the dataset."""
        self.logger.debug(f"Standardizing coordinates for model name {model_name}")

        rename_dict = {}
        available_dims = list(dataset.dims)

        # Example: Check for common lat/lon names and map to 'lat'/'lon'
        # Uses config to find model-specific names (simplified lookup)
        lat_name = self._get_model_dim_name('yc', available_dims, model_name, self.config_manager) # yc -> lat
        lon_name = self._get_model_dim_name('xc', available_dims, model_name, self.config_manager) # xc -> lon
        time_name = self._get_model_dim_name('tc', available_dims, model_name, self.config_manager) # tc -> time
        level_name = self._get_model_dim_name('zc', available_dims, model_name, self.config_manager) # zc -> lev

        if lat_name and lat_name in available_dims and lat_name != 'lat':
             rename_dict[lat_name] = 'lat'
        if lon_name and lon_name in available_dims and lon_name != 'lon':
             rename_dict[lon_name] = 'lon'
        if time_name and time_name in available_dims and time_name != 'time':
             rename_dict[time_name] = 'time'
        if level_name and level_name in available_dims and level_name != 'lev':
             rename_dict[level_name] = 'lev'

        if rename_dict:
            self.logger.debug(f"Renaming dimensions: {rename_dict}")
            dataset = dataset.rename(rename_dict) # Use xarray's rename method

        return dataset

    def _apply_unit_conversions(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply unit conversions to the dataset."""
        # Iterate through each variable in the dataset
        for var_name, var in dataset.data_vars.items():
            if 'units' in var.attrs:
                units = str(var.attrs['units']).lower()

                # Example: Convert Kelvin to Celsius for Temperature
                if units == 'k' and var_name.lower() in ['temp', 'temperature']:
                    self.logger.debug(f"Converting {var_name} from K to C")
                    var_data_celsius = var.values - 273.15 # Simple conversion
                    # Create a new DataArray with converted data and update units attribute
                    dataset[var_name] = xr.DataArray(
                        var_data_celsius, dims=var.dims, coords=var.coords, attrs=var.attrs
                    )
                    dataset[var_name].attrs['units'] = 'C'

        return dataset

    # ... methods like _handle_missing_values, regrid, compute_difference ...
    # The _get_model_dim_name helper (not shown in detail here) looks up dimension
    # names based on the source type in the ConfigManager's meta_coords section.
```

**Explanation:**

*   The `DataProcessor` takes a `DataSource` object containing an `xarray.Dataset`.
*   Its `process_data_source` method extracts the `dataset` and calls internal methods like `_standardize_coordinates`, `_handle_missing_values`, and `_apply_unit_conversions`.
*   `_standardize_coordinates` looks at the dataset's dimensions and renames common ones (like `latitude` or `XLAT`) to standard eViz names (`lat`, `lon`, `lev`, `time`) using `xarray`'s `rename` method. It uses a helper `_get_model_dim_name` that relies on the [Configuration System](01_configuration_system_.md) (specifically `meta_coords`) to find the correct names for different source types.
*   `_apply_unit_conversions` checks the `units` attribute of variables and performs simple conversions (like K to C) if needed, updating the data values and the `units` attribute.
*   Other methods like `regrid` or `compute_difference` (not shown fully) handle more advanced data manipulation tasks.

### Integrator and Transformer

*   **`DataIntegrator` (`eviz/lib/data/pipeline/integrator.py`):** This component has methods like `integrate_data_sources` (which uses `xarray.merge` or `xarray.concat` to combine datasets from different files) and `integrate_variables` (which can perform arithmetic operations like addition or subtraction on variables within a dataset to create a new composite variable).
*   **`DataTransformer` (`eviz/lib/data/pipeline/transformer.py`):** This is a placeholder for future transformation logic. Currently, its `_transform_dataset` method simply returns the dataset unchanged.

These components are used by the `DataPipeline` when specific integration or transformation steps are requested, typically orchestrated by the [Source Models](05_source_models_.md) based on the configuration.

## Connection to the Rest of eViz

The Data Processing Pipeline is a crucial link in the eViz chain:

*   It takes instructions and relevant lookups (like model-specific dimension names or unit conversion preferences) from the [Configuration System](01_configuration_system_.md) via the `ConfigManager`.
*   It is primarily used by the [Source Models](05_source_models_.md). Models tell the pipeline *which* files and variables are needed, and the pipeline returns the processed data ready for slicing and plotting.
*   The `DataReader` component *within* the pipeline relies heavily on the [Data Source Abstraction](07_data_source_abstraction_.md) and [Data Source Factory](08_data_source_factory_.md) to handle the variety of input file formats. The factory provides the right tool (a specific `DataSource` subclass), and the abstraction ensures that tool has a standard `load_data` method the Reader can call.

The pipeline hides the complexity of data loading, cleaning, and preparation from the [Source Models](05_source_models_.md), allowing the models to focus on the logic specific to their data type (e.g., how to slice a WRF grid vs. a global grid).

## Conclusion

In this chapter, we learned about the Data Processing Pipeline, which acts as an assembly line to prepare raw data from files for visualization. We explored its main components: the `DataReader` (for loading), the `DataProcessor` (for standardizing and cleaning), the `DataTransformer` (for changing representation), the `DataIntegrator` (for combining), and the main `DataPipeline` class that orchestrates the flow. We saw how a [Source Model](05_source_models_.md) asks the pipeline for data, and the pipeline handles the necessary steps, using the [Configuration System](01_configuration_system_.md) for guidance and relying on the [Data Source Abstraction](07_data_source_abstraction_.md) and [Data Source Factory](08_data_source_factory_.md) to handle different file types.

Now that we've seen how the pipeline processes data, let's take a closer look at two crucial pieces *used by* the Reader within the pipeline: the [Data Source Abstraction](07_data_source_abstraction_.md) and the [Data Source Factory](08_data_source_factory_.md), which allow eViz to read various file formats seamlessly.

[Next Chapter: Data Source Abstraction](07_data_source_abstraction_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
