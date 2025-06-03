# Chapter 3: DataPipeline

Welcome back! In the [previous chapter](02_configmanager_.md), we explored the **ConfigManager**, the central hub for eViz's settings and the source of the "blueprint" that tells the application what to do. We saw how it loads your instructions from configuration files and command-line arguments.

Now that we have the instructions, how do we get the raw data ready to follow those instructions? Raw data, often stored in files, is rarely in the perfect format needed for immediate plotting. It might need cleaning, reorganizing, or combining with other data.

This is where the **DataPipeline** comes in.

## What is DataPipeline and Why Do We Need It?

Think of the `DataPipeline` as an **assembly line** for your data. Raw materials (your data files) enter one end, pass through various workstations where specific tasks are performed (like cleaning, transforming, or combining), and come out the other end as refined, ready-to-use products for plotting.

Why do we need this? Because preparing data for visualization involves multiple distinct steps. You need to:

1.  **Load** the data from a file (or maybe several files).
2.  **Process** it: check for errors, standardize things like coordinate names, handle missing values, maybe convert units.
3.  **Transform** it: reshape it (like averaging over time or regridding to a different resolution) or select specific parts of it (subsetting).
4.  Potentially **Integrate** it: combine it with data from other files or perform calculations between different variables within the same file.

Doing all these steps manually in your own script for every different dataset would be tedious and error-prone. The `DataPipeline` provides a structured, reusable way to define and execute this sequence of operations. It orchestrates these tasks using specialized components, just like an assembly line has different stations for different jobs.

### A Simple Use Case: Getting One File Ready

Let's imagine a simple goal: You have a NetCDF file containing weather data (`weather.nc`), and you want to prepare it for plotting. This means:

1.  Reading the data from `weather.nc`.
2.  Making sure the latitude/longitude coordinates are named consistently (`lat`, `lon`).
3.  If the data is at multiple time steps, maybe averaging it over time to get a single, average state.
4.  Getting the final, processed data ready for the plotting component.

The `DataPipeline` is the system responsible for handling this entire process based on the instructions provided (usually via the [ConfigManager](02_configmanager_.md)).

### The Assembly Line Stations: Pipeline Components

The `DataPipeline` isn't one giant block of code; it's made up of several smaller, specialized components, each responsible for a specific part of the data preparation process. These are the "stations" on our data assembly line:

1.  **DataReader:** This is the entry point. Its job is purely to **read** the raw data from the specified file(s) or URL(s) and load it into memory, typically as an `xarray.Dataset` object (a common format for scientific data). It uses a [DataSourceFactory](04_datasourcefactory_.md) to figure out *how* to read different types of files.
2.  **DataProcessor:** Once the data is loaded, the `DataProcessor` takes over. It focuses on **cleaning and standardizing**. This involves tasks like renaming coordinates to standard names (`latitude` becomes `lat`), handling missing data, ensuring units are consistent, and applying initial corrections or overlays if needed (like calculating specific humidity or dealing with tropopause height based on other files, as seen in the code snippet).
3.  **DataTransformer:** After cleaning, the `DataTransformer` handles operations that **change the shape or extent** of the data. This includes **subsetting** (selecting a specific geographic region or time range), **regridding** (interpolating the data onto a different spatial grid), or **averaging/summing** over specific dimensions (like time or vertical levels).
4.  **DataIntegrator:** This component is used when you need to **combine data** from *multiple* sources or perform calculations *between variables*. For example, merging two datasets from different models, concatenating data from different time periods, or computing the difference or ratio between two variables.

The `DataPipeline` object itself acts as the **manager** of this assembly line, coordinating which components are used and in what order, passing the data from one station to the next.

### How to Use the DataPipeline (Conceptually)

As a user interacting with eViz via the command line, you typically don't *directly* instantiate and call methods on the `DataPipeline` yourself. Instead, you provide instructions through the [ConfigManager](02_configmanager_.md) (using YAML files, as we saw in Chapter 2), and **Autoviz** orchestrates the creation and execution of the `DataPipeline` as part of its overall workflow.

The main worker object for a specific data type (like the `GriddedModel` created by the `GriddedSourceFactory`, concepts we'll touch on later) will receive the [ConfigManager](02_configmanager_.md) and then use it to set up and run the `DataPipeline`.

So, from a user's perspective, "using" the pipeline means setting the right options in your configuration file, such as:

```yaml
# Part of a config file (simplified)

inputs:
  - name: weather.nc
    location: /data/my_weather_files
    exp_id: run_A
    process: true # Tell the pipeline to process this file
    transform: true # Tell the pipeline to transform this file
    transform_params: # Parameters for the transformer
      subset: true
      lat_range: [-30, 30] # Subset to tropical latitudes
      time_average: true # Average over the time dimension

# ... other config settings ...
```

When `Autoviz` runs with this config, it tells the main processing model for this data to use the `DataPipeline`. The pipeline reads these settings from the [ConfigManager](02_configmanager_.md) and executes the requested steps (read, process, subset, time average).

### Inside the DataPipeline: The Workflow

Let's look at how the `DataPipeline` manages this process internally when asked to prepare a file.

When the `DataPipeline` is initialized (it receives the [ConfigManager](02_configmanager_.md) as a blueprint):

1.  It creates instances of its worker components: `DataReader`, `DataProcessor`, `DataTransformer`, and `DataIntegrator`.

When a request comes in to process a specific file (like `weather.nc`), the `DataPipeline`'s `process_file` method is called. Here's the simplified flow:

```{mermaid}
sequenceDiagram
    participant Caller as Model/Factory
    participant DP as DataPipeline
    participant DR as DataReader
    participant DPR as DataProcessor
    participant DTR as DataTransformer
    participant DS as DataSource

    Caller->>DP: process_file("weather.nc", ...)
    DP->>DR: read_file("weather.nc")
    DR->>DR: Uses DataSourceFactory to load file
    DR-->>DS: Return DataSource object (raw data)
    DS-->>DP: Return DataSource (raw data)
    alt if process == true
        DP->>DPR: process_data_source(DataSource)
        DPR->>DPR: Standardize, clean, etc.
        DPR-->>DS: Return DataSource (processed)
        DS-->>DP: Return DataSource (processed)
    end
    alt if transform == true
        DP->>DTR: transform_data_source(DataSource, params)
        DTR->>DTR: Subset, average, regrid, etc.
        DTR-->>DS: Return DataSource (transformed)
        DS-->>DP: Return DataSource (transformed)
    end
    DP->>DP: Store the resulting DataSource
    DP-->>DS: Return the final DataSource
    DS-->>Caller: Return final DataSource (ready for plotting!)
```

This diagram shows the step-by-step process: the `DataPipeline` orchestrates the calls to its sub-components, passing the data along. Each component performs its task and returns the modified data, which the pipeline then passes to the next component.

### Code Walkthrough (Simplified)

Let's look at very simplified snippets from the `eviz/lib/data/pipeline` directory to see this in action.

First, the main `DataPipeline` class (`pipeline.py`):

```python
# eviz/lib/data/pipeline/pipeline.py (simplified)
import logging
# ... other imports ...
from eviz.lib.data.pipeline.reader import DataReader
from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.lib.data.pipeline.transformer import DataTransformer
from eviz.lib.data.pipeline.integrator import DataIntegrator


class DataPipeline:
    """Orchestrates the data processing workflow."""
    def __init__(self, config_manager=None):
        """Initialize a new DataPipeline."""
        self.logger = logging.getLogger(__name__)
        # Create instances of the worker components
        self.reader = DataReader(config_manager)
        self.processor = DataProcessor()
        self.transformer = DataTransformer()
        self.integrator = DataIntegrator()
        self.data_sources = {} # To store processed data sources
        self.config_manager = config_manager

    def process_file(self, file_path, model_name=None,
                    process=True, transform=False, transform_params=None,
                    metadata=None, file_format=None):
        """Process a single file through the pipeline."""
        self.logger.debug(f"Processing file: {file_path}")

        # Step 1: Read the file using the DataReader
        data_source = self.reader.read_file(file_path, model_name, file_format=file_format)

        # Add metadata if provided (often from config)
        if metadata and hasattr(data_source, 'metadata'):
            data_source.metadata.update(metadata)

        # Step 2: Process the data using the DataProcessor (if requested)
        if process:
            data_source = self.processor.process_data_source(data_source)

        # Step 3: Transform the data using the DataTransformer (if requested)
        if transform and transform_params:
            data_source = self.transformer.transform_data_source(data_source, **transform_params)

        # Store the resulting data source
        self.data_sources[file_path] = data_source

        return data_source

    # ... methods for processing multiple files, integrating, etc. ...

    def get_all_data_sources(self):
        """Get all processed data sources."""
        return self.data_sources.copy()

    # ... close method to clean up ...
```

This snippet shows that the `DataPipeline` constructor (`__init__`) creates its worker components (`DataReader`, `DataProcessor`, etc.). The `process_file` method then calls these components in sequence, passing the `data_source` object from one step to the next.

Let's look at a tiny piece of one of the worker components, for example, the `DataProcessor` handling coordinate standardization (`processor.py`):

```python
# eviz/lib/data/pipeline/processor.py (simplified)
import logging
import xarray as xr
# ... other imports ...

@dataclass
class DataProcessor:
    # ... attributes and __post_init__ ...

    def process_data_source(self, data_source):
        """Process a data source."""
        # ... validation ...
        data_source.dataset = self._process_dataset(data_source.dataset)
        # ... other processing steps ...
        return data_source

    def _process_dataset(self, dataset: xr.Dataset) -> xr.Dataset | None:
        """Process an Xarray dataset (called by process_data_source)."""
        if dataset is None:
            return None

        # Calls specific standardization/cleaning methods
        dataset = self._standardize_coordinates(dataset)
        dataset = self._handle_missing_values(dataset)
        dataset = self._apply_unit_conversions(dataset)

        return dataset

    def _standardize_coordinates(self, dataset: xr.Dataset) -> xr.Dataset:
        """Standardize coordinate names."""
        coord_mappings = {
            'latitude': 'lat',
            'longitude': 'lon',
            # ... other mappings ...
        }

        rename_dict = {}
        for old_name, new_name in coord_mappings.items():
            if old_name in dataset.coords and new_name not in dataset.coords:
                rename_dict[old_name] = new_name

        if rename_dict:
            dataset = dataset.rename(rename_dict)
            self.logger.debug(f"Renamed coordinates: {rename_dict}")

        # ... logic for normalizing coordinate values (lat/lon ranges) ...

        return dataset

    # ... other methods like _handle_missing_values, _apply_unit_conversions, regrid, etc. ...
```

This snippet shows that the `DataProcessor` has a main method (`process_data_source`) which calls internal helper methods like `_standardize_coordinates` to do the actual work on the `xarray.Dataset` held within the `DataSource` object.

Similarly, the `DataTransformer` has methods for subsetting or averaging (`transformer.py`):

```python
# eviz/lib/data/pipeline/transformer.py (simplified)
import logging
import xarray as xr
# ... other imports ...

@dataclass()
class DataTransformer:
    # ... attributes and __post_init__ ...

    def transform_data_source(self, data_source, **kwargs):
        """Transform a data source."""
        # ... validation ...
        data_source.dataset = self._transform_dataset(data_source.dataset, **kwargs)
        return data_source

    def _transform_dataset(self, dataset: xr.Dataset, **kwargs) -> xr.Dataset:
        """Transform a Xarray dataset (called by transform_data_source)."""
        if dataset is None:
            return None

        # Calls specific transformation methods based on kwargs (from config)
        if kwargs.get('subset', False):
            dataset = self._subset_dataset(dataset, **kwargs)

        if kwargs.get('time_average', False):
            dataset = self._time_average_dataset(dataset, **kwargs)

        # ... other transformations like regrid, vertical average/sum ...

        return dataset

    def _subset_dataset(self, dataset: xr.Dataset, **kwargs) -> xr.Dataset:
        """Subset a dataset."""
        self.logger.debug("Subsetting dataset")

        lat_range = kwargs.get('lat_range')
        lon_range = kwargs.get('lon_range')
        # ... other ranges ...

        if lat_range is not None and 'lat' in dataset.coords:
            # Use xarray's powerful selection capabilities
            dataset = dataset.sel(lat=slice(lat_range[0], lat_range[1]))

        if lon_range is not None and 'lon' in dataset.coords:
            dataset = dataset.sel(lon=slice(lon_range[0], lon_range[1]))

        # ... apply other subsetting ...

        return dataset

    # ... other methods like _time_average_dataset, _regrid_dataset, etc. ...
```

This pattern holds for `DataIntegrator` as well. Each component focuses on its specific type of operation, and the `DataPipeline` puts it all together, guided by the parameters derived from the [ConfigManager](02_configmanager_.md).

### Conclusion

In this chapter, we introduced the **DataPipeline** as the essential system within eViz that takes raw data from files and prepares it for visualization. We learned it acts like an assembly line, coordinating specialized components: the **DataReader** (loads data), **DataProcessor** (cleans and standardizes), **DataTransformer** (reshapes and subsets), and **DataIntegrator** (combines data).

While you configure its behavior through settings in the [ConfigManager](02_configmanager_.md), the `DataPipeline` object itself is the engine that executes these data preparation steps internally, handing off the data between its specialized components.

Now that we understand how the data flows through the pipeline, the next crucial question is: How does the `DataReader` know *what type* of data it's dealing with (e.g., NetCDF, CSV) and thus *how* to read it correctly? This is the job of the **[DataSourceFactory](04_datasourcefactory_.md)**, which we will explore in the next chapter.

[Next Chapter: DataSourceFactory](04_datasourcefactory_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)