# Chapter 5: Data Pipeline

Welcome back to the eViz tutorial! In the previous chapter, [Chapter 4: Autoviz (Main Application)](04_autoviz__main_application__.md), we saw how the `Autoviz` class acts as the conductor, starting the eViz process and using the [Config Manager](02_config_manager_.md) to understand your visualization goals. But before `Autoviz` can tell the plotting components what to draw, the raw data needs to be prepared.

Scientific data, fresh out of a file, is rarely in the perfect state for immediate plotting. It might need units converting, variables combining, missing values handling, or even combining data from multiple files. Doing these steps manually for every variable and every file would be a huge task.

This is where the **Data Pipeline** comes in.

## What Problem Does the Data Pipeline Solve?

Think of your raw data files as ingredients straight from the farm – they need washing, chopping, maybe some spices added, and sometimes even combining ingredients from different sources to make a complete dish.

The Data Pipeline solves the problem of getting your raw data from its initial state in the file ([Chapter 1: Data Source](01_data_source_.md)) into a clean, standardized, and ready-to-use format for visualization. It's an automated assembly line for your data. It takes the raw data objects loaded by the [Data Source](01_data_source_.md) and puts them through a series of steps: reading (if not already done), processing (cleaning, standardizing), transforming (reshaping), and integrating (combining).

Its job is to perform all the necessary preparation steps automatically, based on the configuration settings you provide, so that the rest of eViz can work with consistent, ready-to-plot data.

## Our Central Use Case: Preparing Data from Multiple Files

Let's say your visualization task involves comparing two different model runs or overlaying observational data onto a model output. This means eViz needs to load data from *multiple* files and potentially combine them into a single dataset before plotting.

The Data Pipeline handles this use case: it reads the data from each file, processes it, and then integrates the data from different files into one coherent dataset, ready for comparison or combined plotting.

## Using the Data Pipeline (Mostly Internal)

As a user, you typically don't interact directly with the `DataPipeline` class. Instead, the [Autoviz](04_autoviz__main_application__.md) class initiates and manages the pipeline based on the configuration you provide via the [Config Manager](02_config_manager_.md).

When `Autoviz.run()` executes (as we saw in the last chapter), it creates a `ConfigurationAdapter`. This adapter then calls `process_configuration()`, and *within this step*, the Data Pipeline is set up and run implicitly, triggered by the input file list and processing instructions found in your YAML config files (parsed by the [YAML Parser](03_yaml_parser_.md)).

Your configuration settings are how you "tell" the pipeline what to do. For example, you might list multiple files in the `inputs` section:

```yaml
# Simplified config snippet
inputs:
  - name: "path/to/model_A_temp.nc"
    exp_id: "model_A"
    to_plot:
      temperature: "map"
  - name: "path/to/model_B_temp.nc"
    exp_id: "model_B"
    to_plot:
      temperature: "map"

outputs:
  # ... output settings ...

# Global input settings for comparison
for_inputs:
  compare: True
  compare_exp_ids: ["model_A", "model_B"]
  # This tells the pipeline integrator to prepare for comparing these sources
```

When the pipeline runs, it sees these inputs and the `compare: True` setting. It knows it needs to:
1.  Read `model_A_temp.nc`.
2.  Process the data from `model_A_temp.nc`.
3.  Read `model_B_temp.nc`.
4.  Process the data from `model_B_temp.nc`.
5.  Integrate (merge/concatenate) the processed datasets from `model_A` and `model_B` so they can be easily compared later.

## Breaking Down the Pipeline: The Stages

The Data Pipeline is made up of several key stages, each handled by a dedicated component class:

1.  **Reader:** (Handled by `DataReader`) Responsible for opening files ([Chapter 1: Data Source](01_data_source_.md)), reading their contents into `xarray.Dataset` objects, and handling patterns (like `*`) or URLs. It uses the [Data Source Factory](01_data_source_.md) to pick the right tool for the job.
2.  **Processor:** (Handled by `DataProcessor`) Takes the loaded `xarray.Dataset` objects and performs various cleaning and standardization steps. This includes:
    *   Validating data.
    *   Standardizing coordinate names (e.g., renaming 'latitude' to 'lat', 'longitude' to 'lon').
    *   Handling missing values.
    *   Applying common unit conversions (like Kelvin to Celsius).
    *   Applying more complex model-specific overlays or conversions (like tropopause height or specific humidity conversion, as seen in the provided code).
    *   Can also handle regridding data between different grids.
3.  **Transformer:** (Handled by `DataTransformer`) Intended for structural changes to the data, like reshaping, resampling, or selecting subsets. (Note: The provided code for `DataTransformer` is currently very basic, suggesting this stage is less developed or used in this specific version).
4.  **Integrator:** (Handled by `DataIntegrator`) Combines data from multiple sources or multiple variables within a single source. This is crucial for comparison plots or plots showing derived quantities. It can merge datasets (e.g., combining variables from different files into one `Dataset`), concatenate datasets (e.g., stacking files along a time dimension), or perform arithmetic operations on variables (like calculating differences or ratios).

## Under the Hood: How It Works

Let's look inside the `DataPipeline` class (`eviz/lib/data/pipeline/pipeline.py`).

The `DataPipeline` class acts as an orchestrator. Its `__init__` method simply creates instances of the four core components: `DataReader`, `DataProcessor`, `DataTransformer`, and `DataIntegrator`. It also holds onto the `config_manager` because the components might need configuration settings.

```python
# Simplified snippet from eviz/lib/data/pipeline/pipeline.py's __init__
class DataPipeline:
    def __init__(self, config_manager=None):
        self.logger = logging.getLogger(__name__)
        # Create instances of the worker components
        self.reader = DataReader(config_manager)
        self.processor = DataProcessor(config_manager)
        self.transformer = DataTransformer()
        self.integrator = DataIntegrator()
        
        self.data_sources = {} # Stores processed DataSource objects
        self.dataset = None # Stores the final integrated dataset
        self.config_manager = config_manager

    # ... other methods ...
```
This shows that the pipeline itself is mostly a container for its workers.

The main methods you'd call on a `DataPipeline` instance (though usually done internally by `Autoviz` via the adapter) are `process_file`, `process_files`, and `integrate_data_sources`.

Let's trace what happens when `process_file` is called:

```{mermaid}
sequenceDiagram
    participant Orchestrator as Autoviz/Adapter
    participant Pipeline as DataPipeline
    participant Reader as DataReader
    participant Processor as DataProcessor
    participant Transformer as DataTransformer
    participant DataSourceObj as DataSource

    Orchestrator->>Pipeline: process_file("file.nc", ...)
    Pipeline->>Reader: read_file("file.nc", ...)
    Reader->>DataSourceObj: Create DataSource instance (via Factory)
    DataSourceObj->>DataSourceObj: load_data("file.nc") -> xarray.Dataset
    Reader-->>Pipeline: return DataSource object
    Pipeline->>Processor: process_data_source(DataSource object)
    Processor->>DataSourceObj: Process dataset (standardize, units, etc.)
    Processor-->>Pipeline: return DataSource object
    Pipeline->>Transformer: transform_data_source(DataSource object, ...)
    Transformer->>DataSourceObj: Transform dataset (if needed)
    Transformer-->>Pipeline: return DataSource object
    Pipeline->>Pipeline: Store processed DataSource in self.data_sources
    Pipeline-->>Orchestrator: return DataSource object
```
This sequence shows how the `process_file` method delegates the work step-by-step to the Reader, then the Processor, then the Transformer. Each step operates on the `DataSource` object and updates its internal `xarray.Dataset`.

Here's a simplified look at the `process_file` method itself:

```python
# Simplified snippet from eviz/lib/data/pipeline/pipeline.py's process_file
def process_file(self, file_path: str, model_name: Optional[str] = None,
                process: bool = True, transform: bool = False,
                transform_params: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                file_format: Optional[str] = None) -> DataSource:
    """Process a single file through the pipeline."""
    self.logger.debug(f"Processing file: {file_path}")

    # Step 1: Read the data using the Reader component
    data_source = self.reader.read_file(file_path, model_name, file_format=file_format)

    # Optional: Add metadata (handled by pipeline or caller)
    if metadata and hasattr(data_source, 'metadata'):
        data_source.metadata.update(metadata)

    # Step 2: Process the data using the Processor component
    if process:
        data_source = self.processor.process_data_source(data_source)

    # Step 3: Transform the data using the Transformer component (if requested)
    if transform and transform_params:
        data_source = self.transformer.transform_data_source(data_source, **transform_params)

    # Store the resulting processed data source
    self.data_sources[file_path] = data_source

    return data_source
```
This code clearly shows the sequential calls to `self.reader`, `self.processor`, and `self.transformer`.

The `integrate_data_sources` method works similarly, but it uses the `DataIntegrator` component and operates on the collection of `DataSource` objects already stored in the pipeline:

```{mermaid}
sequenceDiagram
    participant Orchestrator as Autoviz/Adapter
    participant Pipeline as DataPipeline
    participant Integrator as DataIntegrator
    participant DataSources as Stored DataSources
    participant IntegratedDataset as xarray.Dataset

    Orchestrator->>Pipeline: integrate_data_sources(file_paths=[...], integration_params={...})
    Pipeline->>DataSources: Retrieve specified DataSource objects
    DataSources-->>Pipeline: return List of DataSources
    Pipeline->>Integrator: integrate_data_sources(List of DataSources, integration_params)
    Integrator->>Integrator: Perform merge/concat using xarray
    Integrator-->>Pipeline: return Integrated Dataset (xarray.Dataset)
    Pipeline->>Pipeline: Store integrated dataset in self.dataset
    Pipeline-->>Orchestrator: return Integrated Dataset
```
This sequence shows how integration happens *after* individual files have been processed.

Here's a simplified look at `integrate_data_sources`:

```python
# Simplified snippet from eviz/lib/data/pipeline/pipeline.py's integrate_data_sources
def integrate_data_sources(self, file_paths: Optional[List[str]] = None,
                          integration_params: Optional[Dict[str, Any]] = None) -> xr.Dataset:
    """Integrate data sources into a single dataset."""
    self.logger.debug("Integrating data sources")

    # Get the data sources to integrate (either specified or all processed)
    if file_paths:
        data_sources_to_integrate = [self.data_sources[fp] for fp in file_paths if fp in self.data_sources]
    else:
        data_sources_to_integrate = list(self.data_sources.values())

    if not data_sources_to_integrate:
        self.logger.warning("No data sources to integrate")
        return None

    integration_params = integration_params or {}
    # Call the Integrator component to perform the integration
    self.dataset = self.integrator.integrate_data_sources(data_sources_to_integrate, **integration_params)

    return self.dataset
```
This shows how the pipeline orchestrates the integration task by calling the `DataIntegrator`'s method.

The `DataIntegrator` uses `xarray`'s powerful merging and concatenating capabilities:

```python
# Simplified snippet from eviz/lib/data/pipeline/integrator.py's _merge_datasets
def _merge_datasets(self, datasets: List[xr.Dataset], **kwargs) -> xr.Dataset:
    """Merge multiple datasets along shared dimensions."""
    self.logger.debug("Merging datasets")
    if not datasets:
        return None
    
    # Uses xarray's built-in merge function
    result = xr.merge(datasets, join=kwargs.get('join', 'outer'), compat=kwargs.get('compat', 'override'))
    self.logger.info(f"Successfully merged {len(datasets)} datasets")
    return result
```
This illustrates how the components leverage existing libraries like `xarray` to perform the actual data operations.

Similarly, the `DataProcessor` uses logic for standardizing coordinates and applying conversions:

```python
# Simplified snippet from eviz/lib/data/pipeline/processor.py's _process_dataset
def _process_dataset(self, dataset: xr.Dataset, model_name: str = None) -> Optional[xr.Dataset]:
    """Process a Xarray dataset."""
    if dataset is None:
        return None

    # Calls other methods within the Processor
    dataset = self._standardize_coordinates(dataset, model_name)
    dataset = self._handle_missing_values(dataset)
    dataset = self._apply_unit_conversions(dataset)

    return dataset

# Simplified snippet from eviz/lib/data/pipeline/processor.py's _standardize_coordinates
def _standardize_coordinates(self, dataset: xr.Dataset, model_name: str = None) -> xr.Dataset:
    """Standardize dimension names in the dataset."""
    self.logger.debug(f"Standardizing coordinates for model name {model_name}")
    
    rename_dict = {}
    # Logic here to determine original names (like 'latitude') and map them to standard names ('lat')
    # ... relies on self.config_manager.meta_coords ...
    if 'latitude' in dataset.dims: # Simplified check
        rename_dict['latitude'] = 'lat'
    if 'longitude' in dataset.dims: # Simplified check
         rename_dict['longitude'] = 'lon'

    if rename_dict:
        self.logger.debug(f"Renaming dimensions: {rename_dict}")
        dataset = dataset.rename(rename_dict) # Uses xarray's rename

    return dataset
```
These snippets show the `DataProcessor` calling specific methods for each processing step and utilizing `xarray` for the renaming operation.

The Data Pipeline centralizes and automates these complex data preparation steps, ensuring that by the time the data is handed over for visualization, it is in a consistent and usable format, regardless of its original source or the processing needed.

## Summary

In this chapter, we explored the **Data Pipeline**:

*   It acts as an automated assembly line for data, taking raw files and preparing them for visualization.
*   It solves the problem of manually cleaning, standardizing, and combining data from potentially multiple sources.
*   It is composed of four main stages/components: `DataReader`, `DataProcessor`, `DataTransformer`, and `DataIntegrator`.
*   These components work sequentially on the data, often represented as `xarray.Dataset` objects within `DataSource` containers.
*   You don't typically interact with the `DataPipeline` directly; it's orchestrated internally by `Autoviz` based on your configuration settings read by the [Config Manager](02_config_manager_.md) ([YAML Parser](03_yaml_parser_.md)).
*   It leverages libraries like `xarray` to perform the heavy lifting of data manipulation.

With the data now successfully loaded, processed, and potentially integrated by the Data Pipeline into standardized `xarray.Dataset` objects, eViz needs to understand how to interpret this data within a specific scientific context. This is where the concept of a "Model" or "Source Handler" comes in.

Let's move on to [Chapter 6: Model/Source Handler](06_model_source_handler_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)