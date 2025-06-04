# Chapter 6: Data Processing Pipeline

Welcome back! In the [previous chapters](01_autoviz_application_.md), we've seen how the **Autoviz Application** acts as the main conductor ([Chapter 1](01_autoviz_application_.md)), loads its detailed plan from **Configuration Files** ([Chapter 2](02_configuration_management_.md)), figures out how to read different file types using **Data Source Abstraction** and the **Data Source Factory** ([Chapters 3 & 4](03_data_source_factory_.md), [Chapter 4](04_data_source_factory_.md)), and understands what the data *means* using **Metadata Handling** ([Chapter 5](05_metadata_handling_.md)).

By now, eViz has loaded your data into one or more standardized **xarray Datasets**. Great! But is that data immediately ready for plotting? Often, no.

Your raw data might:
*   Use dimension names (`XLONG`, `YLAT`) different from the standard ones the plotting tools expect (`lon`, `lat`).
*   Contain missing values that need handling.
*   Have variables in units (like Kelvin) that you want to convert (to Celsius) before plotting.
*   Need calculations performed (like finding the difference between two variables).
*   Come from multiple files that need to be combined (e.g., different time steps or different variables in separate files).

This is where the **Data Processing Pipeline** comes in.

## The Data Processing Pipeline: Your Data's Assembly Line

Imagine an assembly line in a factory. Raw materials (your data files) come in at one end, and finished products (data ready for plotting) come out the other. The assembly line has different stations, each performing a specific task to get the product ready.

In eViz, the **Data Processing Pipeline** is that assembly line. It's a sequence of steps or stages that the data goes through after it's loaded and before it's sent to the plotting engine. Each stage performs a specific task to clean, standardize, calculate, or combine the data.

The main goal is to take the raw data from the **xarray Dataset(s)** produced by the **Data Source** and get it into the *exact* shape and format needed by the **Plotting Engine** ([Chapter 7](07_plotting_engine_.md)).

The key stages in the eViz pipeline include:

1.  **Reading:** Loading the data from the file(s) into an **xarray Dataset**. (We touched on this with **Data Source** and **Factory**, but the pipeline orchestrates it).
2.  **Processing:** Standardizing dimensions and coordinates, handling missing values, applying unit conversions, and performing basic calculations.
3.  **Transforming:** More complex transformations (though this stage is less developed in the current eViz compared to others).
4.  **Integrating:** Combining data from multiple sources or multiple variables within a dataset.

Data flows sequentially through these stages, getting closer to its final, plottable form at each step.

## The `DataPipeline` Class: The Pipeline Manager

The core component that manages this assembly line is the `DataPipeline` class, found in `eviz/lib/data/pipeline/pipeline.py`. This class doesn't *do* the processing itself, but it orchestrates the different stages by creating and calling specialized helper objects for each stage:

*   `DataReader`: Manages the reading stage.
*   `DataProcessor`: Manages the processing stage.
*   `DataTransformer`: Manages the transformation stage.
*   `DataIntegrator`: Manages the integration stage.

Think of the `DataPipeline` as the floor manager of the factory, telling each station (Reader, Processor, etc.) when to work and passing the data along.

## How the Pipeline Works (Simplified Flow)

Let's see how the `DataPipeline` fits into the overall eViz flow:

```{mermaid}
sequenceDiagram
    participant A as Autoviz Object
    participant CM as ConfigManager
    participant DP as DataPipeline Object
    participant DR as DataReader
    participant DS as Specific DataSource
    participant XD as xarray Dataset
    participant DPR as DataProcessor
    participant DTR as DataTransformer
    participant DINT as DataIntegrator

    A->>CM: Get File Paths & Config
    CM-->>A: file1.nc, file2.nc, process_options, integration_options
    A->>DP: Create DataPipeline(config_manager)
    A->>DP: process_files([file1.nc, file2.nc], process=True)

    DP->>DR: Create DataReader(config)
    DR->>DS: Ask Factory to create DataSource for file1.nc
    DS->>DS: load_data(file1.nc)
    DS-->>DR: xarray Dataset 1
    DR-->>DP: DataSource 1 (containing Dataset 1)

    DP->>DPR: Create DataProcessor(config)
    DPR->>DPR: process_data_source(DataSource 1)
    DPR->>DPR: _standardize_coordinates(Dataset 1)
    DPR->>DPR: _handle_missing_values(Dataset 1)
    DPR->>DPR: _apply_unit_conversions(Dataset 1)
    DPR-->>DP: Processed DataSource 1

    Note right of DP: Repeat for file2.nc...
    DP-->>A: Dictionary of processed DataSources

    A->>DP: integrate_data_sources(file_paths=[file1.nc, file2.nc], method='merge')
    DP->>DINT: Create DataIntegrator()
    DINT->>DINT: integrate_data_sources([DataSource 1, DataSource 2])
    DINT->>DINT: _merge_datasets([Dataset 1, Dataset 2])
    DINT-->>DP: Integrated xarray Dataset (now a single object)
    DP-->>A: Integrated xarray Dataset

    A->>A: Pass Integrated Dataset to Plotting Engine (Next Chapter)
```

This diagram shows that the **Autoviz Application** creates the **DataPipeline** and tells it which files to process. The `DataPipeline` uses its internal `DataReader` to load the data (which, as we know, uses the **Factory** and **Data Source**). Once loaded, `DataPipeline` sends the resulting **DataSource** object(s) through the `DataProcessor` (and potentially `DataTransformer`). Finally, it can use the `DataIntegrator` to combine data from multiple processed sources into a single **xarray Dataset**, which is then ready for plotting.

## Inside the Pipeline Stages

Let's peek into what happens inside some of these pipeline stages.

### DataReader (`eviz/lib/data/pipeline/reader.py`)

The `DataReader` is responsible for getting the data *from* the files. It heavily relies on the **Data Source Factory** ([Chapter 4](04_data_source_factory_.md)) to get the correct **DataSource** object for a given file path.

Here's a very simplified snippet of its `read_file` method:

```python
# --- File: eviz/lib/data/pipeline/reader.py (Simplified) ---
# ... imports ...
from eviz.lib.data.factory import DataSourceFactory # Uses the Factory

@dataclass
class DataReader:
    # ... attributes ...
    factory: object = field(init=False)

    def __post_init__(self):
        self.factory = DataSourceFactory(self.config_manager) # Create the Factory!

    def read_file(self, file_path: str, model_name: Optional[str] = None, file_format: Optional[str] = None) -> DataSource:
        """Read data from a file or URL."""
        self.logger.debug(f"Reading file: {file_path}")

        # ... code to handle wildcards or check if file exists ...

        # Use the factory to create the right DataSource object
        data_source = self.factory.create_data_source(file_path, model_name, file_format=file_format)

        # Tell the created DataSource object to load the data
        data_source.load_data(file_path)

        # Store and return the loaded DataSource object (with its xarray Dataset inside)
        self.data_sources[file_path] = data_source
        return data_source

    # ... other methods ...
```

This shows how `DataReader` is initialized with a `DataSourceFactory` and its `read_file` method uses the factory to get the right **DataSource** instance, then calls `load_data` on it to get the **xarray Dataset**.

### DataProcessor (`eviz/lib/data/pipeline/processor.py`)

The `DataProcessor` handles common cleaning and standardization tasks. It takes a **DataSource** object (which contains the **xarray Dataset**) and modifies the dataset within it.

Here are simplified examples of a few key methods:

```python
# --- File: eviz/lib/data/pipeline/processor.py (Simplified) ---
# ... imports ...
import xarray as xr
# ... other imports ...

@dataclass
class DataProcessor:
    config_manager: Optional['ConfigManager'] = None
    # ... attributes ...

    def process_data_source(self, data_source: DataSource) -> DataSource:
        """Orchestrates processing steps for a data source."""
        if not data_source.validate_data(): # Basic check
            self.logger.error("Data validation failed")
            return data_source

        self.logger.debug("Starting processing steps...")
        # Call internal methods to apply processing
        data_source.dataset = self._standardize_coordinates(data_source.dataset, data_source.model_name)
        data_source.dataset = self._handle_missing_values(data_source.dataset)
        data_source.dataset = self._apply_unit_conversions(data_source.dataset)
        # ... other processing steps ...

        self.logger.debug("Processing steps finished.")
        return data_source # Return the data source with the modified dataset

    def _standardize_coordinates(self, dataset: xr.Dataset, model_name: str = None) -> xr.Dataset:
        """Standardize dimension names (e.g., 'XLONG' -> 'lon')."""
        self.logger.debug(f"Standardizing coordinates for {model_name}")
        # This method uses the config_manager and meta_coords (from Chapter 5)
        # to map source-specific dimension names to standard names like 'lon', 'lat'.
        # It then renames the dimensions in the dataset.
        rename_dict = {}
        # Example: Find the actual name for the longitude dimension ('xc')
        xc_dim_name = self._get_model_dim_name('xc', list(dataset.dims), model_name, self.config_manager)
        if xc_dim_name and xc_dim_name != 'lon' and xc_dim_name in dataset.dims:
             rename_dict[xc_dim_name] = 'lon'
        # ... similar logic for 'yc', 'zc', 'tc' ...

        if rename_dict:
            self.logger.debug(f"Renaming dimensions: {rename_dict}")
            dataset = dataset.rename(rename_dict) # Use xarray's rename method
        return dataset

    def _handle_missing_values(self, dataset: xr.Dataset) -> xr.Dataset:
        """Replace specific missing value indicators or NaNs."""
        self.logger.debug("Handling missing values")
        # This loops through variables and replaces missing values based on metadata
        # (e.g., the _FillValue attribute) or common representations like NaN.
        for var_name, var in dataset.data_vars.items():
             if '_FillValue' in var.attrs:
                 fill_value = var.attrs['_FillValue']
                 # ... code to replace values matching fill_value with NaN ...
             # ... code to handle other missing value conventions ...
        return dataset # Return the dataset with missing values handled

    def _apply_unit_conversions(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply common unit conversions (e.g., K to C)."""
        self.logger.debug("Applying unit conversions")
        # This loops through variables, checks their 'units' attribute (from metadata),
        # and applies conversions for known cases.
        for var_name, var in dataset.data_vars.items():
            if 'units' in var.attrs:
                units = var.attrs['units'].lower()
                # Example: Convert Kelvin to Celsius
                if units == 'k' and var_name.lower() in ['temp', 'temperature']:
                    var_data = var.values - 273.15
                    # Create a new DataArray with converted data and updated units
                    dataset[var_name] = xr.DataArray(var_data, dims=var.dims, coords=var.coords, attrs=var.attrs)
                    dataset[var_name].attrs['units'] = 'C'
                    self.logger.debug(f"Converted {var_name} from K to C")
        return dataset

    # ... other processing methods like regridding, computing differences ...
```

These snippets show how the `DataProcessor` takes the `xarray Dataset` and applies standard operations like renaming dimensions based on configuration/metadata mappings, replacing specific missing values with standard `NaN`, and converting units like Kelvin to Celsius. It uses `xarray`'s powerful capabilities (`.rename()`, accessing `.attrs`, `.values`) to perform these tasks.

### DataIntegrator (`eviz/lib/data/pipeline/integrator.py`)

The `DataIntegrator` is responsible for combining data. This might mean combining multiple **xarray Datasets** (e.g., if you loaded data from several time-step files) or creating new variables by combining existing ones (e.g., calculating a difference or sum).

Here are simplified examples of its methods:

```python
# --- File: eviz/lib/data/pipeline/integrator.py (Simplified) ---
# ... imports ...
import xarray as xr
# ... other imports ...

@dataclass
class DataIntegrator:
    # ... attributes ...

    def integrate_data_sources(self, data_sources: List[DataSource], **kwargs) -> xr.Dataset:
        """Integrate multiple data sources (datasets) into one."""
        self.logger.debug(f"Integrating {len(data_sources)} data sources")
        if not data_sources:
            return None

        # Get the datasets from the DataSource objects
        datasets_to_integrate = [ds.dataset for ds in data_sources if ds and ds.dataset is not None]

        method = kwargs.get('method', 'merge') # Method can be 'merge' or 'concatenate'

        if method == 'merge':
            # xr.merge combines datasets assuming they have compatible coordinates
            # and will add variables from different datasets if names don't clash.
            result = xr.merge(datasets_to_integrate, join=kwargs.get('join', 'outer'))
        elif method == 'concatenate':
            # xr.concat stacks datasets along a specified dimension (like 'time')
            dim = kwargs.get('dim', 'time')
            result = xr.concat(datasets_to_integrate, dim=dim)
        else:
            self.logger.error(f"Unknown integration method: {method}")
            return None # Or raise error

        self.logger.info(f"Successfully integrated datasets using '{method}'")
        return result # Return the single integrated dataset

    def integrate_variables(self, dataset: xr.Dataset, variables: List[str], operation: str, output_name: str) -> xr.Dataset:
        """Create a new variable by operating on existing variables."""
        self.logger.debug(f"Integrating variables {variables} with operation '{operation}'")

        if not dataset or not variables:
            return dataset # Nothing to do

        # Example: Simple addition of variables
        if operation == 'add':
             # Access DataArrays within the dataset and sum them using xarray's capabilities
             result_data_array = sum(dataset[var] for var in variables if var in dataset.data_vars)
             # Add the new DataArray to the dataset
             dataset[output_name] = result_data_array
             # Add some metadata to the new variable
             dataset[output_name].attrs['long_name'] = f"Sum of {', '.join(variables)}"
             self.logger.info(f"Added new variable '{output_name}' (sum)")
        # ... other operations like 'subtract', 'mean', etc. would be implemented here ...

        return dataset # Return the dataset with the new variable added
```

These snippets show how `DataIntegrator` uses `xarray.merge` or `xarray.concat` to combine multiple datasets into one, and how it can perform calculations (`sum`, `subtract`, etc.) on variables within a dataset using `xarray` operations and add the result as a new variable.

### DataTransformer (`eviz/lib/data/pipeline/transformer.py`)

The `DataTransformer` is intended for more complex transformations, perhaps changing the very structure or representation of the data beyond simple processing. In the current eViz code base, this stage is minimal, with a placeholder method.

```python
# --- File: eviz/lib/data/pipeline/transformer.py (Simplified) ---
# ... imports ...
import xarray as xr
# ... other imports ...

@dataclass()
class DataTransformer:
    # ... attributes ...

    def transform_data_source(self, data_source: DataSource, **kwargs) -> DataSource:
        """Transform a data source."""
        self.logger.debug("Transforming data source (placeholder)")
        # Currently, this just calls a placeholder method
        data_source.dataset = self._transform_dataset(data_source.dataset, **kwargs)
        return data_source

    @staticmethod
    def _transform_dataset(dataset: xr.Dataset) -> xr.Dataset:
        """Placeholder for dataset transformation logic."""
        # TODO: Implement data transformation logic
        return dataset # Returns the dataset unchanged for now
```

This shows that while the `DataTransformer` is part of the pipeline structure, its actual functionality is planned but not fully implemented yet. Data mostly flows through this stage unchanged for now.

## Benefits of the Pipeline Structure

*   **Clear Responsibilities:** Each stage (Reader, Processor, Integrator) has a specific job, making the code easier to understand and maintain.
*   **Modularity:** You can modify or replace a stage (e.g., improve missing value handling in the Processor) without affecting other stages.
*   **Reusability:** The components (like the Processor's unit conversion logic) can potentially be reused for different data sources or workflows.
*   **Flexibility:** By configuring which stages run and in what order (as orchestrated by the `DataPipeline`), you can create different processing workflows for different data types or plotting needs.
*   **Standardized Data:** The pipeline operates on the standard **xarray Dataset**, reinforcing the benefits of **Data Source Abstraction** ([Chapter 3](03_data_source_abstraction_.md)).

## Conclusion

In this chapter, you learned about the **Data Processing Pipeline**, the assembly line that takes raw data from your files and prepares it for visualization. You saw how the `DataPipeline` class orchestrates different stages (`DataReader`, `DataProcessor`, `DataTransformer`, `DataIntegrator`), each responsible for specific tasks like loading, cleaning, standardizing coordinates, handling missing values, converting units, or combining datasets. Data flows through these stages as standard **xarray Datasets**, getting progressively closer to the final state needed for plotting.

Now that the data has been loaded, processed, and potentially integrated, it's finally ready to be turned into actual plots and images!

Ready to see how eViz takes this prepared data and creates visualizations? Let's move on to the next chapter: [Plotting Engine](07_plotting_engine_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
