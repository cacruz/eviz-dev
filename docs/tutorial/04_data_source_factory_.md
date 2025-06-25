# Chapter 4: Data Source Factory

Welcome back! In [Chapter 3: Data Source Abstraction](03_data_source_abstraction_.md), we learned how eViz uses a common interface and the **xarray Dataset** to work with data, regardless of its original file format (NetCDF, CSV, etc.). We saw that specific classes like `NetCDFDataSource` and `CSVDataSource` know *how* to read their particular file types and produce an `xarray.Dataset`.

But this leaves an important question unanswered: when eViz reads your configuration file and sees you want to load `sample_gridded_data.nc`, how does it *know* to use the `NetCDFDataSource` instead of the `CSVDataSource`? How does it automatically pick the right tool for the job?

This is precisely the problem solved by the **Data Source Factory**.

## What is the Data Source Factory?

Imagine you have a specialized workshop filled with skilled craftspeople, each an expert at building one specific type of tool (like a NetCDF reader tool, a CSV reader tool, etc.). When a request comes in to "build a reader tool for `/path/to/my/data/sample.nc`", you don't need to tell the workshop *which* craftsperson to use. You just give them the file path. The workshop has a foreman whose job is to look at the request (specifically, the `.nc` extension in the file path), know which craftsperson specializes in `.nc` files, and tell *that* person to build the tool.

The **Data Source Factory** in eViz is that specialized workshop with its clever foreman.

Its job is to:
1.  Take a file path (or sometimes an explicit format instruction from the configuration).
2.  Inspect the file path (usually by looking at the file extension like `.nc`, `.csv`, `.h5`).
3.  Figure out which specific `DataSource` class (like `NetCDFDataSource`, `CSVDataSource`, `HDF5DataSource`) is the correct one for that file type.
4.  Create an instance of that correct `DataSource` class and give it back to the part of eViz that needs it (like the [Configuration Management](02_configuration_management_.md) system or the [Data Processing Pipeline](05_data_processing_pipeline_.md)).

This design pattern (called the "Factory" pattern) is great because it keeps the code that *needs* a `DataSource` (the "client" code) clean. The client code doesn't need a messy `if/elif/else` block checking file extensions. It just asks the Factory: "Hey Factory, I need a DataSource for this file path, please build one for me!" The Factory handles the complexity of choosing and creating the right one.

## Our Use Case: Automatically Selecting the Correct Data Reader

Based on the configuration from [Chapter 2](02_configuration_management_.md), the `inputs` section lists files:

```yaml
# --- Snippet from a config file ---
inputs:
  - name: my_netcdf_data.nc
    location: /data/dir
    # ... other settings ...
    to_plot: ...
  - name: another_data.csv
    location: /other/dir
    # ... other settings ...
    to_plot: ...
```

When the Configuration Management system processes this, it identifies that it needs to load two different files with two different formats. It needs a way to get the appropriate `DataSource` object for each: a `NetCDFDataSource` for `my_netcdf_data.nc` and a `CSVDataSource` for `another_data.csv`.

This is where the **Data Source Factory** is called into action.

## How the Data Source Factory Works (High-Level)

Let's revisit the data loading flow, focusing on where the Factory fits in:

```{mermaid}
sequenceDiagram
    participant InputConfig as InputConfig
    participant DataSourceFactory as Data Source Factory
    participant DataSourceRegistry as Data Source Registry
    participant NetCDFDataSource as NetCDFDataSource
    participant CSVDataSource as CSVDataSource

    InputConfig->>DataSourceFactory: "Need DataSource for 'data.nc'"
    DataSourceFactory->>DataSourceRegistry: "What class for extension '.nc'?"
    DataSourceRegistry-->>DataSourceFactory: "It's NetCDFDataSource"
    DataSourceFactory->>NetCDFDataSource: Create new NetCDFDataSource()
    NetCDFDataSource-->>DataSourceFactory: Return NetCDFDataSource object
    DataSourceFactory-->>InputConfig: Return NetCDFDataSource object

    InputConfig->>DataSourceFactory: "Need DataSource for 'data.csv'"
    DataSourceFactory->>DataSourceRegistry: "What class for extension '.csv'?"
    DataSourceRegistry-->>DataSourceFactory: "It's CSVDataSource"
    DataSourceFactory->>CSVDataSource: Create new CSVDataSource()
    CSVDataSource-->>DataSourceFactory: Return CSVDataSource object
    DataSourceFactory-->>InputConfig: Return CSVDataSource object

    InputConfig->>InputConfig: Store created DataSource objects

```

As the diagram shows, when the `InputConfig` (part of the Configuration Management system) is setting itself up and figuring out which data readers are needed, it interacts with the `DataSourceFactory`. The Factory, in turn, might use a helper component (the **Data Source Registry**) to look up the correct `DataSource` class based on the file extension, and then it creates the actual instance of that class.

## Inside the Code: Registry and Factory

The Data Source Factory system in eViz involves two main classes working together:

1.  **`DataSourceRegistry`**: This class acts like a simple lookup table. It stores the mapping between file extensions (like `nc`, `csv`, `h5`) and the corresponding `DataSource` class (like `NetCDFDataSource`, `CSVDataSource`, `HDF5DataSource`). Think of this as the foreman's address book, listing which craftsperson handles which material.
2.  **`DataSourceFactory`**: This is the main class you interact with. It's the "workshop" itself. It uses the `DataSourceRegistry` to find the right class and then creates an instance of that class.

Let's look at simplified code snippets for each.

### The Data Source Registry

The `DataSourceRegistry` is defined in `eviz/lib/data/factory/registry.py`.

```python
# --- File: eviz/lib/data/factory/registry.py (simplified) ---
from typing import List, Dict, Type
from eviz.lib.data.sources import DataSource # Import the base class

@dataclass
class DataSourceRegistry:
    _registry: Dict[str, Type] = field(default_factory=dict, init=False) # The lookup dictionary

    def register(self, extensions: List[str], data_source_class: Type[DataSource]) -> None:
        """Register a data source class for the specified file extensions."""
        for ext in extensions:
            ext = ext.lower().lstrip('.') # Clean up extension
            self._registry[ext] = data_source_class # Store the mapping

    def get_data_source_class(self, file_extension: str) -> Type[DataSource]:
        """Get the data source class for the specified file extension."""
        ext = file_extension.lower().lstrip('.') # Clean up extension

        if ext not in self._registry:
            # Handle unsupported extension
            raise ValueError(f"No data source registered for extension: {file_extension}")

        return self._registry[ext] # Return the stored class
```

This `DataSourceRegistry` class is very simple. Its `_registry` is a dictionary. The `register` method adds entries to this dictionary (mapping extension strings to the actual class objects), and the `get_data_source_class` method looks up an extension and returns the class associated with it. It raises an error if the extension isn't found.

### The Data Source Factory

The `DataSourceFactory` is defined in `eviz/lib/data/factory/source_factory.py`.

```python
# --- File: eviz/lib/data/factory/source_factory.py (simplified) ---
import os
from typing import Type, List, Optional
from eviz.lib.data.sources import (
    DataSource, NetCDFDataSource, HDF5DataSource, CSVDataSource, GRIBDataSource, ZARRDataSource
)
from .registry import DataSourceRegistry # Import the Registry

@dataclass
class DataSourceFactory:
    config_manager: Optional[object] = None
    registry: DataSourceRegistry = field(init=False) # Factory *uses* a Registry

    def __post_init__(self):
        """Initialize the Factory and register default sources."""
        self.registry = DataSourceRegistry() # Create the Registry instance
        self._register_default_data_sources() # Populate the Registry

    def _register_default_data_sources(self) -> None:
        """Register the default data source implementations."""
        self.registry.register(['nc', 'netcdf', 'dap'], NetCDFDataSource)
        self.registry.register(['h5', 'hdf5'], HDF5DataSource)
        self.registry.register(['csv', 'txt'], CSVDataSource)
        self.registry.register(['grib', 'grib2'], GRIBDataSource)
        self.registry.register(['zarr'], ZARRDataSource)
        # Add more here as new DataSources are created!

    def create_data_source(self, file_path: str, model_name: Optional[str] = None,
                           reader_type: Optional[str] = None,
                           file_format: Optional[str] = None) -> DataSource:
        """ Create a data source instance for the specified file or URL."""

        # 1. Check for explicit format/reader type override (from config)
        if reader_type is not None or file_format is not None:
            # Simplified: In real code, this looks up the class based on the string
            # For this example, we'll just show the extension lookup below.
            pass # Skip complex override logic for simplicity

        # 2. If no explicit override, determine type from file path/extension
        ext = None
        if os.path.isdir(file_path) and file_path.endswith('.zarr'):
             ext = 'zarr' # Special handling for Zarr directories
        elif file_path.startswith(('http://', 'https://', 'ftp://')):
             # Simplified URL handling: Check path portion for extension
             path = file_path.split('?')[0]
             _, ext = os.path.splitext(path)
        else:
             _, ext = os.path.splitext(file_path) # Get extension from local file

        # Handle missing extension or attempt inference (simplified)
        if not ext:
             # More complex logic here in real code to infer from path name/content
             raise ValueError(f"Could not determine file type for: {file_path}")

        ext = ext[1:] if ext.startswith('.') else ext # Remove the dot

        # 3. Use the Registry to get the correct class
        try:
            data_source_class = self.registry.get_data_source_class(ext)
        except ValueError:
            # Handle unsupported extension raised by Registry
            raise ValueError(f"Unsupported file type extension: {ext}")

        # 4. Create and return an instance of the found class
        return data_source_class(model_name, self.config_manager)

    # ... other helper methods ...
```

When the `DataSourceFactory` is created (`__post_init__`), it immediately creates its `DataSourceRegistry` and calls `_register_default_data_sources`. This method populates the registry with the mappings for the standard file types eViz supports. If you wanted to add support for a new file type, you'd create a new `MyNewDataSource` class inheriting from `DataSource` and add `self.registry.register(['mynewext'], MyNewDataSource)` here (or provide a way for users to register custom sources).

The main method is `create_data_source`. It takes the `file_path` and optionally other hints. In the simplified version above, it focuses on getting the file extension from the path (handling local files, URLs, and Zarr directories). Then, it asks its `registry` for the correct `DataSource` class based on that extension. If the registry finds a class, the factory simply creates an instance of that class (`data_source_class(...)`) and returns it.

This clearly shows how the Factory delegates the *lookup* job to the Registry but keeps the *creation* job for itself.

### Where is the Factory Used?

The `DataSourceFactory` is typically created and held by the Configuration Management system (specifically the `ConfigManager` or its `InputConfig` helper). As seen in the high-level flow diagram and referenced in [Chapter 2](02_configuration_management_.md), when `InputConfig._init_readers()` is called during the configuration setup process, it loops through the list of input files provided in the configuration (`self.app_data.inputs`). For each file entry, it calls the `DataSourceFactory` to get the appropriate `DataSource` object *before* the data is actually loaded. The factory creates these `DataSource` objects, and the `InputConfig` keeps track of them, ready to be told to `load_data()` later in the process.

```python
# --- Snippet from eviz/lib/config/input_config.py (simplified) ---
# (This happens inside InputConfig's initialize method or a method called by it)
from eviz.lib.data.factory import DataSourceFactory
# ... other imports and class definition ...

@dataclass
class InputConfig:
    # ... attributes ...
    _data_source_factory: DataSourceFactory = field(init=False)
    _data_sources_mapping: Dict[str, Type] = field(default_factory=dict) # To store which DS class maps to which file

    def initialize(self):
        # ... parse app_data, build file_list ...

        # Create the Data Source Factory instance
        self._data_source_factory = DataSourceFactory(config_manager=self.config_manager)

        # Now, loop through the file list and ask the factory for the right Data Source *class*
        self._init_readers()

    def _init_readers(self):
        """Determine which data source reader class is needed for each file."""
        for file_index, file_info in self.file_list.items():
             file_path = file_info['filename']
             explicit_format = file_info.get('format') # Check if format is specified in config

             try:
                  # Ask the factory to DETERMINE the correct class for this file
                  # We aren't creating an instance *yet*, just mapping file to class
                  data_source_class = self._data_source_factory.create_data_source(
                      file_path=file_path, # Give the factory the file path
                      file_format=explicit_format # Or give it an explicit format hint
                  ).__class__ # We only need the class object here, not the instance

                  # Store the mapping: filename -> DataSource Class
                  self._data_sources_mapping[file_path] = data_source_class
                  self.logger.debug(f"Mapped {file_path} to {data_source_class.__name__}")

             except ValueError as e:
                  self.logger.error(f"Failed to determine data source for {file_path}: {e}")
                  sys.exit()

        # Note: The actual Data Source *instances* are created later,
        # right before the data is loaded for a specific plotting task.
        # The ConfigManager or Pipeline uses this _data_sources_mapping
        # to know which class to instantiate when needed.
```

This snippet shows how `InputConfig` creates the `DataSourceFactory` and then uses it in its `_init_readers` method. It calls `factory.create_data_source`, but in this specific phase, it often just wants to know *which class* the factory *would* create for that file path (hence the `.__class__` at the end). This mapping is stored, and the actual instance creation and `load_data()` call happen later when eViz is ready to process data for a specific plot (often orchestrated by the [Data Processing Pipeline](05_data_processing_pipeline_.md)).

## Summary

In this chapter, we demystified the **Data Source Factory** in eViz. We learned that:

*   The Factory acts as a smart selector and creator of the correct `DataSource` object based on the input file (usually its extension).
*   It uses a **Data Source Registry** internally as a lookup table mapping extensions to `DataSource` classes.
*   This pattern keeps the rest of the eViz code clean, as it doesn't need to know the specifics of choosing between different data readers. It just asks the Factory.
*   The Factory is typically initialized by the Configuration Management system ([Chapter 2](02_configuration_management_.md)), which then uses it to determine and eventually instantiate the correct `DataSource` classes needed for the files listed in your configuration.

Now that eViz knows *what* data files you want to use (from [Configuration Management](02_configuration_management_.md)), *how* to read different types of data (using [Data Source Abstraction](03_data_source_abstraction_.md) and the Factory), the next step is to figure out *what to do with the data once it's loaded*. How does eViz apply transformations, calculations, or selections to the data before plotting? That's handled by the **Data Processing Pipeline**.

[Data Processing Pipeline](05_data_processing_pipeline_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)