# Chapter 3: Data Source Abstraction

Welcome back! In [Chapter 1: Autoviz Application](01_autoviz_application_.md), we met the director, and in [Chapter 2: Configuration Management](02_configuration_management_.md), we saw how the director gets its script or plan. Now, let's talk about the *data* itself – the actual information eViz needs to visualize.

Scientific data comes in many shapes and sizes, and perhaps more importantly, many different *formats*. You might have data in:
*   NetCDF (`.nc` files) - common in climate and weather.
*   HDF5 (`.h5`, `.hdf5` files) - used across many scientific fields.
*   CSV (`.csv` files) - simple tables of data.
*   GRIB (`.grib`, `.grib2` files) - specific to meteorology.
*   Zarr (`.zarr` directories) - a modern format for cloud-native storage.

Imagine trying to write a program that needs to read all these different types of files. You'd need separate code for each format, handling their unique ways of organizing variables, dimensions, and attributes. That sounds complicated and tedious!

This is where the **Data Source Abstraction** comes to the rescue.

## What is Data Source Abstraction?

Think of **Data Source Abstraction** as a **universal adapter** or a **translator** for your data files. It's a system that hides the messiness of dealing with different file formats.

No matter if your data is in a NetCDF file, a CSV table, or a Zarr store, the Data Source Abstraction makes it appear and behave consistently *once it's loaded* inside eViz.

The key idea is to load data from any supported format and turn it into a standard, easy-to-work-with structure. In eViz, this standard structure is typically an **xarray Dataset**.

An **xarray Dataset** is a powerful tool in Python that's designed for labeled multi-dimensional arrays. It's excellent for scientific data because it understands dimensions (like time, latitude, longitude, level), coordinates associated with those dimensions, and attributes (like units, descriptions).

So, the job of the Data Source Abstraction is to:
1.  Read data from a specific file format (NetCDF, CSV, etc.).
2.  Translate it into a standard `xarray.Dataset`.
3.  Present this `xarray.Dataset` through a common interface, so the rest of eViz doesn't need to know or care what the original file format was.

This makes eViz much more flexible. It can handle new data formats simply by adding a new "adapter" (a new `DataSource` class) without changing the core visualization logic.

## Our Use Case: Loading Data Seamlessly

Based on [Chapter 2: Configuration Management](02_configuration_management_.md), you specify the file you want to load in the `inputs` section of your YAML config file:

```yaml
# --- Snippet from my_gridded_config.yaml ---
inputs:
  - name: sample_gridded_data.nc  # Could be sample.csv, sample.h5, etc.
    location: /path/to/your/data
    exp_id: baseline
    to_plot:
      temperature: xy
```

When eViz processes this configuration, it sees that it needs to load `/path/to/your/data/sample_gridded_data.nc`. The Data Source Abstraction system is responsible for reading this file correctly, whether it's a `.nc`, `.csv`, `.h5`, `.grib`, or `.zarr` file, and turning it into an `xarray.Dataset` that eViz can then use to plot 'temperature'.

The user doesn't type a special command or call a specific function for NetCDF files vs. CSV files. They just point eViz to the file via the config, and the Data Source Abstraction handles the rest.

## How Data Source Abstraction Works (High-Level)

Let's see how this fits into the overall flow we saw in [Chapter 1](01_autoviz_application_.md) and [Chapter 2](02_configuration_management_.md):

```{mermaid}
sequenceDiagram
    participant ConfigMgr as ConfigManager
    participant DataSourceFactory as Data Source Factory
    participant SpecificDataSource as Specific DataSource (e.g., NetCDFDataSource)
    participant DataFile as Data File (.nc, .csv, etc.)
    participant XarrayDataset as xarray.Dataset

    ConfigMgr->>DataSourceFactory: "I need a DataSource for this file..."
    DataSourceFactory->>DataSourceFactory: Figure out file type/format
    DataSourceFactory->>SpecificDataSource: Create the right DataSource object (e.g., NetCDFDataSource)
    SpecificDataSource-->>DataSourceFactory: Return SpecificDataSource object
    DataSourceFactory-->>ConfigMgr: Return SpecificDataSource object
    ConfigMgr->>SpecificDataSource: Call load_data("path/to/file")
    SpecificDataSource->>DataFile: Read data from file (using appropriate library)
    DataFile-->>SpecificDataSource: Raw data
    SpecificDataSource->>XarrayDataset: Translate raw data into xarray.Dataset
    XarrayDataset-->>SpecificDataSource: Return xarray.Dataset object
    SpecificDataSource->>SpecificDataSource: Store xarray.Dataset internally
    SpecificDataSource-->>ConfigMgr: Return xarray.Dataset (optional, it's stored internally)
    ConfigMgr->>SpecificDataSource: Ask for variable 'temperature' (using standard methods)
    SpecificDataSource->>XarrayDataset: Get 'temperature' DataArray
    XarrayDataset-->>SpecificDataSource: Return 'temperature' DataArray
    SpecificDataSource-->>ConfigMgr: Return 'temperature' DataArray
    ConfigMgr->>ConfigMgr: Use 'temperature' for plotting (always the same way)

```

In this simplified flow:

1.  The `ConfigManager` (our plan reader) identifies an input file from the configuration.
2.  It asks the **Data Source Factory** (which we'll cover in the next chapter!) to create the correct `DataSource` object for that specific file type. The factory is like the part of the system that knows which adapter is needed for which type of plug (file format).
3.  The factory creates the appropriate `DataSource` object (e.g., `NetCDFDataSource` for a `.nc` file).
4.  The `ConfigManager` then tells this `DataSource` object to `load_data()`, giving it the file path.
5.  The specific `DataSource` implementation (like `NetCDFDataSource`) uses the right tools (like the `xarray` library, which itself uses lower-level readers) to open the file and read the data.
6.  It converts the data into an `xarray.Dataset` and stores it internally.
7.  Now, the `ConfigManager` (and later other parts of eViz like the [Data Processing Pipeline](05_data_processing_pipeline_.md) and the plotting system) can interact with the loaded data through the standard methods and interface provided by the `DataSource` object, *without* caring if the original file was NetCDF or something else.

## Inside the Code: The DataSource Base Class

The core of the abstraction is the `DataSource` base class, found in `eviz/lib/data/sources/base.py`. This is an "abstract base class" (ABC), meaning you can't use it directly, but other specific data source classes *must* inherit from it and provide implementations for its abstract methods.

Here's a simplified look at the `DataSource` base class:

```python
# --- File: eviz/lib/data/sources/base.py (simplified) ---
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import xarray as xr
import logging
from typing import Any, Dict, Optional, List

@dataclass
class DataSource(ABC):
    """
    Abstract base class that defines the interface for all data sources.
    """
    model_name: Optional[str] = None
    config_manager: Optional[object] = None
    dataset: Optional[xr.Dataset] = field(default=None, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    @abstractmethod
    def load_data(self, file_path: str) -> xr.Dataset:
        """
        Load data from the specified file path into an xarray dataset.
        """
        raise NotImplementedError("Subclasses must implement load_data.")

    def get_field(self, field_name: str) -> Optional[xr.DataArray]:
        """Get a specific field (variable) from the dataset."""
        if self.dataset is None:
             self.logger.error("No data loaded.")
             return None
        try:
            return self.dataset[field_name]
        except KeyError:
            self.logger.error(f"Field '{field_name}' not found.")
            return None

    # ... other common methods like get_metadata, get_dimensions, get_variables, close ...

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying dataset."""
        # This lets you do data_source.mean() instead of data_source.dataset.mean()
        if self.dataset is not None and hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Delegate item access to the underlying dataset."""
        # This lets you do data_source['temp'] instead of data_source.dataset['temp']
        if self.dataset is not None:
            return self.dataset[key]
        raise TypeError(f"'{self.__class__.__name__}' has no dataset loaded")

    # ... other utility methods like _get_model_dim_name ...
```

Here's what's important about this base class:

*   It inherits from `ABC` and defines `load_data` with the `@abstractmethod` decorator. This forces any class inheriting from `DataSource` to provide its *own* version of the `load_data` method.
*   It has a `dataset` attribute (initialized to `None`). This is where the loaded `xarray.Dataset` will be stored by the `load_data` method of the concrete class.
*   It provides common methods like `get_field`, `get_metadata`, etc., that work *on* the stored `dataset`.
*   Crucially, it implements `__getattr__` and `__getitem__`. These are special Python methods that allow you to access attributes (like `.mean()`) and items (like `['temperature']`) directly on the `DataSource` object, just as if you were accessing the underlying `xarray.Dataset`. This makes working with `DataSource` objects feel very natural if you're used to `xarray`.

## Inside the Code: Specific DataSource Implementations

Now let's look at how specific file formats are handled by concrete classes that inherit from `DataSource`. Each of these classes lives in its own file (e.g., `netcdf.py`, `csv.py`, `hdf5.py`, `grib.py`, `zarr.py`) and provides its unique `load_data` method.

### Example: NetCDFDataSource

The `NetCDFDataSource` in `eviz/lib/data/sources/netcdf.py` uses `xarray.open_dataset` (which itself often uses libraries like `netcdf4` or `h5netcdf`) to read NetCDF files.

```python
# --- File: eviz/lib/data/sources/netcdf.py (simplified load_data) ---
import logging
import xarray as xr
from .base import DataSource # Import the base class

class NetCDFDataSource(DataSource):
    """Data source implementation for NetCDF files."""
    # ... __init__ and logger property ...

    def load_data(self, file_path: str) -> xr.Dataset:
        """
        Load data from a NetCDF file or OpenDAP URL into an Xarray dataset.
        """
        self.logger.debug(f"Loading NetCDF data from {file_path}")
        try:
            # xarray makes reading NetCDF super easy!
            # It automatically handles local files, URLs, and OpenDAP
            dataset = xr.open_dataset(file_path, decode_cf=True)

            # Store the loaded dataset in the base class attribute
            self.dataset = dataset
            # Optional: Extract metadata and store it
            # self._extract_metadata(dataset)

            self.logger.debug(f"Successfully loaded NetCDF data.")
            return dataset # Also return the dataset

        except Exception as exc:
            self.logger.error(f"Error loading NetCDF file: {file_path}. Exception: {exc}")
            raise

    # ... other methods specific to NetCDF or common processing ...
```

Notice how simple the `load_data` method is thanks to `xarray`. It opens the file, and `xarray` does the heavy lifting of reading the NetCDF structure and creating the `Dataset`. The resulting `dataset` is stored in `self.dataset`, fulfilling the contract of the base class.

### Example: CSVDataSource

The `CSVDataSource` in `eviz/lib/data/sources/csv.py` uses the `pandas` library to read CSV files and then converts the pandas DataFrame to an `xarray.Dataset`. It also includes basic logic to try and identify columns like 'time', 'lat', and 'lon' and set them as `xarray` coordinates.

```python
# --- File: eviz/lib/data/sources/csv.py (simplified load_data) ---
import pandas as pd
import xarray as xr
from .base import DataSource # Import the base class

class CSVDataSource(DataSource):
    """Data source implementation for CSV files."""
    # ... __init__ and logger property ...

    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from a CSV file or a list of CSV files into an Xarray dataset."""
        self.logger.debug(f"Loading CSV data from {file_path}")

        try:
            # Use pandas to read the CSV
            combined_data = pd.read_csv(file_path)

            # Convert the pandas DataFrame to an xarray Dataset
            dataset = combined_data.to_xarray()

            # Optional: Add specific CSV processing (like identifying time/lat/lon columns)
            dataset = self._process_data(dataset) # Calls an internal method

            # Store the loaded dataset
            self.dataset = dataset
            # Optional: Extract metadata
            # self._extract_metadata(dataset)

            self.logger.debug(f"Successfully loaded CSV data.")
            return dataset

        except Exception as exc:
            self.logger.error(f"Error loading CSV file: {file_path}. Exception: {exc}")
            raise

    # ... other methods specific to CSV processing (_process_data, _extract_metadata) ...
```

Again, the core `load_data` method is straightforward. It uses `pandas.read_csv` to get the data and `to_xarray()` to convert it. Any CSV-specific clean-up or coordinate identification happens in helper methods like `_process_data`, ensuring the `load_data` method itself stays focused on reading and converting to the standard `xarray.Dataset`.

### Other Data Sources

Similarly, `HDF5DataSource` (`hdf5.py`), `GRIBDataSource` (`grib.py`), and `ZARRDataSource` (`zarr.py`) each implement the `load_data` method using appropriate libraries (`h5py`, `cfgrib`/`pynio`, `xarray` with zarr engine) to read their specific format and produce an `xarray.Dataset`.

The important outcome is that once `self.dataset` is populated in *any* of these classes after `load_data` is called, the rest of eViz can interact with it using the common `DataSource` interface and the handy `xarray` delegation provided by `__getattr__` and `__getitem__`. Whether you loaded a NetCDF file or a CSV file, you can now do things like `data_source['temperature'].mean()` or `data_source.dims` or `data_source.get_field('pressure')` in a consistent way.

## Summary

In this chapter, we learned about **Data Source Abstraction**, which is eViz's way of handling different data file formats seamlessly.

*   Different scientific data formats require different reading methods.
*   The Data Source Abstraction uses the **xarray Dataset** as a universal internal representation.
*   An abstract base class `DataSource` defines the common interface, requiring concrete implementations to provide a `load_data` method that returns an `xarray.Dataset`.
*   Specific classes like `NetCDFDataSource`, `CSVDataSource`, etc., inherit from `DataSource` and implement `load_data` using format-specific libraries (`xarray`, `pandas`, `h5py`, `cfgrib`, etc.).
*   Once data is loaded, the `DataSource` object behaves much like an `xarray.Dataset`, allowing the rest of eViz to work with the data consistently, regardless of its origin format.

This abstraction is crucial because it allows eViz's core processing and plotting logic to remain simple and focused on `xarray.Dataset`s, while the complexity of reading various file types is hidden away in the specific `DataSource` classes.

Now that we know *what* the Data Source Abstraction is and how concrete classes implement it, the next question is: how does eViz *automatically choose* and create the correct `DataSource` object (e.g., `NetCDFDataSource` vs `CSVDataSource`) when it encounters a file listed in the configuration? That task is handled by the **Data Source Factory**.

[Data Source Factory](04_data_source_factory_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)