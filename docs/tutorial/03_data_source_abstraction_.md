# Chapter 3: Data Source Abstraction

Welcome back! In the [previous chapter](02_configuration_management_.md), we learned that the **Autoviz Application** gets its detailed instructions for plotting from **Configuration Files**. These YAML files tell eViz *which* data files to use.

But here's a challenge: data comes in many shapes and sizes! You might have weather data in a **NetCDF** file (`.nc`), a satellite image in **HDF5** (`.h5`), station observations in a **CSV** (`.csv`), or forecast model output in **GRIB** (`.grib`). Each of these file formats stores data differently.

How can eViz process and plot data from *all* these different file types using the *same* processing steps and the *same* plotting code? If the plotting code had to know how to read *every single* file format, it would be incredibly complicated!

This is where **Data Source Abstraction** comes in. It's a core concept in eViz that solves this problem by providing a standard way to interact with data, no matter its original format.

## The Problem: Many Data Formats, One Visualization System

Imagine you have different types of electrical plugs in your house: one for your lamp, one for your computer, one for your kitchen mixer. You can't plug your computer directly into the lamp's socket!

Your eViz processing and plotting system is like an electrical appliance that needs a standard plug. But your data files are like devices with different, non-standard plugs.

 eViz needs a way to connect *any* data file (any plug type) to the processing pipeline and plotting engine (the standard socket).

## The Solution: Universal Adapters (Data Source Abstraction)

Data Source Abstraction provides these "universal adapters". It works like this:

1.  **A Standard "Language":** eViz decides on a standard way to represent data *internally* once it's loaded. This standard is the **xarray Dataset**. Think of an xarray Dataset as a smart, labeled container that can hold multi-dimensional data (like temperature over time, latitude, and longitude) along with its coordinates (time values, latitude values, longitude values) and metadata (units, descriptions). It's a very common and powerful format in scientific computing.
2.  **Specialist Adapters:** For each different data file format (NetCDF, CSV, HDF5, GRIB), eViz has a dedicated "specialist" or "adapter" class. This class knows *only* how to read *its* specific file format.
3.  **A Common Blueprint (The Abstract Interface):** All these specialist adapter classes are built using the same blueprint. This blueprint is an **abstract base class** called `DataSource` (`eviz/lib/data/sources/base.py`). This blueprint requires all specialists to have a method (a function) called `load_data`.
4.  **The `load_data` Promise:** The `load_data` method in *every* specialist class promises to do two things:
    *   Take the file path as input.
    *   Return an **xarray Dataset** as output.

So, when eViz needs data from a NetCDF file, it uses the `NetCDFDataSource` specialist. When it needs data from a CSV file, it uses the `CSVDataSource` specialist. But *regardless* of which specialist is used, the result is *always* an xarray Dataset.

This means the rest of the eViz system – the parts that process the data, calculate statistics, or create plots – only ever need to know how to work with an xarray Dataset. They don't need to worry about whether the data originally came from NetCDF, CSV, or something else!

It's like having different power adapters for plugs from different countries. Each adapter is different inside because it handles different pin shapes and voltages, but the *other side* of every adapter is the same standard plug that fits into your device.

## How it Works in eViz (Simplified Flow)

Let's see how this fits into the process we started describing:

1.  You run `python autoviz.py -s gridded`.
2.  The `Autoviz` application uses the [Configuration Management](02_configuration_management_.md) to find out which input files are specified for the 'gridded' source. Let's say the config points to a NetCDF file (`data.nc`).
3.  Based on the source type ('gridded') and potentially the file extension (`.nc`), eViz needs to figure out which specialist adapter class is needed. This step is handled by something called the **Data Source Factory** (which we'll cover in the next chapter!).
4.  The Factory creates the correct specialist object, e.g., a `NetCDFDataSource` object.
5.  The `Autoviz` application then tells *this specific* `NetCDFDataSource` object to `load_data('/path/to/data.nc')`.
6.  The `NetCDFDataSource` object uses Python libraries that know how to read NetCDF files. It reads the data and organizes it into an `xarray Dataset`.
7.  The `NetCDFDataSource` object returns the `xarray Dataset` to the `Autoviz` application (or more accurately, the part of the system like the [Data Processing Pipeline](06_data_processing_pipeline_.md) or [Plotting Engine](07_plotting_engine_.md) that requested the data).
8.  The rest of the system works with this `xarray Dataset`, completely unaware that it came from a NetCDF file.

Here's a simple flow diagram:

```{mermaid}
sequenceDiagram
    participant A as Autoviz Object
    participant CM as ConfigManager
    participant F as Data Source Factory (Next Chapter!)
    participant DS as Specific Data Source (e.g., NetCDFDataSource)
    participant File as Data File (.nc, .csv, etc.)
    participant XD as xarray Dataset

    A->>CM: Get Input File Paths
    CM-->>A: '/path/to/data.nc' (and source type 'gridded')
    A->>F: Which DataSource for 'gridded', '.nc'?
    F-->>A: Returns NetCDFDataSource class
    A->>DS: Create NetCDFDataSource() object
    A->>DS: Call load_data('/path/to/data.nc')
    DS->>File: Read file content
    File-->>DS: Raw data
    DS->>DS: Organize into xarray Dataset
    DS-->>A: Return xarray Dataset
    A->>XD: Work with xarray Dataset (Process, Plot, etc.)
```

This diagram shows how the `Autoviz` object, after getting file info from the `ConfigManager`, uses a Factory (we'll explain this fully in the next chapter) to get the right `DataSource`. It then tells that `DataSource` to load the file, and the `DataSource` returns the data as a standardized `xarray Dataset`.

## Peeking at the Code

Let's look at the core pieces of code that make this possible.

First, the blueprint: the `DataSource` abstract base class (`eviz/lib/data/sources/base.py`). This defines the required methods that any specialist class must implement.

```python
# --- File: eviz/lib/data/sources/base.py (Simplified) ---
from abc import ABC, abstractmethod
import xarray as xr
# ... other imports ...

class DataSource(ABC): # ABC means Abstract Base Class
    """
    Abstract base class that defines the interface for all data sources.
    """
    # ... attributes like model_name, config_manager, dataset, metadata ...
    # These are common to all data sources

    @property # This makes 'logger' usable like a variable
    def logger(self):
        # Code to get a logger for messages
        pass

    @abstractmethod # This method *must* be implemented by subclasses
    def load_data(self, file_path: str) -> xr.Dataset:
        """
        Load data from the specified file path into an xarray dataset.

        Args:
            file_path (str): Path to the data file.

        Returns:
            xarray.Dataset: An xarray dataset containing the loaded data.

        This abstract method must be implemented by all concrete data source classes.
        """
        # This method has no implementation here, it's just a placeholder!
        raise NotImplementedError("Subclasses must implement the load_data method.")

    # ... other common methods like validate_data, get_field, close ...
    # These provide a standard way to interact with the loaded dataset

    # Methods like __getattr__ and __getitem__ allow you to do things like
    # `my_data_source['temperature']` instead of `my_data_source.dataset['temperature']`
    # for convenience.
    # ... __getattr__ and __getitem__ implementations ...
```

This simplified code shows the `DataSource` class inheriting from `ABC` and defining `load_data` with the `@abstractmethod` decorator. This tells Python that any class inheriting from `DataSource` *must* provide its own version of the `load_data` method, and that method *must* accept a file path and return an `xarray.Dataset`.

Now, let's look at how one of the specialists implements this. Here's a *very* simplified `NetCDFDataSource` (`eviz/lib/data/sources/netcdf.py`):

```python
# --- File: eviz/lib/data/sources/netcdf.py (Simplified) ---
import xarray as xr
# ... other imports ...
from .base import DataSource # Import the blueprint

class NetCDFDataSource(DataSource): # Inherit from the blueprint
    """Data source implementation for NetCDF files."""
    # ... __init__ and other methods ...

    def load_data(self, file_path: str) -> xr.Dataset:
        """
        Load data from a NetCDF file or URL into an Xarray dataset.
        """
        self.logger.debug(f"Loading NetCDF data from {file_path}")
        try:
            # Use xarray's built-in function to open NetCDF files
            # xarray understands the NetCDF format very well
            dataset = xr.open_dataset(file_path, decode_cf=True)

            self.logger.debug(f"Loaded NetCDF data: {file_path}")

            # Store the loaded dataset
            self.dataset = dataset

            # Return the xarray Dataset
            return dataset

        except FileNotFoundError as exc:
            self.logger.error(f"Error loading NetCDF file: {file_path}. Exception: {exc}")
            raise # Re-raise the exception
        except Exception as exc:
            self.logger.error(f"Error loading NetCDF data: {file_path}. Exception: {exc}")
            raise # Re-raise the exception
```

This simplified snippet of `NetCDFDataSource` shows that it inherits from `DataSource` and provides its *own* implementation of `load_data`. Inside this method, it uses the `xarray.open_dataset` function, which is specifically designed to read NetCDF files, and then it returns the resulting `xarray.Dataset`.

Other specialist classes, like `CSVDataSource` (`eviz/lib/data/sources/csv.py`), implement `load_data` using different libraries (`pandas` for CSV), but they all conform to the same input (`file_path`) and output (`xarray.Dataset`) contract defined by the `DataSource` base class:

```python
# --- File: eviz/lib/data/sources/csv.py (Simplified) ---
import pandas as pd
import xarray as xr
from .base import DataSource # Import the blueprint

class CSVDataSource(DataSource): # Inherit from the blueprint
    """Data source implementation for CSV files."""
    # ... __init__ and other methods ...

    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from a CSV file into an Xarray dataset."""
        self.logger.debug(f"Loading CSV data from {file_path}")

        try:
            # Use pandas to read the CSV file
            data_frame = pd.read_csv(file_path)

            # Convert the pandas DataFrame to an xarray Dataset
            dataset = data_frame.to_xarray()

            self.logger.debug(f"Loaded and converted CSV data: {file_path}")

            # Store the loaded dataset
            self.dataset = dataset

            # Return the xarray Dataset
            return dataset

        except Exception as exc:
            self.logger.error(f"Error loading CSV file: {file_path}. Exception: {exc}")
            raise
```

Again, the `CSVDataSource.load_data` method uses different internal tools (`pandas`) but delivers the data in the required `xarray.Dataset` format.

The `HDF5DataSource` (`eviz/lib/data/sources/hdf5.py`) and `GRIBDataSource` (`eviz/lib/data/sources/grib.py`) classes work similarly, using appropriate libraries (`h5py` or `xarray`'s engines for HDF5, `cfgrib` or `pynio` for GRIB) but always returning an `xarray.Dataset`.

## Benefits of Data Source Abstraction

*   **Modularity:** The code that reads different file formats is separate from the code that processes and plots the data.
*   **Extensibility:** Adding support for a *new* file format simply requires creating a new class that inherits from `DataSource` and implements `load_data` for that format. The rest of eViz doesn't need to change.
*   **Consistency:** Downstream components ([Data Processing Pipeline](06_data_processing_pipeline_.md), [Plotting Engine](07_plotting_engine_.md)) always receive data in the same `xarray Dataset` structure, simplifying their logic.
*   **Maintainability:** Changes to how a specific file format is read are contained within its dedicated `DataSource` class.

## Conclusion

In this chapter, you've learned about **Data Source Abstraction**, the concept that allows eViz to handle data from various file formats (NetCDF, CSV, HDF5, GRIB) in a consistent way. By using an abstract `DataSource` blueprint and specific implementation classes that all promise to return an **xarray Dataset** from their `load_data` method, eViz can connect any supported data file to its standard processing and plotting capabilities. This is like using universal adapters to make different electrical plugs fit a standard socket.

Now that you understand the idea of having different specialist classes for different data types, the next question is: how does eViz *know* which specialist class (which adapter) to use for a given file? This is the job of the **Data Source Factory**.

Ready to learn how eViz selects the right adapter? Let's move on to the next chapter: [Data Source Factory](04_data_source_factory_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
