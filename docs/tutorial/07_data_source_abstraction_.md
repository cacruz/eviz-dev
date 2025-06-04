# Chapter 7: Data Source Abstraction

Welcome back to the eViz tutorial! In [Chapter 6: Data Processing Pipeline](06_data_processing_pipeline_.md), we saw how the pipeline acts as an assembly line, preparing data for visualization. The first step in that assembly line is reading the raw data from a file, handled by the `DataReader`.

But Earth system science data comes in many flavors! You might have:
*   Large model outputs in NetCDF format (`.nc`).
*   Observation data from satellites in HDF5 format (`.h5`, `.hdf`).
*   Inventory data from monitoring stations in CSV format (`.csv`).
*   Weather model outputs in GRIB format (`.grib`).

Each of these formats requires different Python libraries to read them correctly (`xarray` for NetCDF, `h5py` or `xarray` for HDF5, `pandas` for CSV, `cfgrib` or `pynio` for GRIB).

Imagine the `DataReader` having to know all these different formats and having complex `if/else` statements inside it:

```python
# Imaginary, overly complicated DataReader read logic
if file_path.endswith('.nc'):
    # Use xarray to open NetCDF
    dataset = xr.open_dataset(file_path)
elif file_path.endswith('.h5') or file_path.endswith('.hdf'):
    # Maybe use h5py or a different xarray engine
    dataset = h5py.File(file_path, 'r') # Or xr.open_dataset(..., engine='h5netcdf')
elif file_path.endswith('.csv'):
    # Use pandas to read CSV
    dataset = pd.read_csv(file_path)
    # Need to convert pandas DataFrame to xarray Dataset? More complexity...
elif file_path.endswith('.grib'):
    # Use cfgrib or pynio
    dataset = xr.open_dataset(file_path, engine='cfgrib')
# ... and so on for every format!
```

This would make the `DataReader` messy and hard to maintain. What if you need to add support for a new file format? You'd have to modify this central `DataReader` code.

## The Problem: Handling Diverse File Formats

The challenge is to provide a consistent way for the [Data Processing Pipeline](06_data_processing_pipeline_.md) (and the rest of the application) to interact with data files, *regardless* of their specific format. The part of the pipeline that needs the data shouldn't have to care if it came from a `.nc`, `.h5`, or `.csv` file.

## The Solution: Data Source Abstraction

This is precisely what the **Data Source Abstraction** solves. Instead of the `DataReader` (or any other part of eViz) working directly with the raw file and format-specific libraries, it works with an **abstract representation** of the data source.

Think of it like a universal adapter or a set of specialized plugs. You have a standard wall socket (the rest of the eViz pipeline), and you need to plug in different devices (NetCDF files, CSV files, etc.). The Data Source Abstraction provides the "plugs" – objects that know how to connect a specific file format to the standard socket.

Each "plug" (or Data Source object) knows how to read *its* specific file type, but they all present the same shape to the wall socket.

## Key Concept: The `DataSource` Base Class

The core of this abstraction is the `DataSource` abstract base class (`eviz/lib/data/sources/base.py`). An abstract base class is like a blueprint or a contract. It defines a set of methods that *must* be implemented by any concrete class that inherits from it.

The `DataSource` base class defines the standard interface that the rest of eViz expects from *any* data source object.

```python
# eviz/lib/data/sources/base.py (Simplified)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import xarray as xr
import logging

@dataclass
class DataSource(ABC): # Inherits from ABC (Abstract Base Class)
    """
    Abstract base class that defines the interface for all data sources.
    """
    model_name: Optional[str] = None
    config_manager: Optional[object] = None
    dataset: Optional[xr.Dataset] = field(default=None, init=False) # Will hold loaded data
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @abstractmethod
    def load_data(self, file_path: str) -> xr.Dataset:
        """
        Load data from the specified file path into an xarray dataset.
        This method MUST be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the load_data method.")

    def get_field(self, field_name: str) -> Optional[xr.DataArray]:
        """Get a specific field (variable) from the dataset."""
        if self.dataset is None:
            self.logger.error("No data has been loaded")
            return None
        try:
            return self.dataset[field_name]
        except KeyError:
            self.logger.error(f"Field '{field_name}' not found in dataset")
            return None

    def get_variables(self) -> List[str]:
        """Get the names of variables in the dataset."""
        if self.dataset is None: return []
        return list(self.dataset.data_vars)

    def close(self) -> None:
        """Close the dataset and free resources."""
        if hasattr(self.dataset, 'close'):
            self.dataset.close()

    # --- Magic methods for easier access ---
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access (like .mean) to the underlying dataset."""
        if self.dataset is None: raise AttributeError(...)
        if hasattr(self.dataset, name): return getattr(self.dataset, name)
        raise AttributeError(...)

    def __getitem__(self, key: str) -> Any:
        """Delegate item access (like ['Temperature']) to the underlying dataset."""
        if self.dataset is None: raise TypeError(...)
        return self.dataset[key]

    # ... other common methods ...
```

**Explanation:**

*   `DataSource(ABC)`: This tells Python that `DataSource` is an abstract base class.
*   `@abstractmethod def load_data(...)`: This is the most important part. It declares that any class inheriting from `DataSource` *must* provide its own implementation of the `load_data` method. The base class itself doesn't implement it.
*   `dataset: Optional[xr.Dataset] = field(default=None, init=False)`: All `DataSource` objects have an attribute `dataset` which, once loaded, is expected to hold an `xarray.Dataset`. By standardizing on `xarray`, eViz makes the data uniform *after* it's loaded, regardless of the original file format.
*   `get_field`, `get_variables`, `close`: These are common methods that provide a standard way to interact with the data once it's loaded into the `dataset` attribute.
*   `__getattr__`, `__getitem__`: These are "magic methods" in Python. They allow you to access attributes (like `my_data_source.mean()`) and items (like `my_data_source['Temperature']`) directly on the `DataSource` object, instead of always having to write `my_data_source.dataset.mean()` or `my_data_source.dataset['Temperature']`. This makes the `DataSource` object itself feel more like the data it represents.

## How Specific File Formats Implement the Abstraction

Now, let's see how the specific file types fit into this. For each supported format (NetCDF, HDF5, CSV, GRIB), there is a concrete class that inherits from `DataSource` and provides its own specific `load_data` implementation.

### `NetCDFDataSource` (`eviz/lib/data/sources/netcdf.py`)

This class knows how to read NetCDF files, typically using `xarray`.

```python
# eviz/lib/data/sources/netcdf.py (Simplified)
import xarray as xr
# ... other imports ...
from .base import DataSource # Imports the base class

class NetCDFDataSource(DataSource): # Inherits from DataSource
    """Data source implementation for NetCDF files."""

    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from a NetCDF file or OpenDAP URL."""
        self.logger.debug(f"Loading NetCDF data from {file_path}")
        try:
            # *** This is the format-specific part! ***
            # Use xarray to open the NetCDF file(s)
            if isinstance(file_path, list) or "*" in file_path:
                 # Handle multiple files (using open_mfdataset)
                 dataset = xr.open_mfdataset(file_path, decode_cf=True, combine="by_coords", parallel=True)
                 self.logger.debug(f"Loaded multiple NetCDF files: {file_path}")
            else:
                 # Handle single file or URL
                 dataset = xr.open_dataset(file_path, decode_cf=True)
                 self.logger.debug(f"Loaded single NetCDF file: {file_path}")

            self.dataset = dataset # Store the loaded xarray Dataset
            # ... potentially extract metadata ...
            return dataset
        except Exception as exc:
            self.logger.error(f"Error loading NetCDF data: {file_path}. Exception: {exc}")
            raise
    # ... other methods specific to NetCDF if needed ...
```

**Explanation:**

*   `NetCDFDataSource(DataSource)`: Declares that this class provides an implementation for the `DataSource` contract.
*   `load_data(self, file_path: str)`: This method *implements* the abstract `load_data` method from the base class. Its logic is specifically for NetCDF files: it uses `xarray.open_dataset` or `xarray.open_mfdataset` to read the data.
*   `self.dataset = dataset`: Once the data is loaded, it stores the resulting `xarray.Dataset` in the `dataset` attribute, fulfilling the requirement from the base class.

### `HDF5DataSource` (`eviz/lib/data/sources/hdf5.py`)

This class handles HDF5 files. It first tries `xarray` and then falls back to `h5py` if needed.

```python
# eviz/lib/data/sources/hdf5.py (Simplified)
import xarray as xr
import h5py
# ... other imports ...
from .base import DataSource

class HDF5DataSource(DataSource): # Inherits from DataSource
    """Data source implementation for HDF5 files."""

    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from an HDF5 file."""
        self.logger.debug(f"Loading HDF5 data from {file_path}")
        try:
            try:
                # Try xarray's h5netcdf engine first
                dataset = xr.open_dataset(file_path, engine="h5netcdf")
                self.logger.info(f"Loaded HDF5 using h5netcdf: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed h5netcdf, falling back to h5py: {e}")
                # *** Fallback: Use h5py and manually convert to xarray ***
                dataset = self._load_with_h5py(file_path) # Internal method using h5py
                self.logger.info(f"Loaded HDF5 using h5py: {file_path}")

            self.dataset = dataset # Store the xarray Dataset
            # ... extract metadata ...
            return dataset
        except Exception as exc:
            self.logger.error(f"Error loading HDF5 file: {file_path}. Exception: {exc}")
            raise

    def _load_with_h5py(self, file_path: str) -> xr.Dataset:
        """Internal helper using h5py and converting to xarray."""
        # *** This is the format-specific part! ***
        h5_file = h5py.File(file_path, 'r')
        # ... logic to manually build xarray Dataset from h5py structure ...
        dataset_dict = {}
        coords_dict = {}
        self._process_h5_group(h5_file, dataset_dict, coords_dict) # Recursive reading
        dataset = xr.Dataset(dataset_dict, coords=coords_dict) # Convert to xarray
        h5_file.close()
        return dataset

    # ... _process_h5_group helper and other methods ...
```

**Explanation:**

*   `HDF5DataSource(DataSource)`: Implements the `DataSource` contract for HDF5.
*   `load_data(...)`: The implementation here tries different approaches (using `xarray` with a specific engine or falling back to `h5py` with manual conversion), but the *output* is always an `xarray.Dataset` stored in `self.dataset`. The method encapsulates the HDF5-specific reading details.

### `CSVDataSource` (`eviz/lib/data/sources/csv.py`)

This class handles CSV files, using `pandas` and converting to `xarray`.

```python
# eviz/lib/data/sources/csv.py (Simplified)
import pandas as pd
import xarray as xr
# ... other imports ...
from .base import DataSource

class CSVDataSource(DataSource): # Inherits from DataSource
    """Data source implementation for CSV files."""

    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from a CSV file or a list of CSV files."""
        self.logger.debug(f"Loading CSV data from {file_path}")
        try:
            # *** This is the format-specific part! ***
            # Use pandas to read the CSV file(s)
            if isinstance(file_path, list):
                 combined_data = pd.concat([pd.read_csv(f) for f in file_path], ignore_index=True)
            else:
                 combined_data = pd.read_csv(file_path)

            # *** Convert pandas DataFrame to xarray Dataset ***
            dataset = combined_data.to_xarray()
            # ... potentially do some CSV-specific processing like identifying time/lat/lon columns ...
            dataset = self._process_data(dataset) # Internal CSV processing

            self.dataset = dataset # Store the xarray Dataset
            # ... extract metadata ...
            return dataset
        except Exception as exc:
            self.logger.error(f"Error loading CSV file: {file_path}. Exception: {exc}")
            raise

    def _process_data(self, dataset: xr.Dataset) -> xr.Dataset:
         """Internal helper to process loaded CSV data (e.g., identify coords)."""
         # Looks for columns like 'date', 'time', 'lat', 'lon'
         # and tries to set them as xarray coordinates
         # ... logic using pandas and xarray.assign_coords ...
         return dataset

    # ... other methods ...
```

**Explanation:**

*   `CSVDataSource(DataSource)`: Implements the `DataSource` contract for CSV.
*   `load_data(...)`: Uses `pandas.read_csv` to load the data into a DataFrame, then converts it to an `xarray.Dataset`. It also calls an internal `_process_data` method to handle typical CSV quirks like using columns for coordinates.

### `GRIBDataSource` (`eviz/lib/data/sources/grib.py`)

This class handles GRIB files.

```python
# eviz/lib/data/sources/grib.py (Simplified)
import xarray as xr
# ... other imports ...
from .base import DataSource

class GRIBDataSource(DataSource): # Inherits from DataSource
    """Data source implementation for GRIB files."""

    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from a GRIB file."""
        self.logger.debug(f"Loading GRIB data from {file_path}")
        try:
            # *** This is the format-specific part! ***
            # Use xarray with the cfgrib engine
            dataset = xr.open_dataset(file_path, engine="cfgrib")
            self.logger.info(f"Loaded GRIB file using cfgrib: {file_path}")

            self.dataset = dataset # Store the xarray Dataset
            # ... potentially do some GRIB-specific processing like renaming coords ...
            dataset = self._process_data(dataset) # Internal GRIB processing
            # ... extract metadata ...
            return dataset

        except ImportError:
            self.logger.error("cfgrib not installed. Cannot read GRIB files.")
            raise ImportError("Please install cfgrib to read GRIB files.")
        except Exception as exc:
            self.logger.error(f"Error loading GRIB file: {file_path}. Exception: {exc}")
            raise

    def _process_data(self, dataset: xr.Dataset) -> xr.Dataset:
        """Internal helper to process loaded GRIB data (e.g., rename dims/coords)."""
        # Looks for GRIB-specific coord names (like 'isobaricInhPa')
        # and renames them to standard names ('lev')
        # ... logic using xarray.rename ...
        return dataset

    # ... other methods ...
```

**Explanation:**

*   `GRIBDataSource(DataSource)`: Implements the `DataSource` contract for GRIB.
*   `load_data(...)`: Uses `xarray.open_dataset` with the `cfgrib` engine. It also calls an internal `_process_data` method for GRIB-specific cleanup like standardizing coordinate names.

## How the Data Processing Pipeline Uses the Abstraction

Now that we have our `DataSource` base class (the contract) and concrete implementations (the specialized plugs), let's revisit the `DataReader` from the [Data Processing Pipeline](06_data_processing_pipeline_.md).

As shown in Chapter 6, the `DataReader` doesn't contain the format-specific reading logic itself. Instead, it relies on the [Data Source Factory](08_data_source_factory_.md) (the topic of our next chapter!) to give it the correct `DataSource` object, and then simply calls the standard `load_data` method on that object.

Here's the key snippet from `DataReader` again:

```python
# eviz/lib/data/pipeline/reader.py (Simplified read_file method)
# ... (imports) ...
from eviz.lib.data.factory import DataSourceFactory # Imports the Factory
from eviz.lib.data.sources import DataSource # Imports the base DataSource

class DataReader:
    # ... __init__ method creates self.factory = DataSourceFactory(...) ...

    def read_file(self, file_path: str, model_name: Optional[str] = None, file_format: Optional[str] = None) -> DataSource:
        """Read data from a file."""
        self.logger.debug(f"Reading file: {file_path}")

        # Check cache first...

        try:
            # *** 1. Use the Factory to get the correct DataSource type ***
            # The factory inspects file_path (or uses file_format hint)
            # and returns an instance of NetCDFDataSource, HDF5DataSource, etc.
            data_source = self.factory.create_data_source(file_path, model_name, file_format=file_format)

            # *** 2. Ask the created DataSource object to load its data ***
            # This calls the specific load_data implementation (NetCDF, HDF5, etc.)
            # which is hidden from the DataReader.
            data_source.load_data(file_path)

            # Store and return the loaded DataSource object
            self.data_sources[file_path] = data_source
            return data_source

        except Exception as e:
            self.logger.error(f"Error reading file: {file_path}. Exception: {e}")
            raise # Re-raise the exception

    # ... other methods ...
```

This is the power of the abstraction! The `DataReader` code is simple because it doesn't need to know the format. It just asks the Factory for the right `DataSource` object and trusts that object to know how to `load_data()`.

The `xarray.Dataset` that is stored inside the `DataSource` object (`data_source.dataset`) provides the uniform interface *after* loading. This means the `DataProcessor` (the next step in the pipeline) also doesn't need to know the original file format; it just operates on a standard `xarray.Dataset`, which all `DataSource` implementations produce.

## Connection to the Rest of eViz

The Data Source Abstraction is fundamental to eViz's flexibility:

*   It works closely with the [Data Source Factory](08_data_source_factory_.md) (next chapter), which is responsible for choosing and creating the correct `DataSource` implementation based on the file.
*   The `DataReader` component within the [Data Processing Pipeline](06_data_processing_pipeline_.md) uses this abstraction by asking the Factory for a `DataSource` object and calling its `load_data` method.
*   The [Source Models](05_source_models_.md) interact with the loaded data *through* the `DataSource` object (often accessed via `data_source.dataset`), which provides a consistent `xarray.Dataset` interface regardless of how the data was originally stored.

By abstracting the file reading process, eViz can easily add support for new data formats just by creating a new class that inherits from `DataSource` and implementing its `load_data` method, without changing the core `DataReader` or other pipeline components.

## Conclusion

In this chapter, we learned about Data Source Abstraction, a core concept in eViz that allows the application to handle diverse data file formats in a consistent way. We saw how the `DataSource` abstract base class defines a standard contract with a crucial `load_data` method, and how specific classes like `NetCDFDataSource`, `HDF5DataSource`, `CSVDataSource`, and `GRIBDataSource` provide format-specific implementations while all producing a standard `xarray.Dataset`. This abstraction hides the complexity of file reading from the rest of the [Data Processing Pipeline](06_data_processing_pipeline_.md) and the [Source Models](05_source_models_.md), making eViz more flexible and easier to extend.

We briefly saw how the `DataReader` relies on a "Factory" to get the right `DataSource` object. In the next chapter, we'll explore this Factory and understand how it determines which specific `DataSource` implementation is needed for a given file.

[Next Chapter: Data Source Factory](08_data_source_factory_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
