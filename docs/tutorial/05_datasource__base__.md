# Chapter 5: DataSource (Base)

Welcome back to the eViz tutorial! In the [previous chapter](04_datasourcefactory_.md), we learned about the **DataSourceFactory**. We saw how this clever component acts like a specialized workshop, figuring out exactly *which* type of data reader is needed for a given file (like NetCDF, CSV, or HDF5) and creating an instance of the correct reader class.

But what *are* these reader objects that the factory creates? What do they all have in common, even though they handle different file formats?

This is where the **DataSource (Base)** concept comes in.

## What is DataSource (Base) and Why Do We Need It?

Imagine you have several different containers, like a wooden box, a metal tin, and a plastic bin. They are made of different materials and look different on the outside, but they all serve the same basic purpose: holding things. If you want to put something *into* or take something *out of* any of these containers, you need a standard way to interact with them – maybe they all have a lid you can open, or an opening you can reach into.

In eViz, data can come from many different types of files: NetCDF (`.nc`), HDF5 (`.h5`), CSV (`.csv`), GRIB (`.grib`), and more. Each format stores data differently and requires specific code to read. However, once the data is read into memory, we want to work with it in a standard way. We need to be able to ask: "What variables are in this data?" or "Give me the temperature variable" or "What are the dimensions (like time, latitude, longitude)?", regardless of whether the data originally came from a NetCDF file or a CSV file.

This is the purpose of the **DataSource (Base)**. It is the **standard interface** or the **blueprint** that all specific data reader classes must follow. It defines a common set of methods and properties that any part of eViz can use to interact with loaded data, no matter its original format.

Think of it as defining the rules for our "data containers". Any class that wants to be a valid eViz data source must implement these rules.

**Why is this important?**

*   **Consistency:** Other parts of eViz, like the [DataPipeline](03_datapipeline_.md) components or the plotters ([Plotter](07_plotter_.md)), don't need to know the specific file format being used. They just interact with the `DataSource` object using its standard methods.
*   **Flexibility:** As we saw with the [DataSourceFactory](04_datasourcefactory_.md), you can add support for a brand new file format by creating a new `DataSource` class that follows the base blueprint. The rest of the system doesn't need to change because it only interacts with the standard `DataSource` interface.
*   **Centralized Data:** The `DataSource` object acts as the primary holder for the loaded data, typically stored in a powerful scientific data structure called an **xarray Dataset**.

### The Core Use Case: Accessing Loaded Data and Metadata

The central job of any `DataSource` instance, once created and loaded, is to provide access to the data and its associated information (metadata).

Imagine the [DataPipeline](03_datapipeline_.md)'s `DataProcessor` or `DataTransformer` component receives a `DataSource` object (created by the [DataSourceFactory](04_datasourcefactory_.md) after reading a file). This component doesn't know if it's dealing with `NetCDFDataSource` or `CSVDataSource`. All it needs to know is:

1.  Can I get the main data structure? (Yes, via an internal attribute or property).
2.  Can I ask for a specific variable, like 'Temperature'? (Yes, via a standard method).
3.  Can I get information *about* the data, like its dimensions or units? (Yes, via standard methods).

The `DataSource` base class defines *how* you ask these questions.

### Key Aspects of the DataSource (Base)

The `DataSource` base class achieves its goal by:

1.  **Being Abstract:** It's defined as an **Abstract Base Class (ABC)**. This means you cannot create a `DataSource` object directly. You *must* create an object of a class that *inherits* from `DataSource` (like `NetCDFDataSource`, `CSVDataSource`, etc.). This enforces that specific implementations follow the blueprint.
2.  **Defining a Required Method (`load_data`):** It includes one or more **abstract methods** that *must* be implemented by any class inheriting from it. The most important one is `load_data(file_path)`, which is responsible for actually opening the specified file and reading its contents. The base class doesn't know *how* to load, but it declares that *any* valid `DataSource` *must* have a way to load data.
3.  **Holding the Data (`self.dataset`):** It includes a standard attribute, `self.dataset`, where the loaded data is stored. eViz uses the **xarray Dataset** format for this. xarray is a popular Python library designed for working with labeled multi-dimensional arrays (like climate or ocean model output), making it ideal for scientific data. All specific `DataSource` classes must load their data into this `self.dataset` attribute.
4.  **Holding Metadata (`self.metadata`):** It includes a standard attribute, `self.metadata`, where information *about* the data (like global attributes, variable properties, dimensions) can be stored.
5.  **Providing Standard Access Methods:** It defines concrete methods (methods that are *not* abstract and *do* have implementations) that work with `self.dataset` and `self.metadata`. Examples include methods to get a specific variable (`get_field`), get all metadata (`get_metadata`), list dimensions (`get_dimensions`), or list variables (`get_variables`).
6.  **Providing Convenience Access:** It uses Python's special methods (`__getattr__`, `__getitem__`) to allow users to interact with the `DataSource` object almost as if it *were* the `xarray.Dataset` directly. For example, if you have a `DataSource` instance called `my_data`, you can often just write `my_data['temperature']` instead of `my_data.dataset['temperature']`.

### How to Use DataSource (Conceptually)

As a user running `autoviz.py`, you don't directly "use" the `DataSource` class in the sense of creating instances or calling its methods in your command. The `DataSource` objects are created internally by the [DataPipeline](03_datapipeline_.md) (specifically, its `DataReader` component, which uses the [DataSourceFactory](04_datasourcefactory_.md)).

The `DataSource` instance is then passed along the [DataPipeline](03_datapipeline_.md) for processing and transforming, and finally given to the plotting components ([Plotter](07_plotter_.md)). These components interact with the data through the standard `DataSource` interface.

So, from your perspective, "using" `DataSource` means knowing that once eViz has read your data file, it will be represented internally by an object that behaves according to the `DataSource` blueprint. This allows the rest of the eViz system to work seamlessly with whatever data format you provided.

### Inside DataSource (Base): The Standard Data Container

Let's look at how the `DataSource` base class and its specific implementations work together.

Imagine the [DataPipeline](03_datapipeline_.md)'s `DataReader` asks the [DataSourceFactory](04_datasourcefactory_.md) to create a `DataSource` for `my_data.nc`.

```{mermaid}
sequenceDiagram
    participant DR as DataReader
    participant DSAF as DataSourceFactory
    participant NetCDFDS_Class as NetCDFDataSource (Class)
    participant NetCDFDS_Inst as NetCDFDataSource (Instance)
    participant DataSource_Base as DataSource (Base Definition)
    participant Xarray_DS as xarray.Dataset

    DR->>DSAF: create_data_source("my_data.nc")
    DSAF->>DSAF: Infer type (NetCDF)
    DSAF->>NetCDFDS_Class: Create instance()
    NetCDFDS_Class-->>NetCDFDS_Inst: Return NetCDFDataSource instance
    NetCDFDS_Inst->>NetCDFDS_Inst: Initializes (calls super().__init__)
    NetCDFDS_Inst-->>DSAF: Return instance
    DSAF-->>DR: Return instance (a NetCDFDataSource, but follows DataSource blueprint)

    DR->>NetCDFDS_Inst: data_source.load_data("my_data.nc")
    activate NetCDFDS_Inst
    NetCDFDS_Inst->>NetCDFDS_Inst: calls xr.open_dataset("my_data.nc") # Format-specific logic
    Xarray_DS-->>NetCDFDS_Inst: Returns xarray Dataset
    NetCDFDS_Inst->>NetCDFDS_Inst: self.dataset = xarray Dataset # Store in base class attribute
    NetCDFDS_Inst->>NetCDFDS_Inst: self.metadata = ... # Populate metadata
    deactivate NetCDFDS_Inst
    NetCDFDS_Inst-->>DR: Return NetCDFDataSource (now with loaded data)

    DR->>NetCDFDS_Inst: data_source.get_variables() # Using base method
    activate NetCDFDS_Inst
    NetCDFDS_Inst->>NetCDFDS_Inst: accesses self.dataset.data_vars # Base method uses self.dataset
    NetCDFDS_Inst-->>DR: Returns list of variable names
    deactivate NetCDFDS_Inst
```

1.  The `DataReader` asks the `DataSourceFactory` for a `DataSource` for `my_data.nc`.
2.  The `DataSourceFactory` determines it needs a `NetCDFDataSource` and creates an instance. This instance inherits from the `DataSource` base.
3.  The `DataReader` then calls the standard `load_data` method on the object it received.
4.  Because the object is a `NetCDFDataSource`, its specific implementation of `load_data` runs. This implementation uses `xarray.open_dataset` (or similar NetCDF-reading code) to read the file.
5.  The loaded data, as an `xarray.Dataset`, is stored in the `self.dataset` attribute provided by the `DataSource` base class.
6.  Now that `self.dataset` is populated, any standard method from the `DataSource` base class (like `get_variables()`, `get_dimensions()`, `get_field()`) will work by accessing this `self.dataset`. Other components in the pipeline or plotting routines can call these standard methods without knowing the data's origin.

### Code Walkthrough (Simplified)

Let's look at the base `DataSource` class and a simple example of a class that inherits from it.

First, the base class (`eviz/lib/data/sources/base.py`):

```python
# eviz/lib/data/sources/base.py (simplified)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import logging
import xarray as xr

@dataclass
class DataSource(ABC): # It's an Abstract Base Class
    """
    Abstract base class that defines the interface for all data sources.
    """
    model_name: Optional[str] = None # Optional: Name of the model
    config_manager: Optional[object] = None # Access to config
    
    # This attribute holds the loaded data - crucial!
    dataset: Optional[xr.Dataset] = field(default=None, init=False)
    
    # This attribute holds metadata
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        return logging.getLogger(__name__)

    @abstractmethod # This method MUST be implemented by subclasses
    def load_data(self, file_path: str) -> xr.Dataset:
        """
        Load data from the specified file path into an xarray dataset.
        """
        # The base class just defines the requirement, it doesn't implement it.
        raise NotImplementedError("Subclasses must implement the load_data method.")
    
    def get_field(self, field_name: str) -> Optional[xr.DataArray]:
        """Get a specific field (variable) from the dataset."""
        if self.dataset is None:
            self.logger.error("No data has been loaded")
            return None
        try:
            # This standard method works on the self.dataset attribute
            return self.dataset[field_name]
        except KeyError:
            self.logger.error(f"Field '{field_name}' not found in dataset")
            return None
            
    # ... other standard methods like get_metadata, get_dimensions, get_variables ...
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying dataset."""
        if self.dataset is None:
            raise AttributeError(...) # Handle error
        # Allows calling dataset.mean() or dataset.sel(...) directly on the DataSource object
        return getattr(self.dataset, name)
    
    def __getitem__(self, key: str) -> Any:
        """Delegate item access to the underlying dataset."""
        if self.dataset is None:
            raise TypeError(...) # Handle error
        # Allows accessing variables like data_source['temperature']
        return self.dataset[key]

    # ... close method and utility methods ...
```

This snippet shows the fundamental structure: it's an `ABC`, defines the placeholder for the loaded data (`self.dataset`), has the required `load_data` method, and provides standard ways (`get_field`, etc., and the special `__getattr__`/`__getitem__`) to interact with the data once it's loaded into `self.dataset`.

Now, let's look at a piece of the `NetCDFDataSource` class (`eviz/lib/data/sources/netcdf.py`) to see how it implements the `load_data` method:

```python
# eviz/lib/data/sources/netcdf.py (simplified)
import logging
import xarray as xr
# ... other imports ...
from .base import DataSource # Import the base class!

@dataclass
class NetCDFDataSource(DataSource): # It inherits from DataSource
    """Data source implementation for NetCDF files."""
    model_name: Optional[str] = None
    config_manager: Optional[object] = None
    # Note: It *doesn't* redefine 'dataset' or 'metadata', it gets them from the base class.

    def __post_init__(self):
        # Call the base class's post_init to make sure it's set up
        super().__post_init__() 

    def load_data(self, file_path: str) -> xr.Dataset: # It implements the abstract method!
        """
        Load data from a NetCDF file or OpenDAP URL into an Xarray dataset.
        """
        self.logger.debug(f"Loading NetCDF data from {file_path}")

        try:
            # --- NetCDF-specific loading logic ---
            # Use xarray to open the NetCDF file
            dataset = xr.open_dataset(file_path, decode_cf=True) 
            self.logger.debug(f"Loaded NetCDF file: {file_path}")
            # --- End of NetCDF-specific loading ---

            # Store the loaded xarray Dataset in the attribute provided by the base class
            self.dataset = dataset 
            
            # Optional: Extract format-specific metadata and store it in self.metadata
            self._extract_metadata(dataset) 

            # Return the dataset (though it's also stored internally)
            return self.dataset

        except Exception as exc:
            self.logger.error(f"Error loading NetCDF data: {file_path}. Exception: {exc}")
            raise

    # ... other methods specific to NetCDF processing (like _rename_dims) ...

    def _extract_metadata(self, dataset: xr.Dataset) -> None:
        """Extract metadata from the dataset (NetCDF specific)."""
        if dataset is None:
            return
        # Populate the base class's metadata attribute
        self.metadata["global_attrs"] = dict(dataset.attrs)
        self.metadata["dimensions"] = {dim: dataset.dims[dim] for dim in dataset.dims}
        # ... extract variable metadata etc. ...

    # ... close method ...
```

This simplified snippet shows that `NetCDFDataSource` inherits from `DataSource` and crucially implements the `load_data` method. Inside `load_data`, it uses `xarray.open_dataset` (the NetCDF-specific part) to get the data, and then it assigns the result to `self.dataset`, making it available to all the standard methods defined in the base `DataSource` class. It also populates the base class's `self.metadata`.

The `CSVDataSource`, `HDF5DataSource`, and `GRIBDataSource` classes would follow a very similar pattern: inherit from `DataSource`, implement `load_data` using their specific libraries (pandas for CSV, h5py/xarray for HDF5, cfgrib/pynio for GRIB), and store the result in `self.dataset`.

### Conclusion

In this chapter, we learned that **DataSource (Base)** is the fundamental concept for representing loaded data in eViz. It acts as an Abstract Base Class, defining a standard interface (a blueprint) that all specific data readers (like `NetCDFDataSource`, `CSVDataSource`, etc.) must follow.

This blueprint requires implementations to provide a `load_data` method and store the loaded data in a standard **xarray Dataset** format within the `self.dataset` attribute. By providing standard methods and convenience accessors (`__getattr__`, `__getitem__`) that operate on `self.dataset`, the `DataSource` base class ensures that the rest of the eViz system can interact with any loaded data source in a consistent way, regardless of its original file format.

Now that we understand how data is loaded and represented in a standard way using `DataSource`, the next step in the workflow is visualizing that data. But where do the plots actually appear? They are drawn on a **Figure (Visualization Canvas)**, which we will explore in the next chapter.

[Next Chapter: Figure (Visualization Canvas)](06_figure__visualization_canvas__.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)