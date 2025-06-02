# Chapter 4: DataSourceFactory

Welcome back! In the [previous chapter](03_datapipeline_.md), we explored the **DataPipeline**, the assembly line responsible for taking raw data from files and getting it ready for visualization. We saw how the `DataPipeline` uses components like the `DataReader` to perform the initial loading step.

But this raises a question: How does the `DataReader` know *how* to read a file? Files come in many different formats – NetCDF, CSV, GRIB, HDF5, just to name a few used in scientific contexts. Each format requires specific code to open it and understand its structure. Does the `DataReader` have to contain complex logic to handle *every* possible file type? That sounds complicated and hard to manage!

This is exactly the problem the **DataSourceFactory** solves.

## What is DataSourceFactory and Why Do We Need It?

Imagine you go to a specialized workshop that builds custom tools. You don't tell the workshop owner *exactly* how to forge the metal, shape the handle, or sharpen the blade for a specific type of tool. Instead, you tell them *what kind* of tool you need (e.g., a wrench, a screwdriver, a saw), and they know which expert craftsman and specific tools in their workshop are needed to build *that exact* tool.

In eViz, when the `DataReader` needs to read a file, it doesn't know or care if it's a NetCDF file or a CSV file or something else. All it knows is "I need an object that can give me data from this file path." It asks the **DataSourceFactory** for this object.

The **DataSourceFactory** is like that specialized workshop. Its job is to figure out *what kind* of data source you have (based on the file path, especially the extension like `.nc` or `.csv`, or sometimes an explicit format you specify) and then create the correct specialized object that knows how to read *that specific type* of file.

These specialized objects are called **DataSource** instances (like `NetCDFDataSource`, `CSVDataSource`, etc., which we'll cover in the next chapter). The factory knows which `DataSource` class is needed and creates the instance.

**Why is this useful?**
*   **Simplified DataReader:** The `DataReader` doesn't need to know the details of reading NetCDF or CSV. It just knows how to ask the factory.
*   **Easy to Add New Types:** If you want to add support for a new file type (say, Shapefiles), you just need to create a new `ShapefileDataSource` class and tell the factory how to recognize `.shp` files and map them to this new class. You *don't* have to modify the existing `DataReader` or other parts of the pipeline. This makes the system much more flexible and maintainable.

### A Simple Use Case: Getting the Right Reader for a File

Let's go back to our [DataPipeline](03_datapipeline_.md) example where the `DataReader` needs to process `weather.nc`.

When the `DataReader` is asked to read this file, its internal logic looks something like this:

```python
# Inside DataReader (simplified concept)
def read_file(self, file_path, ...):
    # I don't know HOW to read this file... 🤔
    # Let's ask the factory!
    data_source_factory = DataSourceFactory(...) # Get the factory instance
    
    # Ask the factory to create the correct DataSource object for this file
    specific_reader_object = data_source_factory.create_data_source(file_path, ...)
    
    # Now I have the right object, I can just tell it to read the data
    data_source_object.read() # This calls the read method on the specific reader (e.g., NetCDFDataSource)
    
    return data_source_object
```

The `DataReader` delegates the crucial task of *choosing and creating* the right reader to the `DataSourceFactory`. When `file_path` is `weather.nc`, the factory looks at the `.nc` extension and knows it needs to create a `NetCDFDataSource` object. If the path was `temperatures.csv`, it would create a `CSVDataSource` object.

### How to Use the DataSourceFactory (Conceptually)

As with the [DataPipeline](03_datapipeline_.md), you typically don't interact directly with the `DataSourceFactory` as a user running `autoviz.py`. The `DataPipeline` (specifically its `DataReader` component) is the primary user of the `DataSourceFactory`.

You influence the factory's behavior indirectly through the [ConfigManager](02_configmanager_.md) by specifying the `file_path` and optionally the `file_format` in your configuration:

```yaml
# Part of a config file (used by ConfigManager)

inputs:
  - name: ocean_model_output.nc # .nc -> Factory knows to make NetCDFDataSource
    location: /path/to/data
    exp_id: ocean_run

  - name: sensor_readings.csv # .csv -> Factory knows to make CSVDataSource
    location: /path/to/sensors
    exp_id: sensor_data

  - name: weather_radar # No extension, so maybe specify format
    location: /path/to/radar_data
    exp_id: radar_scan
    file_format: grib # Explicitly tell the factory it's GRIB
# ...
```

When the `DataPipeline` gets these inputs from the [ConfigManager](02_configmanager_.md), its `DataReader` will pass these details to the `DataSourceFactory`, which then creates the appropriate `DataSource` object for each entry.

### Inside the DataSourceFactory: The Workshop's Process

Let's peek inside the `DataSourceFactory` to see how it decides which `DataSource` object to build.

The core mechanism involves a **registry**. Think of the registry as a list of all the specialized craftsmen (the `DataSource` classes) and which types of jobs (file extensions) they are trained for.

Here's a simplified sequence of what happens when the `DataReader` calls `create_data_source('weather.nc', ...)`:

```{mermaid}
sequenceDiagram
    participant DR as DataReader
    participant DSAF as DataSourceFactory
    participant DSR as DataSourceRegistry
    participant NetCDFDS as NetCDFDataSource (Class)

    DR->>DSAF: create_data_source('weather.nc', ...)
    DSAF->>DSAF: Check for explicit format (e.g., 'netcdf')
    DSAF->>DSAF: Infer format from file extension (.nc)
    DSAF->>DSR: get_data_source_class('.nc')
    DSR->>DSR: Look up '.nc' in its internal map
    DSR-->>DSAF: Return NetCDFDataSource class
    DSAF->>NetCDFDS: Create instance (NetCDFDataSource(...))
    NetCDFDS-->>DSAF: Return NetCDFDataSource instance
    DSAF-->>DR: Return NetCDFDataSource instance (the reader object)
```

1.  **Request Received:** The `DataSourceFactory` receives the request from the `DataReader` with the file path (`weather.nc`).
2.  **Format Check/Inference:** It first checks if an explicit `file_format` or `reader_type` was provided (from the [ConfigManager](02_configmanager_.md)). If not, it looks at the file extension (`.nc`).
3.  **Registry Lookup:** It asks its internal `DataSourceRegistry`: "Which `DataSource` class handles files with the extension `.nc`?".
4.  **Registry Response:** The `DataSourceRegistry` looks up `.nc` in its stored mappings and returns the `NetCDFDataSource` class.
5.  **Instance Creation:** The `DataSourceFactory` receives the class and uses it to create a new object (an instance) of `NetCDFDataSource`.
6.  **Return Object:** The factory returns the newly created `NetCDFDataSource` object back to the `DataReader`.

Now, the `DataReader` has the specific object it needs to read `weather.nc`.

### Code Walkthrough (Simplified)

Let's look at small code snippets from `eviz/lib/data/factory` to see these pieces.

First, the `DataSourceRegistry` (`registry.py`). This is where the mapping lives.

```python
# eviz/lib/data/factory/registry.py (simplified)
from typing import List, Dict, Type
from eviz.lib.data.sources import DataSource # The base class (next chapter!)
from dataclasses import dataclass, field

@dataclass
class DataSourceRegistry:
    """Registry for data source types."""
    # This dictionary stores the mapping: extension (string) -> DataSource Class (Type)
    _registry: Dict[str, Type] = field(default_factory=dict, init=False)

    def register(self, extensions: List[str], data_source_class: Type[DataSource]) -> None:
        """Register a data source class for the specified file extensions."""
        for ext in extensions:
            # Store the mapping, ensuring extension is lowercase and without a leading dot
            self._registry[ext.lower().lstrip('.')] = data_source_class
            # Example: _registry['nc'] = NetCDFDataSource

    def get_data_source_class(self, file_extension: str) -> Type[DataSource]:
        """Get the data source class for the specified file extension."""
        ext = file_extension.lower().lstrip('.')
        if ext not in self._registry:
            # If we don't know this extension, raise an error
            raise ValueError(f"No data source registered for extension: {file_extension}")
        return self._registry[ext]

    # ... other methods like is_supported ...
```

This short class is essentially just a dictionary that stores the relationship between file extensions and the Python classes that can handle them. The `register` method adds entries, and `get_data_source_class` retrieves the correct class.

Next, the `DataSourceFactory` itself (`source_factory.py`). This class uses the registry.

```python
# eviz/lib/data/factory/source_factory.py (simplified)
import os
from typing import Type, List, Optional
# Import the specific DataSource classes the factory can create
from eviz.lib.data.sources import (
    DataSource,
    NetCDFDataSource,
    CSVDataSource,
    GRIBDataSource # Example: for .grib files
    # ... other specific DataSource classes ...
)
from .registry import DataSourceRegistry # Import the registry

@dataclass
class DataSourceFactory:
    """Factory for creating data source instances."""
    config_manager: Optional[object] = None # Factory might need config later
    registry: DataSourceRegistry = field(init=False) # The factory HAS a registry

    def __post_init__(self):
        """Initialize the Factory and register default sources."""
        self.registry = DataSourceRegistry() # Create the registry instance
        self._register_default_data_sources() # Populate the registry

    def _register_default_data_sources(self) -> None:
        """Register the built-in data source implementations."""
        # Tell the registry which class handles which extensions
        self.registry.register(['nc', 'netcdf', 'dods'], NetCDFDataSource)
        self.registry.register(['csv', 'txt'], CSVDataSource)
        self.registry.register(['grib'], GRIBDataSource)
        # ... register other default sources ...

    def create_data_source(self, file_path: str, model_name: Optional[str] = None,
                        reader_type: Optional[str] = None,
                        file_format: Optional[str] = None) -> DataSource:
        """ Create a data source instance for the specified file or URL. """

        # 1. Handle explicit format/type first (from config)
        if reader_type is not None or file_format is not None:
             # Simplified: Logic here to map format/type string to a class
             # E.g., if format='netcdf', return NetCDFDataSource(...)
             # If format='csv', return CSVDataSource(...)
             if file_format == 'netcdf' or reader_type == 'netcdf':
                 return NetCDFDataSource(model_name, self.config_manager)
             elif file_format == 'csv' or reader_type == 'csv':
                 return CSVDataSource(model_name, self.config_manager)
             # ... handle other explicit types ...
             else:
                 # Handle unsupported explicit type
                 raise ValueError(f"Unsupported explicit format/reader type: {file_format or reader_type}")

        # 2. If no explicit type, infer from file path
        _, ext = os.path.splitext(file_path) # Get the extension (.nc, .csv)

        # 3. Use the registry to get the correct class
        try:
            data_source_class = self.registry.get_data_source_class(ext)
        except ValueError:
            # If registry doesn't know the extension, raise an error
            raise ValueError(f"Unsupported file type extension: {ext}")

        # 4. Create and return an instance of that class
        return data_source_class(model_name, self.config_manager)

    # ... other utility methods like get_supported_extensions, is_supported ...
```

This code shows the `DataSourceFactory` being initialized with a `DataSourceRegistry`. Its `_register_default_data_sources` method populates this registry with the built-in `DataSource` classes. The key method is `create_data_source`, which first checks for explicit format hints and then, if needed, extracts the file extension and uses the registry (`self.registry.get_data_source_class`) to find the correct class to instantiate.

### Conclusion

In this chapter, we learned about the **DataSourceFactory**, a key component that acts as a specialized workshop for creating the right kind of `DataSource` object needed to read a specific file type. It uses a **DataSourceRegistry** to map file extensions or explicit format requests to the appropriate `DataSource` class (like `NetCDFDataSource` or `CSVDataSource`).

By using a factory pattern, eViz keeps the `DataReader` simple and makes it easy to add support for new data file types without modifying the core data processing logic.

Now that we know how the factory decides *which* object to create, the next logical step is to understand what that object actually *is* and what it does. In the next chapter, we will explore the base class that all these specialized readers inherit from: **[DataSource (Base)](05_datasource__base__.md)**.

[Next Chapter: DataSource (Base)](05_datasource__base__.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)