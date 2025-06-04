# Chapter 4: Data Source Factory

Welcome back! In the [previous chapter](03_data_source_abstraction_.md), we learned about **Data Source Abstraction**. We saw how eViz uses an abstract blueprint (`DataSource`) and specific "specialist" classes (like `NetCDFDataSource`, `CSVDataSource`) that all know how to read their particular file format but always produce a standard **xarray Dataset**. This is like having different power adapters that all convert to the same standard plug.

But this raises a new question: When the eViz system, guided by the configuration files ([Chapter 2: Configuration Management](02_configuration_management_.md)), encounters a line that says `inputs: - name: "/path/to/my/data/file.nc"`, how does it know whether to use the `NetCDFDataSource` or the `CSVDataSource` or some other specialist? It can't just randomly pick one!

This is where the **Data Source Factory** comes in.

## The Problem: Choosing the Right Specialist

Imagine you arrive at a big international electronics store needing a power adapter for your device, but you have no idea which type of foreign plug you have (is it Type C? Type G? Type A?). You need someone smart to look at your plug and tell you exactly which adapter you need from the shelves.

In eViz, the "plug" is your data file (`.nc`, `.csv`, etc.), and the "adapters" are the different `DataSource` classes. The system needs that "someone smart" to look at the file (or information about it) and choose the correct `DataSource` specialist class.

## The Solution: The Data Source Factory

The **Data Source Factory** is like the helpful expert at the electronics store or a smart dispatcher. Its only job is to:

1.  Receive information about the data needed (like the file path or a hint about the file type from the configuration).
2.  Examine that information (most commonly, it looks at the file extension like `.nc` or `.csv`).
3.  Determine which specific `DataSource` implementation class is the correct one to use.
4.  Create and provide an instance of that specific `DataSource` class, ready to be used to load the data.

The Factory doesn't *read* the data itself. It's purely focused on *selecting and creating* the right tool (`DataSource` object) for the job.

## How the Factory Fits In (The Flow)

Let's update our flow diagram to include the Data Source Factory:

```{mermaid}
sequenceDiagram
    participant A as Autoviz Object
    participant CM as ConfigManager
    participant F as Data Source Factory
    participant DS_Base as DataSource Blueprint
    participant DS_NC as NetCDFDataSource Class
    participant DS_CSV as CSVDataSource Class
    participant DS_Obj as Specific DataSource Object (e.g., NetCDFDataSource instance)
    participant File as Data File (.nc, .csv, etc.)
    participant XD as xarray Dataset

    A->>CM: Get Input File Info (path, type hint)
    CM-->>A: '/path/to/data.nc' (and source type 'gridded', maybe format 'netcdf')
    A->>F: Create DataSource for '/path/to/data.nc'?
    F->>F: Examine path, type hint...
    F->>DS_NC: (Identifies) Need NetCDFDataSource
    F-->>A: Returns NetCDFDataSource Class (or creates instance directly)
    A->>DS_Obj: Create NetCDFDataSource() object
    A->>DS_Obj: Call load_data('/path/to/data.nc')
    DS_Obj->>File: Read file content
    File-->>DS_Obj: Raw data
    DS_Obj->>DS_Obj: Organize into xarray Dataset
    DS_Obj-->>A: Return xarray Dataset
    A->>XD: Work with xarray Dataset (Process, Plot)
```

This diagram shows that the `Autoviz` object, after getting details from the `ConfigManager`, asks the **Data Source Factory** to provide the appropriate `DataSource` object for a given file path. The Factory looks at the file path (and maybe other hints), figures out it needs the `NetCDFDataSource`, creates one, and gives it back to `Autoviz`. *Then*, `Autoviz` (or the pipeline component it directs) tells that specific `NetCDFDataSource` object to `load_data`.

Notice that the Factory interacts with the *classes* (`DS_NC`, `DS_CSV`) to decide which one to instantiate. It's the architect that knows how to build different types of houses and picks the right blueprint before construction (creating the object instance) begins.

## Inside the Factory: The `DataSourceFactory` Class

The Data Source Factory functionality in eViz is primarily handled by the `DataSourceFactory` class, located in `eviz/lib/data/factory/source_factory.py`.

Let's look at how it identifies and provides the correct `DataSource` class.

First, the Factory needs a way to know which file extensions map to which `DataSource` classes. This mapping is often stored in a **registry**. The `DataSourceFactory` uses a `DataSourceRegistry` for this.

```python
# --- File: eviz/lib/data/factory/registry.py (Simplified) ---
from typing import Dict, Type
from eviz.lib.data.sources import DataSource
from dataclasses import dataclass, field

@dataclass
class DataSourceRegistry:
    """Registry for data source types."""
    # This dictionary stores the mapping, e.g., {'nc': NetCDFDataSource, 'csv': CSVDataSource}
    _registry: Dict[str, Type[DataSource]] = field(default_factory=dict, init=False)

    def register(self, extensions: list[str], data_source_class: Type[DataSource]) -> None:
        """Register a data source class for the specified file extensions."""
        for ext in extensions:
            # Store the class in the dictionary, using the extension as the key
            self._registry[ext.lower().lstrip('.')] = data_source_class

    def get_data_source_class(self, file_extension: str) -> Type[DataSource]:
        """Get the data source class for the specified file extension."""
        ext = file_extension.lower().lstrip('.')
        if ext not in self._registry:
            raise ValueError(f"No data source registered for extension: {file_extension}")
        # Return the class found in the dictionary
        return self._registry[ext]

    # ... other methods like is_supported ...
```

The `DataSourceRegistry` is a simple class that holds a dictionary mapping file extensions (like `'nc'`, `'csv'`) to the actual `DataSource` class objects (like `NetCDFDataSource`, `CSVDataSource`). The `register` method adds entries to this dictionary, and `get_data_source_class` retrieves the class based on an extension.

Now, let's see the `DataSourceFactory` using this registry.

```python
# --- File: eviz/lib/data/factory/source_factory.py (Simplified) ---
import os
from typing import Type, Optional
from eviz.lib.data.sources import (
    DataSource, NetCDFDataSource, CSVDataSource, HDF5DataSource, GRIBDataSource
)
from .registry import DataSourceRegistry # Import the registry

class DataSourceFactory:
    """Factory for creating data source instances."""

    def __post_init__(self):
        """Initial setup."""
        self.registry = DataSourceRegistry() # Create a registry instance
        self._register_default_data_sources() # Populate it with known types

    def _register_default_data_sources(self) -> None:
        """Register the default data source implementations."""
        self.registry.register(['nc', 'netcdf'], NetCDFDataSource)
        self.registry.register(['csv', 'txt'], CSVDataSource)
        self.registry.register(['h5', 'hdf5'], HDF5DataSource)
        self.registry.register(['grib', 'grib2'], GRIBDataSource)
        # ... register other types ...

    def create_data_source(self, file_path: str, reader_type: Optional[str] = None) -> DataSource:
        """ Create a data source instance. """
        if reader_type is not None:
            # If config gave an explicit type hint, use that first!
            reader_type = reader_type.lower().strip()
            if reader_type == 'csv':
                return CSVDataSource(...) # Create and return the object
            elif reader_type in ['netcdf', 'nc']:
                return NetCDFDataSource(...) # Create and return the object
            # ... handle other explicit types ...
            else:
                raise ValueError(f"Unsupported explicit reader type: {reader_type}")

        # If no explicit hint, try to infer from file extension
        _, ext = os.path.splitext(file_path) # Get the extension (like '.nc')
        if ext:
             ext = ext.lower().lstrip('.') # Clean up the extension ('nc')
             try:
                 # Ask the registry for the class associated with this extension
                 data_source_class = self.registry.get_data_source_class(ext)
                 # Create an instance of that class and return it
                 return data_source_class(...)
             except ValueError:
                 # If extension isn't in registry, it's unsupported
                 raise ValueError(f"Unsupported file type extension: {ext}")

        # Handle cases where no extension and no explicit type hint are available
        # (More complex logic might go here, e.g., sniffing file content)
        raise ValueError(f"Could not determine data source type for: {file_path}")

    # ... other methods ...
```

This simplified `DataSourceFactory` shows the core logic:

1.  In its setup (`__post_init__`), it creates a `DataSourceRegistry` and fills it with mappings like `.nc` -> `NetCDFDataSource`, `.csv` -> `CSVDataSource`, etc.
2.  The main method, `create_data_source`, takes the file path and potentially an explicit `reader_type` hint (which might come from the configuration file).
3.  It first checks for the explicit `reader_type`. If provided and known, it immediately creates and returns the corresponding `DataSource` object (`CSVDataSource(...)`, `NetCDFDataSource(...)`).
4.  If no explicit hint is given, it extracts the file extension from the path (`os.path.splitext`).
5.  It then uses the `registry.get_data_source_class(ext)` method to look up which `DataSource` class is registered for that extension.
6.  Finally, it creates an instance of the found class (`data_source_class(...)`) and returns that object.

If the file extension is unknown or no type can be determined, it raises an error, telling the user the file type isn't supported.

## Benefits of Using a Factory

*   **Decoupling:** The code that *uses* the `DataSource` (like the data processing pipeline) doesn't need to know *how* to choose the right one. It just asks the Factory. This makes the pipeline code simpler and cleaner.
*   **Centralized Logic:** All the rules for mapping file types/hints to `DataSource` classes are in one place (the Factory).
*   **Easy to Add New Types:** To support a new file format (say, GRIB), you just need to:
    1.  Create a new `GRIBDataSource` class inheriting from `DataSource` with its own `load_data` implementation.
    2.  Add a line in the Factory's `_register_default_data_sources` method to register the GRIB extensions (`.grib`, `.grib2`) with the `GRIBDataSource` class.
    You don't need to change any other part of the system!

## Conclusion

You've now learned about the **Data Source Factory**, a crucial component that acts as a smart dispatcher. It receives information about needed data files, examines file extensions or configuration hints, and uses a registry to determine and provide the correct **Data Source** specialist (`NetCDFDataSource`, `CSVDataSource`, etc.) instance required to load that data. This keeps the core processing logic separate from the specifics of file format handling and makes the system easier to extend with new data types.

With the configuration loaded and the correct data source identified and created by the factory, eViz is now ready to start thinking about the *contents* of the data. What variables are available? What are their units? What coordinate systems are used? This is the role of **Metadata Handling**.

Ready to see how eViz understands what's inside your data file? Let's move on to the next chapter: [Metadata Handling](05_metadata_handling_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
