# Chapter 8: Data Source Factory

Welcome back to the eViz tutorial! In our previous chapter, [Chapter 7: Data Source Abstraction](07_data_source_abstraction_.md), we learned how the `DataSource` base class and its specific implementations (like `NetCDFDataSource` and `CSVDataSource`) provide a consistent way to interact with different file formats. We saw that the `DataReader` component in the [Data Processing Pipeline](06_data_processing_pipeline_.md) doesn't need to know *how* to read a NetCDF or CSV file itself; it just needs a `DataSource` object and can call its standard `load_data()` method.

But this raises a new question: How does the `DataReader` get the *correct* `DataSource` object? If it's given `my_weather_data.nc`, how does it know it needs a `NetCDFDataSource`? If it's given `my_station_data.csv`, how does it know it needs a `CSVDataSource`?

Putting complex `if/else` logic into the `DataReader` to check file extensions would bring back the mess we tried to avoid with abstraction!

## The Problem: Choosing the Right Tool

The problem is deciding which specific `DataSource` implementation (the right "plug" or "specialized adapter" from Chapter 7) is needed for a given file path or format hint. This decision needs to happen *before* the `load_data()` method can be called.

## The Solution: Data Source Factory

This is where the **Data Source Factory** comes in!

Think of the Factory like a smart tool dispenser or a specialized builder. You give it a request (like "I need to work with this `.csv` file" or "I need the reader for NetCDF format"), and it looks at the request, figures out which specific tool (which `DataSource` subclass) is appropriate, creates an instance of that tool, and hands it back to you.

The Factory centralizes the logic for selecting the correct `DataSource` type based on criteria like the file extension or an explicit format specified in the configuration.

## Your Eighth Task: Getting the Right Data Source

Imagine the [Data Processing Pipeline](06_data_processing_pipeline_.md)'s `DataReader` needs to process a file called `sensor_readings.csv`. Its task is to get a `DataSource` object that can handle CSV files.

Instead of doing this itself:

```python
# Imaginary bad code in DataReader
if file_path.endswith('.csv'):
    data_source = CSVDataSource(...) # Manually create the CSV reader
elif file_path.endswith('.nc'):
    data_source = NetCDFDataSource(...) # Manually create the NetCDF reader
# ... and so on ...
```

The `DataReader` uses the **Data Source Factory**:

```python
# Good code in DataReader using the Factory
# (Assume 'factory' is an instance of DataSourceFactory)
data_source = factory.create_data_source(file_path) # Ask the Factory!

# Now I have the right DataSource object, I can use its standard methods
data_source.load_data(file_path)
variable_data = data_source.get_field('Temperature')
# ... and pass it along the pipeline ...
```

The `DataReader` simply asks the `factory` to `create_data_source` for the given `file_path`, and the factory handles the messy details of figuring out if it's a CSV, NetCDF, HDF5, etc., and returning the appropriate `DataSource` object.

## Key Concepts in the Data Source Factory

The Data Source Factory is primarily implemented in the `eviz/lib/data/factory` directory. It involves two main parts:

1.  **`DataSourceRegistry`**: This component acts like a simple phonebook or lookup table. It stores which file extensions (like `.nc`, `.csv`, `.h5`) correspond to which `DataSource` class (`NetCDFDataSource`, `CSVDataSource`, `HDF5DataSource`).
2.  **`DataSourceFactory`**: This is the main Factory class. It contains the `DataSourceRegistry` and the logic to use the registry to make the decision and create the object.

## How the Data Source Factory Selects and Creates

Let's see how the Factory, specifically the `DataSourceFactory`, works when the `DataReader` asks it for a `DataSource` for `sensor_readings.csv`.

Here's a simplified sequence diagram:

```{mermaid}
sequenceDiagram
    participant DataReader as Data Reader
    participant Factory as DataSourceFactory
    participant Registry as DataSourceRegistry
    participant CSVDataSourceClass as CSVDataSource (Class)
    participant CSVDataSourceInstance as CSVDataSource (Instance)

    DataReader->>Factory: "Give me a DataSource for 'sensor_readings.csv'!"
    Factory->>Registry: "What class handles file extension '.csv'?"
    Registry-->>Factory: "That's the CSVDataSource class!"
    Factory->>CSVDataSourceClass: "Create a new instance of yourself!"
    CSVDataSourceClass-->>CSVDataSourceInstance: New CSVDataSource instance created
    Factory-->>CSVDataSourceInstance: Receives the new instance
    Factory-->>DataReader: Returns the CSVDataSource instance
    DataReader->>CSVDataSourceInstance: Calls load_data('sensor_readings.csv')
```

The `DataReader` asks the `Factory`, the `Factory` consults its `Registry` to find the right class, the `Factory` then creates an *instance* of that class and gives it back to the `DataReader`. The `DataReader` never had to know the class name `CSVDataSource` itself.

## Diving Deeper into the Code

Let's look at the key parts of the Factory implementation.

### The Phonebook: `DataSourceRegistry` (`eviz/lib/data/factory/registry.py`)

This class simply holds the mapping from extensions to classes.

```python
# eviz/lib/data/factory/registry.py (Simplified)
from typing import List, Dict, Type
from eviz.lib.data.sources import DataSource
from dataclasses import dataclass, field

@dataclass
class DataSourceRegistry:
    """Registry for data source types."""
    _registry: Dict[str, Type] = field(default_factory=dict, init=False)

    def register(self, extensions: List[str], data_source_class: Type[DataSource]) -> None:
        """Register a data source class for extensions."""
        for ext in extensions:
            ext = ext.lower().lstrip('.') # Clean up extension string
            self._registry[ext] = data_source_class

    def get_data_source_class(self, file_extension: str) -> Type[DataSource]:
        """Get the data source class for the extension."""
        ext = file_extension.lower().lstrip('.')
        if ext not in self._registry:
            raise ValueError(f"No data source registered for extension: {ext}")
        return self._registry[ext]

    # ... methods for checking support, getting supported extensions ...
```

**Explanation:**

*   `_registry`: This is a simple Python dictionary where keys are file extensions (like `'csv'`, `'nc'`, `'h5'`) and values are the corresponding `DataSource` *classes* (`CSVDataSource`, `NetCDFDataSource`, `HDF5DataSource`).
*   `register()`: This method is used to add mappings to the registry. For example, the Factory will call `register(['csv', 'dat'], CSVDataSource)` during its setup.
*   `get_data_source_class()`: This is the method the Factory uses. You give it an extension string (like `'csv'`), and it looks it up in the `_registry` and returns the corresponding class (`CSVDataSource`).

### The Builder: `DataSourceFactory` (`eviz/lib/data/factory/source_factory.py`)

This is the main class that users (like the `DataReader`) interact with.

```python
# eviz/lib/data/factory/source_factory.py (Simplified)
import os
from typing import Type, List, Optional
from eviz.lib.data.sources import (
    DataSource,
    NetCDFDataSource,
    HDF5DataSource,
    CSVDataSource,
    GRIBDataSource # Import the specific DataSource classes
)
# ... other imports ...
from .registry import DataSourceRegistry # Import the Registry

@dataclass
class DataSourceFactory:
    """Factory for creating data source instances."""
    config_manager: Optional[object] = None
    registry: DataSourceRegistry = field(init=False) # Holds an instance of the Registry

    def __post_init__(self):
        """Post-initialization setup."""
        self.registry = DataSourceRegistry() # Create the Registry instance
        self._register_default_data_sources() # Populate the registry

    def _register_default_data_sources(self) -> None:
        """Register the default data source implementations."""
        # Call the registry's register method for each supported type
        self.registry.register(['nc', 'nc4', 'netcdf', 'opendap'], NetCDFDataSource)
        self.registry.register(['h5', 'hdf5'], HDF5DataSource)
        self.registry.register(['csv', 'dat'], CSVDataSource)
        self.registry.register(['grib'], GRIBDataSource)
        # ... register other default types ...

    def create_data_source(self, file_path: str, model_name: Optional[str] = None,
                           file_format: Optional[str] = None) -> DataSource:
        """ Create a data source instance for the specified file or URL. """
        
        # 1. Try to determine the file format or extension
        determined_ext = None
        if file_format:
             # Use explicit format hint if provided in config
             determined_ext = file_format.strip().lower().lstrip('.')
        elif is_url(file_path):
             # Try to get extension from URL path
             path = file_path.split('?')[0]
             _, determined_ext = os.path.splitext(path)
        else:
             # Get extension from local file path
             _, determined_ext = os.path.splitext(file_path)

        # Handle cases where extension is missing or needs inference (like WRF names)
        if not determined_ext:
             # ... logic to try inferring from path name, etc. ...
             pass # Simplified

        determined_ext = determined_ext.lstrip('.') # Clean up

        # 2. *** Use the Registry to find the correct class ***
        try:
            data_source_class = self.registry.get_data_source_class(determined_ext)
        except ValueError:
            self.logger.error(f"Unsupported file type: {determined_ext}")
            raise ValueError(f"Unsupported file type: {determined_ext}")

        # 3. *** Create and return an instance of the found class ***
        # This is the Factory's core job!
        return data_source_class(model_name, self.config_manager)

    # ... methods for getting supported extensions, checking support ...
```

**Explanation:**

*   `registry: DataSourceRegistry = field(init=False)`: The Factory holds an instance of the `DataSourceRegistry`.
*   `__post_init__()`: When the `DataSourceFactory` is created, it also creates its `DataSourceRegistry` and calls `_register_default_data_sources()` to fill it with the known file format mappings.
*   `_register_default_data_sources()`: This method is simple; it just makes calls to `self.registry.register()` for each `DataSource` class that eViz supports by default.
*   `create_data_source()`: This is the main entry point for creating `DataSource` objects.
    *   It first tries to figure out the file format or extension, looking at the `file_format` hint (which can come from the [Configuration System](01_configuration_system_.md)), the file path extension, or even attempting inference from the path name for special cases (like the WRF filename hack mentioned in the original code).
    *   Once it has the extension (or format string), it calls `self.registry.get_data_source_class(determined_ext)` to ask the registry for the correct `DataSource` *class*.
    *   Finally, it calls `data_source_class(model_name, self.config_manager)`. This *creates an instance* of the class that the registry returned (e.g., `CSVDataSource(...)`) and passes it the necessary information (`model_name`, `config_manager`). This newly created instance is then returned.

The `DataReader` receives this instance and can immediately start using its `load_data()` method, without needing to know or care whether it got a `NetCDFDataSource`, `CSVDataSource`, or any other specific type. The complexity of selection is hidden within the Factory.

## Connection to the Rest of eViz

The Data Source Factory plays a specific but vital role in the eViz architecture:

*   It is primarily used by the `DataReader`, which is a component within the [Data Processing Pipeline](06_data_processing_pipeline_.md). The `DataReader` relies on the Factory to provide it with the correct `DataSource` object for any given file.
*   It works directly with the [Data Source Abstraction](07_data_source_abstraction_.md) by creating instances of classes that inherit from the `DataSource` base class (`NetCDFDataSource`, `CSVDataSource`, etc.).
*   It may indirectly use information from the [Configuration System](01_configuration_system_.md) (passed via the `config_manager` during creation) to help determine the file type (e.g., an explicit `file_format` setting in the configuration).
*   The [Source Models](05_source_models_.md) benefit from the Factory because the data they receive (via the pipeline) is handled by the correct `DataSource` implementation, ensuring data loading and initial structure are handled appropriately for the file type.

The Factory isolates the logic for choosing the right `DataSource` type, making the `DataReader` simpler and making it easier to add support for new file formats in the future – you only need to create a new `DataSource` subclass and register it with the Factory.

## Conclusion

In this chapter, we learned about the Data Source Factory, a crucial component that centralizes the logic for selecting and creating the correct type of `DataSource` object based on the file format. We saw how the `DataSourceRegistry` acts as a lookup table and how the `DataSourceFactory` uses this registry to return an instance of the appropriate `DataSource` subclass (like `NetCDFDataSource` or `CSVDataSource`) when requested by the `DataReader`. This design keeps the data loading logic clean and flexible, relying on the [Data Source Abstraction](07_data_source_abstraction_.md) to provide a standard interface regardless of the underlying file type.

This concludes our tour of the core components of eViz! We've seen how configuration guides the application, how the core orchestrates the process, how metadata helps setup, how plotting components draw, how source models specialize, how the pipeline processes data, how abstraction hides file details, and finally, how the factory chooses the right tools.

This is the final chapter in this series. Congratulations on making it this far and learning about the foundational pieces of the eViz project!

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
