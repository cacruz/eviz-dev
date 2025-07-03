# Chapter 1: Data Source

Welcome to the first chapter of the eViz tutorial! We're starting our journey by understanding how eViz handles the raw data you want to visualize. Think of eViz as a chef who needs ingredients to cook a meal. The ingredients are your scientific data files, and the "Data Source" is like the system that brings those ingredients into the kitchen, making them ready to be used, no matter how they were originally packaged.

## What Problem Does Data Source Solve?

Scientific data comes in many different formats. You might have climate model output in NetCDF files, observation data in CSV spreadsheets, or other data in HDF5, GRIB, or Zarr formats. Each of these formats is like a different language for packaging data. If eViz had to learn every language perfectly every time it wanted to read a file, it would be very complicated!

The "Data Source" concept solves this by acting as a universal translator. Its job is to take a file (or a collection of files) in *any* supported format and present the data inside in a single, consistent way that eViz can easily understand and work with.

## The Universal Data Language: xarray

The standard language eViz uses internally is based on a powerful Python library called `xarray`. If you've ever used Pandas for working with tables (like in a spreadsheet), xarray is similar but designed for multi-dimensional data – perfect for scientific datasets that have dimensions like time, latitude, longitude, and vertical levels.

So, the main goal of any Data Source in eViz is to read your file and load its contents into an `xarray.Dataset` object. Once the data is in this xarray format, the rest of eViz doesn't need to worry about whether it came from a NetCDF file or a CSV file – it just knows how to work with the `xarray.Dataset`.

## Our First Use Case: Loading a File

Let's imagine you have a NetCDF file named `temperature_data.nc` and you want to load it into eViz and access the 'temperature' variable inside.

How do you tell eViz to open this specific file and give you the data? You use a Data Source.

## Creating a Data Source

You don't manually pick which *specific* Data Source class to use (like `NetCDFDataSource` or `CSVDataSource`). Instead, eViz uses a helper called a `DataSourceFactory`. You give the factory the path to your file, and it figures out the correct type of Data Source to create based on the file extension or other clues.

Here's a simplified example of how you might create and use a Data Source:

```python
# Imagine this is part of eViz's internal logic
from eviz.lib.data.factory.source_factory import DataSourceFactory

# 1. Create the Factory (like hiring the chief translator)
factory = DataSourceFactory()

# 2. Tell the Factory which file you have
# Let's use a placeholder path for now
file_path = "path/to/your/temperature_data.nc"
print(f"Asking factory to create a Data Source for: {file_path}")

# 3. The Factory creates the correct Data Source object
# Based on the '.nc' extension, it will create a NetCDFDataSource
data_source = factory.create_data_source(file_path)
print(f"Created a Data Source object of type: {type(data_source)}")
```

In this code:
*   We import `DataSourceFactory`.
*   We create an instance of the factory.
*   We call `factory.create_data_source()` with the path to our file. The factory inspects the path (like checking the `.nc` extension) and gives us back the appropriate `DataSource` object.

Now that we have the `data_source` object, it's ready to load the data.

## Loading the Data

The Data Source object knows *how* to read its specific file format. To actually load the data into the `xarray.Dataset` format, you call the `load_data` method:

```python
# Continuing from the previous example
# The data_source object was created by the factory

print(f"Loading data from {file_path}...")
# 4. Call the load_data method to read the file contents
# This will read the NetCDF file and store the data in an xarray.Dataset
dataset = data_source.load_data(file_path)
print("Data loaded successfully!")
print(f"Loaded data is an object of type: {type(dataset)}")
print("Variables available in the dataset:", list(dataset.data_vars))
```

The `load_data` method does the heavy lifting. It uses libraries specific to the file format (like `xarray`'s built-in NetCDF support, `h5py` for HDF5, `pandas` for CSV, `cfgrib` for GRIB, `zarr` for Zarr) to read the raw bytes and structure them into a meaningful `xarray.Dataset`.

The `dataset` variable now holds your data in the standard xarray format.

## Accessing Data and Variables

One of the nice things about the `DataSource` object is that it acts a lot like the `xarray.Dataset` it contains. You can often access variables directly using square brackets `[]`, just like you would with the dataset itself.

```python
# Continuing from the previous example
# data_source now has the loaded dataset inside

# 5. Access a variable directly from the Data Source
# This is like saying, "Give me the 'temperature' ingredient"
print("Attempting to get the 'temperature' variable...")
temperature_variable = data_source['temperature'] # Using [] access

print(f"Successfully accessed 'temperature' variable.")
print("Type of the variable data:", type(temperature_variable))
print("Dimensions of the variable:", temperature_variable.dims)
print("Shape of the variable:", temperature_variable.shape)

# You can even access xarray dataset methods directly!
# print("Mean temperature:", data_source['temperature'].mean().values) # Example calculation
```

Notice how we used `data_source['temperature']` directly. This is thanks to a neat feature in the base `DataSource` class (`eviz/lib/data/sources/base.py`) that lets it "delegate" requests for items (like variables) and attributes (like `.mean()`) to the underlying `xarray.Dataset`. This makes working with the `DataSource` object feel very natural if you're familiar with xarray.

Finally, it's good practice to close the data source when you're done to free up system resources:

```python
# Continuing from the previous example

# 6. Close the Data Source when finished
print("Closing the Data Source...")
data_source.close()
print("Data Source closed.")
```

## Under the Hood: How It Works

Let's peek behind the curtain to see how eViz makes this happen.

### The Blueprint: `DataSource` Base Class

At the core is the `DataSource` base class (`eviz/lib/data/sources/base.py`). This class is like a blueprint or a contract. It defines the essential methods that *any* Data Source must have, such as `load_data()` and `close()`. It also holds the loaded `xarray.Dataset` and provides common functionality like accessing variables and dimensions.

The `load_data` method in the base class is marked as `abstract`, meaning it *must* be implemented by any class that inherits from `DataSource`.

### The Translators: Specific Data Source Classes

For each supported file format, there's a specific class that inherits from `DataSource` and provides the concrete implementation for `load_data`.

*   `eviz/lib/data/sources/netcdf.py`: Contains `NetCDFDataSource`. Its `load_data` uses `xarray.open_dataset` with the `'netcdf4'` engine or `xarray.open_mfdataset` for multiple files.
*   `eviz/lib/data/sources/hdf5.py`: Contains `HDF5DataSource`. Its `load_data` tries `xarray.open_dataset` with `'h5netcdf'` but can fall back to using the `h5py` library directly if needed.
*   `eviz/lib/data/sources/csv.py`: Contains `CSVDataSource`. Its `load_data` uses the `pandas` library to read CSVs and then converts the resulting DataFrame to an `xarray.Dataset`.
*   `eviz/lib/data/sources/grib.py`: Contains `GRIBDataSource`. Its `load_data` uses `xarray.open_dataset` with the `'cfgrib'` or `'pynio'` engines.
*   `eviz/lib/data/sources/zarr.py`: Contains `ZARRDataSource`. Its `load_data` uses `xarray.open_dataset` with the `'zarr'` engine.

Each of these classes knows the specific details of *its* format but ultimately produces the same `xarray.Dataset` output.

Here's a tiny snippet from `NetCDFDataSource` showing its `load_data`:

```python
# Inside eviz/lib/data/sources/netcdf.py
class NetCDFDataSource(DataSource):
    # ... (other methods) ...

    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from a NetCDF file or OpenDAP URL."""
        self.logger.debug(f"Loading NetCDF data from {file_path}")
        try:
            # Uses xarray's built-in NetCDF capability
            dataset = xr.open_dataset(file_path, decode_cf=True) # Simplified
            self.logger.debug(f"Loaded single NetCDF file: {file_path}")

            self.dataset = dataset # Store the dataset
            # ... (other processing like dimension renaming) ...
            return dataset

        except Exception as exc:
            self.logger.error(f"Error loading NetCDF data: {file_path}. Exception: {exc}")
            raise
```

You can see it uses `xr.open_dataset`, which is xarray's way of reading NetCDF files. It then stores the resulting dataset in `self.dataset`.

### The Selector: `DataSourceFactory` and `DataSourceRegistry`

The job of picking the *right* specific Data Source class for a given file path belongs to the `DataSourceFactory` (`eviz/lib/data/factory/source_factory.py`).

The factory works with a `DataSourceRegistry` (`eviz/lib/data/factory/registry.py`). The registry is simply a list (or more accurately, a dictionary) that maps file extensions (like 'nc', 'csv', 'h5') to the corresponding Data Source classes (`NetCDFDataSource`, `CSVDataSource`, `HDF5DataSource`).

When you call `factory.create_data_source(file_path)`, the factory does this:

1.  It looks at the `file_path` (checking extension, or if it's a URL).
2.  It asks the `DataSourceRegistry`, "Which class handles files with this extension?"
3.  The registry looks up the extension in its map and tells the factory the correct class (e.g., `NetCDFDataSource` for '.nc').
4.  The factory then creates an instance of that specific class and returns it to you.

Here's a simple sequence diagram showing this process:

```{mermaid}
sequenceDiagram
    participant User as Your Code
    participant Factory as DataSourceFactory
    participant Registry as DataSourceRegistry
    participant NetCDFSource as NetCDFDataSource
    participant Xarray as xarray

    User->>Factory: create_data_source("my_data.nc")
    Factory->>Registry: get_data_source_class(".nc")
    Registry-->>Factory: returns NetCDFDataSource
    Factory->>NetCDFSource: create instance (new NetCDFDataSource(...))
    NetCDFSource-->>Factory: returns instance
    Factory-->>User: returns NetCDFDataSource instance

    User->>NetCDFSource: load_data("my_data.nc")
    NetCDFSource->>Xarray: open_dataset("my_data.nc")
    Xarray-->>NetCDFSource: returns xarray.Dataset
    NetCDFSource->>NetCDFSource: Store dataset internally
    NetCDFSource-->>User: returns xarray.Dataset
```

This shows how the factory acts as the entry point, ensuring you get the right tool for the job (the correct Data Source class) without you needing to know the specific class names or import them directly.

Here's a simplified look at the factory's `create_data_source` method:

```python
# Inside eviz/lib/data/factory/source_factory.py
class DataSourceFactory:
    # ... (initialization and registration of default sources) ...

    def create_data_source(self, file_path: str, model_name=None, reader_type=None, file_format=None) -> DataSource:
        """ Create a data source instance for the specified file or URL. """

        # Simplified logic for demonstration
        if reader_type is not None:
            # Use explicit type if provided
            if reader_type.lower() == 'csv':
                return CSVDataSource(model_name, self.config_manager)
            # ... handle other explicit types ...
            else:
                raise ValueError(f"Unsupported reader type: {reader_type}")

        # Infer from file extension or URL
        if is_opendap_url(file_path):
            return NetCDFDataSource(model_name, self.config_manager) # OpenDAP is like NetCDF over web

        # Get extension (simplified)
        _, ext = os.path.splitext(file_path)
        if ext.startswith('.'):
            ext = ext[1:]

        if not ext:
             raise ValueError(f"Could not determine file type for: {file_path}")

        try:
            # Ask the registry for the correct class
            data_source_class = self.registry.get_data_source_class(ext)
        except ValueError:
            raise ValueError(f"Unsupported file type: {ext}")

        # Create an instance of the found class and return it
        return data_source_class(model_name, self.config_manager)

```

This snippet shows how the factory first checks if an explicit `reader_type` is given, then tries to infer from the file path (checking URLs, extensions), and finally uses the `registry` to get the class. It then creates and returns an instance of that class.

## Summary

In this chapter, we learned that the Data Source concept in eViz is all about standardizing access to scientific data, regardless of its original file format.

*   A `DataSource` object represents a single data file or collection.
*   Its primary job is to load the data into a standard `xarray.Dataset`.
*   Specific classes (like `NetCDFDataSource`, `CSVDataSource`) handle the details of reading different formats.
*   The `DataSourceFactory` automatically selects and creates the correct `DataSource` type based on the file path.
*   You can easily access loaded data and variables through the `DataSource` object, which behaves much like the underlying `xarray.Dataset`.

This abstraction is crucial because it means the rest of eViz (like the parts responsible for plotting or analysis) only needs to know how to work with `xarray.Dataset`s, making the application much more flexible and easier to extend with support for new data formats in the future.

Now that we know how eViz gets the raw data, the next question is: How does eViz know *what* specifically to do with that data (which variables to plot, what kind of plot to make, titles, colors, etc.)? This is where configuration comes in.

Let's move on to the next chapter to learn about the [Config Manager](02_config_manager_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)