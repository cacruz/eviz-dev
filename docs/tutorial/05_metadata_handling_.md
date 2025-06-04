# Chapter 5: Metadata Handling

Welcome back! In the [previous chapter](04_data_source_factory_.md), we learned about the **Data Source Factory**, which figures out *which* specific type of **Data Source** (like `NetCDFDataSource`) is needed to read your data file and provides you with the correct one. We saw how this abstraction allows eViz to work with many different file formats while presenting the data consistently as an **xarray Dataset** ([Chapter 3: Data Source Abstraction](03_data_source_abstraction_.md)).

But imagine you have that **xarray Dataset** – a powerful container of numbers. How does eViz know what those numbers *mean*? Is `variable_001` temperature or pressure? Are the spatial coordinates latitude and longitude, or something else? What units are they in (Kelvin, Celsius, Pascals)?

This is where **Metadata Handling** becomes essential.

## The Problem: Data Without Context is Meaningless

Think of receiving a spreadsheet full of numbers. If the columns aren't labeled (no variable names), you don't know what each column represents. If the numbers are temperatures but you don't know the units (Fahrenheit, Celsius, Kelvin), you can't use them correctly, compare them, or plot them meaningfully. If there are location columns but you don't know if they're latitude/longitude or X/Y coordinates, you can't make a map.

Data files, especially in scientific fields, are usually designed to include this crucial "data about the data" – the **Metadata**.

eViz needs to understand this metadata to:

*   Find the specific variables you want to plot (like 'temperature' or 'precipitation').
*   Interpret the dimensions correctly (e.g., which dimension is time, which is latitude, which is longitude).
*   Know the units of variables to apply conversions if needed or label plots correctly.
*   Get descriptive names or titles for variables or the whole dataset.
*   Understand coordinate systems.

Without robust Metadata Handling, eViz couldn't properly interpret the **xarray Dataset** it gets from the **Data Source**, making accurate processing and plotting impossible.

## What is Metadata in eViz?

Metadata in eViz primarily refers to information stored within your data files (often in the headers or attributes of formats like NetCDF, HDF5) or provided in the configuration files. Key types include:

*   **Variable Names:** The unique identifier for each set of data (e.g., `'tas'`, `'prcp'`, `'temp'`).
*   **Dimensions:** The axes along which the data varies (e.g., `'time'`, `'lat'`, `'lon'`, `'plev'`, `'level'`).
*   **Units:** The physical units of a variable (e.g., `'K'`, `'mm/day'`, `'hPa'`).
*   **Attributes:** Descriptive information attached to a variable or the entire dataset (e.g., `'long_name': 'Surface Air Temperature'`, `'standard_name': 'air_temperature'`, `'title': 'My Climate Simulation'`).
*   **Coordinate Systems:** Information about the geographic projection or coordinate system used.

## How eViz Gets and Stores Metadata

eViz primarily gets metadata in two main ways, and stores it in accessible locations:

1.  **From the Data File Itself (via xarray):** When a **Data Source** loads a file into an **xarray Dataset** ([Chapter 3](03_data_source_abstraction_.md)), `xarray` automatically reads most standard metadata (like dimensions, variable names, units, and attributes) and attaches it directly to the dataset and its variables. This is the primary, most reliable source for file-specific metadata.
    *   You can access global attributes of the dataset using `dataset.attrs`.
    *   You can access attributes of a specific variable using `dataset['variable_name'].attrs`.
    *   Dimensions are available via `dataset.dims` and for a variable via `dataset['variable_name'].dims`.

2.  **From Configuration Files:** Metadata can also come from the configuration files ([Chapter 2: Configuration Management](02_configuration_management_.md)). This includes:
    *   Which variables to plot (listed under `inputs` -> `to_plot`).
    *   Mappings between standard eViz dimension names ('tc', 'xc', 'yc', 'zc' for time, longitude, latitude, vertical level) and the actual dimension names used in your specific data file (e.g., mapping 'lon', 'latitude', 'x' all to 'xc'). These mappings are often stored in separate YAML files like `meta_coordinates.yaml` and loaded via utilities.
    *   Overriding or adding metadata like preferred plot titles, units, or descriptive names for plotting. These are often found in the specifications (`_specs.yaml`) files.

## The Role of `metadump.py` (Revisited)

We first encountered `metadump.py` in [Chapter 1: Autoviz Application](01_autoviz_application_.md) as a tool you could run directly with `--file`. Now you can see its deeper connection to Metadata Handling and [Configuration Management](02_configuration_management_.md).

The main job of `metadump.py` is to **inspect a data file (like NetCDF) and extract its raw metadata**, then **use that extracted metadata to suggest and generate initial configuration files**.

Let's revisit the command:

```bash
python metadump.py /path/to/your/data/file.nc --app my_config.yaml --specs my_specs.yaml
```

When you run this, `metadump.py` does the following (simplified):

1.  It uses `xarray.open_dataset` (similar to a **Data Source** class) to load the data file.
2.  It accesses the `.attrs` and `.dims` of the resulting `xarray Dataset` and its variables to get the raw metadata.
3.  It applies some eViz logic (like checking dimensions to guess if a variable is plottable on an XY map or XT time series).
4.  It uses the extracted metadata (variable names, dimensions, units, descriptions) and its own logic to structure the output.
5.  It writes this structured information into the requested YAML files (`my_config.yaml`, `my_specs.yaml`), presenting the metadata in a way that eViz's [Configuration Management](02_configuration_management_.md) system can load and use.

This process allows you to quickly get a starting point for your configuration based directly on the metadata found *within* your data file, saving you from writing it all manually.

## Accessing Metadata in Code

Once data is loaded into an `xarray Dataset` by a **Data Source**, accessing its metadata is straightforward using `xarray`'s built-in capabilities.

Here's a tiny example demonstrating accessing variable attributes:

```python
# Assume 'dataset' is a loaded xarray.Dataset object
variable_name = 'temperature'

if variable_name in dataset.data_vars:
    temp_data_array = dataset[variable_name]

    # Access variable attributes
    if 'units' in temp_data_array.attrs:
        units = temp_data_array.attrs['units']
        print(f"Variable '{variable_name}' units: {units}") # Output: units: K

    if 'long_name' in temp_data_array.attrs:
        long_name = temp_data_array.attrs['long_name']
        print(f"Variable '{variable_name}' long name: {long_name}") # Output: long name: Surface Air Temperature

    # Access dimensions
    dimensions = temp_data_array.dims
    print(f"Variable '{variable_name}' dimensions: {dimensions}") # Output: dimensions: ('time', 'lat', 'lon')
else:
    print(f"Variable '{variable_name}' not found.")
```

This snippet shows how easy it is to get units, long names, or dimensions directly from a variable (`DataArray`) within the `xarray Dataset`.

## Mapping Dimensions with `meta_coordinates.yaml`

One crucial aspect of metadata handling, especially for gridded data from different models, is figuring out which dimension in the file corresponds to the standard eViz concepts of time, longitude, latitude, and vertical level.

eViz uses configuration files like `meta_coordinates.yaml` (loaded by functions in `eviz/lib/utils.py`, like `read_meta_coords`) to store these mappings. This file might look something like this (simplified):

```yaml
# eviz/lib/config/meta_coordinates.yaml (Simplified)
tc: # Time Coordinate
  gridded: time # For 'gridded' source, time dimension is usually 'time'
  wrf: Times # For 'wrf' source, it might be 'Times'
  lis: time
xc: # X Coordinate (Longitude)
  gridded: lon,longitude # Could be 'lon' or 'longitude'
  wrf: XLONG
  lis: lon
yc: # Y Coordinate (Latitude)
  gridded: lat,latitude # Could be 'lat' or 'latitude'
  wrf: XLAT
  lis: lat
zc: # Z Coordinate (Vertical)
  gridded: plev,level # Could be 'plev' or 'level'
  wrf: Z
  lis: z
```

The `DataSource` classes (specifically the `DataSource` base class and its implementations) use functions like `_get_model_dim_name` (which internally uses `read_meta_coords` from `utils.py`) to look up the *actual* dimension name in the loaded dataset corresponding to 'tc', 'xc', 'yc', or 'zc' based on the `model_name` (source type).

Here's a simplified look at how that mapping logic might be used within a **Data Source**:

```python
# Inside a DataSource method (Simplified)
# (This would likely be called during loading or validation)

# Get the model-specific name for the time dimension
time_dim_name_in_file = self._get_model_dim_name('tc', available_dims=self.get_dimensions())

if time_dim_name_in_file:
    print(f"The time dimension in this dataset is called: {time_dim_name_in_file}")
    # Now we can safely access dataset[time_dim_name_in_file]
    time_coords = self.dataset[time_dim_name_in_file]
    print(f"First time value: {time_coords.values[0]}")
else:
    print("Could not identify a standard time dimension.")

# Similarly for longitude
lon_dim_name_in_file = self._get_model_dim_name('xc', available_dims=self.get_dimensions())
if lon_dim_name_in_file:
     print(f"The longitude dimension is: {lon_dim_name_in_file}")
```

This code snippet shows how `_get_model_dim_name` is used to translate the standard eViz conceptual dimension name ('tc' for time) into the actual name found in the specific dataset being processed (e.g., 'time' or 'Times'). This makes the rest of the code generic, working with 'tc' conceptually, while the metadata handling translates it to the specific dataset's reality.

The function `_get_model_dim_name` itself (seen in the provided code for `eviz/lib/data/sources/base.py` and `metadump.py`) reads from the `meta_coords` dictionary (loaded from the YAML file). It looks up the standard dimension name ('tc'), then looks up the source type ('gridded'), and finds the corresponding dimension name(s) from the config, checking if they exist in the currently loaded dataset's dimensions.

## How Metadata is Used Downstream

Metadata, once accessed and understood, is critical throughout the eViz pipeline:

*   **Data Processing Pipeline ([Chapter 6](06_data_processing_pipeline_.md)):** Uses variable names to select data, dimensions to perform aggregations (e.g., averaging over time), units for potential conversions, and attributes for calculations (e.g., using `standard_name` to identify temperature data for a specific calculation).
*   **Plotting Engine ([Chapter 7](07_plotting_engine_.md)):** Uses variable names and attributes (`long_name`, `units`) for plot titles, axis labels, and color bar labels. Uses dimension information to determine the type of plot possible (XY map needs latitude/longitude dimensions). Uses coordinate system info for map projections.
*   **Validation:** Metadata is checked against expectations (e.g., does this variable have the expected dimensions for an XY plot?).

## Metadata Handling Flow

Here's a simple diagram showing the flow involving metadata:

```{mermaid}
sequenceDiagram
    participant CM as ConfigManager
    participant A as Autoviz Object
    participant F as Data Source Factory
    participant DS_Obj as Specific DataSource Object
    participant File as Data File (.nc, .csv, etc.)
    participant XD as xarray Dataset
    participant UTILS as Utility Functions (read_meta_coords)
    participant META_YAML as meta_coordinates.yaml
    participant PIPELINE as Data Processing Pipeline (Next Chapter!)
    participant PLOTTING as Plotting Engine (Later Chapter!)

    A->>CM: Get Input File Info & Config (incl. specs)
    CM-->>A: File paths, variable names to plot, plot settings
    A->>UTILS: Load standard metadata maps
    UTILS->>META_YAML: Read meta_coordinates.yaml etc.
    META_YAML-->>UTILS: Mappings ('tc' -> 'time')
    UTILS-->>A: Metadata mappings available
    A->>F: Create DataSource for file
    F-->>DS_Obj: DataSource instance
    A->>DS_Obj: Call load_data(file_path)
    DS_Obj->>File: Read data & metadata
    File-->>DS_Obj: Raw data & file metadata
    DS_Obj->>XD: Organize into xarray Dataset (metadata included)
    DS_Obj-->>A: Return xarray Dataset
    A->>PIPELINE: Pass xarray Dataset, Config, and Metadata mappings
    PIPELINE->>XD: Access variable data
    PIPELINE->>XD: Access variable.attrs (units, long_name)
    PIPELINE->>DS_Obj: Use _get_model_dim_name('tc') etc. (using mappings)
    PIPELINE->>PLOTTING: Pass processed DataArrays and Plotting Metadata (titles, labels)
    PLOTTING->>PLOTTING: Create plot using variable data, dimensions, titles, units, etc.
    PLOTTING->>User: Output plot image
```

This diagram illustrates how metadata comes from both the data file (via the `DataSource` and `xarray`) and configuration/utility files (`meta_coordinates.yaml` via `UTILS`). This combined metadata is then used by subsequent steps like the **Data Processing Pipeline** and **Plotting Engine**.

## Conclusion

In this chapter, you've learned that **Metadata Handling** is the process of understanding the "data about the data" contained within your files and configuration. This includes crucial details like variable names, dimensions, units, and attributes. You saw how `metadump.py` can help extract this initially, how `xarray` automatically makes file metadata available once data is loaded, how configuration files provide additional metadata and mappings, and how eViz uses utilities to read standard mappings like those for coordinates. Understanding and utilizing this metadata allows eViz to correctly interpret data, validate inputs, and create meaningful, well-annotated plots.

Now that we understand how eViz loads data and understands what it contains, the next step is to learn how it performs calculations, transformations, and aggregations on that data before plotting. Let's move on to the next chapter: [Data Processing Pipeline](06_data_processing_pipeline_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
