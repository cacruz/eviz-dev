# Chapter 9: Metadata Tool (metadump)

Welcome back! In our eViz journey so far, we've explored many components:
*   The [Autoviz Application](01_autoviz_application_.md) as the main director.
*   [Configuration Management](02_configuration_management_.md) providing the detailed plan in YAML.
*   [Data Source Abstraction](03_data_source_abstraction_.md) and the [Data Source Factory](04_data_source_factory_.md) for reading various data files into a standard format.
*   The [Data Processing Pipeline](05_data_processing_pipeline_.md) for cleaning and preparing data.
*   [Plotter Abstraction](06_plotter_abstraction_.md) and the [Plotter Factory](07_plotter_factory_.md) for creating plots using different backends.
*   [Source/Model Specific Logic](08_source_model_specific_logic_.md) providing the expert knowledge for handling specific data types before plotting.

All these components work together beautifully *if* eViz knows exactly what variables exist in your file, what their dimensions are, what units they use, and so on. But how do you, the user, figure out this information in the first place, especially if you're dealing with a new or unfamiliar data file? How do you know what variable names like `T2`, `PSFC`, or `h500` actually mean, or whether `pressure` is in Pascals or hectopascals?

Sure, you could open the file with specialized software or write a quick Python script using `xarray` or `netcdf4`, but wouldn't it be helpful to have a simple, dedicated tool for this specific task?

This is where the **Metadata Tool**, known in eViz as `metadump.py`, comes in handy.

## What is the Metadata Tool (metadump)?

Think of `metadump.py` as a **data file inspector** or a **quick scanner**. Its sole purpose is to open one or two data files (typically NetCDF, but it can work with others supported by `xarray`), read information *about* the data *within* the file (metadata), and present it to you in a useful format.

It can tell you:
*   What variables are available.
*   What dimensions each variable has (like `time`, `lat`, `lon`, `level`).
*   What the data type is (e.g., `float32`, `int64`).
*   Important attributes associated with variables or the whole dataset (like `units`, `long_name`, `valid_range`).

Beyond simply listing this information, `metadump.py` can also **generate snippets of YAML configuration** based on its findings. This is incredibly helpful because it gives you a starting point for creating the detailed configuration files needed by the main [Autoviz Application](01_autoviz_application_.md).

It helps bridge the gap between having a raw data file and writing the configuration plan ([Chapter 2: Configuration Management](02_configuration_management_.md)) that tells eViz how to visualize it.

## Our Use Case: Inspecting a Data File

Let's imagine you have a NetCDF file named `sample_data.nc` and you want to know what's inside so you can decide what to plot and how to write your configuration file.

The primary way to use the Metadata Tool is to run the `metadump.py` script directly from your command line, giving it the path to your file.

```bash
python metadump.py /path/to/your/data/sample_data.nc
```

When you run this command, `metadump.py` will open the file, analyze its contents, and by default, print a list of variables that it thinks are potentially plottable, along with some basic information like dimensions.

```
INFO :: (main:122) : Plottable variables: ['temperature', 'pressure', 'humidity']
```

This simple output already gives you the variable names you can use in your configuration's `to_plot` section ([Chapter 2: Configuration Management](02_configuration_management_.md)).

### Getting More Detail: JSON Output

If you want a more structured view of the metadata, including all dimensions and attributes for each variable and the global attributes of the dataset, you can ask for JSON output using the `--json` flag:

```bash
python metadump.py /path/to/your/data/sample_data.nc --json
```

This will print the detailed metadata to your console in JSON format. You can redirect this output to a file:

```bash
python metadump.py /path/to/your/data/sample_data.nc --json metadata.json
```

The `metadata.json` file will contain a comprehensive description like this (simplified snippet):

```json
{
    "global_attributes": {
        "Conventions": "CF-1.6",
        "title": "Sample Model Output",
        "history": "Created on 2023-10-27"
    },
    "variables": {
        "temperature": {
            "dimensions": ["time", "level", "lat", "lon"],
            "data_type": "float32",
            "attributes": {
                "units": "K",
                "long_name": "Air Temperature",
                "valid_range": [200.0, 350.0]
            }
        },
        "pressure": {
            "dimensions": ["time", "level", "lat", "lon"],
            "data_type": "float32",
            "attributes": {
                "units": "Pa",
                "long_name": "Air Pressure",
                "standard_name": "air_pressure"
            }
        },
        // ... other variables ...
    }
}
```
This JSON output is incredibly valuable because it shows you dimension names (`time`, `level`, `lat`, `lon`) and variable attributes (`units`, `long_name`) which are essential for correctly configuring eViz and potentially useful if you need to implement [Source/Model Specific Logic](08_source_model_specific_logic_.md).

### Generating Configuration Snippets: YAML Output

Perhaps the most powerful feature for beginners is `metadump.py`'s ability to generate YAML snippets that you can use as a starting point for your configuration file ([Chapter 2: Configuration Management](02_configuration_management_.md)). You can ask for two types of YAML output:

*   `--app`: Generates the main application configuration structure (`inputs`, `outputs`, `system_opts`).
*   `--specs`: Generates the detailed variable specifications (like contour levels, plot types per variable).

You typically use these together:

```bash
python metadump.py /path/to/your/data/sample_data.nc --app my_config.yaml --specs my_config_specs.yaml --source gridded
```

*   `--app my_config.yaml`: Tells it to generate the main config into `my_config.yaml`.
*   `--specs my_config_specs.yaml`: Tells it to generate the variable specs into `my_config_specs.yaml`.
*   `--source gridded`: Important hint to `metadump.py` about the type of data, which helps it interpret dimensions correctly using `meta_coords`.

After running this, you'll have two files:

`my_config.yaml` (simplified snippet):

```yaml
# --- Generated by metadump.py ---
inputs:
  - name: /path/to/your/data/sample_data.nc # The file path you provided
    to_plot:
      temperature: xt,xy,yz  # Suggested plot types based on dimensions
      pressure: xt,xy,yz     # Suggested plot types based on dimensions
      humidity: xt,xy,yz
outputs:
  print_to_file: yes
  output_dir: null
  print_format: png
  print_basic_stats: true
  make_pdf: false
system_opts:
  use_mp_pool: false
  archive_web_results: true
```

`my_config_specs.yaml` (simplified snippet):

```yaml
# --- Generated by metadump.py ---
humidity:
  units: 'kg/kg'
  name: Specific Humidity
  xtplot:
    time_lev: all
    grid: yes
  xyplot:
    levels: {0.0: []} # Suggested initial contour level
  yzplot:
    contours: []
pressure:
  units: 'Pa'
  name: Air Pressure
  xtplot:
    time_lev: all
    grid: yes
  xyplot:
    levels: {100000.0: []} # Suggested initial contour level (example)
  yzplot:
    contours: []
temperature:
  units: 'K'
  name: Air Temperature
  xtplot:
    time_lev: all
    grid: yes
  xyplot:
    levels: {273.15: []} # Suggested initial contour level (example: 0C)
  yzplot:
    contours: []
```
These generated files provide a solid starting point for your visualization task! You can then edit them to refine the plot types, change contour levels, specify output directories, etc.

### Two-File Comparison Configuration

`metadump.py` also supports generating a configuration for comparing two files.

```bash
python metadump.py file_A.nc file_B.nc --app compare_config.yaml --specs compare_specs.yaml
```

This will generate similar `_specs.yaml` content based on `file_A.nc` (assuming variables match) and an `_app.yaml` that includes both files in the `inputs` list and adds the `for_inputs` section to enable comparison mode ([Chapter 2: Configuration Management](02_configuration_management_.md)), automatically assigning experiment IDs.

`compare_config.yaml` (simplified snippet):

```yaml
# --- Generated by metadump.py ---
inputs:
  - name: file_A.nc
    to_plot:
      temperature: xt,xy,yz
      # ... other variables ...
    exp_id: <random_id_A> # Automatically generated
    exp_name: null
  - name: file_B.nc
    to_plot: {} # No plots listed here, they use the list from file_A's entry
    location: null
    exp_id: <random_id_B> # Automatically generated
    exp_name: null
for_inputs: # This section is added for comparison
  compare:
    ids: <random_id_A>, <random_id_B> # Tells eViz to compare these two
  cmap: coolwarm # Suggested colormap for difference plots
outputs:
  # ... outputs settings ...
system_opts:
  # ... system settings ...
```

This makes setting up comparison plots much quicker.

## How the Metadata Tool Works (High-Level)

`metadump.py` is a standalone script. When you run it, it parses your command-line arguments (file paths, output options, variables to include/exclude, source name). It then uses Python libraries like `xarray` and `PyYAML` to do its job.

Here's a simple flow for generating YAML configuration:

```{mermaid}
sequenceDiagram
    participant User
    participant MetadumpScript as metadump.py
    participant MetadataExtractor as MetadataExtractor
    participant DataFile as Data File (.nc)
    participant Xarray as xarray library
    participant PyYAML as PyYAML library
    participant OutputFiles as my_config.yaml, my_config_specs.yaml

    User->>MetadumpScript: Run 'python metadump.py file.nc --app app.yaml --specs specs.yaml'
    MetadumpScript->>MetadumpScript: parse_command_line()
    MetadumpScript->>MetadataExtractor: Create MetadataExtractor(config)
    MetadataExtractor->>DataFile: Open file.nc
    DataFile-->>MetadataExtractor: Raw File Handle
    MetadataExtractor->>Xarray: xr.open_dataset()
    Xarray-->>MetadataExtractor: Return xarray.Dataset object
    MetadataExtractor->>MetadataExtractor: Store Dataset, setup coords
    MetadataExtractor->>MetadataExtractor: process()
    MetadataExtractor->>MetadataExtractor: _generate_specs_dict()
    MetadataExtractor->>MetadataExtractor: Loop through variables
    MetadataExtractor->>MetadataExtractor: Check dimensions, get attributes, determine plot types (using helper functions like is_plottable, get_model_dim_name)
    MetadataExtractor-->>MetadataExtractor: Build specs dictionary
    MetadataExtractor->>MetadataExtractor: _generate_app_dict()
    MetadataExtractor-->>MetadataExtractor: Build app dictionary
    MetadataExtractor->>PyYAML: yaml.dump(specs_dict)
    PyYAML-->>MetadataExtractor: Return specs YAML string
    MetadataExtractor->>OutputFiles: Write specs YAML string to specs.yaml
    MetadataExtractor->>PyYAML: yaml.dump(app_dict)
    PyYAML-->>MetadataExtractor: Return app YAML string
    MetadataExtractor->>OutputFiles: Write app YAML string to app.yaml
    MetadumpScript-->>User: Process complete

```
The `autoviz.py` script itself provides a shortcut to run `metadump.py`. If you run `python autoviz.py --file /path/to/my_data.nc`, the `autoviz.py` script's `main` function detects the `--file` argument and uses Python's `subprocess` module to simply run `metadump.py /path/to/my_data.nc` for you, effectively delegating the task. If `--vars` is also included, it passes those along to `metadump.py`.

```python
# --- File: autoviz.py (simplified snippet from main) ---
def main():
    # ... parsing arguments ...
    args = parse_command_line()

    # --- Simplified metadump check ---
    # If --file is used, just run metadump.py instead of Autoviz
    if args.file and args.vars: # If --file AND --vars
        subprocess.run(['python',
                        'metadump.py', args.file[0], # Pass the file path
                        '--vars', *args.vars])      # Pass the vars
        sys.exit() # Exit after metadump finishes
    elif args.file: # If only --file is used
        subprocess.run(['python',
                        'metadump.py', args.file[0]]) # Pass the file path
        sys.exit() # Exit after metadump finishes
    # --- End metadump check ---

    # ... rest of Autoviz logic (only runs if --file is NOT used) ...
    # autoviz = Autoviz(...)
    # autoviz.run()
```

This snippet from `autoviz.py` shows that `autoviz --file` isn't a mode *within* Autoviz, but rather a simple wrapper that executes the `metadump.py` script as a separate process and then exits. This reinforces that `metadump.py` is a distinct utility.

## Inside the Code: The `MetadataExtractor` Class

The core logic of `metadump.py` is encapsulated in the `MetadataExtractor` class.

```python
# --- File: metadump.py (simplified MetadataExtractor __init__ and process) ---
# ... imports and MetadumpConfig dataclass ...

class MetadataExtractor:
    """Main class for extracting metadata and generating configuration files."""

    def __init__(self, config: MetadumpConfig):
        self.config = config
        # Opens the dataset(s) using xarray.open_dataset
        self.dataset = self._open_dataset(config.filepath_1)
        self.dataset_2 = self._open_dataset(config.filepath_2) if config.filepath_2 else None
        # Reads the standard metadata mapping file
        self.meta_coords = u.read_meta_coords()
        # Sets up internal references to standard dimension names like self.tc, self.xc, etc.
        self._setup_coordinates()

        if self.dataset_2:
            self._validate_datasets() # Check if datasets are compatible

    def _open_dataset(self, filepath: Optional[str]) -> Optional[xr.Dataset]:
        # Uses xarray.open_dataset internally - simple!
        # ... error handling ...
        return xr.open_dataset(filepath, decode_cf=True)

    def process(self) -> None:
        """Main processing method to generate all required outputs."""
        # Check if JSON output was requested
        if self.config.json_output:
            self._generate_json_metadata()
            return # Stop here if only JSON is needed

        # Otherwise, generate YAML dictionaries
        specs_dict = self._generate_specs_dict()
        app_dict = self._generate_app_dict()

        # Write dictionaries to files if output paths were provided
        if self.config.specs_output:
            self._write_specs_yaml(specs_dict)
        if self.config.app_output:
            self._write_app_yaml(app_dict)

        # If no output files specified, just list plottable variables
        if not (self.config.specs_output or self.config.app_output):
            filtered_vars = self.get_plottable_vars()
            logger.info(f"Plottable variables: {filtered_vars}")

    # ... other methods like _setup_coordinates, _validate_datasets,
    #     _generate_json_metadata, _get_plottable_vars,
    #     _generate_specs_dict, _generate_app_dict, _write_specs_yaml, _write_app_yaml ...
```

The `MetadataExtractor` is initialized with a `MetadumpConfig` object containing the parsed command-line arguments. Its `__init__` method immediately opens the dataset(s) using `xarray` and loads the `meta_coords` file (which maps standard dimension names like 'tc', 'xc' to model-specific names like 'time', 'longitude'). The `process` method then acts as the main dispatcher, calling different internal methods based on whether JSON or YAML output is requested. If no output files are specified, it simply prints the list of plottable variables.

### Generating JSON Metadata

The `_generate_json_metadata` method builds the dictionary structure needed for the JSON output.

```python
# --- File: metadump.py (simplified _generate_json_metadata) ---
# ... imports and MetadataExtractor class ...

    def _generate_json_metadata(self) -> None:
        """Generate and save JSON metadata for the dataset."""
        metadata = {
            "global_attributes": self._get_json_compatible_attrs(self.dataset.attrs), # Get dataset attributes
            "variables": {} # Dictionary to hold variable info
        }

        # Loop through each variable (data_vars) in the dataset
        for var_name, da in self.dataset.data_vars.items():
            # Check if the variable should be included based on --vars or --ignore
            if self._should_include_var(var_name):
                metadata["variables"][var_name] = {
                    "dimensions": list(da.dims), # Get dimensions as a list
                    "data_type": str(da.dtype),   # Get data type
                    "attributes": self._get_json_compatible_attrs(da.attrs) # Get variable attributes
                }

        # Use Python's json library to write the dictionary to a file
        with open(self.config.json_output or "ds_metadata.json", "w") as json_file:
            json.dump(metadata, json_file, indent=4)
        logger.debug(f"Saved metadata to {self.config.json_output or 'ds_metadata.json'}")

    def _get_json_compatible_attrs(self, attrs: Dict) -> Dict:
        """Convert attributes to JSON-compatible format (handles numpy types)."""
        # Uses the helper function json_compatible to convert numpy types
        return {k: json_compatible(v) for k, v in attrs.items()}

# --- File: metadump.py (simplified json_compatible utility) ---
# ... imports ...
# Utility function to handle non-standard types like numpy floats/ints/arrays
def json_compatible(value: Any) -> Any:
    if isinstance(value, (np.float32, np.float64)):
        return float(value) # Convert numpy float to standard Python float
    elif isinstance(value, (np.int32, np.int64, np.int16)):
        return int(value) # Convert numpy int to standard Python int
    # ... recursive calls for lists and dictionaries ...
    return value # Return other types as is
```

This code shows how `metadump` iterates through the `data_vars` of the `xarray.Dataset`, extracts dimensions and attributes, formats them, and builds the nested dictionary structure that gets written to JSON. The `json_compatible` helper is necessary because standard Python `json` might not know how to handle `numpy` specific data types directly.

### Generating YAML Configuration

Generating YAML involves two main steps: building the `specs` dictionary (`_generate_specs_dict`) and building the `app` dictionary (`_generate_app_dict`).

```python
# --- File: metadump.py (simplified _generate_specs_dict and _process_variable) ---
# ... imports and MetadataExtractor class ...

    def _generate_specs_dict(self) -> Dict:
        """Generate the specifications dictionary for YAML output."""
        specs_dict = {}
        plottable_vars = self.get_plottable_vars() # Get the list of variables to include

        for var_name in plottable_vars:
            var = self.dataset[var_name]
            # Delegate processing of each variable to a helper method
            specs_dict[var_name] = self._process_variable(var_name, var)

        return specs_dict

    def _process_variable(self, var_name: str, var: xr.DataArray) -> Dict:
        """Process a single variable and return its metadata dictionary."""
        temp_dict = {}

        # Add basic metadata (units, long_name)
        if 'units' in var.attrs: temp_dict['units'] = var.attrs['units']
        if 'long_name' in var.attrs: temp_dict['name'] = var.attrs['long_name']

        # Determine suggested plot types based on dimensions
        n_non_time_dims = len([dim for dim in var.dims if dim != self.tc]) # Count dims excluding time
        if self.zc and len(self.dataset.coords[self.zc]) == 1: # If vertical dim exists but has only 1 level
            n_non_time_dims -= 1 # Don't count it towards multi-dimensional plots

        # Add plot configurations based on counts and dimension names
        if has_multiple_time_levels(self.dataset, var_name, self.tc): # If more than one time step
            temp_dict['xtplot'] = {"time_lev": "all", "grid": "yes"} # Suggest XT plot

        if n_non_time_dims >= 2: # If at least 2 non-time, non-single-vertical dims
            default_lev = float(self.dataset.coords[self.zc][0].values) if self.zc else 0 # Suggest first level if applicable
            temp_dict['xyplot'] = dict(levels={default_lev: []}) # Suggest XY plot
            if self.tc and self.dataset[self.tc].ndim > 1: # If also has time
                temp_dict['xyplot']['time_lev'] = 1 # Suggest first time level

        if n_non_time_dims >= 3 and all("soil_layers" not in dim for dim in var.dims): # If 3+ non-time dims, not soil layers
            temp_dict['yzplot'] = dict(contours=[]) # Suggest YZ plot
            if self.tc and self.dataset[self.tc].ndim > 1: # If also has time
                temp_dict['yzplot']['time_lev'] = 1 # Suggest first time level

        return temp_dict

# --- File: metadump.py (simplified get_plottable_vars and is_plottable) ---
# ... imports ...
# Helper function to check if a variable looks plottable (enough spatial/vertical/time dimensions)
def is_plottable(ds: xr.Dataset, var: str,
                 space_coords: Set[str], zc: Optional[str],
                 tc: Optional[str]) -> bool:
    """Determine if a variable is plottable based on dimensions."""
    var_dims = set(ds[var].dims)
    # Checks if required dimension sets are present for 2D, 3D, or 4D fields
    if space_coords.issubset(var_dims) and len(var_dims) == 2: return True # 2D space (lon, lat)
    # ... similar checks for 3D space, 4D space-time, 2D space-time ...
    return False # Not plottable if no recognizable dimension combo

def get_plottable_vars(self) -> List[str]:
    """Get list of plottable variables based on configuration and dimension check."""
    # Filter variables based on --vars or --ignore first
    if self.config.vars: return self.config.vars
    # Then filter based on dimension check using the helper function
    plottable = [var for var in self.dataset.data_vars
                if is_plottable(self.dataset, var, self.space_coords,
                              self.zc, self.tc)]
    # Apply ignore list again just in case
    # ... ignore list logic ...
    return plottable

# --- File: eviz/lib/utils.py (simplified get_model_dim_name) ---
# ... imports ...
# Helper utility used by MetadataExtractor._setup_coordinates and _process_variable
def get_model_dim_name(dims: List[str], dim_name: str,
                      meta_coords: Dict, source: str = 'gridded') -> Optional[str]:
    """Get the model-specific dimension name (e.g., 'lat' for 'yc')."""
    # Looks up standard dim_name ('tc', 'xc', 'yc', 'zc') in the loaded meta_coords dictionary
    # Then finds the actual dimension name used in the dataset based on the 'source' hint
    # ... logic to check meta_coords[dim_name][source] against available dims ...
    pass # Simplified for clarity
```

The `_generate_specs_dict` method iterates through potentially plottable variables (determined by `get_plottable_vars`, which uses the `is_plottable` helper to check dimensions). For each variable, `_process_variable` looks at its dimensions and suggests relevant plot types (`xt`, `xy`, `yz`), adding basic attributes like units and name from the variable's attributes. It uses the standard dimension names (`self.tc`, `self.xc`, etc.) which were figured out in `__init__` using the `get_model_dim_name` utility function and the `meta_coords` file.

The `_generate_app_dict` method builds the main application configuration dictionary.

```python
# --- File: metadump.py (simplified _generate_app_dict) ---
# ... imports and MetadataExtractor class ...

    def _generate_app_dict(self) -> Dict:
        """Generate the application dictionary for YAML output."""
        app_dict = {
            "inputs": [{
                "name": self.config.filepath_1, # Add the first file path
                # Add the suggested plot types for each variable
                "to_plot": self._get_plot_types() 
            }],
            "outputs": {
                "print_to_file": "yes",
                "output_dir": None, # User needs to fill this in
                "print_format": "png",
                "print_basic_stats": True,
                "make_pdf": False
            },
            "system_opts": {
                "use_mp_pool": False, # Default to no multiprocessing
                "archive_web_results": True
            }
        }

        # If a second file was provided (--filepaths fileA fileB)
        if self.config.filepath_2:
            self._add_comparison_config(app_dict) # Add comparison specific config

        return app_dict

    def _get_plot_types(self) -> Dict[str, str]:
        """Get plot types for each plottable variable (used in _generate_app_dict)."""
        plot_types = {}
        for var_name in self.get_plottable_vars():
            types = []
            # Same logic as in _process_variable, but just returns the list of types as a comma-separated string
            if has_multiple_time_levels(self.dataset, var_name, self.tc):
                types.append("xt")
            if is_plottable(self.dataset, var_name, self.space_coords, self.zc, self.tc):
                types.append("xy")
                if self.zc and "soil_layers" not in var_name:
                    types.append("yz")
            plot_types[var_name] = ",".join(types) # e.g., "xt,xy,yz"
        return plot_types

    # --- File: metadump.py (simplified _add_comparison_config) ---
    # ... imports and MetadataExtractor class ...

    def _add_comparison_config(self, app_dict: Dict) -> None:
        """Add comparison configuration for two-file cases."""
        # Generate random IDs for experiment names
        exp_id_1 = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        exp_id_2 = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))

        # Add exp_id to the first input entry
        app_dict['inputs'][0]['exp_id'] = exp_id_1
        app_dict['inputs'][0]['exp_name'] = None

        # Add the second file as a new input entry
        app_dict['inputs'].append({
            "name": self.config.filepath_2,
            "to_plot": {}, # Don't list plots here, they reference the first entry
            "location": None,
            "exp_id": exp_id_2,
            "exp_name": None
        })

        # Add the for_inputs section to enable comparison mode
        app_dict['for_inputs'] = {
            "compare": {"ids": f"{exp_id_1}, {exp_id_2}"},
            "cmap": "coolwarm" # Suggest a comparison colormap
        }
```

The `_generate_app_dict` starts with a basic structure for the `inputs`, `outputs`, and `system_opts` sections. It calls `_get_plot_types` (which uses logic similar to `_process_variable`) to populate the `to_plot` list for the first input file. If a second file was provided, `_add_comparison_config` is called to add that file to the `inputs` list and include the `for_inputs` section needed for comparison plots ([Chapter 2: Configuration Management](02_configuration_management_.md)).

Finally, the `_write_specs_yaml` and `_write_app_yaml` methods simply use the `yaml.dump` function from the PyYAML library to convert the generated Python dictionaries into YAML format and write them to the specified output files.

## Why Use the Metadata Tool?

*   **Quick Inspection:** Easily see what's inside a file without writing code or using heavy software.
*   **Identify Plottable Variables:** Get a quick list of variables eViz is likely able to plot based on dimension analysis.
*   **Understand Metadata:** View dimension names, units, long names, etc., which are crucial for configuration and understanding the data.
*   **Generate Boilerplate Config:** Get a head start on writing your eViz configuration files, reducing manual effort and potential errors. This is especially useful for comparison plots where the `inputs` and `for_inputs` sections need careful setup.
*   **Support for New Data:** If you're adding support for a new data format or model (requiring new [Data Source Abstraction](03_data_source_abstraction_.md) or [Source/Model Specific Logic](08_source_model_specific_logic_.md)), `metadump.py --json` is invaluable for understanding the raw structure of the files you need to read.

In essence, `metadump.py` is a developer and user-friendly utility that complements the core eViz visualization engine by making it easier to understand the input data and generate the necessary configuration.

## Summary

In this final conceptual chapter, we explored the **Metadata Tool (metadump)** (`metadump.py`), a separate command-line utility within the eViz project.

*   `metadump.py` is designed to inspect data files (like NetCDF) and extract metadata.
*   You run it directly from the command line, providing one or two file paths.
*   By default, it lists potentially plottable variables.
*   Using the `--json` flag, you can get detailed metadata (dimensions, attributes, etc.) in JSON format.
*   Using the `--app` and `--specs` flags, you can automatically generate boilerplate YAML configuration files (`_app.yaml` and `_specs.yaml`) based on the file's contents, including suggested plot types and comparison setup if two files are provided.
*   It uses `xarray` to open and read the data files and `PyYAML` to generate YAML output.
*   The `autoviz.py` script provides a convenient wrapper (`autoviz.py --file ...`) for running `metadump.py`.
*   The tool is invaluable for understanding your data files and quickly generating the starting point for your visualization configuration, making the process of using the core eViz application much smoother.

This concludes our conceptual overview of the main components of the `eviz-dev` project. We've covered how the application runs, how configuration drives the process, how data is read and processed abstractly, how plotters are managed, how source-specific logic handles data types, and now, how a utility tool helps you understand your data to get started. Armed with this knowledge, you should be well-equipped to understand the structure of eViz and how its different parts work together to turn your scientific data into visualizations!

Thank you for joining this tutorial journey!

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)