# Chapter 2: Config Manager

Welcome back to the eViz tutorial! In the previous chapter, [Chapter 1: Data Source](01_data_source_.md), we learned how eViz acts like a universal translator to read your scientific data files and prepare them in a standard format (`xarray.Dataset`). Now that eViz has the ingredients (your data), it needs the recipe: instructions on *what* to do with that data.

How does eViz know which variables you want to plot, what kind of plot you need for each variable, where to save the results, and how to compare different datasets? This is where the **Config Manager** comes in.

## What Problem Does Config Manager Solve?

Imagine you have a complex dataset with dozens of variables, and you want to create many different plots: a map for temperature, a vertical profile for ozone, a time series for pressure, and maybe compare the temperature map from one file to another.

If you had to write code to specify every single detail (filename, variable name, plot type, title, color scale, output file path, comparison settings) for every plot, your scripts would become incredibly long and hard to manage.

The **Config Manager** solves this by centralizing all your settings and instructions. You provide eViz with one or more configuration files (usually written in a human-readable format like YAML), and the Config Manager reads everything from these files and holds it in one place. It acts like the application's "brain," making sure all the different parts of eViz can easily access the settings they need.

Think of the Config Manager as the main control panel where you dial in all your preferences for the visualization process.

## Our Central Use Case: Loading Settings

The primary way you interact with the configuration is by telling eViz which configuration files to load. eViz will then create a `ConfigManager` object that contains all the settings read from these files.

Let's imagine you have a configuration file named `my_settings.yaml` that specifies your input files, output directory, and which variables to plot.

## Creating the Config Manager

You don't usually create the `ConfigManager` directly. Instead, eViz handles this internally based on the configuration files you provide when you start the application. However, understanding how it's structured helps.

The `ConfigManager` relies on a core `Config` object, which itself uses a [YAML Parser](03_yaml_parser_.md) (we'll cover that in the next chapter) to read the files.

Here's a simplified look at how you might get a `ConfigManager` instance within eViz:

```python
# Imagine this is part of eViz's startup process
from eviz.lib.config.config import Config
from eviz.lib.config.config_manager import ConfigManager
# from eviz.lib.config.app_data import AppData # Used internally

# Assume 'source_names' and 'config_files' come from command line or defaults
source_names_list = ["my_data_source"]
config_file_paths = ["path/to/your/my_settings.yaml"]

print(f"Loading settings from: {config_file_paths}")

# 1. The core Config object reads the files
# (It internally uses the YAML Parser)
base_config = Config(
    source_names=source_names_list,
    config_files=config_file_paths
)
print(f"Base config object created: {type(base_config)}")

# 2. The ConfigManager wraps the base config and adds more logic
config_manager = ConfigManager(
    input_config=base_config.input_config,
    output_config=base_config.output_config,
    system_config=base_config.system_config,
    history_config=base_config.history_config,
    config=base_config # Pass the base config object
)
print(f"Config Manager object created: {type(config_manager)}")

print("Configuration loaded successfully!")
```

In this snippet:

*   We import the necessary `Config` and `ConfigManager` classes.
*   We create a `Config` object, passing it the list of source names (often related to experiments or models) and the paths to the configuration files. The `Config` object does the heavy lifting of reading and parsing the files.
*   We then create the `ConfigManager`, passing it instances of the sub-configurations (`input_config`, `output_config`, etc.) and the main `config` object itself. The `ConfigManager` instance `config_manager` is what other parts of the application will use.

## Accessing Settings

Once you have the `config_manager` object, accessing any setting is easy. The `ConfigManager` provides direct access to common settings and also allows you to "reach through" to the settings held by its sub-configurations.

Let's say your `my_settings.yaml` file includes settings like this:

```yaml
# Simplified my_settings.yaml
outputs:
  output_dir: "./my_visualization_output"
  print_format: "png"
  make_pdf: True

inputs:
  # ... your file list here ...
  for_inputs:
    compare: True
    compare_exp_ids: ["experiment_A", "experiment_B"]
    # ... other input settings ...

# ... other sections like specs, system_opts, etc. ...
```

You can access these settings directly from the `config_manager` object:

```python
# Continuing from the previous example with the config_manager object

print(f"Output directory is: {config_manager.output_dir}")
print(f"Print format is: {config_manager.print_format}")
print(f"Should we make a PDF? {config_manager.make_pdf}")
print(f"Is comparison mode active? {config_manager.compare}")
print(f"Experiment IDs to compare: {config_manager.compare_exp_ids}")
```

Notice how you can access `output_dir`, `print_format`, and `make_pdf` directly from `config_manager`, even though these settings are actually managed by the `OutputConfig` sub-object internally. Similarly, `compare` and `compare_exp_ids` come from the `InputConfig`. The `ConfigManager` hides this internal structure and provides a convenient single interface.

This is a key benefit: any part of eViz that needs a setting just asks the `config_manager`, regardless of which specific configuration sub-object actually holds that setting.

## Breaking Down Complexity: Sub-Configurations

The `ConfigManager` doesn't hold *all* the settings data directly. It delegates responsibility to specialized sub-configuration objects. Looking at the code snippets (like `eviz/lib/config/config_manager.py`), you'll see attributes like:

*   `input_config`: An instance of `InputConfig` (`eviz/lib/config/input_config.py`). This handles everything related to the input data: the list of files, how to read them, settings for comparison or overlaying datasets.
*   `output_config`: An instance of `OutputConfig` (`eviz/lib/config/output_config.py`). This manages settings about the output visualizations: where to save them, file format (PNG, PDF, GIF), whether to add logos, etc.
*   `system_config`: An instance of `SystemConfig` (`eviz/lib/config/system_config.py`). This covers application-wide system settings, like using multiprocessing pools or archiving results.
*   `history_config`: An instance of `HistoryConfig` (`eviz/lib/config/history_config.py`). This handles settings related to tracking configuration history.
*   `config`: An instance of the base `Config` class (`eviz/lib/config/config.py`). This object holds the raw parsed data (`app_data`, `spec_data`, etc.) and initializes the sub-configurations mentioned above.

Why do this? It follows the principle of "separation of concerns." Each type of configuration (inputs, outputs, system) has its own class responsible for managing those specific settings. This makes the code easier to understand, maintain, and extend.

The `ConfigManager` then acts as a **facade** – it provides a simplified, unified interface to all these underlying configuration objects.

## Under the Hood: How It Works

Let's lift the hood and see how the `ConfigManager` ties everything together.

1.  **Loading and Parsing:** The process starts when the main application provides the list of config files to the core `Config` object. The `Config` object uses the [YAML Parser](03_yaml_parser_.md) to read the raw data from these files into structured Python dictionaries.
2.  **Structuring Raw Data:** The `Config` object then populates an `AppData` dataclass (`eviz/lib/config/app_data.py`) with the main sections from the YAML (like `inputs`, `outputs`, `system_opts`, `plot_params`). It also holds other parsed data like `spec_data` (details about variables and plots) and `map_params`.
3.  **Initializing Sub-Configurations:** Inside its `__post_init__` method (which runs automatically after the object is created), the `Config` object creates instances of `InputConfig`, `OutputConfig`, `SystemConfig`, and `HistoryConfig`. It passes the `AppData` object to each of them. Then, it calls the `initialize()` method on each sub-config. This allows each sub-config to process the raw `AppData` and set up its specific attributes (like parsing the file list in `InputConfig` or setting the output directory in `OutputConfig`).
    ```python
    # Simplified snippet from eviz/lib/config/config.py's __post_init__
    def __post_init__(self):
        # 1. Parse YAML (next chapter!)
        self.yaml_parser = YAMLParser(...)
        self.yaml_parser.parse()
        # 2. Store raw data in AppData and other attributes
        self.app_data = AppData(**self.yaml_parser.app_data)
        self.spec_data = self.yaml_parser.spec_data
        # ... other data from parser ...

        # 3. Initialize sub-configurations
        self.input_config = InputConfig(self.source_names, self.config_files)
        self.output_config = OutputConfig()
        self.system_config = SystemConfig()
        self.history_config = HistoryConfig()

        # 4. Assign app_data to sub-configs
        self._assign_app_data_to_subconfigs()

        # 5. Call initialize on all configs
        self.initialize() # This calls initialize() on all sub-configs too
    ```
4.  **ConfigManager Wraps Up:** The `ConfigManager` is then created, receiving these initialized sub-configuration objects and the base `Config` object. Its `__post_init__` method does final setup, like calling `setup_comparison()`.
    ```python
    # Simplified snippet from eviz/lib/config/config_manager.py's __post_init__
    def __post_init__(self):
        """Initialize the ConfigManager after construction."""
        # Pass itself to input_config (if needed for factory)
        self.input_config.config_manager = self # CC: Is this necessary? Yes, for DataSourceFactory!
        # Perform application-specific setup based on config
        self.setup_comparison()
        # ... other ConfigManager specific setup ...
    ```
5.  **Unified Access (`__getattr__`):** The magic that allows you to access properties like `config_manager.output_dir` or `config_manager.compare` directly is Python's `__getattr__` method. When you try to access an attribute (like `output_dir`) on the `ConfigManager` instance, Python first checks if the `ConfigManager` itself has that attribute. If not, it calls `__getattr__`. The `ConfigManager`'s `__getattr__` (see `eviz/lib/config/config_manager.py`) checks if the attribute exists on its `config` object or any of its sub-config objects (`input_config`, `output_config`, etc.). If it finds it, it returns that value; otherwise, it raises an `AttributeError`. This creates the seamless facade.
    ```python
    # Simplified snippet from eviz/lib/config/config_manager.py's __getattr__
    def __getattr__(self, name):
        # 1. Check the base config object first
        if hasattr(self.config, name):
            return getattr(self.config, name)

        # 2. Check in other sub-config objects
        for config in [self.input_config, self.output_config, self.system_config,
                       self.history_config]:
            if hasattr(config, name):
                return getattr(config, name)

        # 3. If not found anywhere, raise error
        raise AttributeError(...)
    ```

Here's a simplified sequence diagram showing how a request for `config_manager.output_dir` might flow:

```{mermaid}
sequenceDiagram
    participant App as eViz Application Code
    participant CM as ConfigManager
    participant Config as Config
    participant OutputConfig as OutputConfig

    App->>CM: Access output_dir (config_manager.output_dir)
    CM->>CM: Does CM have 'output_dir'? No -> calls __getattr__('output_dir')
    CM->>Config: Does Config have 'output_dir'? No
    CM->>OutputConfig: Does OutputConfig have 'output_dir'? Yes
    OutputConfig-->>CM: Returns self.output_dir value
    CM-->>App: Returns the value
```

This shows how the `ConfigManager` routes the request to the correct place (`OutputConfig`) without the application code needing to know the specific structure.

## Application-Specific Logic

Beyond just holding settings, the `ConfigManager` also contains logic that depends on multiple settings or relates directly to eViz's core processes. Examples from the code include:

*   `setup_comparison()`: This method in `ConfigManager` uses the `compare_exp_ids` or `overlay_exp_ids` from the `InputConfig` (via `self.input_config._compare_exp_ids`) to build internal lists (`a_list`, `b_list`) of file indices that correspond to the experiments being compared. This pre-computation based on config makes later processing easier.
*   `get_model_dim_name(dim_name)`: This is a crucial method in `ConfigManager`. It takes a generic dimension name (like `'yc'` for latitude or `'zc'` for vertical level) and looks up the *actual* dimension name used in the currently loaded data source (e.g., 'latitude', 'lev'). It uses `meta_coords` which were loaded by the `Config` object from a metadata file. This allows eViz to work with different model outputs that might use different names for the same concept.

```python
# Simplified snippet from eviz/lib/config/config_manager.py's get_model_dim_name
def get_model_dim_name(self, dim_name):
    """
    Get model-specific dimension name associated with the source...
    """
    # Get the name of the current source (from input_config)
    source = self.source_names[self.ds_index] # ds_index is managed by CM

    # Look up mapping in meta_coords (loaded by Config)
    if dim_name in self.meta_coords and source in self.meta_coords[dim_name]:
        model_name = self.meta_coords[dim_name][source]
        # Need to check if this name actually exists in the loaded data!
        # (This part is more complex in the real code, involving the pipeline)
        # ... simplified check ...
        # if model_name exists in current dataset dimensions:
        #    return model_name
        # else:
        #    logger.debug(...)
        return model_name # Simplified
    else:
        self.logger.debug(f"No mapping for {dim_name} in source {source}")
        # Fallback logic (checking common names, other sources) happens here...
        return None # Or the inferred common name/fallback

```

This demonstrates how the `ConfigManager` not only holds data but also provides methods that use that data to perform essential tasks needed by the application's workflow.

## Summary

In this chapter, we explored the **Config Manager**, the central hub for all settings and instructions in eViz.

*   It solves the problem of managing complex visualization settings by loading them from configuration files (typically YAML).
*   It provides a single, easy-to-use interface (`config_manager`) to access all settings.
*   It delegates specific types of settings to specialized sub-configurations (`InputConfig`, `OutputConfig`, etc.) for organization.
*   Under the hood, it uses the core `Config` object (which relies on a [YAML Parser](03_yaml_parser_.md)) to read files and initializes the sub-configs.
*   Python's `__getattr__` method helps the `ConfigManager` act as a facade, allowing direct access to settings held by its sub-objects.
*   The `ConfigManager` also includes application-specific logic, such as setting up comparison lists and mapping generic dimension names to model-specific ones.

The Config Manager ensures that all parts of eViz can access the user's desired settings consistently throughout the visualization process.

Now that we know that settings come from YAML files and are managed by the Config Manager, the next logical step is to understand how eViz reads and interprets the structure of those YAML files. Let's move on to [Chapter 3: YAML Parser](03_yaml_parser_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)