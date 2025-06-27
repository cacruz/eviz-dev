# Chapter 2: Configuration Management

Welcome back! In [Chapter 1: Autoviz Application](01_autoviz_application_.md), we met the **Autoviz Application**, which acts like the director of our visualization orchestra. It's the main program you run, and it's responsible for bringing all the other eViz components together.

But how does the director know *what* music to play? How does it know which data files to use, which variables are important, what kind of plots to create, and where to save the results? This is where **Configuration Management** comes in.

Think of **Configuration Management** as the detailed **script** or **plan** that the Autoviz director reads and understands. It's the central control center for all the settings your visualization task needs.

## Why Do We Need Configuration Management?

Imagine you have a complex scientific dataset with many variables, multiple files covering different periods, and specific requirements for how you want the plots to look (colors, titles, maps, etc.).

If you had to write Python code every time you wanted to visualize a different variable or use a different file, it would be a lot of work! Configuration management solves this by allowing you to define all these details in easy-to-read files, separate from the main code.

This separation makes eViz flexible:

1.  **Easy to change settings:** Want to plot a different variable? Just change a line in the configuration file.
2.  **Repeatable tasks:** You can easily run the exact same visualization process on new data files by reusing the same configuration.
3.  **Shareable plans:** You can share your configuration files with colleagues so they can reproduce your visualizations.

In eViz, these configuration files are typically written in the **YAML** format. YAML is a human-friendly data format that's easy to read and write.

## Our Use Case: Visualizing Data Using a Config File

In Chapter 1, we ran `python autoviz.py -s gridded` to tell eViz to visualize the 'gridded' source using its *default* settings.

Now, let's make it more specific. We want to tell eViz: "Visualize data for the 'gridded' source, but use *these specific settings* defined in a configuration file."

Let's say we have a simple YAML file named `my_gridded_config.yaml`.

```yaml
# --- File: my_gridded_config.yaml ---

# Section for defining input data files
inputs:
  - name: sample_gridded_data.nc  # The data file name
    location: /path/to/your/data  # Where the file is located
    exp_id: baseline             # An identifier for this data (useful for comparisons)
    description: My baseline simulation
    to_plot:                     # Which variables from this file you want to plot
      temperature: xy            # Plot 'temperature' variable using 'xy' plot type
      pressure: xy               # Plot 'pressure' variable using 'xy' plot type

# Section for defining output options
outputs:
  output_dir: ./my_gridded_plots # Where to save the plots
  print_to_file: True            # Actually save the plots to files
  print_format: png              # Save as PNG images

# ... other sections for system settings, etc. ...
```

This `my_gridded_config.yaml` file is the "plan". It tells eViz:
*   Look for a file named `sample_gridded_data.nc` in `/path/to/your/data`.
*   Identify this data as `baseline`.
*   For this data, generate plots for the `temperature` and `pressure` variables, using the `xy` plot type for each.
*   Save the generated plots as PNG files in the `./my_gridded_plots` directory.

To run eViz with this specific configuration file, you use the `-c` or `--config` command-line argument:

```bash
python autoviz.py -s gridded -c my_gridded_config.yaml
```

This command tells the `autoviz.py` script to run the `Autoviz` application, specifically targeting the 'gridded' source (`-s gridded`), and importantly, to load its settings from `my_gridded_config.yaml` (`-c my_gridded_config.yaml`).

When you run this, the Configuration Management system in eViz takes over to read and process `my_gridded_config.yaml`.

## How Configuration Management Works (High-Level)

When the `Autoviz` object (the director) is created, one of the very first things it does is set up its Configuration Management system. This is like the director receiving and reading the script.

Here's a simplified look at the flow when you include a config file:

```{mermaid}
sequenceDiagram
    participant User
    participant AutovizScript as autoviz.py
    participant AutovizApp as Autoviz
    participant ConfigMgr as ConfigManager
    participant YAMLFiles as YAML Config Files

    User->>AutovizScript: Run 'python autoviz.py -s gridded -c my_config.yaml'
    AutovizScript->>AutovizScript: parse_command_line() (gets -s and -c)
    AutovizScript->>AutovizApp: Create Autoviz(['gridded'], args)
    AutovizApp->>AutovizApp: __post_init__()
    AutovizApp->>ConfigMgr: create_config(args) # This starts config loading
    ConfigMgr->>ConfigMgr: Initializes internal components (like YAMLParser)
    ConfigMgr->>YAMLFiles: Read YAML config file(s)
    YAMLFiles-->>ConfigMgr: Return configuration data
    ConfigMgr-->>AutovizApp: Return initialized ConfigManager object
    AutovizApp-->>AutovizScript: Autoviz object created with config
    AutovizScript->>AutovizApp: run() # Now AutovizApp knows the plan!
    AutovizApp->>ConfigMgr: Ask ConfigManager for details (e.g., file list, variables)
    ConfigMgr-->>AutovizApp: Provide requested configuration data
    AutovizApp->>AutovizApp: Continue visualization based on config
    AutovizApp-->>AutovizScript: Run method finishes
    AutovizScript-->>User: Print time taken & Exit
```

The key takeaway is that the `create_config(args)` step in `Autoviz.__post_init__` is where the Configuration Management system is built. It reads your specified config files (from `args`), processes them, and provides a structured object (`ConfigManager`) that the rest of the `Autoviz` application can query for information.

## Inside the Code: The Configuration Stack

eViz uses a few classes to handle configuration, working together like a stack:

1.  **`YAMLParser`**: This is the lowest level. Its job is just to read the raw YAML files you provide, handle things like merging multiple files, expanding environment variables (`!path`), and loading basic metadata files (`meta_coords.yaml`, `meta_attrs.yaml`, etc.). It structures the raw data but doesn't add much application logic.
2.  **`Config`**: This class sits on top of `YAMLParser`. It takes the raw data parsed by `YAMLParser` and organizes it into specialized "sub-configuration" objects: `InputConfig`, `OutputConfig`, `SystemConfig`, and `HistoryConfig`. It acts as a central container for these different aspects of the configuration.
3.  **`ConfigManager`**: This is the top-level class that the rest of the eViz application interacts with. It wraps the `Config` object and adds application-specific logic, such as setting up comparison lists (`a_list`, `b_list`), managing the current file/data source being processed (`findex`, `ds_index`), and providing convenient methods to access nested configuration details (like `get_model_dim_name`).

Here's how these pieces fit together conceptually:

```{mermaid}
graph TD
    UserCmd("User Command (-c my_config.yaml)") --> Autoviz("Autoviz Application (Director)")
    Autoviz --> ConfigMgr("ConfigManager (Control Center)")
    ConfigMgr --> Config("Config (Settings Container)")
    Config --> InputConfig("InputConfig (Data Files, Readers)")
    Config --> OutputConfig("OutputConfig (Saving Plots)")
    Config --> SystemConfig("SystemConfig (MP, Archive)")
    Config --> HistoryConfig("HistoryConfig (History Tracking)")
    Config --> YAMLParser("YAMLParser (Raw File Reader)")
    YAMLParser --> YAMLFiles("YAML Config Files")
```

Let's look at some simplified code snippets to see this in action.

First, where `ConfigManager` is created in `Autoviz.__post_init__` (from `eviz/lib/autoviz/base.py`):

```python
# --- File: eviz/lib/autoviz/base.py (simplified) ---
from eviz.lib.config.config import Config
from eviz.lib.config.config_manager import ConfigManager
# ... other imports ...

@dataclass
class Autoviz:
    # ... attributes ...

    def __post_init__(self):
        # ... logging setup ...

        # Create the core Config object
        # This object uses YAMLParser internally to read files
        config = Config(source_names=self.source_names, config_files=self.args.config) # Uses YAMLParser inside

        # Create the ConfigManager, passing the core Config object
        # The ConfigManager will wrap Config and add eViz-specific logic
        self._config_manager = ConfigManager(
            input_config=config.input_config,
            output_config=config.output_config,
            system_config=config.system_config,
            history_config=config.history_config,
            config=config # Pass the core Config object
        )

        # Now initialize the ConfigManager. This triggers initialization
        # of sub-configs and setting up things like comparison lists.
        self._config_manager.initialize() # Calls initialize() on sub-configs
        self.logger.info("ConfigManager initialized.")

        # ... other initialization steps ...
```

Here, `Autoviz` creates the base `Config` object, which handles parsing the YAML files (using `YAMLParser` internally). Then, it creates the `ConfigManager`, passing the initialized sub-configs and the core `Config` object to it. Finally, it calls `_config_manager.initialize()` to finish setting up things specific to the application's workflow.

Next, let's peek at `YAMLParser` (`eviz/lib/config/yaml_parser.py`). Its main job is `_concatenate_yaml`, which reads and merges the YAML files:

```python
# --- File: eviz/lib/config/yaml_parser.py (simplified) ---
import os
from typing import List, Dict, Any
import eviz.lib.utils as u # Utility functions including load_yaml_simple
# ... other imports ...

@dataclass
class YAMLParser:
    config_files: List[str]
    source_names: List[str]
    app_data: Dict[str, Any] = field(default_factory=dict)
    spec_data: Dict[str, Any] = field(default_factory=dict)
    _map_params: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # ... other attributes ...

    def parse(self):
        """Parse YAML files and populate app_data and spec_data."""
        concat = self._concatenate_yaml()
        self._init_map_params(concat) # Processes inputs/outputs into _map_params
        # Load metadata files
        self.meta_coords = u.read_meta_coords()
        self.meta_attrs = u.read_meta_attrs()
        self.species_db = u.read_species_db()

    def _concatenate_yaml(self) -> List[Dict[str, Any]]:
        """Read and merge multiple YAML files and their associated specs."""
        result = {} # This will hold the merged app_data
        # ... other initializations ...

        for index, file_path in enumerate(self.config_files):
            # Load the main config file
            yaml_content = u.load_yaml_simple(file_path)
            yaml_content['source'] = self.source_names[index] # Add source name

            # Merge sections from this file into the main result dict
            if 'inputs' in yaml_content:
                result.setdefault('inputs', []).extend(yaml_content['inputs'])
            # ... merge other sections like for_inputs, system_opts, outputs ...

            # Load associated specs file (e.g., my_gridded_config_specs.yaml)
            specs_file = os.path.join(os.path.dirname(file_path),
                                      f"{os.path.splitext(os.path.basename(file_path))[0]}_specs.yaml")
            if os.path.exists(specs_file):
                specs_content = u.load_yaml_simple(specs_file)
                self.spec_data.update(specs_content) # Merge into spec_data
            # ... handle missing specs file ...

        self.app_data = result # Store the merged application data
        # ... return value ...
```

The `YAMLParser` reads your main config files (`my_gridded_config.yaml`), potentially merges data if multiple files are specified, and also looks for and loads associated "specs" files (like `my_gridded_config_specs.yaml`) which contain detailed settings for specific variables or plot types. It stores the combined application settings in `app_data` and the variable/plot specifications in `spec_data`. It also loads global metadata like `meta_coords` and `meta_attrs` using utility functions.

The `_init_map_params` method within `YAMLParser` is important because it takes the raw `inputs` and `outputs` data and creates the `_map_params` structure. This structure is a list of dictionaries, where each dictionary represents *one specific plot task*. For example, plotting 'temperature' from `sample_gridded_data.nc` using the 'xy' plot type might be one entry in `_map_params`, and plotting 'pressure' from the same file might be another entry. This `_map_params` is crucial for iterating through all the necessary plots later.

Next, the `Config` class (`eviz/lib/config/config.py`). It's mostly a container and initializer:

```python
# --- File: eviz/lib/config/config.py (simplified) ---
from dataclasses import dataclass, field
from typing import List, Dict, Any
from eviz.lib.config.input_config import InputConfig
from eviz.lib.config.output_config import OutputConfig
from eviz.lib.config.system_config import SystemConfig
from eviz.lib.config.history_config import HistoryConfig
from eviz.lib.config.yaml_parser import YAMLParser
from eviz.lib.config.app_data import AppData # Simple data class for app_data structure
# ... other imports ...

@dataclass
class Config:
    # Input parameters
    source_names: List[str]
    config_files: List[str]

    # Data populated by YAMLParser
    app_data: AppData = field(default_factory=AppData) # Holds merged data from YAML sections
    spec_data: Dict[str, Any] = field(default_factory=dict) # Holds data from specs files
    map_params: Dict[int, Dict[str, Any]] = field(default_factory=dict) # Holds processed plot tasks
    meta_coords: dict = field(default_factory=dict)
    meta_attrs: dict = field(default_factory=dict)
    species_db: dict = field(default_factory=dict)
    _specs_yaml_exists: bool = True # Flag from parser

    # Sub-configuration objects (delegation)
    input_config: InputConfig = field(default_factory=InputConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)
    system_config: SystemConfig = field(default_factory=SystemConfig)
    history_config: HistoryConfig = field(default_factory=HistoryConfig)

    def __post_init__(self):
        """Parse YAML, assign data, and initialize sub-configs."""
        # 1. Use YAMLParser to load and structure data
        self.yaml_parser = YAMLParser(config_files=self.config_files, source_names=self.source_names)
        self.yaml_parser.parse()

        # 2. Transfer parsed data from parser to Config attributes
        self.app_data = AppData(**self.yaml_parser.app_data) # Convert dict to AppData object
        self.spec_data = self.yaml_parser.spec_data
        self.map_params = self.yaml_parser.map_params
        self.meta_coords = self.yaml_parser.meta_coords
        self.meta_attrs = self.yaml_parser.meta_attrs
        self.species_db = self.yaml_parser.species_db
        self._specs_yaml_exists = self.yaml_parser._specs_yaml_exists
        # ... transfer other data if needed ...

        # 3. Assign the parsed app_data to the sub-configuration objects
        self._assign_app_data_to_subconfigs()

        # 4. Initialize the sub-configuration objects
        self.initialize() # Calls initialize() on InputConfig, OutputConfig, etc.

    def _assign_app_data_to_subconfigs(self):
        """Assign app_data to all sub-configurations so they can initialize."""
        self.input_config.app_data = self.app_data
        self.output_config.app_data = self.app_data
        self.system_config.app_data = self.app_data
        self.history_config.app_data = self.app_data

    def initialize(self):
        """Initialize all configurations."""
        self.input_config.initialize()
        self.output_config.initialize()
        self.system_config.initialize()
        self.history_config.initialize()

    # ... other methods, properties ...
```

`Config`'s main role is in `__post_init__`. It delegates the raw file reading to `YAMLParser`. Once the parser finishes, `Config` takes the structured data (`app_data`, `spec_data`, `map_params`, etc.) and stores it in its own attributes. Crucially, it then passes the `app_data` to the various sub-configuration objects (`InputConfig`, `OutputConfig`, etc.) and calls their `initialize()` methods.

The sub-configuration objects like `InputConfig` (`eviz/lib/config/input_config.py`) then process *their specific part* of the `app_data`:

```python
# --- File: eviz/lib/config/input_config.py (simplified) ---
import sys
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from eviz.lib.utils import join_file_path
from eviz.lib.config.app_data import AppData
# ... other imports ...

@dataclass
class InputConfig:
    source_names: List[str]
    config_files: List[str]
    app_data: AppData = field(default_factory=AppData) # Receives AppData from Config
    file_list: Dict[int, Dict[str, Any]] = field(default_factory=dict) # Processed list
    # ... other attributes ...

    # Attributes derived from app_data (internal)
    _overlay: bool = field(default=False, init=False)
    _compare: bool = field(default=False, init=False)
    _compare_diff: bool = field(default=False, init=False)
    _compare_exp_ids: List[str] = field(default_factory=list, init=False)
    # ... many other settings derived from app_data['for_inputs'] ...

    def initialize(self):
        """Initialize input configuration from app_data."""
        # Set flags based on self.app_data.for_inputs
        self._overlay = self.app_data.for_inputs.get('overlay', False)
        self._compare = self.app_data.for_inputs.get('compare', False)
        self._compare_diff = self.app_data.for_inputs.get('compare_diff', False)

        # Parse comparison IDs
        self._parse_for_inputs(self.app_data.for_inputs) # This populates _compare_exp_ids etc.

        # Process the raw 'inputs' list into a more structured dictionary (file_list)
        self._get_file_list() # Populates self.file_list

        # Determine reader types based on file extensions/formats in self.file_list
        self._init_readers() # Sets up which data reader objects are needed

        # ... initialize other parameters from self.app_data.for_inputs ...

        self.logger.debug(f"Initialized InputConfig with: compare={self._compare}, ...")

    def _get_file_list(self):
        """ Process the raw 'inputs' list from app_data into a structured file_list."""
        if not self.app_data.inputs:
            self.logger.error("The 'inputs' section... is empty.")
            sys.exit()

        self.logger.debug(f"Processing {len(self.app_data.inputs)} input file entries.")
        for i, entry in enumerate(self.app_data.inputs):
            filename = join_file_path(entry.get('location', ''), entry['name'])
            self.file_list[i] = entry # Copy the original entry
            self.file_list[i]['filename'] = filename # Add the full filename

            # Store format if provided
            if 'format' in entry:
                 self._file_format_mapping[filename] = entry['format']

            self.logger.debug(f"file_list[{i}] = {self.file_list[i]}")

    # ... other methods like _init_readers, _parse_for_inputs, get_reader_for_file ...
```

`InputConfig.initialize()` specifically looks at the `self.app_data.inputs` and `self.app_data.for_inputs` sections. It populates its own attributes (`_compare`, `_overlay`, `_compare_exp_ids`, etc.) based on `for_inputs` settings and creates the `file_list` dictionary by processing the raw `inputs` list, adding the full filename. It also uses the parsed file list to determine which types of data readers are needed ([Data Source Abstraction](03_data_source_abstraction_.md)). `OutputConfig`, `SystemConfig`, and `HistoryConfig` do similar processing for their respective sections in `app_data`.

Finally, the `ConfigManager` (`eviz/lib/config/config_manager.py`) wraps all this. It holds references to the base `Config` object and all the sub-configuration objects. It provides convenient ways for other parts of the application to access *any* configuration setting. It also adds logic specific to eViz's workflow, like setting up the `a_list` and `b_list` for comparisons based on the `exp_id`s and `_compare_exp_ids` parsed by `InputConfig`.

```python
# --- File: eviz/lib/config/config_manager.py (simplified) ---
from dataclasses import dataclass, field
import logging
from typing import Optional, List, Dict, Any
import eviz.lib.utils as u
from eviz.lib.config.config import Config
from eviz.lib.config.input_config import InputConfig
from eviz.lib.config.output_config import OutputConfig
from eviz.lib.config.system_config import SystemConfig
from eviz.lib.config.history_config import HistoryConfig
# ... other imports ...

@dataclass
class ConfigManager:
    # References to sub-configs and the main config object
    input_config: InputConfig
    output_config: OutputConfig
    system_config: SystemConfig
    history_config: HistoryConfig
    config: Config

    # Application state/helpers managed by ConfigManager
    a_list: List[int] = field(default_factory=list) # Indices for comparison group A
    b_list: List[int] = field(default_factory=list) # Indices for comparison group B
    _findex: int = 0 # Current file index
    _ds_index: int = 0 # Current data source index
    # ... other attributes ...

    def __post_init__(self):
        """Initialize the ConfigManager after construction."""
        self.logger.info("Starting ConfigManager initialization")
        # No need to call initialize() on sub-configs here, Config.__post_init__ does it.
        # Now set up comparison lists based on initialized input_config
        self.setup_comparison()

    # Properties that allow easy access to attributes from *any* underlying config object
    # This is often done using a special __getattr__ method or explicit properties
    @property
    def app_data(self):
        """Access to application data from the core config."""
        return self.config.app_data

    @property
    def file_list(self):
        """Access to processed file list from InputConfig."""
        return self.input_config.file_list

    @property
    def compare(self):
        """Flag indicating if comparison mode is active from InputConfig."""
        return self.input_config._compare # Note: accessing the internal attribute

    @property
    def output_dir(self):
        """Access to output directory from OutputConfig."""
        return self.output_config.output_dir

    # ... many other properties for easy access ...

    # Application-specific methods
    def setup_comparison(self):
        """
        Set up comparison between datasets based on config settings.
        Creates a_list and b_list using exp_id and compare_exp_ids from input_config.
        """
        self.a_list = []
        self.b_list = []

        # Check if comparison is enabled using the flags from input_config
        if not (self.input_config._compare or self.input_config._compare_diff or self.input_config._overlay):
            self.logger.debug("Comparison not enabled")
            return

        # Get the list of experiment IDs to compare/overlay
        compare_ids = self.input_config._compare_exp_ids or self.input_config._overlay_exp_ids or []
        if not compare_ids:
            return

        # Map exp_ids from config to their corresponding index in the file_list
        exp_id_indices = {}
        for i, entry in enumerate(self.app_data.inputs): # Use raw inputs for this mapping
            if 'exp_id' in entry:
                exp_id_indices[entry['exp_id']] = i

        # Populate a_list and b_list based on the order in compare_ids
        for i, exp_id in enumerate(compare_ids):
            exp_id = exp_id.strip()
            if exp_id in exp_id_indices:
                if i == 0: # First ID is the 'base' (goes to a_list)
                    self.a_list.append(exp_id_indices[exp_id])
                else: # Subsequent IDs are for comparison (go to b_list)
                    self.b_list.append(exp_id_indices[exp_id])
            else:
                self.logger.warning(f"Could not find entry for exp_id: {exp_id}")

    def get_model_dim_name(self, dim_name):
        """
        Get model-specific dimension name (e.g., 'lat' for 'yc')
        using meta_coords loaded by the parser/config.
        """
        # This method uses self.meta_coords (which comes from self.config.meta_coords)
        # and logic to figure out the actual dimension name in the dataset.
        # ... complex logic involves self.ds_index, self.config.meta_coords, and loaded data sources ...
        pass # Simplified for clarity

    # ... other methods like get_file_index, get_levels, get_dim_names ...
```

The `ConfigManager` is where you'll find methods that combine information from different parts of the configuration or add logic needed by other components. For instance, the `setup_comparison` method reads the comparison IDs parsed by `InputConfig` and the raw inputs list from `app_data` to build the `a_list` and `b_list` which are then used extensively during plotting to decide which files to compare. It also provides properties (like `compare`, `output_dir`) that simply pass through calls to the underlying sub-config objects, making it easier for other parts of eViz to get settings without knowing which specific sub-config holds the value.

When the `GriddedModel` object (or any data-specific model) is later created, it receives a reference to the `ConfigManager`. It then uses this `ConfigManager` to find out:
*   Which files it needs to process (`config_manager.input_config.file_list`)
*   Which variables to plot from those files (`config_manager.map_params`)
*   Specific settings for those variables (from `config_manager.spec_data`)
*   Where to save the output (`config_manager.output_dir`)
*   Details needed for comparisons (`config_manager.a_list`, `config_manager.b_list`)
*   Model-specific dimension names (`config_manager.get_model_dim_name('yc')`)
*   ... and all other settings from the YAML files.

## Summary

In this chapter, we explored **Configuration Management** in eViz, which is like the detailed script or plan the Autoviz Application director follows. We learned that:

*   You tell eViz to use a specific configuration plan using YAML files via the `-c` command-line argument.
*   The Configuration Management system reads these YAML files and organizes all the settings.
*   eViz uses a stack of classes (`YAMLParser`, `Config`, `ConfigManager`, and various sub-configs like `InputConfig`, `OutputConfig`, etc.) to handle this process.
*   `YAMLParser` reads the raw files and basic metadata.
*   `Config` acts as a container for the parsed data and initializes the specialized sub-configuration objects.
*   Sub-configurations process their specific parts of the settings (e.g., `InputConfig` processes the list of input files and comparison settings).
*   `ConfigManager` is the main interface for the rest of the application to access *all* configuration settings and provides application-specific logic like setting up comparison groups.

Now that we understand how eViz reads the plan (configuration), the next step is to understand how it handles the actual data specified in that plan. How does it read different types of data files (like NetCDF, HDF5, CSV) in a unified way? That's where **Data Source Abstraction** comes in.

[Data Source Abstraction](03_data_source_abstraction_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
