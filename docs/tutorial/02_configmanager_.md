# Chapter 2: ConfigManager

Welcome back to the eViz tutorial! In the [previous chapter](01_autoviz__application_orchestrator__.md), we met **Autoviz**, the conductor that orchestrates the entire visualization process. We saw how Autoviz takes your command, figures out which tools (like factories) it needs, and then relies on a "blueprint" to know *exactly* what to do.

But where does this blueprint come from? How does Autoviz, or any other part of eViz, know which data files to read, what variables are important, what colors to use in a plot, or where to save the final images?

This is where the **ConfigManager** comes in.

## What is ConfigManager and Why Do We Need It?

Imagine you're building something complex, like a model airplane or even cooking a detailed recipe. You need clear instructions: a list of parts or ingredients, steps to follow, specifications for how things should look or taste. In eViz, the "blueprint" containing all these instructions is managed by the **ConfigManager**.

The **ConfigManager** is the central brain for *all* configuration settings within eViz. It gathers information from different sources – primarily configuration files you provide and details from your command-line inputs – and makes it available to every part of the application that needs it.

Think of it as the main settings panel. Instead of each component (like the part that reads data or the part that makes plots) having to figure out its settings individually, they all go to the **ConfigManager** for their instructions. This keeps everything organized and ensures that all components are working with the same set of rules.

### A Simple Use Case: Telling eViz What to Plot

Let's revisit our example from Chapter 1: running `python autoviz.py -s gridded`. When you run this, Autoviz knows you want to work with `gridded` data. But it still needs more details. Which specific gridded data file? Which variable *in* that file should it visualize? How should the plot look?

You provide these details in **configuration files**, typically written in YAML format (YAML is just a simple way to write structured data, easy for both humans and computers to read). The **ConfigManager**'s job is to load these YAML files, combine their settings with command-line arguments, and store them.

Let's look at a minimal example of what a configuration YAML file might look like:

```yaml
# my_gridded_config.yaml

inputs:
  - name: my_gridded_data.nc # The data file name
    location: /path/to/your/data # Where to find the file
    exp_id: my_experiment # A unique ID for this data

for_inputs:
  to_plot: # What variables to plot from the input file
    Temperature: xy # Plot Temperature using the default 'xy' method
    Pressure: yz # Plot Pressure using the 'yz' method (e.g., latitude vs. height)

outputs:
  print_to_file: true # Save plots to files
  output_dir: ./my_plots # Directory to save plots
  print_format: png # Save as PNG images

system_opts:
  use_mp_pool: false # Don't use multiprocessing for simplicity
```

This simple file tells eViz:
*   Look for a file named `my_gridded_data.nc` in `/path/to/your/data`.
*   This data belongs to an experiment with ID `my_experiment`.
*   From this file, plot the variables `Temperature` and `Pressure`.
*   Save the resulting plots as PNG files in a directory called `my_plots`.

To use this configuration with `Autoviz`, you would typically run:

```bash
python autoviz.py -s gridded -f my_gridded_config.yaml
```

Here, the `-f my_gridded_config.yaml` part tells `Autoviz` *which* configuration file to load. `Autoviz` then passes this information to the `ConfigManager`.

## ConfigManager's Core Jobs

At a high level, the `ConfigManager` performs several key tasks:

1.  **Loading:** It reads one or more configuration files (like `my_gridded_config.yaml`) and other special metadata files (`meta_coordinates.yaml`, `meta_attributes.yaml`, etc.).
2.  **Parsing:** It understands the structure of the YAML data.
3.  **Combining:** It merges settings from different files and integrates command-line arguments provided by `Autoviz`.
4.  **Organizing:** It structures the loaded settings into logical categories (like input settings, output settings, system settings, etc.).
5.  **Accessing:** It provides a simple way for other parts of eViz to get the settings they need.

This last point is crucial. Instead of reaching directly into the raw YAML data, components ask the `ConfigManager` for specific pieces of information, like "What is the output directory?" or "Which variables should I plot?".

### Inside ConfigManager: The Settings Hub

Let's peek inside the `ConfigManager` to see how it works.

#### The Loading Process (Simplified Walkthrough)

When `Autoviz` creates a `ConfigManager` instance (remember the `create_config` function from Chapter 1?), here's what happens:

1.  **Receive Instructions:** `ConfigManager` receives the list of configuration files (`-f`) and source names (`-s`) from `Autoviz`.
2.  **Parse YAML:** It uses a helper component, the `YAMLParser`, to read the specified YAML files (`my_gridded_config.yaml` in our example).
3.  **Find Associated Specs:** The `YAMLParser` automatically looks for related "specs" files (e.g., `my_gridded_config_specs.yaml`). These files contain detailed information about variables and plot appearances.
4.  **Load Metadata:** The `YAMLParser` also loads global metadata files (`meta_coordinates.yaml`, `meta_attributes.yaml`, `species_database.yaml`) which define standard names and properties.
5.  **Consolidate Data:** The `YAMLParser` combines all this parsed data into structured Python dictionaries.
6.  **Organize into Sub-Configs:** `ConfigManager` takes the consolidated data and distributes it into specialized configuration objects (`InputConfig`, `OutputConfig`, `SystemConfig`, `HistoryConfig`, `PathsConfig`). Each of these handles settings for a specific area of the application.
7.  **Initialize Sub-Configs:** It tells these sub-config objects to initialize themselves, processing their specific settings (e.g., `InputConfig` figures out the list of input files and their formats, `OutputConfig` sets up the output directory).
8.  **Ready for Access:** The `ConfigManager` now holds references to all these initialized sub-config objects and makes their data accessible.

Here's a simplified sequence diagram:

```{mermaid}
sequenceDiagram
    participant A as Autoviz
    participant CM as ConfigManager
    participant YP as YAMLParser
    participant SubC as Sub-Configs (Input, Output, etc.)
    participant YAMLFile as my_gridded_config.yaml
    participant SpecsFile as *_specs.yaml
    participant MetaFiles as meta_*.yaml

    A->>CM: Create ConfigManager(args)
    CM->>CM: Initialize
    CM->>YP: Create YAMLParser(config_files)
    YP->>YAMLFile: Read content
    YP->>SpecsFile: Read content (if exists)
    YP->>MetaFiles: Read content
    YAMLFile-->>YP: Data
    SpecsFile-->>YP: Data
    MetaFiles-->>YP: Data
    YP-->>CM: Return parsed data (app_data, spec_data, etc.)
    CM->>SubC: Create Sub-Configs (pass app_data)
    CM->>SubC: Initialize()
    SubC->>SubC: Process specific settings
    SubC-->>CM: Finished initialization
    CM-->>A: Return ConfigManager instance (ready!)
```

This diagram shows how the `ConfigManager` orchestrates the loading and parsing using `YAMLParser` and then organizes the results into dedicated `Sub-Configs`.

#### Code Structure (Simplified)

Let's look at some minimal code snippets from the `eviz/lib/config` directory to see these pieces.

First, the main `ConfigManager` class:

```python
# eviz/lib/config/config_manager.py (simplified)
from dataclasses import dataclass, field
# ... other imports ...
from eviz.lib.config.config import Config # The core config object
from eviz.lib.config.input_config import InputConfig
from eviz.lib.config.output_config import OutputConfig
# ... other sub-configs ...

@dataclass
class ConfigManager:
    """
    Enhanced configuration manager for the eViz application.
    """
    # ConfigManager holds references to the main Config object
    # and specific sub-configs
    config: Config # This holds the combined parsed data
    input_config: InputConfig
    output_config: OutputConfig
    # ... other sub-configs ...

    # Other internal state or data specific to ConfigManager
    a_list: List[int] = field(default_factory=list) # Used for comparison indexing
    b_list: List[int] = field(default_factory=list)
    _findex: int = 0 # Current file index being processed
    _ds_index: int = 0 # Current data source index

    def __post_init__(self):
        """Initialize the ConfigManager after construction."""
        self.logger.info("Starting ConfigManager initialization")
        # Some initialization logic happens here,
        # including potentially setting up comparison lists
        self.setup_comparison()

    # ... properties for easy access (see below) ...

    def setup_comparison(self):
        """Set up comparison lists based on config settings."""
        # This method reads the 'compare'/'overlay' settings
        # from input_config and app_data and builds a_list and b_list.
        pass # Simplified

    def get_model_dim_name(self, dim_name):
        """
        Get model-specific dimension name (delegates to internal logic
        that uses meta_coords and current file info).
        """
        # This is where ConfigManager provides utility methods
        # accessing organized data from its sub-configs or meta files.
        pass # Simplified

    # ... other utility methods ...

    # --- How ConfigManager provides easy access ---
    # This is a key feature: ConfigManager acts as a facade.
    # You can access settings from sub-configs directly through ConfigManager.

    @property
    def paths(self):
        """Access to paths configuration (delegates to self.config.paths)."""
        return self.config.paths # Get paths from the core config object

    @property
    def app_data(self):
        """Access to application data (delegates to self.config.app_data)."""
        return self.config.app_data # Get app_data from the core config object

    @property
    def output_dir(self):
        """Access to output directory (delegates to output_config)."""
        return self.output_config.output_dir # Get output_dir from the OutputConfig

    @property
    def compare(self):
        """Access to compare flag (delegates to input_config)."""
        return self.input_config._compare # Get compare flag from InputConfig

    # ... many more properties like these ...

```

This shows that `ConfigManager` is essentially a container for other configuration objects (`Config`, `InputConfig`, etc.). Its `__post_init__` method kicks off the initialization of these sub-parts. The important pattern here is the use of `@property` methods. These allow you to write `config_manager.output_dir` and the `ConfigManager` automatically fetches the `output_dir` from its `output_config` object. This simplifies accessing settings.

The core `Config` class is responsible for using the `YAMLParser`:

```python
# eviz/lib/config/config.py (simplified)
from dataclasses import dataclass, field
from typing import List, Dict, Any
# ... other imports ...
from eviz.lib.config.yaml_parser import YAMLParser
from eviz.lib.config.app_data import AppData

@dataclass
class Config:
    """
    Main configuration class that delegates responsibilities to sub-configurations.
    """
    source_names: List[str]
    config_files: List[str]

    # Data populated by the YAMLParser
    app_data: AppData = field(default_factory=AppData) # Application-wide data
    spec_data: Dict[str, Any] = field(default_factory=dict) # Specs data
    map_params: Dict[int, Dict[str, Any]] = field(default_factory=dict) # Organized plot data
    meta_coords: dict = field(default_factory=dict) # Coordinate mappings
    # ... other metadata ...

    # Sub-configuration objects (created and held by Config)
    input_config: InputConfig = field(init=False) # Created in __post_init__
    output_config: OutputConfig = field(init=False)
    # ... other sub-configs ...


    def __post_init__(self):
        # Create and use the YAMLParser
        self.yaml_parser = YAMLParser(config_files=self.config_files, source_names=self.source_names)
        self.yaml_parser.parse() # This reads and parses all files

        # Assign parsed data from YAMLParser to Config attributes
        self.app_data = AppData(**self.yaml_parser.app_data)
        self.spec_data = self.yaml_parser.spec_data
        self.map_params = self.yaml_parser.map_params
        # ... assign other parsed data ...

        # Create the sub-config objects
        self.input_config = InputConfig(self.source_names, self.config_files)
        self.output_config = OutputConfig()
        # ... create other sub-configs ...

        # Assign the app_data to the sub-configs
        self._assign_app_data_to_subconfigs()

        # Initialize the sub-configs
        self.initialize()

    def _assign_app_data_to_subconfigs(self):
        """Assign app_data to all sub-configurations."""
        self.input_config.app_data = self.app_data
        self.output_config.app_data = self.app_data
        # ... assign to others ...

    def initialize(self):
        """Initialize all configurations (calls initialize on sub-configs)."""
        self.input_config.initialize()
        self.output_config.initialize()
        # ... initialize others ...

    # ... properties and methods ...

```

The `Config` object's main job is to use the `YAMLParser` to load everything and then create and manage the specialized sub-config objects (`InputConfig`, `OutputConfig`, etc.), passing them the relevant data.

Finally, the `YAMLParser` is the component that actually reads the files:

```python
# eviz/lib/config/yaml_parser.py (simplified)
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
# ... other imports ...
import eviz.lib.utils as u # Utility functions for file loading

@dataclass
class YAMLParser:
    config_files: List[str] # List of config files to read
    source_names: List[str]

    # Data that will be populated after parsing
    app_data: Dict[str, Any] = field(default_factory=dict)
    spec_data: Dict[str, Any] = field(default_factory=dict)
    _map_params: Dict[int, Dict[str, Any]] = field(default_factory=dict) # Internal mapping for plots
    meta_attrs: dict = field(default_factory=dict)
    meta_coords: dict = field(default_factory=dict)
    # ... other parsed data ...


    def parse(self):
        """Parse YAML files and populate attributes."""
        # 1. Read and merge the main config files
        concat = self._concatenate_yaml()

        # 2. Initialize the plotting structure from the combined data
        self._init_map_params(concat)

        # 3. Read standard metadata files
        self.meta_coords = u.read_meta_coords() # Uses a utility to read meta_coordinates.yaml
        self.meta_attrs = u.read_meta_attrs()   # Uses a utility to read meta_attributes.yaml
        # ... read others ...


    def _concatenate_yaml(self) -> List[Dict[str, Any]]:
        """Read and merge multiple YAML files and their associated specs."""
        concat = []
        result = {} # This will hold the combined app_data structure

        for file_path in self.config_files:
            yaml_content = u.load_yaml_simple(file_path) # Use utility to load YAML
            concat.append(yaml_content)

            # Merge sections like 'inputs', 'outputs', 'system_opts' into result
            if 'inputs' in yaml_content:
                result.setdefault('inputs', []).extend(yaml_content['inputs'])
            # ... merge other sections ...

            # Look for and load associated specs files
            specs_file = os.path.join(os.path.dirname(file_path), f"{os.path.splitext(os.path.basename(file_path))[0]}_specs.yaml")
            if os.path.exists(specs_file):
                specs_content = u.load_yaml_simple(specs_file)
                self.spec_data.update(specs_content) # Add specs to spec_data
            # ... handle missing specs file ...

        self.app_data = result # Store the combined app_data
        return concat # Return combined data for _init_map_params

    def _init_map_params(self, concat: List[Dict[str, Any]]):
        """Organize data specifically for plotting routines (like which file/field pairs to plot)."""
        _maps = {}
        # Logic to iterate through the combined input data (concat or self.app_data)
        # and create entries in _maps for each file/variable/plot_type combination.
        # This logic is somewhat complex as it handles different config structures
        # and comparison/overlay settings.
        # ... simplified logic ...
        self._map_params = _maps # Store the result

    # ... properties and methods ...

```

The `YAMLParser` is the workhorse for file reading. It uses helper functions (`u.load_yaml_simple`) to load the YAML content, finds related files (like `_specs.yaml`), and combines everything into dictionaries (`app_data`, `spec_data`).

By combining these components, `ConfigManager` provides a single, well-organized source of truth for all eViz settings. Any part of the application that needs to know configuration details can simply ask the `ConfigManager`.

### Conclusion

In this chapter, we learned that the **ConfigManager** is eViz's central repository for all configuration settings. It loads parameters from YAML files (and associated specs/metadata files) and command-line arguments, organizes them into logical sub-configurations, and provides an easy way for other parts of the system to access this information.

It takes the raw instructions you provide in files and through the command line and turns them into a structured, accessible blueprint used throughout the application, ensuring that everything runs according to your specifications.

With `Autoviz` orchestrating the process and `ConfigManager` providing the detailed instructions, the next step is to understand how the data itself flows through the system to be read, processed, and prepared for plotting. This is the job of the **[DataPipeline](03_datapipeline_.md)**, which we'll explore in the next chapter.

[Next Chapter: DataPipeline](03_datapipeline_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)