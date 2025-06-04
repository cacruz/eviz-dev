# Chapter 1: Configuration System

Welcome to the eViz tutorial! This first chapter is all about the **Configuration System**. Think of this system as the **control panel** for eViz. It's where you give eViz instructions on what you want it to do, without needing to change the program's code itself.

Why do we need a control panel? Imagine you want to visualize data from different weather models, compare them, or plot different variables. If you had to change the program code every time you wanted to do something slightly different, it would be a huge hassle! The Configuration System solves this by letting you describe your task using simple text files.

## Your First Task: Plotting a Variable from a File

Let's imagine a very common task: You have a data file (let's say `my_weather_data.nc` in a folder called `~/data/model_output`) and you want eViz to plot a specific variable from it, for example, `Temperature`. You also want to save the resulting image as a PNG file in a specific folder, like `./my_plots`.

How do you tell eViz to do this? You use the Configuration System, primarily through **YAML files**.

## What is YAML?

YAML stands for "YAML Ain't Markup Language". It's a human-friendly data format, perfect for configuration files because it's easy to read and write. It uses indentation (spaces!) to show structure, like bullet points in an outline.

Here's a *very* simplified example of what a configuration YAML file for our task might look like:

```yaml
# my_config.yaml

# Section for input files
inputs:
  - name: my_weather_data.nc
    location: ~/data/model_output
    exp_id: BaseRun # An identifier for this data

# Section for general settings related to inputs
for_inputs:
  to_plot:
    Temperature: xy # Tell eViz to plot the 'Temperature' variable as an 'xy' plot (e.g., a map)

# Section for output settings
outputs:
  print_to_file: True # Yes, save the plot to a file
  output_dir: ./my_plots # Save it here
  print_format: png # Save it as a PNG image
```

**Explanation:**

*   `inputs`: This section lists the data files you want eViz to load. We specify the `name` and `location` of our data file and give it an `exp_id` (Experiment ID) for easy reference.
*   `for_inputs`: This section holds settings that apply *to* the input data or how they should be handled. Here, we use `to_plot` to list the variables we want to visualize and the type of plot (`xy` is common for horizontal maps).
*   `outputs`: This section controls how eViz should save or display the results. We tell it to `print_to_file` (save to disk), where (`output_dir`), and in what `print_format`.

This simple YAML file contains all the basic instructions eViz needs to load the file, find the `Temperature` variable, plot it, and save the image.

## How eViz Handles Configuration (The Big Picture)

When you run eViz, you point it to one or more of these YAML files. eViz then performs these steps:

1.  **Read the Files:** It reads the content of the specified YAML files.
2.  **Combine Settings:** If you provide multiple files (e.g., one for inputs, another for plot styles), it combines the settings from all of them.
3.  **Organize the Data:** It takes the raw information from the YAML files and organizes it neatly into different categories (input settings, output settings, system settings, etc.).
4.  **Prepare for Use:** It makes these organized settings easily accessible to all other parts of the eViz application.

Think of it like this:

```{mermaid}
sequenceDiagram
    participant User
    participant YAMLFile as Configuration File (YAML)
    participant YAMLParser as YAML Parser
    participant AppData as Raw Data Container
    participant Config as Central Config Object
    participant ConfigManager as Application Interface
    participant OtherParts as Other eViz Components

    User->>eViz: Run eViz with config file(s)
    eViz->>YAMLParser: "Read these YAML files!"
    YAMLParser->>YAMLFile: Reads content
    YAMLParser-->>AppData: Raw data loaded
    AppData->>Config: Data passed to main Config
    Config->>AppData: Reads & organizes data
    Config->>InputConfig: Initializes Input Config
    Config->>OutputConfig: Initializes Output Config
    Config->>...: Initializes other Configs
    Config-->>ConfigManager: Provides organized Config
    ConfigManager->>Config: Wraps main Config
    ConfigManager->>OtherParts: Provides easy access to settings
    OtherParts->>ConfigManager: "What file should I load?"
    ConfigManager-->>OtherParts: "Here's the path from the config!"
    OtherParts->>ConfigManager: "Where should I save the plot?"
    ConfigManager-->>OtherParts: "Here's the output directory!"
```

The key players here are `YAMLParser`, `AppData`, `Config`, and `ConfigManager`.

## Diving Deeper into the Code (Simplified)

Let's look at the core Python classes that make this system work. We'll keep the code snippets very short and focus on their purpose.

### `AppData`: Just the Raw Data

This is a simple container for the information read directly from the YAML files.

```python
# eviz/lib/config/app_data.py
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class AppData:
    """Data class for application-level configuration."""
    inputs: Dict[str, Any] = field(default_factory=dict)
    for_inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    system_opts: Dict[str, Any] = field(default_factory=dict)
    # ... other sections like history, plot_params
```

`AppData` holds the raw dictionary-like structures that come from parsing the YAML. It doesn't do much by itself, it just stores the data.

### `YAMLParser`: The Reader

This class's main job is to read the YAML files you provide and fill the `AppData` container.

```python
# eviz/lib/config/yaml_parser.py
from dataclasses import dataclass, field
from typing import List, Dict, Any
import eviz.lib.utils as u # Helper for loading YAML

@dataclass
class YAMLParser:
    config_files: List[str] # List of YAML files to read
    source_names: List[str] # Names associated with sources
    app_data: Dict[str, Any] = field(default_factory=dict) # Where raw data goes
    spec_data: Dict[str, Any] = field(default_factory=dict) # Data about variables/plots

    def parse(self):
        """Read YAML files and populate app_data and spec_data."""
        concat = self._concatenate_yaml()
        # ... initializes other internal data structures
        # This method is the main entry point for reading configs

    def _concatenate_yaml(self) -> List[Dict[str, Any]]:
        """Read and merge multiple YAML files."""
        result = {}
        for file_path in self.config_files:
            yaml_content = u.load_yaml_simple(file_path)
            # Merge contents into result dictionary (simplified logic)
            if 'inputs' in yaml_content:
                 result.setdefault('inputs', []).extend(yaml_content['inputs'])
            if 'for_inputs' in yaml_content:
                 result.setdefault('for_inputs', {}).update(yaml_content['for_inputs'])
            # ... merge other top-level sections

            # Also look for associated specs files (e.g., my_config_specs.yaml)
            specs_file = "..." # Construct specs filename
            if os.path.exists(specs_file):
                 specs_content = u.load_yaml_simple(specs_file)
                 self.spec_data.update(specs_content)

        self.app_data = result # Store the merged data
        return [result] # Return a list (simplified)
```

The `YAMLParser` reads the files, handles combining settings if you have multiple config files, and also looks for separate "specs" files which contain detailed information about variables and how to plot them.

### `Config`: The Organizer

This is the central hub. It takes the raw `AppData` and creates specialized configuration objects (`InputConfig`, `OutputConfig`, etc.) to manage different areas of the settings.

```python
# eviz/lib/config/config.py
from dataclasses import dataclass, field
from typing import List, Dict, Any
from eviz.lib.config.app_data import AppData
from eviz.lib.config.input_config import InputConfig
from eviz.lib.config.output_config import OutputConfig
from eviz.lib.config.system_config import SystemConfig
from eviz.lib.config.history_config import HistoryConfig
from eviz.lib.config.yaml_parser import YAMLParser

@dataclass
class Config:
    """Main configuration class that delegates responsibilities."""
    source_names: List[str]
    config_files: List[str]
    app_data: AppData = field(default_factory=AppData) # Raw data from parser
    spec_data: Dict[str, Any] = field(default_factory=dict) # Specs data

    def __post_init__(self):
        # 1. Parse YAML files
        self.yaml_parser = YAMLParser(config_files=self.config_files, source_names=self.source_names)
        self.yaml_parser.parse()

        # 2. Get raw data from parser
        self.app_data = AppData(**self.yaml_parser.app_data)
        self.spec_data = self.yaml_parser.spec_data
        # ... get other data like map_params, meta_coords, etc. from parser

        # 3. Create specialized config objects (delegation)
        self.input_config = InputConfig(self.source_names, self.config_files)
        self.output_config = OutputConfig()
        self.system_config = SystemConfig()
        self.history_config = HistoryConfig()

        # 4. Give raw data to the specialized config objects
        self._assign_app_data_to_subconfigs()

        # 5. Initialize the specialized config objects
        self.initialize()

    def _assign_app_data_to_subconfigs(self):
        """Assign app_data to all sub-configurations."""
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
```

`Config` is like the main switchboard. It uses the `YAMLParser` to get the raw data, then creates smaller, dedicated objects (`InputConfig`, `OutputConfig`, etc.) and gives them the relevant pieces of the data to manage. This keeps things organized – `InputConfig` only worries about inputs, `OutputConfig` only about outputs, and so on.

### Specialized Configs (`InputConfig`, `OutputConfig`, etc.)

These classes handle the details for a specific part of the configuration. They take the relevant section of `AppData` and turn it into easy-to-access attributes.

Here's a peek at `OutputConfig`:

```python
# eviz/lib/config/output_config.py
from dataclasses import dataclass, field
import os
from typing import Dict, Any

@dataclass
class OutputConfig:
    app_data: Dict[str, Any] = field(default_factory=dict) # Will receive output section

    # Default values for output settings
    output_dir: str = "./output_plots"
    print_to_file: bool = False
    print_format: str = "png"
    add_logo: bool = False
    # ... other output settings

    def initialize(self):
        """Initialize output configuration."""
        # Get the 'outputs' section from app_data
        outputs = self.app_data.get('outputs', {})

        # Update attributes based on the YAML settings
        self.add_logo = outputs.get('add_logo', self.add_logo) # Use YAML value or default
        self.print_to_file = outputs.get('print_to_file', self.print_to_file)
        self.print_format = outputs.get('print_format', self.print_format)
        # ... set other attributes

        # Ensure output directory exists if saving to file
        self._set_output_dir()

    def _set_output_dir(self):
        """Set the output directory."""
        if self.print_to_file:
            if not os.path.exists(self.output_dir):
                # Create the directory if it doesn't exist
                os.makedirs(self.output_dir)
```

Each specialized config class (`InputConfig`, `SystemConfig`, `HistoryConfig`) works similarly: it receives the relevant part of the `AppData`, and its `initialize` method processes that data to set up its own properties (like `output_dir`, `print_to_file`, `use_mp_pool`, etc.). This makes the settings readily available as Python object attributes.

### `ConfigManager`: The Application's Gateway

This class is what most other parts of eViz interact with. It wraps the `Config` object and adds application-specific logic or provides convenient ways to access settings.

```python
# eviz/lib/config/config_manager.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from eviz.lib.config.config import Config
from eviz.lib.config.input_config import InputConfig
from eviz.lib.config.output_config import OutputConfig
# ... import other config classes

@dataclass
class ConfigManager:
    """Enhanced configuration manager for the eViz application."""
    # ConfigManager holds instances of the specialized configs
    input_config: InputConfig
    output_config: OutputConfig
    system_config: SystemConfig
    history_config: HistoryConfig
    config: Config # Also holds the central Config object

    # ... other attributes like a_list, b_list for comparisons, internal state

    def __post_init__(self):
        """Initialize the ConfigManager."""
        # Perform application-specific setup, e.g., figuring out which files to compare
        self.setup_comparison()

    # Example of a convenient property that delegates to a sub-config
    @property
    def output_dir(self):
        """The directory to write output files to."""
        return self.output_config.output_dir

    # Example of application-specific logic using config data
    def setup_comparison(self):
        """Figure out which files/experiments should be compared based on config."""
        self.a_list = []
        self.b_list = []
        # Check input_config for 'compare' or 'compare_diff' settings
        if self.input_config._compare or self.input_config._compare_diff:
            # ... logic to find files based on exp_id and populate a_list, b_list
            pass # Simplified
            
    # Magic method to easily access attributes from wrapped config objects
    def __getattr__(self, name):
         # If the attribute isn't found directly in ConfigManager,
         # check the wrapped config objects (config, input_config, etc.)
         if hasattr(self.config, name):
             return getattr(self.config, name)
         # ... check other sub-configs
         raise AttributeError(...)
```

The `ConfigManager` is the main object that other parts of eViz will ask for information. It can directly provide settings from the sub-configs (like `output_dir`) or perform slightly more complex tasks that combine settings from different places (like `setup_comparison`). The `__getattr__` magic method makes it even easier – you can often access settings like `config_manager.use_cartopy` even if `use_cartopy` is technically stored in `input_config`, because `ConfigManager` knows to look there if it doesn't have the attribute itself.

## How Configuration Powers eViz

The Configuration System, through `ConfigManager`, provides the central source of truth for the entire application.

*   The [Autoviz Application Core](02_autoviz_application_core_.md) reads settings from `ConfigManager` to know which data sources to process, which variables to plot, and what kind of plots to make.
*   The [Metadata Generator (metadump)](03_metadata_generator__metadump__.md) might use configuration settings to know what kind of metadata to extract or where to save it.
*   [Plotting Components](04_plotting_components_.md) get instructions from `ConfigManager` on everything from the variable name to plot, to the colormap to use, to the output directory.
*   The [Data Processing Pipeline](06_data_processing_pipeline_.md) and the [Data Source Abstraction](07_data_source_abstraction_.md) use information from the configuration (via `ConfigManager` and `InputConfig`) to find file paths, determine file formats, and know which data sources and variables are relevant. The [Data Source Factory](08_data_source_factory_.md) also relies on configuration to determine which reader type (like NetCDF or CSV) to use for a given file.

In essence, the Configuration System is the brain that receives the user's high-level goals (defined in YAML) and translates them into concrete instructions that guide every other part of eViz.

## Conclusion

In this chapter, we learned that the Configuration System is eViz's control panel, allowing users to specify tasks using easy-to-read YAML files. We saw how these files are read by `YAMLParser`, stored raw in `AppData`, organized by `Config` into specialized configuration objects like `InputConfig` and `OutputConfig`, and finally made easily accessible to the rest of the application through `ConfigManager`.

This system ensures that eViz is flexible and can be directed to handle various data and plotting tasks just by changing text files, without touching the underlying code.

Now that we understand how eViz takes instructions, let's move on to the next chapter to see how the main part of the application uses these instructions to actually perform the visualization tasks.

[Next Chapter: Autoviz Application Core](02_autoviz_application_core_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
