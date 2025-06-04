# Chapter 2: Autoviz Application Core

Welcome back to the eViz tutorial! In [Chapter 1: Configuration System](01_configuration_system_.md), we learned how to tell eViz what we want to do by creating and using configuration files. We saw how the Configuration System acts like the control panel, taking our instructions and organizing them neatly.

Now, in this chapter, we'll look at the **Autoviz Application Core**. If the Configuration System is the control panel, the Autoviz Application Core is the **engine**. It's the central part of the program that reads those instructions from the control panel and actually *does* the work of finding data, processing it, and creating the plots.

It's the part that runs when you type `autoviz` on your command line!

## Your Second Task: Running eViz with Configuration

Let's continue with our task from Chapter 1: Plotting the `Temperature` variable from the `my_weather_data.nc` file using our `my_config.yaml` file.

How does the Autoviz Application Core use that configuration to make the plot?

The main way you interact with the Autoviz Application Core is by running the `autoviz.py` script from your terminal.

```bash
python autoviz.py -f my_config.yaml
```

This simple command tells Python to run the `autoviz.py` script and provides the path to our configuration file using the `-f` or `--configfile` argument.

You could also use a directory with configuration files for different "sources" using the `-c` or `--config` argument and specify which source to use with `-s` or `--sources`:

```bash
# Assuming my_config.yaml is in ./configs/gridded/gridded.yaml
python autoviz.py -c ./configs -s gridded
```

When you run one of these commands, the Autoviz Application Core springs into action!

## What Happens When You Run `autoviz.py`?

Let's trace the journey inside eViz when you run the command. Think of the Autoviz Application Core as the conductor of an orchestra. It doesn't play all the instruments itself, but it tells each section (like the woodwinds, strings, or percussion) when and how to play together to create the final piece (the visualization).

Here's a simplified view of the main steps:

1.  **Start the Engine:** The `autoviz.py` script begins execution.
2.  **Read Instructions:** It reads the command-line arguments you provided (like which config file to use).
3.  **Set up the Control Panel:** It uses those arguments to initialize the [Configuration System](01_configuration_system_.md) we talked about in Chapter 1. This creates the `ConfigManager` object, which holds all your settings from the YAML file.
4.  **Identify Data Sources:** It looks at the configuration (via `ConfigManager`) to figure out which types of data you want to process (like 'gridded' data from NetCDF files).
5.  **Find the Right Tool:** For each data type, it finds the specialized "factory" designed to handle that type of data. (We'll learn more about the [Data Source Factory](08_data_source_factory_.md) later).
6.  **Create the Worker:** It asks the factory to create a "model" object specifically designed to work with that data type and your configuration settings. (These are the [Source Models](05_source_models_.md), which we'll cover in another chapter).
7.  **Give the Worker Instructions:** It gives the model object the `ConfigManager`, so the model knows *exactly* what to do (which variables to plot, output directory, etc.).
8.  **Start the Work:** It tells the model object to start its process. This model will then use the [Data Processing Pipeline](06_data_processing_pipeline_.md) to read the data and the [Plotting Components](04_plotting_components_.md) to create the visualizations, all guided by the configuration.

Here’s a very simple diagram showing the flow:

```{mermaid}
sequenceDiagram
    participant CLI as Command Line
    participant AutovizScript as autoviz.py Script
    participant AutovizCore as Autoviz Class (The Engine)
    participant ConfigMgr as ConfigManager (Control Panel)
    participant SourceFactory as Source Factory (Tool Finder)
    participant SourceModel as Source Model (The Worker)
    participant OtherParts as Other eViz Parts (Pipeline, Plotting)

    CLI->>AutovizScript: Run with config file!
    AutovizScript->>AutovizCore: Create Autoviz instance (with args)
    AutovizCore->>ConfigMgr: Initialize ConfigManager (using args)
    ConfigMgr-->>AutovizCore: Provides configuration
    AutovizCore->>SourceFactory: "Which factory for 'gridded'?"
    SourceFactory-->>AutovizCore: "Here's the Gridded Factory!"
    AutovizCore->>SourceModel: "Factory, create a Gridded Model!"
    SourceModel-->>AutovizCore: "Here's the Gridded Model!"
    AutovizCore->>SourceModel: "Model, here's the config, now run!"
    SourceModel->>OtherParts: Uses config to find data, process, plot...
    OtherParts-->>SourceModel: Creates plots, etc.
    SourceModel-->>AutovizCore: (Completes task)
    AutovizCore-->>AutovizScript: (Completes task)
    AutovizScript-->>CLI: Done!
```

This diagram illustrates how the `Autoviz` class in the core orchestrates the process, relying heavily on the `ConfigManager` for instructions and delegating the actual data handling and plotting work to the appropriate `SourceModel`.

## Diving Deeper into the Code

Let's look at the code snippets related to this orchestration role.

First, the `autoviz.py` script's `main` function:

```python
# --- File: autoviz.py ---
# ... (imports and parse_command_line function) ...

def main():
    """
    Main driver for the autoviz plotting tool.
    """
    args = parse_command_line() # Get instructions from command line

    # --- Metadata extraction mode (special case) ---
    # (Skipped for this chapter - see Chapter 3)
    # if args.file: ... sys.exit() ...
    # --- End metadata extraction mode ---

    # Setup logging
    logger_setup(...)

    # Get source names from args (e.g., ['gridded'])
    input_sources = [s.strip() for s in args.sources[0].split(',')]

    # Process each source
    for source in input_sources:
        print(f"Processing source: {source}")
        # Create a specific args for this source (simplified)
        source_args = argparse.Namespace(**vars(args))
        source_args.sources = [source]

        # *** This is where the core Autoviz class is used! ***
        autoviz = Autoviz([source], args=source_args)
        autoviz.run() # Tell the Autoviz engine to run!

    print(f"Time taken = ...")

# ... (if __name__ == "__main__": main()) ...
```

The `main` function is simple. It parses the command line, sets up logging, and then, for each source type requested (`-s gridded,wrf`), it creates an `Autoviz` object and calls its `run()` method. The `Autoviz` object is the heart of the Autoviz Application Core.

Now, let's look inside the `Autoviz` class (`eviz/lib/autoviz/base.py`).

```python
# --- File: eviz/lib/autoviz/base.py ---
# ... (imports and helper functions like get_config_path_from_env, create_config, get_factory_from_user_input) ...

@dataclass
class Autoviz:
    """
    Main class for automatic visualization... orchestrates the entire process.
    """
    source_names: list # e.g., ['gridded']
    args: Namespace = None # Command line args
    # ... other attributes ...
    _config_manager: ConfigManager = None # Holds the ConfigManager

    # ... logger property ...

    def __post_init__(self):
        """Initialize the Autoviz instance."""
        self.logger.info("Start init")
        # Simplified args handling for notebooks (not shown here)

        # 1. Find the right factories for the source names
        self.factory_sources = get_factory_from_user_input(self.source_names)
        if not self.factory_sources:
             raise ValueError(...)

        # 2. *** Create the ConfigManager using the arguments ***
        self._config_manager = create_config(self.args)

        # TODO: enable processing of S3 buckets

    def run(self):
        """
        Execute the visualization process.
        """
        self.logger.info("Running Autoviz")

        # Optional check for missing files (simplified)
        self._check_input_files()

        # The ConfigurationAdapter helps process the config (more advanced)
        # self.config_adapter = ConfigurationAdapter(self._config_manager)

        # Data integration or composite field modes handled here (skipped for clarity)
        # if hasattr(self.args, 'integrate'): ...
        # if hasattr(self.args, 'composite'): ... return

        # 3. Use the factories to create model instances and tell them to run
        for factory in self.factory_sources:
            self.logger.info(f"Creating model instance for factory: {type(factory).__name__}")
            # *** Create the Source Model instance ***
            model = factory.create_root_instance(self._config_manager)

            # Pass map parameters if available (optional)
            # if hasattr(model, 'set_map_params') and self._config_manager.map_params:
            #     model.set_map_params(self._config_manager.map_params)

            # *** Tell the model to execute its process! ***
            self.logger.info("Calling model()")
            model() # This call starts the data processing and plotting!

        # Cleanup (simplified)
        # finally: self.config_adapter.close()

    # ... (set_data, _check_input_files, set_output methods) ...
```

Let's break down the `Autoviz` class:

*   **`__post_init__(self)`:** This method runs automatically right after an `Autoviz` object is created. It's the setup phase.
    *   `get_factory_from_user_input(self.source_names)`: It calls a helper function to look up the correct factory class based on the source names you provided (like 'gridded').
    *   `create_config(self.args)`: **Crucially,** this calls the `create_config` function (shown below) which uses the command-line arguments (`self.args`) to set up the entire [Configuration System](01_configuration_system_.md), resulting in the `_config_manager` object.
*   **`run(self)`:** This is the main method that orchestrates the visualization process.
    *   It iterates through the `factory_sources` it found during initialization.
    *   `factory.create_root_instance(self._config_manager)`: For each factory, it asks the factory to create the actual `SourceModel` object. It passes the `_config_manager` to the model so the model knows all the settings.
    *   `model()`: It then simply calls the model object's `()` method (making it "callable"). This single call to the model object is the trigger that starts the [Data Processing Pipeline](06_data_processing_pipeline_.md), reads the necessary data based on the configuration, and invokes the [Plotting Components](04_plotting_components_.md) to create the plots.

Let's look at the helper functions used by `Autoviz`:

```python
# --- File: eviz/lib/autoviz/base.py ---
# ... (imports) ...
from eviz.lib.config.config import Config # Need this!
from eviz.lib.config.config_manager import ConfigManager # Need this!
# ... (get_config_path_from_env) ...

def create_config(args) -> ConfigManager:
    """
    Create a ConfigManager instance from command-line arguments.
    """
    source_names = args.sources[0].split(',')
    config_dir = args.config # Path from -c
    config_file = args.configfile # Path from -f

    # Determine config files based on args (simplified logic)
    if config_file:
        # Use specific file(s) provided by -f
        config = Config(source_names=source_names, config_files=config_file)
    else:
        # Use files based on source name in config dir (from -c or env var)
        # ... logic to find config_files like ['path/to/configs/gridded/gridded.yaml'] ...
        config = Config(source_names=source_names, config_files=config_files)

    # *** This calls the Config object which initializes all sub-configs ***
    # We get back the initialized specialized configs
    input_config = config.input_config
    output_config = config.output_config
    system_config = config.system_config
    history_config = config.history_config

    # *** Wrap them in a ConfigManager and return it ***
    return ConfigManager(input_config, output_config, system_config, history_config, config=config)
```

The `create_config` function is the bridge to the [Configuration System](01_configuration_system_.md). It takes the command-line arguments and passes the relevant parts (source names, config file paths) to the `Config` class constructor. As we saw in Chapter 1, the `Config` object then does all the heavy lifting of parsing, organizing, and initializing the settings. Finally, `create_config` wraps the resulting specialized config objects within a `ConfigManager` and returns it. This `ConfigManager` is then used by the `Autoviz` class.

```python
# --- File: eviz/lib/autoviz/base.py ---
# ... (imports) ...
from eviz.models.source_factory import (GriddedSourceFactory, WrfFactory, ...) # Import factories

def get_factory_from_user_input(inputs) -> list:
    """
    Return factory classes associated with user input sources.
    """
    # This dictionary maps source names (strings from -s argument)
    # to the actual Python factory class instances.
    mappings = {
        "gridded": GriddedSourceFactory(),
        "wrf": WrfFactory(),
        "lis": LisFactory(),
        # ... other mappings ...
    }
    factories = []
    for i in inputs: # Loop through requested source names
        if i not in mappings:
            print(f"ERROR: '{i}' is not a valid source name.")
            import sys
            sys.exit(1)
        factories.append(mappings[i]) # Add the corresponding factory instance
    return factories # Return the list of factories
```

The `get_factory_from_user_input` function is straightforward. It takes the list of source names from the command line (`['gridded']`) and uses a dictionary (`mappings`) to find the correct "factory" object (`GriddedSourceFactory()`) for each name. This is how the Autoviz Core knows which specific tools it will need to process the data.

## How the Autoviz Core Connects Everything

The Autoviz Application Core (`Autoviz` class and supporting functions like `create_config` and `get_factory_from_user_input`) acts as the central hub:

*   It receives instructions from the user via command-line arguments, which guide the creation of the [Configuration System](01_configuration_system_.md) (`ConfigManager`).
*   It uses the configuration to identify the necessary [Source Models](05_source_models_.md) (like `GriddedModel`, `WrfModel`, etc.) and obtains their corresponding factory objects ([Data Source Factory](08_data_source_factory_.md)).
*   It instructs these factories to create instances of the specific models, passing the `ConfigManager` to them.
*   By calling `model()`, it triggers the model's execution, which involves using the [Data Processing Pipeline](06_data_processing_pipeline_.md) (which includes the [Data Source Abstraction](07_data_source_abstraction_.md)) to load and prepare data, and then using the [Plotting Components](04_plotting_components_.md) to generate the visualizations, all according to the settings in the `ConfigManager`.

The [Metadata Generator (metadump)](03_metadata_generator__metadump__.md) is a slightly different path handled directly by the `autoviz.py` script *before* the `Autoviz` class is even created, as shown in the commented-out section of the `main` function. It's a utility function rather than part of the core visualization workflow orchestrated by the `Autoviz` class.

## Conclusion

The Autoviz Application Core, primarily embodied by the `Autoviz` class in `eviz/lib/autoviz/base.py`, is the engine that drives the visualization process. It takes the user's high-level requests (via command line and configuration) and orchestrates the different parts of the eViz system – the [Configuration System](01_configuration_system_.md), the [Source Models](05_source_models_.md) (found via the [Data Source Factory](08_data_source_factory_.md)), the [Data Processing Pipeline](06_data_processing_pipeline_.md), and the [Plotting Components](04_plotting_components_.md) – to produce the desired visualizations. It doesn't do the data reading or plotting itself, but it makes sure the right components are created, configured, and told to run in the correct order.

Now that we've seen the main engine, let's take a brief detour to look at the [Metadata Generator (metadump)](03_metadata_generator__metadump__.md), a helpful tool that works alongside the core visualization process.

[Next Chapter: Metadata Generator (metadump)](03_metadata_generator__metadump__.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
