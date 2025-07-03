# Chapter 4: Autoviz (Main Application)

Welcome back to the eViz tutorial! In the last few chapters, we've been building up our understanding of eViz's core components: how it gets the raw data ([Chapter 1: Data Source](01_data_source_.md)), how it manages all the settings and instructions you provide ([Chapter 2: Config Manager](02_config_manager_.md)), and how it reads those instructions from YAML files ([Chapter 3: YAML Parser](03_yaml_parser_.md)).

Now, we're ready to look at the component that brings all of these together and actually *runs* the visualization process: the **Autoviz** class.

## What Problem Does Autoviz Solve?

Imagine you have all the pieces ready – your data files, your configuration files specifying exactly what plots to make, and the components to read data and manage settings. How does the application actually *start* and coordinate the work?

The `Autoviz` class is the **main entry point** and **director** of the entire eViz application when you run it in automatic visualization mode. It's the high-level orchestrator that:

1.  Reads the command-line instructions you give it.
2.  Uses the [Config Manager](02_config_manager_.md) to load and process all your settings from the configuration files.
3.  Identifies *which* specific types of data handling and plotting are needed based on your configuration (like figuring out if you're working with gridded model output, observational data, etc.).
4.  Sets up the overall workflow, often involving what's called a "Data Pipeline" (which we'll cover in the next chapter).
5.  Triggers the actual data loading, processing, and visualization steps.

Think of `Autoviz` as the conductor leading an orchestra. The musicians are the other components (Data Sources, Config Manager, Plotters, etc.), the score is your configuration, and `Autoviz` is the one who reads the score, tells everyone when to start playing, and keeps them all in sync to produce the final musical piece (your visualizations!).

## Our Central Use Case: Running eViz

The most common way you interact with the `Autoviz` concept is simply by running the eViz command-line script, typically named `autoviz.py`.

When you type a command like this in your terminal:

```bash
python autoviz.py -s gridded -c /path/to/my/config_dir
```

...you are invoking the code that lives inside the `autoviz.py` script, and the core of that script is creating and running an `Autoviz` instance.

## Using Autoviz (from the Command Line)

As a user, you primarily interact with `Autoviz` through the command line arguments provided to the `autoviz.py` script. The script takes these arguments and uses them to configure and run the `Autoviz` object.

For example, let's revisit a command:

```bash
python autoviz.py -s gridded -c /path/to/my/config_dir
```

Here:
*   `python autoviz.py` starts the main script.
*   `-s gridded` tells the script (and eventually `Autoviz`) that you are working with 'gridded' data sources.
*   `-c /path/to/my/config_dir` tells the script (and eventually `Autoviz` and its [Config Manager](02_config_manager_.md)) where to find your configuration files.

The `autoviz.py` script handles parsing these arguments and setting up the `Autoviz` class.

## Under the Hood: How It Works

Let's peek behind the curtain of the `autoviz.py` script and the `Autoviz` class itself (`eviz/lib/autoviz/base.py`).

When you run `python autoviz.py ...`, the `main()` function in `autoviz.py` is executed.

### Step 1: Parse Command Line Arguments

The `main()` function starts by parsing the command-line arguments you provided using Python's `argparse` library.

```python
# Simplified snippet from autoviz.py's main()
def main():
    # Parse command line arguments
    args = parse_command_line() # parse_command_line is defined in autoviz.py
    # args is now an object like: Namespace(sources=['gridded'], config=['/path/to/my/config_dir'], ...)
    print(f"Parsed arguments: {args}")

    # ... rest of main()
```

The `parse_command_line()` function (also in `autoviz.py`) understands all the options like `-s`, `-c`, `-f`, `--file`, `--vars`, etc. It returns an `argparse.Namespace` object containing the values you specified.

### Step 2: Set Up Logging and Handle Special Modes

Before creating the `Autoviz` instance, the `main()` function sets up logging based on `-v` and `-l` arguments and checks for special modes like `--file` (which triggers metadata dumping instead of plotting) or `--composite`.

```python
# Continuing simplified main() snippet from autoviz.py
    # ... (after parsing args) ...

    # Handle special modes like --file (for metadata dumping)
    if args.file:
        # Code here calls metadump.py instead of Autoviz
        print("Detected --file argument, running metadump...")
        subprocess.run(['python', 'metadump.py', args.file[0]])
        sys.exit() # Exit after dumping metadata

    # Set up logging based on args.verbose and args.log
    print("Setting up logging...")
    logger_setup('autoviz', log=args.log, verbose=args.verbose)
    # Now print statements will go through the logger

    # ... rest of main()
```

If you used `--file`, the script stops here after running a different tool (`metadump.py`). Otherwise, it proceeds to set up the logger and prepare to create `Autoviz`.

### Step 3: Create the Autoviz Instance

This is the core step. The `main()` function creates an instance of the `Autoviz` class.

```python
# Continuing simplified main() snippet from autoviz.py
    # ... (after logging setup) ...

    # The user can specify multiple sources separated by commas (e.g., -s wrf,lis)
    input_sources = [s.strip() for s in args.sources[0].split(',')]

    # Create an Autoviz instance for each source (simplified here for one source)
    source = input_sources[0]
    print(f"Creating Autoviz instance for source: {source}")
    # The Autoviz class expects a list of source names
    autoviz = Autoviz([source], args=args)
    print(f"Autoviz object created: {type(autoviz)}")

    # ... Call autoviz.run() next ...
```

The `Autoviz` class constructor (`__init__`) takes a list of source names (like `['gridded']`) and the parsed `args` object. The `__post_init__` method of the `Autoviz` class then performs the main setup:

1.  **Creates the Config Manager:** It calls a helper function `create_config(args)` (defined in `eviz/lib/autoviz/base.py`) to build the [Config Manager](02_config_manager_.md). As we learned in [Chapter 2](02_config_manager_.md) and [Chapter 3](03_yaml_parser_.md), `create_config` uses the [YAML Parser](03_yaml_parser_.md) to read the configuration files specified in `args` (either by `-f` or `-c`).
    ```python
    # Simplified snippet from eviz/lib/autoviz/base.py's __post_init__
    def __post_init__(self):
        self.logger.info("Autoviz initialization")
        # ... (handle default args if none provided) ...

        # 1. Get factories for the specified source names
        self.factory_sources = get_factory_from_user_input(self.source_names)
        if not self.factory_sources:
            raise ValueError(f"No factories found for sources: {self.source_names}")
        print(f"Found factories: {[type(f) for f in self.factory_sources]}")

        # 2. Create and initialize the ConfigManager
        self._config_manager = create_config(self.args)
        print(f"Config Manager created: {type(self._config_manager)}")
        # The ConfigManager now holds all settings from the YAML files
        # and has initialized its sub-configs (InputConfig, OutputConfig, etc.)

        # TODO: enable processing of S3 buckets
    ```
2.  **Identifies Factories:** It calls `get_factory_from_user_input(self.source_names)` to determine *which* kind of specific data handling and plotting logic is needed for the specified source(s) (like `GriddedSourceFactory` for 'gridded' or `WrfFactory` for 'wrf'). These factories will be used later to create the actual data "Models" (which we'll see in Chapter 6).
    ```python
    # Simplified snippet from eviz/lib/autoviz/base.py's get_factory_from_user_input
    def get_factory_from_user_input(inputs) -> list:
        """
        Return factory classes associated with user input sources.
        """
        mappings = {
            "gridded": GriddedSourceFactory(),
            "wrf": WrfFactory(),
            "omi": OmiFactory(),
            # ... other mappings ...
        }
        factories = []
        for i in inputs:
            if i not in mappings:
                print(f"ERROR: '{i}' is not a valid source name...")
                sys.exit(1)
            factories.append(mappings[i])
        return factories
    ```
    This function simply looks up the source name (like 'gridded') in a predefined dictionary to find the correct Factory class (like `GriddedSourceFactory`).

After `__post_init__` finishes, the `autoviz` object is fully set up with the configuration loaded and the correct factories identified.

### Step 4: Run the Visualization Process

Finally, the `main()` function calls the `run()` method on the created `autoviz` instance.

```python
# Continuing simplified main() snippet from autoviz.py
    # ... (after creating autoviz instance) ...

    print("Calling autoviz.run()...")
    autoviz.run()
    print("autoviz.run() finished.")

    end_time = time.time()
    print(f"Total time taken = {timer(start_time, end_time)}")
```

The `Autoviz.run()` method is where the main action happens. It orchestrates the workflow:

1.  **Check Input Files:** It calls `_check_input_files()` to make sure the files listed in the configuration actually exist, logging warnings for any that don't.
    ```python
    # Simplified snippet from Autoviz.run()
    def run(self):
        self.logger.info("Executing Autoviz.run()")
        _start_time = time.time()
        self._config_manager.input_config.start_time = _start_time

        # 1. Check input files
        self._check_input_files()
        # Logs warnings if files are missing

        # ... rest of run()
    ```
2.  **Process Configuration (using Adapter):** It creates a `ConfigurationAdapter` and calls its `process_configuration()` method. This is a crucial step where the configuration is prepared for the data processing "pipeline". *Importantly, the Data Pipeline itself is often initiated or run implicitly as part of this configuration processing step.* We'll dive into the Data Pipeline in the next chapter.
    ```python
    # Continuing simplified Autoviz.run() snippet
    def run(self):
        # ... (before) ...

        # 2. Create a ConfigurationAdapter
        self.config_adapter = ConfigurationAdapter(self._config_manager)
        print(f"ConfigurationAdapter created: {type(self.config_adapter)}")

        # 3. Process the configuration - this triggers the pipeline setup/run
        self.logger.info("Processing configuration using adapter")
        self.config_adapter.process_configuration()
        print("Configuration processing complete.")
        # After this step, data should be loaded by the pipeline

        # ... rest of run()
    ```
3.  **Retrieve Data Sources:** It tries to get the loaded data sources (the `xarray.Dataset` objects from [Chapter 1](01_data_source_.md)) from the pipeline that was run in the previous step.
    ```python
    # Continuing simplified Autoviz.run() snippet
    def run(self):
        # ... (after processing config) ...

        # 4. Get loaded data sources from the pipeline
        all_data_sources = {}
        try:
             # Assumes the pipeline is available via the config manager after process_configuration
            all_data_sources = self._config_manager._pipeline.get_all_data_sources()
            print(f"Retrieved {len(all_data_sources)} data sources.")
        except Exception as e:
             self.logger.error(f"Error accessing pipeline or data sources: {e}")
             print(f"Error retrieving data: {e}")

        if not all_data_sources:
            self.logger.error("No data sources were loaded. Cannot proceed.")
            print("ERROR: No data sources loaded. Check inputs and config.")
            return # Stop if no data loaded

        # ... rest of run()
    ```
4.  **Trigger Model Execution:** For each identified source (using the `factory_sources`), it creates a specific "model" instance (like `GriddedSource` or `WrfSource`). These model instances represent the data for one source *and* contain the logic for plotting and analysis specific to that source type. `Autoviz` then passes the list of plotting tasks (`map_params` from the [Config Manager](02_config_manager_.md), originally from the [YAML Parser](03_yaml_parser_.md)) to the model and calls the model instance (using `model()`), which starts the plotting loop.
    ```python
    # Continuing simplified Autoviz.run() snippet
    def run(self):
        # ... (after getting data sources) ...

        # 5. Trigger plotting for each source/model
        for factory in self.factory_sources:
            print(f"Using factory {type(factory)} to create model...")
            # Create the specific model instance (e.g., GriddedSource, WrfSource)
            model = factory.create_root_instance(self._config_manager)
            print(f"Created model: {type(model)}")

            # Pass the list of plot tasks to the model
            if hasattr(model, 'set_map_params') and self._config_manager.map_params:
                print(f"Setting map_params on model ({len(self._config_manager.map_params)} tasks).")
                model.set_map_params(self._config_manager.map_params)
            else:
                self.logger.warning("Model doesn't support map_params or no tasks defined.")

            # Execute the model - this triggers the plotting process!
            print("Executing model() to start plotting...")
            model() # This calls the __call__ method of the model instance

        print("All models executed.")

        # ... (cleanup like config_adapter.close()) ...
    ```

This final step, `model()`, is where the visualization logic actually gets initiated. The model knows how to use the loaded data, the configuration settings (including `map_params`), and the plotting backends ([Chapter 8: Plotter Backend](08_plotter_backend_.md)) to generate the images.

Here's a simple sequence diagram showing the high-level flow when you run `autoviz.py`:

```{mermaid}
sequenceDiagram
    participant Shell as User (Shell)
    participant MainScript as autoviz.py (main function)
    participant AutovizObj as Autoviz Instance
    participant ConfigMgr as ConfigManager
    participant YAMLParser as YAML Parser
    participant Factories as Source Factories
    participant Adapter as ConfigurationAdapter
    participant Pipeline as Data Pipeline (concept)
    participant ModelObj as Model Instance (e.g., GriddedSource)

    Shell->>MainScript: python autoviz.py -s ... -c ...
    MainScript->>MainScript: parse_command_line()
    MainScript->>MainScript: logger_setup()
    MainScript->>AutovizObj: Create Autoviz([source], args)
    AutovizObj->>AutovizObj: __post_init__()
    AutovizObj->>Factories: get_factory_from_user_input([source])
    Factories-->>AutovizObj: return [Factory instance]
    AutovizObj->>ConfigMgr: create_config(args)
    ConfigMgr->>YAMLParser: Parse config files
    YAMLParser-->>ConfigMgr: return parsed data
    ConfigMgr-->>AutovizObj: return ConfigManager instance
    AutovizObj-->>MainScript: return Autoviz instance

    MainScript->>AutovizObj: run()
    AutovizObj->>AutovizObj: _check_input_files()
    AutovizObj->>Adapter: Create ConfigurationAdapter(ConfigMgr)
    AutovizObj->>Adapter: process_configuration()
    Adapter->>Pipeline: Setup & Run Pipeline (uses ConfigMgr)
    Pipeline-->>Adapter: Data loaded/processed
    Adapter-->>AutovizObj: Processing finished (pipeline available via ConfigMgr)
    AutovizObj->>Pipeline: get_all_data_sources() (via ConfigMgr)
    Pipeline-->>AutovizObj: return loaded data sources
    AutovizObj->>Factories: create_root_instance(ConfigMgr) for each factory
    Factories-->>AutovizObj: return Model Instance
    AutovizObj->>ModelObj: set_map_params(...)
    AutovizObj->>ModelObj: Call model() (__call__)
    ModelObj->>ModelObj: Orchestrates plotting using data, config, plotters
    ModelObj-->>AutovizObj: Plotting complete for this model

    AutovizObj->>Adapter: close() (cleanup)
    AutovizObj-->>MainScript: run() complete
    MainScript-->>Shell: Exit / print time taken
```

This diagram illustrates how the `Autoviz` object is created, sets up the configuration and identifies the necessary tools (factories), triggers the processing via the adapter/pipeline, and finally initiates the plotting process by executing the specific model instances.

## Summary

In this chapter, we looked at the **Autoviz** class, the main orchestrator of the eViz automatic visualization application.

*   It's the primary entry point when running `autoviz.py`.
*   It reads command-line arguments to guide its setup.
*   It relies heavily on the [Config Manager](02_config_manager_.md) (which in turn uses the [YAML Parser](03_yaml_parser_.md)) to load all application settings.
*   It identifies the specific data handling "factories" and "models" needed based on the configured data sources.
*   Its `run()` method orchestrates the core process, involving checking files, processing configuration via a `ConfigurationAdapter` (which triggers the Data Pipeline), and finally executing the relevant model instances to generate the visualizations based on the plot tasks defined in the configuration (`map_params`).

`Autoviz` provides the structure and sequence for the entire automatic visualization workflow. Now that we know how the process is started and configured, the next logical step is to understand the central workflow component triggered by `Autoviz`: the **Data Pipeline**.

Let's move on to [Chapter 5: Data Pipeline](05_data_pipeline_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)