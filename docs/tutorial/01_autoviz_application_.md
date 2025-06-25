# Chapter 1: Autoviz Application

Welcome to the first chapter of the eViz tutorial! This is where we'll get introduced to the main character of our story: the **Autoviz Application**. Think of this as the central hub or the main program you interact with to create beautiful visualizations from your scientific data using eViz.

## What is the Autoviz Application?

Imagine you have a bunch of scientific data files (like weather model outputs, satellite measurements, etc.) and you want to turn them into plots and maps automatically. You don't want to manually write code for every single file or every single type of data. That's exactly what eViz is designed for!

The **Autoviz Application** is the part of eViz that takes your instructions (like "process this type of data", "use these settings", "save the pictures here") and makes it all happen. It's the top-level controller, the conductor of the entire visualization orchestra.

Here's a simple way to think about it:

*   **You:** The user, who wants visualizations.
*   **Autoviz Application:** The helpful assistant you talk to. You tell it what you need using commands or configuration files.
*   **Other eViz components:** The specialized tools and experts the assistant uses behind the scenes (like data readers, processors, plotters, etc.).

The Autoviz Application's main job is to understand what you want, set up everything correctly, bring the right tools together, and tell them to start creating the visualizations.

## Our First Use Case: Running a Simple Visualization

Let's start with the most basic task you might want to do with eViz: tell it to visualize data from a specific *source* using some predefined settings.

A *source* in eViz terminology refers to a type of data, often related to a specific model or observation platform (like 'gridded' for generic grid-based data, 'wrf' for WRF model output, 'omi' for OMI satellite data, etc.). eViz comes with default settings for many common sources.

So, our goal is to tell the Autoviz Application: "Please visualize data for the 'gridded' source."

How do we do that? We use the main program file, `autoviz.py`, directly from your command line (like your terminal or command prompt).

```bash
python autoviz.py -s gridded
```

This simple command is the most common way to kick off the Autoviz Application. Let's break down what's happening:

*   `python autoviz.py`: This tells your computer to run the Python script named `autoviz.py`.
*   `-s gridded`: This is an *argument* you're passing to the script. `-s` is a short way of saying `--sources`, and `gridded` is the specific source name you want to process.

When you run this command, the Autoviz Application (specifically, the code inside `autoviz.py` and the `Autoviz` class) springs into action.

## How Autoviz.py Gets Started

Let's peek at the beginning of the `autoviz.py` file to see how it handles your command.

```python
# --- File: autoviz.py ---
import sys
import time
import subprocess
import argparse

# ... other imports and functions ...

def main():
    """
    Main driver for the autoviz plotting tool.
    """
    start_time = time.time() # Remember when we started
    args = parse_command_line() # Read your command line arguments

    # --- Simplified metadump check (we'll skip this for now) ---
    # if args.file:
    #     subprocess.run(...)
    #     sys.exit()
    # ----------------------------------------------------------

    # Setup logging so we can see messages
    verbose = int(args.verbose[0] if isinstance(args.verbose, list) else '1')
    log = int(args.log[0] if isinstance(args.log, list) else '1')
    logger_setup('autoviz', log=log, verbose=verbose)

    # Convert comma-separated sources (if any) into a list
    input_sources = [s.strip() for s in args.sources[0].split(',')]

    # The core action: Create and run the Autoviz object!
    for source in input_sources:
        print(f"Processing source: {source}")
        # Create a temporary args for this specific source
        source_args = argparse.Namespace(**vars(args))
        source_args.sources = [source]

        # THIS is where the main work is delegated
        autoviz = Autoviz([source], args=source_args)
        autoviz.run() # Tell the Autoviz object to do its job

    print(f"Time taken = {timer(start_time, time.time())}")


if __name__ == "__main__":
    main() # When you run 'python autoviz.py', the main() function is called
```

When you run `python autoviz.py -s gridded`:

1.  The special `if __name__ == "__main__":` block at the bottom runs.
2.  It calls the `main()` function.
3.  `main()` first records the start time.
4.  `parse_command_line()` reads your `-s gridded` input and stores it in an `args` object.
5.  Logging is set up so eViz can print helpful messages as it works.
6.  It takes the source names from `args.sources` (in our case, just `['gridded']`).
7.  It loops through each source (just 'gridded' in this example).
8.  Inside the loop, it creates an instance of the `Autoviz` class, passing the source name (`['gridded']`) and the collected arguments (`args`).
9.  Crucially, it then calls the `.run()` method on the `autoviz` object. **This is where the real orchestration begins.**
10. Finally, after the loop finishes, it prints how long the whole process took.

So, while `autoviz.py` is the entry point, the heavy lifting of *managing* the visualization process is done by the `Autoviz` class.

## The Autoviz Class: The Conductor

Let's look at the `Autoviz` class definition in `eviz/lib/autoviz/base.py`. This is the heart of the Autoviz Application concept.

```python
# --- File: eviz/lib/autoviz/base.py ---
import glob
import os
import logging
import time
from typing import Optional
from argparse import Namespace
from dataclasses import dataclass, field

# ... imports for config and factories ...

@dataclass
class Autoviz:
    """
    Main class for automatic visualization... orchestrates the entire process.
    """
    source_names: list # The list of sources you want to process (like ['gridded'])
    args: Namespace = None # The command line arguments you provided
    # ... other attributes and methods ...

    def __post_init__(self):
        """
        Initialize the Autoviz instance after dataclass initialization.
        """
        self.logger.info("Start init")
        # If no args were provided (e.g., creating in a script), set up defaults
        if not self.args:
            self.args = Namespace(...) # Set up default args
        
        # Figure out which specific "factories" are needed for these sources
        # Factories know how to create the right tools for each source type
        self.factory_sources = get_factory_from_user_input(self.source_names)
        
        # Create the configuration manager. This object reads config files
        # and knows ALL the details about what, where, and how to plot.
        self._config_manager = create_config(self.args)

        # ... other initialization ...

    def run(self):
        """
        Execute the visualization process.
        """
        self.logger.info("Start run")
        # ... timing and initial setup ...

        # Check if the required input files actually exist
        self._check_input_files()

        # An adapter helps process the complex configuration details
        self.config_adapter = ConfigurationAdapter(self._config_manager)

        try:
            self.logger.info("Processing configuration using adapter")
            self.config_adapter.process_configuration() # This sets up the pipeline!

            # Get the objects representing the data we loaded based on config
            all_data_sources = self._config_manager._pipeline.get_all_data_sources()

            if not all_data_sources:
                 self.logger.error("No data sources were loaded...")
                 return # Can't plot if no data

            # --- Simplified composite field check (skip for now) ---
            # if hasattr(self.args, 'composite') and self.args.composite:
            #     ... handle composite plotting ...
            #     return
            # ----------------------------------------------------

            # NOW, for each source factory we identified earlier...
            for factory in self.factory_sources:
                # Ask the factory to create the specific "model" object
                # This model object knows how to work with THIS type of data (e.g., gridded)
                model = factory.create_root_instance(self._config_manager)

                # If the model needs map settings from config, give them to it
                # if hasattr(model, 'set_map_params') ... model.set_map_params(...)

                # Tell the model to do its work (loading data, processing, plotting!)
                model() # Calling the model instance executes its main logic

        finally:
            # Clean up resources if necessary
            self.config_adapter.close()
        self.logger.info("Run finished")

    # ... other methods like set_data, _check_input_files, set_output ...
```

When the `Autoviz` object is created (in the `main()` function of `autoviz.py`), its `__post_init__` method runs:

1.  It receives the `source_names` (like `['gridded']`) and the `args` object.
2.  It calls `get_factory_from_user_input` (we'll learn more about [Data Source Factory](04_data_source_factory_.md) later!) to find the right "factory" for the source(s). A factory is like a blueprint creator – it knows *how* to create the specific tools needed for a type of data (like 'gridded'). For 'gridded', it finds `GriddedSourceFactory`.
3.  It calls `create_config` (we'll dive into [Configuration Management](02_configuration_management_.md) next!) to set up the `ConfigManager`. This object reads configuration files (or uses defaults) to figure out all the details: where the data files are, which variables to plot, what kind of plots to make, where to save them, etc.

Then, when `autoviz.run()` is called:

1.  It first checks if the input files specified in the configuration actually exist (`_check_input_files`).
2.  It uses a `ConfigurationAdapter` (a helpful middleman) to process the detailed configuration loaded by the `ConfigManager`. This step is crucial as it sets up the processing steps needed.
3.  It retrieves the data that was loaded and prepared during the configuration processing.
4.  If data was successfully loaded, it loops through the factories it found earlier.
5.  For each factory, it asks the factory to `create_root_instance`. This instance is typically an object that represents the specific *type* of data being processed (like a `GriddedModel` object for the 'gridded' source). This object contains the logic specific to handling that data type and generating plots. We'll explore this in [Source/Model Specific Logic](08_source_model_specific_logic_.md).
6.  It then "calls" this model object (`model()`). This is where the model object executes its main routine: loading data, applying processing steps (from the [Data Processing Pipeline](05_data_processing_pipeline_.md)), creating plots using [Plotter Abstraction](06_plotter_abstraction_.md) and [Plotter Factory](07_plotter_factory_.md)), and saving them.

## Putting It Together (Simple Flow)

Let's visualize the simple flow when you run `python autoviz.py -s gridded`:

```{mermaid}
sequenceDiagram
    participant User
    participant AutovizScript as autoviz.py
    participant AutovizApp as Autoviz
    participant ConfigMgr as ConfigManager
    participant Factory as GriddedSourceFactory
    participant Model as GriddedModel

    User->>AutovizScript: Run 'python autoviz.py -s gridded'
    AutovizScript->>AutovizScript: parse_command_line()
    AutovizScript->>AutovizApp: Create Autoviz(['gridded'], args)
    AutovizApp->>AutovizApp: __post_init__()
    AutovizApp->>Factory: get_factory_from_user_input(['gridded'])
    Factory-->>AutovizApp: Return GriddedSourceFactory
    AutovizApp->>ConfigMgr: create_config(args)
    ConfigMgr-->>AutovizApp: Return ConfigManager
    AutovizApp-->>AutovizScript: Autoviz object created
    AutovizScript->>AutovizApp: run()
    AutovizApp->>AutovizApp: _check_input_files()
    AutovizApp->>ConfigMgr: Process configuration (via adapter)
    ConfigMgr-->>AutovizApp: Data sources available
    AutovizApp->>Factory: create_root_instance(ConfigManager)
    Factory-->>AutovizApp: Return GriddedModel
    AutovizApp->>Model: Call Model instance ()
    Model->>ConfigMgr: Get instructions
    Model->>Model: Load, Process, Plot Data
    Model-->>AutovizApp: Plotting complete
    AutovizApp-->>AutovizScript: Run method finishes
    AutovizScript-->>User: Print time taken & Exit

```

This diagram shows how the `autoviz.py` script acts as the initial handler, but the `Autoviz` object is where the main coordination happens. It relies heavily on the [Configuration Management](02_configuration_management_.md) to know what to do and uses factories ([Data Source Factory](04_data_source_factory_.md), [Plotter Factory](07_plotter_factory_.md)) to get the right tools ([Data Source Abstraction](03_data_source_abstraction_.md), [Plotter Abstraction](06_plotter_abstraction_.md)) and models ([Source/Model Specific Logic](08_source_model_specific_logic_.md)) to perform the actual work.

## Summary

In this first chapter, we learned that the **Autoviz Application** is the main entry point for using eViz. You interact with it primarily by running the `autoviz.py` script from the command line. The `Autoviz` class within eViz acts as the central coordinator. It reads your instructions, loads the necessary configuration, finds the right tools (factories and models) for your specific data source, and tells them to start the process of loading, processing, and visualizing your data.

The Autoviz Application doesn't do the data handling or plotting itself; it's a director that brings together all the other specialized components of eViz to perform the task you requested based on the configuration.

Now that we know the Autoviz Application is the director, the next logical question is: How does the director know *what* movie to make? How does it know which data files to use, which variables are important, what kind of plots are needed, and where to save them? That information comes from the **Configuration Management**.

Ready to see how eViz learns the details of your visualization task? Let's move on!

[Configuration Management](02_configuration_management_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)