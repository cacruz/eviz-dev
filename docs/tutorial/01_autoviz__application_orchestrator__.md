# Chapter 1: Autoviz (Application Orchestrator)

Welcome to the eViz tutorial! This first chapter introduces you to the very heart of the eViz application: **Autoviz**. Think of Autoviz as the conductor of an orchestra, the general manager of a project, or the main switchboard for everything eViz does.

## What is Autoviz and Why Do We Need It?

Imagine you want to create a beautiful visualization (like a map or a graph) from some scientific data, say, climate model outputs. This seemingly simple task involves several steps:

1.  Figure out what type of data you have.
2.  Find the specific data files you need.
3.  Read the data from those files.
4.  Apply any necessary calculations or adjustments to the data.
5.  Decide how the plot should look (colors, titles, axes, etc.).
6.  Actually generate the plot and save it.

Doing all these steps manually for different types of data or different plots can be complicated and repetitive.

This is where **Autoviz** comes in. Its main job is to automate and manage this entire process for you. It's the application's central hub that takes your request (usually given through the command line) and orchestrates all the different parts of eViz to fulfill it.

In short, Autoviz:
*   **Listens** to what you want to do (reads your command).
*   **Plans** how to do it (loads the right rules and settings from configuration files).
*   **Delegates** tasks to the right experts (like finding the component that knows how to read your specific data type).
*   **Oversees** the whole operation from start to finish.

It ties together all the other components of eViz (which you'll learn about in later chapters, like the [ConfigManager](02_configmanager_.md), [DataPipeline](03_datapipeline_.md), and [Plotter](07_plotter_.md)) to get the job done.

### How to Use Autoviz: Your First Command

Since Autoviz is the main entry point, you interact with it primarily through the `autoviz.py` script run from your terminal.

Let's look at a very common way to use it: telling eViz what kind of data you want to visualize.

```bash
python autoviz.py -s gridded
```

This simple command tells the `autoviz.py` script to start the eViz application and focus on data sources categorized as `gridded`.

What happens when you run this?
1.  You launch the `autoviz.py` script using Python.
2.  The script starts up Autoviz.
3.  Autoviz reads your command (`-s gridded`).
4.  Based on `-s gridded`, Autoviz figures out which specific rules (configurations) and data handling tools (factories) are needed for 'gridded' data.
5.  It then uses these tools and rules to find the data (based on configuration), process it, and generate the predefined plots for that data type.

This single command kicks off a whole sequence of operations managed by Autoviz.

### Inside Autoviz: The Conductor's Flow

Let's peek behind the curtain to see the main steps Autoviz takes when you run a command like `python autoviz.py -s gridded`.

Imagine Autoviz as the main boss receiving an order:

1.  **Receive the Order:** The `autoviz.py` script gets your command-line arguments (`-s gridded`).
2.  **Understand the Request:** It parses these arguments to understand exactly what you want to do (which data sources to process, which configuration to use, etc.).
3.  **Find the Right Tools:** Based on the requested data sources (like `gridded`), it identifies and prepares the specialized components ("factories") that know how to handle that specific type of data.
4.  **Load the Blueprint:** It loads the detailed instructions ("configuration") for this task. This includes where to find the data, what variables to look at, how the plots should appear, where to save the results, and much more. This step heavily involves the [ConfigManager](02_configmanager_.md).
5.  **Start the Work:** It initiates the main processing flow. This involves telling the specialized data handling components to read and process the data, and then instructing the plotting components to create the visualizations based on the configuration.
6.  **Oversee and Report:** Autoviz manages the overall workflow, handles potential issues, and eventually finishes the process, often reporting how long it took.

Here’s a simplified sequence diagram showing this interaction:

```{mermaid}
sequenceDiagram
    participant User
    participant CLI as autoviz.py Script
    participant A as Autoviz Instance
    participant CM as ConfigManager
    participant FF as DataSourceFactory
    participant Model as Data/Plotting Logic

    User->>CLI: Run command (e.g., python autoviz.py -s gridded)
    CLI->>A: Create Autoviz instance (pass args)
    A->>FF: Get factory for 'gridded'
    FF-->>A: Return GriddedFactory
    A->>CM: Create ConfigManager (pass args)
    CM-->>A: Return ConfigManager
    A->>A: Process configuration (using ConfigAdapter)
    A->>A: Check input files
    A->>Model: Tell factory to create Model instance (pass config)
    Model-->>A: Return Model instance
    A->>Model: Call Model's run/process method
    Model->>Model: Read data, process, plot
    Model-->>A: Finished
    A-->>CLI: Finished
    CLI-->>User: Display time taken, status
```

This diagram shows how your command triggers a series of actions, orchestrated by the `Autoviz` instance, involving other key parts of the system like the `ConfigManager` and `DataSourceFactory` (which you'll learn about soon!).

### Code Walkthrough (Simplified)

Let's look at tiny snippets from the `autoviz.py` and `eviz/lib/autoviz/base.py` files to see how the steps above are implemented.

First, in `autoviz.py`, the `main()` function is the very first code that runs when you execute the script. It starts by understanding your command:

```python
# autoviz.py
import argparse
# ... other imports ...

def parse_command_line() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(...) # Setup how arguments are expected
    parser.add_argument('-s', '--sources', ...) # Define the -s argument
    # ... define other arguments ...
    args = parser.parse_args()

    # Simple check to make sure we know what to do
    if not args.file and not args.sources:
        parser.error("The --sources argument is required unless --file is specified.")

    return args

def main():
    """Main driver for the autoviz plotting tool."""
    start_time = time.time()
    args = parse_command_line() # Step 2: Understand the Request

    # ... code for --file/metadump.py omitted for clarity ...

    # Process each source if multiple were given (e.g., -s wrf,lis)
    # Simplified here for a single source like -s gridded
    source = args.sources[0] # Get the first source name

    # Step 3, 4, 5: Find Tools, Load Blueprint, Start Work
    # This happens *inside* the Autoviz class
    autoviz = Autoviz([source], args=args) # Create the Autoviz instance
    autoviz.run() # Tell Autoviz to start the process!

    print(f"Time taken = {timer(start_time, time.time())}")

if __name__ == "__main__":
    main()
```

This snippet shows that `main()` calls `parse_command_line()` to get your instructions (`args`), then creates an `Autoviz` object, and finally tells that object to `run()`.

Now, let's look at the `Autoviz` class itself in `eviz/lib/autoviz/base.py`.

```python
# eviz/lib/autoviz/base.py
import logging
# ... other imports ...
from eviz.lib.config.config_manager import ConfigManager
from eviz.models.root_factory import GriddedFactory, WrfFactory # Example factories
# ... other factories ...

def get_factory_from_user_input(inputs) -> list:
    """Return factory classes associated with user input sources."""
    mappings = {
        "gridded": GriddedFactory(), # Map 'gridded' to GriddedFactory
        "wrf": WrfFactory(), # Map 'wrf' to WrfFactory
        # ... other mappings ...
    }
    factories = []
    for i in inputs: # For each source name provided (-s gridded wrf)
        if i not in mappings:
             # Handle unknown source
             pass # Simplified error handling
        factories.append(mappings[i]) # Get the corresponding factory
    return factories

def create_config(args) -> ConfigManager:
    """Create a ConfigManager instance from command-line arguments."""
    # This function loads configuration based on the args and source names.
    # It involves finding config files, reading them, and setting up sub-configs.
    # We will cover this in detail in the next chapter!
    print("Autoviz is loading your configuration...") # Simplified
    config_manager = ConfigManager(...) # Actual complex logic here
    return config_manager


@dataclass # Makes it easy to create objects with these properties
class Autoviz:
    """Main class for automatic visualization."""
    source_names: list # The list of sources you provided (-s gridded)
    args: Namespace = None
    # ... other attributes ...

    def __post_init__(self):
        """Initialize the Autoviz instance."""
        self.logger.info("Start init")
        # If not run from CLI, create default args (e.g., in a script)
        if not self.args:
             self.args = Namespace(...) # Default args
             
        # Step 3: Find the Right Tools
        self.factory_sources = get_factory_from_user_input(self.source_names)
        if not self.factory_sources:
            raise ValueError(...) # Handle no factories found

        # Step 4: Load the Blueprint
        self._config_manager = create_config(self.args) # Use the helper function

    def run(self):
        """Execute the visualization process (Step 5 starts here)."""
        self._config_manager.input_config.start_time = time.time()

        self._check_input_files() # Check if data files exist

        # Prepares config for use by other components
        self.config_adapter = ConfigurationAdapter(self._config_manager)
        self.config_adapter.process_configuration() # Processes the blueprint

        all_data_sources = {}
        try:
             # Get the data sources prepared by the config pipeline
             if hasattr(self._config_manager, '_pipeline') and self._config_manager._pipeline is not None:
                all_data_sources = self._config_manager._pipeline.get_all_data_sources()
             # ... simplified error handling ...
        except Exception as e:
             self.logger.error(f"Error accessing pipeline: {e}")

        if not all_data_sources:
            self.logger.error("No data sources were loaded. Check files.")
            # ... display missing files ...
            return # Stop if no data

        # Handle special composite plotting request if any
        if hasattr(self.args, 'composite') and self.args.composite:
             # ... logic for composite plots ...
             return

        # Now, tell each factory to create the main "Model" object
        # and run it! The Model object handles the actual data processing and plotting.
        for factory in self.factory_sources:
            self.logger.info(f"Using factory: {type(factory).__name__}")
            # This creates the main object responsible for reading data and plotting
            model = factory.create_root_instance(self._config_manager)

            # Potentially set up map parameters if needed
            if hasattr(model, 'set_map_params') and self._config_manager.map_params:
                 model.set_map_params(self._config_manager.map_params)

            # THIS IS WHERE THE MAGIC HAPPENS! Tell the model to do its work.
            model() # This calls the __call__ method on the model object


    # ... other methods like set_data, _check_input_files, set_output ...
```

This simplified code shows that when an `Autoviz` object is created (`__post_init__`), it first finds the right data `factory` using `get_factory_from_user_input` and then loads the project `configuration` using `create_config`. When `autoviz.run()` is called, it processes the configuration using a `ConfigurationAdapter`, checks for data files, and then loops through the selected factories, asking each one to create a `model` object (the main worker for that data type) and then telling that model to execute its tasks (`model()`).

### Conclusion

In this chapter, we learned that **Autoviz** is the central orchestrator of the eViz application. It's the first thing that runs when you execute `autoviz.py` and it's responsible for understanding your request, loading the necessary configuration, getting the right tools ready (factories), and kicking off the data processing and visualization workflow.

It acts as the bridge between your command and the complex internal operations of eViz.

Now that you know how Autoviz starts the process and relies on a "blueprint", the next logical step is to understand how that blueprint is managed. In the next chapter, we will dive into the **[ConfigManager](02_configmanager_.md)**, which is crucial for telling Autoviz and other components how to behave.

[Next Chapter: ConfigManager](02_configmanager_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)