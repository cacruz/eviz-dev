# Chapter 1: Autoviz Application

Welcome to the eViz tutorial! This first chapter introduces you to the heart of the eViz system for creating visualizations: the **Autoviz Application**.

Imagine you're directing a large orchestra. You have all sorts of musicians (different data files), instruments (plotting tools), and music sheets (configuration settings). But you need a conductor to bring it all together and make beautiful music (visualizations)!

In eViz, the **Autoviz Application**, specifically the `autoviz.py` script and the `Autoviz` class it uses, acts as that conductor. It's the central control tower that takes your instructions and coordinates all the different parts of the system to produce the final images or plots you want.

## Your First Command: Making eViz Play!

Let's start with a simple use case: you have some standard gridded climate data (like temperature or precipitation) and you want eViz to automatically create some plots for it.

You'll interact with the Autoviz Application primarily through the command line, by running the `autoviz.py` script.

A very basic command to tell eViz to process 'gridded' data would look like this:

```bash
python autoviz.py -s gridded
```

Let's break this down:

*   `python autoviz.py`: This is how you tell your computer to run the main eViz script.
*   `-s gridded`: This is your main instruction to the Autoviz Application. `-s` stands for 'source', and `gridded` tells the application that the data you're interested in comes from a "gridded" source type. eViz knows how to handle different types of data sources, and 'gridded' is a common one.

When you run this command, the Autoviz Application wakes up! What happens next?

## What the Autoviz Application Does (The Conductor's Job)

When you run `python autoviz.py -s gridded`, the script does a few key things behind the scenes:

1.  **Listens to Instructions:** It first reads the command-line arguments you provided (like `-s gridded`).
2.  **Checks for Special Requests:** It checks if you're asking it to do something else, like just extract information *about* a file using `--file` (more on this later). If not, it proceeds to set up for visualization.
3.  **Gathers the Plan (Configuration):** Based on your instructions (like `-s gridded`), it figures out where to find the detailed plans (configuration settings) for processing 'gridded' data. These plans tell eViz *which* data files to look for, *what variables* to plot, *how* the plots should look, and where to save them. (We'll dive deep into [Configuration Management](02_configuration_management_.md) in the next chapter).
4.  **Prepares the Team (Sets up Data Handling):** It gets ready to handle the actual data files according to the configuration.
5.  **Tells the Experts What to Do (Creates the Model):** Based on the 'gridded' source type, it finds the specific part of eViz that knows how to process and plot 'gridded' data. Think of this as assigning the right section of the orchestra (e.g., the string section) to play a specific part.
6.  **Starts the Performance (Runs the Visualization):** It finally tells the expert part (the data model) to go ahead and perform the task – loading data, processing it, and creating the visualizations based on the plan.

Here's a simplified look at this flow:

```{mermaid}
sequenceDiagram
    participant User
    participant A as autoviz.py Script
    participant B as Autoviz Object
    participant C as Configuration Manager
    participant D as Data Model (e.g., Gridded Model)

    User->>A: Run python autoviz.py -s gridded
    A->>A: Parse command line args
    A->>B: Create Autoviz(sources=['gridded'], args)
    B->>C: Create and load configuration
    B->>B: Check inputs, setup processing
    B->>D: Create Data Model (using Factory - later chapter)
    B->>D: Tell Data Model to Run (run its main method)
    D->>User: Generate and save plots
```

This diagram shows that you, the User, interact with the `autoviz.py` script. The script then creates an `Autoviz` object, which is the main coordinator. The `Autoviz` object loads the configuration and then creates and runs the appropriate `Data Model` for your data type (`Gridded Model` in this case), which actually does the work of making plots.

## A Peek at the Code

Let's look at tiny, simplified parts of the `autoviz.py` file (from the `autoviz.py` and `eviz/lib/autoviz/base.py` files in the project) to see where this happens. Don't worry about understanding every detail – just the main structure.

First, the `autoviz.py` script itself has a `main()` function, which is the starting point:

```python
# --- File: autoviz.py (Simplified) ---
import argparse
import subprocess # Used for metadump.py

from eviz.lib.autoviz.base import Autoviz # Import the main class

def parse_command_line():
    # ... (code to set up arguments like -s, --file) ...
    parser = argparse.ArgumentParser(...)
    parser.add_argument('-s', '--sources', ...) # Defines -s
    parser.add_argument('--file', ...)       # Defines --file
    # ... more arguments ...
    args = parser.parse_args()
    return args

def main():
    args = parse_command_line() # 1. Get instructions

    # 2. Check for special request (--file)
    if args.file:
        # If --file is used, run the metadump.py script instead
        subprocess.run(['python', 'metadump.py', args.file[0]])
        sys.exit() # Stop here

    # 3, 4, 5, 6. If no --file, proceed with visualization
    # ... (logging setup omitted) ...

    # Parse sources (handles comma-separated like 'wrf,lis')
    input_sources = [s.strip() for s in args.sources[0].split(',')]

    # Process each source (like 'gridded')
    for source in input_sources:
        print(f"Processing source: {source}")
        # Create the main conductor object (Autoviz instance)
        autoviz = Autoviz([source], args=args)
        # Tell the conductor to start the performance!
        autoviz.run()

    # ... (timing info omitted) ...

if __name__ == "__main__":
    main()
```

This simplified `main` function shows the two main paths: either you're just asking about a file (`--file`), in which case it runs `metadump.py`, OR you're asking it to visualize data using `--sources`, in which case it creates an `Autoviz` object and calls its `run()` method.

Now let's look at the `Autoviz` class in `eviz/lib/autoviz/base.py`. This is the actual conductor object.

```python
# --- File: eviz/lib/autoviz/base.py (Simplified) ---
from argparse import Namespace
from dataclasses import dataclass # Used for simple classes
# ... other imports for config and factories ...
from eviz.lib.config.config_manager import ConfigManager

# Helper function to create config (details in next chapter)
def create_config(args) -> ConfigManager:
    # ... code to find and load config files based on args ...
    pass # We'll see this in detail in Chapter 2

# Helper function to get the right "factory" (details in later chapter)
def get_factory_from_user_input(inputs) -> list:
    # ... code to map source names ('gridded') to the right factory ...
    pass # We'll see this in detail in Chapter 4

@dataclass # Makes creating this object easier
class Autoviz:
    """
    Main class for automatic visualization... orchestrates the entire process.
    """
    source_names: list # e.g., ['gridded']
    args: Namespace = None # The command line arguments

    # ... other attributes and methods ...

    def __post_init__(self):
        """Initial setup after creating Autoviz object."""
        # 3. Gathers the Plan (Configuration)
        self._config_manager = create_config(self.args)

        # 5. Tells the Experts What to Do (Gets the right factory)
        self.factory_sources = get_factory_from_user_input(self.source_names)
        if not self.factory_sources:
             # Handle error if source name is unknown
            raise ValueError(...)


    def run(self):
        """
        Execute the visualization process.
        """
        # ... (timing and file checks omitted) ...

        # 4. Prepares the Team (Uses ConfigurationAdapter)
        # This adapter helps process the config for the pipeline
        self.config_adapter = ConfigurationAdapter(self._config_manager)
        self.config_adapter.process_configuration()


        # 6. Starts the Performance!
        # Now, use the factories to create the specific model(s)
        # and tell each model to run its visualization process.
        for factory in self.factory_sources:
            # Use the factory to create the specific model (e.g., GriddedModel)
            model = factory.create_root_instance(self._config_manager)
            # Tell the model to execute its steps (load data, process, plot)
            model() # Calling model() is like calling its main execution method

        # ... (cleanup omitted) ...

```

This simplified `Autoviz` class shows how it uses the command-line arguments (`self.args`) and the `source_names` (like `['gridded']`) to load the configuration (`create_config`) and find the right "factory" (`get_factory_from_user_input`). The factory is like a specialized builder that knows how to create the specific type of data model needed for 'gridded' data. Finally, the `run()` method orchestrates the main process: preparing the configuration using the `ConfigurationAdapter` (details on this and the [Data Processing Pipeline](06_data_processing_pipeline_.md) will come later) and then looping through the created models, telling each one to execute itself (`model()`), which triggers the actual data loading, processing, and [Plotting Engine](07_plotting_engine_.md) calls.

So, when you run `python autoviz.py -s gridded`, the Autoviz Application:
1.  Reads `-s gridded`.
2.  Creates an `Autoviz` object.
3.  The `Autoviz` object loads the 'gridded' configuration.
4.  The `Autoviz` object gets the 'gridded' factory.
5.  The `Autoviz` object uses the factory to create a 'Gridded Model' instance.
6.  The `Autoviz` object calls the `run()` method of that 'Gridded Model' instance, which then proceeds to find the data files specified in the configuration, process them, and create plots as defined by the configuration.

## Conclusion

In this first chapter, you've learned that the **Autoviz Application**, initiated by running `autoviz.py`, is the main entry point and coordinator for the eViz system. It takes your high-level instructions (like which data source type to use), loads the necessary configuration, and directs the appropriate parts of the system (like data models and plotting engines) to perform the visualization task.

The next crucial piece of the puzzle is understanding *where* Autoviz gets the detailed instructions on *what* specifically to plot, from *which* files, and *how* it should look. This is handled by the system's configuration.

Ready to learn how Autoviz gets its plan? Let's move on to the next chapter: [Configuration Management](02_configuration_management_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
