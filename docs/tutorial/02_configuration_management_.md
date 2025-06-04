# Chapter 2: Configuration Management

Welcome back! In the [previous chapter](01_autoviz_application_.md), we met the **Autoviz Application**, the "conductor" that orchestrates the entire visualization process in eViz. We saw how a simple command like `python autoviz.py -s gridded` tells the conductor which type of data source to work with.

But where does the conductor get the detailed instructions? Where does it learn *which* specific data file to use, *what variables* inside that file are interesting, *how* to draw the plots (what colors, what contour levels), and *where* to save the final images?

Imagine the conductor getting a huge box of sheet music! This "sheet music" for eViz comes in the form of **Configuration Files**. These files are like the detailed blueprint or instruction manual for your visualization task.

This chapter is all about **Configuration Management** – the part of eViz that handles reading, understanding, and using these crucial instruction files.

## Why Do We Need Configuration Files?

Think about building something complex, like a house. You don't just tell the builder "build me a house." You provide blueprints, material lists, paint colors, and landscaping plans.

Similarly, for eViz to create plots, it needs detailed instructions:

*   What is the **input data file**? (e.g., `my_simulation_output.nc`)
*   Which **variables** should be plotted? (e.g., 'temperature', 'precipitation_flux')
*   What **type** of plot for each variable? (e.g., a map, a time series)
*   What **settings** for the plot? (e.g., contour levels, color map, title)
*   Where should the **output images** go? (e.g., into a folder named `my_plots`)

If these instructions were hardcoded directly into the eViz program, you'd have to change the code every time you wanted to plot a different file, use different settings, or save to a different place. That would be incredibly tedious and error-prone!

**Configuration Management solves this** by putting all these instructions into external files that you can easily create, read, and modify *without* changing the eViz code itself.

## eViz Uses YAML Files for Configuration

eViz uses files written in **YAML** (which stands for "YAML Ain't Markup Language") for its configuration. YAML is popular because it's designed to be easy for humans to read and write, while also being easy for computers to parse.

YAML files use indentation and simple `key: value` pairs to structure data. They often end with `.yaml` or `.yml`.

Here's a tiny, made-up example of what a very simple eViz-like configuration snippet might look like:

```yaml
# This is a comment in YAML
outputs:
  output_dir: "my_latest_plots" # Where to save images
  print_format: "png"         # Save as PNG files

inputs:
  - name: "/path/to/my/data/file.nc" # The main data file
    to_plot:
      temperature: "xy,xt" # Plot temperature as a map (xy) and time series (xt)
      pressure: "xy"       # Plot pressure just as a map
```

This little snippet tells eViz:
*   Save output files into a directory called `my_latest_plots`.
*   Save them as PNG images.
*   Process one input file: `/path/to/my/data/file.nc`.
*   For that file, plot the variable 'temperature' using both 'xy' (map) and 'xt' (time series) plot types.
*   Also plot the variable 'pressure' using the 'xy' (map) plot type.

You can see it's structured with main sections like `outputs` and `inputs`, and then details nested inside.

## How Does Autoviz Use Configuration?

When you run `python autoviz.py -s gridded`, the **Autoviz Application** (specifically the `Autoviz` class) doesn't *guess* what to do. Instead, its first major step is to load the configuration.

Based on the `-s gridded` argument, the system knows to look for configuration settings relevant to 'gridded' data. This often involves loading default configuration files that are part of the eViz project, perhaps located in a `config/` directory, and potentially overriding parts of that configuration with settings from other files or command-line arguments.

The `Autoviz` object uses a dedicated component called the **ConfigManager** (from `eviz/lib/config/config_manager.py`) to handle finding, loading, and combining all these configuration pieces.

Let's revisit the sequence diagram from Chapter 1 and add the Configuration Management step:

```{mermaid}
sequenceDiagram
    participant User
    participant A as autoviz.py Script
    participant B as Autoviz Object
    participant C as ConfigManager
    participant Y as YAML Files
    participant D as Data Model (later)

    User->>A: Run python autoviz.py -s gridded
    A->>A: Parse command line args
    A->>B: Create Autoviz(sources=['gridded'], args)
    B->>C: Create ConfigManager
    C->>Y: Read default and user YAML files
    C-->>B: Provide combined configuration
    B->>B: Use config to set up processing
    B->>D: Create Data Model (using config)
    B->>D: Tell Data Model to Run (using config details)
    D->>User: Generate and save plots (based on config)
```

As you can see, the `Autoviz` object relies heavily on the `ConfigManager` to get the complete set of instructions from the YAML files *before* it starts setting up the data processing and plotting.

## Creating Configuration Files with `metadump.py`

Manually writing YAML files, especially for large datasets with many variables, can be time-consuming and error-prone. Fortunately, eViz provides a helper tool specifically for this: `metadump.py`.

`metadump.py` is designed to inspect standard data files (like NetCDF files often used for gridded data), figure out what variables and dimensions they contain, and automatically generate initial YAML configuration files based on that information.

Remember in Chapter 1, we saw that if you run `autoviz.py` with the `--file` argument, it actually runs `metadump.py`?

```bash
python autoviz.py --file /path/to/my/data/file.nc
```

This command essentially tells eViz: "Don't make plots, just inspect this file using the `metadump` tool." By default, `metadump.py` will print information about the file and plottable variables to your screen.

However, the real power of `metadump.py` is generating the YAML files you need. You can tell it to create the main "application" file (`.yaml`) and a separate "specifications" file (`_specs.yaml`).

Here's an example command to use `metadump.py` to generate config files for a NetCDF file:

```bash
python metadump.py /path/to/your/data/file.nc --app my_data.yaml --specs my_data_specs.yaml
```

*   `python metadump.py`: Runs the tool.
*   `/path/to/your/data/file.nc`: The data file you want to inspect.
*   `--app my_data.yaml`: Tells it to generate the main application configuration file and name it `my_data.yaml`. This file will contain settings like the input file path, output directory, and which variables from the file to plot (just listing their names).
*   `--specs my_data_specs.yaml`: Tells it to generate the specifications configuration file and name it `my_data_specs.yaml`. This file will contain variable-specific details like units and default plot settings (e.g., initial contour levels).

After running this, you'll have `my_data.yaml` and `my_data_specs.yaml` files that you can then open, review, and edit to customize your plots before running `autoviz.py` with these configuration files. (How `autoviz.py` is *specifically* told to use *these* files depends on command line arguments handled by `create_config` within `ConfigManager`, but the core idea is that `metadump.py` creates the files, and `ConfigManager` loads them).

## Inside Configuration Management: Reading YAML

The core task of the Configuration Management system is reading the YAML files. This is handled by functions, many of which live in the `eviz/lib/utils.py` file we saw briefly in the last chapter.

The `yaml` Python library is used for this. The `load_yaml_simple` function in `utils.py` is a straightforward example:

```python
# --- File: eviz/lib/utils.py (Simplified) ---
import yaml
import os

def load_yaml_simple(file_path: str) -> dict:
    """Load a YAML file."""
    if not os.path.exists(file_path):
        # It's good practice to check if the file exists!
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with open(file_path, 'r') as file:
        # yaml.safe_load is the standard way to parse YAML safely
        config = yaml.safe_load(file)
        # eViz also often expands environment variables like $HOME
        config = expand_env_vars(config)
        return config

def expand_env_vars(obj):
    # ... (code to look for strings like $HOME or ${MY_VAR} and replace them) ...
    pass # This lets you use environment variables in your YAML paths!

# ... other functions ...
```

This function simply takes a file path, opens the file, uses `yaml.safe_load` to turn the YAML content into a Python dictionary, and then applies a function (`expand_env_vars`) to replace any environment variables found within the text. The result is a Python dictionary that mirrors the structure of your YAML file.

The `ConfigManager` (`eviz/lib/config/config_manager.py`) uses functions like this to load one or more YAML files. It's responsible for finding the right files based on the source name and command-line arguments, potentially loading multiple files (e.g., a base config and then an override config), and merging their settings into a single, master configuration dictionary that the rest of the system can easily access.

## Key Information Stored in Configuration

While the full structure can be complex, the configuration files typically contain sections like:

*   **`inputs`**: Defines the data files to process, their types, and which variables from each file should be considered for plotting. This is where you'd specify the path to `my_simulation_output.nc`.
*   **`outputs`**: Specifies where to save plots, the desired image formats (PNG, PDF), and other output-related settings.
*   **`system_opts`**: Controls internal eViz behavior, like whether to use parallel processing.
*   **Variable Specifications**: Detailed settings for individual variables and plot types, defining things like color maps, contour levels, titles, units, etc. This is often stored in a separate specifications file generated by `metadump.py`.

By changing values in these YAML files, you completely control how eViz runs and what plots it generates without touching the core code.

## Conclusion

In this chapter, you learned that **Configuration Management** is the system responsible for taking your instructions from YAML files and making them available to the eViz **Autoviz Application**. These files act as the detailed plan, specifying everything from input data paths to plot appearance. You also saw how the `metadump.py` tool can help you get started by automatically generating initial configuration files based on your data.

Understanding configuration is key because it's how you customize eViz's behavior for your specific data and visualization needs.

Now that we know *where* the instructions come from, the next logical step is to understand how eViz handles the actual data files specified in these instructions. Let's move on to the next chapter: [Data Source Abstraction](03_data_source_abstraction_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
