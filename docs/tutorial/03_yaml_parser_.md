# Chapter 3: YAML Parser

Welcome back to the eViz tutorial! In the last chapter, [Chapter 2: Config Manager](02_config_manager_.md), we learned that the Config Manager is the central brain of eViz, holding all the settings and instructions for your visualization tasks. But where does the Config Manager get those instructions from? It gets them from configuration files you provide, typically written in a format called YAML.

This is where the **YAML Parser** comes in.

## What Problem Does YAML Parser Solve?

You write your eViz configuration in human-readable YAML files. These files look like structured notes with key-value pairs, lists, and nested sections – easy for you to understand and edit.

```yaml
# Example snippet from a config file
outputs:
  output_dir: ./my_plots
  print_format: png

inputs:
  - name: temperature_data.nc
    exp_id: control
    to_plot:
      temperature: xy
```

While you can read this, your computer program needs this information in a format it can process easily, like Python dictionaries and lists. The raw text in the YAML file needs to be translated into this structured data.

The **YAML Parser** is the dedicated component that performs this translation. Its job is to read the text from one or more YAML configuration files, understand their structure, and convert it into Python data structures that eViz's [Config Manager](02_config_manager_.md) (and other parts of the application) can use.

Think of it as a language interpreter. It reads the blueprint written in the "YAML language" and converts it into step-by-step instructions and data structures the application's "brain" ([Config Manager](02_config_manager_.md)) can follow.

## Our Central Use Case: Reading Configuration Files

The main task of the YAML Parser is straightforward: take a list of paths to configuration files and produce the structured data contained within them.

Let's use the example from the previous chapter, where you have a configuration file like `my_settings.yaml`.

## Creating and Using the YAML Parser (Simplified)

In the real eViz application, the [Config Manager](02_config_manager_.md) creates and uses the `YAMLParser` internally. You usually don't interact with the parser directly. However, to understand it, let's look at a simplified example of how it would be used:

```python
# Imagine this simplified code is inside the Config object (used by Config Manager)
from eviz.lib.config.yaml_parser import YAMLParser

# Assume these come from application inputs
config_file_paths = ["path/to/your/my_settings.yaml"]
source_names_list = ["my_data_source"] # Often related to experiments/models

print(f"Initializing YAML Parser for files: {config_file_paths}")

# 1. Create an instance of the YAMLParser
# You give it the paths to the main config files and source names
yaml_parser = YAMLParser(
    config_files=config_file_paths,
    source_names=source_names_list
)
print(f"YAML Parser object created: {type(yaml_parser)}")
```

In this code:
*   We import the `YAMLParser` class.
*   We create an instance, providing the list of configuration file paths (`config_files`) and the corresponding list of `source_names` (these lists are often linked, where each file corresponds to a source/model).

Now that we have the `yaml_parser` object, it's ready to read and process the files.

## Parsing the Files

To actually read the content from the files and translate it into structured data, you call the `parse()` method:

```python
# Continuing from the previous example with the yaml_parser object

print("Calling the parse() method...")
# 2. Call the parse() method
# This method reads the files, merges them, and extracts the data
yaml_parser.parse()
print("Parsing complete!")
```

The `parse()` method does the main work. It opens each file listed in `config_files`, reads its YAML content, and processes it. This involves:

*   Reading the main sections like `inputs`, `outputs`, `system_opts`, etc.
*   Handling cases where multiple configuration files are provided (merging their settings).
*   Looking for associated "specs" files (files named like `my_settings_specs.yaml` next to `my_settings.yaml`) which contain detailed plotting parameters for specific variables.
*   Organizing all this information into specific Python data structures.

## Accessing the Parsed Data

After calling `parse()`, the `yaml_parser` object holds the extracted data in various attributes. The most important ones are:

*   `app_data`: A dictionary containing the main application-level settings from the config files (like output directory, input file lists, system options).
*   `spec_data`: A dictionary containing the detailed plotting specifications, usually loaded from the associated `_specs.yaml` files.
*   `map_params`: A structured dictionary (organized numerically) that lists every single plot task eViz needs to perform, derived from the `inputs` and `to_plot`/`variables` sections of your config.

You can access these like any other object attribute:

```python
# Continuing from the previous example after calling parse()

print("\nAccessing parsed data:")

# 3. Access the main application data
print(f"Type of app_data: {type(yaml_parser.app_data)}")
# Example: Print some keys from app_data if it's not empty
if yaml_parser.app_data:
    print(f"Keys in app_data: {list(yaml_parser.app_data.keys())}")
else:
    print("app_data is empty (check config file content).")


# 4. Access the detailed plotting specs
print(f"\nType of spec_data: {type(yaml_parser.spec_data)}")
if yaml_parser.spec_data:
     print(f"Keys in spec_data (e.g., variable names): {list(yaml_parser.spec_data.keys())}")
else:
     print("spec_data is empty (check for _specs.yaml files).")


# 5. Access the organized list of plotting tasks
print(f"\nType of map_params: {type(yaml_parser.map_params)}")
print(f"Number of plot tasks identified: {len(yaml_parser.map_params)}")
# You could inspect the first task:
# if yaml_parser.map_params:
#     first_task_key = list(yaml_parser.map_params.keys())[0]
#     print(f"Example first task ({first_task_key}): {yaml_parser.map_params[first_task_key]}")
```

The `app_data` and `spec_data` dictionaries contain the raw information read from the files, just organized into Python dictionaries. The `map_params` attribute is a more processed structure, specifically built by the parser to outline *what* needs to be plotted for *each* input file and variable combination.

## Under the Hood: How It Works

Let's look at the internal process of the `YAMLParser` when you call `parse()`.

1.  **Concatenate YAML:** The `parse()` method first calls `_concatenate_yaml()`. This is where the file reading happens. It loops through the list of `config_files`. For each file:
    *   It uses a utility function (`u.load_yaml_simple`) to read the YAML content into a Python dictionary.
    *   It adds the corresponding `source_name` to the dictionary.
    *   It merges sections (like `inputs`, `outputs`, `system_opts`, `for_inputs`) from the current file into a single main result dictionary (`self.app_data`). This handles the case of loading multiple configuration files – later files can override settings from earlier ones or add to lists.
    *   It checks for a corresponding `_specs.yaml` file (e.g., `my_settings_specs.yaml` for `my_settings.yaml`) and loads its content into the `self.spec_data` dictionary, merging across multiple files.
2.  **Initialize Map Parameters:** After `_concatenate_yaml()` has populated `self.app_data` and `self.spec_data`, the `parse()` method calls `_init_map_params()`. This method goes through the `inputs` section within `self.app_data`. For each input file listed, and for each variable specified in its `to_plot` or `variables` section, it creates an entry in the `self._map_params` dictionary. Each entry is a dictionary describing a single visualization task (which file, which variable, which source, which plot type, any comparison settings, etc.). This structure is crucial for the later stages of the data pipeline.
3.  **Load Metadata:** Finally, `parse()` loads additional metadata like `meta_coords`, `meta_attrs`, and `species_db` using utility functions. These are often static YAML files describing standard variable/dimension names and attributes, used for consistency and lookups.

Let's look at simplified snippets from the `YAMLParser` class (`eviz/lib/config/yaml_parser.py`):

Here's a simplified `parse` method:

```python
# Simplified snippet from eviz/lib/config/yaml_parser.py
def parse(self):
    """Parse YAML files and populate data attributes."""
    print("Executing YAMLParser.parse()")

    # Step 1: Read and merge the main config and specs files
    self._concatenate_yaml() # This populates self.app_data and self.spec_data
    print(f"Concatenated YAML. app_data keys: {list(self.app_data.keys())}")
    print(f"spec_data keys: {list(self.spec_data.keys())}")

    # Step 2: Process inputs and specs to create plot task list (map_params)
    self._init_map_params([]) # [] is a placeholder for concatenated data
    print(f"Initialized map_params. Number of tasks: {len(self.map_params)}")

    # Step 3: Load additional metadata
    # self.meta_coords = u.read_meta_coords() # Simplified away utility call
    # self.meta_attrs = u.read_meta_attrs()   # Simplified away utility call
    # self.species_db = u.read_species_db()   # Simplified away utility call
    print("Loaded metadata (meta_coords, meta_attrs, species_db).")

    # Data is now ready in self.app_data, self.spec_data, self._map_params
    print("Parsing finished.")

```
This snippet shows the sequence of the three main steps within the `parse` method. It relies on `_concatenate_yaml` and `_init_map_params` to do the actual work.

Now, a very simplified look at `_concatenate_yaml`:

```python
# Simplified snippet from eviz/lib/config/yaml_parser.py
def _concatenate_yaml(self) -> List[Dict[str, Any]]:
    """Read and merge multiple YAML files and their associated specs."""
    print("Executing _concatenate_yaml()")
    result = {}
    self.spec_data = {} # Initialize spec_data

    for index, file_path in enumerate(self.config_files):
        print(f"  Reading file: {file_path}")

        # Use a utility to load the file content
        # yaml_content = u.load_yaml_simple(file_path) # Simplified away utility call
        # --- Imagine load_yaml_simple reads the file and returns a dict ---
        yaml_content = {"inputs": [...], "outputs": {...}} # Placeholder dict
        # --------------------------------------------------------------------

        # Add source name
        yaml_content['source'] = self.source_names[index]

        # Merge main sections into result (simple update logic)
        if 'inputs' in yaml_content:
            result.setdefault('inputs', []).extend(yaml_content['inputs']) # Append lists
        if 'outputs' in yaml_content:
             # Simple update - note: real code merges more carefully
            result.setdefault('outputs', {}).update(yaml_content['outputs'])
        # ... handle other sections like for_inputs, system_opts ...

        # Check for and load specs file
        # simplified_specs_path = file_path.replace('.yaml', '_specs.yaml') # Simplified path logic
        # if os.path.exists(simplified_specs_path): # Simplified check
        #     specs_content = u.load_yaml_simple(simplified_specs_path) # Simplified utility call
        #     self.spec_data.update(specs_content) # Merge specs
        # else:
        #      print(f"  No specs file found for {file_path}")

    self.app_data = result # Store the merged main data
    print(f"  Finished merging files. app_data keys: {list(self.app_data.keys())}")
    # The real method returns the list of individual file dicts, useful for map_params

    return [self.app_data] # Simplified return value

```

This snippet focuses on the core idea of looping through files, loading them (represented by the placeholder dictionary), and merging their content into the `result` dictionary (`self.app_data`) and `self.spec_data`. It shows the basic logic of combining information from multiple sources.

Finally, a *very* simplified look at `_init_map_params`:

```python
# Simplified snippet from eviz/lib/config/yaml_parser.py
def _init_map_params(self, concat: List[Dict[str, Any]]):
    """Organize data for plotting routines into _map_params."""
    print("Executing _init_map_params()")
    _maps = {}
    map_counter = 0

    # In reality, this loops through the 'inputs' section of self.app_data
    # and checks against self.spec_data for details.
    # --- Simplified logic ---
    if 'inputs' in self.app_data:
        for input_entry in self.app_data['inputs']:
            filename = input_entry.get('name', 'unknown_file')
            source_name = self.source_names[0] # Simplified: assume first source
            exp_id = input_entry.get('exp_id', 'default_exp')

            # Assuming 'to_plot' lists variables and their types
            to_plot_vars = input_entry.get('to_plot', {}) # e.g., {'temp': 'map', 'pressure': 'profile'}

            for field_name, plot_type_value in to_plot_vars.items():
                # Create an entry in _maps for this specific plot task
                _maps[map_counter] = {
                    'source_name': source_name,
                    'exp_id': exp_id,
                    'filename': filename,
                    'field': field_name,
                    'to_plot': [plot_type_value] # Store plot type as a list
                    # ... other details like field_specs, comparison info etc.
                }
                print(f"    Created plot task: {map_counter} ({field_name} as {plot_type_value})")
                map_counter += 1

    # ----------------------

    self._map_params = _maps # Store the generated task list
    print(f"  Finished initializing map_params. Total tasks: {len(self._map_params)}")

```
This snippet shows the core loop of processing the `inputs` section from `app_data` and generating individual entries in `_map_params` for each variable that needs plotting from each file. It's like creating a list of specific "jobs" for the plotting system.

The `YAMLParser` is designed to handle the complexities of reading and combining configuration from potentially multiple files and extracting all the necessary details into structured attributes (`app_data`, `spec_data`, `map_params`) that the rest of eViz can easily use.

## Connecting Back to Config Manager

The results of the `YAMLParser` (the `app_data`, `spec_data`, `map_params`, etc., attributes) are exactly what the [Config Manager](02_config_manager_.md) needs. When the `Config` object (which the [Config Manager](02_config_manager_.md) wraps) is initialized, it creates a `YAMLParser` instance, calls its `parse()` method, and then uses the populated attributes from the parser to set up its own structure, initialize the sub-configurations ([InputConfig](02_config_manager_.md#breaking-down-complexity-sub-configurations), [OutputConfig](02_config_manager_.md#breaking-down-complexity-sub-configurations), etc.), and make all the settings readily available.

## Summary

In this chapter, we focused on the **YAML Parser**:

*   It's the component responsible for reading the human-readable instructions in YAML configuration files.
*   It translates the YAML text into structured Python data (dictionaries, lists).
*   Its main task is handled by the `parse()` method.
*   It reads multiple config files, merges settings, and loads associated "specs" files.
*   It organizes the results into key attributes like `app_data` (general settings), `spec_data` (detailed plot specifications), and `map_params` (a list of specific plot tasks).
*   The [Config Manager](02_config_manager_.md) uses the output of the `YAMLParser` to manage the application's settings.

Now that we understand how eViz reads the raw data ([Data Source](01_data_source_.md)) and how it loads and manages the instructions ([Config Manager](02_config_manager_.md), using the YAML Parser), we're ready to look at the main application itself that orchestrates these steps to create visualizations.

Let's move on to [Chapter 4: Autoviz (Main Application)](04_autoviz__main_application__.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
