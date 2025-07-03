# Chapter 6: Model/Source Handler

Welcome back to the eViz tutorial! In the previous chapter, [Chapter 5: Data Pipeline](05_data_pipeline_.md), we learned how eViz uses an automated assembly line to take raw data from files, process it, and prepare it into a standardized format, primarily `xarray.Dataset`.

Now that the data is cleaned and ready, eViz needs to figure out exactly *how* to visualize it. An `xarray.Dataset` containing temperature data from a global climate model looks different from a dataset with ozone observations from a satellite or soil moisture from a land surface model. They use different variable names, different coordinate names, might have different structures (gridded vs. points/swath), and may require specific handling (like dealing with staggered grids in some models, or knowing which columns in observation data represent latitude and longitude).

This is where the **Model/Source Handler** concept comes in.

## What Problem Does Model/Source Handler Solve?

Imagine you have processed data for:
*   Temperature from the WRF atmospheric model.
*   Soil moisture from the LIS land surface model.
*   CO2 concentration from Airnow observational stations.
*   Ozone column amount from OMI satellite data.

All of these might end up in `xarray.Dataset` objects after the pipeline, but they have different characteristics:
*   WRF might use dimension names like 'XLAT' and 'XLONG' and require specific logic to handle its grid.
*   LIS might use 'north_south' and 'east_west' and have NaN values in coordinates that need careful handling.
*   Airnow data is point-based, not gridded, and comes from CSV files with specific column names ('Latitude', 'Longitude', 'Value').
*   OMI data is swath-based, often in HDF5, and might need specific data fields extracted and re-projected for plotting.

A single, generic plotting function wouldn't know how to correctly interpret the dimensions of WRF data, the coordinate issues in LIS, or the point structure of Airnow data.

**Model/Source Handlers** solve this by acting as specialized experts for each type of data source or model. They understand the unique structure, naming conventions, and specific processing or plotting needs of *their* data type. They provide the bridge between the generic processed data (`xarray.Dataset`) and the specific logic needed to prepare that data for a particular plot type (like a map or a time series).

Think of them as expert translators or guides for specific scientific domains. Once the [Data Pipeline](05_data_pipeline_.md) has done the general cleaning, the Model/Source Handler steps in to interpret the data according to the rules of its specific model or source.

## Our Central Use Case: Preparing Data for a Map Plot

Let's say your configuration asks for a map plot (`xy` or `sc` type) of the variable 'temperature' from a WRF model output file.

The core question is: How does eViz know how to take the 'temperature' variable from *that specific* WRF `xarray.Dataset` and get it ready for a 2D map plot? It uses a WRF Model/Source Handler.

## Using the Concept (Mostly Internal)

As with the [Data Pipeline](05_data_pipeline_.md), you don't usually create Model/Source Handler objects directly in your configuration or user code. Instead, they are instantiated internally by the main [Autoviz](04_autoviz__main_application__.md) application class.

Recall from [Chapter 4: Autoviz](04_autoviz__main_application__.md) that when you run `autoviz.py`, you specify a source type using the `-s` flag (e.g., `-s wrf`, `-s lis`, `-s gridded`, `-s airnow`, `-s omi`, `-s crest`).

[Autoviz](04_autoviz__main_application__.md) uses a **Source Factory** (like `WrfFactory`, `LisFactory`, `GriddedSourceFactory`, `AirnowFactory`, `OmiFactory`, `CrestFactory`) based on this `-s` input. The role of this factory is precisely to create the *correct* Model/Source Handler instance (`Wrf`, `Lis`, `GriddedSource`, `Airnow`, `Omi`, `Crest`).

```python
# Imagine this is simplified logic inside Autoviz.run()
# from eviz.models.source_factory import WrfFactory, GriddedSourceFactory # etc.
# from eviz.models.esm.wrf import Wrf # etc.

# This comes from the -s flag provided by the user
source_name_from_args = 'wrf'
print(f"User requested source type: {source_name_from_args}")

# Autoviz gets the appropriate factory
# See eviz/lib/autoviz/base.py -> get_factory_from_user_input
# Simplified example:
from eviz.models.source_factory import GriddedSourceFactory, WrfFactory

def get_factory(name):
    if name == 'gridded': return GriddedSourceFactory()
    if name == 'wrf': return WrfFactory()
    # ... handle other types ...
    return None

factory = get_factory(source_name_from_args)
print(f"Found factory: {type(factory)}")

# The factory creates the specific Model/Source Handler instance
# See eviz/models/source_factory.py -> create_root_instance
config_manager = "..." # Imagine ConfigManager is already created
model_handler = factory.create_root_instance(config_manager)
print(f"Created Model/Source Handler object: {type(model_handler)}")

# Now 'model_handler' is an instance of the Wrf class, ready to process WRF data
```

The `model_handler` object created here is the specific expert for your chosen data type. It holds a reference to the [Config Manager](02_config_manager_.md) (and thus the [Data Pipeline](05_data_pipeline_.md) and loaded data) and knows how to interpret settings and data fields according to its specialized knowledge.

## Breaking Down the Concept: Handlers and Factories

1.  **Base Handler (`BaseSource`):**
    *   The foundation is the `BaseSource` abstract base class (`eviz/models/base.py`).
    *   It defines the *minimum contract* for any handler: it needs a `logger`, a main `plot()` method (or similar entry point like `__call__` in `GenericSource`), and methods for managing data sources (though these are now often delegated to the [Data Pipeline](05_data_pipeline_.md)).
    *   It ensures all handlers have a common interface, even though their implementations differ.
    ```python
    # Simplified snippet from eviz/models/base.py
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    import logging
    from eviz.lib.config.config_manager import ConfigManager

    @dataclass
    class BaseSource(ABC):
        config_manager: ConfigManager

        @property
        @abstractmethod
        def logger(self) -> logging.Logger:
            """Abstract property for the logger instance."""
            pass

        @abstractmethod
        def plot(self): # Or __call__ in GenericSource
            """Abstract method for the top-level plotting routine."""
            pass

        # ... other abstract methods like add_data_source (often delegated now) ...

        def __post_init__(self):
            self.logger.info(f"Initializing {self.__class__.__name__}")

    ```

2.  **Generic Source (`GenericSource`):**
    *   `GenericSource` (`eviz/models/source_base.py`) is a base class that adds more common functionality on top of `BaseSource`, implementing the `__call__` method which orchestrates the plotting loop (single plots, comparison plots, etc.) based on the `map_params` from the [Config Manager](02_config_manager_.md). Most specific handlers inherit from `GenericSource` or `GriddedSource`/`ObsSource` which inherit from `GenericSource`.

3.  **Specific Handlers (`GriddedSource`, `Wrf`, `Lis`, `ObsSource`, `Airnow`, `Omi`, `Ghg`, `Crest`, etc.):**
    *   These are the concrete classes that implement the logic for specific data types.
    *   `GriddedSource` (`eviz/models/gridded_source.py`) is a general handler for regular gridded data. It implements common methods for extracting XY, YZ, XT, TX slices.
    *   `ObsSource` (`eviz/models/obs_source.py`) is a general handler for observational data, adding logic for handling irregular grids, swath data, and extracting geographical extents.
    *   `Wrf` (`eviz/models/esm/wrf.py`) inherits from `GenericSource` (or potentially a more specific base like `NuWrf`) and adds WRF-specific logic, like understanding WRF's staggered grid dimension names (`_stag`), approximating pressure levels, and specific coordinate processing.
    *   `Lis` (`eviz/models/esm/lis.py`) inherits from `GenericSource` (or `NuWrf`) and includes LIS-specific coordinate fixing logic.
    *   `Airnow` (`eviz/models/obs/inventory/airnow.py`) inherits from `ObsSource` and has specific methods for handling the structure of Airnow CSV data (treating rows as points, columns as variables, etc.) and plotting scattering (`sc`) data.
    *   `Omi` (`eviz/models/obs/satellite/omi.py`) inherits from `ObsSource` and includes logic for extracting fields and their coordinates from OMI HDF5 files, handling fill values, and reconstructing lat/lon.
    *   `Ghg` (`eviz/models/obs/inventory/ghg.py`) inherits from `ObsSource` and is specialized for time series GHG data, handling CSV parsing and uncertainty data.
    *   `Crest` (`eviz/models/esm/crest.py`) is interesting because it *contains* instances of `GriddedSource` and `ObsSource` and *delegates* plotting tasks to the appropriate internal handler based on whether the specific data variable being plotted looks like gridded or observational data. This is an example of a handler that can handle mixed data types by using other handlers internally.

    ```python
    # Simplified snippet showing structure and specific methods
    # from eviz.models.gridded_source import GriddedSource
    # from eviz.models.obs_source import ObsSource
    # from eviz.models.esm.wrf import Wrf
    # from eviz.models.obs.satellite.omi import Omi

    # Inside GriddedSource (eviz/models/gridded_source.py)
    class GriddedSource(GenericSource):
        def _extract_xy_simple(self, data_array: xr.DataArray) -> xr.DataArray:
            # Logic to squeeze/mean over time/vertical dims for a simple XY map
            # Uses self.config_manager.get_model_dim_name('tc') etc.
            print("GriddedSource: Extracting simple XY slice...")
            data2d = data_array.squeeze()
            # ... (apply mean over time/vertical dims if present) ...
            return data2d

    # Inside Wrf (eviz/models/esm/wrf.py)
    class Wrf(GenericSource): # Actually inherits NuWrf, which inherits GenericSource
        def _process_coordinates(self, data2d, dim1, dim2, *args):
             # WRF specific coordinate handling (staggered grids, projecting)
             print("Wrf: Processing WRF specific coordinates...")
             # Uses self.config_manager.get_model_dim_name('xc') etc.
             # ... logic to get WRF specific lon/lat arrays ...
             return data2d, xs, ys, *args # Return processed coordinates


    # Inside Omi (eviz/models/obs/satellite/omi.py)
    class Omi(ObsSource): # Actually inherits ObsSource, which inherits GenericSource
         def _prepare_field_to_plot(self, ds_short, field_name, *args):
             # OMI specific data extraction and coordinate reconstruction
             print(f"Omi: Preparing field '{field_name}' for plotting...")
             # Calls utility to extract field, handle fill value, reconstruct lat/lon
             data2d, lats, lons = extract_field_with_coords(ds_short, field_name)
             return data2d, lons, lats, field_name, *args # Return data and reconstructed coords

    # Inside Crest (eviz/models/esm/crest.py)
    class Crest(GenericSource):
        def __post_init__(self):
             super().__post_init__()
             # Crest holds instances of other handlers
             self.gridded_handler = GriddedSource(self.config_manager)
             self.obs_handler = ObsSource(self.config_manager)

        def process_plot(self, data_array, field_name, *args):
             # Crest decides which internal handler to use
             is_obs = self._is_observational_data(data_array) # Logic to check data shape/dims
             if is_obs:
                 print("Crest: Delegating to ObsSource handler...")
                 handler = self.obs_handler
             else:
                 print("Crest: Delegating to GriddedSource handler...")
                 handler = self.gridded_handler

             # Call the corresponding method on the selected handler
             if hasattr(handler, 'process_plot'): # Assuming process_plot exists on the handler
                 handler.process_plot(data_array, field_name, *args)
             # ... or call more specific methods like _process_xy_plot ...

    ```
    These snippets highlight how specific handlers implement methods (often starting with `_process_` or `_extract_`) that contain the tailored logic for their data type. They use the `config_manager` to access general settings and dimension name mappings.

4.  **Source Factories:**
    *   As mentioned, factories (`eviz/models/source_factory.py`) are simple classes whose sole purpose is to create the correct handler instance when requested by [Autoviz](04_autoviz__main_application__.md).

    ```python
    # Simplified snippet from eviz/models/source_factory.py
    from dataclasses import dataclass
    # from eviz.models.gridded_source import GriddedSource
    # from eviz.models.esm.wrf import Wrf # etc.

    class BaseSourceFactory:
         # Abstract base with create_root_instance method

    @dataclass
    class GriddedSourceFactory(BaseSourceFactory):
         def create_root_instance(self, config_manager):
             return GriddedSource(config_manager) # Creates a GriddedSource object

    @dataclass
    class WrfFactory(BaseSourceFactory):
         def create_root_instance(self, config_manager):
             return Wrf(config_manager) # Creates a Wrf object

    # ... similar factories for Lis, Airnow, Omi, Ghg, Crest ...
    ```
    The factory pattern keeps the logic for *which* handler to create separate from the code that *uses* the handler, making the system more modular.

## Under the Hood: Orchestration and Delegation

Here's a sequence diagram showing the main flow once [Autoviz](04_autoviz__main_application__.md) has created a specific Model/Source Handler (e.g., `Wrf`):

```{mermaid}
sequenceDiagram
    participant Autoviz as Autoviz (in run())
    participant ConfigMgr as ConfigManager
    participant Pipeline as DataPipeline
    participant Handler as Wrf Instance
    participant Dataset as xarray.Dataset
    participant Figure as Figure Object
    participant Plotter as Plotter Backend

    Autoviz->>Handler: Call handler() (or handler.plot())
    Handler->>ConfigMgr: Get map_params (list of plot tasks)
    Handler->>ConfigMgr: Get access to Pipeline
    loop For each plot task (variable, plot type, file) in map_params
        Handler->>Pipeline: Get processed data (e.g., data_source for file)
        Pipeline-->>Handler: Return DataSource object
        Handler->>Dataset: Access xarray.Dataset from DataSource
        Handler->>Handler: Select variable (e.g., dataset['temperature'])
        Handler->>Handler: Call specific method to prepare data (e.g., _prepare_field_to_plot)
        Handler->>Handler: Apply model-specific logic (_extract_xy_data, _process_coordinates etc.)
        Handler->>Figure: Create Figure (Figure.create_eviz_figure)
        Handler->>Figure: Set up axes (figure.set_axes, init_ax_opts)
        Figure-->>Handler: Return Figure object (often implicitly)
        Handler->>Plotter: Call plotting method (e.g., plotter.plot_2d) with prepared data & Figure
        Plotter->>Plotter: Draw plot on Figure axes
        Plotter-->>Handler: Return plot result/Figure
        Handler->>Autoviz: Save/Display plot (via utils.print_map)
    end
    Autoviz-->>Autoviz: Continue/Finish
```

This diagram shows that the Model/Source Handler `Handler` is the component that iterates through the desired plots (`map_params`), retrieves the necessary data (via the [Data Pipeline](05_data_pipeline_.md)), applies its specialized knowledge to prepare the data for the specific plot type, and then uses the [Figure](07_figure_.md) object to set up the plot canvas and the [Plotter Backend](08_plotter_backend_.md) to do the actual drawing.

Here are some simplified code snippets illustrating this delegation and specialized logic:

```python
# Simplified snippet from eviz/models/source_base.py's __call__ method (or similar plot entry point)
class GenericSource(BaseSource):
     # ... __init__ etc ...

     def __call__(self):
         self.logger.info(f"Executing model handler: {self.__class__.__name__}")

         # This loop is the core orchestration
         for plot_task_key, plot_params in self.config_manager.map_params.items():
             field_name = plot_params.get('field')
             filename = plot_params.get('filename')
             plot_types = plot_params.get('to_plot', [])

             if not field_name or not filename or not plot_types:
                  self.logger.warning(f"Skipping incomplete plot task: {plot_params}")
                  continue

             # Get the loaded data for this file from the pipeline
             data_source = self.config_manager.pipeline.get_data_source(filename)
             if not data_source or not hasattr(data_source, 'dataset'):
                  self.logger.error(f"Could not get data source for file: {filename}")
                  continue

             if field_name not in data_source.dataset:
                  self.logger.error(f"Field '{field_name}' not found in dataset from {filename}")
                  continue

             data_array = data_source.dataset[field_name]
             file_index = self.config_manager.get_file_index(filename)

             # Process comparison plots first if needed (more complex logic omitted)
             if self.config_manager.compare or self.config_manager.overlay:
                 # ... logic to handle pairs/multiple files ...
                 # Delegates to _process_xy_comparison_plots etc.
                 pass
             else:
                 # Process single plots
                 for plot_type in plot_types:
                      # This is where the handler uses its specific logic
                      # Calls a method that is implemented/overridden in subclasses
                      # to prepare the data specific to this field and plot type
                      prepared_data_tuple = self._prepare_field_to_plot(
                           data_array, field_name, file_index, plot_type, None # Figure created later
                      )

                      if prepared_data_tuple:
                           # Creates the Figure and calls the Plotter (simplified delegation)
                           figure = Figure.create_eviz_figure(self.config_manager, plot_type)
                           plot_result = self.create_plot(field_name, prepared_data_tuple, figure)
                           # Saves/shows the plot
                           pu.print_map(self.config_manager, plot_type, file_index, plot_result)

     # This method must be implemented in subclasses (or in base classes like Gridded/Obs)
     # It knows how to slice/process the data_array for the specific plot_type
     # using model-specific dimension names, levels, etc.
     def _prepare_field_to_plot(self, data_array, field_name, file_index, plot_type, figure, **kwargs):
          raise NotImplementedError("Subclass must implement _prepare_field_to_plot")

     # This method is often in GenericSource or its parents, it chooses the right plotting function
     # from the plotter backend based on the plot_type and calls it.
     def create_plot(self, field_name, prepared_data_tuple, figure):
          # Gets the appropriate plotter backend object (e.g., MatplotlibPlotter)
          plotter = self.config_manager.get_plotter_backend() # Simplified access

          # Selects the correct plotting function based on plot_type
          if 'xy' in prepared_data_tuple[4] or 'sc' in prepared_data_tuple[4]: # Check plot_type
               plotter_method = plotter.plot_2d # Or plotter.plot_scatter
          elif 'xt' in prepared_data_tuple[4] or 'tx' in prepared_data_tuple[4]:
               plotter_method = plotter.plot_line # Or plotter.plot_time_series
          # ... handle other plot types ...
          else:
               self.logger.error(f"Unsupported plot type: {prepared_data_tuple[4]}")
               return None

          # Call the plotter method, passing the prepared data and figure details
          # prepared_data_tuple structure: (data2d, dim1_coords, dim2_coords, field_name, plot_type, file_index, figure)
          return plotter_method(*prepared_data_tuple) # Unpack the tuple and call plotter


```
The `__call__` method (or `plot()`) loops through the tasks. The core of the handler's specialization is in methods like `_prepare_field_to_plot` and the specific `_extract_` or `_process_` methods they call (e.g., `_extract_xy_data` in `GriddedSource`, `_process_coordinates` in `Wrf`, `_prepare_field_to_plot` in `Airnow`/`Omi`/`Ghg`). These methods know how to slice, select, and format the data *specific to their model/source type* before handing it off to the plotting backend.

The `create_plot` method bridges the Handler to the [Plotter Backend](08_plotter_backend_.md), selecting the right plotting function based on the plot type requested.

## Summary

In this chapter, we learned about the **Model/Source Handler**:

*   It's the "expert interpreter" for data from a specific scientific model or source (like WRF, LIS, Airnow, OMI).
*   It understands the unique characteristics, naming conventions, and structural details of its data type.
*   Based on the plot tasks defined in the configuration ([Config Manager](02_config_manager_.md)), it knows how to select the correct variable and extract/process the appropriate slice or subset of data ([xarray.Dataset](01_data_source_.md) processed by [Data Pipeline](05_data_pipeline_.md)) needed for a particular plot type.
*   Specific handler classes (like `Wrf`, `Lis`, `ObsSource`, `Airnow`, `Omi`, `Ghg`, `Crest`) implement the tailored logic.
*   Source Factories are used by [Autoviz](04_autoviz__main_application_.md) to create the correct handler instance based on user input.
*   The handler orchestrates the final plotting steps by preparing the data and passing it to the [Figure](07_figure_.md) setup and [Plotter Backend](08_plotter_backend_.md).

The Model/Source Handler is crucial because it allows eViz to handle the diversity of scientific data formats and structures, providing model-specific intelligence required before data can be successfully visualized.

Now that we know how eViz selects and prepares the data for a specific plot using the appropriate handler, the next step is to understand the container where the visualization happens: the **Figure**.

Let's move on to [Chapter 7: Figure](07_figure_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)