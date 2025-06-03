# Chapter 8: Model Handler (BaseSource)

Welcome back! In the [previous chapters](05_datasource__base__.md), [Chapter 6: Figure (Visualization Canvas)](06_figure__visualization_canvas__.md), and [Chapter 7: Plotter](07_plotter_.md), we've covered how eViz loads your data into a standard format ([DataSource (Base)](05_datasource__base__.md)), how it prepares the drawing canvas ([Figure (Visualization Canvas)](06_figure__visualization_canvas__.md)), and how it uses the blueprint from the [ConfigManager](02_configmanager_.md) to draw the data onto the canvas ([Plotter](07_plotter_.md)).

But scientific data comes in many shapes and forms, often with quirks specific to its source. A climate model like WRF might name its dimensions differently than another like GEOS. An observational dataset from a satellite like OMI might have a completely different data structure or need specialized unit conversions. Who is responsible for understanding these nuances and coordinating the steps needed *specifically* for that type of data?

This is the role of the **Model Handler**.

## What is a Model Handler and Why Do We Need It?

Imagine you're managing a team of specialists:
*   A data librarian ([DataSourceFactory](04_datasourcefactory_.md) and [DataSource (Base)](05_datasource__base__.md)) who knows how to find and open different kinds of data books.
*   A canvas preparer ([Figure (Visualization Canvas)](06_figure__visualization_canvas__.md)) who gets the art canvas ready, sets its size, and adds grids if needed.
*   An artist ([Plotter](07_plotter_.md)) who knows how to apply paint to a canvas based on instructions.
*   A project manager ([ConfigManager](02_configmanager_.md)) who holds all the project instructions.

You also have raw data coming from different projects: Project A data looks one way, Project B data looks another. You can't just give Project A's raw data directly to the artist; it might need special processing, conversion, or slicing specific to how Project A organizes its data.

You need a **specialized expert** for each project's data. This expert receives the raw data, understands its unique structure and characteristics, applies any necessary project-specific steps, and then directs the canvas preparer and the artist using the project instructions.

In eViz, the **Model Handler** is this specialized expert. It's the layer that understands the specific characteristics and common visualization needs of data coming from a particular scientific source (like a specific climate model or a satellite instrument). It coordinates the entire visualization process *for that source type*.

### The Blueprint: BaseSource

Just as all the specialized data container classes followed the [DataSource (Base)](05_datasource__base__.md) blueprint, all these specialized Model Handler experts follow a common blueprint called **BaseSource**.

**BaseSource** is an **Abstract Base Class (ABC)**. It defines the essential actions that *any* Model Handler must be able to perform to be a valid part of the eViz system. It sets the contract, saying "If you are a Model Handler, you *must* have methods for X, Y, and Z." It doesn't provide the *implementation* for these methods; it just declares that they must exist.

For example, a core requirement for any Model Handler is to be able to generate plots. So, `BaseSource` defines an abstract `plot()` method. Every specific Model Handler (like the one for `GriddedSource` data, or `WRF` data) *must* provide its own version of the `plot()` method, containing the specific logic for orchestrating plots for *that* data type.

This ensures that higher-level components, like [Autoviz (Application Orchestrator)](01_autoviz__application_orchestrator__.md), can interact with *any* Model Handler object in a standard way by calling its required methods, without needing to know the specific data source details.

### The Common Ground: The GenericSource Class

While `BaseSource` defines the pure blueprint, many data sources share common visualization workflows (like plotting single fields, generating simple plots, or creating comparison plots). To avoid duplicating this common logic in every specific Model Handler (like `GriddedSource`, `WRF`, `LIS`, etc.), eViz uses an intermediate class called **GenericSource**.

The **GenericSource** class inherits from `BaseSource` and provides concrete implementations for these common workflows. It contains the logic for:

1.  Receiving the [ConfigManager](02_configmanager_.md) and accessing its settings.
2.  Determining *which* kind of plot is requested (single, comparison, simple) based on the [ConfigManager](02_configmanager_.md) settings.
3.  Getting the necessary data from the [DataPipeline](03_datapipeline_.md).
4.  Creating the appropriate [Figure (Visualization Canvas)](06_figure__visualization_canvas__.md) based on the plot type and configuration.
5.  Calling the correct [Plotter](07_plotter_.md) method (`single_plots`, `comparison_plots`) and passing it the data, figure, and configuration.
6.  Handling common tasks like looping through fields or levels specified in the configuration.

Most specific Model Handlers in eViz (like `GriddedSource`, `WRF`, `LIS`, `Geos`) inherit from `GenericSource` to get all this common functionality for free. They then add their *own* specialized logic on top.

### Specialization: Specific Model Handlers (e.g., GriddedSource, WRF)

Classes like `GriddedSource`, `WRF`, `LIS`, `Geos`, `Airnow`, etc., are the actual, concrete **Model Handlers**. Each of these classes:

1.  Inherits from `GenericSource` (or sometimes a class that inherits from `GenericSource`, like `NuWrf`).
2.  Implements any abstract methods required by `BaseSource` that `GenericSource` didn't cover (though `GenericSource` covers most of the core plotting ones).
3.  Provides specialized methods for its data source, such as:
    *   Understanding source-specific dimension names (e.g., 'XLON' and 'XLAT' in WRF vs. 'lon' and 'lat' elsewhere).
    *   Extracting 2D slices (XY, YZ, XT, TX) from multi-dimensional data in a way specific to that source's structure.
    *   Applying source-specific calculations or unit conversions that happen *after* the general [DataPipeline](03_datapipeline_.md) processing.
    *   Handling source-specific metadata or coordinate system details (like LIS handling NaN coordinates or GEOS parsing HISTORY.rc).

This is where the "specialized interpreter" aspect comes in. These classes contain the "Rosetta Stone" for their particular data format and common analysis patterns.

### Core Use Case: Orchestrating the Plotting Workflow for a Specific Source

The main job of an instance of a specific Model Handler (like a `GriddedSource` object or a `Wrf` object) is to oversee the entire visualization process for the data files assigned to it by the [ConfigManager](02_configmanager_.md).

When [Autoviz (Application Orchestrator)](01_autoviz__application_orchestrator__.md) runs, it first uses the `BaseSourceFactory` ([DataSourceFactory](04_datasourcefactory_.md) concept) to create the correct Model Handler instance based on your input (e.g., `-s gridded` creates a `GriddedSource` instance).

Then, [Autoviz](01_autoviz__application_orchestrator__.md) tells this instance to run its main task, typically by calling its `plot()` method (or its `__call__` method, which in the `GenericSource` class calls `plot()`).

```python
# Inside Autoviz.run() (simplified)
# ... after getting the config_manager and data sources ...

for factory in self.factory_sources: # Loop through factories for requested sources
    self.logger.info(f"Using factory: {type(factory).__name__}")
    # Factory creates the Model Handler instance (e.g., GriddedSource instance)
    model = factory.create_root_instance(self._config_manager)

    # Optional: Set map parameters (often loaded by ConfigManager now)
    # if hasattr(model, 'set_map_params') and self._config_manager.map_params:
    #      model.set_map_params(self._config_manager.map_params)

    # THIS IS WHERE THE MODEL HANDLER ORCHESTRATES ITS WORK!
    # This calls the __call__ method on the model object, which runs model.plot()
    model()
```

When `model()` is called, the specific Model Handler instance (inheriting from `GenericSource` which inherits from `BaseSource`) takes over and orchestrates the steps:

1.  **Read Configuration:** It uses the `config_manager` to find out which fields to plot, what plot types are requested, which levels/times to use, and all styling options.
2.  **Get Data:** It asks the `config_manager.pipeline` ([DataPipeline](03_datapipeline_.md)) for the already loaded and potentially processed data ([DataSource (Base)](05_datasource__base__.md) objects containing `xarray.Dataset`s).
3.  **Source-Specific Data Prep:** It calls its own specialized methods (like `_get_xy`, `_get_yz`, etc.) to extract the correct 2D slice for plotting, applying any necessary source-specific logic like dimension renaming or vertical level handling.
4.  **Prepare Canvas:** It uses the [Figure (Visualization Canvas)](06_figure__visualization_canvas__.md) factory method (`Figure.create_eviz_figure`) to create the appropriate canvas with the right number of subplots and projection.
5.  **Draw:** It calls the methods on the [Plotter](07_plotter_.md) (like `plotter.single_plots`), passing the prepared data slice, the figure, and the relevant configuration options.
6.  **Save/Show Output:** After plotting, it uses utility functions to save the figure to a file or display it, based on the configuration.

### Inside Model Handler: The Orchestration

Let's trace the flow focusing on the specific Model Handler (e.g., `GriddedSource`) coordinating the process after `Autoviz` calls its main method (`model()` -> `GenericSource.plot()`):

```{mermaid}
sequenceDiagram
    participant A as Autoviz
    participant Factory as BaseSourceFactory
    participant Handler as Specific Model Handler (e.g., GriddedSource)
    participant CM as ConfigManager
    participant DP as DataPipeline
    participant Fig as Figure
    participant Plotter as Plotter
    participant Data as Data (xarray.DataArray)

    A->>Factory: create_root_instance(config_manager)
    Factory-->>Handler: Return Handler instance (e.g., GriddedSource)
    A->>Handler: handler() (calls handler.plot())
    activate Handler
    Handler->>Handler: (Inherited from GenericSource) Check config type (single/compare)
    Handler->>Handler: (Inherited from GenericSource) Call _single_plots()
    activate Handler
    Handler->>CM: Get map_params (fields to plot)
    CM-->>Handler: Return map_params
    loop over fields/levels/times
        Handler->>CM: Get filename/field from map_params
        Handler->>DP: get_data_source(filename)
        DP-->>Handler: Return DataSource (with Dataset)
        Handler->>Handler: Call source-specific _get_field_to_plot(data_array, ...)
        activate Handler
        Handler->>Handler: Extract/slice data (e.g., _get_xy, _get_yz)
        deactivate Handler
        Handler->>Fig: create_eviz_figure(config, plot_type)
        Fig-->>Handler: Return Figure instance
        Handler->>Plotter: Create SinglePlotter()
        Handler->>Plotter: plotter.single_plots(config, data_to_plot, ...)
        activate Plotter
        Plotter->>Plotter: (Calls plotting helpers)
        Plotter->>Fig: Add elements to Axes (using Figure's methods)
        Plotter->>CM: Get styling from config
        Plotter-->>Handler: Plotting complete for this item
        deactivate Plotter
        Handler->>Handler: (Inherited from GenericSource) Call print_map (save/show figure)
    end
    deactivate Handler
    Handler-->>A: Plotting complete
    deactivate Handler
    A-->>User: Done
```

This diagram shows that the specific Model Handler (`GriddedSource` in this example) is the central coordinator once it's created. It drives the process, getting instructions from the [ConfigManager](02_configmanager_.md), fetching data from the [DataPipeline](03_datapipeline_.md), preparing the canvas with the [Figure (Visualization Canvas)](06_figure__visualization_canvas__.md), and telling the [Plotter](07_plotter_.md) exactly what to draw and where.

### Code Walkthrough (Simplified)

Let's look at simplified snippets to see how `BaseSource`, `GenericSource`, and a specific handler like `GriddedSource` fit together.

First, the `BaseSource` (in `eviz/models/base.py`):

```python
# eviz/models/base.py (simplified)
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

# BaseSource is an ABC
@dataclass
class BaseSource(ABC):
    """
    Abstract base class that defines the interface for all Model Handlers.
    Defines the contract that specific handlers must follow.
    """
    config_manager: object # Placeholder for ConfigManager access

    @property
    @abstractmethod # Must be implemented by subclasses
    def logger(self) -> logging.Logger:
        """Abstract property for the logger instance."""
        pass

    @abstractmethod # Must be implemented by subclasses
    def plot(self):
        """Abstract method for the top-level plotting routine."""
        pass

    # Although implemented in GenericSource now, these were originally abstract
    # methods defining the contract for data source management.
    @abstractmethod
    def add_data_source(self, name, data_source): pass
    @abstractmethod
    def get_data_source(self, name): pass
    @abstractmethod
    def load_data_sources(self, file_list): pass

    def __post_init__(self):
        """Default initialization behavior (optional for subclasses to call)."""
        self.logger.info("Initializing BaseSource")
        # Subclasses would typically call super().__post_init__()
```

This shows that `BaseSource` enforces that any class inheriting from it must provide a `logger` and a `plot` method, among others originally defined here.

Next, the `GenericSource` class (in `eviz/models/root.py`), which inherits from `BaseSource`:

```python
# eviz/models/source_base.py (simplified)
from dataclasses import dataclass
import logging
# ... other imports like SinglePlotter, ComparisonPlotter ...
from eviz.lib.config.config_manager import ConfigManager
from eviz.models.base import BaseSource  # Inherits from BaseSource!


# ... other imports ...

@dataclass
class GenericSource(BaseSource):  # Inherits from BaseSource
    """
    Provides common orchestration and plotting logic for Model Handlers.
    Implements the BaseSource interface.
    """
    config_manager: ConfigManager  # Requires a ConfigManager

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)  # Provides the required logger

    def __post_init__(self):
        self.logger.info("Start init GenericSource")
        super().__post_init__()  # Call the base class init
        self.config = self.config_manager.config
        self.app = self.config_manager.app_data
        self.specs = self.config_manager.spec_data
        # ... initializes other common attributes ...
        self.use_mp_pool = self.app.system_opts.get('use_mp_pool', False)

    # This implements the abstract plot() method from BaseSource
    def plot(self):
        """Top-level orchestration for generating plots."""
        self.logger.info("Generate plots.")

        if not self.config_manager.spec_data:
            # No SPECS config? Do simple plots.
            plotter = SimplePlotter()
            self._simple_plots(plotter)  # Calls helper method for simple plots
        else:
            # SPECS config exists - check for comparison/overlay modes
            if self.config_manager.compare and not self.config_manager.compare_diff:
                plotter = ComparisonPlotter(self.config_manager.compare_exp_ids)
                self._side_by_side_plots(plotter)  # Calls helper for side-by-side
            # ... other comparison/overlay checks ...
            else:
                # Default: Single plots using SPECS
                plotter = SinglePlotter()
                self._single_plots(plotter)  # Calls helper for single plots

        # ... Logic to handle output files (print_to_file) ...
        self.logger.info("Done.")

    # GenericSource also provides concrete implementation for common plotting workflows:
    def _single_plots(self, plotter):
        """Generate single plots based on map_params from config."""
        self.logger.info("Generating single plots (in GenericSource)")

        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for single plotting.")
            return

        # Loop through plot requests from config (map_params)
        for idx, params in self.config_manager.map_params.items():
            field_name = params.get('field')
            if not field_name: continue

            filename = params.get('filename')
            # Get data source from the pipeline (delegating this responsibility)
            data_source = self.config_manager.pipeline.get_data_source(filename)
            if not data_source or not hasattr(data_source, 'dataset'): continue

            field_data_array = data_source.dataset.get(field_name)
            if field_data_array is None: continue

            plot_types = params.get('to_plot', ['xy'])
            if isinstance(plot_types, str): plot_types = [pt.strip() for pt in plot_types.split(',')]

            # Loop through each requested plot type for this field
            for plot_type in plot_types:
                # --- This is where specific handlers might override logic ---
                # GenericSource calls a helper (_process_plot) which might be implemented
                # differently in GriddedSource, WRF, etc.
                self._process_plot(field_data_array, field_name, idx, plot_type, plotter)
                # ------------------------------------------------------------

    # ... _comparison_plots, _side_by_side_plots, _simple_plots methods omitted ...
    # ... data source methods (add_data_source, etc.) are implemented here
    #     but delegate to the pipeline ...
    # ... other utility methods omitted ...
```

This shows `GenericSource` inheriting from `BaseSource` and providing the main `plot()` method that dispatches to other helper methods (`_single_plots`, `_comparison_plots`) based on the configuration. It also shows how it retrieves data from the [DataPipeline](03_datapipeline_.md) and prepares to call a plotter. Notice how `_single_plots` calls `_process_plot`, which is a key point for specific handlers to add their logic.

Finally, let's look at the `GriddedSource` class (in `eviz/models/esm/gridded.py`), which inherits from `GenericSource`:

```python
# eviz/models/esm/gridded_source.py (simplified)
from dataclasses import dataclass
import logging
# ... other imports like xarray, Figure, DataProcessor ...
from eviz.models.source_base import GenericSource  # Inherits from GenericSource!


@dataclass
class GriddedSource(GenericSource):  # Inherits from GenericSource
    """
    Model Handler for generic gridded data (NetCDF).
    Adds specialized methods for slicing gridded data.
    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)  # Provides logger

    def __post_init__(self):
        self.logger.info("Start init GriddedSource")
        super().__post_init__()  # Calls GenericSource.__post_init__ and BaseSource.__post_init__
        # GriddedSource might initialize its own DataProcessor or other components
        self.processor = DataProcessor(self.config_manager)

    # GriddedSource overrides or adds methods to handle its specific data structure
    # This method is called by GenericSource's _single_plots
    def _process_plot(self, data_array: xr.DataArray, field_name: str, file_index: int,
                      plot_type: str, plotter):
        """
        Process a single plot type for a given field (GriddedSource specific).
        Orchestrates slicing, figure creation, and plotting calls.
        """
        self.logger.info(f"GriddedSource: Plotting {field_name}, {plot_type} plot")

        # Create the figure (Figure factory method used here)
        figure = Figure.create_eviz_figure(self.config_manager, plot_type)
        # Initialize axis options for this field/plot type from config
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        # Determine time levels to plot (e.g., single, all, range) from config
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        num_times = data_array[tc_dim].size if tc_dim in data_array.dims else 1
        time_levels = range(num_times) if time_level_config == 'all' else [
            time_level_config]

        # Handle levels or zsum based on config
        levels = self.config_manager.get_levels(field_name, plot_type + 'plot')
        do_zsum = self.config_manager.ax_opts.get('zsum', False)

        if levels:
            # Loop over levels defined in SPECS config
            for level_val in levels.keys():
                self.config_manager.level = level_val  # Set current level in config
                for t in time_levels:
                    # --- GriddedSource specific data slicing ---
                    # Calls helper method to get the 2D slice
                    field_to_plot = self._get_field_to_plot(data_array, field_name,
                                                            file_index, plot_type,
                                                            figure, t, level=level_val)
                    # -------------------------------------
                    if field_to_plot:
                        # Call the plotter (passing prepared data, figure, config)
                        plotter.single_plots(self.config_manager, field_to_plot=field_to_plot,
                                             level=level_val)
                        # Save the map (using utility method)
                        pu.print_map(self.config_manager, plot_type,
                                     self.config_manager.findex, figure,
                                     level=level_val)
        # ... Similar logic for zsum or if no levels specified ...

    # --- GriddedSource specific data slicing methods ---
    def _get_field_to_plot(self, data_array: xr.DataArray, field_name: str,
                           file_index: int, plot_type: str, figure, time_level,
                           level=None) -> tuple:
        """Prepare the data array and coordinates for plotting (GriddedSource specific)."""
        if data_array is None: return None

        # Get expected dimension names from config (e.g., 'lon', 'lat')
        dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
        data2d = None

        # Call specific slicing method based on plot type
        if 'yz' in plot_type:
            data2d = self._get_yz(data_array, time_lev=time_level)  # GriddedSource specific _get_yz
        elif 'xt' in plot_type:
            data2d = self._get_xt(data_array, time_lev=time_level)  # GriddedSource specific _get_xt
        # ... etc. for tx, xy, polar ...
        elif 'xy' in plot_type or 'polar' in plot_type:
            data2d = self._get_xy(data_array, level=level, time_lev=time_level)  # GriddedSource specific _get_xy
        else:
            return None

        if data2d is None: return None

        # Extract coordinates based on determined dimension names
        x_values = data2d[dim1_name].values if dim1_name in data2d.coords else None
        y_values = data2d[dim2_name].values if dim2_name in data2d.coords else None
        # ... Error handling for coordinates ...

        # Return data and coordinates in the tuple format expected by Plotter
        return data2d, x_values, y_values, field_name, plot_type, file_index, figure

    # Example of a GriddedSource specific slicing method
    def _get_xy(self, data_array, level, time_lev):
        """ Extract XY slice (latlon) from a DataArray (GriddedSource specific logic) """
        if data_array is None: return None

        d_temp = data_array.copy()
        tc_dim = self.config_manager.get_model_dim_name('tc')  # Get time dim name from config
        zc_dim = self.config_manager.get_model_dim_name('zc')  # Get vertical dim name from config

        # Apply time selection based on config/input time_lev
        if tc_dim and tc_dim in d_temp.dims:
            # ... slicing logic using d_temp.isel({tc_dim: time_lev}) ...
            pass  # Simplified

        # Apply vertical level selection based on config/input level
        has_vertical_dim = zc_dim and zc_dim in d_temp.dims
        if has_vertical_dim:
            if level is not None:
                # ... slicing logic using d_temp.isel({zc_dim: lev_idx}) ...
                pass  # Simplified

        data2d = d_temp.squeeze()

        # Handle additional dimensions if still > 2
        if len(data2d.dims) > 2:
            # ... logic to reduce dimensions (e.g., take mean or first index) ...
            pass  # Simplified

        # Apply time/vertical averaging/summing if requested in config (ax_opts)
        if self.config_manager.ax_opts.get('tave', False):
            # ... apply time average using apply_mean ...
            pass  # Simplified
        if self.config_manager.ax_opts.get('zsum', False):
            # ... apply vertical sum using apply_zsum ...
            pass  # Simplified

        # Apply unit conversion if needed (using apply_conversion utility)
        data2d.attrs = data_array.attrs.copy()
        return apply_conversion(self.config_manager, data2d, data_array.name)

    # ... Other GriddedSource specific methods like _get_yz, _get_xt, _get_tx, _select_yrange ...

```

This shows how `GriddedSource` inherits from `GenericSource`, providing its own `_process_plot` method (which overrides or adds to whatever `GenericSource` might have had, although in this case `GenericSource` delegates this step). This method then calls *further* `GriddedSource`-specific helper methods like `_get_field_to_plot` and `_get_xy` to perform the actual data slicing and preparation based on its understanding of gridded NetCDF data dimensions and structures. These specific methods then return the prepared data slice and coordinates, which are passed to the [Plotter](07_plotter_.md).

The `Wrf`, `Lis`, `Geos`, `Airnow`, `Omi`, etc., classes follow a similar pattern, inheriting from `GenericSource` (or a class in between) and implementing methods like `_get_field_to_plot` and specific slicing methods (`_get_xy`, `_get_xt`, etc.) with logic tailored to their unique data formats and structures. For instance, the `Wrf` code shows specific handling for staggered grids and pressure levels, while `Lis` has logic for fixing NaN coordinates. `Airnow` and `Omi` handle non-gridded or satellite data structures differently.

### Conclusion

In this chapter, we learned about the **Model Handler**, the crucial layer that provides specialized understanding and orchestration for data coming from different scientific sources. We saw how **BaseSource** defines the fundamental contract for any Model Handler, and how the **GenericSource** class provides common plotting workflow implementations that most specific handlers inherit.

Finally, we looked at how concrete Model Handler classes like **GriddedSource**, **WRF**, and others inherit from `GenericSource` and implement source-specific logic for data slicing, processing, and coordination, ensuring that the entire visualization pipeline correctly handles the unique characteristics of each data type.

The Model Handler effectively brings together the configuration ([ConfigManager](02_configmanager_.md)), data loading/processing ([DataPipeline](03_datapipeline_.md), [DataSourceFactory](04_datasourcefactory_.md), [DataSource (Base)](05_datasource__base__.md)), canvas preparation ([Figure (Visualization Canvas)](06_figure__visualization_canvas__.md)), and drawing ([Plotter](07_plotter_.md)) for a specific data source type, acting as the specialized brain for that part of the visualization task.

With this chapter, we have completed the cycle of understanding the core components that take your data from file to plot in eViz!

This is the last chapter covering core eViz concepts in this initial tutorial series. You now have a foundational understanding of the main pieces and how they interact, orchestrated by [Autoviz (Application Orchestrator)](01_autoviz__application_orchestrator__.md) based on your [ConfigManager](02_configmanager_.md) blueprint.

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)