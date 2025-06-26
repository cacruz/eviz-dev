# Chapter 8: Source/Model Specific Logic

Welcome back! In our journey so far, we've seen how the [Autoviz Application](01_autoviz_application_.md) directs the process, how [Configuration Management](02_configuration_management_.md) provides the detailed plan, how [Data Source Abstraction](03_data_source_abstraction_.md) and the [Data Source Factory](04_data_source_factory_.md) handle reading different file formats into a standard `xarray.Dataset`, and how the [Data Processing Pipeline](05_data_processing_pipeline_.md) cleans and prepares that data. We also learned about [Plotter Abstraction](06_plotter_abstraction_.md) and the [Plotter Factory](07_plotter_factory_.md) for creating plots using a standard interface.

Now, imagine you have loaded your data into an `xarray.Dataset`, and it's been through the pipeline. You're ready to plot! But different types of scientific data, even when in the same `xarray` format, can have subtle (or not-so-subtle!) differences.

For example:
*   A weather model like WRF might use specific dimension names or grid structures (like "staggered" grids) that need special handling before calculating averages or extracting slices.
*   A land surface model like LIS might store coordinates in a unique way or have NaN values that need attention.
*   Satellite observations might be in a "swath" format (irregular, track-based data) instead of a nice rectangular grid, requiring different approaches for plotting geographical extent or creating scatter plots.
*   A simple inventory dataset (like greenhouse gas emissions) might be just a time series in a CSV, needing very different plotting logic than a 3D model output.

The generic plotting tools know *how* to draw a map or a line graph if you give them the right 2D data and coordinates. But extracting exactly the right 2D slice, handling model-specific coordinates, or calculating metrics unique to that data type requires specialized knowledge.

This is where **Source/Model Specific Logic** comes in. It's the layer in eViz that provides the "expert knowledge" for different types of data sources (like WRF, LIS, observational data, etc.).

## What is Source/Model Specific Logic?

Think of the **Source/Model Specific Logic** as the **specialized experts** in our visualization workshop. We have general tools (the Plotters) and an assembly line for data preparation (the Pipeline), but before the data goes *to* a plotting tool, it needs to be correctly prepared *by someone who understands this particular type of data*.

For instance:
*   When plotting a variable from WRF data, you need the "WRF expert" to figure out the actual latitude and longitude coordinates for a staggered grid.
*   When plotting satellite data, you need the "Observational data expert" to calculate the geographical bounding box of the data swath so the map zooms correctly.
*   When plotting a box plot from observational data, you need the "Observational data expert" to gather the data points correctly, potentially handling missing values or sampling large datasets.

In eViz, these experts are represented by **specialized Python classes**. These classes inherit from a **base class** (`GenericSource`) and implement methods or override behavior to handle the unique characteristics of their data source.

These specialized classes are responsible for:
1.  Knowing how to extract the correct *slice* of data (`_extract_xy_data`, `_extract_xt_data`, etc.) needed for a specific plot type (XY map, XT time series, etc.), considering the data's dimensions and the user's configuration (level, time, area average, etc.).
2.  Understanding their specific coordinate systems and how to get the correct coordinate values for plotting.
3.  Adding any necessary data processing steps *after* the generic pipeline but *before* plotting (like calculating differences for comparison plots, handling model-specific data structures).
4.  Orchestrating the use of the [Plotter Factory](07_plotter_factory_.md) and the chosen Plotter objects to generate the final images for *their specific data type*.

## Our Use Case: Plotting WRF vs. Observational Data

Let's say you want to plot temperature (`T2`) from a WRF model output file *and* temperature (`temperature`) from a separate observational dataset using an XY map plot (`xy`).

Your configuration might look something like this:

```yaml
# --- Snippet from a config file ---
inputs:
  - name: wrf_data.nc
    location: /path/to/wrf
    source: wrf # Explicitly specify the source type
    to_plot:
      T2: xy     # Plot WRF T2 as an XY map

  - name: obs_data.h5
    location: /path/to/obs
    source: obs # Explicitly specify the source type
    to_plot:
      temperature: xy # Plot Obs temperature as an XY map
# ...
outputs:
  plotting_backend: matplotlib
  # ... other settings ...
```

When you run eViz with this config, the Source/Model Specific Logic layer will handle the differences:

1.  The [Data Source Factory](04_data_source_factory_.md) ([Chapter 4](04_data_source_factory_.md)) will be asked to create instances for the `wrf` source and the `obs` source. It will create a `Wrf` object and an `ObsSource` object.
2.  The `Wrf` object will be responsible for orchestrating the plotting of `T2`. When it prepares the data for the XY plot, it will use its specific methods (`_extract_xy_data`, `_process_coordinates`) that know how to handle WRF's grid and dimensions.
3.  The `ObsSource` object will be responsible for orchestrating the plotting of `temperature`. When it prepares the data for the XY plot, it will use *its* specific methods (`_extract_xy_data`, `apply_extent_to_config`) that know how to handle observational data, such as calculating the map extent from the data itself.
4.  Both objects will use the same generic [Data Processing Pipeline](05_data_processing_pipeline_.md) components for initial processing (like unit conversion) and the same [Plotter Factory](07_plotter_factory_.md) and [Plotter Abstraction](06_plotter_abstraction_.md) to create the `MatplotlibXYPlotter`. But the *inputs* they give to the plotter (the 2D data array, the coordinate arrays, the map extent settings) will be tailored by their source-specific logic.

The core plotting code doesn't need to know the difference between WRF and Obs data; it just receives a prepared 2D array and associated information. The source-specific classes do the preparation.

## How Source/Model Specific Logic Works (High-Level)

This logic primarily resides in classes that inherit from `GenericSource`. These classes are the "root instances" created by the [Data Source Factory](eviz/models/source_factory.py) as seen in [Chapter 1: Autoviz Application](01_autoviz_application_.md).

Here's a simplified flow, focusing on the `Autoviz` application handing off control to the specific model instance:

```{mermaid}
sequenceDiagram
    participant AutovizApp as Autoviz
    participant SourceFactory as Source Factory
    participant ConfigMgr as ConfigManager
    participant GriddedModel as GriddedSource Instance
    participant ObsModel as ObsSource Instance
    participant DataArray as xarray.DataArray
    participant PlotterFactory as Plotter Factory
    participant MatplotlibXY as MatplotlibXYPlotter

    AutovizApp->>ConfigMgr: Initialized Config
    AutovizApp->>SourceFactory: get_factory_from_user_input(['gridded', 'obs'])
    SourceFactory-->>AutovizApp: Return GriddedSourceFactory, ObsSourceFactory
    AutovizApp->>GriddedModel: GriddedSourceFactory.create_root_instance(ConfigMgr)
    GriddedModel-->>AutovizApp: Return GriddedSource object
    AutovizApp->>ObsModel: ObsSourceFactory.create_root_instance(ConfigMgr)
    ObsModel-->>AutovizApp: Return ObsSource object

    AutovizApp->>GriddedModel: Call run() / () # Start plotting for Gridded
    GriddedModel->>ConfigMgr: Get plot tasks (e.g., T2: xy)
    GriddedModel->>GriddedModel: _process_single_plots() / process_plot()
    GriddedModel->>GriddedModel: _prepare_field_to_plot(T2) # Specific slicing for T2
    GriddedModel->>GriddedModel: _process_coordinates() # Specific coordinate handling
    GriddedModel->>PlotterFactory: create_plotter('xy', 'matplotlib')
    PlotterFactory-->>GriddedModel: Return MatplotlibXYPlotter

    GriddedModel->>MatplotlibXY: plot(config, prepared_T2_data)
    MatplotlibXY->>MatplotlibXY: (Draws using Matplotlib)
    GriddedModel->>MatplotlibXY: save(filename)
    MatplotlibXY-->>GriddedModel: Plot saved

    AutovizApp->>ObsModel: Call run() / () # Start plotting for Obs
    ObsModel->>ConfigMgr: Get plot tasks (e.g., temperature: xy)
    ObsModel->>ObsModel: _process_single_plots() / process_plot()
    ObsModel->>ObsModel: _prepare_field_to_plot(temperature) # Specific slicing for temperature
    ObsModel->>ObsModel: apply_extent_to_config() # Specific extent calc
    ObsModel->>PlotterFactory: create_plotter('xy', 'matplotlib')
    PlotterFactory-->>ObsModel: Return MatplotlibXYPlotter

    ObsModel->>MatplotlibXY: plot(config, prepared_temperature_data)
    MatplotlibXY->>MatplotlibXY: (Draws using Matplotlib)
    ObsModel->>MatplotlibXY: save(filename)
    MatplotlibXY-->>ObsModel: Plot saved
    AutovizApp-->>User: Processed complete

```

This diagram shows that the `Autoviz` application delegates the main `run()` or `()` call to the specific "root instance" (like `GriddedSource` or `ObsSource`) obtained from the Source Factory. That instance then takes over, orchestrating the plotting process for its data source, calling its *own* specialized methods to prepare the data before ultimately requesting a generic Plotter from the [Plotter Factory](07_plotter_factory_.md).

## Inside the Code: Base and Generic Sources

The foundation for these specialized classes is laid by two base classes:

1.  **`BaseSource`**: (in `eviz/models/base.py`) This is a simple abstract base class. It defines the absolute minimum interface that *any* source handler class *must* provide, like a `logger` property and a `plot()` method (though in practice, the `plot` logic is often implemented in `GenericSource`). It also has abstract methods related to data sources (`add_data_source`, `get_data_source`, `load_data_sources`), but these are largely handled by the Pipeline now.

    ```python
    # --- File: eviz/models/base.py (simplified) ---
    from abc import ABC, abstractmethod
    import logging
    from eviz.lib.config.config_manager import ConfigManager

    @dataclass
    class BaseSource(ABC):
        config_manager: ConfigManager # All sources need config

        @property
        @abstractmethod
        def logger(self) -> logging.Logger:
            """Abstract property for the logger instance."""
            pass # Must be implemented by subclasses

        @abstractmethod
        def plot(self):
            """Abstract method for the top-level plotting routine."""
            pass # Must be implemented by subclasses

        # ... other abstract methods, often handled by pipeline now ...
    ```

    This class primarily ensures that any source handler has access to the `ConfigManager` and logging, and defines the entry point (`plot`) for the visualization process.

2.  **`GenericSource`**: (in `eviz/models/source_base.py`) This class inherits from `BaseSource`. It provides the **default implementation** for plotting orchestration and common data preparation steps that apply to many (especially gridded) sources. It handles the main loop over plot tasks (`map_params`), calls the [Plotter Factory](07_plotter_factory_.md), and contains helper methods for extracting data slices and preparing them for the plotter.

    ```python
    # --- File: eviz/models/source_base.py (simplified) ---
    from dataclasses import dataclass
    import logging
    import xarray as xr # GenericSource works heavily with xarray
    from eviz.models.base import BaseSource
    from eviz.lib.autoviz.plotting.factory import PlotterFactory # Uses the plotter factory
    from eviz.lib.autoviz.figure import Figure # Manages matplotlib figures/axes

    @dataclass
    class GenericSource(BaseSource):
        config_manager: ConfigManager

        def __post_init__(self):
            super().__post_init__() # Call parent init
            self.logger.info("Initializing GenericSource")
            # Get config details for easy access
            self.config = self.config_manager.config 
            self.app = self.config_manager.app_data
            self.specs = self.config_manager.spec_data
            # ... other init ...
            self.data2d_list = [] # Used for comparisons

        def plot(self):
            """Generate plots based on current configuration."""
            self.logger.info("Generate plots (GenericSource).")
            if not self.config_manager.spec_data:
                # Handle simple plots without SPECS (often delegated)
                pass # Simplified
            elif self.config_manager.compare or self.config_manager.compare_diff or self.config_manager.overlay:
                 # Handle comparison/overlay plots (delegated)
                 self.process_side_by_side_plots() # Example delegation
            else:
                # Handle standard single plots
                self.process_single_plots() # Example delegation

        def process_single_plots(self):
            """Generate single plots for each field."""
            self.logger.info("Generating single plots (GenericSource)")
            # Loop through each field and plot type specified in config (map_params)
            for idx, params in self.config_manager.map_params.items():
                field_name = params.get('field')
                if not field_name: continue
                filename = params.get('filename')
                # Get the data array for this field from the pipeline
                data_source = self.config_manager.pipeline.get_data_source(filename)
                if not data_source or not data_source.dataset or field_name not in data_source.dataset: continue
                field_data_array = data_source.dataset[field_name]
                # Get requested plot types (e.g., ['xy', 'xt'])
                plot_types = params.get('to_plot', ['xy'])
                if isinstance(plot_types, str): plot_types = [pt.strip() for pt in plot_types.split(',')]

                # For each plot type requested for this field:
                for plot_type in plot_types:
                    self.logger.info(f"Plotting {field_name}, {plot_type} plot")
                    # orchestrate the actual plotting process
                    self.process_plot(field_data_array, field_name, idx, plot_type)

        def process_plot(self, data_array, field_name, file_index, plot_type):
            """Process a single plot for the given field and type."""
            self.register_plot_type(field_name, plot_type) # Keep track of plot type for this field
            figure = Figure.create_eviz_figure(self.config_manager, plot_type) # Manage figures/axes
            self.config_manager.ax_opts = figure.init_ax_opts(field_name) # Init plot options

            # *** This is a key method where specialized classes often override or call helpers ***
            field_to_plot = self._prepare_field_to_plot(data_array, 
                                                        field_name,
                                                        file_index, 
                                                        plot_type, 
                                                        figure,
                                                        time_level=self.config_manager.ax_opts.get('time_lev', 0))

            if field_to_plot:
                # *** Use the Plotter Factory to get the right plotter ***
                plot_result = self.create_plot(field_name, field_to_plot)
                # *** Use a utility to print/save the plot via the plotter's methods ***
                pu.print_map(self.config_manager, plot_type, self.config_manager.findex, plot_result)

        def create_plot(self, field_name, data_to_plot):
             """Create a plotter and generate the plot using it."""
             backend = getattr(self.config_manager, 'plot_backend', 'matplotlib')
             plot_type = self.get_plot_type(field_name)
             # Call the factory from Chapter 7!
             plotter = PlotterFactory.create_plotter(plot_type, backend)
             if plotter is None: return None
             return plotter.plot(self.config_manager, data_to_plot)

        def _prepare_field_to_plot(self, data_array, field_name, file_index, plot_type, figure, time_level, level=None):
             """Prepare the 2D data array and coordinates for plotting."""
             # Get dimension names based on plot type (e.g., 'xc', 'yc' for 'xy')
             dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
             data2d = None

             # Call specialized methods to extract the correct 2D slice
             if 'xy' in plot_type or 'polar' in plot_type:
                 # *** This calls a method often implemented/overridden in subclasses ***
                 data2d = self._extract_xy_data(data_array, level=level, time_lev=time_level)
             # ... elif for yz, xt, tx, etc. calling _extract_yz_data, _extract_xt_data ...
             elif 'box' in plot_type:
                 # *** Box plots need special data prep, often overridden in ObsSource/Ghg ***
                 data2d = self._extract_box_data(data_array, time_lev=time_level)
             # ... handle other plot types ...

             if data2d is None: return None

             # *** Process coordinates, potentially calling methods overridden by subclasses ***
             # ... logic to get x, y coordinates from data2d based on dim1_name, dim2_name ...
             # ... potentially call a method like _process_coordinates(data2d, dim1_name, dim2_name) ...

             # Return the prepared data tuple needed by plotters
             return data2d, x_coords, y_coords, field_name, plot_type, file_index, figure

        # ... other helper methods like _extract_xy_data (default implementation),
        # _extract_yz_data, _extract_xt_data, _set_time_config, process_comparison_plots, etc.
    ```

`GenericSource` is where the core plotting workflow lives. It iterates through the plot requests and uses its helper methods (like `_prepare_field_to_plot`) to get the data ready. Crucially, many of these helper methods (`_extract_xy_data`, `_extract_xt_data`, `_prepare_field_to_plot` itself, and methods for comparison plots) are designed to be **overridden or extended by subclasses** to inject source-specific logic.

## Inside the Code: Specialized Source Classes

Classes like `GriddedSource`, `ObsSource`, `Wrf`, `Lis`, `Grib`, `Ghg`, `Airnow`, `Omi`, and `Crest` inherit from `GenericSource` (or sometimes from each other) and provide the unique handling.

Let's look at a few examples:

### `GriddedSource` (`eviz/models/gridded_source.py`)

This is often the *first* level of specialization beyond `GenericSource`. It provides implementations specifically for regular gridded data, building upon `GenericSource`'s framework. It has default implementations for `_extract_xy_simple`, `_extract_yz_simple`, `_process_xy_plot`, etc.

```python
# --- File: eviz/models/gridded_source.py (simplified) ---
from dataclasses import dataclass
import logging
import xarray as xr
from eviz.models.source_base import GenericSource # Inherits from GenericSource

@dataclass
class GriddedSource(GenericSource):
    """Specialized functionality for handling gridded ESM data."""

    def __post_init__(self):
        super().__post_init__()
        self.logger.info("Initializing GriddedSource")
        # Often initializes components needed for gridded processing, like DataProcessor
        # self.processor = DataProcessor(self.config_manager)

    def _extract_xy_data(self, data_array: xr.DataArray, level, time_lev):
        """
        Extract XY slice (latlon) from a DataArray.
        (Default gridded implementation)
        """
        if data_array is None: return None
        d_temp = data_array.copy()
        tc_dim = self.config_manager.get_model_dim_name('tc') # Get standard time dim name
        zc_dim = self.config_manager.get_model_dim_name('zc') # Get standard vertical dim name

        # Default logic for selecting time/level dimensions if they exist
        if tc_dim and tc_dim in d_temp.dims:
             if isinstance(time_lev, int):
                 d_temp = d_temp.isel({tc_dim: time_lev}) # Select specific time
        if zc_dim and zc_dim in d_temp.dims:
             if level is not None:
                  d_temp = d_temp.sel({zc_dim: level}, method='nearest') # Select nearest level

        data2d = d_temp.squeeze() # Remove dimensions of size 1
        # ... More logic to ensure exactly 2 dimensions (XY) ...

        # Apply units conversion (delegated to a utility function)
        return apply_conversion(self.config_manager, data2d, data_array.name)

    # ... implements/overrides _process_xy_plot, _process_xt_plot, etc.
    # which orchestrate getting data and calling create_plot ...
```

`GriddedSource` provides the basic logic for handling time and vertical level selections common in gridded model data within its `_extract_xy_data` method. Other gridded models might inherit from *it*.

### `ObsSource` (`eviz/models/obs_source.py`)

This class is for observational data, which often has different characteristics. It overrides methods to handle things like determining geographical extent from potentially scattered or swath data.

```python
# --- File: eviz/models/obs_source.py (simplified) ---
from dataclasses import dataclass
import logging
import xarray as xr
from eviz.models.source_base import GenericSource # Inherits from GenericSource

@dataclass
class ObsSource(GenericSource):
    """Specialized functionality for handling observational data."""

    def __post_init__(self):
        super().__post_init__()
        self.logger.info("Initializing ObsSource")

    def get_data_extent(self, data_array: xr.DataArray) -> list:
        """
        Extract the geographical extent (bounding box) from an xarray DataArray.
        (Specific logic for observational data)
        """
        if data_array is None: return [-180, 180, -90, 90]
        try:
            xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
            yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'
            
            # Look for 1D or 2D coordinate arrays, calculate min/max
            if xc_dim in data_array.coords and yc_dim in data_array.coords:
                 lon_vals = data_array[xc_dim].values
                 lat_vals = data_array[yc_dim].values
                 lon_min = np.nanmin(lon_vals) # Use numpy nanmin
                 lat_min = np.nanmin(lat_vals)
                 # ... calculate lon_max, lat_max ...
                 return [lon_min, lon_max, lat_min, lat_max]
            # ... additional logic to find extent in attributes or other places ...
        except Exception as e:
            self.logger.error(f"Error extracting extent: {e}")
            return [-180, 180, -90, 90]

    def apply_extent_to_config(self, data_array: xr.DataArray, field_name: str = None):
        """Extract extent and apply it to the config for the plotter."""
        extent = self.get_data_extent(data_array)
        self.config_manager.ax_opts['extent'] = extent # Set the map extent in config
        # Also calculate and set central lon/lat
        self.config_manager.ax_opts['central_lon'] = (extent[0] + extent[1]) / 2
        self.config_manager.ax_opts['central_lat'] = (extent[2] + extent[3]) / 2

    def _prepare_field_to_plot(self, data_array, field_name, file_index, plot_type, figure, time_level, level=None):
        """
        Prepare the 2D data array and coordinates for plotting.
        (Overrides GenericSource to add extent handling for XY/Polar plots)
        """
        # Call the parent method first to get the 2D slice
        field_to_plot_tuple = super()._prepare_field_to_plot(
            data_array, field_name, file_index, plot_type, figure, time_level, level)

        if field_to_plot_tuple is None: return None

        data2d, x_coords, y_coords, field_name, plot_type, file_index, figure = field_to_plot_tuple

        # For XY or Polar plots, apply the calculated extent
        if 'xy' in plot_type or 'polar' in plot_type:
            if data2d is not None:
                 # *** Add the ObsSource specific step ***
                 self.apply_extent_to_config(data2d, field_name)

        return data2d, x_coords, y_coords, field_name, plot_type, file_index, figure

    # ... adds/overrides methods like _process_box_plot, _extract_box_data,
    #     _process_pearson_plot, _extract_pearson_data, etc.
```

`ObsSource` adds methods like `get_data_extent` and modifies `_prepare_field_to_plot` to ensure that map plots created from observational data automatically zoom to the data's coverage area. It also includes logic for specific observational plot types like `box` and `pearson` within its `_process_..._plot` methods, often implemented by adding `_extract_..._data` helpers.

### `Wrf` (`eviz/models/esm/wrf.py`) and `Lis` (`eviz/models/esm/lis.py`)

These classes specialize in specific Earth System Models. They often inherit from `GenericSource` (or an intermediate base like `NuWrf`) and override methods related to coordinate handling, vertical level selection, or time averaging to match the specific conventions of their model output files.

```python
# --- File: eviz/models/esm/wrf.py (simplified) ---
from dataclasses import dataclass
import xarray as xr
from eviz.models.esm.nuwrf import NuWrf # Often inherit from intermediate base

@dataclass
class Wrf(NuWrf): # Or GenericSource depending on class hierarchy
    """Define WRF specific model data and functions."""

    def __post_init__(self):
        super().__post_init__()
        self.logger.info("Initializing WrfSource")
        self.source_name = 'wrf'
        # Often calls model-specific domain initialization
        # self._init_wrf_domain() # Needs access to loaded data

    def _init_wrf_domain(self, data_source):
         """WRF-specific initialization (e.g., processing sigma levels)."""
         # Accesses data_source.dataset to get variables like 'P_TOP', 'ZNW', 'ZNU'
         # Calculates pressure levels (self.levs, self.levf) from sigma coordinates
         pass # Simplified

    def _extract_xy_data(self, data_array: xr.DataArray, level, time_lev):
         """
         Extract XY slice for WRF data.
         (Overrides parent to handle WRF specifics like time dimension name)
         """
         if data_array is None: return None
         d_temp = data_array.copy()
         # WRF uses 'Time' dimension name
         tc_dim = 'Time' if 'Time' in d_temp.dims else super()._get_time_dimension_name(d_temp)
         zc_dim = self.config_manager.get_model_dim_name('zc') # Get standard vertical dim name

         # Apply time selection using the WRF 'Time' dim
         if tc_dim in d_temp.dims:
             if isinstance(time_lev, int):
                 d_temp = d_temp.isel({tc_dim: time_lev})

         # Apply level selection - may need WRF-specific vertical coordinate handling
         if zc_dim and zc_dim in d_temp.dims:
              if level is not None:
                   # *** WRF-specific vertical level selection ***
                   # Calls a method that maps requested pressure level to WRF model level index
                   d_temp = self._apply_vertical_level_selection(d_temp, data_array.name, level)

         data2d = d_temp.squeeze()
         # ... more logic ...
         return apply_conversion(self.config_manager, data2d, data_array.name)

    def _apply_vertical_level_selection(self, data2d, field_name, level):
        """
        Apply vertical level selection for WRF data, handling staggered grids
        and converting pressure levels to model indices.
        (Specific WRF implementation)
        """
        zname = self.get_field_dim_name(data2d, 'zc', field_name) # Get actual dim name

        if zname and zname in data2d.dims and level is not None:
             # Use the pre-calculated WRF pressure levels (self.levs)
             # to find the index corresponding to the requested 'level' (pressure)
             difference_array = np.absolute(self.levs - level)
             index = difference_array.argmin() # Find index of nearest level
             self.logger.debug(f"Selecting WRF model level index {index} for pressure level {level}")
             return data2d.isel({zname: index}).squeeze()

        return data2d # Return as is if no vertical dim or level not specified

    # ... overrides _process_coordinates to return WRF grid coordinates,
    #     _extract_xt_data with WRF time handling, etc.
```

`Wrf` demonstrates overriding `_extract_xy_data` to use its own logic for vertical level selection (`_apply_vertical_level_selection`), which needs to map user-requested pressure levels to the internal WRF sigma coordinate indices using data processed during initialization (`_init_wrf_domain`). It also overrides `_process_coordinates` to correctly provide the staggered grid coordinates to the plotter.

### `Crest` (`eviz/models/esm/crest.py`)

`Crest` provides an interesting example of a source handler that doesn't implement all plotting logic itself, but instead **delegates** to instances of `GriddedSource` and `ObsSource` based on the *characteristics of the data* it encounters.

```python
# --- File: eviz/models/esm/crest.py (simplified) ---
from dataclasses import dataclass
import logging
import xarray as xr
from eviz.models.source_base import GenericSource
from eviz.models.gridded_source import GriddedSource # Needs Gridded handler
from eviz.models.obs_source import ObsSource       # Needs Obs handler

@dataclass
class Crest(GenericSource):
    """Source handler for CREST data, which can be gridded or observational."""

    def __post_init__(self):
        super().__post_init__()
        self.logger.info("Initializing CrestSource")
        self.source_name = 'crest'
        # *** Create instances of the handlers it might need to delegate to ***
        self.gridded_handler = GriddedSource(self.config_manager)
        self.obs_handler = ObsSource(self.config_manager)

    def _is_observational_data(self, data_array):
        """
        Determine if the data array should be treated as observational data.
        (Checks for 2D coords, irregular grids, attributes, limited extent)
        """
        # ... implementation checks characteristics of data_array ...
        pass # Simplified, see full code snippet

    def process_plot(self, data_array, field_name, file_index, plot_type):
        """
        Process a plot, delegating to the appropriate handler based on data type.
        (Overrides GenericSource.process_plot)
        """
        self.register_plot_type(field_name, plot_type)
        figure = Figure.create_eviz_figure(self.config_manager, plot_type)
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        # *** Decide which handler to use based on the data characteristics ***
        is_obs = self._is_observational_data(data_array)

        if is_obs:
            self.logger.info(f"Delegating {field_name} plot to ObsSource handler")
            handler = self.obs_handler
            # For Obs data, apply extent early as needed by ObsSource methods
            handler.apply_extent_to_config(data_array, field_name) # Explicitly call ObsSource logic
        else:
            self.logger.info(f"Delegating {field_name} plot to GriddedSource handler")
            handler = self.gridded_handler

        # *** Delegate the actual plot processing to the chosen handler ***
        # Call the corresponding method on the selected handler instance
        if plot_type == 'xy':
            handler._process_xy_plot(data_array, field_name, file_index, plot_type, figure)
        elif plot_type == 'box':
            handler._process_box_plot(data_array, field_name, file_index, plot_type, figure)
        # ... elif for other plot types calling handler's methods ...
        else:
             self.logger.warning(f"Plot type {plot_type} not handled by Crest delegation logic.")

        # Note: The handler's _process_..._plot method will internally call
        # handler._prepare_field_to_plot and handler.create_plot(...)
```

`Crest` checks the properties of the data (`_is_observational_data`) and then explicitly calls the corresponding plotting method (`_process_xy_plot`, `_process_box_plot`, etc.) on the appropriate handler instance (`self.gridded_handler` or `self.obs_handler`). This allows a single "crest" source type in the config to handle diverse data outputs produced by that framework.

### Other Specialized Classes

The other classes (`Grib`, `Ghg`, `Airnow`, `Omi`) follow similar patterns, inheriting from `GriddedSource` or `ObsSource` and overriding methods as needed:
*   `Grib`: Inherits from `GriddedSource`, overrides methods like `_prepare_field_to_plot` to handle GRIB-specific coordinate access and extent setting (`_set_grib_extents`).
*   `Ghg`: Inherits from `ObsSource`, adds methods like `process_data` to specifically read and structure CSV data into an `xarray.Dataset` (although the pipeline's `CSVDataSource` might do this now), and overrides methods like `_prepare_field_to_plot` to handle time series and uncertainty data extraction for plots like `xt` and `bar`.
*   `Airnow`: Inherits from `ObsSource`, likely overrides `_prepare_field_to_plot` to handle its specific CSV/point data structure and setting a fixed geographic extent for plots over the US.
*   `Omi`: Inherits from `ObsSource`, overrides methods like `process_single_plots` and `_prepare_field_to_plot` to handle unpacking hierarchical HDF5 structures (like those from satellite data) and ensuring correct latitude/longitude arrays are used (via a helper like `extract_field_with_coords`).

In summary, these specialized classes are the workhorses that bridge the gap between the generic `xarray.Dataset` and the specific requirements of a particular plotting task for a particular type of data, using the generic tools ([Data Processing Pipeline](05_data_processing_pipeline_.md), [Plotter Factory](07_plotter_factory_.md), [Plotter Abstraction](06_plotter_abstraction_.md)) coordinated by the `GenericSource` framework.

## Where are these Classes Created and Used?

As shown in the high-level diagram and referenced in [Chapter 1](01_autoviz_application_.md) and [Chapter 4](04_data_source_factory_.md), these `Source/Model Specific Logic` classes are the "root instances" created by the **Source Factory** (`eviz/models/source_factory.py`).

```python
# --- File: eviz/models/source_factory.py (simplified) ---
from dataclasses import dataclass
from eviz.lib.config.config_manager import ConfigManager
# Import the specialized classes
from eviz.models.gridded_source import GriddedSource
from eviz.models.esm.wrf import Wrf
from eviz.models.obs_source import ObsSource
# ... import other specific source classes ...


class BaseSourceFactory: # Abstract base for factories
    # ... abstractmethod create_root_instance ...
    pass

@dataclass
class GriddedSourceFactory(BaseSourceFactory):
    """Factory for creating GriddedSource instances."""
    def create_root_instance(self, config_manager: ConfigManager):
        # Factory knows *how* to create this specific type
        return GriddedSource(config_manager) # Creates the GriddedSource object

@dataclass
class WrfFactory(BaseSourceFactory):
    """Factory for creating Wrf instances."""
    def create_root_instance(self, config_manager: ConfigManager):
        return Wrf(config_manager) # Creates the Wrf object

@dataclass
class ObsSourceFactory(BaseSourceFactory):
    """Factory for creating ObsSource instances."""
    def create_root_instance(self, config_manager: ConfigManager):
        return ObsSource(config_manager) # Creates the ObsSource object

# ... other factories for Lis, Ghg, Crest, etc. ...


# Function used by Autoviz to get the right factory
def get_factory_from_user_input(source_names):
    """Looks up the factory for a given source name."""
    # Simple dictionary lookup mapping source names (from config) to factory classes
    factory_registry = {
        'gridded': GriddedSourceFactory,
        'wrf': WrfFactory,
        'obs': ObsSourceFactory,
        'crest': CrestFactory,
        'ghg': GhgFactory,
        # ... etc. ...
    }
    
    factories = []
    for name in source_names:
        if name in factory_registry:
            factories.append(factory_registry[name]()) # Create an instance of the FACTORY
        else:
            raise ValueError(f"Unknown source type: {name}")
    return factories
```

The `get_factory_from_user_input` function (called by `Autoviz` in Chapter 1) finds the *factory* class for the requested source name (e.g., `GriddedSourceFactory` for 'gridded'). Then, the `Autoviz` object calls `create_root_instance` on *that factory* to get the actual source handler object (`GriddedSource` instance). It's this `GriddedSource` or `Wrf` or `ObsSource` instance that the `Autoviz.run()` method then calls (`model()`), initiating the specialized plotting logic.

## Summary

In this chapter, we explored the **Source/Model Specific Logic** layer in eViz.

*   Different scientific data sources (models like WRF, LIS; observations like OMI, Airnow; generic gridded data) have unique structures and require specialized handling before plotting.
*   eViz uses a hierarchy of classes (`BaseSource` -> `GenericSource` -> Specialized Sources like `GriddedSource`, `ObsSource`, `Wrf`, `Lis`, `Grib`, `Ghg`, `Airnow`, `Omi`, `Crest`) to manage this.
*   `GenericSource` provides the common framework for plotting orchestration and default data preparation steps.
*   Specialized classes inherit from `GenericSource` (or other intermediate classes) and override or add methods (`_extract_xy_data`, `_process_coordinates`, `get_data_extent`, `_extract_box_data`, `process_data`, etc.) to implement the logic specific to their data type.
*   These specialized classes use their expert knowledge to prepare the correct data and coordinate inputs before requesting a generic Plotter from the [Plotter Factory](07_plotter_factory_.md).
*   These source-specific classes are the main objects ("root instances") created by the Source Factory and are responsible for the overall visualization process for their data type, working with the [Configuration Management](02_configuration_management_.md), [Data Processing Pipeline](05_data_processing_pipeline_.md), and Plotters.

Understanding this layer is key to adding support for new data types or customizing how existing data types are handled and plotted.

Now that you know how eViz uses source-specific experts to process and plot data, you might be wondering how you can inspect your data files to understand their structure (like dimension names, variable names, attributes) so you know what to put in your configuration or how to implement a new source handler. That's where the **Metadata Tool (metadump)** comes in handy.

[Metadata Tool (metadump)](09_metadata_tool__metadump__.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)