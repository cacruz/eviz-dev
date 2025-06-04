# Chapter 8: Model Implementations

Welcome back! Over the last few chapters, we've built a picture of how eViz works. The **Autoviz Application** ([Chapter 1: Autoviz Application](01_autoviz_application_.md)) orchestrates everything, guided by **Configuration Files** ([Chapter 2: Configuration Management](02_configuration_management_.md)). It figures out how to read your data using **Data Source Abstraction** and the **Data Source Factory** ([Chapters 3 & 4](03_data_source_abstraction_.md), [Chapter 4](04_data_source_factory_.md)), understands what's inside using **Metadata Handling** ([Chapter 5: Metadata Handling](05_metadata_handling_.md)), and prepares it for plotting using the **Data Processing Pipeline** ([Chapter 6: Data Processing Pipeline](06_data_processing_pipeline_.md)). Finally, the **Plotting Engine** ([Chapter 7: Plotting Engine](07_plotting_engine_.md)) takes this prepared data and draws the visualizations.

The result of the pipeline is a nice, clean **xarray Dataset**. The plotting engine knows how to draw standard plot types (maps, time series) from xarray data. So, are we done? Almost!

Here's the final piece of the puzzle: Data from different scientific models (like GEOS, WRF, LIS) or observation inventories (like Airnow, OMI) often have *unique* characteristics or require *specific* sequences of steps that aren't totally generic.

*   Maybe GEOS data stores a particular variable name differently.
*   Maybe WRF data uses a special kind of vertical coordinate or has dimensions that need specific handling ('staggered' dimensions).
*   Maybe LIS data has quirks with its grid coordinates (like NaNs) that need fixing *before* plotting.
*   Maybe OMI satellite data needs special handling to extract the main data variable from a complex HDF5 structure and reconstruct standard latitude/longitude.
*   Maybe Airnow data needs specific logic to handle station locations and plot them as points.
*   Maybe you need to perform a calculation (like a difference or sum) that is commonly needed *only* for data from a specific model.

The generic **Data Processing Pipeline** handles common tasks, but it doesn't know about all these specific nuances. The **Plotting Engine** draws based on standard data structures, but it doesn't know *how* to get the exact slice or transformation needed for a specific model's quirky data.

This is where **Model Implementations** come in.

## Model Implementations: The Specialized Chefs

Imagine the **Data Processing Pipeline** gives you perfectly prepared ingredients (like chopped vegetables, measured flour, etc.) as a standard output – the **xarray Dataset**.

But making a specific dish, say, a complex layered cake (visualizing WRF data) or a delicate sushi roll (visualizing OMI data), requires a specialized chef who knows the *exact* steps, temperatures, and order of operations unique to *that* dish, even starting from the same prepped ingredients.

In eViz, **Model Implementations** (often just called "Models" in the code, e.g., `Wrf`, `Lis`, `Geos`, `Airnow`, `Omi`) are these specialized chefs. They are Python classes specifically designed to handle the unique requirements of data from a particular source type.

Their main job is to:

1.  Know how to work with the **xarray Dataset** obtained from the pipeline *for their specific data type*.
2.  Apply any necessary **source-specific logic** that the generic pipeline doesn't cover. This might involve:
    *   Selecting specific variables or slices in a way unique to that model's data structure.
    *   Handling model-specific coordinate systems or dimension names.
    *   Performing calculations or transformations specific to that model's data.
    *   Preparing the data *exactly* in the format (often a 2D slice and its coordinates) that the **Plotting Engine** expects for a particular plot type.
3.  Orchestrate the calls to the **Plotting Engine**, feeding it the prepared data and any necessary model-specific plot details (like map extents calculated from model data).

They act as the bridge between the generic **Data Processing Pipeline** and the generic **Plotting Engine**, applying the "recipe" unique to their data source type.

## How Model Implementations Fit In (The Flow)

Let's see how these specialized chefs fit into the overall flow. Remember, the **Autoviz Application** is the conductor.

```{mermaid}
sequenceDiagram
    participant A as Autoviz Object
    participant SF as Source Factory
    participant Model as Specific Model (e.g., Omi)
    participant CM as ConfigManager
    participant DP as DataPipeline
    participant XD as xarray Dataset (Processed)
    participant PE as Plotting Engine

    A->>CM: Get Configuration (includes source type, e.g., 'omi')
    CM-->>A: ConfigManager object
    A->>SF: Get Factory for 'omi' source type
    SF-->>A: OmiFactory class
    A->>Model: Use OmiFactory to create Omi() instance (passing ConfigManager)
    Model->>DP: Tell DataPipeline to load/process data (using config)
    DP-->>Model: Processed xarray Dataset(s)
    Model->>Model: Apply Omi-specific data processing (e.g., restructure HDF5 vars, extract coords)
    Model->>PE: Call Plotting Engine with prepared data and plot info
    PE->>PE: Generate Plot
    PE-->>Model: Plotting Complete
    Model-->>A: Task Complete
```

As shown in the diagram:
1.  The **Autoviz Application** determines the source type from the configuration.
2.  It asks the **Data Source Factory** ([Chapter 4: Data Source Factory](04_data_source_factory_.md)) to provide the correct "root instance" for that source type. The Factory doesn't just create a generic `DataSource` object here; it creates the top-level **Model Implementation** object (`Omi` in this example).
3.  This specific **Model** object (`Omi` instance) receives the `ConfigManager`.
4.  The **Model** then tells the **Data Processing Pipeline** ([Chapter 6: Data Processing Pipeline](06_data_processing_pipeline_.md)) to load and process the data files specified in the configuration.
5.  The **Model** receives the processed **xarray Datasets** from the pipeline.
6.  Crucially, the **Model** then applies its *own* specialized logic to the data (e.g., extracting the right part of the OMI dataset, potentially fixing coordinates, or applying unit conversions *after* the generic pipeline processing if needed in a specific way).
7.  Finally, the **Model** object calls the **Plotting Engine** ([Chapter 7: Plotting Engine](07_plotting_engine_.md)), passing it the *specifically prepared* data slice and any plotting parameters unique to this data type (like the correct map extent).

The `BaseSource` class (`eviz/models/base.py`) acts as the blueprint for all Model Implementations, requiring them to have methods like `plot()` (the main method often called by `Autoviz`) and methods for handling data sources.

The `GenericSource` class (`eviz/models/source_base.py`) provides common functionality inherited by many models, such as initializing the ConfigManager, setting up multiprocessing, and containing the generic logic to call different Plotter types (`_single_plots`, `_comparison_plots`, etc.) based on the configuration (`self.config_manager.spec_data`).

## Peeking at the Code

Let's look at how this specialized logic appears in the code.

First, let's see the base `BaseSource` class blueprint:

```python
# --- File: eviz/models/base.py (Simplified) ---
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.data.sources import DataSource


@dataclass
class BaseSource(ABC): # Abstract Base Class
    """This class defines an abstract base class for autoviz data sources (Models)."""
    config_manager: ConfigManager # All models need access to config

    @property
    @abstractmethod
    def logger(self) -> logging.Logger: # All models need a logger
        """Abstract property for the logger instance."""
        pass

    @abstractmethod
    def plot(self): # All models MUST implement a plot method
        """Abstract method for the top-level plotting routine."""
        pass

    # ... other abstract methods for data source handling ...

    def __post_init__(self):
        """Optional default behavior for initialization."""
        self.logger.info(f"Initializing {self.__class__.__name__}") # Logs which model is starting
```

This shows `BaseSource` defines the essential contract for any class acting as a top-level Model Implementation – it must take a `config_manager` and implement a `plot` method.

Many models inherit from `GenericSource` (`eviz/models/source_base.py`), which provides common plotting logic based on the configuration:

```python
# --- File: eviz/models/source_base.py (Simplified) ---
# ... imports ...
from eviz.lib.autoviz.plotter import SimplePlotter, ComparisonPlotter, SinglePlotter
from eviz.models.base import BaseSource # Inherits from BaseSource

@dataclass
class GenericSource(BaseSource):
    """This class defines gridded interfaces and plotting for all supported sources."""
    config_manager: ConfigManager

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__) # Provides a default logger

    def __post_init__(self):
        super().__post_init__() # Calls the parent's init (BaseSource)
        # Initialize common attributes like config, app, specs, plotting flags
        self.config = self.config_manager.config
        self.app = self.config_manager.app_data
        self.specs = self.config_manager.spec_data
        # ... other common initializations ...

    def __call__(self):
         """Allows the object to be called like a function."""
         self.plot() # Calling the model object runs its plot method

    def plot(self):
        """Top-level interface for gridded (NetCDF) maps (and other types)."""
        self.logger.info("Generate plots.")

        # This method decides which plotter type to use based on config
        if not self.config_manager.spec_data:
            plotter = SimplePlotter()
            self._simple_plots(plotter) # Call simple plotting logic
        elif self.config_manager.compare or self.config_manager.compare_diff:
             plotter = ComparisonPlotter(...)
             self._comparison_plots(plotter) # Call comparison logic
        # ... other checks for overlay, single plot ...
        else:
            plotter = SinglePlotter()
            self._single_plots(plotter) # Call single plotting logic

        # ... handles printing output file paths ...

    # These methods are often overridden or extended by specific models
    def _simple_plots(self, plotter): pass
    def _single_plots(self, plotter): pass
    def _comparison_plots(self, plotter): pass

    # ... other common methods ...
```

`GenericSource` provides the standard `plot()` method that acts as the main entry point. It checks the configuration (`self.config_manager`) to decide whether to make simple plots, comparison plots, or single plots using the appropriate `Plotter` class ([Chapter 7: Plotting Engine](07_plotting_engine_.md)). It calls internal methods like `_single_plots` which are often where specific Model Implementations add their unique data preparation steps before calling the `Plotter`. The `__call__` method makes the Model object executable directly (like `model_instance()`).

Now, let's look at a simplified example from the `Omi` Model (`eviz/models/obs/satellite/omi.py`). OMI data often comes in HDF5 files with complex internal paths (like `HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/ColumnAmountO3`) which are tricky for generic code to handle directly. The `Omi` model knows how to flatten these or find the right field.

```python
# --- File: eviz/models/obs/satellite/omi.py (Simplified) ---
import logging
# ... other imports ...
import xarray as xr
from dataclasses import dataclass
# ... other imports, including Figure and utils ...
from eviz.models.obs_source import ObsSource # Inherits from ObsSource, which inherits from GenericSource

@dataclass
class Omi(ObsSource): # Omi is a type of observation source model
    """ Define OMI satellite data and functions."""
    # ... attributes ...

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__) # Provides its logger

    def __post_init__(self):
        self.logger.info("Start init Omi Model")
        super().__post_init__() # Call parent init (ObsSource -> GenericSource)

    # Methods required by BaseSource (often implemented in parent ObsSource/GenericSource)
    # def add_data_source(...): pass
    # def get_data_source(...): pass
    # def load_data_sources(...): pass

    # Often, the model overrides _single_plots to handle its specific workflow
    def _single_plots(self, plotter):
        """Generate single plots for OMI data."""
        self.logger.info("Generating single plots for OMI")

        # Get processed data sources from the pipeline (loaded earlier)
        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for OMI plotting.")
            return

        # Loop through the items specified in the 'map_params' part of the config
        for idx, params in self.config_manager.map_params.items():
            field_name = params.get('field') # Variable name from config (e.g., 'ColumnAmountO3')
            filename = params.get('filename') # File path from config

            # Get the specific DataSource object from the pipeline
            data_source = self.config_manager.pipeline.get_data_source(filename)
            if not data_source or not hasattr(data_source, 'dataset') or data_source.dataset is None:
                 self.logger.warning(f"No data source found for {filename}")
                 continue

            # --- OMI-Specific Data Preparation ---
            # OMI data often has complex internal paths like 'HDFEOS/GRIDS/.../ColumnAmountO3'.
            # This model needs to simplify this structure.
            # The Data Source (HDF5DataSource) loaded the raw structure, now the Omi model
            # prepares it for plotting.
            new_names = {name: name.split('/')[-1] for name in data_source.dataset.data_vars}
            ds_short = data_source.dataset.rename(new_names) # Rename variables to just their final name

            if field_name not in ds_short:
                 self.logger.warning(f"Field {field_name} not found in OMI dataset after renaming.")
                 continue

            # --- Continue with generic plotting setup (often in parent class, but simplified here) ---
            self.config_manager.findex = idx # Set file index in config for plotting
            self.config_manager.pindex = idx # Set plot index in config for plotting
            self.config_manager.axindex = 0 # Default to first axis if multiple

            # Get plot types requested in config for this field
            plot_types = params.get('to_plot', ['xy']) # Default to 'xy' for OMI maps
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]

            # Process each requested plot type
            for plot_type in plot_types:
                # OMI specific: prepare the exact 2D data and coordinates tuple
                field_to_plot = self._get_field_to_plot(ds_short, field_name, idx, plot_type, None) # None for figure initially

                if field_to_plot:
                    # Use the plotter from the parent GenericSource._single_plots method
                    # (In real code, _single_plots would pass the plotter instance down)
                    # Simplified call:
                    self.logger.info(f"Plotting {field_name} as {plot_type} plot (via OMI model)")
                    figure = Figure.create_eviz_figure(self.config_manager, plot_type) # Create figure here
                    self.config_manager.ax_opts = figure.init_ax_opts(field_name) # Init axis options
                    # Call the plotter method from parent/passed instance
                    plotter.single_plots(self.config_manager, field_to_plot=field_to_plot, figure=figure) # Pass figure

                    # Print/Save the figure
                    pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)
                    plt.close(figure) # Close figure

        # ... handle gif creation ...


    # --- OMI-Specific Data Preparation Method ---
    def _get_field_to_plot(self, ds_short: xr.Dataset, field_name: str,
                           file_index: int, plot_type: str, figure, time_level=None,
                           level=None) -> tuple:
        """Prepare the OMI data array and coordinates for plotting."""
        if field_name not in ds_short:
            self.logger.error(f"Field {field_name} not found in OMI dataset.")
            return None

        data_array = ds_short[field_name]

        # OMI often doesn't have standard lat/lon coords attached in the data vars,
        # and might have specific fill values. Use a helper for this.
        data2d, lats, lons = extract_field_with_coords(data_array) # Call OMI-specific helper

        # OMI is typically level-less data, so level arg is ignored.
        # OMI is typically a single time slice per file, so time_level is ignored.

        # For XY plots, we need the 2D data array and the 1D lat/lon coordinate arrays
        if 'xy' in plot_type or 'sc' in plot_type or 'po' in plot_type:
            # Return data in the format expected by the generic plotter functions
            return data2d, lons, lats, field_name, plot_type, file_index, figure
        # ... logic for other OMI plot types if needed ...
        else:
            self.logger.error(f"Unsupported plot type {plot_type} for OMI data.")
            return None


# --- OMI-Specific Coordinate/Data Extraction Helper ---
# Often defined outside the main class or in a dedicated OMI utils file
def extract_field_with_coords(da: xr.DataArray, lat_bounds=(-90, 90), lon_bounds=(-180, 180)) -> tuple:
    """
    Extracts data array, reconstructs lat/lon coords assuming regular global grid,
    and masks invalid values.
    """
    if da is None:
         return None, None, None

    # Assuming a standard global grid structure for demonstration
    n_lat, n_lon = da.shape[-2], da.shape[-1] # Get dimensions from the DataArray shape
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    # Construct 1D coordinate arrays (simplified)
    lat_step = (lat_max - lat_min) / n_lat
    lon_step = (lon_max - lon_min) / n_lon
    lats = np.linspace(lat_max - lat_step / 2, lat_min + lat_step / 2, n_lat)
    lons = np.linspace(lon_min + lon_step / 2, lon_max - lon_step / 2, n_lon)

    # Handle potential multi-dimensional data (e.g., time dimension) by squeezing
    data2d = da.squeeze()

    # Mask invalid values using _FillValue attribute
    fill_value = data2d.attrs.get('_FillValue', None)
    if fill_value is not None:
        data2d = data2d.where(data2d != fill_value)

    return data2d, lats, lons # Return the prepared data and coords
```

This simplified `Omi` model code shows:
1.  It inherits from `ObsSource` (which inherits from `GenericSource`), getting common functionality.
2.  It overrides `_single_plots` to manage the plotting process specifically for OMI data.
3.  Inside `_single_plots`, after getting the data from the pipeline, it performs an OMI-specific step: simplifying the variable names (`data_source.dataset.rename`).
4.  It calls an internal helper method, `_get_field_to_plot`, which is *also* OMI-specific.
5.  `_get_field_to_plot` calls another helper `extract_field_with_coords` that knows how to get the main data and *reconstruct* the standard lat/lon coordinates from the OMI dataset's structure and dimensions. It also handles specific OMI metadata like `_FillValue` for masking.
6.  `_get_field_to_plot` then returns the data (`data2d`), longitudes (`lons`), and latitudes (`lats`) in the exact tuple format expected by the generic `SinglePlotter`'s underlying plotting functions (`_single_xy_plot` etc.).
7.  Finally, `_single_plots` calls the `plotter.single_plots` method (from the generic plotting engine), passing the `config_manager`, the prepared `field_to_plot` tuple, and the `figure` object.

This illustrates the core pattern: the specific Model Implementation (`Omi`) intercepts the data after the generic pipeline, applies its unique processing (`rename`, `extract_field_with_coords`), and then hands the *specifically prepared* data off to the generic **Plotting Engine**.

Other models like `Wrf` (`eviz/models/esm/wrf.py`) and `Lis` (`eviz/models/esm/lis.py`) have their own `_get_field_to_plot` or similar methods (`_process_coordinates` in WRF/LIS) that handle *their* specific coordinate systems, staggering, or dimension naming before returning data in the standard format for plotting. The `Airnow` model (`eviz/models/obs/inventory/airnow.py`) has logic to read CSV points and prepare them for scatter plotting, and the `Ghg` model (`eviz/models/obs/inventory/ghg.py`) focuses on processing CSV time series data and handling uncertainty for plotting time series plots.

## Benefits of Model Implementations

*   **Encapsulation:** All the specific knowledge and quirky logic for a particular data source are kept together in one place (its Model Implementation class).
*   **Separation of Concerns:** The generic pipeline doesn't need to know about WRF's staggered grid, and the generic plotting engine doesn't need to know how to extract OMI coordinates from an HDF5 file. The Model Implementation handles that translation.
*   **Modularity:** You can add support for a completely new data source (say, sea ice data) by creating a new Model Implementation class without modifying the core pipeline or plotting engine.
*   **Maintainability:** If a change is needed for how WRF vertical levels are handled, you only need to modify the `Wrf` class.

## Conclusion

In this chapter, you've learned about **Model Implementations**. These are specialized Python classes (like `Wrf`, `Lis`, `Geos`, `Airnow`, `Omi`) that act as expert handlers for data from specific scientific models or observation inventories. They receive processed data from the **Data Processing Pipeline**, apply any necessary source-specific logic (like handling unique coordinates, dimensions, or data structures), and prepare the data precisely for the **Plotting Engine**. They bridge the gap between the generic eViz components and the unique characteristics of different scientific datasets, making the system flexible and extensible.

This concludes the conceptual overview of the core eViz components. We've gone from the initial command line instruction to the final plotted image, understanding the role of each major system part.

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
