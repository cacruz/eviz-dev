# Chapter 5: Source Models

Welcome back to the eViz tutorial! In our journey so far, we've learned how the [Configuration System](01_configuration_system_.md) acts as eViz's control panel, how the [Autoviz Application Core](02_autoviz_application_core_.md) is the engine that reads those instructions and orchestrates the process, how the [Metadata Generator (metadump)](03_metadata_generator__metadump__.md) helps create configurations, and how the [Plotting Components](04_plotting_components_.md) are the artists that actually draw the visualizations.

Now, let's talk about the **Source Models**. If the Autoviz Application Core is the conductor of the orchestra, the Source Models are the **specialized section leaders** – the first violins, the lead trumpeter, the principal oboist. They don't conduct the whole piece, but they know *exactly* how to play their specific part and lead their section (the data for their type).

## What is the Problem Source Models Solve?

Earth System Models produce data in various formats (NetCDF, GRIB, HDF5) and structures (regular grids, regional grids, unstructured points). Observational data comes from satellites (often HDF5) or inventories (often CSV).

Imagine trying to write a single piece of code that knows how to:
*   Find the 'latitude' dimension in a generic global NetCDF file (might be 'lat', 'latitude', 'yc').
*   Find the 'latitude' dimension in a regional WRF file (might have a special name like 'XLAT').
*   Select a specific pressure level from a 3D variable in a global model.
*   Select a specific vertical *soil* layer from a 3D variable in a regional LIS model (which uses different vertical coordinates).
*   Extract time series data from a CSV file of CO2 concentrations.
*   Handle the specific projection or coordinate system needed for a regional WRF or LIS domain.
*   Find latitude/longitude coordinates which might be separate variables in one file type but dimensions in another.

This would make the main visualization engine (`Autoviz` core) incredibly complicated! It would need `if/else` statements checking the data source type at every step.

## Source Models: Specialized Data Handlers

This is exactly what Source Models are for. They are **specialized classes** designed to handle the unique characteristics of different **types of data sources**. Each Source Model knows the quirks of its data:

*   How to identify key dimensions (time, level, lat, lon).
*   How to extract specific slices (e.g., an XY map at a certain level, a YZ cross-section averaged over longitude, an XT time series at a specific point).
*   How to handle its specific coordinate system or grid.
*   How to prepare the data in a format ready for the [Plotting Components](04_plotting_components_.md).

Instead of one giant, complex worker, eViz has many smaller, specialized workers (the Source Models).

## Your Fifth Task: Plotting Data from a Specific Source Type

Let's go back to our `Temperature` example from `my_weather_data.nc`. In earlier chapters, we just said eViz plots it. But now, we know that the type of data matters.

When you run `autoviz.py` with a configuration like this (let's assume `my_weather_data.nc` is generic gridded NetCDF):

```yaml
# my_config.yaml
inputs:
  - name: my_weather_data.nc
    # ... other details ...
    to_plot:
      Temperature: xy # Generic gridded data often plots as a map

# ... other config sections ...
```

And you run eViz like this:

```bash
python autoviz.py -c ./configs -s gridded # Telling eViz it's 'gridded' source type
```

The `Autoviz` Application Core doesn't just know *what* to plot (`Temperature`, `xy`), it also knows *what type* of data (`gridded`). Based on the `-s gridded` flag (or the `source` type specified in config), it will use the [Data Source Factory](08_data_source_factory_.md) to create a `GriddedSource` model instance.

This `GriddedSource` model is then given the `ConfigManager` (our control panel). It knows that for 'gridded' data, 'xy' plots usually involve 'lat' and 'lon' dimensions (or their aliases configured for the source). It knows how to slice the `Temperature` variable from the dataset (obtained via the [Data Processing Pipeline](06_data_processing_pipeline_.md) and [Data Source Abstraction](07_data_source_abstraction_.md)) to get a 2D latitude-longitude slice, possibly selecting a specific time or level if needed by the configuration.

Then, it takes that 2D slice and the instructions from the `ConfigManager` (like contour levels, colormap) and hands them over to the [Plotting Components](04_plotting_components_.md) to draw the final image.

If your data was from a WRF model:

```yaml
# wrf_config.yaml
inputs:
  - name: my_wrf_output.nc
    # ... other details ...
    to_plot:
      T2: xy # Temperature at 2m often plotted as a map
      PH: yz # Geopotential Height might be plotted as a cross-section

# ... other config sections ...
```

You might run:

```bash
python autoviz.py -c ./configs -s wrf # Telling eViz it's 'wrf' source type
```

In this case, the `Autoviz` core would create a `Wrf` Source Model. The `Wrf` model knows that its latitude/longitude dimensions might be called 'XLAT'/'XLONG', that vertical levels might relate to sigma coordinates, and it has specific logic in methods like `_get_xy`, `_get_yz` to handle extracting data slices according to WRF's structure.

## Key Concepts in Source Models

1.  **`BaseSource`:** This is the top-level abstract class. It defines the minimum requirements for *any* Source Model in eViz. All Source Models must inherit from `BaseSource` and implement its abstract methods (like `plot`). Think of it as the blueprint that ensures all specialized workers have the essential tools and methods.

    ```python
    # eviz/models/base.py (Simplified)
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    import logging
    from eviz.lib.config.config_manager import ConfigManager

    @dataclass
    class BaseSource(ABC):
        """Abstract base class for autoviz data sources (Source Models)."""
        config_manager: ConfigManager # All models need config

        @property
        @abstractmethod
        def logger(self) -> logging.Logger:
            """Abstract property for the logger."""
            pass

        @abstractmethod
        def plot(self):
            """Abstract method for the top-level plotting routine."""
            pass
            
        # ... other abstract methods related to data sources (simplified in actual code) ...
    ```
    `BaseSource` ensures consistency. Any class claiming to be a Source Model must have a logger and a `plot` method.

2.  **`GenericSource`:** This class inherits from `BaseSource` and provides common functionality for many **gridded** data sources, particularly those based on NetCDF. It implements the basic plotting logic like figuring out if a SPECS file is used (single/comparison plot) or not (simple plot) and calling the appropriate methods (`_simple_plots`, `_single_plots`, `_comparison_plots`). It contains generic methods for extracting XY, YZ, XT data slices, but these often need to be *overridden* by more specific models if the data structure is unusual.

    ```python
    # eviz/models/source_base.py (Simplified)
    from dataclasses import dataclass
    import logging
    from eviz.lib.autoviz.plotter import SimplePlotter, ComparisonPlotter, SinglePlotter
    from eviz.lib.config.config_manager import ConfigManager
    from eviz.models.base import BaseSource

    @dataclass
    class GenericSource(BaseSource):
        """Base class providing common functionality for gridded sources."""
        config_manager: ConfigManager

        @property
        def logger(self) -> logging.Logger:
            # Provides a default logger implementation
            return logging.getLogger(__name__)

        def __call__(self):
             # Makes the instance callable, main entry point from Autoviz core
            self.plot()

        def plot(self):
            """Top-level plotting interface."""
            self.logger.info("Generate plots.")

            if not self.config_manager.spec_data:
                plotter = SimplePlotter()
                self._simple_plots(plotter) # Calls simple plot logic
            elif self.config_manager.compare or self.config_manager.overlay:
                plotter = ComparisonPlotter(self.config_manager.compare_exp_ids)
                self._comparison_plots(plotter) # Calls comparison plot logic
            else:
                plotter = SinglePlotter()
                self._single_plots(plotter) # Calls single plot logic

            # ... Handles saving files after plots are generated ...

        # Abstract methods from BaseSource are implemented here (simplified)
        def load_data_sources(self, file_list: list): pass
        def get_data_source(self, name: str): pass
        def add_data_source(self, name: str, data_source): pass

        # Methods for handling different plot types (often overridden)
        def _simple_plots(self, plotter): # Logic for simple plots
             # ... uses self._get_field_for_simple_plot and plotter.simple_plot ...
             pass
        def _single_plots(self, plotter): # Logic for plots with SPECS
             # ... uses self._get_field_to_plot and plotter.single_plots ...
             pass
        def _comparison_plots(self, plotter): # Logic for comparison plots
             # ... uses self._get_field_to_plot_compare and plotter.comparison_plots ...
             pass

        # Data extraction helpers (often overridden)
        def _get_xy(self, data_array, level, time_lev): pass
        def _get_yz(self, data_array, time_lev): pass
        def _get_xt(self, data_array, time_lev): pass
        # ... other helpers ...
    ```
    `GenericSource` handles the general flow. It gets the `ConfigManager`, decides *which* kind of plotting (simple, single, comparison) is needed based on the config, creates the appropriate `Plotter` from the [Plotting Components](04_plotting_components_.md), and then calls internal methods (`_single_plots`, etc.) to manage the process for individual plots.

3.  **Specific Source Models (`GriddedSource`, `Wrf`, `Lis`, `Geos`, `ObsSource`, `Airnow`, `Ghg`, `Omi`, etc.):** These are the concrete classes that handle specific data types or sources.
    *   `GriddedSource` (in `eviz/models/gridded_source.py`) is the default for generic NetCDF data. It inherits from `GenericSource` and provides basic implementations of `_get_xy`, `_get_yz`, `_get_xt` that work for standard NetCDF dimensions ('lon', 'lat', 'lev', 'time').
    *   `Wrf` (in `eviz/models/esm/wrf.py`) inherits from `GenericSource`. It overrides methods like `_get_xy`, `_get_yz`, `_get_xt`, and `_process_coordinates` to handle WRF's specific dimension names ('XLAT', 'XLONG', staggered grids), vertical coordinate systems (sigma), and regional domain boundaries.
    *   `Lis` (in `eviz/models/esm/lis.py`) also inherits from `GenericSource`. It similarly overrides methods to handle LIS's unique grid structure and coordinates, including fixing potential NaN coordinate values.
    *   `Geos` (in `eviz/models/esm/geos.py`) inherits from `GriddedSource` and adds specific functionality related to parsing GEOS HISTORY.rc files, although its core plotting logic relies heavily on the `GriddedSource` base.
    *   `ObsSource` (in `eviz/models/obs_source.py`) inherits from `GenericSource` but is tailored for unstructured observational data. It provides methods appropriate for scatter plots (`_get_field_to_plot` handling lat/lon differently) and time series.
    *   Specific observation models like `Airnow` (in `eviz/models/obs/inventory/airnow.py`) and `Ghg` (in `eviz/models/obs/inventory/ghg.py`) inherit from `ObsSource` or `GenericSource` and add logic specific to their file format (like CSV for GHG) or data structure (like point data for Airnow). `Ghg`, for example, has specific `process_data` logic to read CSVs and `_get_field_to_plot` logic to handle time series and potentially associated uncertainty data.

This inheritance structure allows sharing common logic in `GenericSource` while providing specialized handling where needed in the child classes.

## How Source Models Work Internally

Let's visualize the flow from the `Autoviz` core calling a Source Model:

```{mermaid}
sequenceDiagram
    participant AutovizCore as Autoviz Class (Engine)
    participant SourceFactory as Source Factory (Tool Finder)
    participant SpecificModel as Specific Source Model (Worker, e.g., GriddedSource)
    participant ConfigMgr as ConfigManager (Instructions)
    participant DataPipeline as Data Processing Pipeline
    participant Plotter as Plotter Subclass (Artist)
    participant Figure as Figure (Canvas)

    AutovizCore->>SourceFactory: "Give me the model for 'gridded'!"
    SourceFactory-->>AutovizCore: Returns GriddedSource class
    AutovizCore->>SpecificModel: Creates GriddedSource instance (with ConfigMgr)
    AutovizCore->>SpecificModel: Calls model() (which calls plot())
    SpecificModel->>ConfigMgr: Checks config (e.g., for SPECS, compare)
    SpecificModel->>SpecificModel: Calls _single_plots() (or _simple_plots, etc.)
    SpecificModel->>ConfigMgr: Gets details for specific plot (field, type)
    SpecificModel->>DataPipeline: "Get data for 'Temperature' from file X!"
    DataPipeline-->>SpecificModel: Returns data (e.g., xarray DataArray)
    SpecificModel->>SpecificModel: Calls _get_xy() (or _get_yz, etc.)
    SpecificModel->>SpecificModel: Applies slicing, averaging, coordinate handling...
    SpecificModel-->>SpecificModel: Returns processed 2D data & coords
    SpecificModel->>Plotter: "Use SinglePlotter!"
    SpecificModel->>Figure: Creates Figure (with ConfigMgr, plot type)
    SpecificModel->>Plotter: "Plot this data (processed data, coords, ConfigMgr, Figure)!"
    Plotter->>ConfigMgr: Gets plot styling (levels, cmap, title info)
    Plotter->>Figure: Draws plot on Figure Axes
    Figure->>Plotter: Drawing complete
    Plotter-->>SpecificModel: Plotting for this variable/level/time is done
    SpecificModel->>SpecificModel: Loops if more plots needed (times, levels)
    SpecificModel->>HelperUtilities: "Save/display the Figure!"
    HelperUtilities-->>SpecificModel: Output handled
    SpecificModel-->>AutovizCore: Processing for this source type complete
```

This diagram shows how the specific Source Model instance is the central orchestrator *for its data type*. It takes the overall task from the `AutovizCore`, uses the `ConfigManager` to understand the details, gets data via the pipeline, processes the data into the correct 2D/1D slice needed for plotting, and then uses the `Plotter` and `Figure` from the [Plotting Components](04_plotting_components_.md) to create the visual output.

Let's look at a simplified example of a specific Source Model overriding a method from `GenericSource`. Here's `Wrf` adding its own coordinate processing logic:

```python
# eviz/models/esm/wrf.py (Simplified snippet)
# ... (imports and other methods) ...

@dataclass
class Wrf(GenericSource): # Inherits from GenericSource
    """Define WRF specific model data and functions."""
    source_name: str = 'wrf' # Adds a source name attribute

    # ... overrides __post_init__ and other methods ...

    def _process_coordinates(self, data2d, dim1_name, dim2_name, field_name, plot_type, file_index, figure):
        """Process coordinates for WRF plots. Overrides GenericSource method."""
        # If not an XY or TX plot, just return data and None coords
        if 'xt' in plot_type or 'yz' in plot_type: # Added YZ here for clarity in snippet
            return data2d, None, None, field_name, plot_type, file_index, figure
        
        # For spatial plots (XY, TX)
        try:
             # WRF coordinates might be separate variables (XLAT, XLONG)
             # or part of the data_array itself depending on how it's read
             # Assume they are variables accessible via data2d.coords or similar
             # The actual WRF code snippet is more complex, looking up via config
             xs = data2d.coords['XLONG'].values # Example lookup specific to WRF
             ys = data2d.coords['XLAT'].values  # Example lookup specific to WRF

             # Calculate and set the map extent for WRF's regional grid
             latN = np.max(ys)
             latS = np.min(ys)
             lonW = np.min(xs)
             lonE = np.max(xs)
             self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
             self.config_manager.ax_opts['central_lon'] = np.mean([lonW, lonE])
             self.config_manager.ax_opts['central_lat'] = np.mean([latS, latN])

             return data2d, xs, ys, field_name, plot_type, file_index, figure

        except KeyError as e:
             self.logger.error(f"Error finding WRF coordinates for {field_name}: {e}")
             return None # Return None if coordinates aren't found as expected

    # ... overrides _get_xy, _get_yz, _get_xt etc. with WRF specific logic ...
```
This snippet shows how `Wrf` overrides the `_process_coordinates` method. Instead of assuming standard 'lon'/'lat' dimensions, it looks for WRF-specific coordinate names ('XLONG', 'XLAT' in this simplified example) and calculates the map extent based on those regional coordinates. This specialized logic is contained entirely within the `Wrf` model, keeping `GenericSource` and other models simpler.

Another example, showing `Ghg` handling its data which comes from CSV:

```python
# eviz/models/obs/inventory/ghg.py (Simplified snippet)
# ... (imports and other methods) ...

@dataclass
class Ghg(ObsSource): # Inherits from ObsSource (which inherits GenericSource)
    """Define Greenhouse Gas (GHG) inventory data and functions."""
    source_data: Any = None # Data often loaded differently (e.g., from CSV via Pandas)

    # ... overrides __post_init__ ...

    def process_data(self, filename: str, field_name: str) -> xr.Dataset:
        """
        Process CSV data for GHG. Overrides or extends base class logic.
        This method would interact with the pipeline's CSV reader.
        """
        data_source = self.config_manager.pipeline.get_data_source(filename)
        if not data_source or not hasattr(data_source, 'dataset'):
             self.logger.error(f"No data source available for {filename}")
             return None

        # Assuming the pipeline reader loaded a Pandas DataFrame from the CSV
        if isinstance(data_source.dataset, pd.DataFrame):
             df = data_source.dataset
             if field_name not in df.columns:
                 self.logger.error(f"Field {field_name} not found in {filename}")
                 return None

             # Convert the relevant part of the DataFrame to xarray Dataset for compatibility
             # This is a crucial step for many observational models!
             time_col = df.columns[0] # Assume time is first column
             ds = xr.Dataset({
                 field_name: xr.DataArray(
                     data=df[field_name].values,
                     dims=[time_col], # Time is the only dimension for a time series
                     coords={time_col: df[time_col].values},
                     attrs={'units': self._infer_units(field_name), 'long_name': field_name}
                 )
             })
             # Check for uncertainty column and add it
             for potential_unc_col in ['unc', 'uncertainty', f'{field_name}_unc']:
                 if potential_unc_col in df.columns:
                      ds[potential_unc_col] = xr.DataArray(
                          data=df[potential_unc_col].values,
                          dims=[time_col],
                          coords={time_col: df[time_col].values},
                          attrs={'units': self._infer_units(potential_unc_col), 'long_name': potential_unc_col}
                      )
                      break # Found uncertainty data, stop looking

             return ds
        else:
             # If data is already an xarray (maybe from a different reader), use it
             return data_source.dataset.get(field_name) # Return the DataArray for the field


    def _get_field_to_plot(self, data_array: xr.DataArray, field_name: str, file_index: int, plot_type: str, figure, time_level=None, full_dataset=None) -> tuple:
        """
        Prepare data for GHG plotting (mostly time series). Overrides GenericSource/ObsSource.
        """
        if data_array is None:
            self.logger.error(f"No data array provided for field {field_name}")
            return None

        # For GHG, the main plot type is 'xt' (time series)
        if 'xt' in plot_type or 'bar' in plot_type:
            # The data_array itself is the time series
            # Time values are the x-axis
            time_dim = list(data_array.coords.keys())[0]
            time_values = data_array.coords[time_dim].values

            # Check for uncertainty data in the *full_dataset* (processed by process_data)
            unc_data = None
            if full_dataset is not None: # full_dataset is the xr.Dataset from process_data
                for potential_unc_field in ['unc', 'uncertainty', f'{field_name}_unc']:
                    if potential_unc_field in full_dataset:
                        unc_data = full_dataset[potential_unc_field]
                        break

            # If uncertainty data found, store it in config_manager.ax_opts
            # The plotter will look here for error bars
            if unc_data is not None:
                 self.config_manager.ax_opts['uncertainty_data'] = unc_data

            # Return the data_array (the time series) and time_values as x-coords
            # y-coords are None for a 1D time series plot
            return data_array, time_values, None, field_name, plot_type, file_index, figure

        # Handle other plot types if necessary (e.g., maybe a map of total emissions?)
        # For now, assume only time series and bar plots are relevant for this model
        else:
            self.logger.warning(f"Plot type {plot_type} not fully supported for GHG data.")
            return None # Or return basic time series data

    # ... other methods ...
```
This `Ghg` snippet shows how it overrides `process_data` to specifically handle reading a Pandas DataFrame (assuming the pipeline did the CSV reading) and converting it into an xarray Dataset with identified time and uncertainty variables. It also overrides `_get_field_to_plot` to ensure it gets the time dimension correctly for time series plots and looks for associated uncertainty data to pass along via the `ConfigManager` for the plotter to use.

These examples highlight how Source Models encapsulate data-specific knowledge, providing a consistent interface (`plot()`, `_get_field_to_plot`) that the `GenericSource` and `Plotting Components` can rely on, while handling the internal details of their particular data format or structure.

## Connection to the Rest of eViz

Source Models are integrated into the eViz workflow as follows:

*   The [Autoviz Application Core](02_autoviz_application_core_.md) receives the requested source type (e.g., 'gridded', 'wrf') from the command line or configuration.
*   It uses the [Data Source Factory](08_data_source_factory_.md) to find and instantiate the correct Source Model class based on this type.
*   It passes the central `ConfigManager` (from the [Configuration System](01_configuration_system_.md)) to the Source Model during creation.
*   The Source Model then uses the `ConfigManager` to determine which files, variables, plot types, and styling options are needed.
*   It interacts with the [Data Processing Pipeline](06_data_processing_pipeline_.md) (which uses the [Data Source Abstraction](07_data_source_abstraction_.md)) to load and access the raw data from the specified files.
*   It processes this raw data into the appropriate structure (e.g., a 2D slice, a 1D time series) required for plotting.
*   Finally, it calls the appropriate methods on instances from the [Plotting Components](04_plotting_components_.md) (like `SinglePlotter.plot` or `ComparisonPlotter.plot`), passing the processed data, coordinates, and the `ConfigManager` for styling information.

Source Models are the specialized middle layer – they translate the generic plotting requests from `GenericSource` and the plotting capabilities of the [Plotting Components](04_plotting_components_.md) into operations that work correctly for their specific type of data.

## Conclusion

In this chapter, we learned about Source Models, eViz's specialized workers designed to handle the unique characteristics of different Earth System Model and observational data types. We saw how the `BaseSource` provides a core contract, `GenericSource` offers common gridded data logic, and specific models like `GriddedSource`, `Wrf`, `Lis`, and `Ghg` override methods to implement data-type-specific processing and coordinate handling. These models take instructions from the `ConfigManager`, get data via the pipeline, prepare it, and direct the [Plotting Components](04_plotting_components_.md) to generate the final visualizations.

Now that we understand how specialized models handle the data based on its type, let's delve into the next layer: the [Data Processing Pipeline](06_data_processing_pipeline_.md), which is responsible for the steps of loading, reading, and manipulating the data itself.

[Next Chapter: Data Processing Pipeline](06_data_processing_pipeline_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
