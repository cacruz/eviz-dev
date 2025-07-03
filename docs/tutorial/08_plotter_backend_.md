# Chapter 8: Plotter Backend

Welcome back to the eViz tutorial! In our last chapter, [Chapter 7: Figure](07_figure_.md), we saw how eViz sets up the canvas – the `Figure` object with its `Axes` – ready for data to be drawn. We also learned that the [Model/Source Handler](06_model_source_handler_.md) has prepared the specific data subset (like a 2D slice or a time series) needed for the plot.

Now, we have the data and the canvas. The final step is to take that prepared data and actually *draw* it onto the axes using a specific plotting library like Matplotlib, HvPlot, or Altair.

This is the job of the **Plotter Backend**.

## What Problem Does Plotter Backend Solve?

Imagine you want to create a map plot of temperature. You have the 2D temperature data and the map axes ready. But how do you actually *draw* the colored contours or pixels on the map?

*   If you use Matplotlib, you might use functions like `ax.contourf()` or `ax.pcolormesh()`.
*   If you use HvPlot, you might call a `.hvplot()` method on your data array.
*   If you use Altair, you might convert your data to a pandas DataFrame and use `alt.Chart().mark_rect().encode(...)`.

Each plotting library has its own unique way of doing things. If the [Model/Source Handler](06_model_source_handler_.md) had to contain code for *every* possible plotting library for *every* type of plot, it would become extremely complicated!

The **Plotter Backend** solves this by separating the *what* (what data to plot, what kind of plot conceptually - map, time series) from the *how* (how to draw it using Matplotlib, HvPlot, or Altair). It's a dedicated component that knows how to use the tools of a *specific* plotting library to create a *specific type* of plot.

Think of it as hiring a specialized artist. You give them the prepared materials (your data) and the canvas ([Figure](07_figure_.md) and Axes) and tell them *what* kind of picture you want (a contour map, a line graph). You also tell them *which tools* to use (Matplotlib brushes, HvPlot paints). The artist (the Plotter Backend) knows exactly which functions from that specific toolset to call to draw the picture you described.

## Our Central Use Case: Drawing an XY Map Plot

Let's revisit the example where you want to draw an XY (latitude-longitude) map of temperature. The [Model/Source Handler](06_model_source_handler_.md) has given you the 2D temperature data array and the corresponding latitude and longitude coordinates. The [Figure](07_figure_.md) has provided a Matplotlib or Cartopy Axes object ready for plotting. Your configuration specifies that you want this done using the 'matplotlib' backend.

The core question is: How does eViz draw this specific map using Matplotlib? It uses a `MatplotlibXYPlotter`.

## Using the Concept (Mostly Internal)

As a user, you primarily control which Plotter Backend is used through your configuration file, specifically the `backend` setting, often found under `system_opts`.

```yaml
# Simplified config snippet
system_opts:
  backend: "matplotlib" # Or "hvplot", "altair"
  # ... other system settings ...

outputs:
  # ... output settings ...

inputs:
  # ... input files and plot tasks ...
```

The [Model/Source Handler](06_model_source_handler_.md), after getting the [Figure](07_figure_.md) ready and preparing the data for a specific plot task (e.g., an XY map of temperature), needs to get the correct Plotter Backend instance. It uses a **Plotter Factory** to do this.

Here's a simplified look at how a [Model/Source Handler](06_model_source_handler_.md) might get and use a plotter:

```python
# Imagine this simplified code is inside a Model/Source Handler method
# like GenericSource.create_plot or a method it calls

from eviz.lib.autoviz.plotting.factory import PlotterFactory
# config_manager object is available via self.config_manager
# figure object is already created and ready (from Chapter 7)
# prepared_data_tuple holds (data2d, x, y, field_name, plot_type, findex, figure)

data2d, x, y, field_name, plot_type, findex, figure = prepared_data_tuple

# 1. Get the desired backend from the configuration
backend = self.config_manager.system_config.backend # Access via Config Manager

# 2. Use the Plotter Factory to create the correct plotter instance
print(f"Requesting plotter for type='{plot_type}' and backend='{backend}'")
plotter = PlotterFactory.create_plotter(plot_type, backend)
print(f"Created plotter object: {type(plotter)}")

# 3. Call the plot method on the plotter instance
# Pass the necessary information: config, the prepared data, and the figure/axes
# The specific plot method (e.g., plot_2d) is often called internally by a generic method
# like `create_plot` in GenericSource, which selects the right backend method.
# Let's show calling a generic plot method for simplicity here:
self.logger.debug(f"Calling plotter.plot() with data for {field_name}...")

# The actual data passed can vary depending on the plot type and backend
# For Matplotlib, it's often (config, ax, data2d, x, y, field_name, fig, ax_opts, ...)
# For HvPlot/Altair, it might be (config, data_to_plot_tuple)
# Simplified call:
plotter.plot(self.config_manager, prepared_data_tuple) # The plotter knows what to do

self.logger.debug("Plotting complete.")

# The result (the populated figure for Matplotlib, or the hv/altair object)
# is now available via plotter.get_plot_object() or implicitly in the fig object
# and can be saved/displayed.
```

This shows that the [Model/Source Handler](06_model_source_handler_.md) acts as the client, asking the `PlotterFactory` for the right tool (the specific plotter instance) and then telling that tool to perform its primary job (`plot`).

## Breaking Down the Concept: Classes and Roles

The Plotter Backend concept in eViz involves several classes working together:

1.  **Base Plotter (`BasePlotter`):**
    *   Found in `eviz/lib/autoviz/plotting/base.py`.
    *   This is the fundamental abstract base class for *all* plotters, regardless of type or backend.
    *   It defines the core contract: any plotter *must* have a `plot()` method (or equivalent), a `save()` method, and a `show()` method.
    *   This ensures a consistent interface for any plotter object.
    ```python
    # Simplified snippet from eviz/lib/autoviz/plotting/base.py
    from abc import ABC, abstractmethod
    import logging

    class BasePlotter(ABC):
        """Base class for all plotters."""
        def __init__(self):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.plot_object = None # Can hold the plot object (e.g., Matplotlib figure, HvPlot object)

        @abstractmethod
        def plot(self, config, data_to_plot): # Signature can vary slightly in subclasses
            """Create a plot from the given data."""
            pass

        @abstractmethod
        def save(self, filename, **kwargs):
            """Save the plot to a file."""
            pass

        @abstractmethod
        def show(self):
            """Display the plot."""
            pass

        def get_plot_object(self):
            """Return the underlying plot object."""
            return self.plot_object
    ```
    This defines the basic behavior expected from any plotter.

2.  **Plot Type Base Classes (`XYPlotter`, `YZPlotter`, `XTPlotter`, `ScatterPlotter`, etc.):**
    *   Also found in `eviz/lib/autoviz/plotting/base.py`.
    *   These are intermediate abstract base classes, inheriting from `BasePlotter`.
    *   They represent the *type* or *shape* of the visualization (a 2D map, a vertical profile, a time series, a scatter plot).
    *   While they inherit the abstract methods, concrete backend implementations will inherit from these *type-specific* bases. This is mainly for organization and clarity, indicating what kind of plot the inheriting class is designed to create.
    ```python
    # Simplified snippet from eviz/lib/autoviz/plotting/base.py
    # (following BasePlotter definition)

    class XYPlotter(BasePlotter):
        """Base class for XY (lat-lon) plotters."""
        @abstractmethod
        def plot(self, config, data_to_plot): # Still abstract, but contextually for XY plots
            """Create an XY plot from the given data."""
            pass

    class XTPlotter(BasePlotter):
        """Base class for XT (time-series) plotters."""
        @abstractmethod
        def plot(self, config, data_to_plot): # Still abstract, but contextually for XT plots
            """Create an XT plot from the given data."""
            pass

    # ... similar classes for YZPlotter, ScatterPlotter, etc. ...
    ```
    These classes categorize plotters by their visual output type.

3.  **Backend Implementation Classes (`MatplotlibXYPlotter`, `HvplotXTPlotter`, `AltairScatterPlotter`, etc.):**
    *   Found in subdirectories like `eviz/lib/autoviz/plotting/backends/matplotlib/`, `hvplot/`, and `altair/`.
    *   These are the concrete classes that do the *actual drawing*.
    *   They inherit from a *plot type base class* (e.g., `MatplotlibXYPlotter` inherits from `XYPlotter`) and implement the `plot()`, `save()`, and `show()` methods using the functions and syntax of their specific library (Matplotlib, HvPlot, or Altair).
    *   They receive the prepared data and the [Figure](07_figure_.md)/Axes object (often embedded within `data_to_plot` or passed separately, depending on the backend).
    ```python
    # Simplified snippet from eviz/lib/autoviz/plotting/backends/matplotlib/xy_plot.py
    import matplotlib as mpl
    import numpy as np
    # from cartopy.mpl.geoaxes import GeoAxes # Needed for map axes

    from .base import MatplotlibBasePlotter # A Matplotlib-specific base class often used

    class MatplotlibXYPlotter(MatplotlibBasePlotter): # Inherits from MatplotlibBasePlotter (which likely inherits XYPlotter or BasePlotter)
        def __init__(self):
            super().__init__()
            self.fig = None
            self.ax = None # Store the Axes object

        def plot(self, config, data_to_plot):
            """Create an XY plot using Matplotlib."""
            data2d, x, y, field_name, plot_type, findex, fig = data_to_plot
            self.fig = fig # Store the Figure
            self.ax = fig.get_axes()[config.axindex] # Get the correct Axes for this plot

            self.logger.debug(f"Plotting {field_name} on Matplotlib Axes")

            # --- This is where the actual Matplotlib drawing happens ---
            # Use the prepared data (x, y, data2d) and the axes (self.ax)
            # Use config or fig.ax_opts for styling (colormap, levels, etc.)

            # Example: Create filled contours
            # This is the core Matplotlib drawing call
            cfilled = self.ax.contourf(x, y, data2d,
                                      levels=config.ax_opts.get('clevs'), # Get levels from config/figure options
                                      cmap=config.ax_opts.get('use_cmap')) # Get colormap

            # Add contour lines if needed
            # clines = self.ax.contour(x, y, data2d, levels=...)

            # Add colorbar using the Figure's helper method
            self.fig.colorbar_eviz(cfilled, self.ax, config.ax_opts)

            # Set title, labels etc. using the Figure's helper methods
            # self.fig.plot_text(field_name, self.ax, ...)

            # ---------------------------------------------------------

            self.plot_object = self.fig # Store the Figure object
            self.logger.debug("Matplotlib plotting complete.")
            return self.fig # Return the figure object

        def save(self, filename, **kwargs):
             if self.plot_object:
                  self.plot_object.savefig(filename, **kwargs) # Use Matplotlib savefig

        def show(self):
             if self.plot_object:
                  self.plot_object.show() # Use Matplotlib show

    ```
    ```python
    # Simplified snippet from eviz/lib/autoviz/plotting/backends/hvplot/xt_plot.py
    import numpy as np
    import pandas as pd
    import holoviews as hv
    import hvplot.xarray # Register hvplot accessor
    import hvplot.pandas  # Register hvplot accessor

    from eviz.lib.autoviz.plotting.base import XTPlotter

    class HvplotXTPlotter(XTPlotter):
        def __init__(self):
            super().__init__()
            self.plot_object = None
            # Initialize HvPlot/HoloViews extension (usually bokeh)
            try:
                hv.extension('bokeh')
            except Exception as e:
                self.logger.warning(f"Could not initialize HvPlot extension: {e}")

        def plot(self, config, data_to_plot):
            """Create an XT plot using HvPlot."""
            data2d, _, _, field_name, plot_type, findex, _ = data_to_plot # Get data array

            self.logger.debug(f"Plotting {field_name} using HvPlot")

            # --- This is where the actual HvPlot drawing happens ---
            # HvPlot often works directly on xarray DataArrays or pandas DataFrames

            if hasattr(data2d, 'hvplot'): # Check if it's an xarray DataArray
                 # Use the .hvplot accessor provided by `hvplot.xarray`
                 # Need to get the time dimension name
                 time_dim = config.get_model_dim_name('tc') or data2d.dims[0]
                 plot = data2d.hvplot.line(
                     x=time_dim, # Specify the x-axis dimension
                     title=field_name,
                     ylabel=config.get_units(field_name, data2d), # Get units
                     width=800, height=400, # Set size
                     tools=['hover', 'pan', 'wheel_zoom'] # Add interactive tools
                 )
            else:
                 # Convert to pandas DataFrame if not xarray
                 # Need time coordinates, let's assume they are available somehow
                 # (Simplified conversion here)
                 time_coords = self._get_time_coords(data2d) # Internal helper
                 df = pd.DataFrame({'time': time_coords, 'value': data2d.values})
                 plot = df.hvplot.line(
                      x='time', y='value',
                      title=field_name,
                      ylabel=config.get_units(field_name, data2d),
                      width=800, height=400,
                      tools=['hover', 'pan', 'wheel_zoom']
                 )

            # -----------------------------------------------------

            self.plot_object = plot # Store the HoloViews/HvPlot object
            self.logger.debug("HvPlotting complete.")
            return plot # Return the plot object

        def save(self, filename, **kwargs):
             if self.plot_object:
                  # HvPlot/HoloViews saving varies, might need `hvplot.save` or `bokeh.io.export_png`
                  # Example (might be simplified/placeholder depending on actual eViz code):
                  try:
                      hv.save(self.plot_object, filename, **kwargs)
                  except Exception as e:
                       self.logger.error(f"Error saving hvplot {filename}: {e}")

        def show(self):
             if self.plot_object:
                  # HvPlot/HoloViews show depends on the extension (bokeh, etc.)
                  try:
                      hv.show(self.plot_object)
                  except Exception as e:
                       self.logger.error(f"Error showing hvplot: {e}")

    ```
    ```python
    # Simplified snippet from eviz/lib/autoviz/plotting/backends/altair/scatter_plot.py
    import numpy as np
    import pandas as pd
    import altair as alt

    from eviz.lib.autoviz.plotting.base import ScatterPlotter

    class AltairScatterPlotter(ScatterPlotter):
        def __init__(self):
            super().__init__()
            self.plot_object = None
            # Initialize Altair renderer
            try:
                 alt.renderers.enable('default')
                 alt.data_transformers.disable_max_rows() # Useful for large data
            except Exception as e:
                 self.logger.warning(f"Could not initialize Altair renderer: {e}")

        def plot(self, config, data_to_plot):
            """Create a Scatter plot using Altair."""
            # Scatter plot data is typically (x, y, z_color_data, ...)
            x_data, y_data, z_data, field_name, plot_type, findex, _ = data_to_plot

            self.logger.debug(f"Plotting {field_name} as Scatter using Altair")

            # --- This is where the actual Altair drawing happens ---
            # Altair works best with pandas DataFrames
            df = self._convert_to_dataframe(x_data, y_data, z_data) # Internal helper

            if df.empty:
                 self.logger.warning("DataFrame is empty, cannot create Altair plot")
                 return None # Return None or a placeholder chart

            # Basic Altair chart setup
            chart = alt.Chart(df).mark_circle().encode( # Use mark_circle for scatter
                 x=alt.X('x:Q', title=config.ax_opts.get('xlabel', 'X')), # Get labels from config
                 y=alt.Y('y:Q', title=config.ax_opts.get('ylabel', 'Y')),
                 tooltip=['x', 'y', 'z'] # Add basic tooltips
            ).properties(
                 title=field_name,
                 width=800, height=500
            ).interactive() # Make it interactive (zoom/pan)

            # Add color encoding if z_data exists
            if z_data is not None and 'z' in df.columns:
                 chart = chart.encode(
                      color=alt.Color('z:Q',
                                     scale=alt.Scale(scheme=config.ax_opts.get('use_cmap', 'viridis')), # Get colormap
                                     title=config.get_units(field_name, z_data)) # Add colorbar label/units
                 )

            # -----------------------------------------------------

            self.plot_object = chart # Store the Altair chart object
            self.logger.debug("Altair plotting complete.")
            return chart # Return the chart object

        def save(self, filename, **kwargs):
             if self.plot_object:
                  # Altair saving uses the chart's save method
                  try:
                      self.plot_object.save(filename, **kwargs)
                  except Exception as e:
                       self.logger.error(f"Error saving Altair plot {filename}: {e}")

        def show(self):
             if self.plot_object:
                  # Altair show uses the chart's show method
                  try:
                       self.plot_object.show()
                  except Exception as e:
                       self.logger.error(f"Error showing Altair plot: {e}")

    ```
    These snippets show that the core drawing logic is specific to each backend implementation class, calling functions from `matplotlib.pyplot`, using `.hvplot` accessors, or building `alt.Chart` objects. They use the `config` or `ax_opts` (passed via `data_to_plot` or `figure`) to get styling information.

4.  **Plotter Factory (`PlotterFactory`):**
    *   Found in `eviz/lib/autoviz/plotting/factory.py`.
    *   This class has a static method `create_plotter` that takes the `plot_type` (e.g., 'xy', 'xt', 'sc') and the desired `backend` ('matplotlib', 'hvplot', 'altair').
    *   It looks up the correct concrete plotter class in a predefined dictionary and returns an instance of it. This hides the details of class names and imports from the code that *uses* the plotter.
    ```python
    # Simplified snippet from eviz/lib/autoviz/plotting/factory.py
    # Import all the specific plotter implementation classes...
    from .backends.matplotlib.xy_plot import MatplotlibXYPlotter
    from .backends.matplotlib.xt_plot import MatplotlibXTPlotter
    # ... import other Matplotlib plotters ...
    from .backends.hvplot.xy_plot import HvplotXYPlotter
    from .backends.hvplot.xt_plot import HvplotXTPlotter
    # ... import other HvPlot plotters ...
    from .backends.altair.xy_plot import AltairXYPlotter
    from .backends.altair.xt_plot import AltairXTPlotter
    # ... import other Altair plotters ...

    class PlotterFactory:
        """Factory for creating appropriate plotters."""

        @staticmethod
        def create_plotter(plot_type, backend="matplotlib"):
            """Create a plotter for the given plot type and backend."""
            # Dictionary mapping (plot_type, backend) to plotter class
            plotters = {
                ("xy", "matplotlib"): MatplotlibXYPlotter,
                ("xt", "matplotlib"): MatplotlibXTPlotter,
                # ... other matplotlib mappings ...

                ("xy", "hvplot"): HvplotXYPlotter,
                ("xt", "hvplot"): HvplotXTPlotter,
                # ... other hvplot mappings ...

                ("xy", "altair"): AltairXYPlotter,
                ("xt", "altair"): AltairXTPlotter,
                # ... other altair mappings ...
            }

            key = (plot_type, backend)
            if key in plotters:
                return plotters[key]() # Create and return an instance
            else:
                raise ValueError(f"No plotter available for plot_type={plot_type}, backend={backend}")

    ```
    This factory is the crucial piece that allows eViz to dynamically select the correct plotting logic at runtime based on the configuration.

## Under the Hood: The Drawing Request

Here's a sequence diagram showing how a [Model/Source Handler](06_model_source_handler_.md) uses the Plotter Backend to draw a specific plot:

```{mermaid}
sequenceDiagram
    participant Handler as Model/Source Handler
    participant ConfigMgr as ConfigManager
    participant Figure as Figure Object
    participant PlotterFactory as PlotterFactory (Class)
    participant MatplotlibXYPlotter as MatplotlibXYPlotter (Instance)
    participant Matplotlib as matplotlib

    Handler->>ConfigMgr: Get requested backend ('matplotlib')
    ConfigMgr-->>Handler: Return 'matplotlib'
    Handler->>PlotterFactory: create_plotter('xy', 'matplotlib')
    PlotterFactory->>PlotterFactory: Lookup ('xy', 'matplotlib') -> MatplotlibXYPlotter
    PlotterFactory->>MatplotlibXYPlotter: Create instance (new MatplotlibXYPlotter())
    MatplotlibXYPlotter-->>PlotterFactory: Return instance
    PlotterFactory-->>Handler: Return MatplotlibXYPlotter instance (plotter)

    Handler->>Handler: Prepare data_to_plot tuple
    Handler->>MatplotlibXYPlotter: plot(ConfigMgr, data_to_plot) # Pass prepared data, config, figure/axes
    MatplotlibXYPlotter->>ConfigMgr: Get ax_opts, units etc.
    MatplotlibXYPlotter->>Figure: Get Axes object for this plot
    Figure-->>MatplotlibXYPlotter: Return Axes (ax)
    MatplotlibXYPlotter->>Matplotlib: Call ax.contourf(x, y, data2d, ...)
    Matplotlib-->>MatplotlibXYPlotter: Drawing complete on Axes
    MatplotlibXYPlotter->>MatplotlibXYPlotter: Store figure in self.plot_object
    MatplotlibXYPlotter-->>Handler: Return figure object

    Handler->>Handler: Call plotting utils to save/display figure
```

This diagram illustrates how the Handler gets the right plotter via the Factory, then calls the plotter's `plot` method, which in turn uses the specific library's functions (`matplotlib` calls here) to draw the data onto the axes provided by the [Figure](07_figure_.md).

For different backends, the calls within the specific plotter's `plot` method would change (e.g., calling `data.hvplot` or building `alt.Chart`), but the overall flow from the Handler's perspective (Get plotter -> Call plot) remains consistent, demonstrating the power of this abstraction.

## Summary

In this chapter, we explored the **Plotter Backend**:

*   It's the component responsible for drawing the prepared data onto the figure using a specific plotting library.
*   It solves the problem of making eViz compatible with multiple plotting libraries (Matplotlib, HvPlot, Altair) without changing the main visualization logic.
*   It uses a hierarchy of classes: `BasePlotter` (general contract), plot type base classes (`XYPlotter`, `XTPlotter`, etc. - interface for plot shapes), and concrete backend implementation classes (e.g., `MatplotlibXYPlotter`, `HvplotXTPlotter`, `AltairScatterPlotter`) that contain the library-specific drawing code.
*   The `PlotterFactory` is used to select and create the correct backend implementation instance at runtime based on the configuration and the requested plot type.
*   [Model/Source Handlers](06_model_source_handler_.md) use the factory to get a plotter and then call its `plot` method, passing the data, configuration, and the [Figure](07_figure_.md)/Axes.

The Plotter Backend completes the core visualization workflow. With data sourced and processed, configuration loaded, the main application orchestrating, data interpreted by a specific handler, the canvas prepared by the Figure, the Plotter Backend finally brings the data to life visually.

This concludes our deep dive into the core components of the eViz visualization process. We've covered everything from reading the initial data file all the way to drawing the final plot on the screen or saving it to a file.

Now that you know how eViz uses source-specific experts to process and plot data, you might be wondering how you can inspect your data files to understand their structure (like dimension names, variable names, attributes) so you know what to put in your configuration or how to implement a new source handler. That's where the **Metadata Tool (metadump)** comes in handy.

[Metadata Tool (metadump)](09_metadata_tool__metadump__.md)


---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
