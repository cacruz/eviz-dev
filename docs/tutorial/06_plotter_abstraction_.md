# Chapter 6: Plotter Abstraction

Welcome back! So far in our journey through eViz:
*   [Chapter 1: Autoviz Application](01_autoviz_application_.md) introduced the main program that runs everything.
*   [Chapter 2: Configuration Management](02_configuration_management_.md) showed us how eViz reads your detailed plan from YAML files.
*   [Chapter 3: Data Source Abstraction](03_data_source_abstraction_.md) explained how eViz can load data from different file formats into a standard `xarray.Dataset`.
*   [Chapter 4: Data Source Factory](04_data_source_factory_.md) revealed how eViz automatically picks the right tool (DataSource) to read your specific data file.
*   [Chapter 5: Data Processing Pipeline](05_data_processing_pipeline_.md) showed us how your data is cleaned, standardized, and prepared after loading.

Now that your data is loaded, processed, and ready, the next step is to visualize it! But scientific data can be visualized in many ways: as maps, time series, scatter plots, box plots, and more. Also, you might want to use different plotting libraries (like Matplotlib for static plots or HoloViews/hvPlot for interactive ones).

How does eViz handle all these different types of plots and different plotting libraries without getting tangled up in complex code? The answer is **Plotter Abstraction**.

## What is Plotter Abstraction?

Imagine you have a set of different **drawing tools**. You might have:
*   A paintbrush specifically designed for painting geographical maps.
*   A ruler and pencil set for drawing precise line graphs (like time series).
*   A special statistical box-drawing tool for box plots.

Each tool is designed for a specific *type* of drawing.

Now, imagine you have different **brands** of these tools. You might have a Matplotlib-brand map paintbrush and a HoloViews-brand map paintbrush. Both are for maps, but they work slightly differently because they come from different companies.

**Plotter Abstraction** in eViz is like having a **universal instruction manual** that works for *any* drawing tool, regardless of its *type* (map, line graph, etc.) or its *brand* (Matplotlib, HoloViews, etc.).

The universal instruction manual says things like:
*   "To draw something, follow these basic steps (using your specific tool's method)." (`plot` method)
*   "To save your drawing, follow these basic steps." (`save` method)
*   "To show your drawing, follow these basic steps." (`show` method)

All drawing tools in eViz agree to follow this universal manual. This means the main eViz program doesn't need to know the specific details of a Matplotlib map tool or a HoloViews time series tool. It just needs to know: "Okay, I have a tool here; I'll tell it to `plot()`, then maybe `save()`, then maybe `show()`."

It separates **what** you want to visualize (e.g., temperature on an XY plane, pressure over time) from **how** it actually gets drawn by a specific plotting library.

## Our Use Case: Plotting Temperature as a Map or Time Series

Let's revisit our configuration from [Chapter 2: Configuration Management](02_configuration_management_.md). Your config file tells eViz *what* to plot:

```yaml
# --- Snippet from a config file ---
inputs:
  - name: sample_data.nc
    # ... other settings ...
    to_plot:
      temperature: xy  # Plot 'temperature' as an XY map
      pressure: xt     # Plot 'pressure' as an XT time series
```

This tells eViz that for the variable `temperature`, you want an `xy` plot type, and for `pressure`, you want an `xt` plot type.

Later in the configuration, you might also specify which **plotting backend** (library) to use, for example, Matplotlib or HvPlot:

```yaml
# --- Snippet from a config file ---
outputs:
  plotting_backend: matplotlib # Or 'hvplot'
  # ... other settings ...
```

The Plotter Abstraction system makes it possible for eViz to handle these instructions:
1.  Find the processed data for `temperature` and `pressure` (from the [Data Processing Pipeline](05_data_processing_pipeline_.md)).
2.  Identify that `temperature` needs an `xy` plot and `pressure` needs an `xt` plot.
3.  Know that you want to use the `matplotlib` backend.
4.  Get the correct **Plotter object** capable of drawing an **XY plot** using **Matplotlib**.
5.  Get the correct **Plotter object** capable of drawing an **XT plot** using **Matplotlib**.
6.  Use the standard `plot()`, `save()`, and `show()` instructions on these objects, regardless of whether they are XY or XT plotters, or Matplotlib or HvPlot plotters (if you had chosen 'hvplot').

## How Plotter Abstraction Works (High-Level)

Let's see a simplified flow focusing on how a Plotter object is obtained and used once the data is ready and the config specifies a plot type and backend.

```{mermaid}
sequenceDiagram
    participant DataReady as Processed Data (xarray.DataArray)
    participant ConfigMgr as ConfigManager
    participant PlotterFactory as Plotter Factory
    participant BasePlotter as BasePlotter (Interface)
    participant XYPlotterBase as XYPlotter (Abstract Type)
    participant MatplotlibXYPlotter as MatplotlibXYPlotter (Concrete Impl)
    participant MatplotlibLib as Matplotlib Library

    ConfigMgr->>ConfigMgr: Determine plot needs (variable, type, backend)
    ConfigMgr->>PlotterFactory: "Need Plotter for var 'temp', type 'xy', backend 'matplotlib'"
    PlotterFactory->>PlotterFactory: Look up which class handles XY plots for Matplotlib
    PlotterFactory->>MatplotlibXYPlotter: Create new MatplotlibXYPlotter()
    MatplotlibXYPlotter-->>PlotterFactory: Return MatplotlibXYPlotter object
    PlotterFactory-->>ConfigMgr: Return MatplotlibXYPlotter object (implements BasePlotter)

    ConfigMgr->>MatplotlibXYPlotter: Call plot(config, DataReady) (using BasePlotter interface)
    MatplotlibXYPlotter->>MatplotlibLib: Call specific Matplotlib functions (e.g., contourf, pcolormesh)
    MatplotlibLib-->>MatplotlibXYPlotter: Matplotlib creates plot internally
    MatplotlibXYPlotter-->>ConfigMgr: Plotting step complete (Plotter object holds the plot)

    ConfigMgr->>MatplotlibXYPlotter: Call save(filename) (using BasePlotter interface)
    MatplotlibXYPlotter->>MatplotlibLib: Call specific Matplotlib function (e.g., fig.savefig)
    MatplotlibLib-->>MatplotlibXYPlotter: Plot saved to file
    MatplotlibXYPlotter-->>ConfigMgr: Saving step complete

    ConfigMgr->>MatplotlibXYPlotter: Call show() (using BasePlotter interface)
    MatplotlibXYPlotter->>MatplotlibLib: Call specific Matplotlib function (e.g., plt.show)
    MatplotlibLib-->>MatplotlibXYPlotter: Plot displayed on screen
    MatplotlibXYPlotter-->>ConfigMgr: Showing step complete
```

This diagram illustrates:
1.  The `ConfigManager` knows *what* plot is needed from the configuration.
2.  It asks the **Plotter Factory** (we'll cover this in [Chapter 7](07_plotter_factory_.md)!) to *create* the right specific Plotter object based on the plot *type* (`xy`) and chosen *backend* (`matplotlib`).
3.  The Factory finds and creates the `MatplotlibXYPlotter`.
4.  Crucially, this `MatplotlibXYPlotter` object understands and implements the methods defined in the `BasePlotter` interface (`plot`, `save`, `show`).
5.  The `ConfigManager` (or the part of the code orchestrating plotting) simply calls these standard methods on the Plotter object it received. It doesn't need to know it's a Matplotlib XY plotter internally.
6.  The `MatplotlibXYPlotter`'s implementation of `plot`, `save`, and `show` then uses the actual Matplotlib library functions to perform the drawing and saving.

This abstraction means eViz can easily swap plotting backends or add new plot types just by creating new classes that follow the `BasePlotter` rules.

## Inside the Code: The BasePlotter and Plot Types

The core of the abstraction starts with the `BasePlotter` abstract class in `eviz/lib/autoviz/plotting/base.py`.

```python
# --- File: eviz/lib/autoviz/plotting/base.py (simplified) ---
from abc import ABC, abstractmethod
import logging

class BasePlotter(ABC):
    """Base class for all plotters."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.plot_object = None # This will hold the library's plot object (e.g., Matplotlib figure)
    
    @abstractmethod
    def plot(self, config, data_to_plot):
        """Create a plot from the given data. Must be implemented by subclasses."""
        pass # Abstract methods have no implementation here
        
    @abstractmethod
    def save(self, filename, **kwargs):
        """Save the plot to a file. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def show(self):
        """Display the plot. Must be implemented by subclasses."""
        pass
    
    def get_plot_object(self):
        """Return the underlying plot object (useful for advanced use)."""
        return self.plot_object
```

*   `BasePlotter(ABC)`: This tells Python that `BasePlotter` is an abstract base class. You cannot create an instance of `BasePlotter` directly.
*   `@abstractmethod`: These decorators mark the `plot`, `save`, and `show` methods as abstract. Any class that inherits from `BasePlotter` *must* provide its own concrete implementation for these methods.
*   `plot_object`: This attribute is where the actual plotting library's object (like a Matplotlib `Figure` or a HoloViews `Layout`) will be stored after the `plot` method runs.

Below the `BasePlotter`, `base.py` also defines abstract classes for different *types* of plots. These inherit from `BasePlotter` and might refine the `plot` method signature slightly or just serve as markers for organization.

```python
# --- File: eviz/lib/autoviz/plotting/base.py (simplified plot types) ---
# ... BasePlotter class ...

class XYPlotter(BasePlotter):
    """Base class for XY (lat-lon) plotters."""    
    @abstractmethod # Still abstract!
    def plot(self, config, data_to_plot):
        """Create an XY plot from the given data."""
        pass

class XTPlotter(BasePlotter):
    """Base class for XT (time-series) plotters."""    
    @abstractmethod # Still abstract!
    def plot(self, config, data_to_plot):
        """Create an XT plot from the given data."""
        pass

# ... YZPlotter, ScatterPlotter, BoxPlotter, etc. follow a similar pattern ...
```

These classes like `XYPlotter` and `XTPlotter` still don't have the actual drawing code. They define the *concept* of an XY or XT plotter within the eViz system. The real work is done by classes that inherit from these *and* are specific to a plotting backend.

## Inside the Code: Backend-Specific Implementations

The concrete, runnable Plotter classes live within backend-specific directories, like `eviz/lib/autoviz/plotting/backends/matplotlib/` and `eviz/lib/autoviz/plotting/backends/hvplot/`.

These classes inherit from the plot-type abstract classes (`XYPlotter`, `XTPlotter`, etc.) and provide the *actual* code using the corresponding plotting library.

### Example: Matplotlib XY Plotter

This class draws XY maps using Matplotlib. It's in `eviz/lib/autoviz/plotting/backends/matplotlib/xy_plot.py`.

```python
# --- File: eviz/lib/autoviz/plotting/backends/matplotlib/xy_plot.py (simplified) ---
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # Used for maps
# Import the base XYPlotter type
from eviz.lib.autoviz.plotting.base import XYPlotter
# Note: It also inherits from MatplotlibBasePlotter for common Matplotlib helpers

class MatplotlibXYPlotter(XYPlotter): # Inherits from XYPlotter
    """Matplotlib implementation of XY plotting."""
    def __init__(self):
        super().__init__() # Call BasePlotter.__init__
        self.fig = None # Specific Matplotlib figure attribute
        self.ax = None  # Specific Matplotlib axes attribute
            
    def plot(self, config, data_to_plot):
        """Create an XY plot using Matplotlib."""
        # data_to_plot is a tuple like (data2d, x, y, field_name, plot_type, findex, fig)
        data2d, x, y, field_name, plot_type, findex, fig = data_to_plot
        
        self.fig = fig # Store the Matplotlib figure
        # Logic to get the correct axes from the figure based on config...
        self.ax = fig.get_axes()[config.axindex] # Simplified
        
        # --- Actual Matplotlib plotting calls ---
        # This calls a helper method specific to Matplotlib XY plotting
        self._plot_xy_data(config, self.ax, data2d, x, y, field_name, fig, config.ax_opts, 0, plot_type, findex)
        # ... (Details of _plot_xy_data include calls to ax.contourf, ax.pcolormesh, etc.) ...
        
        # Logic for colorbars, titles, etc.
        # ...
        
        self.plot_object = fig # Store the figure in the base attribute
        return fig

    def save(self, filename, **kwargs):
        """Save the Matplotlib plot to a file."""
        if self.fig is not None:
            try:
                # Use Matplotlib's savefig method
                self.fig.savefig(filename, **kwargs)
                self.logger.info(f"Saved Matplotlib XY plot to {filename}")
            except Exception as e:
                self.logger.error(f"Error saving Matplotlib plot: {e}")
        else:
            self.logger.warning("No Matplotlib figure to save")

    def show(self):
        """Display the Matplotlib plot."""
        if self.fig is not None:
            try:
                # Use Matplotlib's show method
                plt.show()
            except Exception as e:
                self.logger.error(f"Error showing Matplotlib figure: {e}")
        else:
            self.logger.warning("No Matplotlib figure to show")
```

This class implements the `plot`, `save`, and `show` methods required by `BasePlotter` (via `XYPlotter`). Its `plot` method contains the specific Matplotlib code needed to create an XY map (using `ax.contourf`, `ax.pcolormesh`, etc. within its helper methods). Its `save` and `show` methods use the standard Matplotlib functions `fig.savefig` and `plt.show`.

### Example: HvPlot XT Plotter

This class draws XT time series plots using the HvPlot/HoloViews library. It's in `eviz/lib/autoviz/plotting/backends/hvplot/xt_plot.py`.

```python
# --- File: eviz/lib/autoviz/plotting/backends/hvplot/xt_plot.py (simplified) ---
import holoviews as hv
import hvplot.xarray # Important: registers hvplot accessor
from eviz.lib.autoviz.plotting.base import XTPlotter # Import the base XTPlotter type

class HvplotXTPlotter(XTPlotter): # Inherits from XTPlotter
    """HvPlot implementation of XT (time-series) plotting."""    
    def __init__(self):
        super().__init__() # Call BasePlotter.__init__
        # Set up the hvplot environment (specific to this backend)
        try:
            hv.extension('bokeh')
        except Exception as e:
            self.logger.warning(f"Could not initialize HoloViews extension: {e}")
    
    def plot(self, config, data_to_plot):
        """Create an interactive XT plot using HvPlot."""
        # data_to_plot is a tuple like (data2d, x, y, field_name, plot_type, findex, fig)
        data2d, _, _, field_name, plot_type, findex, _ = data_to_plot
        
        if data2d is None:
             self.logger.warning("No data to plot")
             return None
        
        # --- Actual HvPlot plotting calls ---
        # Assuming data2d is an xarray DataArray with a time dimension
        time_dim = config.get_model_dim_name('tc') # Get the standard time dimension name
        
        try:
            # Use the hvplot accessor on the xarray DataArray
            plot = data2d.hvplot.line( # HvPlot method for line plots
                x=time_dim,          # Specify the x-axis dimension
                title=field_name,    # Use field name for title
                xlabel='Time',
                ylabel=data2d.attrs.get('units', 'n.a.'), # Use units attribute
                width=800, height=400,
                tools=['hover'] # Add interactive tools
                # ... other plot options ...
            )
            
            self.logger.debug("Successfully created hvplot")
            self.plot_object = plot # Store the HvPlot object in the base attribute
            
            return plot
            
        except Exception as e:
            self.logger.error(f"Error creating hvplot: {e}")
            return None

    def save(self, filename, **kwargs):
        """Save the HvPlot plot to an HTML file."""
        if self.plot_object is not None:
            try:
                # Ensure .html extension for interactive plots
                if not filename.endswith('.html'):
                    filename += '.html'
                # Use HoloViews save method
                hv.save(self.plot_object, filename)
                self.logger.info(f"Saved interactive plot to {filename}")
            except Exception as e:
                self.logger.error(f"Error saving plot: {e}")
        else:
            self.logger.warning("No plot to save")
    
    def show(self):
        """Display the HvPlot plot (in notebook or browser)."""
        if self.plot_object is not None:
            try:
                # Try to display in a Jupyter notebook
                from IPython.display import display
                display(self.plot_object)
            except ImportError:
                # If not in a notebook, save temporarily and open in browser
                import tempfile, webbrowser, os
                temp_file = os.path.join(tempfile.gettempdir(), 'eviz_hvplot_xt.html')
                hv.save(self.plot_object, temp_file)
                webbrowser.open(f"file://{temp_file}")
                self.logger.info(f"Opening plot in browser: {temp_file}")
            except Exception as e:
                self.logger.error(f"Error showing plot: {e}")
        else:
            self.logger.warning("No plot to show")
```

This class implements the `plot`, `save`, and `show` methods for XT plots using HvPlot. Its `plot` method uses the `hvplot` accessor on the `xarray.DataArray` to quickly generate an interactive line plot. Its `save` and `show` methods use HoloViews/HvPlot specific ways to save to HTML and display the plot.

Notice how different the internal code is between `MatplotlibXYPlotter.plot` and `HvplotXTPlotter.plot` (using Matplotlib functions vs. the `hvplot` accessor). But from the perspective of the code *using* these objects, they both just have a `.plot()`, `.save()`, and `.show()` method. This is the power of abstraction!

## Summary

In this chapter, we explored **Plotter Abstraction** in eViz. We learned that:

*   Plotter Abstraction provides a standard interface (`BasePlotter` with `plot`, `save`, `show` methods) for all visualization tools in eViz.
*   Abstract classes like `XYPlotter` and `XTPlotter` define different *types* of plots, inheriting from `BasePlotter`.
*   Concrete classes like `MatplotlibXYPlotter` and `HvplotXTPlotter` implement these plot types using specific plotting libraries (backends).
*   This system allows the core eViz logic to request a Plotter object based on plot type and backend, and then interact with it using the standard interface, without needing to know the backend-specific implementation details.
*   It makes it easy to switch between plotting libraries or add support for new plot types.

Now that eViz has the processed data and understands the concept of different plotters following a standard interface, the final piece is: how does eViz *automatically select* and create the *correct* backend-specific Plotter class instance (`MatplotlibXYPlotter`, `HvplotXTPlotter`, etc.) based on your configuration? That task is handled by the **Plotter Factory**.

[Plotter Factory](07_plotter_factory_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)