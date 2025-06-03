# Chapter 7: Plotter

Welcome back! In the [previous chapter](06_figure__visualization_canvas__.md), we learned about the **Figure**, the eViz component that acts as our canvas and easel, providing the drawing areas (the "axes") upon which visualizations are created. We saw how it gets ready, setting up subplots and map projections based on your configuration.

Now that we have the loaded data ([DataSource](05_datasource__base__.md)) and the blank canvas ready (`Figure`/Axes), who actually *does* the drawing? Who takes the numbers from your data and turns them into colorful contour lines, shaded areas, scattered points, or time series plots on those axes?

This is the job of the **Plotter**.

## What is Plotter and Why Do We Need It?

Think of the `Plotter` as the **artist** in our visualization process. The artist receives the data (the subject matter), knows *what kind* of picture to draw (e.g., a contour map, a graph, a scatter plot), has the instructions on *how* it should look (colors, titles, labels – from the [ConfigManager](02_configmanager_.md)), and is given the canvas areas (`Axes` from the `Figure`) to draw on.

Its main purpose is to contain the specific code and logic required to translate different types of data into visual elements using a plotting library like Matplotlib.

Why is having a dedicated `Plotter` component useful?

1.  **Separation of Concerns:** It keeps the drawing logic separate from data loading, processing, or figure setup. The [DataPipeline](03_datapipeline_.md) handles data, the `Figure` handles the canvas, and the `Plotter` handles the drawing. This makes each part simpler and easier to understand and maintain.
2.  **Specialization:** Different types of plots (contour maps, time series, scatter plots) require very different plotting code. The `Plotter` can delegate to specialized functions or classes for each type.
3.  **Configuration-Driven:** The `Plotter` uses the detailed instructions from the [ConfigManager](02_configmanager_.md) (specifically the "specs" data) to customize the look of the plot – which colors to use, what range of values to color, where to put labels, etc.

In short, the `Plotter` is where the data finally meets the canvas and becomes a visualization, guided by your configuration settings.

### A Simple Use Case: Drawing a Contour Map

Let's imagine our goal is to draw a contour map of temperature data (`Temperature` variable) on a geographical world map canvas that has already been prepared by the `Figure`.

The `Plotter`'s role in this specific task would be:

1.  Receive the temperature data (likely an `xarray.DataArray`).
2.  Receive the geographical axes object from the `Figure`.
3.  Look up the configuration settings for how the `Temperature` contour map should look (e.g., color map, contour levels, units for the colorbar label).
4.  Call the appropriate Matplotlib/Cartopy functions (like `contourf` for filled contours, `contour` for line contours, `colorbar` to add a colorbar) and pass the data, the axes, and the styling options derived from the configuration.
5.  Add titles and labels to the axes based on the configuration.

The end result is the data visually represented on the canvas.

### Key Aspects of the Plotter

The `Plotter` in eViz (specifically the `Plotter` class and its related helper classes like `SinglePlotter` in `eviz/lib/autoviz/plotter.py`) has these characteristics:

1.  **Receives Data and Axes:** It doesn't create the data or the axes; it receives them as inputs from the component orchestrating the visualization process (the Model Handler, coming in [Chapter 8](08_model_handler__abstractroot__.md)).
2.  **Depends on Configuration:** It relies heavily on the [ConfigManager](02_configmanager_.md) to get detailed instructions on *how* to draw.
3.  **Delegates Drawing:** The main `Plotter` classes often don't contain all the drawing code directly. They delegate to smaller, specialized helper functions (like `_single_xy_plot`, `_single_yz_plot`, etc., seen in the code provided) that handle the specifics of drawing a particular plot type.
4.  **Uses Matplotlib/Cartopy:** At the lowest level, the plotting helper functions use the standard plotting commands from the Matplotlib and Cartopy libraries.

### How to Use the Plotter (Conceptually)

As a user running `autoviz.py`, you typically do not directly instantiate or call methods on the `Plotter` classes yourself. The internal workflow handles this for you.

The primary "user" of the `Plotter` is usually the main processing logic for a specific data type (the "Model Handler" that you'll learn about in [Chapter 8](08_model_handler__abstractroot__.md)). This handler will:

1.  Get the processed data (as a [DataSource](05_datasource__base__.md) object containing an `xarray.Dataset`).
2.  Get the [ConfigManager](02_configmanager_.md) object with all settings.
3.  Get or create the `Figure` and its `Axes` objects.
4.  Create an instance of the appropriate `Plotter` class (e.g., `SinglePlotter`).
5.  Call a method on the `Plotter` instance (like `single_plots`), passing the relevant data, the `Figure` object, and the configuration.

So, your "use" of the `Plotter` is indirect – you define *what* to plot and *how* it should look in your configuration files ([ConfigManager](02_configmanager_.md)), and the system ensures the `Plotter` receives these instructions and executes the drawing commands on the `Figure`'s axes.

### Inside Plotter: The Artist at Work

Let's trace the steps once the main processing logic (Model Handler) is ready to draw a plot, for example, our temperature contour map (`xy` plot type).

```{mermaid}
sequenceDiagram
    participant MH as Model Handler
    participant Fig as Figure Instance
    participant Axes as Matplotlib Axes
    participant PlotterI as SinglePlotter Instance
    participant ConfigM as ConfigManager
    participant PlotFunc as _single_xy_plot()
    participant MPL as Matplotlib/Cartopy

    MH->>Fig: Get Axes (figure.get_axes())
    Fig-->>Axes: Return Axes object(s)
    MH->>PlotterI: Create SinglePlotter()
    MH->>PlotterI: single_plots(config, data, axes, field_name, ...)
    activate PlotterI
    PlotterI->>PlotterI: plot(config, data, axes, ...) # Calls internal plot method
    activate PlotterI
    PlotterI->>PlotterI: Checks plot_type ('xyplot')
    PlotterI->>PlotFunc: Call _single_xy_plot(config, data, ..., fig, axes, ...)
    deactivate PlotterI
    activate PlotFunc
    PlotFunc->>ConfigM: Get styling options (colormap, levels)
    ConfigM-->>PlotFunc: Return options
    PlotFunc->>MPL: Call contourf(data, levels, cmap, ..., ax=axes)
    MPL-->>PlotFunc: Return contour object (cfilled)
    PlotFunc->>MPL: Call colorbar(cfilled, ax=axes, fig=fig, ...)
    MPL-->>PlotFunc: Create and add colorbar
    PlotFunc->>MPL: Add titles/labels (axes.set_title, axes.set_xlabel, ...)
    MPL-->>PlotFunc: Apply titles/labels
    deactivate PlotFunc
    PlotterI-->>MH: Return (Plotting complete for this item)
    deactivate PlotterI
```

1.  The Model Handler has the loaded data, the configuration, and the prepared `Figure` (including its `Axes`).
2.  It retrieves the list of `Axes` objects from the `Figure` (using `figure.get_axes()`).
3.  It creates a `SinglePlotter` instance.
4.  It calls a method on the `SinglePlotter` (`single_plots` in this case), passing the `ConfigManager`, the data for the specific variable and time step, the `Figure` object itself, and other details like the field name and plot type.
5.  The `SinglePlotter`'s `plot` method receives this information. It looks at the `plot_type` (e.g., `'xyplot'`) and decides which specialized helper function to call (`_single_xy_plot`).
6.  The helper function (`_single_xy_plot`) receives the data (`data2d`, `x`, `y`), the `ConfigManager`, and the `Figure`/`Axes`.
7.  Inside the helper function, it accesses styling information from the `ConfigManager` (e.g., via `config.ax_opts`).
8.  It then calls the actual Matplotlib/Cartopy drawing functions, passing the data, the axes (`ax`), and the styling options. For a contour plot, this would be `ax.contourf(...)` and `fig.colorbar(...)`. It also sets titles and labels directly on the `ax` object.
9.  Once the helper function finishes, the data has been drawn onto the specified axes within the figure.

### Code Walkthrough (Simplified)

Let's look at simplified snippets from `eviz/lib/autoviz/plotter.py`.

First, the main `Plotter` class and the `SinglePlotter` worker:

```python
# eviz/lib/autoviz/plotter.py (simplified)
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
# ... other imports ...
from cartopy.mpl.geoaxes import GeoAxes # Important for map axes

# Base Plotter class - Can act as a factory or base for specific plotters
@dataclass()
class Plotter:
    @staticmethod
    def simple_plot(config, data_to_plot):
        """
        Create a basic plot without specific config overrides.
        Delegates to SimplePlotter.
        """
        no_specs_plotter = SimplePlotter()
        no_specs_plotter.plot(config, data_to_plot)
        # pu.output_basic handles saving/showing the figure (omitted detail)

# SinglePlotter - Handles plotting based on detailed SPECS configuration
@dataclass()
class SinglePlotter(Plotter): # Can inherit or be used independently

    def single_plots(self, config: ConfigManager, field_to_plot: tuple,
                     level: int = None):
        """
        Entry point to create plots using SPECS data for a single file/field.
        This method orchestrates the specific plotting calls.
        """
        self.plot(config, field_to_plot, level)

    @staticmethod
    def plot(config, field_to_plot, level):
        """
        Selects and calls the appropriate plotting helper function
        based on the plot type.
        """
        # field_to_plot is a tuple containing:
        # (data2d, dim1, dim2, field_name, plot_type, findex, fig)
        # We need the plot_type and the figure object (fig) and the data.
        data2d, dim1, dim2, field_name, plot_type, findex, fig = field_to_plot

        # Append 'plot' to match function names like _single_xyplot
        plotting_method_name = '_' + plot_type + 'plot'

        # Based on the plot type string, call the corresponding helper function.
        # This is a simplified lookup; the actual code uses if/elif chain.
        if plotting_method_name == '_xyplot':
            _single_xy_plot(config, field_to_plot, level) # Call the XY plot helper
        elif plotting_method_name == '_yzplot':
            _single_yz_plot(config, field_to_plot) # Call the YZ plot helper
        # ... add elif for other plot types like _single_xt_plot, _single_polar_plot, etc.
        else:
             logging.getLogger(__name__).error(f"Unknown plot type: {plot_type}")

        # Note: Saving/showing the figure happens *after* plotting is complete,
        # usually back in the Model Handler or a dedicated output function.
        # The plotter's job is just to DRAW on the axes provided.
```

This snippet shows that `SinglePlotter` receives the request and the crucial `field_to_plot` tuple (which contains the data and the `Figure` object `fig`). Its `plot` method then uses the `plot_type` string to decide which specific helper function (like `_single_xy_plot`) to call. It passes all the necessary information, including the `config`, `data`, and importantly, the `fig` object and its axes, to this helper.

Now, let's look at a piece of one of those helper functions, `_single_xy_plot`, and the `_plot_xy_data` helper it calls:

```python
# eviz/lib/autoviz/plotter.py (simplified)
# ... imports ...
# from eviz.lib.autoviz.figure import Figure # Needed for type hinting, but simplified
import cartopy.crs as ccrs # For map projections

def _single_xy_plot(config: ConfigManager, data_to_plot: tuple, level: int) -> None:
    """
    Helper function called by SinglePlotter to create a single XY plot.
    Sets up the axes based on config and calls the data plotting helper.
    """
    # Unpack the tuple received from SinglePlotter.plot()
    data2d, x, y, field_name, plot_type, findex, fig = data_to_plot

    ax_opts = config.ax_opts # Get plotting options from config

    # Get the correct axes from the figure.
    # This logic can be complex depending on comparison/overlay modes.
    # For a simple single plot, it often just gets the first axis.
    ax = fig.get_axes() # Get the list of axes from the Figure
    if isinstance(ax, list):
         ax = ax[0] # For a single plot, just grab the first one

    if data2d is None:
        return # Nothing to plot

    # Update axes options based on the specific field and plot type
    ax_opts = fig.update_ax_opts(field_name, ax, 'xy', level=level) # Updates ax_opts based on specs

    # Add text (like title or level info) to the plot - uses a Figure method
    fig.plot_text(field_name, ax, 'xy', level=level, data=data2d)

    # Now, call the specific helper that draws the contours and colorbar
    _plot_xy_data(config, ax, data2d, x, y, field_name, fig, ax_opts, level,
                  plot_type, findex)

# Helper called by _single_xy_plot to perform the actual drawing
def _plot_xy_data(config, ax, data2d, x, y, field_name, fig, ax_opts, level,
                  plot_type, findex):
    """
    Helper function to plot XY data (e.g., contour map) on a single axes.
    This function calls Matplotlib/Cartopy drawing functions.
    """
    # Handle fill values (replace them with NaN so Matplotlib ignores them)
    if 'fill_value' in config.spec_data.get(field_name, {}).get('xyplot', {}):
        fill_value = config.spec_data[field_name]['xyplot']['fill_value']
        data2d = data2d.where(data2d != fill_value, np.nan) # Use xarray's where method

    # Check if the axis is a Cartopy GeoAxes and if Cartopy is in use for this figure
    is_cartopy_axis = isinstance(ax, GeoAxes)
    data_transform = ccrs.PlateCarree() if is_cartopy_axis else None # Set transform if using Cartopy

    # --- This is where the actual drawing happens! ---
    # Call a helper to create filled contours using Matplotlib's contourf
    # Pass the axis (ax), data (data2d, x, y), and styling options (cmap, levels)
    # Note: _filled_contours is another internal helper that handles levels/norms
    cfilled = _filled_contours(config, field_name, ax, x, y, data2d, transform=data_transform)

    # Set up map ticks/labels if using Cartopy
    if fig.use_cartopy and is_cartopy_axis:
         if 'extent' in ax_opts:
              _set_cartopy_ticks(ax, ax_opts['extent'])
         else:
              _set_cartopy_ticks(ax, [-180, 180, -90, 90])


    # Add the colorbar using another helper function
    # Pass the contour object (cfilled), the figure (fig), axis (ax), and config/data
    if cfilled is not None: # Only add colorbar if contours were drawn
        _set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d)

    # Add line contours if requested
    if ax_opts.get('line_contours', False) and cfilled is not None:
        # Call another helper for line contours using Matplotlib's contour
        _line_contours(fig, ax, ax_opts, x, y, data2d, transform=data_transform)

    # ... Add other elements like borders, coastlines, etc. if needed (often done when setting up axes in Figure)

    # The function finishes, the data is now drawn on 'ax'
```

This extended snippet shows the heart of the `Plotter`:
1.  `_single_xy_plot` gets the data and `fig` object and retrieves the specific `ax` to draw on.
2.  It calls `_plot_xy_data`, passing the `config`, the specific `ax`, the `data2d` (the numbers to plot), the coordinate data (`x`, `y`), the `fig` object, and the `ax_opts` (styling).
3.  `_plot_xy_data` uses Matplotlib functions (often via smaller helpers like `_filled_contours` and `_set_colorbar`) to draw onto the provided `ax`. It uses the `fig` object to add the colorbar. It uses the `ax_opts` (from the `config` and `fig.update_ax_opts`) to control colors, levels, and labels.

This flow demonstrates how the `Plotter` acts as the go-between, taking the raw data and the prepared canvas and applying the styling rules from the configuration to create the final image elements.

### Conclusion

In this chapter, we learned about the **Plotter**, the eViz component responsible for the crucial task of drawing your data onto the prepared visualization canvas (`Figure`'s `Axes`). It acts as the artist, taking the data and detailed styling instructions from the [ConfigManager](02_configmanager_.md) and using libraries like Matplotlib and Cartopy to add visual elements like contours, colors, and labels to the plot axes.

While you define the plot appearance through configuration, the `Plotter` classes and their helper functions contain the specific code that executes the drawing commands, ensuring your data is accurately and beautifully represented.

Now that we've covered loading data ([DataSource](05_datasource__base__.md)), managing configuration ([ConfigManager](02_configmanager_.md)), preparing the canvas (`Figure`), and drawing the data (`Plotter`), the next chapter will introduce the **Model Handler**. This is the component that brings *all* these pieces together, orchestrating the entire process for a specific type of data source, from reading to plotting.

[Next Chapter: Model Handler (BaseSource)](08_model_handler__abstractroot__.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)