# Chapter 4: Plotting Components

Welcome back to the eViz tutorial! In [Chapter 1: Configuration System](01_configuration_system_.md), we set up the control panel with instructions. In [Chapter 2: Autoviz Application Core](02_autoviz_application_core_.md), we saw how the main engine reads those instructions and orchestrates the whole process. And in [Chapter 3: Metadata Generator (metadump)](03_metadata_generator__metadump__.md), we learned how to quickly get a starting configuration for a new data file.

Now, we arrive at the exciting part: **Plotting Components**. All the steps so far have been about getting ready – defining what to plot, where the data is, and setting up the environment. But how does eViz actually turn the processed data into the images you see? That's the job of the Plotting Components.

Think of this set of components as the **Art Studio** of eViz. It provides the canvas, the artists who know how to draw different types of pictures (plots), and some helpers to format and save the final artwork.

## Your Fourth Task: Creating the Plot Image

Let's stick with our ongoing task: Plotting the `Temperature` variable from our `my_weather_data.nc` file, guided by our configuration. Once the [Data Processing Pipeline](06_data_processing_pipeline_.md) (which we'll cover later) has loaded and potentially manipulated the `Temperature` data, it needs to be visualized.

This is where the Plotting Components take over. They take the prepared data (usually as a NumPy array or an xarray DataArray) and instructions (from the [Configuration System](01_configuration_system_.md) via `ConfigManager`) and generate the plot image.

## Key Concepts in Plotting Components

The Plotting Components primarily revolve around two main ideas:

1.  **The Canvas (`Figure`):** Just like a painter needs a canvas, plots need a surface to be drawn upon. In eViz, this is represented by the `Figure` class, which is an enhanced version of Matplotlib's `Figure`. It's the virtual paper where everything is placed.
2.  **The Artists (`Plotter` subclasses):** Different plots require different drawing techniques. A map is drawn differently than a time series, or a vertical profile. `Plotter` subclasses are like specialized artists. Each `Plotter` subclass knows how to take data and draw a *specific type* of plot (like a contour map, a line graph, or a scatter plot).

There are also **Helper Utilities** (in `eviz/lib/autoviz/utils.py`) that assist with tasks like saving the figure to a file, adding titles, or formatting specific elements like colorbars.

## How Plotting Components Create the Visualization

The flow typically happens *after* the data has been loaded and processed by a [Source Model](05_source_models_.md). The model, having the prepared data and access to the `ConfigManager`, decides which type of plot is needed and delegates the drawing job to the appropriate `Plotter`.

Here's a simplified look at the process:

```{mermaid}
sequenceDiagram
    participant SourceModel as Source Model (Worker)
    participant ConfigMgr as ConfigManager (Instructions)
    participant Plotter as Plotter Subclass (Artist)
    participant Figure as Figure (Canvas)
    participant Matplotlib as Matplotlib Backend
    participant ImageFile as Plot Image File

    SourceModel->>ConfigMgr: "What plot type for Temperature?"
    ConfigMgr-->>SourceModel: "xyplot, xtplot (from specs)"
    SourceModel->>Plotter: "Use SinglePlotter for xyplot!"
    SourceModel->>Figure: "Factory, create a Figure (for xyplot, with config)!"
    Figure-->>SourceModel: Returns eViz Figure instance
    SourceModel->>Plotter: "Plot this data (data, config, figure)!"
    Plotter->>Figure: "Draw plot elements on your Axes!"
    Figure->>Matplotlib: Makes low-level drawing calls (contourf, plot)
    Matplotlib-->>Figure: Drawing happens
    Figure->>Plotter: Drawing complete
    Plotter->>Figure: "Add title, colorbar, etc."
    Figure->>Matplotlib: Makes more drawing calls
    Matplotlib-->>Figure: Adding elements happens
    Plotter-->>SourceModel: Plotting done
    SourceModel->>HelperUtilities: "Save this Figure to file!"
    HelperUtilities->>Figure: "Get canvas data & save!"
    Figure->>ImageFile: Writes image file (PNG, PDF, etc.)
    ImageFile-->>HelperUtilities: File saved
    HelperUtilities-->>SourceModel: Task complete
```

As you can see, the `SourceModel` acts as the manager, preparing data and telling the `Plotter` what to draw and where (`Figure`). The `Plotter` uses the `ConfigManager` for details on *how* to draw it, making calls to the `Figure` object, which in turn uses the underlying Matplotlib library.

## Diving Deeper into the Code

Let's look at the core components involved in plotting.

### The Canvas: `Figure` (`eviz/lib/autoviz/figure.py`)

The `Figure` class is the central object representing the plot area.

```python
# --- Simplified eviz/lib/autoviz/figure.py ---
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
# ... other imports like cartopy ...

@dataclass
class Figure(mfigure.Figure):
    """
    Enhanced Figure class inheriting from matplotlib's Figure with eViz framework customizations.
    """
    config_manager: Any  # Needs ConfigManager
    plot_type: str       # e.g., 'xy', 'xt'

    # Attributes initialized after Matplotlib Figure is created
    _gridspec = None
    _subplots = (1, 1) # Default layout
    _use_cartopy = False
    axes_array = [] # List of axes on the figure
    _ax_opts: dict = field(default_factory=dict) # Stores plotting options

    def __post_init__(self):
        # Matplotlib Figure setup happens in super().__init__
        # eViz customizations start here:
        
        # Determine subplot layout (e.g., 1x1, 1x2 for comparison)
        self._set_compare_diff_subplots() 
        self._init_frame() # Set figure size based on layout

        # Important: Pass Matplotlib-specific args to the parent class
        super().__init__(**self.get_matplotlib_kwargs())
        
        self._init_frame() # (Called again, check if needed once or twice)

    def get_matplotlib_kwargs(self):
        # Helper to extract standard Figure kwargs
        kwargs = {} # ... logic to pull figsize etc from config or defaults ...
        return kwargs

    def set_axes(self) -> "Figure":
        """Create axes objects on the figure based on the layout and plot type."""
        if 'tx' in self.plot_type or 'sc' in self.plot_type or 'xy' in self.plot_type:
            self._use_cartopy = True # Decide if Cartopy projection is needed

        self.create_subplot_grid() # Set up the grid (e.g., 1 row, 2 columns)
        self.create_subplots()     # Add the actual Axes objects

        return self # Return self for chaining

    def create_subplots(self):
        """Add subplots (Axes) to the figure."""
        if self.use_cartopy:
            return self._create_subplots_crs() # Create Cartopy GeoAxes
        else:
            # Create standard Matplotlib Axes
            for i in range(self._subplots[0]):
                for j in range(self._subplots[1]):
                    ax = self.add_subplot(self.gs[i, j])
                    self.axes_array.append(ax)
            return self

    @staticmethod
    def create_eviz_figure(config_manager, plot_type, field_name=None, nrows=None, ncols=None) -> "Figure":
        """Factory method to create an eViz Figure instance."""
        # Logic to determine nrows, ncols based on config (comparison, overlay)
        # ... (simplified) ...
        fig = Figure(config_manager, plot_type, nrows=nrows, ncols=ncols)
        return fig # Return the created Figure

    # ... methods for adding colorbars, text, saving, showing, etc. ...
```

**Explanation:**

*   The `Figure` class inherits from `matplotlib.figure.Figure`, meaning it *is* a Matplotlib Figure with all its standard abilities.
*   `__post_init__` runs setup logic after the object is created. It takes the `config_manager` and `plot_type` to understand the context. It determines the layout (like how many subplots are needed for a comparison plot) and initializes attributes like `_ax_opts` which will store plotting styles for the current axes.
*   The `set_axes` method is crucial. It calls `create_subplot_grid` to define the grid structure and then `create_subplots` to add the actual drawing areas (`Axes`) to the figure, deciding whether to use standard Matplotlib axes or geographical axes from Cartopy based on the `plot_type`.
*   `create_eviz_figure` is a recommended way to get a `Figure` object. It includes logic to handle different layout requirements specified in the configuration (like comparison plots).

### The Artists: `Plotter` Subclasses (`eviz/lib/autoviz/plotter.py`)

The `Plotter` classes are responsible for taking the data and drawing on the `Figure`'s axes.

```python
# --- Simplified eviz/lib/autoviz/plotter.py ---
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# ... other imports like cartopy ...

from eviz.lib.config.config_manager import ConfigManager
# ... import eViz Figure class ...

@dataclass()
class Plotter:
    """Base class for plotters."""
    # This base class doesn't do much drawing itself
    pass

@dataclass()
class SimplePlotter(Plotter):
    """Plotter for creating basic plots without detailed specifications."""

    @staticmethod
    def plot(config: ConfigManager, field_to_plot: tuple):
        """ Create a basic plot (like ncview) based on plot_type. """
        # field_to_plot is a tuple containing data, dims, field name, etc.
        data2d, dim1, dim2, field_name, plot_type, findex, fig = field_to_plot

        if plot_type == 'xy':
            _simple_xy_plot(config, field_to_plot) # Call helper function to draw
        elif plot_type == 'yz':
            _simple_yz_plot(config, field_to_plot) # Call helper function to draw
        # ... other simple plot types ...
        else:
             print(f"ERROR: Simple plot type {plot_type} not implemented")

        # Basic output handling
        # pu.output_basic(config, field_to_plot[3]) # Call helper utility

@dataclass()
class SinglePlotter(Plotter):
    """Plotter for creating plots using detailed specifications from config."""

    @staticmethod
    def plot(config: ConfigManager, field_to_plot: tuple, level: int = None):
        """ Create a single plot using specs data. """
        # field_to_plot contains data, dims, field name, figure instance, etc.
        data2d, x, y, field_name, plot_type_key, findex, fig = field_to_plot
        
        # The key from config is 'xyplot', 'xtplot', etc.
        plot_type = plot_type_key.replace('plot', '') # Get 'xy', 'xt', etc.

        # Initialize axes options from config/specs for this field
        fig.init_ax_opts(field_name) 
        
        # Get the correct axes instance from the figure (important for subplots)
        ax = fig.get_axes() # Note: this returns a list

        # Call specific helper functions based on the plot type
        if plot_type == 'yz':
            _single_yz_plot(config, field_to_plot) 
        elif plot_type == 'xt':
            _single_xt_plot(config, field_to_plot)
        # ... other single plot types ...
        elif plot_type == 'xy':
            # _single_xy_plot handles getting the correct axes from the list
            _single_xy_plot(config, field_to_plot, level)
        # ... more plot types ...
        else:
             print(f"ERROR: Single plot type {plot_type_key} not implemented")

# Helper functions doing the actual drawing (simplified examples)
def _single_xy_plot(config, data_to_plot, level):
    """Helper to draw a single XY map using Matplotlib/Cartopy."""
    data2d, x, y, field_name, plot_type_key, findex, fig = data_to_plot

    ax_temp = fig.get_axes() # Get list of axes from the figure
    # Logic here to select the correct 'ax' from ax_temp based on comparison layout, etc.
    ax = ax_temp[config.axindex] # Assuming axindex is set by the SourceModel

    # Get/Update plotting options for this specific plot (e.g., contour levels)
    ax_opts = fig.update_ax_opts(field_name, ax, 'xy', level=level)

    if data2d is None: return # Handle empty data

    # Use Matplotlib functions to draw on the selected axes (ax)
    if fig.use_cartopy:
        # Draw filled contours with Cartopy transformation
        cfilled = _filled_contours(config, field_name, ax, x, y, data2d, transform=fig.projection)
        _set_cartopy_ticks(ax, ax_opts.get('extent')) # Add geographical labels/grid
    else:
        # Draw filled contours with standard Matplotlib
        cfilled = _filled_contours(config, field_name, ax, x, y, data2d)

    # Add contour lines if requested
    if ax_opts.get('line_contours', False):
         _line_contours(fig, ax, ax_opts, x, y, data2d, transform=fig.projection if fig.use_cartopy else None)

    # Add colorbar and title using figure methods or helpers
    if cfilled:
        _set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d)
    fig.plot_text(field_name, ax, 'xy', level=level, data=data2d) # Add title/stats

def _filled_contours(config, field_name, ax, x, y, data2d, transform=None):
    """Helper to create filled contours."""
    # Determine contour levels and colormap from config/specs
    _create_clevs(field_name, config.ax_opts, data2d)
    clevs = config.ax_opts['clevs']
    cmap = config.ax_opts.get('use_diff_cmap' if config.compare else 'use_cmap')

    # Use Matplotlib's contourf function
    cfilled = ax.contourf(x, y, data2d, levels=clevs, cmap=cmap, extend=config.ax_opts['extend_value'], transform=transform)
    return cfilled

# ... other _single_..._plot helper functions and plotting utilities ...
```

**Explanation:**

*   `Plotter` is a simple base class.
*   `SimplePlotter` handles requests for basic plots. Its `plot` method receives the `config` and the data/metadata tuple (`field_to_plot`). It then calls specific `_simple_..._plot` helper functions to do the actual drawing. These simple plots typically don't use the detailed settings from the `specs.yaml`.
*   `SinglePlotter` handles requests for more customized plots that *do* use the `specs.yaml` configuration. Its `plot` method also receives the `config` and data. It initializes or updates axes options (`ax_opts`) using `fig.init_ax_opts` or `fig.update_ax_opts` (which pull settings from `config.spec_data`). It then calls the corresponding `_single_..._plot` helper function (like `_single_xy_plot`).
*   The actual Matplotlib/Cartopy drawing calls (`ax.contourf`, `ax.plot`, etc.) happen inside the private helper functions like `_single_xy_plot` or `_filled_contours`. These helpers use the `ax_opts` dictionary (derived from the `ConfigManager` and `specs.yaml`) to control the appearance (colormap, contour levels, line styles, etc.). They interact directly with the `ax` (Axes) object from the `Figure` to draw elements.

### Helper Utilities (`eviz/lib/autoviz/utils.py`)

This module contains various functions used by the `Plotter` classes and the `SourceModel` for tasks related to figures and output.

```python
# --- Simplified eviz/lib/autoviz/utils.py ---
import matplotlib.pyplot as plt
import os
import json
from PIL import Image # Used for GIF creation

import numpy as np
import matplotlib.ticker as mticker # For formatters

# ... other imports ...

def print_map(config, plot_type, findex, fig, level=None) -> None:
    """Save or display a plot."""
    # Determine output directory and filename from config
    output_dir = config.output_config.output_dir # Get from ConfigManager
    print_to_file = config.output_config.print_to_file
    print_format = config.output_config.print_format

    # Build the filename based on field name, file index, level, etc.
    # ... (logic to build filename string like Temperature_1_0.png) ...
    filename = os.path.join(output_dir, final_filename)

    if print_to_file:
        fig.tight_layout()
        fig.savefig(filename, dpi=300) # Use Figure's savefig
        # Optional: Add logo
        if config.output_config.add_logo:
             add_logo_ax(fig) # Add logo after layout is set, before saving
             fig.savefig(filename, dpi=300) # Save again with logo

        # Optional: Dump metadata JSON
        # dump_json_file(...)

        logger.debug(f"Figure saved to {filename}")
    else:
        plt.tight_layout()
        plt.show() # Display using Matplotlib's show

def add_logo_ax(fig, desired_width_ratio=0.10) -> None:
    """Adds image logo to figure using axes coordinates."""
    # ... logic to load logo image ...
    logo_ax = fig.add_axes([left, bottom, width, height], zorder=10) # Add axes specifically for logo
    logo_ax.imshow(logo) # Draw logo on its own axes
    logo_ax.axis('off') # Hide axis border/ticks

# ... functions for creating GIFs, PDFs, formatting ticks, colorbar formatters, etc. ...

class OOMFormatter(mticker.ScalarFormatter):
    # Custom formatter for scientific notation on colorbars
    pass

class FlexibleOOMFormatter(mticker.ScalarFormatter):
    # Another custom formatter for colorbars
    pass
```

**Explanation:**

*   `print_map` is a utility function that takes the `config` and the `fig` (Figure) object and handles the final output step: either saving the figure to a file (using settings from `config.output_config`) or displaying it. It also orchestrates adding logos or generating metadata files if configured.
*   `add_logo_ax` is an example of a helper function that modifies the figure (by adding a logo image) before it's saved.
*   The module also contains various formatting utilities, like classes for controlling how numbers appear on colorbars (`OOMFormatter`, `FlexibleOOMFormatter`) or functions to determine appropriate font sizes based on the number of subplots (`legend_font_size`, `axis_tick_font_size`, etc.).

## Connection to the Rest of eViz

The Plotting Components are the final step in the visualization pipeline orchestrated by the [Autoviz Application Core](02_autoviz_application_core_.md).

*   The `ConfigManager` (from the [Configuration System](01_configuration_system_.md)) is the single source of truth, providing all the detailed instructions (`ax_opts`, `use_cartopy`, output path, file format) that the `Figure` and `Plotter` subclasses need to draw and save the plot correctly.
*   The [Source Models](05_source_models_.md) (coming next!) are the clients of the Plotting Components. They prepare the data using the [Data Processing Pipeline](06_data_processing_pipeline_.md), then create the necessary `Figure` object and call the appropriate `Plotter`'s `plot` method, passing the data and configuration.
*   The underlying Matplotlib library (and optionally Cartopy for maps) does the low-level drawing, but the eViz `Figure` and `Plotter` classes manage *what* is drawn and *how* it's styled, making it consistent with the configuration system.

In summary, Plotting Components take the data and instructions and perform the actual drawing and saving, turning abstract data into visual representations.

## Conclusion

In this chapter, we explored the Plotting Components, which are responsible for generating the final visualization images. We learned about the `Figure` class, eViz's customized canvas, and the `Plotter` subclasses (`SimplePlotter`, `SinglePlotter`) which act as specialized artists, knowing how to draw different plot types. We also saw how helper utilities assist with tasks like saving the output and adding extra elements like logos.

These components work closely with the `ConfigManager` for styling and output instructions, and are called by the [Source Models](05_source_models_.md) after the data processing is complete.

Now that we understand how the plots are drawn, let's move on to learn more about the [Source Models](05_source_models_.md) themselves – the workers that prepare the data and manage the plotting process for specific data types.

[Next Chapter: Source Models](05_source_models_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
