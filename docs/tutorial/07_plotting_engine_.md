# Chapter 7: Plotting Engine

Welcome back! In the [previous chapter](06_data_processing_pipeline_.md), we followed our data through the **Data Processing Pipeline**. We saw how it's loaded from various formats ([Chapter 3: Data Source Abstraction](03_data_source_abstraction_.md), [Chapter 4: Data Source Factory](04_data_source_factory_.md)), how eViz understands its properties using **Metadata Handling** ([Chapter 5: Metadata Handling](05_metadata_handling_.md)), and how the pipeline cleans, standardizes, and integrates it into neat **xarray Datasets**.

Now, we have the data, processed and prepared, ready to be seen! But numbers in an `xarray Dataset` aren't a picture. How does eViz turn these numbers and their associated coordinates and metadata into the maps, time series, and vertical profiles you see?

This is the job of the **Plotting Engine**.

## The Plotting Engine: Bringing Data to Life

Imagine you are an artist. The **Data Processing Pipeline** has just given you your materials: a beautiful canvas, high-quality paints mixed to the perfect colors, and brushes cleaned and ready. Your task as the artist is to take these materials and create a visual masterpiece – a painting!

In eViz, the **Plotting Engine** is that artist. It's the component responsible for the final step: taking the processed data and the detailed instructions from the **Configuration Files** ([Chapter 2: Configuration Management](02_configuration_management_.md)) and rendering them into visual plots.

It uses powerful Python libraries like **Matplotlib** (a standard plotting library) and **Cartopy** (specialized for drawing maps) to handle the actual drawing.

## What the Plotting Engine Does (The Artist's Tasks)

The Plotting Engine performs several key tasks to create a visualization:

1.  **Sets up the Canvas:** It creates the figure (the entire window or image file) and the axes (the individual plot areas within the figure). Think of this as preparing your canvas and deciding where each part of the painting will go.
2.  **Selects the Right Technique:** Based on the dimensions of the data and the requested plot type from the configuration (e.g., 'xy' for map, 'xt' for time series, 'yz' for profile), it chooses the appropriate plotting method.
3.  **Applies the Paint:** It uses Matplotlib or Cartopy functions to draw the data onto the axes. This might involve drawing filled contours for a map (`contourf`), drawing lines for a time series (`plot`), or scattering points (`scatter`).
4.  **Adds the Details:** It adds crucial visual elements like:
    *   Titles and axis labels (often using metadata like variable names and units).
    *   Colorbars to show the data scale.
    *   Gridlines, coastlines, and political boundaries (especially for maps using Cartopy).
    *   Specific contour lines with labels.
5.  **Applies the Style:** It reads styling instructions from the configuration (like color maps, contour levels, font sizes, line styles) and makes sure the plot looks the way you want.
6.  **Finishes the Piece:** It manages the final output, either saving the plot to an image file (like PNG or PDF) or displaying it on your screen.

## Core Components: Figure and Plotter

In eViz, the main components handling the plotting are:

*   **`Figure`:** An enhanced version of Matplotlib's `Figure` class (`eviz/lib/autoviz/figure.py`). It represents the entire canvas and manages things like the overall size, the arrangement of subplots (multiple plots in one figure), and map projections (if using Cartopy). It's like the canvas and the layout plan combined.
*   **`Plotter`:** A class (`eviz/lib/autoviz/plotter.py`) that acts as the conductor *for the plotting process itself*. It receives the processed data and configuration and directs the creation of specific plot types by calling dedicated functions. It's like the artist's brain, deciding *how* to paint each part based on the plan.

Helper functions for tasks like formatting numbers, adding colorbars, and saving files are found in `eviz/lib/autoviz/utils.py`.

## How the Plotting Engine Fits In (Simplified Flow)

Let's see the final steps involving the Plotting Engine:

```{mermaid}
sequenceDiagram
    participant A as Autoviz Object/Model
    participant CM as ConfigManager
    participant DP as DataProcessing Pipeline
    participant XD as xarray Dataset (Processed)
    participant PE as Plotting Engine (Plotter)
    participant FIG as Figure Object
    participant MPL_CR as Matplotlib/Cartopy
    participant Output as Image File/Display

    A->>DP: Process Data
    DP-->>A: Processed xarray Dataset(s) + Config
    A->>PE: Pass Dataset(s) & Config
    PE->>FIG: Create Figure (using Config for layout/size)
    PE->>FIG: Set up subplots (using Config/data dims)
    PE->>PE: Determine plot type needed (e.g., 'xy')
    PE->>MPL_CR: Call plotting functions (_single_xy_plot etc.)
    MPL_CR->>FIG: Draw data onto Figure axes
    PE->>MPL_CR: Add plot elements (titles, labels, colorbars, gridlines) using Dataset metadata & Config styling
    MPL_CR-->>PE: Plot is drawn
    PE->>Output: Save Figure to file / Display Figure
```

This diagram shows that the Plotting Engine receives the processed data (as `xarray Dataset(s)`) and the `ConfigManager` from the main `Autoviz` or Model object. It uses the `Figure` class to set up the plot canvas, then uses the config and data to figure out what specific plot to create. It calls the underlying Matplotlib/Cartopy functions to draw everything onto the `Figure`, adds all the details and styling, and finally handles the output.

## Peeking at the Code

Let's look at some very simplified snippets to see the core pieces.

First, the `Figure` class (`eviz/lib/autoviz/figure.py`). It inherits from `matplotlib.figure.Figure`:

```python
# --- File: eviz/lib/autoviz/figure.py (Simplified) ---
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# ... other imports ...

class Figure(mfigure.Figure): # Inherits from Matplotlib's Figure
    """
    Enhanced Figure class inheriting from matplotlib's Figure with eViz customizations.
    """
    def __init__(self, config_manager, plot_type, *, nrows=None, ncols=None, **kwargs):
        # Store config and plot type
        self.config_manager = config_manager
        self.plot_type = plot_type
        self._subplots = (nrows if nrows is not None else 1, ncols if ncols is not None else 1)
        self._use_cartopy = False # Flag for map plots
        self.axes_array = [] # To store subplot axes

        # Call the parent (Matplotlib Figure) constructor
        super().__init__(**kwargs)

        # Initialize eViz specific things like frame layout based on config
        self._init_frame()

    def set_axes(self) -> "Figure":
        """
        Set figure axes objects based on required subplots and plot type.
        """
        if 'tx' in self.plot_type or 'sc' in self.plot_type or 'xy' in self.plot_type or 'po' in self.plot_type:
             # Map/scatter/polar plots need Cartopy projection
            self._use_cartopy = True

        self.create_subplot_grid() # Set up the grid (e.g., 1x1, 1x2)
        self.create_subplots()    # Create the actual axes objects

        return self

    def create_subplots(self):
        """
        Create subplots based on the gridspec (subplot grid) and projection.
        """
        if self.use_cartopy:
            # Create subplots with Cartopy projection
            return self._create_subplots_crs()
        else:
            # Create standard Matplotlib subplots
            for i in range(self._subplots[0]):
                for j in range(self._subplots[1]):
                    ax = self.add_subplot(self.gs[i, j]) # Add subplot to grid
                    self.axes_array.append(ax)
            return self

    def _create_subplots_crs(self) -> "Figure":
        """Create subplots with cartopy projections."""
        # Determine the projection (e.g., PlateCarree) based on config
        map_projection = self.get_projection()

        for i in range(self._subplots[0]):
            for j in range(self._subplots[1]):
                 # Add subplot specifying the projection
                ax = self.add_subplot(self.gs[i, j], projection=map_projection)
                self.axes_array.append(ax)

            # Add common map features like coastlines
        for ax in self.axes_array:
             ax.coastlines()
             # ... add other features like borders, land ...

        return self

    def get_projection(self, projection=None) -> Optional[ccrs.Projection]:
         """Get a Cartopy projection instance based on config."""
         # ... logic to read projection name and extent from config ...
         # Default to PlateCarree if none specified
         if projection is None:
             return ccrs.PlateCarree()
         # ... logic to return specific Cartopy projection instances ...
         return ccrs.PlateCarree() # Fallback
```

This simplified `Figure` class shows its basic structure: it holds the configuration, knows the plot type, determines the subplot layout (`_subplots`), and decides whether to use Cartopy based on the plot type. Its `set_axes` method orchestrates the creation of the subplot grid and the actual axes objects (`create_subplots`), handling the special case of Cartopy projections.

Next, the `Plotter` class (`eviz/lib/autoviz/plotter.py`). There are different `Plotter` classes (like `SimplePlotter` for basic plots, `SinglePlotter` for spec-driven plots, `ComparisonPlotter`), but they share a common pattern: they have a `plot` method that directs the drawing. Let's look at a simplified `SinglePlotter`:

```python
# --- File: eviz/lib/autoviz/plotter.py (Simplified) ---
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from eviz.lib.autoviz.figure import Figure # Import our custom Figure class
# ... other imports and helper plotting functions ...

@dataclass()
class SinglePlotter():
    """
    Handles plotting for a single field using specifications.
    """
    def single_plots(self, config, field_to_plot, level=None):
        # This method is often called from the Model (next chapter)
        self.plot(config, field_to_plot, level)

    @staticmethod # It's a static method because it doesn't need 'self' state
    def plot(config, field_to_plot, level):
        """ Create a single plot using specs data """
        # field_to_plot tuple contains: (data2d, dim1_coords, dim2_coords, field_name, plot_type_short, file_index, figure_object)
        data2d, x_coords, y_coords, field_name, plot_type_short, findex, fig = field_to_plot

        # Determine the full plot type string (e.g., 'xyplot')
        plot_type_full = plot_type_short + 'plot'

        # Create the Figure object if not already provided (often is by the model)
        if fig is None:
             # The Figure factory method handles layout based on config/plot type
             fig = Figure.create_eviz_figure(config, plot_type_full, field_name)

        # Initialize axes options based on config/specs for this field
        fig.init_ax_opts(field_name)

        # Get the axes objects from the figure
        axes = fig.set_axes().get_axes()
        ax = axes[config.axindex] # Select the correct axis if multiple subplots

        # --- Direct the plotting based on plot type ---
        if plot_type_full == 'xyplot':
            # Call a dedicated helper function for XY maps
            _single_xy_plot(config, ax, data2d, x_coords, y_coords, field_name, fig, fig.ax_opts, level, plot_type_short, findex)
        elif plot_type_full == 'xtplot':
            # Call a dedicated helper function for Time Series
            _single_xt_plot(config, ax, data2d, field_name, fig, fig.ax_opts) # Simplified args
        # ... elif for other plot types like 'yzplot', 'polarplot', 'scplot' ...
        else:
            logger.error(f'Plot type {plot_type_full} is not implemented')
            return # Stop if type is unknown

        # --- Handle output ---
        # Use a helper function from utils to save or show the figure
        pu.print_map(config, plot_type_full, findex, fig, level=level)
        plt.close(fig) # Close the figure after saving/showing to free memory
```

This simplified `SinglePlotter.plot` method shows how it receives the processed data (`data2d`, `x_coords`, `y_coords`) and configuration (`config`). It gets or creates a `Figure` object, initializes axes options based on the detailed specs (`fig.init_ax_opts`), sets up the figure's axes (`fig.set_axes()`), and then calls a specific helper function (`_single_xy_plot`, `_single_xt_plot`, etc.) to do the actual drawing based on the plot type. Finally, it uses `pu.print_map` (from `eviz/lib/autoviz/utils.py`) to save or show the resulting figure and closes the figure.

Let's look at a *highly* simplified version of one of the specific plotting helper functions, like `_single_xy_plot` (also in `eviz/lib/autoviz/plotter.py`):

```python
# --- File: eviz/lib/autoviz/plotter.py (Simplified, inside _single_xy_plot function) ---
# This function gets called by SinglePlotter.plot if plot_type is 'xyplot'

def _single_xy_plot(config, ax, data2d, x, y, field_name, fig, ax_opts, level, plot_type_short, findex):
    """Helper function to plot XY (map) data on a single axes."""

    # --- Apply the paint (Draw the data) ---
    # Determine if we need Cartopy transform (depends on Figure's use_cartopy flag)
    data_transform = ccrs.PlateCarree() if fig.use_cartopy else None

    # Use Matplotlib/Cartopy contourf for filled contours
    cfilled = ax.contourf(x, y, data2d,
                          levels=ax_opts['clevs'], # Get contour levels from processed config/specs
                          cmap=ax_opts['use_cmap'], # Get colormap from config/specs
                          extend=ax_opts['extend_value'],
                          norm=colors.BoundaryNorm(ax_opts['clevs'], ncolors=256, clip=False), # Set color normalization
                          transform=data_transform) # Apply Cartopy transform if needed

    # --- Add the details (Plot elements and styling) ---

    # Add contour lines if enabled in config/specs
    if ax_opts.get('line_contours', False):
        ax.contour(x, y, data2d, levels=ax_opts['clevs'], colors="black", linewidths=0.5, transform=data_transform)
        # Also add contour labels using ax.clabel(...)

    # Add colorbar
    _set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d) # Uses a helper function

    # Add plot title and labels (uses metadata/config)
    fig.plot_text(field_name, ax, plot_type_short, level=level, data=data2d) # Uses a Figure method

    # Set axis ticks and labels (Cartopy has special formatters)
    if fig.use_cartopy:
        _set_cartopy_ticks(ax, ax_opts['extent']) # Uses a helper function
    else:
        # Set standard Matplotlib ticks/labels
        ax.set_xlabel(...)
        ax.set_ylabel(...)

    # Add gridlines if enabled
    if ax_opts['add_grid']:
         fig.add_grid(ax) # Uses a Figure method
```

This highly simplified function shows how the specific plotting logic combines the `xarray Dataset` (passed as `data2d`, `x`, `y`) with the styling and setting information obtained from the `config` and `ax_opts` (which came from the configuration/specs). It makes direct calls to Matplotlib (`contourf`, `contour`) and Cartopy (via the `transform` argument and Cartopy-specific tick helpers) to draw the plot elements and relies on helper functions (`_set_colorbar`, `fig.plot_text`) to add common features and styling.

## Summary of Plotting Engine Process

When the **Autoviz Application** (or more specifically, the relevant **Model** class) determines that a particular variable from a processed dataset needs to be plotted:

1.  It calls the **Plotting Engine** (specifically, one of the `Plotter` classes).
2.  The `Plotter` receives the processed `xarray DataArray` (or `Dataset`) and the `ConfigManager`.
3.  The `Plotter` gets or creates a `Figure` object, determining the overall canvas size and subplot layout based on the configuration (e.g., comparison plots need multiple subplots).
4.  The `Figure` sets up the subplot axes, using Cartopy for map-like plots (`xy`, `tx`, `sc`, `po`) to handle projections and add base map features like coastlines.
5.  The `Plotter` identifies the specific plot type (e.g., `xyplot`, `xtplot`) from the configuration.
6.  The `Plotter` calls a dedicated internal function (like `_single_xy_plot`, `_single_xt_plot`) responsible for drawing that specific plot type.
7.  This dedicated function uses the data from the `xarray` object and the specific plotting options from the configuration (read via `config.ax_opts`) to make calls to Matplotlib/Cartopy drawing functions (`contourf`, `plot`, `scatter`).
8.  Additional elements like colorbars, titles, labels, and grids are added, again using data metadata (units, long name) and configuration options for styling.
9.  Finally, a utility function (`pu.print_map`) is called to either save the completed `Figure` to a file or display it interactively.
10. The `Figure` object is then closed to free up memory.

This component is where all the planning, data loading, processing, and configuration comes together to produce the final visual output.

## Conclusion

In this chapter, you've been introduced to the **Plotting Engine**, the component in eViz that acts as the artist. It takes the prepared **xarray Dataset** from the **Data Processing Pipeline** and, guided by the detailed instructions in the **Configuration Files**, uses libraries like Matplotlib and Cartopy to draw maps, time series, and other plots. You saw the roles of the `Figure` class in managing the canvas and subplots, and the `Plotter` class in directing the specific drawing tasks based on plot type and styling from the configuration and data metadata.

We've now covered most of the major components of the eViz system that work together to load, process, and visualize data based on your configuration. But who is the mastermind that ties the Data Processing Pipeline and the Plotting Engine together for specific types of data, like 'gridded' or 'wrf'? This is the role of the **Model Implementations**.

Ready to see how specific data types are orchestrated through the system? Let's move on to the final conceptual chapter: [Model Implementations](08_model_implementations_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
