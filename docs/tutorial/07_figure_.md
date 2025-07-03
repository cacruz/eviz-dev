# Chapter 7: Figure

Welcome back to the eViz tutorial! In our previous chapters, we've seen how eViz handles the raw data ([Chapter 1: Data Source](01_data_source_.md), [Chapter 5: Data Pipeline](05_data_pipeline_.md)), loads and manages your visualization instructions ([Chapter 2: Config Manager](02_config_manager_.md), [Chapter 3: YAML Parser](03_yaml_parser_.md)), is orchestrated by the main application ([Chapter 4: Autoviz (Main Application)](04_autoviz__main_application__.md)), and how specialized handlers interpret data specific to your model or source ([Chapter 6: Model/Source Handler](06_model_source_handler_.md)).

Now that a [Model/Source Handler](06_model_source_handler_.md) has retrieved and prepared the data it needs for a specific plot, where does that data actually *go*? It needs a canvas to be drawn on! This is where the **Figure** concept comes in.

## What Problem Does Figure Solve?

In any plotting library, you need a container for your plot. In Python's popular Matplotlib library (which eViz uses under the hood), this is typically a `Figure` object, which acts like a blank canvas. Within this figure, you add one or more `Axes` objects, which are the actual areas where data is plotted (think of them as the individual graphs within the canvas).

```python
# Simple Matplotlib example (outside of eViz)
import matplotlib.pyplot as plt

# Create a Figure (the canvas) and an Axes (the plotting area)
fig, ax = plt.subplots(1, 1) # 1 row, 1 column layout

# Now you can plot data ON the 'ax' object
ax.plot([1, 2, 3], [4, 5, 6])
ax.set_title("My Simple Plot")
plt.show() # Display the figure
```

While a standard Matplotlib `Figure` provides the basic canvas, scientific visualizations often require more:
*   Complex layouts (side-by-side comparisons, multiple rows/columns).
*   Handling geographical projections for maps (requiring a special type of Axes, like Cartopy's `GeoAxes`).
*   Consistently adding elements like main titles, colorbars, grid lines across different plot types and comparison views, often with specific styling based on your configuration.
*   Managing plot-specific settings (like contour levels, color maps, axis limits) that are defined in your [Config Manager](02_config_manager_.md)'s `spec_data`.

The **Figure** class in eViz (`eviz/lib/autoviz/figure.py`) is a specialized extension of the standard Matplotlib `Figure`. It's customized to handle these scientific visualization needs, acting as a tailored canvas that knows how to arrange subplots, set up map projections, apply configuration-defined styling, and manage common plot elements consistently.

Think of it as a specialized stage manager. It doesn't perform the acting (drawing the data), but it sets up the stage (the figure), arranges the lighting and props (axes, titles, colorbars), and ensures everything is positioned correctly according to the script (your configuration).

## Our Central Use Case: Setting Up a Multi-Panel Map with Projection

Let's say your configuration asks for a map plot of temperature from two different model runs, displayed side-by-side for comparison. This requires:
1.  A single figure (the canvas).
2.  Two plotting areas (axes) side-by-side within that figure.
3.  Each axes needing a geographical projection (e.g., Plate Carree) so the map looks correct.
4.  Applying specific title settings, potentially colorbar options, etc., defined in your config for the 'temperature' variable.

The eViz `Figure` class is designed to set up exactly this kind of multi-panel, geographically projected plotting space based on your configuration.

## Using the Concept (Mostly Internal)

Like the [Data Pipeline](05_data_pipeline_.md) and [Model/Source Handlers](06_model_source_handler_.md), you generally don't create `Figure` objects directly in your YAML configuration or user code. Instead, the [Model/Source Handler](06_model_source_handler_.md) that's currently processing a plot task is responsible for requesting and setting up the necessary `Figure` object *before* it calls the [Plotter Backend](08_plotter_backend_.md) to draw the data.

The `Figure` class provides a special class method (`create_eviz_figure`) which acts like a factory function. The [Model/Source Handler](06_model_source_handler_.md) calls this method, passing the current [Config Manager](02_config_manager_.md) and the type of plot being requested (e.g., 'xyplot', 'scplot').

Here's a simplified look at how a [Model/Source Handler](06_model_source_handler_.md) might create and prepare a `Figure`:

```python
# Imagine this simplified code is inside a Model/Source Handler method
# like GenericSource.__call__ or a method it calls

from eviz.lib.autoviz.figure import Figure
# config_manager object is available via self.config_manager

# ... (Handler determines plot_type, field_name from map_params) ...
plot_type = 'xyplot' # e.g., map plot
field_name = 'temperature'
print(f"Handler needs a figure for {field_name} ({plot_type})")

# 1. Use the factory method to create the Figure instance
# This method looks at config_manager (for comparison, overlay)
# and plot_type to decide the layout (nrows, ncols)
figure = Figure.create_eviz_figure(
    config_manager=self.config_manager, # Pass the config
    plot_type=plot_type,                # Specify plot type
    field_name=field_name               # Specify field name (for specs lookup)
)
print(f"Created Figure object: {type(figure)}")
print(f"Initial layout: {figure.subplots} rows, {figure.subplots[1]} columns") # Accesses the _subplots property

# 2. Tell the Figure object to set up the actual plotting areas (Axes)
# This method creates the Matplotlib/Cartopy Axes objects and adds them to the figure
figure.set_axes()
print(f"Figure axes created. Number of axes: {len(figure.get_axes())}") # Accesses the axes_array

# 3. Initialize axis options based on config
# This loads settings like contour levels, colormaps, projection from spec_data
ax_options = figure.init_ax_opts(field_name)
print(f"Initialized axis options from spec_data for {field_name}")
# print(f"Example option: clevs = {ax_options.get('clevs')}") # Show an option

# Now, the 'figure' object is ready. It has the correct layout,
# the right type of axes (with or without projection), and
# axis-specific settings loaded from the config.
# The Handler would then pass this 'figure' and its axes to the Plotter Backend.
```

This example shows the key steps: use `create_eviz_figure` to get the Figure object, then call `set_axes` to make the axes, and `init_ax_opts` to load specific settings for those axes.

## Breaking Down the Concept: Key Parts

The `Figure` class builds upon Matplotlib and Cartopy to provide eViz's specialized functionality. Here are some key internal parts:

1.  **Inheritance:** The `Figure` class inherits directly from `matplotlib.figure.Figure`. This means it *is* a Matplotlib figure and has all its standard capabilities, allowing eViz to use familiar Matplotlib functions for the actual drawing.
    ```python
    # Simplified inheritance declaration
    import matplotlib.figure as mfigure

    class Figure(mfigure.Figure):
        # ... rest of the class ...
        pass # Inherits methods like add_subplot, savefig, show etc.
    ```
    This ensures compatibility with the underlying plotting library.

2.  **Configuration and Plot Type:** The `Figure` stores the `config_manager` and the `plot_type` it's being used for. This allows its methods to access configuration settings (like comparison mode, output size, specific plot options from `spec_data`) and tailor the setup based on whether it's a map (`xy`, `sc`), time series (`tx`), profile (`yz`, `xt`), etc.
    ```python
    # Simplified init
    def __init__(self, config_manager, plot_type, *, nrows=None, ncols=None, **kwargs):
        # Store references
        self.config_manager = config_manager
        self.plot_type = plot_type
        self._logger = logging.getLogger(__name__)

        # Determine initial subplot layout based on config/args
        if nrows is not None and ncols is not None:
            self._subplots = (nrows, ncols)
        else:
             # Calls _set_compare_diff_subplots internally or similar logic
             # to determine default layout based on self.config_manager.compare etc.
             self._subplots = (1, 1) # Placeholder - real logic is more complex

        # Call the parent Matplotlib Figure constructor
        super().__init__(**kwargs)

        # Further eViz setup
        self._init_frame() # Sets figure size based on _subplots
        self._ax_opts = {} # Initialize axis options dictionary
    ```
    The `__init__` method receives the essential context (`config_manager`, `plot_type`) and calls the parent Matplotlib constructor.

3.  **Layout (`_subplots`, `gs`, `axes_array`):**
    *   `_subplots` (a tuple like `(1, 1)` or `(1, 2)`) stores the determined grid layout (rows, columns). It's set during initialization, considering comparison modes (`_set_compare_diff_subplots` method handles this logic).
    *   `gs` (a `matplotlib.gridspec.GridSpec` object) is created by `create_subplot_grid()` based on `_subplots`. It defines the geometry of the grid within the figure, allowing for precise placement of axes.
    *   `axes_array` (a list) stores the actual `matplotlib.axes.Axes` or `cartopy.mpl.geoaxes.GeoAxes` objects created within the grid. The [Plotter Backend](08_plotter_backend_.md) will plot data *on* these axes objects.
    ```python
    # Simplified snippet from create_subplot_grid
    def create_subplot_grid(self) -> "Figure":
        """Create a grid of subplots based on the figure frame layout."""
        # self._subplots holds the determined (nrows, ncols)
        # Use GridSpec to manage the grid geometry
        if self.config_manager.compare and not self.config_manager.compare_diff:
             # Adjust spacing for side-by-side comparison
             self.gs = gridspec.GridSpec(*self._subplots, wspace=0.3)
        else:
             # Default GridSpec for single plots or difference plots
             self.gs = gridspec.GridSpec(*self._subplots)

        # Optional: Adjust figure size based on determined layout
        # self._init_frame() sets self._frame_params based on _subplots
        # and uses self.set_size_inches(...)

        return self

    # Simplified snippet from create_subplots
    def create_subplots(self):
        """Create subplots based on gridspec and projection."""
        self.axes_array = [] # Initialize the list of axes
        if self.use_cartopy: # Check if a map projection is needed
            return self._create_subplots_crs() # Create Cartopy axes
        else:
            for i in range(self._subplots[0]): # Loop through rows
                for j in range(self._subplots[1]): # Loop through columns
                    # Add a standard Matplotlib Axes object to the grid
                    ax = self.add_subplot(self.gs[i, j])
                    self.axes_array.append(ax) # Store the created axes
            return self
    ```
    These methods work together to determine the grid and populate it with the correct number and type of axes.

4.  **Geographical Projection (`_use_cartopy`, `_projection`, `get_projection`, `_create_subplots_crs`):**
    *   `_use_cartopy` is a boolean flag set by `set_axes()` or `_get_fig_ax()` if the `plot_type` indicates a map (`xy`, `sc`).
    *   `get_projection()` determines the specific Cartopy projection object (like `ccrs.PlateCarree`, `ccrs.Mercator`, etc.) to use. It looks for a 'projection' setting in the `_ax_opts` (which comes from the [Config Manager](02_config_manager_.md)'s `spec_data`) or defaults to Plate Carree. It also handles map extents and central coordinates based on configuration.
    *   `_create_subplots_crs()` is called by `create_subplots()` if `_use_cartopy` is true. It uses the projection from `get_projection()` when calling `self.add_subplot()`, creating `GeoAxes` objects instead of standard `Axes`. It also adds common map features like coastlines and borders.
    ```python
    # Simplified snippet from _create_subplots_crs
    def _create_subplots_crs(self) -> "Figure":
        """Create subplots with cartopy projections."""
        # Determine the projection to use based on _ax_opts or default
        map_projection = self.get_projection(self._ax_opts.get('projection'))

        self.axes_array = [] # Initialize the list of axes
        for i in range(self._subplots[0]):
            for j in range(self._subplots[1]):
                # Add a Cartopy GeoAxes object with the determined projection
                ax = self.add_subplot(self.gs[i, j], projection=map_projection)
                self.axes_array.append(ax) # Store the created axes

        # Add map features to each GeoAxes
        for ax in self.axes_array:
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            # ... add other features ...

        return self

    # Simplified snippet from get_projection
    def get_projection(self, projection=None) -> Optional[ccrs.Projection]:
        """Get projection parameter."""
        # Look for extent/projection name in self._ax_opts
        extent = self._ax_opts.get('extent', [-180, 180, -90, 90]) # Default to global
        projection_name = projection or self._ax_opts.get('projection')

        if projection_name is None or projection_name.lower() == 'platecarree':
            # Default or explicit Plate Carree
            self._projection = ccrs.PlateCarree()
        elif projection_name.lower() == 'mercator':
             self._projection = ccrs.Mercator(...) # Use central_lon from _ax_opts
        # ... handle other projection names ...
        else:
             self._logger.warning(f"Unknown projection: {projection_name}")
             self._projection = ccrs.PlateCarree() # Fallback

        # If extent is specified, set the limits for the projection
        if extent:
             self._ax_opts['extent'] = extent # Store used extent
             # This is applied to the axes later by the Plotter Backend using ax.set_extent(extent, crs=ccrs.PlateCarree())

        return self._projection
    ```
    This shows how the Figure class intelligently sets up the correct type of axes and their associated projections and extents based on the plot type and config.

5.  **Axis Options (`_ax_opts`, `init_ax_opts`, `update_ax_opts`, `apply_rc_params`):**
    *   `_ax_opts` is a dictionary holding specific settings for the axes *before* the plotting happens (or is updated during plotting). This includes things like contour levels (`clevs`), color maps (`use_cmap`, `use_diff_cmap`), color scale type (`cscale`, `zscale`), title options, grid options, and raw Matplotlib runtime configuration parameters (`rc_params`).
    *   `init_ax_opts(field_name)` populates `_ax_opts` by looking up plot-specific settings for the given `field_name` and `plot_type` within the [Config Manager](02_config_manager_.md)'s `spec_data`. It applies defaults if settings aren't found.
    *   `update_ax_opts(...)` can modify `_ax_opts` later, for instance, to set difference plot contour levels during a comparison run.
    *   `apply_rc_params()` takes the `rc_params` from `_ax_opts` and applies them globally to Matplotlib, affecting font sizes, line styles, etc.
    ```python
    # Simplified snippet from init_ax_opts
    def init_ax_opts(self, field_name) -> Dict[str, Any]:
        """Initialize map options for a given field."""
        plot_type_short = self.plot_type[:2] # Get 'xy', 'yz', 'tx', 'sc', 'po'
        # Look up settings in spec_data for this field and plot type
        spec = self.config_manager.spec_data.get(field_name, {}).get(f"{plot_type_short}plot", {})

        # Define default options
        defaults = {
            'use_cmap': self.config_manager.input_config._cmap, # Default cmap from config
            'clevs': None, # No contour levels by default
            'create_clevs': False, # Don't automatically create levels by default
            'projection': None, # No specific projection by default
            'extent': None, # No specific extent by default
            'rc_params': {}, # Empty dict for rcParams
            # ... many other defaults ...
        }

        # Populate _ax_opts, taking values from spec if they exist, otherwise using defaults
        new_ax_opts = {}
        for key, default_value in defaults.items():
            if key == 'rc_params':
                 # Handle rc_params separately to merge instead of overwrite
                 # Combine existing rc_params from _ax_opts (if any) with new ones from spec
                 existing = self._ax_opts.get('rc_params', {}).copy()
                 from_spec = spec.get('rc_params', {})
                 existing.update(from_spec) # Merge
                 new_ax_opts[key] = existing
            else:
                 new_ax_opts[key] = spec.get(key, default_value) # Use spec value or default

        self._ax_opts = new_ax_opts # Store the initialized options

        # Apply rc_params immediately
        self.apply_rc_params(default_params=self.get_default_plot_params()) # Uses default Matplotlib params

        return self._ax_opts
    ```
    This method is crucial for pulling the detailed plotting instructions from your YAML configuration (`spec_data`) and making them available to the Figure and later the [Plotter Backend](08_plotter_backend_.md).

6.  **Helper Methods (`plot_text`, `colorbar_eviz`, `add_grid`, etc.):** The `Figure` class includes various helper methods to add common plot elements.
    *   `plot_text()`: Adds titles (main figure title via `suptitle_eviz`, subplot titles via `ax.set_title`), field names, level information, and potentially basic stats to the axes. It handles the complex logic for comparison plot titles based on the subplot's position within the grid.
    *   `colorbar_eviz()`: A wrapper around creating a colorbar, often using `make_axes_locatable` to place it nicely next to an axes.
    *   `add_grid()`: Adds grid lines to axes, optionally with specific tick locations.
    ```python
    # Simplified snippet from plot_text (showing title logic for comparison)
    def _plot_text(self, field_name, ax, pid, level=None, data=None, **kwargs):
        # ... (determine font sizes etc.) ...

        # Check if it's a comparison plot layout
        if self.config_manager.compare or self.config_manager.compare_diff:
            geom = pu.get_subplot_geometry(ax) # Helper to get axes position (rows, cols, row_idx, col_idx)
            title_string = 'Placeholder'
            if geom and geom[0] == (3, 1): # Example: 3 rows, 1 column layout
                 if geom[1:] == (0, 1, 1, 1): # Position of the bottom plot (often the difference)
                      title_string = "Difference (top - middle)"
                 elif geom[1:] in [(1, 1, 0, 1), (0, 1, 0, 1)]: # Positions of top/middle plots
                      # Use _set_axes_title to get title based on file/exp config
                      title_string = self._set_axes_title(self.config_manager.findex)
            # ... handle other comparison layouts like (2,2), (1,2) ...

            ax.set_title(title_string, loc=kwargs.get('location', 'left'), fontsize=title_fontsize)
            return # Don't add other text like field name in comparison titles

        # ... (Non-comparison title logic and field name text placement) ...
    ```
    This illustrates how `plot_text` uses the figure's layout information and the config manager to generate informative titles for complex plots.

## Under the Hood: Figure Creation Flow

Here's a simplified sequence diagram showing how a `Figure` object is typically created and set up:

```{mermaid}
sequenceDiagram
    participant Handler as Model/Source Handler
    participant FigureClass as Figure (Class)
    participant FigureInstance as Figure (Instance)
    participant ConfigMgr as ConfigManager
    participant Matplotlib as matplotlib
    participant Cartopy as cartopy.crs

    Handler->>FigureClass: create_eviz_figure(ConfigMgr, plot_type, field_name)
    FigureClass->>ConfigMgr: Check config (compare, overlay)
    ConfigMgr-->>FigureClass: Return relevant config values
    FigureClass->>FigureClass: Determine nrows, ncols
    FigureClass->>FigureInstance: Create Figure(ConfigMgr, plot_type, nrows, ncols, ...)
    FigureInstance->>FigureInstance: __init__(...)
    FigureInstance->>Matplotlib: super().__init__(...) # Call parent Matplotlib Figure
    Matplotlib-->>FigureInstance: Return Matplotlib Figure base
    FigureInstance->>FigureInstance: _init_frame() # Set figure size/shape
    FigureInstance->>FigureInstance: _ax_opts = {} # Init options dict
    FigureInstance->>FigureInstance: init_ax_opts(field_name)
    FigureInstance->>ConfigMgr: Lookup spec_data for field_name, plot_type
    ConfigMgr-->>FigureInstance: Return plot specs
    FigureInstance->>FigureInstance: Populate _ax_opts with specs/defaults
    FigureInstance->>FigureInstance: apply_rc_params() # Apply Matplotlib settings
    FigureInstance->>Matplotlib: mpl.rcParams.update(...)
    Matplotlib-->>FigureInstance: Settings applied
    FigureInstance-->>FigureClass: Return Figure instance
    FigureClass-->>Handler: Return Figure instance (figure)

    Handler->>FigureInstance: set_axes()
    FigureInstance->>FigureInstance: create_subplot_grid()
    FigureInstance->>Matplotlib: gridspec.GridSpec(...)
    Matplotlib-->>FigureInstance: Return GridSpec (gs)
    FigureInstance->>FigureInstance: create_subplots()
    alt if use_cartopy
        FigureInstance->>FigureInstance: _create_subplots_crs()
        FigureInstance->>FigureInstance: get_projection()
        FigureInstance->>Cartopy: ccrs.PlateCarree() or other
        Cartopy-->>FigureInstance: Return projection object
        FigureInstance->>Matplotlib: self.add_subplot(gs[...], projection=...) # Create GeoAxes
        Matplotlib-->>FigureInstance: Return GeoAxes
        loop for each subplot
            FigureInstance->>FigureInstance: add GeoAxes to axes_array
        end
        FigureInstance->>Cartopy: Add features (coastlines, borders)
    else else (no cartopy)
        loop for each subplot
            FigureInstance->>Matplotlib: self.add_subplot(gs[...]) # Create standard Axes
            Matplotlib-->>FigureInstance: Return Axes
            FigureInstance->>FigureInstance: add Axes to axes_array
        end
    end
    FigureInstance-->>Handler: return Figure instance
```
This diagram shows how `create_eviz_figure` sets up the instance and loads options, and then `set_axes` is called to populate the figure with the actual plotting areas (`axes_array`), handling the complexities of grid layout and geographical projections based on the stored configuration and plot type.

## Summary

In this chapter, we learned about the **Figure** class:

*   It's eViz's specialized canvas for scientific plots, extending Matplotlib's `Figure`.
*   It solves the problem of setting up complex plot layouts, handling geographical projections (via Cartopy), and managing plot-specific settings consistently.
*   [Model/Source Handlers](06_model_source_handler_.md) use the `Figure.create_eviz_figure()` factory method to instantiate and prepare a `Figure` object.
*   Key steps in preparation include determining the subplot layout (`_subplots`), creating the grid structure (`gs`), generating the correct type of axes (`axes_array`, using `_create_subplots_crs` for maps), setting up geographical projections (`get_projection`), and loading axis-specific plot options from the configuration (`init_ax_opts`).
*   It provides helper methods for adding common elements like titles, colorbars, and grids, often with logic tailored for comparison plots or map types.

The `Figure` class ensures that by the time eViz is ready to draw data, it has a perfectly prepared space – the right number of panels, the correct map projection (if needed), and all the necessary plot settings ready to be applied to the axes.

Now that the data is ready ([Data Pipeline](05_data_pipeline_.md), [Model/Source Handler](06_model_source_handler_.md)) and the canvas is set up (Figure), the final step is to actually draw the data onto the axes. This is the job of the **Plotter Backend**.

Let's move on to [Chapter 8: Plotter Backend](08_plotter_backend_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)