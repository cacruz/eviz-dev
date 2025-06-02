# Chapter 6: Figure (Visualization Canvas)

Welcome back! In the [previous chapter](05_datasource__base__.md), we learned about the fundamental **DataSource (Base)** concept, which provides a standard way to represent and access your data once it's loaded into eViz, regardless of its original file format.

Now that the data is loaded and ready, the next step is to visualize it! But where does this visualization actually *happen*? Where is the drawing space, the backdrop upon which all the colorful maps and graphs are created?

This is the role of the **Figure (Visualization Canvas)**.

## What is Figure and Why Do We Need It?

Think of creating a visualization like an artist creating a painting. The artist needs a **canvas** to paint on and an **easel** to hold the canvas steady. In eViz, the **Figure** is essentially this combined canvas and easel for your plots.

At its core, eViz's `Figure` is built upon Matplotlib's powerful `Figure` object. Matplotlib is the primary plotting library eViz uses. A Matplotlib `Figure` is the top-level container for all the elements of a plot – the axes, titles, legends, etc.

However, eViz needs to do more than just create a basic Matplotlib figure. It needs to:

1.  **Manage Multiple Plots:** Often, you want to show several plots side-by-side (like comparing different simulation runs or showing different variables from the same file). The `Figure` needs to organize this layout into **subplots**.
2.  **Handle Map Projections:** For geographical data, eViz needs to display maps correctly using different map projections (like Plate Carree, Mercator, etc.). The `Figure` must integrate with libraries like Cartopy (`cartopy.crs`) to set this up for map plots.
3.  **Incorporate eViz Settings:** The size, layout, and specific features (like adding coastlines or borders to maps) need to be configured based on your eViz settings provided via the [ConfigManager](02_configmanager_.md) (Chapter 2).

The eViz `Figure` abstraction takes Matplotlib's basic `Figure` and adds these capabilities, making it a prepared, ready-to-use canvas specifically tailored for eViz visualizations.

### A Simple Use Case: Preparing a Single Canvas for a Map

Imagine you want to visualize a single variable on a map of the globe. Before you can draw the data, you need a canvas ready. This canvas should:

1.  Be a standard plotting figure.
2.  Be set up to handle geographical coordinates (using a map projection like Plate Carree).
3.  Perhaps already include basic geographical features like coastlines.

Getting this specific canvas ready is the central job of the `Figure` in this simple case. If you needed two maps side-by-side, the `Figure` would prepare a canvas with two distinct drawing areas (subplots), both set up for maps.

### Key Aspects of the Figure (Visualization Canvas)

The eViz `Figure` class (`eviz.lib.autoviz.figure.Figure`) has several important characteristics:

1.  **Inheritance:** It inherits directly from `matplotlib.figure.Figure`. This means it *is* a Matplotlib Figure and has all the standard Matplotlib capabilities, while adding eViz-specific logic.
2.  **Subplot Management:** It uses `matplotlib.gridspec.GridSpec` internally to define the grid layout for multiple subplots. It figures out how many rows and columns are needed based on the eViz configuration ([ConfigManager](02_configmanager_.md)), especially for comparison plots.
3.  **Axes Container:** It keeps track of all the individual drawing areas (the subplots, which are `matplotlib.axes.Axes` or `cartopy.mpl.geoaxes.GeoAxes` objects) it creates. These are the areas where plotting functions will actually draw the data.
4.  **Map Projection Handling:** For plot types requiring maps, it uses Cartopy projections (`ccrs.Projection`) when creating the axes, ensuring they are geographically aware. It can look up the desired projection in the configuration.
5.  **Configuration Aware:** It receives the [ConfigManager](02_configmanager_.md) upon initialization and uses it to determine layout, size, projection, and other visual settings.

### How to Use the Figure (Conceptually)

As a user running `autoviz.py`, you don't directly create a `Figure` object yourself. Instead, the internal eViz workflow handles this. Typically, the main processing logic for a data source (the "Model Handler" which we'll cover later in [Chapter 8](08_model_handler__abstractroot__.md)) will:

1.  Receive the loaded data ([DataSource](05_datasource__base__.md)) and the configuration ([ConfigManager](02_configmanager_.md)).
2.  Determine what kind of plots are requested (e.g., map plots, time series).
3.  Use a helper method (like the `create_eviz_figure` factory method on the `Figure` class) to create an eViz `Figure` instance tailored for the specific plot type and configuration (e.g., single plot, side-by-side comparison, map projection needed).
4.  Call a method on the `Figure` (like `set_axes()`) to actually create the required subplot axes within the figure.
5.  Pass the `Figure` object and its created axes to the component responsible for drawing ([Plotter](07_plotter_.md)).

So, from your perspective, "using" the `Figure` means setting configuration options (in your YAML files or via command line) that influence how the `Figure` is created and set up (e.g., how many plots, which map projection). The system takes care of creating the `Figure` and giving it to the drawing routines.

### Inside Figure: Setting up the Canvas

Let's look at the internal process of how an eViz `Figure` is typically created and initialized.

When the main plotting logic (e.g., in the [Plotter](07_plotter_.md) or the Model Handler) needs a figure, it calls a helper method like `Figure.create_eviz_figure()`, passing the [ConfigManager](02_configmanager_.md) and the type of plot needed.

```{mermaid}
sequenceDiagram
    participant Caller as Plotting Logic
    participant FigFactory as Figure.create_eviz_figure()
    participant FigClass as Figure (Class)
    participant FigInst as Figure (Instance)
    participant ConfigM as ConfigManager
    participant GS as GridSpec

    Caller->>FigFactory: create_eviz_figure(config_manager, plot_type, ...)
    FigFactory->>ConfigM: Check config for layout (compare, subplots)
    ConfigM-->>FigFactory: Return layout info (e.g., nrows=1, ncols=2)
    FigFactory->>FigClass: Create new Figure(config_manager, plot_type, nrows=1, ncols=2)
    FigClass-->>FigInst: Return Figure instance (inherits from mpl.Figure)
    activate FigInst
    FigInst->>FigInst: _init_frame() # Determine size based on layout
    FigInst->>GS: Create GridSpec(1, 2) # Setup the grid layout
    GS-->>FigInst: Return GridSpec instance
    deactivate FigInst
    FigFactory-->>FigInst: Return Figure instance
    FigInst->>Caller: Return Figure instance (Figure is now created)

    Caller->>FigInst: figure.set_axes() # Tell Figure to create axes
    activate FigInst
    FigInst->>FigInst: create_subplot_grid() # (Already done in init, sets self.gs)
    FigInst->>FigInst: create_subplots() # Create Axes objects
    loop over grid (1 row, 2 cols)
        FigInst->>FigInst: add_subplot(gs[i,j], projection=...)
        FigInst->>FigInst: Store created Axes in self.axes_array
    end
    deactivate FigInst
    FigInst-->>Caller: Return Figure instance (Figure now has Axes ready for drawing)
```

1.  The plotting logic decides it needs a figure and calls the `create_eviz_figure` factory method.
2.  The factory method consults the [ConfigManager](02_configmanager_.md) to determine the desired subplot layout (e.g., for comparison plots).
3.  It creates a new `Figure` instance, passing the configuration and layout info.
4.  During the `Figure`'s initialization (`__init__` and `_init_frame`), it sets up the `matplotlib.gridspec.GridSpec` based on the requested layout.
5.  The initialized `Figure` object is returned. At this point, the canvas and its grid are defined, but there are no drawing areas yet.
6.  The plotting logic then calls `figure.set_axes()`.
7.  The `set_axes` method calls `create_subplots`, which iterates through the defined `GridSpec` and calls `self.add_subplot()` for each grid cell. This is where the actual `Axes` objects are created. If it's a map plot, it will create `GeoAxes` with the correct Cartopy projection.
8.  The created `Axes` objects are stored internally (e.g., in `self.axes_array`).
9.  The `Figure` object, now fully set up with its drawing areas, is returned to the plotting logic, ready to be passed to the [Plotter](07_plotter_.md).

### Code Walkthrough (Simplified)

Let's look at some simplified snippets from `eviz/lib/autoviz/figure.py` to see these pieces.

First, the class definition and initialization:

```python
# eviz/lib/autoviz/figure.py (simplified)
import logging
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt # Often used alongside figures
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs # For map projections

# Figure inherits from matplotlib's Figure
class Figure(mfigure.Figure):
    """
    Enhanced Figure class inheriting from matplotlib's Figure.
    Manages subplot layout and map projections.
    """
    def __init__(self, config_manager, plot_type,
        *,
        nrows=None,
        ncols=None,
        **kwargs,
    ):
        # Internal storage for grid and axes
        self._gridspec = None
        self._subplots = (1, 1) # Default to a single subplot
        self.axes_array = []   # List to hold the created Axes objects

        # eViz-specific attributes
        self.config_manager = config_manager
        self.plot_type = plot_type
        self._logger = logging.getLogger(__name__)

        # Determine initial number of subplots based on provided args
        if nrows is not None and ncols is not None:
            self._subplots = (nrows, ncols)

        # --- Simplify: Logic to set _subplots based on config happens here ---
        # (See _set_compare_diff_subplots in the full code)
        # For example, if config_manager.compare is True, it might set _subplots to (1, 2)

        # Call the parent (matplotlib.figure.Figure) constructor
        # Remove nrows/ncols as they are handled by GridSpec, not mpl Figure directly
        if 'nrows' in kwargs: del kwargs['nrows']
        if 'ncols' in kwargs: del kwargs['ncols']
        super().__init__(**kwargs)

        # Set up the initial frame/size and GridSpec
        self._init_frame()
        self.create_subplot_grid() # Creates self.gs (the GridSpec)


    def _init_frame(self):
        """
        Determines figure size based on _subplots and plot type,
        sets self._frame_params. (Simplified)
        """
        # Logic here looks at self._subplots and self.plot_type
        # and sets the figure size using self.set_size_inches(...)
        # For simplicity, we omit the details.
        pass # Simplified


    def create_subplot_grid(self) -> "Figure":
        """Create the GridSpec based on the determined subplot layout."""
        # Use the determined number of rows and columns (self._subplots)
        # to create the GridSpec.
        self.gs = gridspec.GridSpec(*self._subplots)

        # Adjust spacing if needed (e.g., for comparison plots - see full code)

        return self # Return self to allow method chaining

    # ... other methods omitted ...
```

This snippet shows that the `Figure` inherits from `mfigure.Figure`, stores the configuration and plot type, initializes the subplot layout (`self._subplots`), and then creates the `GridSpec` (`self.gs`) in `create_subplot_grid`.

Next, the factory method and the `set_axes`/`create_subplots` methods:

```python
# eviz/lib/autoviz/figure.py (simplified)
# ... imports and class definition ...

    @staticmethod
    def create_eviz_figure(config_manager,
                           plot_type,
                           field_name=None, nrows=None, ncols=None) -> "Figure":
        """
        Factory method to create an eViz Figure instance.
        Handles layout logic based on config.
        """
        # Determine layout (nrows, ncols) based on config (comparison, overlay, etc.)
        # The logic here determines the final nrows and ncols, often overriding
        # the default 1x1 or adapting for comparison views.
        # For simplicity, assume it sets nrows and ncols correctly based on config.
        
        # Use the determined layout to create the Figure instance
        fig = Figure(config_manager, plot_type, nrows=nrows, ncols=ncols)
        # Note: The Figure's __init__ and create_subplot_grid already ran

        return fig # Return the newly created Figure instance


    def set_axes(self) -> "Figure":
        """
        Create the actual Matplotlib Axes objects (subplots) inside the Figure.
        This is where the drawing areas are added.
        """
        # Check if map projection is needed based on plot type
        self._use_cartopy = ('tx' in self.plot_type or 'sc' in self.plot_type or 'xy' in self.plot_type)

        # Create the Axes objects, potentially with Cartopy projection
        self.create_subplots()

        # Return self to allow method chaining
        return self


    def create_subplots(self):
        """
        Create subplots based on the GridSpec and projection requirements.
        Populates self.axes_array.
        """
        if self.use_cartopy:
            # If Cartopy is needed, figure out the projection
            map_projection = self.get_projection() # Method to get ccrs.Projection

            # Iterate through the grid and add Axes with the projection
            for i in range(self._subplots[0]):
                for j in range(self._subplots[1]):
                    # Add a subplot, specifying the GridSpec cell and the projection
                    ax = self.add_subplot(self.gs[i, j], projection=map_projection)
                    # Add basic map features like coastlines (simplified)
                    if hasattr(ax, 'coastlines'):
                        ax.coastlines()
                        # ax.add_feature(cfeature.BORDERS) # Example feature
                    self.axes_array.append(ax) # Store the created axis
        else:
            # If no Cartopy, just add standard Axes
            for i in range(self._subplots[0]):
                for j in range(self._subplots[1]):
                    # Add a standard subplot
                    ax = self.add_subplot(self.gs[i, j])
                    self.axes_array.append(ax) # Store the created axis

        # Return self to allow method chaining
        return self

    def get_projection(self, projection_name=None) -> ccrs.Projection | None:
        """Helper method to get a Cartopy projection instance."""
        # This method looks up the projection name (either provided or from config)
        # and returns the corresponding cartopy.crs.Projection object.
        # Defaults to PlateCarree if none specified.
        if projection_name is None:
             return ccrs.PlateCarree()
        # ... lookup logic based on projection_name ...
        # Example: if projection_name == 'Mercator': return ccrs.Mercator(...)
        pass # Simplified

    def get_axes(self) -> list:
        """Return the list of created Axes objects."""
        # This is how plotting routines get access to the drawing areas
        return self.axes_array

    @property
    def use_cartopy(self):
        """Boolean property indicating if Cartopy is being used for these axes."""
        return self._use_cartopy

    # ... other methods omitted ...
```

This snippet shows how the `create_eviz_figure` factory method is used to instantiate the `Figure`, how `set_axes` triggers the creation of the actual drawing areas, and how `create_subplots` adds either standard `Axes` or Cartopy `GeoAxes` to the figure based on whether `use_cartopy` is true, storing them in `self.axes_array`. The `get_axes()` method provides the crucial way for the plotting logic to retrieve these ready-to-use drawing areas.

### Conclusion

In this chapter, we introduced the **Figure (Visualization Canvas)**, the eViz component built on Matplotlib that serves as the canvas and easel for your plots. It manages the overall size and layout, sets up subplots using `GridSpec`, and integrates with Cartopy for map projections.

By using eViz's `Figure`, the complex details of setting up the plotting space, especially for multiple plots or geographical visualizations, are handled automatically based on your configuration. The end result is a `Figure` object containing one or more `Axes` objects, prepared and ready to receive the data visualizations.

With the data loaded ([DataSource](05_datasource__base__.md)) and the canvas prepared (`Figure`), the next step is to actually draw the data onto the canvas. This is the responsibility of the **[Plotter](07_plotter_.md)**, which we will explore in the next chapter.

[Next Chapter: Plotter](07_plotter_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)