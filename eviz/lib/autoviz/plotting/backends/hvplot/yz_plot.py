import numpy as np
import pandas as pd
import hvplot.xarray  # register the hvplot method with xarray objects
import holoviews as hv
import hvplot.pandas  # noqa
from holoviews import opts

from eviz.lib.autoviz.plotting.base import YZPlotter


class HvplotYZPlotter(YZPlotter):
    """
    YZ Plotter implementation for hvplot backend.

    This class creates latitude-height (YZ) plots using hvplot, which are
    typically used for visualizing zonal means of atmospheric variables.
    """

    def __init__(self):
        """Initialize the YZ Plotter."""
        super().__init__()
        try:
            hv.extension('bokeh')
        except Exception as e:
            self.logger.warning(f"Could not initialize HoloViews/hvplot extensions: {e}")   

    def plot(self, config_manager, plot_data):
        """
        Create a YZ (latitude-height) plot using hvplot.

        Parameters:
            config_manager: The configuration manager object
            plot_data: Tuple containing (data_array, x, y, field_name, plot_type, file_index, figure)
                data_array: xarray.DataArray containing the data to plot
                x: x-coordinates (latitude values)
                y: y-coordinates (height/pressure levels)
                field_name: Name of the field being plotted
                plot_type: Type of plot ('yz')
                file_index: Index of the file being plotted
                figure: Figure object to plot on

        Returns:
            holoviews.Element: The created plot
        """
        data_array, x, y, field_name, plot_type, file_index, figure = plot_data

        # Get plot configuration
        ax_opts = config_manager.ax_opts
        cmap = ax_opts.get("cmap", "viridis")
        title = ax_opts.get("title", field_name)
        colorbar_label = ax_opts.get("colorbar_label", "")

        # Get dimensions from data
        if hasattr(data_array, "dims"):
            dims = list(data_array.dims)
            if len(dims) != 2:
                raise ValueError(f"Expected 2D data for YZ plot, got {len(dims)}D")

            # Typically for YZ plots, first dimension is vertical level, second is latitude
            y_dim, x_dim = dims
        else:
            # Fallback if dims not available
            y_dim, x_dim = "lev", "lat"

        # Set up plot options
        plot_opts = {
            "cmap": cmap,
            "colorbar": True,
            "title": title,
            "clabel": colorbar_label,
            "xlabel": "Latitude",
            "ylabel": "Pressure (hPa)" if "pressure" in y_dim.lower() else "Level",
            "invert_yaxis": "pressure" in y_dim.lower() or "lev" in y_dim.lower(),
            "width": ax_opts.get("figsize", (8, 6))[0] * 100,
            "height": ax_opts.get("figsize", (8, 6))[1] * 100,
        }

        # Handle contour levels if specified
        if "levels" in ax_opts:
            plot_opts["levels"] = ax_opts["levels"]

        # Handle colorbar range if specified
        if "vmin" in ax_opts and "vmax" in ax_opts:
            plot_opts["clim"] = (ax_opts["vmin"], ax_opts["vmax"])

        # Create the plot
        plot = data_array.hvplot.contourf(x=x_dim, y=y_dim, **plot_opts)

        # Add additional styling
        plot = plot.opts(
            opts.Contours(
                line_width=0.5,
                tools=["hover"],
            )
        )

        # Store the plot in the figure object if provided
        if figure and hasattr(figure, "plot"):
            figure.plot = plot

        return plot

    def show(self):
        """Display the plot."""
        pass

    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        pass
