# eviz/lib/autoviz/plotting/backends/hvplot/xy_plot.py
import numpy as np
# import hvplot.xarray
# import holoviews as hv
from bokeh.models import HoverTool

from ....plotting.base import XYPlotter

class HvplotXYPlotter(XYPlotter):
    """HvPlot implementation of XY plotting."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
    
    def plot(self, config, data_to_plot):
        """Create an interactive XY plot using HvPlot.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created HvPlot object
        """
        pass
        # data2d, x, y, field_name, plot_type, findex, _ = data_to_plot
        
        # if data2d is None:
        #     self.logger.warning("No data to plot")
        #     return None
        
        # # Get axes options from config
        # ax_opts = config.ax_opts
        
        # # Handle fill values
        # if 'fill_value' in config.spec_data[field_name]['xyplot']:
        #     fill_value = config.spec_data[field_name]['xyplot']['fill_value']
        #     data2d = data2d.where(data2d != fill_value, np.nan)
        
        # # Get colormap
        # cmap = ax_opts.get('use_cmap', 'viridis')
        
        # # Get title
        # title = field_name
        # if 'name' in config.spec_data[field_name]:
        #     title = config.spec_data[field_name]['name']
        
        # # Get units
        # units = "n.a."
        # if 'units' in config.spec_data[field_name]:
        #     units = config.spec_data[field_name]['units']
        # elif hasattr(data2d, 'attrs') and 'units' in data2d.attrs:
        #     units = data2d.attrs['units']
        # elif hasattr(data2d, 'units'):
        #     units = data2d.units
        
        # # Create hover tool
        # hover = HoverTool(
        #     tooltips=[
        #         ("x", "@x"),
        #         ("y", "@y"),
        #         (field_name, "@image{0.00} " + units)
        #     ]
        # )
        
        # # Create the plot
        # plot = data2d.hvplot.contourf(
        #     x=x.name if hasattr(x, 'name') else 'x',
        #     y=y.name if hasattr(y, 'name') else 'y',
        #     cmap=cmap,
        #     title=title,
        #     width=800,
        #     height=500,
        #     colorbar=True,
        #     clabel=units,
        #     tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', hover]
        # )
        
        # # Store the plot object
        # self.plot_object = plot
        
        # return plot
    
    def save(self, filename, **kwargs):
        """Save the plot to an HTML file."""
        if self.plot_object is not None:
            import holoviews as hv
            hv.save(self.plot_object, filename)
            self.logger.info(f"Saved interactive plot to {filename}")
        else:
            self.logger.warning("No plot to save")
    
    def show(self):
        """Display the plot."""
        if self.plot_object is not None:
            import holoviews as hv
            hv.render(self.plot_object)
        else:
            self.logger.warning("No plot to show")
