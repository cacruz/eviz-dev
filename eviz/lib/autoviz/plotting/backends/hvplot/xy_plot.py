import numpy as np
import pandas as pd
import logging
import holoviews as hv
import hvplot.xarray  # register the hvplot method with xarray objects
import hvplot.pandas
from eviz.lib.autoviz.plotting.base import XYPlotter


class HvplotXYPlotter(XYPlotter):
    """HvPlot implementation of XY plotting."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._apply_numpy_compatibility_patch()

        # Set up HoloViews and hvplot extensions
        try:
            hv.extension('bokeh')
            self.logger.debug("Successfully initialized HoloViews and hvplot extensions")
        except Exception as e:
            self.logger.warning(f"Could not initialize HoloViews/hvplot extensions: {e}")   

    def _apply_numpy_compatibility_patch(self):
        """Apply compatibility patch for NumPy 1.20+ with older HoloViews/hvplot."""
        try:
            import numpy as np
            if not hasattr(np, 'bool'):
                self.logger.debug("Applying NumPy compatibility patch for bool")
                np.bool = bool
            
            # Add other deprecated NumPy aliases that might be needed
            if not hasattr(np, 'int'):
                np.int = int
            if not hasattr(np, 'float'):
                np.float = float
            if not hasattr(np, 'complex'):
                np.complex = complex
            if not hasattr(np, 'object'):
                np.object = object
            if not hasattr(np, 'str'):
                np.str = str
        except Exception as e:
            self.logger.warning(f"Failed to apply NumPy compatibility patch: {e}")        

    def plot(self, config, data_to_plot):
        """Create an interactive XY plot using HvPlot.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created HvPlot object
        """
        data2d, x, y, field_name, plot_type, findex, _ = data_to_plot
         
        if data2d is None:
            self.logger.warning("No data to plot")
            return None
        
        self.logger.debug(f"Data shape: {data2d.shape}")
        self.logger.debug(f"X coords shape: {x.shape if hasattr(x, 'shape') else 'scalar'}")
        self.logger.debug(f"Y coords shape: {y.shape if hasattr(y, 'shape') else 'scalar'}")
        
        ax_opts = config.ax_opts
        
        if 'fill_value' in config.spec_data[field_name]['xyplot']:
            fill_value = config.spec_data[field_name]['xyplot']['fill_value']
            data2d = data2d.where(data2d != fill_value, np.nan)
        
        cmap = ax_opts.get('use_cmap', 'viridis')
        
        title = field_name
        if 'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        units = "n.a."
        if 'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        elif hasattr(data2d, 'attrs') and 'units' in data2d.attrs:
            units = data2d.attrs['units']
        elif hasattr(data2d, 'units'):
            units = data2d.units
        
        try:
            x_dim = config.get_model_dim_name('xc')
            y_dim = config.get_model_dim_name('yc')
            
            plot = data2d.hvplot(
                x=x_dim,
                y=y_dim,
                cmap=cmap,
                title=title,
                width=800,
                height=500,
                colorbar=True,
                clabel=units,
                tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'hover']
            )
            
            self.logger.debug("Successfully created hvplot")
            self.plot_object = plot
            
            return plot
            
        except Exception as e:
            self.logger.error(f"Error creating hvplot: {e}")
            
            try:
                self.logger.info("Trying alternative approach with HoloViews")
                
                if hasattr(data2d, 'values'):
                    z_values = data2d.values
                else:
                    z_values = np.array(data2d)
                
                x_values = x.values if hasattr(x, 'values') else np.array(x)
                y_values = y.values if hasattr(y, 'values') else np.array(y)
                
                image = hv.Image((x_values, y_values, z_values), 
                                kdims=[x_dim, y_dim], 
                                vdims=[field_name])
                
                plot = image.opts(
                    cmap=cmap,
                    colorbar=True,
                    title=title,
                    width=800,
                    height=500,
                    tools=['hover'],
                    xlabel=x_dim,
                    ylabel=y_dim,
                    clabel=units
                )
                
                self.plot_object = plot
                return plot
                
            except Exception as e2:
                self.logger.error(f"Alternative approach also failed: {e2}")
                return None
    
    def save(self, filename, **kwargs):
        """Save the plot to an HTML file."""
        if self.plot_object is not None:
            try:
                # Ensure filename has .html extension
                if not filename.endswith('.html'):
                    filename += '.html'
                
                # Save using holoviews
                hv.save(self.plot_object, filename)
                self.logger.info(f"Saved interactive plot to {filename}")
            except Exception as e:
                self.logger.error(f"Error saving plot: {e}")
        else:
            self.logger.warning("No plot to save")

    
    def show(self):
        """Display the plot."""
        if self.plot_object is not None:
            try:
                # Try to display in notebook
                from IPython.display import display
                display(self.plot_object)
            except ImportError:
                # If not in a notebook, save to a temporary file and open in browser
                import tempfile
                import webbrowser
                import os
                
                temp_file = os.path.join(tempfile.gettempdir(), 'eviz_hvplot.html')
                try:
                    hv.save(self.plot_object, temp_file)
                    webbrowser.open(f"file://{temp_file}")
                    self.logger.info(f"Opening plot in browser: {temp_file}")
                except Exception as e:
                    self.logger.error(f"Error saving temporary file: {e}")
        else:
            self.logger.warning("No plot to show")
