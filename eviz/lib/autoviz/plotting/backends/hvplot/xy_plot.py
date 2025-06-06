import numpy as np
import xarray
import logging
import hvplot.xarray  # This registers hvplot with xarray
import holoviews as hv
from ....plotting.base import XYPlotter

class HvplotXYPlotter(XYPlotter):
    """HvPlot implementation of XY plotting."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
        # Set up HoloViews extension and default renderer
        try:
            hv.extension('bokeh')
        except Exception as e:
            self.logger.warning(f"Could not initialize HoloViews extension: {e}")
        
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
        
        # Handle fill values
        if 'fill_value' in config.spec_data[field_name]['xyplot']:
            fill_value = config.spec_data[field_name]['xyplot']['fill_value']
            data2d = data2d.where(data2d != fill_value, np.nan)
        
        # Get colormap
        cmap = ax_opts.get('use_cmap', 'viridis')
        
        # Get title
        title = field_name
        if 'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        # Get units
        units = "n.a."
        if 'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        elif hasattr(data2d, 'attrs') and 'units' in data2d.attrs:
            units = data2d.attrs['units']
        elif hasattr(data2d, 'units'):
            units = data2d.units
        
        try:
            # Get dimension names
            x_dim = x.name if hasattr(x, 'name') else 'x'
            y_dim = y.name if hasattr(y, 'name') else 'y'
            
            self.logger.debug(f"X dimension: {x_dim}")
            self.logger.debug(f"Y dimension: {y_dim}")
            
            # Use the simpler hvplot approach that we know works
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
            
            # Try using HoloViews directly as a fallback
            try:
                self.logger.debug("Trying alternative approach with HoloViews")
                
                # Convert xarray DataArray to numpy arrays
                if hasattr(data2d, 'values'):
                    z_values = data2d.values
                else:
                    z_values = np.array(data2d)
                
                x_values = x.values if hasattr(x, 'values') else np.array(x)
                y_values = y.values if hasattr(y, 'values') else np.array(y)
                
                # Create a HoloViews Image object directly
                image = hv.Image((x_values, y_values, z_values), 
                                kdims=[x_dim, y_dim], 
                                vdims=[field_name])
                
                # Apply styling options
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
