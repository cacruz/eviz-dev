import numpy as np
import logging
import holoviews as hv
import pandas as pd
from ....plotting.base import BasePlotter


class HvplotLinePlotter(BasePlotter):
    """HvPlot implementation of Line plotting."""
    
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
        """Create an interactive Line plot using HvPlot.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created HvPlot object
        """
        data, x, y, field_name, plot_type, findex, _ = data_to_plot
         
        if data is None:
            self.logger.warning("No data to plot")
            return None
        
        self.logger.debug(f"Data shape: {data.shape if hasattr(data, 'shape') else 'scalar'}")
        
        ax_opts = config.ax_opts
        
        # Handle fill values if specified
        if 'lineplot' in config.spec_data[field_name] and 'fill_value' in config.spec_data[field_name]['lineplot']:
            fill_value = config.spec_data[field_name]['lineplot']['fill_value']
            data = data.where(data != fill_value, np.nan)
        
        title = field_name
        if 'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        units = "n.a."
        if 'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        elif hasattr(data, 'attrs') and 'units' in data.attrs:
            units = data.attrs['units']
        elif hasattr(data, 'units'):
            units = data.units
        
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif hasattr(data, 'to_dataframe'):
                df = data.to_dataframe().reset_index()
            else:
                if x is not None and y is not None:
                    # If x and y are provided directly
                    df = pd.DataFrame({'x': x, 'y': y})
                elif hasattr(data, 'dims') and len(data.dims) >= 1:
                    # For 1D xarray DataArrays
                    x_dim = data.dims[0]
                    df = pd.DataFrame({
                        'x': data[x_dim].values,
                        'y': data.values
                    })
                else:
                    # Simple arrays
                    df = pd.DataFrame({
                        'x': np.arange(len(data)),
                        'y': data
                    })
            
            self.logger.debug(f"DataFrame shape: {df.shape}")
            self.logger.debug(f"DataFrame columns: {df.columns}")
            
            x_col = 'x' if 'x' in df.columns else df.columns[0]
            y_col = 'y' if 'y' in df.columns else df.columns[1]
            
            plot = df.hvplot.line(
                x=x_col,
                y=y_col,
                title=title,
                xlabel=x_col,
                ylabel=f"{title} ({units})",
                width=800,
                height=500,
                tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'hover'],
                legend='top'
            )
            
            if ax_opts.get('markers', False):
                scatter = df.hvplot.scatter(
                    x=x_col,
                    y=y_col,
                    color='red',
                    size=50
                )
                plot = plot * scatter
            
            self.logger.debug("Successfully created hvplot line plot")
            self.plot_object = plot
            
            return plot
            
        except Exception as e:
            self.logger.error(f"Error creating hvplot line plot: {e}")
            
            try:
                self.logger.debug("Trying alternative approach with HoloViews")
                
                # Convert to DataFrame if needed
                if not isinstance(data, pd.DataFrame):
                    if x is not None and y is not None:
                        df = pd.DataFrame({'x': x, 'y': y})
                    else:
                        df = pd.DataFrame({
                            'x': np.arange(len(data)),
                            'y': data if hasattr(data, '__len__') else [data]
                        })
                else:
                    df = data
                
                # Get x and y column names
                x_col = 'x' if 'x' in df.columns else df.columns[0]
                y_col = 'y' if 'y' in df.columns else df.columns[1]
                
                curve = hv.Curve(df, x_col, y_col)
                
                plot = curve.opts(
                    title=title,
                    width=800,
                    height=500,
                    tools=['hover'],
                    xlabel=x_col,
                    ylabel=f"{title} ({units})"
                )
                
                if ax_opts.get('markers', False):
                    scatter = hv.Scatter(df, x_col, y_col).opts(color='red', size=8)
                    plot = plot * scatter
                
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
                
                temp_file = os.path.join(tempfile.gettempdir(), 'eviz_hvplot_line.html')
                try:
                    hv.save(self.plot_object, temp_file)
                    webbrowser.open(f"file://{temp_file}")
                    self.logger.info(f"Opening plot in browser: {temp_file}")
                except Exception as e:
                    self.logger.error(f"Error saving temporary file: {e}")
        else:
            self.logger.warning("No plot to show")