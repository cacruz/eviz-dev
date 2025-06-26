import numpy as np
import pandas as pd
import logging
import holoviews as hv
from eviz.lib.autoviz.plotting.base import ScatterPlotter


class HvplotScatterPlotter(ScatterPlotter):
    """HvPlot implementation of scatter plotting."""

    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            hv.extension('bokeh')
        except Exception as e:
            self.logger.warning(f"Could not initialize HoloViews extension: {e}")
    
    def plot(self, config, data_to_plot):
        """Create an interactive scatter plot using HvPlot.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (x_data, y_data, z_data, field_name, plot_type, findex, fig)
                where z_data is optional and can be used for coloring points
        
        Returns:
            The created HvPlot object
        """
        x_data, y_data, z_data, field_name, plot_type, findex, _ = data_to_plot
        
        if x_data is None or y_data is None:
            return None

        ax_opts = config.ax_opts
        
        title = field_name
        if 'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        units = "n.a."
        if 'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        elif z_data is not None:
            if hasattr(z_data, 'attrs') and 'units' in z_data.attrs:
                units = z_data.attrs['units']
            elif hasattr(z_data, 'units'):
                units = z_data.units
        
        x_label = ax_opts.get('xlabel', x_data.name if hasattr(x_data, 'name') else 'X')
        y_label = ax_opts.get('ylabel', y_data.name if hasattr(y_data, 'name') else 'Y')
        
        cmap = ax_opts.get('use_cmap', 'viridis')
        
        marker_size = ax_opts.get('marker_size', 8)
        
        try:
            df = self._convert_to_dataframe(x_data, y_data, z_data)
            
            if z_data is not None:
                plot = df.hvplot.scatter(
                    x='x',
                    y='y',
                    c='z',
                    cmap=cmap,
                    title=title,
                    xlabel=x_label,
                    ylabel=y_label,
                    clabel=units,
                    colorbar=True,
                    size=marker_size,
                    width=800,
                    height=500,
                    tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'hover']
                )
            else:
                # Simple scatter plot
                plot = df.hvplot.scatter(
                    x='x',
                    y='y',
                    title=title,
                    xlabel=x_label,
                    ylabel=y_label,
                    size=marker_size,
                    width=800,
                    height=500,
                    tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'hover']
                )
            
            if ax_opts.get('add_regression', False):
                try:
                    from scipy import stats
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x'], df['y'])
                    line_x = np.array([df['x'].min(), df['x'].max()])
                    line_y = slope * line_x + intercept
                    
                    line_df = pd.DataFrame({'x': line_x, 'y': line_y})
                    line_plot = line_df.hvplot.line(
                        x='x',
                        y='y',
                        color='red',
                        line_width=2
                    )
                    
                    r_squared = r_value**2
                    text = hv.Text(df['x'].min() + 0.1 * (df['x'].max() - df['x'].min()),
                                  df['y'].max() - 0.1 * (df['y'].max() - df['y'].min()),
                                  f'R² = {r_squared:.3f}')
                    
                    plot = plot * line_plot * text
                    
                except Exception as e:
                    self.logger.warning(f"Error adding regression line: {e}")
            
            self.logger.debug("Successfully created hvplot")
            self.plot_object = plot
            
            return plot
            
        except Exception as e:
            self.logger.error(f"Error creating hvplot: {e}")
            
            try:
                self.logger.debug("Trying alternative approach with HoloViews")
                
                x_values = x_data.values if hasattr(x_data, 'values') else np.array(x_data)
                y_values = y_data.values if hasattr(y_data, 'values') else np.array(y_data)
                
                if z_data is not None:
                    z_values = z_data.values if hasattr(z_data, 'values') else np.array(z_data)
                    
                    scatter = hv.Scatter((x_values, y_values, z_values), 
                                        kdims=[x_label, y_label], 
                                        vdims=['Color'])
                    
                    plot = scatter.opts(
                        color='Color',
                        cmap=cmap,
                        colorbar=True,
                        title=title,
                        width=800,
                        height=500,
                        size=marker_size,
                        tools=['hover']
                    )
                else:
                    # Create simple scatter
                    scatter = hv.Scatter((x_values, y_values), 
                                        kdims=[x_label, y_label])
                    
                    plot = scatter.opts(
                        title=title,
                        width=800,
                        height=500,
                        size=marker_size,
                        tools=['hover']
                    )
                
                self.plot_object = plot
                return plot
                
            except Exception as e2:
                self.logger.error(f"Alternative approach also failed: {e2}")
                return None
    
    def _convert_to_dataframe(self, x_data, y_data, z_data=None):
        """Convert data to pandas DataFrame.
        
        Args:
            x_data: X-coordinate values
            y_data: Y-coordinate values
            z_data: Optional Z values for coloring points
            
        Returns:
            pandas DataFrame with columns 'x', 'y', and optionally 'z'
        """
        try:
            x_values = x_data.values if hasattr(x_data, 'values') else np.array(x_data)
            y_values = y_data.values if hasattr(y_data, 'values') else np.array(y_data)
            
            if z_data is not None:
                z_values = z_data.values if hasattr(z_data, 'values') else np.array(z_data)
                
                min_len = min(len(x_values), len(y_values), len(z_values))
                df = pd.DataFrame({
                    'x': x_values[:min_len],
                    'y': y_values[:min_len],
                    'z': z_values[:min_len]
                })
            else:
                min_len = min(len(x_values), len(y_values))
                df = pd.DataFrame({
                    'x': x_values[:min_len],
                    'y': y_values[:min_len]
                })
            
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {e}")
            if z_data is not None:
                return pd.DataFrame({
                    'x': [0, 1, 2],
                    'y': [0, 1, 2],
                    'z': [0, 1, 2]
                })
            else:
                return pd.DataFrame({
                    'x': [0, 1, 2],
                    'y': [0, 1, 2]
                })
    
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
                from IPython.display import display
                display(self.plot_object)
            except ImportError:
                # If not in a notebook, save to a temporary file and open in browser
                import tempfile
                import webbrowser
                import os
                
                temp_file = os.path.join(tempfile.gettempdir(), 'eviz_hvplot_scatter.html')
                try:
                    hv.save(self.plot_object, temp_file)
                    webbrowser.open(f"file://{temp_file}")
                    self.logger.info(f"Opening plot in browser: {temp_file}")
                except Exception as e:
                    self.logger.error(f"Error saving temporary file: {e}")
        else:
            self.logger.warning("No plot to show")
