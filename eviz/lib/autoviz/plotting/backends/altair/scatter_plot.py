import numpy as np
import pandas as pd
import altair as alt
import logging
from ....plotting.base import ScatterPlotter


class AltairScatterPlotter(ScatterPlotter):
    """Altair implementation of scatter plotting."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
        # Set Altair renderer to default
        alt.renderers.enable('default')
        # Increase max rows limit for larger datasets
        alt.data_transformers.disable_max_rows()
    
    def plot(self, config, data_to_plot):
        """Create an interactive scatter plot using Altair.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (x_data, y_data, z_data, field_name, plot_type, findex, fig)
                where z_data is optional and can be used for coloring points
        
        Returns:
            The created Altair chart object
        """
        x_data, y_data, z_data, field_name, plot_type, findex, _ = data_to_plot
        
        if x_data is None or y_data is None:
            self.logger.warning("No data to plot")
            return None
        
        self.logger.debug(f"X data shape: {x_data.shape if hasattr(x_data, 'shape') else 'scalar'}")
        self.logger.debug(f"Y data shape: {y_data.shape if hasattr(y_data, 'shape') else 'scalar'}")
        if z_data is not None:
            self.logger.debug(f"Z data shape: {z_data.shape if hasattr(z_data, 'shape') else 'scalar'}")
        
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
        
        df = self._convert_to_dataframe(x_data, y_data, z_data)
        
        self.logger.debug(f"DataFrame shape: {df.shape}")
        self.logger.debug(f"DataFrame columns: {df.columns}")
        
        try:
            x_label = ax_opts.get('xlabel', x_data.name if hasattr(x_data, 'name') else 'X')
            y_label = ax_opts.get('ylabel', y_data.name if hasattr(y_data, 'name') else 'Y')
            
            cmap = ax_opts.get('use_cmap', 'viridis')
            # Map common matplotlib colormaps to Vega schemes
            cmap_mapping = {
                'viridis': 'viridis',
                'plasma': 'plasma',
                'inferno': 'inferno',
                'magma': 'magma',
                'cividis': 'cividis',
                'rainbow': 'rainbow',
                'jet': 'rainbow',  # Approximate
                'Blues': 'blues',
                'Reds': 'reds',
                'Greens': 'greens',
                'YlOrRd': 'yelloworangered',
                'RdBu': 'redblue',
                'coolwarm': 'redblue'
            }
            vega_scheme = cmap_mapping.get(cmap, 'viridis')
            
            marker_size = ax_opts.get('marker_size', 60)
            
            if z_data is not None:
                chart = alt.Chart(df).mark_circle(size=marker_size).encode(
                    x=alt.X('x:Q', title=x_label),
                    y=alt.Y('y:Q', title=y_label),
                    color=alt.Color('z:Q', 
                                   scale=alt.Scale(scheme=vega_scheme),
                                   title=units)
                )
            else:
                chart = alt.Chart(df).mark_circle(size=marker_size).encode(
                    x=alt.X('x:Q', title=x_label),
                    y=alt.Y('y:Q', title=y_label)
                )
            
            # Add tooltips
            if z_data is not None:
                chart = chart.encode(
                    tooltip=[
                        alt.Tooltip('x:Q', title=x_label),
                        alt.Tooltip('y:Q', title=y_label),
                        alt.Tooltip('z:Q', title=field_name)
                    ]
                )
            else:
                chart = chart.encode(
                    tooltip=[
                        alt.Tooltip('x:Q', title=x_label),
                        alt.Tooltip('y:Q', title=y_label)
                    ]
                )
            
            if ax_opts.get('add_regression', False):
                regression = chart.transform_regression('x', 'y').mark_line(color='red')
                chart = alt.layer(chart, regression)
            
            chart = chart.properties(
                width=800,
                height=500,
                title=title
            ).interactive()
            
            if ax_opts.get('add_grid', True):
                chart = chart.configure_axis(
                    grid=True,
                    gridColor='lightgray',
                    gridOpacity=0.5
                )
            
        except Exception as e:
            self.logger.error(f"Error creating Altair chart: {e}")
            # Create a simple fallback chart
            chart = alt.Chart(df).mark_circle().encode(
                x='x:Q',
                y='y:Q'
            ).properties(
                width=800,
                height=500,
                title=f"{title} (fallback visualization)"
            )
        
        self.plot_object = chart
        
        return chart
    
    def _convert_to_dataframe(self, x_data, y_data, z_data=None):
        """Convert data to pandas DataFrame for Altair.
        
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
            if not filename.endswith('.html'):
                filename += '.html'
            
            self.plot_object.save(filename)
            self.logger.info(f"Saved interactive plot to {filename}")
        else:
            self.logger.warning("No plot to save")
    
    def show(self):
        """Display the plot."""
        if self.plot_object is not None:
            # Display the chart
            try:
                from IPython.display import display
                display(self.plot_object)
            except ImportError:
                # If not in a notebook, save to a temporary file and open in browser
                import tempfile
                import webbrowser
                import os
                
                temp_file = os.path.join(tempfile.gettempdir(), 'eviz_altair_scatter.html')
                self.plot_object.save(temp_file)
                webbrowser.open('file://' + temp_file)
                self.logger.info(f"Opening plot in browser: {temp_file}")
        else:
            self.logger.warning("No plot to show")
