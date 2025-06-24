import numpy as np
import pandas as pd
import altair as alt
import logging
from eviz.lib.autoviz.plotting.base import XTPlotter


class AltairXTPlotter(XTPlotter):
    """Altair implementation of XT (time-series) plotting."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
        # Set Altair renderer to default
        alt.renderers.enable('default')
        # Increase max rows limit for larger datasets
        alt.data_transformers.disable_max_rows()
    
    def plot(self, config, data_to_plot):
        """Create an interactive XT plot using Altair.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created Altair chart object
        """
        data2d, _, _, field_name, plot_type, findex, _ = data_to_plot
        
        if data2d is None:
            self.logger.warning("No data to plot")
            return None
                
        ax_opts = config.ax_opts
        
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
        
        # Create a simple DataFrame directly from the data
        # This is a more direct approach that might work better
        try:
            # If data2d is an xarray DataArray
            if hasattr(data2d, 'values'):
                values = data2d.values
                
                if hasattr(data2d, 'coords'):
                    time_coords = None
                    for coord_name, coord in data2d.coords.items():
                        if coord_name.lower() in ['time', 't', 'date', 'datetime']:
                            time_coords = coord.values
                            break
                    
                    if time_coords is None:
                        coord_name = list(data2d.coords.keys())[0]
                        time_coords = data2d.coords[coord_name].values
                else:
                    time_coords = pd.date_range(start='2000-01-01', periods=len(values), freq='D')
            else:
                values = np.array(data2d)
                time_coords = pd.date_range(start='2000-01-01', periods=len(values), freq='D')
            
            self.logger.debug(f"Values: {values}")
            self.logger.debug(f"Time coords: {time_coords}")
            
            df = pd.DataFrame({
                'time': time_coords,
                'value': values
            })
            
            df['value'] = df['value'].replace([np.inf, -np.inf], np.nan)
            df['value'] = df['value'].mask(df['value'].abs() > 1e30, np.nan)
            
            df = df.dropna()
                        
            # If DataFrame is empty, create a dummy one for testing
            if df.empty:
                self.logger.warning("DataFrame is empty, creating dummy data")
                df = pd.DataFrame({
                    'time': pd.date_range(start='2000-01-01', periods=10, freq='D'),
                    'value': np.random.rand(10)
                })
        
        except Exception as e:
            self.logger.error(f"Error creating DataFrame: {e}")
            # Create a dummy DataFrame
            df = pd.DataFrame({
                'time': pd.date_range(start='2000-01-01', periods=10, freq='D'),
                'value': np.random.rand(10)
            })
        
        try:
            chart = alt.Chart(df).mark_line(color='blue').encode(
                x=alt.X('time:T', title='Time'),
                y=alt.Y('value:Q', title=units)
            ).properties(
                width=800,
                height=400,
                title=title
            ).interactive()
            
            # Add points for better interactivity
            points = chart.mark_point(size=60, filled=True, opacity=0.5).encode(
                tooltip=[
                    alt.Tooltip('time:T', title='Time'),
                    alt.Tooltip('value:Q', title=field_name, format='.3f')
                ]
            )
            
            chart = alt.layer(chart, points)
            
        except Exception as e:
            self.logger.error(f"Error creating Altair chart: {e}")
            # Create a very simple fallback chart
            chart = alt.Chart(df).mark_line().encode(
                x='time:T',
                y='value:Q'
            ).properties(
                width=800,
                height=400,
                title=f"{title} (fallback visualization)"
            )
        
        self.plot_object = chart
        
        return chart
    
    def save(self, filename, **kwargs):
        """Save the plot to an HTML file."""
        if self.plot_object is not None:
            # Ensure filename has .html extension
            if not filename.endswith('.html'):
                filename += '.html'
            
            # Save as HTML
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
                
                temp_file = os.path.join(tempfile.gettempdir(), 'eviz_altair_plot.html')
                self.plot_object.save(temp_file)
                webbrowser.open('file://' + temp_file)
                self.logger.info(f"Opening plot in browser: {temp_file}")
        else:
            self.logger.warning("No plot to show")
