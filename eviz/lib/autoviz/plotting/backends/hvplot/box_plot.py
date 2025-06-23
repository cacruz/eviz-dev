import numpy as np
import pandas as pd
import logging
import holoviews as hv
import hvplot.xarray  # register the hvplot method with xarray objects
from ....plotting.base import BoxPlotter


class HvplotBoxPlotter(BoxPlotter):
    """HvPlot implementation of Box plotting."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            hv.extension('bokeh')
            self.logger.debug("Successfully initialized HoloViews and hvplot extensions")
        except Exception as e:
            self.logger.warning(f"Could not initialize HoloViews/hvplot extensions: {e}")   
        
    def plot(self, config, data_to_plot):
        """Create an interactive Box plot per time step using HvPlot.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data, categories, values, field_name,
                          plot_type, findex, fig)
        
        Returns:
            The created HvPlot object

        Notes:

            * Each box summarizes the distribution of data_to_plot values across the
              entire spatial grid (all lat/lon points) at that time.
            * For each datetime, the boxplot shows:
                * Median: Middle value of data over all grid cells
                * Interquartile Range (IQR):
                * Q1 (25th percentile) = lower edge of the box
                * Q3 (75th percentile) = upper edge of the box
                * Whiskers: Range of values within 1.5×IQR of Q1 and Q3
                * Outliers: Any grid points with unusually high or low value at that time

        """
        data, _, _, field_name, plot_type, findex, _ = data_to_plot
         
        if data is None:
            self.logger.warning("No data to plot")
            return None
        
        self.logger.debug(f"Data shape: {data.shape if hasattr(data, 'shape') else 'scalar'}")
        
        ax_opts = config.ax_opts
        
        # Handle fill values if specified
        if 'boxplot' in config.spec_data[field_name] and 'fill_value' in config.spec_data[field_name]['boxplot']:
            fill_value = config.spec_data[field_name]['boxplot']['fill_value']
            data = data.where(data != fill_value, np.nan)
        
        # Get title
        title = field_name
        if 'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        # Get units
        units = "n.a."
        if 'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        elif hasattr(data, 'attrs') and 'units' in data.attrs:
            units = data.attrs['units']
        elif hasattr(data, 'units'):
            units = data.units
        
        try:
            df = data            
            self.logger.debug(f"DataFrame shape: {df.shape}")
            self.logger.debug(f"DataFrame columns: {df.columns}")
            
            plot = df.hvplot.box(
                y='value',
                by='time',
                title=title,
                ylabel=f"{title} ({units})",
                width=800,
                height=500,
                tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'hover'],
                legend='top'
            )
            
            self.logger.debug("Successfully created hvplot box plot")
            self.plot_object = plot
            
            return plot
            
        except Exception as e:
            self.logger.error(f"Error creating hvplot box plot: {e}")
            
    
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
                
                temp_file = os.path.join(tempfile.gettempdir(), 'eviz_hvplot_box.html')
                try:
                    hv.save(self.plot_object, temp_file)
                    webbrowser.open(f"file://{temp_file}")
                    self.logger.info(f"Opening plot in browser: {temp_file}")
                except Exception as e:
                    self.logger.error(f"Error saving temporary file: {e}")
        else:
            self.logger.warning("No plot to show")