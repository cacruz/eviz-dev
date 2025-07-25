import pandas as pd
import logging
import holoviews as hv
import hvplot.xarray  # register the hvplot method with xarray objects
import hvplot.pandas  # noqa
from eviz.lib.autoviz.plotting.base import BoxPlotter


class HvplotBoxPlotter(BoxPlotter):
    """HvPlot implementation of Box plotting."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            hv.extension('bokeh')
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
        """
        data, _, _, field_name, plot_type, findex, _ = data_to_plot
        
        if data is None:
            self.logger.warning("No data to plot")
            return None
        
        if isinstance(data, pd.DataFrame):
            df = data
            self.logger.debug(f"Using provided DataFrame with {len(df)} rows")
        else:
            self.logger.warning("Expected DataFrame for box plot, got something else")
            return None
        
        title = field_name
        if hasattr(config, 'spec_data') and \
                field_name in config.spec_data \
                and 'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        units = "n.a."
        if hasattr(config, 'spec_data') and \
                field_name in config.spec_data and \
                'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        elif hasattr(data, 'attrs') and 'units' in data.attrs:
            units = data.attrs['units']
        elif hasattr(data, 'units'):
            units = data.units
        
        try:
            category_col = None
            if 'time' in df.columns:
                category_col = 'time'
            elif 'category' in df.columns:
                category_col = 'category'
            else:
                for col in df.columns:
                    if col != 'value':
                        category_col = col
                        break
            
            if category_col is None:
                self.logger.warning("No category column found for box plot")
                return None
            
            plot = df.hvplot.box(
                y='value',
                by=category_col,
                title=title,
                ylabel=f"{title} ({units})",
                width=400,
                height=400,
                tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'hover'],
                legend='top_left'
            )
            
            self.logger.debug("Successfully created hvplot box plot")
            self.plot_object = plot
            
            return plot
            
        except Exception as e:
            self.logger.error(f"Error creating hvplot box plot: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def show(self):
        """Display the plot."""
        pass

    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        pass
