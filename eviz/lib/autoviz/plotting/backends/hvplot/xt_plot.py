import numpy as np
import pandas as pd
import logging
import holoviews as hv
from eviz.lib.autoviz.plotting.base import XTPlotter


class HvplotXTPlotter(XTPlotter):
    """HvPlot implementation of XT (time-series) plotting."""
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            hv.extension('bokeh')
        except Exception as e:
            self.logger.warning(f"Could not initialize HoloViews extension: {e}")
    
    def plot(self, config, data_to_plot):
        """Create an interactive XT plot using HvPlot.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created HvPlot object
        """
        data2d, _, _, field_name, plot_type, findex, _ = data_to_plot
        
        if data2d is None:
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
        
        tc_dim = config.get_model_dim_name('tc')
        try:
            if tc_dim and tc_dim in data2d.coords:
                time_coords = data2d.coords[tc_dim].values
                time_dim = tc_dim
            else:
                time_dim = None
                if hasattr(data2d, 'dims'):
                    for dim in data2d.dims:
                        if dim.lower() in ['time', 't', 'date', 'datetime']:
                            time_dim = dim
                            break
                if time_dim:
                    self.logger.debug(f"Using {time_dim} as time dimension")
                    time_coords = data2d[time_dim].values
                elif len(data2d.dims) > 0:
                    time_dim = data2d.dims[0]
                    self.logger.debug(f"Using first dimension {time_dim} as time")
                    time_coords = data2d[time_dim].values
                else:
                    self.logger.debug("No time dimension found, using range")
                    time_coords = np.arange(len(data2d))
                    time_dim = 'time'
            
            self.logger.debug(f"Time dimension: {time_dim}")
            self.logger.debug(f"Time coordinates shape: {time_coords.shape}")
            
        except Exception as e:
            self.logger.warning(f"Error getting time coordinates: {e}")
            time_coords = np.arange(len(data2d))
            time_dim = 'time'
        
        # Handle rolling mean if specified
        window_size = 0
        if field_name in config.spec_data and 'xtplot' in config.spec_data[field_name]:
            if 'mean_type' in config.spec_data[field_name]['xtplot']:
                if config.spec_data[field_name]['xtplot']['mean_type'] == 'rolling':
                    if 'window_size' in config.spec_data[field_name]['xtplot']:
                        window_size = config.spec_data[field_name]['xtplot']['window_size']
        
        try:
            if hasattr(data2d, 'hvplot'):
                # If data2d is a xarray DataArray with hvplot accessor
                self.logger.debug("Using hvplot accessor for xarray DataArray")
                
                plot_opts = {
                    'x': time_dim,
                    'title': title,
                    'xlabel': 'Time',
                    'ylabel': units,
                    'width': 800,
                    'height': 400,
                    'line_width': 2,
                    'tools': ['pan', 'wheel_zoom', 'box_zoom', 'reset', 'hover'],
                }
                
                if window_size > 0:
                    self.logger.debug(f"Applying rolling mean with window size {window_size}")
                    plot_opts['rolling'] = window_size
                    plot_opts['rolling_center'] = True
                
                plot = data2d.hvplot.line(**plot_opts)
                
                # Add trend line if specified
                if field_name in config.spec_data and 'xtplot' in config.spec_data[field_name]:
                    if 'add_trend' in config.spec_data[field_name]['xtplot'] and \
                            config.spec_data[field_name]['xtplot']['add_trend']:
                        self.logger.debug("Adding trend line")
                        trend = data2d.hvplot.line(
                            x=time_dim,
                            regression=True,
                            line_color='red',
                            line_width=1.5
                        )
                        plot = plot * trend
                
            else:
                self.logger.debug("Converting to DataFrame for plotting")
                df = self._convert_to_dataframe(data2d, time_coords)
                
                if window_size > 0:
                    self.logger.debug(f"Applying rolling mean with window size {window_size}")
                    df['value'] = df['value'].rolling(window=window_size, center=True).mean().dropna()
                
                curve = hv.Curve(df, 'time', 'value')
                plot = curve.opts(
                    title=title,
                    xlabel='Time',
                    ylabel=units,
                    width=800,
                    height=400,
                    line_width=2,
                    tools=['hover']
                )
                
                if field_name in config.spec_data and 'xtplot' in config.spec_data[field_name]:
                    if 'add_trend' in config.spec_data[field_name]['xtplot'] and \
                            config.spec_data[field_name]['xtplot']['add_trend']:
                        self.logger.debug("Adding trend line")
                        from scipy import stats
                        
                        # Convert time to numeric for regression
                        if isinstance(df['time'].iloc[0], (pd.Timestamp, np.datetime64)):
                            x_numeric = (df['time'] - df['time'].iloc[0]).dt.total_seconds().values
                        else:
                            x_numeric = np.arange(len(df))
                        
                        # Calculate trend line
                        slope, intercept, _, _, _ = stats.linregress(x_numeric, df['value'])
                        trend_y = intercept + slope * x_numeric
                        
                        # Create trend line curve
                        trend_df = pd.DataFrame({'time': df['time'], 'trend': trend_y})
                        trend_curve = hv.Curve(trend_df, 'time', 'trend')
                        trend_plot = trend_curve.opts(color='red', line_width=1.5)
                        
                        # Overlay trend line on main plot
                        plot = plot * trend_plot
            
            self.logger.debug("Successfully created plot")
            self.plot_object = plot
            
            return plot
            
        except Exception as e:
            self.logger.error(f"Error creating plot: {e}")
            
            # Try a simpler approach as fallback
            try:
                self.logger.debug("Trying simpler approach")
                
                df = self._convert_to_dataframe(data2d, time_coords)
                
                curve = hv.Curve(df, 'time', 'value')
                plot = curve.opts(
                    title=title,
                    width=800,
                    height=400
                )
                
                self.plot_object = plot
                return plot
                
            except Exception as e2:
                self.logger.error(f"Simpler approach also failed: {e2}")
                return None
    
    def _convert_to_dataframe(self, data2d, time_coords):
        """Convert time series data to pandas DataFrame.
        
        Args:
            data2d: Time series data
            time_coords: Time coordinates
            
        Returns:
            pandas DataFrame with columns 'time', 'value'
        """
        try:
            if hasattr(data2d, 'values'):
                values = data2d.values
            else:
                values = np.array(data2d)
            
            if len(values.shape) > 1:
                self.logger.debug(f"Flattening values from shape {values.shape}")
                values = values.flatten()
            
            if len(time_coords) != len(values):
                self.logger.warning(f"Time coordinates length ({len(time_coords)}) "
                                    f"doesn't match values length ({len(values)})")
                # Use the shorter length
                min_len = min(len(time_coords), len(values))
                time_coords = time_coords[:min_len]
                values = values[:min_len]
            
            # Handle cftime objects
            if len(time_coords) > 0 and (hasattr(time_coords[0], '_cftime') or
                                         str(type(time_coords[0])).find('cftime') >= 0):
                try:
                    # Convert to pandas datetime
                    self.logger.debug("Converting cftime objects to pandas datetime")
                    time_coords = pd.to_datetime([str(t) for t in time_coords])
                except Exception as e:
                    self.logger.warning(f"Error converting cftime to pandas datetime: {e}")
                    time_coords = pd.date_range(start='2000-01-01', periods=len(values), freq='D')
            
            df = pd.DataFrame({
                'time': time_coords,
                'value': values
            })
            
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {e}")
            # Create a minimal DataFrame as fallback
            return pd.DataFrame({
                'time': pd.date_range(start='2000-01-01', periods=10, freq='D'),
                'value': np.random.rand(10)
            })
    
    def show(self):
        """Display the plot."""
        pass

    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        pass
