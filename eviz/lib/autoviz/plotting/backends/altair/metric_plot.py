import numpy as np
import logging
import xarray as xr
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import altair as alt
from eviz.lib.autoviz.plotting.base import XYPlotter


class AltairMetricPlotter(XYPlotter):
    """Altair implementation for metric visualization (e.g., correlation maps)."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
            alt.renderers.enable('default')
            # Set a reasonable max rows limit for Altair
            alt.data_transformers.disable_max_rows()
            self.logger.debug("Successfully initialized Altair")
        except Exception as e:
            self.logger.warning(f"Could not initialize Altair: {e}")
    
    def compute_correlation_map(self, data1: xr.DataArray, data2: xr.DataArray) -> xr.DataArray:
        """Compute pixel-wise Pearson correlation coefficient between two datasets."""
        self.logger.debug("Computing correlation map")
        
        if not isinstance(data1, xr.DataArray) or not isinstance(data2, xr.DataArray):
            self.logger.error("Both inputs must be xarray DataArrays")
            return None
            
        if data1.shape != data2.shape:
            self.logger.warning(f"Data shapes don't match: {data1.shape} vs {data2.shape}, "
                                f"correlation may not be meaningful")
        
        dims1 = list(data1.dims)
        
        # Handle 3D data (time, lat, lon)
        if len(dims1) == 3:
            self.logger.debug(f"Processing 3D data with dimensions {dims1}")            
            time_dim = dims1[0]            
            spatial_dims = dims1[1:]
            
            # Create output array with same spatial coordinates as data1
            # but without the time dimension
            template = data1.isel({time_dim: 0}).copy()
            corr_data = xr.zeros_like(template)
            
            time_len = data1.shape[0]
            y_len, x_len = data1.shape[1], data1.shape[2]
            
            self.logger.debug(f"Computing correlation across {time_len} time points for a "
                              f"{y_len}x{x_len} grid")
            
            # Compute correlation coefficient for each grid point
            for i in range(y_len):
                for j in range(x_len):
                    # Extract time series at this grid point
                    ts1 = data1[:, i, j].values
                    ts2 = data2[:, i, j].values
                    
                    if np.isnan(ts1).all() or np.isnan(ts2).all():
                        corr_data[i, j] = np.nan
                        continue
                        
                    mask = ~np.isnan(ts1) & ~np.isnan(ts2)
                    if np.sum(mask) < 2:  # Need at least 2 points for correlation
                        corr_data[i, j] = np.nan
                        continue
                        
                    try:
                        r, _ = pearsonr(ts1[mask], ts2[mask])
                        corr_data[i, j] = r
                    except Exception as e:
                        self.logger.debug(f"Error computing correlation at point "
                                          f"({i},{j}): {e}")
                        corr_data[i, j] = np.nan
            
            corr_data.attrs['long_name'] = 'Pearson Correlation Coefficient'
            corr_data.attrs['units'] = 'dimensionless'
            corr_data.attrs['description'] = f'Correlation between {data1.name} and {data2.name} across time'
            
            return corr_data
            
        # Handle 2D data
        elif len(dims1) == 2:
            self.logger.debug(f"Processing 2D data with dimensions {dims1}")
            
            corr_data = xr.zeros_like(data1)
            
            # For 2D data, we can't compute pixel-wise correlation across time
            # Instead, compute spatial correlation or return a warning

            flat1 = data1.values.flatten()
            flat2 = data2.values.flatten()
            
            mask = ~np.isnan(flat1) & ~np.isnan(flat2)
            if np.sum(mask) < 2:
                self.logger.error("Not enough valid data points for correlation")
                return None
                
            try:
                r, _ = pearsonr(flat1[mask], flat2[mask])
                corr_data.values.fill(r)
                
                corr_data.attrs['long_name'] = 'Pearson Correlation Coefficient (Spatial)'
                corr_data.attrs['units'] = 'dimensionless'
                corr_data.attrs['description'] = f'Spatial correlation between {data1.name} and {data2.name}'
                
                return corr_data
            except Exception as e:
                self.logger.error(f"Error computing spatial correlation: {e}")
                return None
        else:
            self.logger.error(f"Unsupported data dimensions: {dims1}")
            return None
    
    def plot(self, config: "ConfigManager", data_to_plot: tuple) -> alt.Chart:
        """Create an interactive correlation map using Altair.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
                If data2d is a tuple of two DataArrays, compute correlation between them
                Otherwise, assume data2d is already a correlation map
        
        Returns:
            The created Altair chart object
        """
        if isinstance(data_to_plot[0], tuple) and len(data_to_plot[0]) == 2:
            data1, data2 = data_to_plot[0]
            self.logger.debug("Computing correlation between two datasets")
            data2d = self.compute_correlation_map(data1, data2)
            if data2d is None:
                self.logger.error("Failed to compute correlation map")
                return None
                
            # Reconstruct data_to_plot with the correlation map
            x, y = data_to_plot[1], data_to_plot[2]
            field_name = data_to_plot[3] + "_correlation"
            plot_type, findex, fig = data_to_plot[4], data_to_plot[5], data_to_plot[6]
            data_to_plot = (data2d, x, y, field_name, plot_type, findex, fig)
        else:
            data2d = data_to_plot[0]
         
        if data2d is None:
            self.logger.warning("No data to plot")
            return None
                
        ax_opts = config.ax_opts
        
        cmap = ax_opts.get('use_cmap', 'RdBu_r')
        
        field_name = data_to_plot[3]
        title = field_name
        if hasattr(config, 'spec_data') and field_name in config.spec_data and 'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        units = "dimensionless"
        
        try:
            x_dim = config.get_model_dim_name('xc') or 'lon'
            y_dim = config.get_model_dim_name('yc') or 'lat'
            
            x_coords = data2d[x_dim].values
            y_coords = data2d[y_dim].values
            
            x_min, x_max = np.nanmin(x_coords), np.nanmax(x_coords)
            y_min, y_max = np.nanmin(y_coords), np.nanmax(y_coords)
            
            self.logger.debug(f"Data extent: lon [{x_min}, {x_max}], lat [{y_min}, {y_max}]")
            
            if 'extent' in config.ax_opts:
                # Use the provided extent
                x_min, x_max, y_min, y_max = config.ax_opts['extent']
                self.logger.debug(f"Using provided extent: lon [{x_min}, {x_max}], lat [{y_min}, {y_max}]")
            else:
                # For continental US, we can set a fixed extent if the data is in that region
                is_us_region = (
                    x_min >= -130 and x_max <= -65 and
                    y_min >= 23 and y_max <= 51
                )
                
                if is_us_region:
                    # Set extent to continental US with a small buffer
                    x_min, x_max = -130, -65
                    y_min, y_max = 23, 51
                    self.logger.debug(f"Setting extent to continental US: "
                                      f"lon [{x_min}, {x_max}], lat [{y_min}, {y_max}]")
            
            irregular_grid = False
            if len(x_coords.shape) > 1 or len(y_coords.shape) > 1:
                irregular_grid = True

            # For irregular grids, we need to convert to a regular grid
            if irregular_grid:
                # Create a new regular grid with reasonable resolution
                new_x = np.linspace(x_min, x_max, 100)
                new_y = np.linspace(y_min, y_max, 100)
                
                # Create a new DataArray on the regular grid
                new_coords = {x_dim: new_x, y_dim: new_y}
                new_data = xr.DataArray(
                    np.zeros((len(new_y), len(new_x))),
                    coords=new_coords,
                    dims=[y_dim, x_dim]
                )
                
                # Interpolate the data to the regular grid
                try:
                    
                    points = np.column_stack([x_coords.flatten(), y_coords.flatten()])
                    values = data2d.values.flatten()
                    
                    valid = ~np.isnan(values)
                    points = points[valid]
                    values = values[valid]
                    
                    X, Y = np.meshgrid(new_x, new_y)
                    
                    interp_values = griddata(points, values, (X, Y), method='linear')
                    
                    new_data.values = interp_values
                    
                    data2d = new_data
                    self.logger.debug("Successfully interpolated irregular grid to regular grid")
                except Exception as e:
                    self.logger.warning(f"Failed to interpolate to regular grid: {e}")
            
            # Convert to DataFrame for Altair
            df = data2d.to_dataframe(name='correlation').reset_index()
                       
            if x_dim in df.columns:
                df[x_dim] = df[x_dim].astype(float)
            if y_dim in df.columns:
                df[y_dim] = df[y_dim].astype(float)
            df['correlation'] = df['correlation'].astype(float)
            
            # Filter the DataFrame to only include points within our desired extent
            df = df[(df[x_dim] >= x_min) & (df[x_dim] <= x_max) & 
                    (df[y_dim] >= y_min) & (df[y_dim] <= y_max)]
            
            base = alt.Chart(df).encode(
                x=alt.X(f'{x_dim}:Q', 
                    title='Longitude',
                    scale=alt.Scale(domain=[x_min, x_max])),  # Set explicit domain for x
                y=alt.Y(f'{y_dim}:Q', 
                    title='Latitude',
                    scale=alt.Scale(domain=[y_min, y_max]))   # Set explicit domain for y
            )
            
            try:
                square_plot = base.mark_square(size=10).encode(
                    color=alt.Color('correlation:Q', 
                                scale=alt.Scale(domain=[-1, 1], scheme=cmap),
                                title=units)
                )
                chart = square_plot
            except Exception:
                chart = base.mark_rect().encode(
                    color=alt.Color('correlation:Q', 
                                scale=alt.Scale(domain=[-1, 1], scheme=cmap),
                                title=units)
                )
            
            # Set chart properties with appropriate aspect ratio for US map
            # Continental US is roughly 1.7 times wider than tall
            width = 800
            height = int(width / 1.7)  # Approximately correct aspect ratio for US
            
            chart = chart.properties(
                title=title,
                width=width,
                height=height
            ).interactive()
            
            # Add tooltip
            chart = chart.encode(
                tooltip=[
                    alt.Tooltip(f'{x_dim}:Q', title='Longitude', format='.3f'),
                    alt.Tooltip(f'{y_dim}:Q', title='Latitude', format='.3f'),
                    alt.Tooltip('correlation:Q', title='Correlation', format='.3f')
                ]
            )
            
            # Try to add US state boundaries if vega_datasets is available
            try:
                from vega_datasets import data
                states_data = data.us_10m.url
                states = alt.topo_feature(states_data, 'states')
                
                # Create a chart with US state boundaries
                state_borders = alt.Chart(states).mark_geoshape(
                    filled=False,
                    stroke='black',
                    strokeWidth=0.5
                ).project(
                    type='albersUsa'  # Use Albers USA projection which is good for US maps
                )
                
                # We can't directly overlay these because they use different coordinate systems
                # Instead, we'll create a layered chart with both elements
                
                # For now, just return the correlation map
                self.plot_object = chart
                
                return chart
            except Exception as e:
                self.logger.debug(f"Could not add state boundaries: {e}")
                
            self.plot_object = chart
            
            return chart
            
        except Exception as e:
            self.logger.error(f"Error creating correlation map with Altair: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def show(self):
        """Display the plot."""
        pass

    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        pass
