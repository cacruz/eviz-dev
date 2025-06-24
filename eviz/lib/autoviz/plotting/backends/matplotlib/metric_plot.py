import numpy as np
import logging
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from eviz.lib.autoviz.plotting.base import XYPlotter


class MatplotlibMetricPlotter(XYPlotter):
    """Matplotlib implementation for metric visualization (e.g., correlation maps)."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compute_correlation_map(self, data1, data2):
        """Compute pixel-wise Pearson correlation coefficient between two datasets."""
        self.logger.debug("Computing correlation map")
        
        if not isinstance(data1, xr.DataArray) or not isinstance(data2, xr.DataArray):
            self.logger.error("Both inputs must be xarray DataArrays")
            return None
            
        # Regrid?
        if data1.shape != data2.shape:
            self.logger.warning(f"Data shapes don't match: {data1.shape} vs {data2.shape}, correlation may not be meaningful")
        
        dims1 = list(data1.dims)
        
        # Handle 3D data (time, lat, lon)
        if len(dims1) == 3:
            self.logger.debug(f"Processing 3D data with dimensions {dims1}")
            
            # Time dimension is usually the first dimension
            time_dim = dims1[0]            
            spatial_dims = dims1[1:]
            
            # Create output array with same spatial coordinates as data1
            # but without the time dimension
            template = data1.isel({time_dim: 0}).copy()
            corr_data = xr.zeros_like(template)
            
            time_len = data1.shape[0]
            y_len, x_len = data1.shape[1], data1.shape[2]
            
            self.logger.debug(f"Computing correlation across {time_len} time points for a {y_len}x{x_len} grid")
            
            # Compute correlation coefficient for each grid point
            for i in range(y_len):
                for j in range(x_len):
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
                        self.logger.debug(f"Error computing correlation at point ({i},{j}): {e}")
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
            self.logger.warning("Input data is 2D, computing spatial correlation instead of pixel-wise temporal correlation")
            
            flat1 = data1.values.flatten()
            flat2 = data2.values.flatten()
            
            mask = ~np.isnan(flat1) & ~np.isnan(flat2)
            if np.sum(mask) < 2:
                self.logger.error("Not enough valid data points for correlation")
                return None
                
            try:
                r, _ = pearsonr(flat1[mask], flat2[mask])
                # Fill the entire map with this value
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
    
    def plot(self, config, data_to_plot):
        """Create a correlation map using Matplotlib.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
                If data2d is a tuple of two DataArrays, compute correlation between them
                Otherwise, assume data2d is already a correlation map
        
        Returns:
            The created Matplotlib figure and axes
        """
        if isinstance(data_to_plot[0], tuple) and len(data_to_plot[0]) == 2:
            # We have two datasets to correlate
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
            # Assume data2d is already a correlation map
            data2d = data_to_plot[0]
         
        if data2d is None:
            self.logger.warning("No data to plot")
            return None
        
        self.logger.debug(f"Data shape: {data2d.shape}")
        
        ax_opts = config.ax_opts
        
        cmap_name = ax_opts.get('use_cmap', 'RdBu_r')
        
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
            
            fig = data_to_plot[6] if data_to_plot[6] is not None else plt.figure(figsize=(10, 8))
            
            # Check if we should use a map projection
            use_map = ax_opts.get('use_map', True)
            
            if use_map and -180 <= np.min(x_coords) <= 180 and -90 <= np.min(y_coords) <= 90:
                try:                    
                    proj = ax_opts.get('projection', 'PlateCarree')
                    if isinstance(proj, str):
                        if hasattr(ccrs, proj):
                            projection = getattr(ccrs, proj)()
                        else:
                            self.logger.warning(f"Unknown projection: {proj}, using PlateCarree")
                            projection = ccrs.PlateCarree()
                    else:
                        projection = ccrs.PlateCarree()
                    
                    ax = fig.add_subplot(111, projection=projection)
                    
                    ax.coastlines(resolution='50m', linewidth=0.5)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                    ax.add_feature(cfeature.STATES, linewidth=0.3)
                    
                    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                     linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    
                    if 'extent' in ax_opts:
                        ax.set_extent(ax_opts['extent'], crs=ccrs.PlateCarree())
                    
                except ImportError:
                    self.logger.warning("Cartopy not available, using regular axes")
                    ax = fig.add_subplot(111)
            else:
                # Use regular axes
                ax = fig.add_subplot(111)
            
            if len(x_coords.shape) == 1 and len(y_coords.shape) == 1:
                # Regular grid
                X, Y = np.meshgrid(x_coords, y_coords)
                im = ax.pcolormesh(X, Y, data2d.values, 
                                  cmap=cmap_name, 
                                  vmin=-1, vmax=1,
                                  shading='auto')
            else:
                # Irregular grid - use contourf instead
                im = ax.contourf(x_coords, y_coords, data2d.values,
                                levels=np.linspace(-1, 1, 21),
                                cmap=cmap_name,
                                extend='both')
            
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label(units)
            
            ax.set_title(title)
            ax.set_xlabel(x_dim)
            ax.set_ylabel(y_dim)
            
            self.plot_object = (fig, ax)
            
            return (fig, ax)
            
        except Exception as e:
            self.logger.error(f"Error creating correlation map with Matplotlib: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        if self.plot_object is not None:
            try:
                fig, ax = self.plot_object
                
                # Get DPI from kwargs or use default
                dpi = kwargs.get('dpi', 300)
                
                # Save figure
                fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                self.logger.info(f"Saved correlation map to {filename}")
            except Exception as e:
                self.logger.error(f"Error saving plot: {e}")
        else:
            self.logger.warning("No plot to save")

    def show(self):
        """Display the plot."""
        if self.plot_object is not None:
            try:
                fig, ax = self.plot_object
                plt.show()
            except Exception as e:
                self.logger.error(f"Error showing plot: {e}")
        else:
            self.logger.warning("No plot to show")
