import matplotlib as mpl
import numpy as np
import logging
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
from .base import MatplotlibBasePlotter


class MatplotlibMetricPlotter(MatplotlibBasePlotter):
    """Matplotlib implementation for metric visualization (e.g., correlation maps)."""
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)

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
        self.config = config
        field_name = data_to_plot[3]        
        if isinstance(data_to_plot[0], tuple) and len(data_to_plot[0]) == 2:
            # Two datasets to correlate
            data1, data2 = data_to_plot[0]
            
            # Store original data for R² calculation
            self._original_data = (data1, data2)
            
            self.logger.debug("Computing correlation between two datasets")
            # Get corr plot settings from for_inputs
            corr_settings = config.app_data.for_inputs['correlation']
            corr_method = corr_settings['method']
            data2d = self._compute_correlation_map(corr_method, data1, data2)
            if data2d is None:
                self.logger.error("Failed to compute correlation map")
                return None

            method_name = {
                'pearson': 'Pearson',
                'spearman': 'Spearman',
                'cross': 'Cross'
            }.get(corr_method, 'Correlation')
                
            # Reconstruct data_to_plot with the correlation map
            x, y = data_to_plot[1], data_to_plot[2]
            if 'name' in config.spec_data[data_to_plot[3]]:
                field_name = config.spec_data[data_to_plot[3]]['name'] + " " + method_name.capitalize() + " correlation"
            plot_type, findex, fig = data_to_plot[4], data_to_plot[5], data_to_plot[6]
            data_to_plot = (data2d, x, y, field_name, plot_type, findex, fig)
        else:
            # Assume data2d is already a correlation map
            data2d = data_to_plot[0]
            
            # Clear original data reference since we don't have the original datasets
            if hasattr(self, '_original_data'):
                del self._original_data
                
            # Try to determine correlation method from attributes
            method_name = 'Correlation'
            if hasattr(data2d, 'attrs') and 'correlation_method' in data2d.attrs:
                corr_method = data2d.attrs['correlation_method']
                method_name = {
                    'pearson': 'Pearson',
                    'spearman': 'Spearman',
                    'cross': 'Cross'
                }.get(corr_method, 'Correlation')

        self.source_name = config.source_names[config.ds_index]
        self.units = "R value"
        self.fig = fig
        
        ax_opts = config.ax_opts
        if not config.compare and not config.compare_diff:
            fig.set_axes()
        
        ax_temp = fig.get_axes()
        axes_shape = fig.subplots
        
        if axes_shape == (3, 1):
            if ax_opts['is_diff_field']:
                self.ax = ax_temp[2]
            else:
                self.ax = ax_temp[config.axindex]
        elif axes_shape == (2, 2):
            if ax_opts['is_diff_field']:
                self.ax = ax_temp[2]
                if config.ax_opts['add_extra_field_type']:
                    self.ax = ax_temp[3]
            else:
                self.ax = ax_temp[config.axindex]
        elif axes_shape == (1, 2) or axes_shape == (1, 3):
            if isinstance(ax_temp, list):
                self.ax = ax_temp[config.axindex]
            else:
                self.ax = ax_temp
        else:
            self.ax = ax_temp[0]

        ax_opts = fig.update_ax_opts(field_name, self.ax, 'corr', level=0)
        with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
            fig.plot_text(field_name, self.ax, 'corr', level=0, data=data2d)
        
        self._plot_correlation_data(config, self.ax, data2d, x, y, field_name,
                                fig, ax_opts, findex, method_name)
        return fig

    def _plot_correlation_data(self, config, ax, data2d, x, y, field_name,
                        fig, ax_opts, findex, method_name='Pearson'):
        with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
            # Use RdBu_r colormap if not specified
            cmap_name = ax_opts.get('use_cmap', 'RdBu_r')

            is_cartopy_axis = False
            try:
                is_cartopy_axis = isinstance(ax, GeoAxes)
            except ImportError:
                pass

            data_transform = ccrs.PlateCarree()

            vmin, vmax = None, None
            self._create_clevs(field_name, ax_opts, data2d, vmin, vmax)

            # For cross-correlation, the range might be different
            if method_name.lower() == 'cross':
                vmin_default, vmax_default = 0, 1  # Cross-correlation is 0 to 1
            else:
                vmin_default, vmax_default = -1, 1  # Pearson and Spearman are -1 to 1

            if len(x.shape) == 1 and len(y.shape) == 1:
                X, Y = np.meshgrid(x, y)
                cfilled = ax.pcolormesh(X, Y, data2d.values, 
                                    cmap=cmap_name, 
                                    vmin=ax_opts.get('vmin', vmin_default), 
                                    vmax=ax_opts.get('vmax', vmax_default),
                                    shading='auto')
            else:
                if fig.use_cartopy and is_cartopy_axis:
                    cfilled = self.filled_contours(config, field_name, ax, x, y, data2d, 
                                                vmin=vmin, vmax=vmax, transform=data_transform)
                    if 'extent' in ax_opts:
                        self.set_cartopy_ticks(ax, ax_opts['extent'])
                    else:
                        self.set_cartopy_ticks(ax, [-180, 180, -90, 90])
                else:
                    cfilled = self.filled_contours(config, field_name, ax, x, y, data2d,
                                                vmin=vmin, vmax=vmax)

            if cfilled is None:
                self.set_const_colorbar(cfilled, fig, ax)
            else:
                if 'colorbar_label' not in ax_opts:
                    ax_opts['colorbar_label'] = f'{method_name} Correlation'
                self.set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d)
            
            # Calculate and display R² value
            if hasattr(self, '_original_data') and len(self._original_data) == 2:
                data1, data2 = self._original_data
                r_squared = self._calculate_r_squared(data1, data2)
            else:
                # Estimate R² from correlation values if original data is not available
                r_squared = self._estimate_r_squared_from_correlation(data2d)
        
            if not np.isnan(r_squared):
                r_squared_text = f'R² = {r_squared:.3f}'
                
                # Position the text in the upper right corner
                x_pos = 0.92
                y_pos = 1.01
                
                ax.text(x_pos, y_pos, r_squared_text, 
                    transform=ax.transAxes, 
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3),
                    fontsize=10)

    def _compute_correlation_map(self, corr_method, data1, data2):
        """Compute pixel-wise correlation coefficient between two datasets."""
        self.logger.debug("Computing correlation map")
        
        if not isinstance(data1, xr.DataArray) or not isinstance(data2, xr.DataArray):
            self.logger.error("Both inputs must be xarray DataArrays")
            return None
            
        # Regrid?
        if data1.shape != data2.shape:
            self.logger.warning(f"Data shapes don't match: {data1.shape} vs {data2.shape}, "
                                f"correlation may not be meaningful")
        
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
            
            self.logger.info(f"Computing {corr_method} correlation across {time_len} time points "
                            f"for a {y_len}x{x_len} grid")
            
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
                        if corr_method == 'pearson':
                            r, _ = pearsonr(ts1[mask], ts2[mask])
                            corr_data[i, j] = r
                        elif corr_method == 'spearman':
                            r, _ = spearmanr(ts1[mask], ts2[mask])
                            corr_data[i, j] = r
                        elif corr_method == 'cross':
                            # Normalize the data
                            ts1_norm = (ts1[mask] - np.mean(ts1[mask])) / (np.std(ts1[mask]) or 1)
                            ts2_norm = (ts2[mask] - np.mean(ts2[mask])) / (np.std(ts2[mask]) or 1)
                            
                            # Calculate cross-correlation
                            cross_corr = correlate(ts1_norm, ts2_norm, mode='valid') / len(ts1_norm)
                            
                            # Use the maximum absolute correlation value
                            corr_data[i, j] = np.max(np.abs(cross_corr))
                    except Exception as e:
                        self.logger.debug(f"Error computing {corr_method} correlation at point "
                                        f"({i},{j}): {e}")
                        corr_data[i, j] = np.nan
            
            # Set appropriate attributes based on correlation method
            method_name = {
                'pearson': 'Pearson',
                'spearman': 'Spearman',
                'cross': 'Cross'
            }.get(corr_method, 'Correlation')
            
            corr_data.attrs['long_name'] = f'{method_name} Correlation Coefficient'
            corr_data.attrs['units'] = 'dimensionless'
            corr_data.attrs['description'] = f'{method_name} correlation between {data1.name} and {data2.name} across time'
            corr_data.attrs['correlation_method'] = corr_method
            
            return corr_data
            
        # Handle 2D data
        elif len(dims1) == 2:
            self.logger.debug(f"Processing 2D data with dimensions {dims1}")
            
            corr_data = xr.zeros_like(data1)
            
            # For 2D data, we can't compute pixel-wise correlation across time
            # Instead, compute spatial correlation or return a warning
            self.logger.warning(f"Input data is 2D, computing spatial {corr_method} correlation instead "
                                f"of pixel-wise temporal correlation")
            
            flat1 = data1.values.flatten()
            flat2 = data2.values.flatten()
            
            mask = ~np.isnan(flat1) & ~np.isnan(flat2)
            if np.sum(mask) < 2:
                self.logger.error("Not enough valid data points for correlation")
                return None
                
            try:
                if corr_method == 'pearson':
                    r, _ = pearsonr(flat1[mask], flat2[mask])
                elif corr_method == 'spearman':
                    r, _ = spearmanr(flat1[mask], flat2[mask])
                elif corr_method == 'cross':
                    # Normalize the data
                    flat1_norm = (flat1[mask] - np.mean(flat1[mask])) / (np.std(flat1[mask]) or 1)
                    flat2_norm = (flat2[mask] - np.mean(flat2[mask])) / (np.std(flat2[mask]) or 1)
                    
                    # Calculate cross-correlation
                    cross_corr = correlate(flat1_norm, flat2_norm, mode='valid') / len(flat1_norm)
                    
                    # Use the maximum absolute correlation value
                    r = np.max(np.abs(cross_corr))
                
                # Fill the entire map with this value
                corr_data.values.fill(r)
                
                # Set appropriate attributes based on correlation method
                method_name = {
                    'pearson': 'Pearson',
                    'spearman': 'Spearman',
                    'cross': 'Cross'
                }.get(corr_method, 'Correlation')
                
                corr_data.attrs['long_name'] = f'{method_name} Correlation Coefficient (Spatial)'
                corr_data.attrs['units'] = 'dimensionless'
                corr_data.attrs['description'] = f'Spatial {method_name} correlation between {data1.name} and {data2.name}'
                corr_data.attrs['correlation_method'] = corr_method
                
                return corr_data
            except Exception as e:
                self.logger.error(f"Error computing spatial {corr_method} correlation: {e}")
                return None
        else:
            self.logger.error(f"Unsupported data dimensions: {dims1}")
            return None


    def _calculate_r_squared(self, data1, data2):
        """
        Calculate the coefficient of determination (R²) between two datasets.
        
        Args:
            data1 (xarray.DataArray): First dataset
            data2 (xarray.DataArray): Second dataset
            
        Returns:
            float: The R² value
        """
        # Flatten the arrays and remove NaN values
        flat1 = data1.values.flatten()
        flat2 = data2.values.flatten()
        
        mask = ~np.isnan(flat1) & ~np.isnan(flat2)
        if np.sum(mask) < 2:
            self.logger.error("Not enough valid data points for R² calculation")
            return np.nan
        
        x = flat1[mask]
        y = flat2[mask]
        
        # Calculate R² directly
        correlation_matrix = np.corrcoef(x, y)
        if correlation_matrix.size >= 4:  # At least a 2x2 matrix
            correlation_xy = correlation_matrix[0, 1]
            r_squared = correlation_xy ** 2
        else:
            r_squared = np.nan
        
        return r_squared

    def _estimate_r_squared_from_correlation(self, corr_data):
        """
        Estimate R² from correlation values.
        
        Args:
            corr_data (xarray.DataArray): Correlation data
            
        Returns:
            float: Estimated R² value
        """
        # Flatten the correlation values and remove NaNs
        corr_values = corr_data.values.flatten()
        corr_values = corr_values[~np.isnan(corr_values)]
        
        if len(corr_values) == 0:
            return np.nan
        
        # For spatial correlation maps, we want the mean R²
        r_squared_values = corr_values ** 2
        return np.mean(r_squared_values)
