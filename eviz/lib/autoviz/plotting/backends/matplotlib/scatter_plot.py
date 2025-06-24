import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from .base import MatplotlibBasePlotter


class MatplotlibScatterPlotter(MatplotlibBasePlotter):
    """Matplotlib implementation of scatter plotting."""
    def __init__(self):
        super().__init__()
        self.fig = None
        self.ax = None
    
    def plot(self, config, data_to_plot):
        """Create a scatter plot using Matplotlib.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (x_data, y_data, z_data, field_name, plot_type, findex, fig)
                where z_data is optional and can be used for coloring points
        
        Returns:
            The created figure
        """
        data2d, x, y, field_name, plot_type, findex, fig = data_to_plot
        
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
        
        if x is None or y is None:
            return fig
        
        ax_opts = fig.update_ax_opts(field_name, self.ax, 'sc', level=0)
        fig.plot_text(field_name, self.ax, 'sc', level=0)
        
        self._plot_scatter_data(config, self.ax, fig, ax_opts, x, y, data2d, field_name, findex)
        
        self.plot_object = fig
        
        return fig

    def _plot_scatter_data(self, config, ax, fig, ax_opts, x, y, data2d, field_name, findex):
        """ Create a single scatter using SPECS data
            config (Config) : configuration used to specify data sources
            data_to_plot (dict) : dict with plotted data and specs

        Parameters:
            config (Config) : configuration used to specify data sources
            data_to_plot (tuple) : dict with plotted data and specs
        """
        with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
            is_cartopy_axis = False
            try:
                is_cartopy_axis = isinstance(ax, GeoAxes)
            except ImportError:
                pass
            
            if self.fig.use_cartopy and is_cartopy_axis:
                # ax.stock_img()
                scat = ax.scatter(x, y, c=data2d, cmap=ax_opts['use_cmap'], s=5,
                                transform=ccrs.PlateCarree())
            else:
                scat = ax.scatter(x, y, c=data2d, cmap=ax_opts['use_cmap'], s=2)

            if scat is None:
                self.logger.error("Scatter plot failed")
                return 
            else:
                self._set_cartopy_ticks_alt(ax, ax_opts['extent'])
                self.set_colorbar(config, scat, fig, ax, ax_opts, findex, field_name, data2d)

            ax.set_title(f'{field_name}')   

    def _plot_scatter_data_alt(self, config, ax, ax_opts, x_data, y_data, z_data, field_name, findex):
        """Helper method that plots the scatter data."""        
        with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
            marker_size = ax_opts.get('marker_size', 20)
            marker_style = ax_opts.get('marker_style', 'o')
            cmap = ax_opts.get('use_cmap', 'viridis')
            alpha = ax_opts.get('alpha', 0.7)
            
            units = "n.a."
            if 'units' in config.spec_data[field_name]:
                units = config.spec_data[field_name]['units']
            elif hasattr(z_data, 'attrs') and 'units' in z_data.attrs:
                units = z_data.attrs['units']
            elif hasattr(z_data, 'units'):
                units = z_data.units
            
            x_values = x_data.values if hasattr(x_data, 'values') else np.array(x_data)
            y_values = y_data.values if hasattr(y_data, 'values') else np.array(y_data)
            
            if z_data is not None:
                z_values = z_data.values if hasattr(z_data, 'values') else np.array(z_data)
                
                scatter = ax.scatter(x_values, y_values, c=z_values, 
                                    s=marker_size, marker=marker_style, 
                                    cmap=cmap, alpha=alpha)
                
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(units)
            else:
                scatter = ax.scatter(x_values, y_values, 
                                    s=marker_size, marker=marker_style, 
                                    alpha=alpha)
            
            x_label = ax_opts.get('xlabel', x_data.name if hasattr(x_data, 'name') else 'X')
            y_label = ax_opts.get('ylabel', y_data.name if hasattr(y_data, 'name') else 'Y')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            if ax_opts.get('add_grid', True):
                ax.grid(True, linestyle='--', alpha=0.7)
            
            if ax_opts.get('add_regression', False):
                try:
                    from scipy import stats
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
                    line_x = np.array([min(x_values), max(x_values)])
                    line_y = slope * line_x + intercept
                    
                    ax.plot(line_x, line_y, 'r-', linewidth=2)
                    
                    r_squared = r_value**2
                    ax.text(0.05, 0.95, f'$R^2 = {r_squared:.3f}$', 
                            transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                except Exception as e:
                    self.logger.warning(f"Error adding regression line: {e}")
