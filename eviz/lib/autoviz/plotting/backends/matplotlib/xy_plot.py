import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes

from .base import MatplotlibBasePlotter


class MatplotlibXYPlotter(MatplotlibBasePlotter):
    """Matplotlib implementation of XY plotting."""
    def __init__(self):
        super().__init__()
        self.fig = None
        self.ax = None

    def plot(self, config, data_to_plot):
        """Create an XY plot using Matplotlib.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created figure
        """
        data2d, x, y, field_name, plot_type, findex, fig = data_to_plot

        if data2d is None:
            return fig
        
        # Store the figure for later use
        self.fig = fig
        
        # Initialize collections for shared colorbar if in comparison mode
        if config.shared_cbar and (config.compare or config.compare_diff):
            # Use as class attributes for persistence across multiple plots
            if not hasattr(MatplotlibXYPlotter, 'cfilled_objects'):
                MatplotlibXYPlotter.cfilled_objects = []
            if not hasattr(MatplotlibXYPlotter, 'axes_list'):
                MatplotlibXYPlotter.axes_list = []
        
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

        ax_opts = fig.update_ax_opts(field_name, self.ax, 'xy', level=0)
        fig.plot_text(field_name, self.ax, 'xy', level=0, data=data2d)
        
        cfilled = self._plot_xy_data(config, self.ax, data2d, x, y, field_name, fig, ax_opts, 0, plot_type, findex)

        # For comparison plots, collect cfilled objects for shared colorbar
        if config.shared_cbar and (config.compare or config.compare_diff) and cfilled is not None:
            MatplotlibXYPlotter.cfilled_objects.append(cfilled)
            MatplotlibXYPlotter.axes_list.append(self.ax)
            
            # Check if this is the last plot in the comparison series
            if hasattr(config, 'compare_exp_ids') and len(MatplotlibXYPlotter.cfilled_objects) == len(config.compare_exp_ids):
                # Adjust subplot positions to make room for colorbar
                fig.subplots_adjust(right=0.85)
                self.add_shared_colorbar(fig, MatplotlibXYPlotter.cfilled_objects, MatplotlibXYPlotter.axes_list, field_name, config)
                # Clear the lists for the next plot
                MatplotlibXYPlotter.cfilled_objects = []
                MatplotlibXYPlotter.axes_list = []    

        return fig

    def _plot_xy_data(self, config, ax, data2d, x, y, field_name, fig, ax_opts, level, plot_type, findex):
        """Helper function to plot XY data on a single axes."""
        if 'fill_value' in config.spec_data[field_name]['xyplot']:
            fill_value = config.spec_data[field_name]['xyplot']['fill_value']
            data2d = data2d.where(data2d != fill_value, np.nan)

        is_cartopy_axis = False
        try:
            is_cartopy_axis = isinstance(ax, GeoAxes)
        except ImportError:
            pass
                
        data_transform = ccrs.PlateCarree()

        # Get colorbar limits for comparison plots
        vmin, vmax = None, None
        if config.compare or config.compare_diff:
            if not hasattr(config, '_comparison_cbar_limits'):
                config._comparison_cbar_limits = {}
                
            if field_name in config._comparison_cbar_limits:
                vmin, vmax = config._comparison_cbar_limits[field_name]

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
            # Store colorbar limits for the first plot in a comparison
            if (config.compare or config.compare_diff) and config.axindex == 0:
                vmin, vmax = cfilled.get_clim()
                if not hasattr(config, '_comparison_cbar_limits'):
                    config._comparison_cbar_limits = {}
                config._comparison_cbar_limits[field_name] = (vmin, vmax)

            # For comparison plots, suppress individual colorbars
            if config.shared_cbar and (config.compare or config.compare_diff):
                # Don't create individual colorbars when using shared colorbar
                ax_opts['suppress_colorbar'] = True
            else:
                # Create individual colorbar for non-comparison plots
                self.set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d)


            if ax_opts.get('line_contours', False):
                if fig.use_cartopy and is_cartopy_axis:
                    self.line_contours(fig, ax, ax_opts, x, y, data2d, transform=data_transform)
                else:
                    self.line_contours(fig, ax, ax_opts, x, y, data2d)

        if config.compare_diff or config.compare:
            name = field_name
            if 'name' in config.spec_data[field_name]:
                name = config.spec_data[field_name]['name']

            level_text = None
            if config.ax_opts.get('zave', False):
                level_text = ' (Column Mean)'
            elif config.ax_opts.get('zsum', False):
                level_text = ' (Total Column)'
            else:
                if str(level) == '0':
                    level_text = ''
                elif level is not None:
                    level_text = f"@ {level} {'Pa' if level > 10000 else 'mb'}"

            if level_text:
                name = name + level_text

            fig.suptitle_eviz(name,
                            fontweight='bold',
                            fontstyle='italic',
                            fontsize=self._image_font_size(fig.subplots))

            if config.add_logo:
                self._add_logo_ax(fig, desired_width_ratio=0.05)

        return cfilled
            
    def show(self):
        """Display the plot."""
        if self.fig is not None:
            try:
                # Use the custom show_eviz method if available
                if hasattr(self.fig, 'show_eviz'):
                    self.fig.show_eviz()
                else:
                    # Fall back to regular plt.show()
                    plt.show()
            except Exception as e:
                self.logger.error(f"Error showing figure: {e}")
        else:
            self.logger.warning("No figure to show")
