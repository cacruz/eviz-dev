import matplotlib as mpl
import numpy as np
from matplotlib.ticker import FormatStrFormatter
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
            data_to_plot: Tuple containing (x, y, z_data, field_name, plot_type, findex, fig)
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
        
        # Add shared colorbar if enabled
        if config.compare and config.shared_cbar:
            self.add_shared_colorbar(fig, config._filled_contours, field_name, config)
        
        return fig

    def _plot_scatter_data(self, config, ax, fig, ax_opts, x, y, data2d, field_name, findex):
        """Create a single scatter plot using SPECS data.

        Parameters:
            config (Config): Configuration with data source and plotting options.
            ax (matplotlib axis): The axis to plot on.
            fig (matplotlib figure): The figure object.
            ax_opts (dict): Axis-specific options including colormap and levels.
            x, y (array-like): Coordinates for scatter points.
            data2d (xarray or array-like): Data values for coloring.
            field_name (str): The field being plotted.
            findex (int): Index of this field in the comparison sequence.
        """
        with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
            if 'fill_value' in config.spec_data[field_name].get('xyplot', {}):
                fill_value = config.spec_data[field_name]['xyplot']['fill_value']
                data2d = data2d.where(data2d != fill_value, np.nan)

            masked_data = np.ma.masked_invalid(data2d)

            is_cartopy_axis = False
            try:
                from cartopy.mpl.geoaxes import GeoAxes
                is_cartopy_axis = isinstance(ax, GeoAxes)
            except ImportError:
                pass

            data_transform = ccrs.PlateCarree() if is_cartopy_axis else None

            # Use shared vmin/vmax if available for consistent colorbar across plots
            vmin, vmax = None, None
            if config.compare or not config.compare_diff:
                if not hasattr(config, '_comparison_cbar_limits'):
                    config._comparison_cbar_limits = {}

                if field_name in config._comparison_cbar_limits:
                    vmin, vmax = config._comparison_cbar_limits[field_name]

            # Adaptive point size
            npts = len(x)
            if npts > 1e5:
                point_size = 1
            elif npts > 1e4:
                point_size = 2
            else:
                point_size = 5

            scatter_kwargs = dict(
                x=x,
                y=y,
                c=masked_data,
                cmap=ax_opts['use_cmap'],
                s=point_size,
                vmin=vmin,
                vmax=vmax,
                # edgecolors='k',
                # linewidths=0.2,
                alpha=0.7,
                transform=data_transform
            )

            scat = ax.scatter(**scatter_kwargs)

            # Set Cartopy-specific settings
            if is_cartopy_axis and self.fig.use_cartopy:
                # ax.stock_img()
                if 'extent' in ax_opts:
                    self._set_cartopy_ticks_alt(ax, ax_opts['extent'])
                else:
                    self.set_cartopy_ticks(ax, [-180, 180, -90, 90])

            if (config.compare or not config.compare_diff) and config.axindex == 0:
                config._comparison_cbar_limits[field_name] = scat.get_clim()

            # Suppress individual colorbars if shared_bar is enabled
            if config.shared_cbar:
                ax_opts['suppress_colorbar'] = True

            # Add colorbar or dummy handler
            if scat is None:
                self.set_const_colorbar(scat, fig, ax)
            else:
                self.set_colorbar(config, scat, fig, ax, ax_opts, findex, field_name, data2d)

            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # Title and logo
            title_text = config.spec_data[field_name].get('name', field_name)
            if config.compare_diff:
                fig.suptitle_eviz(
                    title_text,
                    fontweight='bold',
                    fontstyle='italic',
                    fontsize=self._image_font_size(fig.subplots)
                )
            elif config.compare:
                fig.suptitle_eviz(
                    title_text,
                    fontweight='bold',
                    fontstyle='italic',
                    fontsize=self._image_font_size(fig.subplots)
                )
                if config.add_logo:
                    self._add_logo_ax(fig, desired_width_ratio=0.05)

            # Collect color-filled objects for shared colorbar handling
            if not hasattr(config, '_filled_contours'):
                config._filled_contours = []
            config._filled_contours.append(scat)

    def _plot_scatter_data_alt(self, config, ax, fig, ax_opts, x, y, data2d, field_name, findex):
        """ Create a single scatter using SPECS data
            config (Config) : configuration used to specify data sources
            data_to_plot (dict) : dict with plotted data and specs

        Parameters:
            config (Config) : configuration used to specify data sources
            data_to_plot (tuple) : dict with plotted data and specs
        """
        if 'fill_value' in config.spec_data[field_name]['xyplot']:
            fill_value = config.spec_data[field_name]['xyplot']['fill_value']
            data2d = data2d.where(data2d != fill_value, np.nan)

        # Check if we're using Cartopy and if the axis is a GeoAxes
        is_cartopy_axis = False
        try:
            is_cartopy_axis = isinstance(ax, GeoAxes)
        except ImportError:
            pass

        data_transform = ccrs.PlateCarree()

        vmin, vmax = None, None
        if config.compare or not config.compare_diff:
            # Check if we've stored limits for this field in the config
            if not hasattr(config, '_comparison_cbar_limits'):
                config._comparison_cbar_limits = {}
                
            if field_name in config._comparison_cbar_limits:
                vmin, vmax = config._comparison_cbar_limits[field_name]

        if self.fig.use_cartopy and is_cartopy_axis:
            ax.stock_img()
            scat = ax.scatter(x, y, c=data2d, cmap=ax_opts['use_cmap'], s=5,
                            transform=data_transform)
            if 'extent' in ax_opts:
                self._set_cartopy_ticks_alt(ax, ax_opts['extent'])
            else:
                self.set_cartopy_ticks(ax, [-180, 180, -90, 90])
        else:
            scat = ax.scatter(x, y, c=data2d, cmap=ax_opts['use_cmap'], s=2)

        if scat is None:
            self.set_const_colorbar(scat, fig, ax)
        else:
            # Store colorbar limits for the first plot in a comparison
            if (config.compare or not config.compare_diff) and config.axindex == 0:
                # Get the limits used in the plot
                vmin, vmax = scat.get_clim()
                config._comparison_cbar_limits[field_name] = (vmin, vmax)

            # Suppress individual colorbars if shared_bar is enabled
            if config.shared_cbar:
                ax_opts['suppress_colorbar'] = True

            self.set_colorbar(config, scat, fig, ax, ax_opts, findex, field_name, data2d)
            if ax_opts.get('line_contours', False):
                if fig.use_cartopy and is_cartopy_axis:
                    self.line_contours(fig, ax, ax_opts, x, y, data2d, transform=data_transform)
                else:
                    self.line_contours(fig, ax, ax_opts, x, y, data2d)

        if config.compare_diff:
            name = field_name
            if 'name' in config.spec_data[field_name]:
                name = config.spec_data[field_name]['name']

            fig.suptitle_eviz(name, 
                            fontweight='bold',
                            fontstyle='italic',
                            fontsize=self._image_font_size(fig.subplots))
        
        elif config.compare:
            fig.suptitle_eviz(text=field_name, 
                            fontweight='bold',
                            fontstyle='italic',
                            fontsize=self._image_font_size(fig.subplots))

            if config.add_logo:
                self._add_logo_ax(fig, desired_width_ratio=0.05)

        # Collect filled contour objects for shared colorbar
        if not hasattr(config, '_filled_contours'):
            config._filled_contours = []
        config._filled_contours.append(scat)
