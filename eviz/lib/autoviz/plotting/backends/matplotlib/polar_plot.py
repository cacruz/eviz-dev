from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .base import MatplotlibBasePlotter
import eviz.lib.autoviz.utils as pu

class MatplotlibPolarPlotter(MatplotlibBasePlotter):
    """Matplotlib implementation of polar plotting."""
    
    def __init__(self):
        super().__init__()
        self.fig = None
        self.ax = None
    
    def plot(self, config, data_to_plot):
        """Create a polar plot using Matplotlib.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created figure
        """
        data2d, x, y, field_name, plot_type, findex, fig = data_to_plot
        
        self.fig = fig
        
        if data2d is None:
            self.logger.warning("No data to plot")
            return fig
        
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
        
        ax_opts = fig.update_ax_opts(field_name, self.ax, 'po')
        # fig.plot_text(field_name, self.ax, 'po', level=0, data=data2d)
        
        self._plot_polar_data(config, self.ax, data2d, x, y, field_name, ax_opts, findex)
        
        self.plot_object = fig
        
        return fig
    
    def _plot_polar_data(self, config, ax, data2d, x, y, field_name, ax_opts, findex):
        """Helper function to plot polar data."""
        
        if ax_opts['use_pole'] == 'south':
            projection = ccrs.SouthPolarStereo()
            extent_lat = -60  # Southern limit for South Polar plot
        else:
            projection = ccrs.NorthPolarStereo()
            extent_lat = 60  # Northern limit for North Polar plot

        ax = self.fig.add_subplot(1, 1, 1, projection=projection)

        ax.set_extent([-180, 180, extent_lat, 90 if ax_opts['use_pole'] == 'north' else -90],
                    ccrs.PlateCarree())
        if ax_opts['boundary']:
            theta = np.linspace(0, 2 * np.pi, 100)
            center = [0.5, 0.5]
            radius = 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)

        self._create_clevs(field_name, ax_opts, data2d)
        clevs = pu.formatted_contours(ax_opts['clevs'])

        extend_value = "both"
        if ax_opts['clevs'][0] == 0:
            extend_value = "max"
        norm = colors.BoundaryNorm(ax_opts['clevs'], ncolors=256, clip=False)

        trans = ccrs.PlateCarree()

        try:
            # contourf
            pcm = ax.contourf(x, y, data2d,
                            cmap=ax_opts['use_cmap'],
                            levels=clevs,
                            transform=trans,
                            extend=extend_value,
                            norm=norm)

            plot_success = True
        except Exception as e:
            self.logger.warning(f"contourf failed: {e}")
            try:
                pcm = ax.imshow(
                    data2d,
                    extent=[-180, 180, -90 if ax_opts['use_pole'] == 'south' else 0,
                            0 if ax_opts['use_pole'] == 'south' else 90],
                    transform=ccrs.PlateCarree(),
                    cmap=ax_opts['use_cmap'],
                    aspect='auto'
                )
                plot_success = True
            except Exception as e2:
                self.logger.error(f"All plotting methods failed: {e2}")
                plot_success = False

        if not plot_success:
            self.logger.error("Failed to create polar plot")
            return

        if ax_opts['line_contours']:
            clines = ax.contour(x, y, data2d,
                                levels=ax_opts['clevs'], colors="black",
                                linewidths=0.5, alpha=0.5, linestyles='solid',
                                transform=trans)
            ax.clabel(clines, inline=1, fontsize=8,
                    inline_spacing=10, colors="black",
                    rightside_up=True,  # fmt=contour_format,
                    use_clabeltext=True)
        else:
            _ = ax.contour(x, y, data2d, linewidths=0.0)

        ax.add_feature(cfeature.BORDERS, zorder=10, linewidth=0.5, edgecolor='grey')
        ax.add_feature(cfeature.LAKES, alpha=0.9)
        ax.add_feature(cfeature.LAND, color='silver', zorder=1, facecolor=0.9)
        ax.add_feature(cfeature.COASTLINE, zorder=10, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=0)

        cbar = plt.colorbar(pcm, ax=ax, shrink=0.5, pad=0.05)
        if 'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        else:
            try:
                units = data2d.units
            except Exception as e:
                self.logger.error(f"{e}: Please specify {field_name} units in specs file")
                units = "n.a."
        if units == '1':
            units = '%'
        if ax_opts['clabel'] is None:
            cbar_label = units
        else:
            cbar_label = ax_opts['clabel']
        cbar.set_label(label=cbar_label, size=12, weight='bold')

        if ax_opts['add_grid']:
            _ = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.75,
                            linestyle='--')

        if 'name' in config.spec_data[field_name]:
            ax.set_title(config.spec_data[field_name]['name'], y=1.03, fontsize=14,
                        weight='bold')
        else:
            ax.set_title(self.source_name, y=1.03, fontsize=14, weight='bold')   # TODO: check source_name
    
    def _convert_to_polar(self, x, y):
        """Convert Cartesian coordinates to polar coordinates."""
        try:
            # Check if x and y are already in polar form
            if hasattr(x, 'name') and 'theta' in x.name.lower():
                theta = x.values
                r = y.values
            elif hasattr(y, 'name') and 'theta' in y.name.lower():
                theta = y.values
                r = x.values
            else:
                # Convert from Cartesian to polar
                if hasattr(x, 'values'):
                    x_values = x.values
                else:
                    x_values = np.array(x)
                
                if hasattr(y, 'values'):
                    y_values = y.values
                else:
                    y_values = np.array(y)
                
                # Create meshgrid if x and y are 1D
                if len(x_values.shape) == 1 and len(y_values.shape) == 1:
                    X, Y = np.meshgrid(x_values, y_values)
                else:
                    X, Y = x_values, y_values
                
                # Convert to polar coordinates
                theta = np.arctan2(Y, X)
                r = np.sqrt(X**2 + Y**2)
            
            return theta, r
            
        except Exception as e:
            self.logger.error(f"Error converting to polar coordinates: {e}")
            # Return dummy coordinates
            return np.linspace(0, 2*np.pi, 100), np.linspace(0, 1, 100)
