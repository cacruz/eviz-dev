import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib as mpl

from eviz.lib.autoviz.plotting.base import XYPlotter

class MatplotlibXYPlotter(XYPlotter):
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
        
        # Store the figure for later use
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
        
        if data2d is None:
            return fig
        
        ax_opts = fig.update_ax_opts(field_name, self.ax, 'xy', level=0)
        fig.plot_text(field_name, self.ax, 'xy', level=0, data=data2d)
        
        self._plot_xy_data(config, self.ax, data2d, x, y, field_name, fig, ax_opts, 0, plot_type, findex)
        
        self.plot_object = fig
        
        return fig
    
    def _plot_xy_data(self, config, ax, data2d, x, y, field_name, fig, ax_opts, level, plot_type, findex):
        """Helper function to plot XY data on a single axes."""
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
        if fig.use_cartopy and is_cartopy_axis:
            cfilled = self._filled_contours(config, field_name, ax, x, y, data2d, transform=data_transform)
            if 'extent' in ax_opts:
                self._set_cartopy_ticks(ax, ax_opts['extent'])
            else:
                self._set_cartopy_ticks(ax, [-180, 180, -90, 90])
        else:
            cfilled = self._filled_contours(config, field_name, ax, x, y, data2d)

        if cfilled is None:
            self._set_const_colorbar(cfilled, fig, ax)
        else:
            self._set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d)
            if ax_opts.get('line_contours', False):
                if fig.use_cartopy and is_cartopy_axis:
                    self._line_contours(fig, ax, ax_opts, x, y, data2d, transform=data_transform)
                else:
                    self._line_contours(fig, ax, ax_opts, x, y, data2d)

        if config.compare_diff:
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
                else:
                    if level is not None:
                        if level > 10000:
                            level_text = '@ ' + str(level) + ' Pa'
                        else:
                            level_text = '@ ' + str(level) + ' mb'

            if level_text:
                name = name + level_text

            fig.suptitle_eviz(name, 
                            fontweight='bold',
                            fontstyle='italic',
                            fontsize=self._image_font_size(fig.subplots))
        
        elif config.compare:
            if not config.is_regional:
                fig.set_size_inches(16, 6)
            fig.suptitle_eviz(text=config.map_params[findex].get('field', 'No name'), 
                            fontweight='bold',
                            fontstyle='italic',
                            fontsize=self._image_font_size(fig.subplots))

            if config.add_logo:
                self._add_logo_ax(fig, desired_width_ratio=0.05)
    
    def _filled_contours(self, config, field_name, ax, x, y, data2d, transform=None):
        """Plot filled contours."""
        from matplotlib import colors
        
        self._create_clevs(field_name, config.ax_opts, data2d)
        norm = colors.BoundaryNorm(config.ax_opts['clevs'], ncolors=256, clip=False)
        if config.compare:
            cmap_str = config.ax_opts['use_diff_cmap']
        else:
            cmap_str = config.ax_opts['use_cmap']

        # Check for constant field
        vmin, vmax = np.nanmin(data2d), np.nanmax(data2d)
        if np.isclose(vmin, vmax):
            self.logger.debug("Fill with a neutral color and print text")
            ax.set_facecolor('whitesmoke')
            ax.text(0.5, 0.5, 'zero field', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, color='gray', fontweight='bold')
            return None

        try:
            if np.all(np.diff(config.ax_opts['clevs']) > 0):
                cfilled = ax.contourf(x, y, data2d,
                                    levels=config.ax_opts['clevs'],
                                    cmap=cmap_str,
                                    extend=config.ax_opts['extend_value'],
                                    norm=norm,
                                    transform=transform)
                if config.ax_opts['cmap_set_under']:
                    cfilled.cmap.set_under(config.ax_opts['cmap_set_under'])
                if config.ax_opts['cmap_set_over']:
                    cfilled.cmap.set_over(config.ax_opts['cmap_set_over'])
                ax.set_aspect('auto')
                return cfilled
            else:
                raise ValueError("Contour levels must be increasing")
        except ValueError as e:
            self.logger.error(f"Error: {e}")
            try:
                cfilled = ax.contourf(x, y, data2d, extend='both',
                                    transform=transform)
            except Exception:
                cfilled = ax.contourf(x, y, data2d, extend='both')

            return cfilled
    
    def _line_contours(self, fig, ax, ax_opts, x, y, data2d, transform=None):
        """Add line contours to the plot."""
        import eviz.lib.autoviz.utils as pu
        
        with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
            contour_format = pu.contour_format_from_levels(
                pu.formatted_contours(ax_opts['clevs']),
                scale=ax_opts['cscale'])
            clines = ax.contour(x, y, data2d, levels=ax_opts['clevs'], colors="black",
                                alpha=0.5, transform=transform)
            if len(clines.allsegs) == 0 or all(len(seg) == 0 for seg in clines.allsegs):
                self.logger.warning("No contours were generated. Skipping contour labeling.")
                return
            ax.clabel(clines, inline=1, fontsize=pu.contour_label_size(fig.subplots),
                    colors="black", fmt=contour_format)
    
    def _create_clevs(self, field_name, ax_opts, data2d):
        """Create contour levels for the plot."""
        if ax_opts['clevs']:
            return
        dmin = data2d.min(skipna=True).values
        dmax = data2d.max(skipna=True).values
        self.logger.debug(f"dmin: {dmin}, dmax: {dmax}")

        range_val = abs(dmax - dmin)
        precision = max(0, int(np.ceil(-np.log10(range_val)))) if range_val != 0 else 6
        if range_val <= 9.0:
            precision = 1
        ax_opts['clevs_prec'] = precision
        self.logger.debug(f"range_val: {range_val}, precision: {precision}")

        # Generate levels
        if not ax_opts.get('create_clevs', True):
            clevs = np.around(np.linspace(dmin, dmax, 10), decimals=precision)
        else:
            clevs = np.around(np.linspace(dmin, dmax, ax_opts.get('num_clevs', 10)),
                            decimals=precision)
            clevs = np.unique(clevs)  # Remove duplicates

        # Check if levels are strictly increasing
        # If not enough unique levels, regenerate with more precision or fallback
        if len(set(clevs)) <= 2:
            self.logger.debug(f"Not enough unique contour levels for {field_name}.")
            # Try with more levels and higher precision
            clevs = np.linspace(dmin, dmax, 10)
            clevs = np.unique(np.around(clevs, decimals=6))
            if len(clevs) <= 2:
                # As a last resort, just use [dmin, dmax]
                clevs = np.array([dmin, dmax])

        # Ensure strictly increasing
        clevs = np.unique(clevs)  # Remove duplicates, again
        ax_opts['clevs'] = clevs

        self.logger.debug(f'Created contour levels for {field_name}: {ax_opts["clevs"]}')
        if ax_opts['clevs'][0] == 0.0:
            ax_opts['extend_value'] = "max"
    
    def _set_colorbar(self, config, cfilled, fig, ax, ax_opts, findex, field_name, data2d):
        """Add a colorbar to the plot."""
        import eviz.lib.autoviz.utils as pu
        
        try:
            source_name = config.source_names[config.ds_index]
            if ax_opts['cbar_sci_notation']:
                fmt = pu.FlexibleOOMFormatter(min_val=data2d.min().compute().item(),
                                            max_val=data2d.max().compute().item(),
                                            math_text=True)
            else:
                fmt = pu.OOMFormatter(prec=ax_opts['clevs_prec'], math_text=True)

            if not fig.use_cartopy:
                cbar = fig.colorbar(cfilled)
            else:
                cbar = fig.colorbar(cfilled, ax=ax,
                                    orientation='vertical' if config.compare or config.compare_diff else 'horizontal',
                                    pad=pu.cbar_pad(fig.subplots),
                                    fraction=pu.cbar_fraction(fig.subplots),
                                    ticks=ax_opts.get('clevs', None),
                                    format=fmt,
                                    shrink=pu.cbar_shrink(fig.subplots))

            # Use the following ONLY with the FlexibleOOMFormatter()
            if ax_opts['cbar_sci_notation']:
                cbar.ax.text(1.05, -0.5, r'$\times 10^{%d}$' % fmt.oom,
                            transform=cbar.ax.transAxes, va='center', ha='left', fontsize=12)

            # Get units for the colorbar
            units = self._get_units(config, field_name, data2d, source_name, findex)

            if ax_opts['clabel'] is None:
                cbar_label = units
            else:
                cbar_label = ax_opts['clabel']
            cbar.set_label(cbar_label, size=pu.bar_font_size(fig.subplots))

            # Set font size for colorbar ticks
            for t in cbar.ax.get_xticklabels():
                t.set_fontsize(pu.contour_tick_font_size(fig.subplots))
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(pu.contour_tick_font_size(fig.subplots))

        except Exception as e:
            self.logger.error(f"Failed to add colorbar: {e}")
    
    def _get_units(self, config, field_name, data2d, source_name, findex):
        """Get units for the field."""
        try:
            if field_name in config.spec_data and 'units' in config.spec_data[field_name]:
                return config.spec_data[field_name]['units']
            
            if hasattr(data2d, 'attrs') and 'units' in data2d.attrs:
                return data2d.attrs['units']
            elif hasattr(data2d, 'units'):
                return data2d.units
            
            # Try to get units from the reader
            reader = None
            if source_name in config.readers:
                if isinstance(config.readers[source_name], dict):
                    readers_dict = config.readers[source_name]
                    if 'NetCDF' in readers_dict:
                        reader = readers_dict['NetCDF']
                    elif readers_dict:
                        reader = next(iter(readers_dict.values()))
                else:
                    reader = config.readers[source_name]

            if reader and hasattr(reader, 'datasets'):
                if findex in reader.datasets and 'vars' in reader.datasets[findex]:
                    field_var = reader.datasets[findex]['vars'].get(field_name)
                    if field_var and hasattr(field_var, 'attrs') and 'units' in field_var.attrs:
                        return field_var.attrs['units']
                    elif field_var and hasattr(field_var, 'units'):
                        return field_var.units
            
            return "n.a."
        except Exception as e:
            self.logger.warning(f"Error getting units: {e}")
            return "n.a."
    
    def _set_const_colorbar(self, cfilled, fig, ax):
        """Add a colorbar for a constant field."""
        _ = fig.colorbar(cfilled, ax=ax, shrink=0.5)
    
    def _set_cartopy_ticks(self, ax, extent, labelsize=10):
        """Add gridlines and tick labels to a Cartopy map."""
        if not extent or len(extent) != 4:
            self.logger.warning(f"Invalid extent {extent}, using default")
            extent = [-180, 180, -90, 90]
        
        try:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        except Exception as e:
            self.logger.warning(f"Could not set extent: {e}")
        
        try:
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.8, 
                color='gray', 
                alpha=0.6, 
                linestyle='--'
            )
            
            gl.top_labels = False
            gl.bottom_labels = True
            gl.left_labels = True
            gl.right_labels = False
            
            gl.xlabel_style = {'size': labelsize, 'rotation': 0}
            gl.ylabel_style = {'size': labelsize, 'rotation': 0}
            
            return True
        except Exception as e:
            self.logger.error(f"Could not set ticks and labels: {e}")
            return False
    
    def _image_font_size(self, subplots):
        """Get appropriate font size based on subplot layout."""
        import eviz.lib.autoviz.utils as pu
        return pu.image_font_size(subplots)
    
    def _add_logo_ax(self, fig, desired_width_ratio=0.05):
        """Add a logo to the figure."""
        import eviz.lib.autoviz.utils as pu
        return pu.add_logo_ax(fig, desired_width_ratio)
    
    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        if self.fig is not None:
            self.fig.savefig(filename, **kwargs)
            self.logger.info(f"Saved plot to {filename}")
        else:
            self.logger.warning("No figure to save")
    

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
