import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import logging
from matplotlib.ticker import FixedLocator
from eviz.lib.autoviz.utils import FlexibleOOMFormatter, OOMFormatter
from eviz.lib.autoviz.plotting.base import BasePlotter
import eviz.lib.autoviz.utils as pu
from eviz.lib.autoviz.utils import bar_font_size, contour_tick_font_size


class MatplotlibBasePlotter(BasePlotter):
    """Base class for all Matplotlib plotters with common functionality."""

    def plot(self, config, data_to_plot):
        pass

    def __init__(self):
        super().__init__()
        self.fig = None
        self.ax = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def filled_contours(self, config, field_name, ax, x, y, data2d, 
                        transform=None, vmin=None, vmax=None):
        """Plot filled contours."""
        # Check if data is all NaN
        if np.isnan(data2d).all():
            self.logger.warning(f"All values are NaN for {field_name}. Cannot create contour plot.")
            ax.set_facecolor("whitesmoke")
            ax.text(
                0.5,
                0.5,
                "No valid data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=16,
                color="gray",
                fontweight="bold",
            )
            return None

        # Create contour levels if they don't exist
        if (
            "clevs" not in config.ax_opts
            or config.ax_opts["clevs"] is None
            or len(config.ax_opts["clevs"]) == 0
        ):
            self._create_clevs(field_name, config.ax_opts, data2d)

        norm = colors.BoundaryNorm(config.ax_opts["clevs"], ncolors=256, clip=False)

        if config.compare:
            cmap_str = config.ax_opts["use_diff_cmap"]
        else:
            cmap_str = config.ax_opts["use_cmap"]

        # Check for constant field
        data_vmin, data_vmax = np.nanmin(data2d), np.nanmax(data2d)
        if np.isclose(data_vmin, data_vmax):
            self.logger.debug("Fill with a neutral color and print text")
            ax.set_facecolor("whitesmoke")
            ax.text(
                0.5,
                0.5,
                "Constant field",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=16,
                color="gray",
                fontweight="bold",
            )
            return None

        try:
            if np.all(np.diff(config.ax_opts["clevs"]) > 0):
                cfilled = ax.contourf(
                    x,
                    y,
                    data2d,
                    levels=config.ax_opts["clevs"],
                    cmap=cmap_str,
                    extend=config.ax_opts["extend_value"],
                    norm=colors.Normalize(vmin=vmin, vmax=vmax),
                    transform=transform,
                )

                # Set under/over colors if specified
                if config.ax_opts["cmap_set_under"]:
                    cfilled.cmap.set_under(config.ax_opts["cmap_set_under"])
                if config.ax_opts["cmap_set_over"]:
                    cfilled.cmap.set_over(config.ax_opts["cmap_set_over"])

                ax.set_aspect("auto")
                return cfilled
            else:
                raise ValueError("Contour levels must be increasing")
        except ValueError as e:
            self.logger.error(f"Error: {e}")
            try:
                cfilled = ax.contourf(x, y, data2d, extend="both", transform=transform)
            except Exception:
                cfilled = ax.contourf(x, y, data2d, extend="both")

            return cfilled

    def _create_clevs(self, field_name, ax_opts, data2d, vmin=None, vmax=None):
        """Create contour levels for the plot."""
        # Check if clevs already exists and is not empty
        if 'clevs' in ax_opts and ax_opts['clevs'] is not None and len(ax_opts['clevs']) > 0:
            return
        
        # Check if data is all NaN
        if np.isnan(data2d).all():
            self.logger.warning(f"All values are NaN for {field_name}. Cannot create "
                                f"contour levels.")
            # Set default contour levels to avoid errors
            ax_opts['clevs'] = np.array([0, 1])
            ax_opts['clevs_prec'] = 0
            return
        
        # Use provided vmin and vmax if available
        if vmin is not None and vmax is not None:
            dmin, dmax = vmin, vmax
        else:
            # Get min/max values, skipping NaN values
            dmin = np.nanmin(data2d)
            dmax = np.nanmax(data2d)
        self.logger.debug(f"dmin: {dmin}, dmax: {dmax}")
        
        # Check if min equals max (constant field)
        if np.isclose(dmin, dmax):
            self.logger.debug(f"Constant field detected for {field_name}: {dmin}")
            # Create simple contour levels around the constant value
            ax_opts['clevs'] = np.array([dmin - 0.1, dmin, dmin + 0.1])
            ax_opts['clevs_prec'] = 1
            return
        
        # Calculate appropriate precision
        range_val = abs(dmax - dmin)
        precision = max(0, int(np.ceil(-np.log10(range_val)))) if range_val != 0 else 6
        if range_val <= 9.0:
            precision = 1
        ax_opts['clevs_prec'] = precision
        self.logger.debug(f"range_val: {range_val}, precision: {precision}")
        
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

    def line_contours(self, fig, ax, ax_opts, x, y, data2d, transform=None):
        """Add line contours to the plot."""
        import eviz.lib.autoviz.utils as pu

        with mpl.rc_context(rc=ax_opts.get("rc_params", {})):
            try:
                # Check if clevs exists and has enough levels
                if (
                    "clevs" not in ax_opts
                    or ax_opts["clevs"] is None
                    or len(ax_opts["clevs"]) < 2
                ):
                    self.logger.warning("Not enough contour levels for line contours")
                    return

                try:
                    formatted_clevs = pu.formatted_contours(ax_opts["clevs"])
                    contour_format = pu.contour_format_from_levels(
                        formatted_clevs, scale=ax_opts.get("cscale", None)
                    )
                except IndexError:
                    # Handle the case where contour_format_from_levels fails
                    self.logger.warning("Could not determine contour format, using default")
                    contour_format = "%.1f"

                clines = ax.contour(
                    x,
                    y,
                    data2d,
                    levels=ax_opts["clevs"],
                    colors="black",
                    alpha=0.5,
                    transform=transform,
                )

                if len(clines.allsegs) == 0 or all(
                    len(seg) == 0 for seg in clines.allsegs
                ):
                    self.logger.warning("No contours were generated. Skipping contour labeling.")
                    return

                ax.clabel(
                    clines,
                    inline=1,
                    fontsize=pu.contour_label_size(fig.subplots),
                    colors="black",
                    fmt=contour_format,
                )
            except Exception as e:
                self.logger.error(f"Error adding contour lines: {e}")

    def set_colorbar(self, config, cfilled, fig, ax, ax_opts, findex, field_name, data2d):
        """Add a colorbar to the plot."""
        try:
            # Skip colorbar creation if suppressed (for shared colorbar)
            if ax_opts.get("suppress_colorbar", False):
                self.logger.debug(f"Suppressing individual colorbar for {field_name}")
                return None

            source_name = config.source_names[config.ds_index]

            # Create formatter for colorbar ticks
            if ax_opts["cbar_sci_notation"]:
                fmt = pu.FlexibleOOMFormatter(
                    min_val=data2d.min().compute().item(),
                    max_val=data2d.max().compute().item(),
                    math_text=True,
                )
            else:
                fmt = pu.OOMFormatter(prec=ax_opts["clevs_prec"], math_text=True)

            if not fig.use_cartopy:
                cbar = fig.colorbar(cfilled)
            else:
                cbar = fig.colorbar(
                    cfilled,
                    ax=ax,
                    orientation="vertical"
                    if config.compare or config.compare_diff
                    else "horizontal",
                    pad=pu.cbar_pad(fig.subplots),
                    fraction=pu.cbar_fraction(fig.subplots),
                    ticks=ax_opts.get("clevs", None),
                    format=fmt,
                    shrink=pu.cbar_shrink(fig.subplots),
                )

            # Add scientific notation if requested
            if ax_opts["cbar_sci_notation"]:
                cbar.ax.text(
                    1.05,
                    -0.5,
                    r"$\times 10^{%d}$" % fmt.oom,
                    transform=cbar.ax.transAxes,
                    va="center",
                    ha="left",
                    fontsize=12,
                )

            units = self.get_units(config, field_name, data2d, source_name, findex)

            if ax_opts["clabel"] is None:
                cbar_label = units
            else:
                cbar_label = ax_opts["clabel"]
            cbar.set_label(cbar_label, size=bar_font_size(fig.subplots))

            for t in cbar.ax.get_xticklabels():
                t.set_fontsize(pu.contour_tick_font_size(fig.subplots))
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(pu.contour_tick_font_size(fig.subplots))

            return cbar

        except Exception as e:
            self.logger.error(f"Failed to add colorbar: {e}")
            return None

    def add_shared_colorbar(self, fig, cfilled_objects, field_name, config):
        """Add a shared colorbar for all plots.
        
        Args:
            fig: The figure object
            cfilled_objects: List of filled contour objects
            field_name: Name of the field being plotted
            config: Configuration manager
            
        Returns:
            The created colorbar object
        """
        if not cfilled_objects:
            self.logger.warning("No filled contour objects found for shared colorbar")
            return None
        
        # Get the first cfilled object to use as reference
        cfilled = cfilled_objects[0]
        
        # Get colorbar parameters
        source_name = config.source_names[config.ds_index]
        ax_opts = config.ax_opts
        
        # Create formatter for colorbar ticks
        if ax_opts['cbar_sci_notation']:
            fmt = FlexibleOOMFormatter(min_val=cfilled.norm.vmin,
                                    max_val=cfilled.norm.vmax,
                                    math_text=True)
        else:
            fmt = OOMFormatter(prec=ax_opts['clevs_prec'], math_text=True)
        
        # Adjust figure layout to allocate space for the colorbar
        fig.subplots_adjust(right=0.85)
        
        # Create a new axis for the colorbar
        cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(cfilled, cax=cbar_ax,
                            orientation='vertical',
                            pad=pu.cbar_pad(fig.subplots),
                            fraction=pu.cbar_fraction(fig.subplots),
                            ticks=ax_opts.get('clevs', None),
                            format=fmt,
                            shrink=pu.cbar_shrink(fig.subplots))
        
        # Add scientific notation if requested
        if ax_opts['cbar_sci_notation']:
            cbar.ax.text(1.05, -0.05, r'$\times 10^{%d}$' % fmt.oom,
                        transform=cbar.ax.transAxes, va='center', ha='left',
                        fontsize=bar_font_size(fig.subplots))
        
        # Set colorbar label
        units = self.get_units(config, field_name, None, source_name, config.findex)
        
        if ax_opts['clabel'] is None:
            cbar_label = units
        else:
            cbar_label = ax_opts['clabel']
        
        cbar.set_label(cbar_label, size=bar_font_size(fig.subplots))
        
        # Set tick font size
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(contour_tick_font_size(fig.subplots))
        
        return cbar

    @staticmethod
    def set_const_colorbar(cfilled, fig, ax):
        _ = fig.colorbar(cfilled, ax=ax, shrink=0.5)

    def get_units(self, config, field_name, data2d, source_name, findex):
        """Get units for the field."""
        try:
            if (
                field_name in config.spec_data
                and "units" in config.spec_data[field_name]
            ):
                return config.spec_data[field_name]["units"]

            if hasattr(data2d, "attrs") and "units" in data2d.attrs:
                return data2d.attrs["units"]
            elif hasattr(data2d, "units"):
                return data2d.units

            # Try to get units from the reader
            reader = None
            if source_name in config.readers:
                if isinstance(config.readers[source_name], dict):
                    readers_dict = config.readers[source_name]
                    if "NetCDF" in readers_dict:
                        reader = readers_dict["NetCDF"]
                    elif readers_dict:
                        reader = next(iter(readers_dict.values()))
                else:
                    reader = config.readers[source_name]

            if reader and hasattr(reader, "datasets"):
                if findex in reader.datasets and "vars" in reader.datasets[findex]:
                    field_var = reader.datasets[findex]["vars"].get(field_name)
                    if (
                        field_var
                        and hasattr(field_var, "attrs")
                        and "units" in field_var.attrs
                    ):
                        return field_var.attrs["units"]
                    elif field_var and hasattr(field_var, "units"):
                        return field_var.units

            return "n.a."
        except Exception as e:
            self.logger.warning(f"Error getting units: {e}")
            return "n.a."

    def set_cartopy_ticks(self, ax, extent, labelsize=10):
        """Add gridlines and tick labels to a Cartopy map."""
        import cartopy.crs as ccrs

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
                color="gray",
                alpha=0.6,
                linestyle="--",
            )

            gl.top_labels = False
            gl.bottom_labels = True
            gl.left_labels = True
            gl.right_labels = False

            gl.xlabel_style = {"size": labelsize, "rotation": 0}
            gl.ylabel_style = {"size": labelsize, "rotation": 0}

            return True
        except Exception as e:
            self.logger.error(f"Could not set ticks and labels: {e}")
            return False

    def _set_cartopy_ticks_alt(self, ax, extent, labelsize=10):
        """
        Adds gridlines and tick labels (in degrees) outside the map for Lambert and PlateCarree.
        Places longitude labels below the map, latitude on the left.
        """
        import numpy as np

        if not extent or len(extent) != 4:
            self.logger.warning(f"Invalid extent {extent}, using default")
            extent = [-180, 180, -90, 90]

        try:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        except Exception as e:
            self.logger.warning(f"Could not set extent: {e}")

        try:
            xticks_deg = np.arange(extent[0], extent[1] + 1, 10)
            yticks_deg = np.arange(extent[2], extent[3] + 1, 10)

            # Use Cartopy's gridlines just for visual grid, no labels
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=False,
                linewidth=0.8,
                color="gray",
                alpha=0.6,
                linestyle="--",
            )
            gl.xlocator = FixedLocator(xticks_deg)
            gl.ylocator = FixedLocator(yticks_deg)

            # Compute projected x/y positions of tick lines
            x_tick_positions = []
            for lon in xticks_deg:
                try:
                    x, _ = ax.projection.transform_point(
                        lon, extent[2], ccrs.PlateCarree()
                    )
                    x_tick_positions.append(x)
                except:
                    continue

            y_tick_positions = []
            for lat in yticks_deg:
                try:
                    _, y = ax.projection.transform_point(
                        extent[0], lat, ccrs.PlateCarree()
                    )
                    y_tick_positions.append(y)
                except:
                    continue

            # Now map projected positions to geographic labels using the original values
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels([f"{lon}°" for lon in xticks_deg], fontsize=labelsize)
            ax.tick_params(axis="x", direction="out", pad=5)

            ax.set_yticks(y_tick_positions)
            ax.set_yticklabels([f"{lat}°" for lat in yticks_deg], fontsize=labelsize)
            ax.tick_params(axis="y", direction="out", pad=5)

            return True

        except Exception as e:
            self.logger.error(f"Could not set ticks and labels: {e}")
            return False

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
            # If figure is not registered with pyplot, register it
            if plt.fignum_exists(self.fig.number if hasattr(self.fig, "number") else 1):
                plt.figure(self.fig.number)
            else:
                # For custom Figure classes that aren't managed by pyplot
                try:
                    # Try to show the figure directly if it has a show method
                    if hasattr(self.fig, "show_eviz"):
                        self.fig.show_eviz()
                    elif hasattr(self.fig, "show"):
                        self.fig.show()
                    else:
                        # Fall back to pyplot.show() which will show all figures
                        plt.show()
                except Exception as e:
                    self.logger.error(f"Error showing figure: {e}")
                    # Last resort: just call plt.show() to display any figures
                    plt.show()
        else:
            self.logger.warning("No figure to show")

    @staticmethod
    def _legend_font_size(subplots):
        """Determine appropriate font size for legends based on subplot layout."""
        return pu.legend_font_size(subplots)

    @staticmethod
    def _image_font_size(subplots):
        """Get appropriate font size based on subplot layout."""
        return pu.image_font_size(subplots)

    @staticmethod
    def _add_logo_ax(fig, desired_width_ratio=0.05):
        """Add a logo to the figure."""
        return pu.add_logo_ax(fig, desired_width_ratio)
