import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import logging
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from eviz.lib.autoviz.utils import FlexibleOOMFormatter, OOMFormatter
from eviz.lib.autoviz.plotting.base import BasePlotter
import eviz.lib.autoviz.utils as pu
from eviz.lib.autoviz.utils import bar_font_size, contour_tick_font_size


DEFAULT_CONTOUR_LABELSIZE = 12
DEFAULT_COLORBAR_LABELSIZE = 10
DEFAULT_COLORBAR_TICKFORMAT = "%.1f"


class MatplotlibBasePlotter(BasePlotter):
    """Base class for all Matplotlib plotters with common functionality."""

    def __init__(self):
        super().__init__()
        self.fig = None
        self.ax = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def plot(self, config, data_to_plot):
        pass

    def filled_contours(
        self, config, field_name, ax, x, y, data2d, transform=None, vmin=None, vmax=None
    ):
        """Plot filled contours."""
        # Check if data is all NaN
        if np.isnan(data2d).all():
            self.logger.warning(
                f"All values are NaN for {field_name}. Cannot create contour plot."
            )
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

        # Check if field was marked as constant during level creation
        if config.ax_opts.get("is_constant_field", False):
            self.logger.debug("Rendering constant field with neutral color and text")
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

        if config.compare:
            cmap_str = config.ax_opts["use_diff_cmap"]
        else:
            cmap_str = config.ax_opts["use_cmap"]

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
        self.logger.debug(f"Create contour levels for {field_name}")
        # Check if clevs already exists and is not empty
        if (
            "clevs" in ax_opts
            and ax_opts["clevs"] is not None
            and len(ax_opts["clevs"]) > 0
        ):
            return

        if np.isnan(data2d).all():
            self.logger.warning("All values are NaN! Cannot create contour levels.")
            # Set default contour levels to avoid errors
            ax_opts["clevs"] = np.array([0, 1])
            ax_opts["clevs_prec"] = 0
            return

        if vmin is not None and vmax is not None:
            dmin, dmax = vmin, vmax
        else:
            # Get min/max values, skipping NaN values
            dmin = np.nanmin(data2d)
            dmax = np.nanmax(data2d)
        self.logger.debug(f"dmin: {dmin}, dmax: {dmax}")

        # Use a single consistent threshold
        variation_threshold = float(ax_opts.get("variation_threshold", 1e-12))
        data_range = np.abs(dmax - dmin)

        self.logger.debug(f"range: {data_range}, threshold: {variation_threshold}")

        # Determine if field should be treated as constant based on relative variation
        max_abs_value = max(np.abs(dmin), np.abs(dmax))

        # For very small values, use absolute threshold
        # For larger values, use relative threshold (e.g., 0.01% of the maximum absolute value)
        if max_abs_value < 1e-10:
            # Very small values - use absolute threshold
            is_constant = data_range < variation_threshold
        else:
            # Larger values - use relative threshold
            relative_threshold = max(variation_threshold, 1e-6 * max_abs_value)
            is_constant = data_range < relative_threshold

        if is_constant:
            self.logger.debug(
                "Field variation below threshold — will be treated as constant in plotting."
            )
            # Set a flag to indicate this should be rendered as constant field
            ax_opts["is_constant_field"] = True
            # Still create minimal contour levels to avoid errors
            center = (dmin + dmax) / 2
            ax_opts["clevs"] = np.array(
                [center - variation_threshold, center, center + variation_threshold]
            )
            ax_opts["clevs_prec"] = max(
                0, int(-np.floor(np.log10(variation_threshold)))
            )
            return

        # Normal case - create proper contour levels
        ax_opts["is_constant_field"] = False

        # Calculate appropriate precision
        range_val = data_range
        precision = max(0, int(np.ceil(-np.log10(range_val)))) if range_val != 0 else 6
        if 1.0 <= range_val <= 9.0:
            precision = 1
        elif 0.1 <= range_val < 1.0:
            precision = 2

        ax_opts["clevs_prec"] = precision
        self.logger.debug(f"range_val: {range_val}, precision: {precision}")

        # Create contour levels
        num_levels = int(ax_opts.get("num_clevs", 10))
        if not ax_opts.get("create_clevs", True):
            clevs = np.around(np.linspace(dmin, dmax, 10), decimals=precision)
        else:
            clevs = np.around(np.linspace(dmin, dmax, num_levels), decimals=precision)
            clevs = np.unique(clevs)  # Remove duplicates

        # Check if levels are strictly increasing
        if len(set(clevs)) <= 2:
            self.logger.debug("Not enough unique contour levels.")
            # Try with more levels and higher precision
            clevs = np.linspace(dmin, dmax, 10)
            clevs = np.unique(np.around(clevs, decimals=6))
            if len(clevs) <= 2:
                # As a last resort, just use [dmin, dmax]
                clevs = np.array([dmin, dmax])

        # Ensure strictly increasing
        clevs = np.unique(clevs)
        ax_opts["clevs"] = clevs

        self.logger.debug(f"Created contour levels: {ax_opts['clevs']}")
        if ax_opts["clevs"][0] == 0.0:
            ax_opts["extend_value"] = "max"

    def line_contours(self, fig, ax, ax_opts, x, y, data2d, transform=None):
        """Add line contours to the plot."""
        with mpl.rc_context(rc=ax_opts.get("rc_params", {})):
            try:
                # Check if clevs exists and has enough levels
                if (
                    "clevs" not in ax_opts
                    or ax_opts["clevs"] is None
                    or len(ax_opts["clevs"]) < 2
                ):
                    return

                try:
                    formatted_clevs = pu.formatted_contours(ax_opts["clevs"])
                    contour_format = pu.contour_format_from_levels(
                        formatted_clevs, scale=ax_opts.get("cscale", None)
                    )
                except IndexError:
                    # Handle the case where contour_format_from_levels fails
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
                    return
                self.clabel_with_default_fontsize(
                    ax, clines, fmt=contour_format, fontsize=8
                )
            except Exception as e:
                self.logger.error(f"Error adding contour lines: {e}")

    def set_colorbar(
        self, config, cfilled, fig, ax, ax_opts, findex, field_name, data2d
    ):
        """Add a colorbar to the plot."""
        self.logger.debug(f"Create colorbar for {field_name}")
        try:
            # Skip colorbar creation if suppressed (for shared colorbar)
            if ax_opts.get("suppress_colorbar", False):
                return None

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

            if ax_opts["clabel"] is None:
                cbar_label = self.units
            else:
                cbar_label = ax_opts["clabel"]
            self.style_colorbar(
                cbar, ax_opts, data2d, fmt=fmt, fontsize=8, label=cbar_label
            )

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
        self.logger.debug(f"Adding shared colorbar for {field_name}")

        # First, check if we already have a shared colorbar and remove it
        if hasattr(fig, "_shared_colorbar_ax") and fig._shared_colorbar_ax in fig.axes:
            self.logger.debug("Removing existing shared colorbar")
            fig._shared_colorbar_ax.remove()

        # Filter out None values
        valid_contours = [c for c in cfilled_objects if c is not None]

        if not valid_contours:
            self.logger.warning("No valid contours for shared colorbar")
            return None

        self.logger.debug(
            f"Found {len(valid_contours)} valid contours for shared colorbar"
        )
        for i, contour in enumerate(valid_contours):
            self.logger.debug(f"Contour {i} clim: {contour.get_clim()}")

        # Get the min and max values across all contours for THIS field only
        vmin = min(c.get_clim()[0] for c in valid_contours)
        vmax = max(c.get_clim()[1] for c in valid_contours)

        self.logger.debug(f"Shared colorbar range for {field_name}: {vmin} to {vmax}")

        # Create a new axes for the colorbar
        colorbar_width = getattr(config, "colorbar_width", 0.02)
        cbar_ax = fig.add_axes([0.92, 0.15, colorbar_width, 0.7])

        fig._shared_colorbar_ax = cbar_ax

        # Create the colorbar using the first valid contour
        cbar = fig.colorbar(valid_contours[0], cax=cbar_ax)

        # Encompass all data for this field
        cbar.mappable.set_clim(vmin, vmax)

        tick_font_size = 8  # Fixed size for shared colorbar
        cbar.ax.tick_params(labelsize=tick_font_size)
        cbar.set_label(self.units, size=10)

        return cbar

    @staticmethod
    def set_const_colorbar(cfilled, fig, ax):
        _ = fig.colorbar(cfilled, ax=ax, shrink=0.5)

    def get_units(self, config, field_name, data2d, findex):
        """Get units for the field."""
        self.logger.debug(f"Get units for {field_name}")
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
            if self.source_name in config.readers:
                if isinstance(config.readers[self.source_name], dict):
                    readers_dict = config.readers[self.source_name]
                    if "NetCDF" in readers_dict:
                        reader = readers_dict["NetCDF"]
                    elif readers_dict:
                        reader = next(iter(readers_dict.values()))
                else:
                    reader = config.readers[self.source_name]

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

    def show(self):
        """Display the plot."""
        pass

    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        pass

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

    @staticmethod
    def clabel_with_default_fontsize_mpl(contour, **kwargs):
        """Label contours with a default font size if not specified (QuadContourSet)."""
        if "fontsize" not in kwargs:
            kwargs["fontsize"] = 12
        return contour.ax.clabel(contour, **kwargs)

    @staticmethod
    def clabel_with_default_fontsize(ax, contour, **kwargs):
        """Label contours with a default font size if not specified (GeoContourSet)."""
        kwargs.setdefault("fontsize", DEFAULT_CONTOUR_LABELSIZE)
        return ax.clabel(contour, **kwargs)

    @staticmethod
    def style_colorbar(cbar, ax_opts, data, fmt="%.1f", fontsize=8, label=None):
        """Style the colorbar with a given format and font size."""
        # Set tick label font size
        cbar.ax.tick_params(labelsize=fontsize)

        # Set the formatter properly
        if isinstance(fmt, str):
            formatter = FormatStrFormatter(fmt)
        else:
            formatter = FuncFormatter(fmt)

        cbar.ax.xaxis.set_major_formatter(formatter)
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.update_ticks()

        # Set label if provided
        if label:
            cbar.set_label(label, fontsize=fontsize)

        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        max_val = max(abs(vmin), abs(vmax))

        clevs = ax_opts.get("clevs", None)
        num_clevs = len(clevs) if clevs is not None else 0

        really_small_vals = max_val < 1e-3 or num_clevs > 10
        really_large_vals = max_val >= 1e3 or num_clevs > 10

        # Rotate labels for horizontal colorbars with really small/large values
        if really_small_vals or really_large_vals:
            if cbar.orientation == "horizontal":
                for label in cbar.ax.get_xticklabels():
                    label.set_rotation(45)

        # Add scientific notation if requested
        if ax_opts["cbar_sci_notation"]:
            cbar.ax.text(
                1.05,
                -0.5,
                r"$\times 10^{%d}$" % fmt.oom,
                transform=cbar.ax.transAxes,
                va="center",
                ha="left",
                fontsize=8,
            )
