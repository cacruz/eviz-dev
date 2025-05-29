import logging
from typing import Any, Dict
import matplotlib as mpl
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator

import numpy as np

from eviz.lib.autoviz.utils import get_subplot_geometry
import eviz.lib.autoviz.utils as pu


class Figure(mfigure.Figure):
    """
    Enhanced Figure class inheriting from matplotlib's Figure with eViz framework customizations.

    Parameters:
    config_manager (ConfigManager): Representation of the model configuration 
    plot_type (str): Type of plot to be created
    
    Attributes:
    - _rindex: Row index for multiple subplots
    - _ax_opts: Dictionary of axis options
    - _frame_params: Dictionary of frame parameters
    - _subplots: Tuple defining subplot layout
    - _use_cartopy: Flag to indicate use of Cartopy projection
    """
    def __init__(self, config_manager, plot_type, 
        *,
        nrows=None,
        ncols=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        %(figure.figure)s

        Other parameters
        ----------------
        %(figure.format)s
        **kwargs
            Passed to `matplotlib.figure.Figure`.

        See also
        --------
        matplotlib.figure.Figure
        """     
        self._gridspec = None
        self._panel_dict = {"left": [], "right": [], "bottom": [], "top": []}
        self._subplot_dict = {}  # subplots indexed by number
        self._subplot_counter = 0  # avoid add_subplot() returning an existing subplot
        self._projection = None
        self._subplots = (1, 1)

        # Initialize eViz-specific attributes
        self.config_manager = config_manager
        self.plot_type = plot_type
        self._logger = logging.getLogger(__name__)
        
        # Initialization defaults
        self._rindex = 0
        self._ax_opts = {}
        self._frame_params = {}
        
        # If nrows and ncols are provided, use them to set _subplots
        if nrows is not None and ncols is not None:
            self._subplots = (nrows, ncols)
            
        self._use_cartopy = False
        self.gs = None
        self.axes_array = []

        # Remove nrows and ncols from kwargs to avoid passing them to matplotlib.figure.Figure
        if 'nrows' in kwargs:
            del kwargs['nrows']
        if 'ncols' in kwargs:
            del kwargs['ncols']
            
        super().__init__(**kwargs)
        
        # Post-initialization setup
        self._logger.debug("Create figure, axes")
        
        if self.config_manager.add_logo:
            self.EVIZ_LOGO = plt.imread('eviz/lib/_static/ASTG_logo.png')
        
        self._init_frame()

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def _init_frame(self):
        """Set shape and size for pre-defined figure frames."""
        self._set_compare_diff_subplots()
        _frame_params = {}
        rindex = 0
        _frame_params[rindex] = list()
        
        if self.config_manager.compare and not self.config_manager.compare_diff:
            # Side-by-side comparison
            if self._subplots[1] == 3:
                # [nrows, ncols, width, height] - wider for 3 columns
                _frame_params[rindex] = [1, 3, 18, 6]
            else:
                _frame_params[rindex] = [1, 2, 12, 6]  # Original 2-column layout
        elif self.config_manager.compare_diff:
            # Comparison with difference
            if self._subplots == (3, 1):
                _frame_params[rindex] = [3, 1, 8, 12]
            elif self._subplots == (2, 2):
                _frame_params[rindex] = [2, 2, 12, 8]
        else:
            # Single plots - keep original logic
            if self._subplots == (1, 1):
                _frame_params[rindex] = [1, 1, None, None]
            elif self._subplots == (2, 1):
                _frame_params[rindex] = [2, 1, 8, 10]
            elif self._subplots == (3, 1):
                _frame_params[rindex] = [3, 1, 8, 12]
            elif self._subplots == (2, 2):
                _frame_params[rindex] = [2, 2, 14, 10]
            elif self._subplots == (3, 4):
                _frame_params[rindex] = [3, 4, 12, 16]
                
        self._frame_params = _frame_params

    def _set_compare_diff_subplots(self):
        """Set subplots for comparison plots."""
        try:
            # Handle simple side-by-side comparison
            if self.config_manager.compare and not self.config_manager.compare_diff:
                # Get the number of variables to compare from the config
                if hasattr(self.config_manager, 'compare_exp_ids'):
                    num_vars = len(self.config_manager.compare_exp_ids)
                    self._subplots = (1, num_vars)
                else:
                    self._subplots = (1, 2)  # Default to side by side layout
                return
                
            # Handle comparison with difference plots
            extra_diff_plot = self.config_manager.extra_diff_plot
            if not self.config_manager.compare_diff:
                extra_diff_plot = False
                
            if self.config_manager.spec_data and extra_diff_plot:
                self._subplots = (2, 2)
            elif self.config_manager.compare_diff:
                self._subplots = (3, 1)
            else:
                self._subplots = (1, 1)  # fallback for single plots
                
        except Exception as e:
            self.logger.warning(f"Error setting subplot layout: {str(e)}, using default")
            self._subplots = (1, 1)

    def set_axes(self):
        """
        Set figure axes objects based on required subplots.

        Returns:
            tuple: (figure, axes) objects for the given plot type
        """
        if 'tx' in self.plot_type or 'sc' in self.plot_type or 'xy' in self.plot_type:
            self._use_cartopy = True

        self.create_subplot_grid()
        self.create_subplots()

        return self

    def reset_axes(self, ax):
        """Remove all plotted data, colorbars, and titles from either a Matplotlib Axes
        or Cartopy GeoAxes.
        """
        if self is None:
            raise ValueError("Figure is None! It may have been closed or deleted.")

        for self.artist in ax.lines + ax.collections + ax.patches + ax.images:
            self.artist.remove()

        # Cartopy GeoAxes
        if hasattr(ax, "coastlines"):  
            ax.cla()  

        colorbars = [cbar_ax for cbar_ax in self.axes if cbar_ax is not ax]
        for cbar_ax in colorbars:
            if "colorbar" in str(cbar_ax):
                self.delaxes(cbar_ax)  

        ax.set_title("")

        # apply changes
        self.canvas.draw_idle()

    def _get_fig_ax(self):
        """
        Initialize figure and axes objects for all plots based on plot type.

        Returns:
            tuple: (figure, axes) objects for the given plot type
        """
        if "po" in self.plot_type:
            return self

        if 'tx' in self.plot_type or 'sc' in self.plot_type or 'xy' in self.plot_type:
            self._use_cartopy = True

        self.create_subplot_grid()
        self.create_subplots()

        return self
    
    def get_fig_ax(self):
        return self._get_fig_ax()

    def get_axes(self):
        # Always return a list of axes, even for a single axes
        return self.axes_array

    def create_subplot_grid(self):
        """Create a grid of subplots based on the figure frame layout."""
        # Hack to distinguish regional plots, which look better in square aspect ratio
        if ('tx' in self.plot_type or 'sc' in self.plot_type or 'xy' in self.plot_type) and  'extent' in self._ax_opts:
            if self._ax_opts['extent'] != [-180, 180, -90, 90]:
                self._frame_params[self._rindex][2] = 8
                self._frame_params[self._rindex][3] = 8
        if self._frame_params[self._rindex][2] and self._frame_params[self._rindex][3]:
            figsize = (self._frame_params[self._rindex][2], self._frame_params[self._rindex][3])
            self.set_size_inches(figsize)
        
        # Create GridSpec with appropriate spacing for side-by-side plots
        if self.config_manager.compare and not self.config_manager.compare_diff:
            # Adjust spacing based on number of columns
            if self._subplots[1] > 2:
                self.gs = gridspec.GridSpec(*self._subplots, wspace=0.25)  # Slightly tighter spacing for 3+ columns
            else:
                self.gs = gridspec.GridSpec(*self._subplots, wspace=0.3)  # Original spacing for 2 columns
        else:
            self.gs = gridspec.GridSpec(*self._subplots)
            
        return self

    def create_subplots(self):
        """
        Create subplots based on the gridspec (subplot grid) and projection requirements.
        """
        if self.use_cartopy:
            return self._create_subplots_crs()
        else:
            for i in range(self._subplots[0]):
                for j in range(self._subplots[1]):
                    ax = self.add_subplot(self.gs[i, j])
                    self.axes_array.append(ax)
            return self

    def _create_subplots_crs(self):
        """Create subplots with cartopy projections."""
        # Determine the projection to use
        map_projection = None
        # Check if we have a field_name and can get projection from spec_data
        if hasattr(self, 'field_name') and self.field_name:
            if (self.config_manager.spec_data and
                self.field_name in self.config_manager.spec_data):
                # Check for projection at the top level of the field spec
                if 'projection' in self.config_manager.spec_data[self.field_name]:
                    projection_name = self.config_manager.spec_data[self.field_name]['projection']
                    map_projection = self.get_projection(projection_name)
                    self._logger.debug(f"Using projection '{projection_name}' for field {self.field_name}")
                # Also check in the plot-type specific section
                elif 'projection' in self.config_manager.spec_data[self.field_name].get(f"{self.plot_type[:2]}plot", {}):
                    projection_name = self.config_manager.spec_data[self.field_name][f"{self.plot_type[:2]}plot"]['projection']
                    map_projection = self.get_projection(projection_name)
                    self._logger.debug(f"Using projection '{projection_name}' for field {self.field_name}")

        # If no projection found from field_name, check ax_opts
        if map_projection is None and 'projection' in self._ax_opts:
            map_projection = self.get_projection(self._ax_opts['projection'])

        # Default to PlateCarree if no projection specified
        if map_projection is None:
            map_projection = self.get_projection()

        for i in range(self._subplots[0]):
            for j in range(self._subplots[1]):
                ax = self.add_subplot(self.gs[i, j], projection=map_projection)
                self.axes_array.append(ax)

        for ax in self.axes_array:
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND, edgecolor='black')
            ax.add_feature(cfeature.LAKES, edgecolor='black')

        return self

    @staticmethod
    def create_eviz_figure(config_manager, plot_type, nrows=None, ncols=None):
        """
        Factory method to create an eViz Figure instance.
        
        Args:
            config_manager (ConfigManager): Configuration manager
            plot_type (str): Type of plot
            nrows (int, optional): Number of rows in the subplot grid
            ncols (int, optional): Number of columns in the subplot grid
        
        Returns:
            Figure: An instance of the eViz Figure class
        """
        # If nrows and ncols are not provided, determine them based on the configuration
        if nrows is None or ncols is None:
            if config_manager.compare and not config_manager.compare_diff:
                # For side-by-side comparison, use 1x2 layout
                nrows, ncols = 1, 2
            elif config_manager.compare_diff:
                # For comparison with difference, use layout from config
                nrows, ncols = config_manager.input_config._comp_panels
            else:
                # For single plots, use 1x1 layout
                nrows, ncols = 1, 1
        
        return Figure(config_manager, plot_type, nrows=nrows, ncols=ncols)

    def get_gs_geometry(self):
        if self.gs:
            return self.gs.get_geometry()
        else:
            return None

    def have_multiple_axes(self):
        return self.axes is not None and (self.axes.numRows > 1 or self.axes.numCols > 1)

    def have_nontrivial_grid(self):
        return self.gs.nrows > 1 or self.gs.ncols > 1

    def _set_fig_axes_regional(self, use_cartopy_opt):
        pass

    def savefig_eviz(self, *args, **kwargs):
        # Custom savefig behavior
        super().savefig(*args, **kwargs)
        # Do more custom stuff

    def show_eviz(self, *args, **kwargs):
        """
        Display the figure with any custom processing.
        Change to show() to override overrides matplotlib's plt.show() with 
        custom behavior if needed.
        """
        # Any custom pre-show processing
        
        # Call the parent method or use plt.show() if needed
        plt.figure(self.number)  # Make sure this figure is active
        plt.show(*args, **kwargs)
        # Any custom post-show processing

    def get_projection(self, projection=None):
        """Get projection parameter."""
        # Default values for extent and central coordinates
        extent = [-180, 180, -90, 90]  # global default
        central_lon = 0.0
        central_lat = 0.0

        # Try to get extent from config_manager.ax_opts
        if hasattr(self.config_manager, 'ax_opts') and self.config_manager.ax_opts:
            if 'extent' in self.config_manager.ax_opts:
                extent = self.config_manager.ax_opts['extent']
            if 'central_lon' in self.config_manager.ax_opts:
                central_lon = self.config_manager.ax_opts['central_lon']
            if 'central_lat' in self.config_manager.ax_opts:
                central_lat = self.config_manager.ax_opts['central_lat']
        # Also check in _ax_opts
        elif 'extent' in self._ax_opts:
            if self._ax_opts['extent'.lower()] == 'conus':
                extent = [-120, -70, 24, 50.5]
            else:
                extent = self._ax_opts['extent']
            # Calculate central coordinates from extent if not provided
            central_lon = np.mean(extent[:2])
            central_lat = np.mean(extent[2:])

        if projection is None:
            self._ax_opts['extent'] = extent
            self._projection = ccrs.PlateCarree()
            return ccrs.PlateCarree()
        
        options = {
            'mercator': ccrs.Mercator(
                central_longitude=central_lon,
            ),
            'robinson': ccrs.Robinson(
                central_longitude=central_lon,
            ),
            'orthographic': ccrs.Orthographic(
                central_longitude=central_lon,
                central_latitude=central_lat,
            ),
            'mollweide': ccrs.Mollweide(
                central_longitude=central_lon,
            ),
            'lambert': ccrs.LambertConformal(
                central_longitude=central_lon,
                central_latitude=central_lat,
                standard_parallels=(extent[2], extent[3])
            ),
            'albers': ccrs.AlbersEqualArea(
                central_longitude=central_lon,
                central_latitude=central_lat,
                standard_parallels=(extent[2], extent[3])
            ),
            'stereo': ccrs.Stereographic(
                central_latitude=central_lat,
                central_longitude=central_lon
            ),
            'ortho': ccrs.Orthographic(
                central_latitude=central_lat,
                central_longitude=central_lon
            ),
            'polar': ccrs.NorthPolarStereo(central_longitude=central_lon),
        }
        self._ax_opts['extent'] = extent
        self._projection = options.get(projection)

        return self._projection

    def set_ax_opts_diff_field(self, ax):
        """ Modify axes internal state based on user-defined options

        Note:
            Only relevant for comparison plots.
        """
        geom = pu.get_subplot_geometry(ax)
        if geom[0] == (3, 1) and geom[1:] == (0, 1, 1, 1):
            self._ax_opts['is_diff_field'] = True
        if geom[0] == (2, 2):
            if geom[1:] == (0, 1, 1, 0):
                self._ax_opts['is_diff_field'] = True
            if geom[1:] == (0, 0, 1, 1):
                self._ax_opts['is_diff_field'] = True
                self._ax_opts['add_extra_field_type'] = True

    def init_ax_opts(self, field_name):
        """Initialize map options for a given field."""
        plot_type = "polar" if self.plot_type.startswith("po") else self.plot_type[:2]
        spec = self.config_manager.spec_data.get(field_name, {}).get(f"{plot_type}plot", {})
        defaults = {
            'rc_params': None,
            'boundary': None,
            'use_pole': 'north',
            'profile_dim': None,
            'zsum': None,
            'zave': None,
            'tave': None,
            'taverange': 'all',
            'cmap_set_over': None,
            'cmap_set_under': None,
            'use_cmap': self.config_manager.input_config._cmap,
            'use_diff_cmap': self.config_manager.input_config._cmap,
            'cscale': None,
            'zscale': 'linear',
            'cbar_sci_notation': False,
            'custom_title': False,
            'add_grid': False,
            'line_contours': True,
            'add_tropp_height': False,
            'torder': None,
            'add_trend': False,
            'projection': None,
            'num_clevs': 10,
            'time_lev': 0,
            'is_diff_field': False,
            'add_extra_field_type': False,
            'clabel': None,
            'create_clevs': False,
            'clevs_prec': 0,
            'clevs': None,
            'plot_title': None,
            'extend_value': 'both',
            'norm': 'both',
            'overlay': False,
            'contour_linestyle': {
                'lines.linewidth': 0.5,
                'lines.linestyle': 'solid'
            },
            'time_series_plot_linestyle': {
                'lines.linewidth': 1,
                'lines.linestyle': 'solid'
            },
            'colorbar_fontsize': {
                'colorbar.fontsize': 8
            },
            'axes_fontsize': {
                'axes.fontsize': 10
            },
            'title_fontsize': {
                'title.fontsize': 10
            },
            'subplot_title_fontsize': {
                'subplot_title.fontsize': 12
            }
        }
        self._ax_opts = {key: spec.get(key, defaults[key]) for key in defaults}
        
        # Merge rc_params from YAML if present
        rc_params_from_yaml = spec.get('rc_params', {})
        rc_keys = set(mpl.rcParams.keys())
        # Filter for valid rcParams
        filtered_rc_params = {k: v for k, v in rc_params_from_yaml.items() if k in rc_keys}
        # Merge with any rcParams already in _ax_opts
        if self._ax_opts.get('rc_params'):
            self._ax_opts['rc_params'].update(filtered_rc_params)
        else:
            self._ax_opts['rc_params'] = filtered_rc_params        

        return self._ax_opts

    @staticmethod
    def add_grid(ax, lines=True, locations=None):
        """Add a grid to the plot."""
        if lines:
            ax.grid(lines, alpha=0.5, which="minor", ls=":")
            ax.grid(lines, alpha=0.7, which="major")
        if locations:
            assert len(locations) == 4, "Invalid grid locations"
            ax.xaxis.set_minor_locator(MultipleLocator(locations[0]))
            ax.xaxis.set_major_locator(MultipleLocator(locations[1]))
            ax.yaxis.set_minor_locator(MultipleLocator(locations[2]))
            ax.yaxis.set_major_locator(MultipleLocator(locations[3]))

    def colorbar_eviz(self, mappable):
        """
        Create a colorbar
        https://joseph-long.com/writing/colorbars/
        """
        last_axes = plt.gca()
        ax = mappable.axes
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = self.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar

    @property
    def projection(self) -> ccrs.Projection:
        return self._projection

    @property
    def frame_params(self):
        return self._frame_params

    @property
    def subplots(self):
        return self._subplots

    @property
    def use_cartopy(self):
        return self._use_cartopy

    @property
    def ax_opts(self):
        return self._ax_opts

    def update_ax_opts(self, field_name, ax, pid, level=None):
        """ Set (or reset) some map options

        Parameters:
            field_name (str) : Name of field that needs axes options updated
            ax (Axes) : Axes object
            pid (str) : Plot type identifier
            level (int) : Vertical level (optional, default=None)

        Returns:
            Updated axes internal state
        """
        if not self.config_manager.compare or not self.config_manager.compare_diff:
            return self._update_single_plot(field_name, pid, level)

        geom = get_subplot_geometry(ax)
        if self._subplots == (3, 1) and geom[1:] == (0, 1, 1, 1):
            self._ax_opts['line_contours'] = False
            self._set_clevs(field_name, f"{pid}plot",
                            f"diff_{level}" if level is not None else "diffcontours")
        elif self._subplots == (2, 2) and geom[1:] == (0, 1, 1, 0):
            self._set_clevs(field_name, f"{pid}plot",
                            f"diff_{level}" if level is not None else "diffcontours")
        elif self._subplots == (2, 2) and geom[1:] == (0, 0, 1, 1):
            self._ax_opts['line_contours'] = False
            diff_opt = f"diff_{self.config_manager.extra_diff_plot}"
            self._ax_opts['clevs'] = self.config_manager.yaml_parser.spec_data[field_name].get(diff_opt, None)
        else:
            self._set_clevs(field_name, f"{pid}plot",
                            level if isinstance(level, int) else "contours")

        # Optionally, update rc_params if new ones are found in the spec
        plot_type = "polar" if self.plot_type.startswith("po") else self.plot_type[:2]
        self.config_manager.spec_data.get(field_name, {}).get(f"{plot_type}plot", {})

        return self._ax_opts
    
    def _update_single_plot(self, field_name, pid, level):
        """Update axes options for single subplot case."""
        plot_type_map = {
            'yz': 'yzplot', 'yzave': 'yzaveplot', 'xy': 'xyplot',
            'xyave': 'xyaveplot', 'tx': 'txplot', 'polar': 'polarplot'
        }
        plot_key = plot_type_map.get(pid, None)
        if plot_key:
            self._set_clevs(field_name, plot_key,
                            level if isinstance(level, int) else "contours")
        return self._ax_opts

    def _set_clevs(self, field_name, ptype, ctype):
        """ Helper function for update_ax_opts(): sets contour levels """
        if isinstance(ctype, int):
            if ctype in self.config_manager.spec_data[field_name][ptype]['levels']:
                self._ax_opts['clevs'] = self.config_manager.spec_data[field_name][ptype]['levels'][ctype]
                if not self._ax_opts['clevs']:
                    self._ax_opts['create_clevs'] = True
            else:
                self._ax_opts['create_clevs'] = True

        else:
            if ctype in self.config_manager.spec_data[field_name][ptype]:
                self._ax_opts['clevs'] = self.config_manager.spec_data[field_name][ptype][ctype]
                if not self._ax_opts['clevs']:
                    self._ax_opts['create_clevs'] = True
            else:
                self._ax_opts['create_clevs'] = True

    def plot_text(self, field_name, ax, pid, level=None, data=None, *args, **kwargs):
        """Add text to a map.

        Parameters:
            field_name (str): Name of the field
            ax (Axes or list of Axes): Axes object(s)
            pid (str): Plot type identifier
            level (int): Vertical level (optional, default=None)
            data (Any): xarray Data for basic stats (optional)
            *args: Additional positional arguments for customization
            **kwargs: Additional keyword arguments for customization
        """
        if isinstance(ax, list):  # Check if ax is a list
            for single_ax in ax:
                self._plot_text(field_name, single_ax, pid, level, data, **kwargs)
        else:
            self._plot_text(field_name, ax, pid, level, data, **kwargs)

    def _plot_text(self, field_name, ax, pid, level=None, data=None, **kwargs):
        """Add text to a single axes."""
        fontsize = kwargs.get('fontsize', pu.subplot_title_font_size(self._subplots))
        loc = kwargs.get('location', 'left')

        findex = self.config_manager.findex
        sname = self.config_manager.config.map_params[findex]['source_name']
        geom = pu.get_subplot_geometry(ax) if self.config_manager.compare or self.config_manager.compare_diff else None

        # Handle plot titles for comparison cases
        if self.config_manager.compare or self.config_manager.compare_diff:
            if geom and geom[0] == (3, 1):  # (3,1) subplot structure
                if geom[1:] == (0, 1, 1, 1):  # Bottom plot
                    title_string = "Difference (top - middle)"
                elif geom[1:] in [(1, 1, 0, 1), (0, 1, 0, 1)]:  # Top/Middle plots
                    title_string = self._set_axes_title(findex)
            elif self._subplots == (2, 2):  # (2,2) subplot structure
                if geom[1:] == (0, 1, 1, 0):
                    title_string = "Difference (left - right)"
                elif geom[1:] == (0, 0, 1, 1):  # Extra diff plot
                    diff_labels = {
                        "percd": ("% Diff", "%"),
                        "percc": ("% Change", "%"),
                        "ratio": ("Ratio Diff", "ratio"),
                    }
                    diff_type = self.config_manager.extra_diff_plot
                    title_string, self._ax_opts['clabel'] = diff_labels.get(
                        diff_type, ("Difference (left - right)", None))
                    self._ax_opts['line_contours'] = False
                else:  # Default case
                    title_string = self._set_axes_title(findex)
            elif geom and (geom[0] == (1, 2) or geom[0] == (1, 3)):
                title_string = self._set_axes_title(findex)
            else:  # Default title for comparison
                title_string = 'Placeholder'
            ax.set_title(title_string, loc=loc, fontsize=fontsize)
            return

        # Non-comparison case
        level_text = self._format_level_text(level)
        name = self._get_field_name(field_name, sname, findex)

        left, width = 0, 1.0
        bottom, height = 0, 1.0
        right = left + width
        top = bottom + height
        title_string = self._set_axes_title(findex)

        if 'yz' in pid:
            if self.config_manager.print_basic_stats:
                # plt.rc('text', usetex=True)
                fmt = self._basic_stats(data)
                ax.text(right, top, fmt, transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=10)

            if self.config_manager.use_history:
                ax.set_title(self.config_manager.history_expid + " (" + self.config_manager.history_expdsc + ")")
            else:
                ax.set_title(title_string, loc=loc, fontsize=8)

            ax.text(0.5 * (left + right), bottom + top + 0.1,
                    name, fontweight='bold',
                    fontstyle='italic',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    transform=ax.transAxes)

        elif 'xy' in pid:
            if self.config_manager.print_basic_stats:
                fmt = self._basic_stats(data)
                ax.text(right, top, fmt, transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=10)
                loc = 'left'

            if self.config_manager.real_time and not self.config_manager.print_basic_stats:
                ax.text(right, top, self.config_manager.real_time,
                        ha='right', va='bottom', fontsize=10,
                        transform=ax.transAxes)
            if self.config_manager.use_history:
                ax.set_title(self.config_manager.history_expid + " (" + self.config_manager.history_expdsc + ")", fontsize=10)
            else:
                ax.set_title(title_string, loc=loc, fontsize=8)

            ax.text(0.5 * (left + right), bottom + top + 0.1,
                    name + level_text, 
                    fontweight=kwargs.get('fontweight', 'bold'),
                    fontstyle=kwargs.get('fontstyle', 'italic'),
                    fontsize=kwargs.get('fontsize', 14),
                    horizontalalignment=kwargs.get('ha', 'center'),
                    verticalalignment=kwargs.get('va', 'center'),
                    transform=ax.transAxes)

        elif 'tx' in pid:
            if self.config_manager.use_history:
                ax.set_title(self.config_manager.history_expid + " (" + self.config_manager.history_expdsc + ")", fontsize=10)
            else:
                ax.set_title(title_string, loc=kwargs.get('loc', 'right'), fontsize=kwargs.get('fontsize', 10))

            ax.text(0.5 * (left + right), bottom + top + 0.5,
                    name,
                    fontweight=kwargs.get('fontweight', 'bold'),
                    fontstyle=kwargs.get('fontstyle', 'normal'),
                    fontsize=kwargs.get('fontsize', 12),
                    horizontalalignment=kwargs.get('ha', 'center'),
                    verticalalignment=kwargs.get('va', 'center'),
                    transform=ax.transAxes)
        elif 'po' in pid:
            pass
        else:  # 'xt' and others
            if self.config_manager.use_history:
                ax.set_title(self.config_manager.history_expid + " (" + self.config_manager.history_expdsc + ")")
            else:
                ax.set_title(title_string, loc=loc, fontsize=fontsize)

            ax.text(0.5 * (left + right), bottom + top + 0.1,
                    name,
                    fontweight=kwargs.get('fontweight', 'bold'),
                    fontstyle=kwargs.get('fontstyle', 'italic'),
                    fontsize=kwargs.get('fontsize', 14),
                    horizontalalignment=kwargs.get('ha', 'center'),
                    verticalalignment=kwargs.get('va', 'center'),
                    transform=ax.transAxes)
        
    def _set_axes_title(self, findex):
        if self.config_manager.get_file_description(findex):
            return self.config_manager.get_file_description(findex)
        elif self.config_manager.get_file_exp_name(findex):
            return self.config_manager.get_file_exp_name(findex)
        else:
            if self.config_manager.ax_opts['custom_title']:
                return self.config_manager.ax_opts['custom_title']
        return None
        
    @staticmethod
    def _basic_stats(data):
        """ Basic stats for a given field """
        # datamin = data.min().values
        # datamax = data.max().values
        datamean = data.mean().values
        datastd = data.std().values
        return f"\nMean:{datamean:.2e}\nStd:{datastd:.2e}"

    def _format_level_text(self, level):
        """Format level annotation text based on level value."""
        if self.config_manager.ax_opts.get('zave'):
            return ' (Column Mean)'
        if self.config_manager.ax_opts.get('zsum'):
            return ' (Total Column)'
        if level is None or str(level) == '0':
            return ''
        return f"@ {level} {'Pa' if level > 10000 else 'mb'}"

    def _get_field_name(self, field_name, sname, findex):
        """Get the field name from the reader's dataset."""
        try:
            # First, use the field name from spec_data if available
            if field_name in self.config_manager.spec_data:
                if 'name' in self.config_manager.spec_data[field_name]:
                    return self.config_manager.spec_data[field_name]['name']
            
            # Try to get the name from the reader
            # First check if we're dealing with the new reader structure
            reader = None
            if sname in self.config_manager.readers:
                if isinstance(self.config_manager.readers[sname], dict):
                    # New structure - get the primary reader
                    readers_dict = self.config_manager.readers[sname]
                    if 'NetCDF' in readers_dict:
                        reader = readers_dict['NetCDF']
                    elif readers_dict:
                        reader = next(iter(readers_dict.values()))
                else:
                    # Old structure - direct access
                    reader = self.config_manager.readers[sname]
            
            # If we found a reader, try to get the field name
            if reader and hasattr(reader, 'datasets'):
                if findex in reader.datasets and 'vars' in reader.datasets[findex]:
                    var_attrs = reader.datasets[findex]['vars'][field_name].attrs
                    if 'long_name' in var_attrs:
                        return var_attrs['long_name']
                        
            # Default to the field name if we couldn't find anything better
            return field_name
            
        except (KeyError, AttributeError, IndexError) as e:
            self.logger.warning(f"Error getting field name for {field_name}: {e}")
            return field_name
 
    @staticmethod
    def get_default_plot_params() -> Dict[str, Any]:
        """
        Return default matplotlib plot parameters.

        Returns:
            dict: Default plot parameters.
        """
        return {
            'image.origin': 'lower',
            'image.interpolation': 'nearest',
            'image.cmap': 'gray',
            'axes.grid': False,
            'savefig.dpi': 150,
            'axes.labelsize': 10,
            'axes.titlesize': 14,
            'font.size': 10,
            'legend.fontsize': 6,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'font.family': 'sans-serif',
        }