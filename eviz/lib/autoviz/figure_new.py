import logging
import os
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator
from dataclasses import dataclass, field
from eviz.lib.autoviz.config import Config
from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.lib.autoviz.plot_utils import get_subplot_geometry, subplot_title_font_size
import eviz.lib.autoviz.plot_utils as pu
import matplotlib.gridspec as mgridspec
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

class Figure(matplotlib.figure.Figure):
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

    def __init__(self, config_manager, plot_type, *args, **kwargs):
        # Call the parent class's __init__ method
        super().__init__(*args, **kwargs)
        
        # Initialize eViz-specific attributes
        self.config_manager = config_manager
        self.plot_type = plot_type
        self._logger = logging.getLogger(__name__)
        
        # Initialization defaults
        self._rindex = 0
        self._ax_opts = {}
        self._frame_params = {}
        self._subplots = (1, 1)
        self._use_cartopy = False
        self.gs = None

        # Post-initialization setup
        self._logger.debug("Create figure, axes")
        
        if self.config_manager.add_logo:
            self.EVIZ_LOGO = plt.imread('eviz/lib/_static/ASTG_logo.png')
        
        self._init_frame()

        if self.config_manager.compare:
            # This might need adjustment based on your specific requirements
            self.axes = self._get_fig_ax()[1]

    # The rest of the methods remain largely the same as in the original class,
    # but are now methods of the class instead of static or separate methods

    def _init_frame(self):
        """ Get shape and geometry for each figure frame """
        self._init_subplots()
        # TODO: clean up
        _frame_params = {}
        rindex = 0
        _frame_params[rindex] = list()
        if self.config_manager.compare:
            if self._subplots == (3, 1):
                _frame_params[rindex] = [3, 1, 8, 12]  # nrows, ncols, width, height
            elif self._subplots == (2, 2):
                _frame_params[rindex] = [2, 2, 12, 8]
        else:
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

    def _init_subplots(self):
        """ Get subplots for each frame """
        _subplots = (1, 1)
        try:
            extra_diff_plot = self.config_manager.extra_diff_plot
            if not self.config_manager.compare:
                extra_diff_plot = False
            if self.config_manager.spec_data and extra_diff_plot:
                _subplots = (2, 2)
            else:
                _subplots = (1, 1)
        except Exception as e:
            self.logger.warning(f"key error: {str(e)}, returning default")
        finally:
            if self.config_manager.compare and not self.config_manager.extra_diff_plot:
                _subplots = (3, 1)
        self._subplots = _subplots

    def create_subplot_grid(self):
        if self._frame_params[self._rindex][2] and self._frame_params[self._rindex][3]:
            figsize = (self._frame_params[self._rindex][2], self._frame_params[self._rindex][3])
            self.set_size_inches(figsize)
        
        gs = gridspec.GridSpec(*self._subplots)
        return gs

    def create_subplots(self, gs):
        """
        Create subplots based on the gridspec and projection requirements.
        """
        if self._use_cartopy:
            return self._create_subplots_crs(gs)
        else:
            axes = []
            for i in range(self._subplots[0]):
                for j in range(self._subplots[1]):
                    ax = self.add_subplot(gs[i, j])
                    axes.append(ax)
            return axes

    def _create_subplots_crs(self, gs):
        axes = []
        if 'projection' in self._ax_opts:
            map_projection = self.get_projection(self._ax_opts['projection'])
        else:
            map_projection = ccrs.PlateCarree()

        for i in range(self._subplots[0]):
            for j in range(self._subplots[1]):
                ax = plt.subplot(gs[i, j], projection=map_projection)
                axes.append(ax)
                # Add gridlines
                gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.add_feature(cfeature.LAND, edgecolor='black')
                ax.add_feature(cfeature.LAKES, edgecolor='black')
        return axes


    # Optional factory method to create Figure instances
    def create_eviz_figure(config_manager, plot_type, *args, **kwargs):
        """
        Factory method to create an eViz Figure instance.
        
        Args:
            config_manager (ConfigManager): Configuration manager
            plot_type (str): Type of plot
            *args: Additional positional arguments for Figure
            **kwargs: Additional keyword arguments for Figure
        
        Returns:
            Figure: An instance of the eViz Figure class
        """
        return Figure(config_manager, plot_type, *args, **kwargs)    
    
    def create_subplot_grid0(self):
        if self._frame_params[self._rindex][2] and self._frame_params[self._rindex][3]:
            fig = plt.figure(figsize=(self._frame_params[self._rindex][2], self._frame_params[self._rindex][3]))
        else:
            fig = plt.figure()
        gs = gridspec.GridSpec(*self._subplots)
        return fig, gs

    def create_subplots0(self, gs):
        if self._use_cartopy:
            return self.create_subplots_crs(gs)
        else:
            axes = []
            for i in range(self._subplots[0]):
                for j in range(self._subplots[1]):
                    ax = plt.subplot(gs[i, j])
                    axes.append(ax)
            return axes

    def get_gs_geometry(self):
        if self.gs:
            return self.gs.get_geometry()
        else:
            return None

    def have_multiple_axes(self):
        return self.axes is not None and (self.axes.numRows > 1 or self.axes.numCols > 1)

    def have_nontrivial_grid(self):
        return self.gs.nrows > 1 or self.gs.ncols > 1

    @staticmethod
    def show():
        plt.show()

    def _get_fig_ax(self):
        """
        Initialize figure and axes objects for all plots based on plot type.

        Returns:
            tuple: (figure, axes) objects for the given plot type
        """
        # Most of the logic remains the same as the original method
        # Adjust as needed to work with the new class structure
        if 'yz' in self.plot_type or 'xt' in self.plot_type:
            self._use_cartopy = False
        elif 'tx' in self.plot_type or 'sc' in self.plot_type or 'xy' in self.plot_type:
            self._use_cartopy = True

        gs = self.create_subplot_grid()
        axtemp = self.create_subplots(gs)

        if isinstance(axtemp, list) and len(axtemp) == 1:
            axes = axtemp[0]
        else:
            axes = axtemp

        return self, axes
    
    # def _get_fig_ax(self):
    #     """ Initialize figure and axes objects for all plots based on plot type

    #     Returns:
    #         fig (Figure) : figure object for the given plot type
    #         ax (Axes) : Axes object for the given plot type
    #     """
    #     if 'yz' in self.plot_type or 'xt' in self.plot_type:
    #         fig, axtemp = self._set_fig_axes_global(use_cartopy_opt=False)
    #     elif 'tx' in self.plot_type or 'sc' in self.plot_type or 'xy' in self.plot_type:
    #         self._use_cartopy = True
    #         fig, axtemp = self._set_fig_axes_global(use_cartopy_opt=True)
    #     else:  # 'po'
    #         fig, axtemp = self._set_fig_axes_global(use_cartopy_opt=False)

    #     if isinstance(axtemp, list) and len(axtemp) == 1:
    #         ax = axtemp[0]
    #     else:
    #         ax = axtemp
    #     return fig, ax

    def get_fig_ax(self):
        return self._get_fig_ax()

    def _set_fig_axes_regional(self, use_cartopy_opt):
        pass

    def savefig_eviz(self, *args, **kwargs):
        # Custom savefig behavior
        result = super().savefig(*args, **kwargs)
        # Do more custom stuff
        return result
    
    def show_eviz(self, *args, **kwargs):
        """
        Display the figure with any custom processing.
        Change to show() to override overrides matplotlib's plt.show() with 
        custom behavior if needed.
        """
        # Any custom pre-show processing
        
        # Call the parent method or use plt.show() if needed
        plt.figure(self.number)  # Make sure this figure is active
        result = plt.show(*args, **kwargs)
        
        # Any custom post-show processing
    
        return result
    
    def _set_fig_axes_global(self, use_cartopy_opt):
        """ Helper function to instantiate figure and axes objects """
        # No longer need to create a new figure since self is the figure
        
        # Create GridSpec on the current figure
        gs = self.add_gridspec(*self._subplots)
        axes = self.create_subplots(gs)

        # Create axes based on the plot type
        if "tx" in self.plot_type:
            # Set figure size if specified
            if self._frame_params[self._rindex][2] and self._frame_params[self._rindex][3]:
                self.set_size_inches(self._frame_params[self._rindex][2], self._frame_params[self._rindex][3])
            
            gs = self.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 6], hspace=0.05)
            axes = []
            
            # Use self for figure reference
            map_projection = self.get_projection(self._ax_opts.get('projection', 'default'))
            axes.append(self.add_subplot(gs[0, 0], projection=map_projection))
            axes.append(self.add_subplot(gs[1, 0]))
        
        elif "xt" in self.plot_type:
            axes = plt.subplot(gs[0, 0])
        else:
            if self.config_manager.add_logo:
                if self._frame_params[self._rindex][2] and self._frame_params[self._rindex][3]:
                    self.set_figwidth(self._frame_params[self._rindex][2])
                    self.set_figheight(self._frame_params[self._rindex][3])
                gs = gridspec.GridSpec(2, 2, width_ratios=[1, 10], height_ratios=[10, 1], wspace=0.05, hspace=0.05)
                axes = gs.figure.axes[0]
                return self, axes
                    
        self.gs = gs
        self._ax_opts['use_cartopy'] = use_cartopy_opt
        return self, axes  # Return self as the figure

        # if "po" in self.plot_type:
        #     fig, ax = plt.subplots()
        #     return fig, ax
        # map_projection = ccrs.PlateCarree()
        # if 'projection' in self._ax_opts:
        #     map_projection = self.get_projection(self._ax_opts['projection'])

        # # Default
        # fig, gs = self.create_subplot_grid()
        # axes = self.create_subplots(gs)

        # if "tx" in self.plot_type:
        #     if self._frame_params[self._rindex][2] and self._frame_params[self._rindex][3]:
        #         fig = plt.figure(figsize=(self._frame_params[self._rindex][2], self._frame_params[self._rindex][3]))
        #     else:
        #         fig = plt.figure()
        #     gs = mgridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 6], hspace=0.05)
        #     axes = list()
        #     axes.append(fig.add_subplot(gs[0, 0], projection=map_projection))
        #     axes.append((fig.add_subplot(gs[1, 0])))
        # elif "xt" in self.plot_type:
        #     axes = plt.subplot(gs[0, 0])
        # else:
        #     if self.config_manager.add_logo:
        #         fig = plt.figure()
        #         if self._frame_params[self._rindex][2] and self._frame_params[self._rindex][3]:
        #             fig.set_figwidth(self._frame_params[self._rindex][2])
        #             fig.set_figheight(self._frame_params[self._rindex][3])
        #         gs = gridspec.GridSpec(2, 2, width_ratios=[1, 10], height_ratios=[10, 1], wspace=0.05, hspace=0.05)
        #         axes = gs.figure.axes[0]
        #         return fig, axes
        # self.gs = gs
        # self._ax_opts['use_cartopy'] = use_cartopy_opt
        # return fig, axes

    @classmethod
    def get_projection(cls, projection):
        """Retrieve projection object."""
        projections = {
            'lambert': ccrs.LambertConformal(),
            'albers': ccrs.AlbersEqualArea(),
            'stereo': ccrs.Stereographic(),
            'ortho': ccrs.Orthographic(),
            'polar': ccrs.NorthPolarStereo(),
            'mercator': ccrs.Mercator()
        }
        return projections.get(projection, ccrs.PlateCarree())

    def create_subplots_crs(self, gs):
        axes = []
        if 'projection' in self._ax_opts:
            map_projection = self.get_projection(self._ax_opts['projection'])
        else:
            map_projection = ccrs.PlateCarree()

        for i in range(self._subplots[0]):
            for j in range(self._subplots[1]):
                ax = plt.subplot(gs[i, j], projection=map_projection)
                axes.append(ax)
                # Add gridlines
                gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.add_feature(cfeature.LAND, edgecolor='black')
                ax.add_feature(cfeature.LAKES, edgecolor='black')
        return axes


    def init_ax_opts(self, field_name):
        """Initialize map options for a given field."""
        plot_type = "polar" if self.plot_type.startswith("po") else self.plot_type[:2]
        spec = self.config_manager.spec_data.get(field_name, {}).get(f"{plot_type}plot", {})
        defaults = {
            'boundary': None, 'use_pole': 'north', 'profile_dim': None, 'zsum': None, 'zave': None, 'tave': None,
            'taverange': 'all', 'cmap_set_over': None, 'cmap_set_under': None, 'use_cmap': self.config_manager.input_config._cmap,
            'use_diff_cmap': self.config_manager.input_config._cmap, 'cscale': None, 'zscale': 'linear', 'cbar_sci_notation': False,
            'custom_title': False, 'add_grid': False, 'line_contours': True, 'add_tropp_height': False,
            'torder': None, 'add_trend': False, 'use_cartopy': False, 'projection': None, 'extent': [-180, 180, -90, 90],
            'central_lon': 0.0, 'central_lat': 0.0, 'num_clevs': 10, 'time_lev': 0, 'is_diff_field': False,
            'add_extra_field_type': False, 'clabel': None, 'create_clevs': False, 'clevs_prec': 0,
            'clevs': None, 'plot_title': None, 'extend_value': 'both', 'norm': 'both',
            'contour_linestyle': {'lines.linewidth': 0.5, 'lines.linestyle': 'solid'},
            'time_series_plot_linestyle': {'lines.linewidth': 1, 'lines.linestyle': 'solid'},
            'colorbar_fontsize': {'colorbar.fontsize': 8}, 'axes_fontsize': {'axes.fontsize': 10},
            'title_fontsize': {'title.fontsize': 10}, 'subplot_title_fontsize': {'subplot_title.fontsize': 12}
        }
        self._ax_opts = {key: spec.get(key, defaults[key]) for key in defaults}
        return self._ax_opts

    def add_grid(self, ax, lines=True, locations=None):
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

    def colorbar_foo(self, mappable):
        """Attach a colorbar to a plot."""
        ax = mappable.axes
        fig = ax.figure
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        return fig.colorbar(mappable, cax=cax)

    def colorbar(self, mappable):
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
    def frame_params(self):
        return self._frame_params

    @property
    def subplots(self):
        return self._subplots

    def update_ax_opts(self, field_name, ax, pid, level=None):
        """Set or reset map options."""
        if not self.config_manager.input_config.compare:
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

    def plot_text(self, field_name, ax, pid, level=None, data=None):
        """Add text to a map.

        Parameters:
            field_name (str): Name of the field
            ax (Axes or list of Axes): Axes object(s)
            pid (str): Plot type identifier
            level (int): Vertical level (optional, default=None)
            data (Any): xarray Data for basic stats (optional)
         Returns:
            Updated axes internal state
       """
        if isinstance(ax, list):  # Check if ax is a list
            for single_ax in ax:
                self._plot_text(field_name, single_ax, pid, level, data)
        else:
            self._plot_text(field_name, ax, pid, level, data)

    def _plot_text(self, field_name, ax, pid, level=None, data=None):
        """Add text to a single axes."""
        fontsize = pu.subplot_title_font_size(self._subplots)
        findex = self.config_manager.findex
        sname = self.config_manager.config.map_params[findex]['source_name']
        ds_index = self.config_manager.ds_index
        # ds_index = self.config_manager.config.map_params[findex]['source_index']
        geom = pu.get_subplot_geometry(ax) if self.config_manager.compare else None

        # Handle plot titles for comparison cases
        if self.config_manager.compare:
            if geom and geom[0] == (3, 1):  # (3,1) subplot structure
                if geom[1:] == (0, 1, 1, 1):  # Bottom plot
                    self._ax_opts['plot_title'] = "Difference (top - middle)"
                elif geom[1:] in [(1, 1, 0, 1), (0, 1, 0, 1)]:  # Top/Middle plots
                    self._axes_title(ax, findex)
            elif self._subplots == (2, 2):  # (2,2) subplot structure
                if geom[1:] == (0, 1, 1, 0):
                    self._ax_opts['plot_title'] = "Difference (left - right)"
                elif geom[1:] == (0, 0, 1, 1):  # Extra diff plot
                    diff_labels = {
                        "percd": ("% Diff", "%"),
                        "percc": ("% Change", "%"),
                        "ratio": ("Ratio Diff", "ratio"),
                    }
                    diff_type = self.config_manager.extra_diff_plot
                    self._ax_opts['plot_title'], self._ax_opts['clabel'] = diff_labels.get(
                        diff_type, ("Difference (left - right)", None))
                    self._ax_opts['line_contours'] = False
                else:  # Default case
                    self._axes_title(ax, findex)
            else:  # Default title for comparison
                plot_title = os.path.basename(
                    self.config_manager.readers[sname].datasets[ax.get_subplotspec().colspan.start][
                        'filename'])
                ax.set_title(plot_title, fontsize=fontsize)
            ax.set_title(self._ax_opts.get('plot_title', ""), fontsize=fontsize)
            return

        # Non-comparison case
        level_text = self._format_level_text(level)
        name = self._get_field_name(field_name, sname, findex, ds_index)

        loc = None
        left, width = 0, 1.0
        bottom, height = 0, 1.0
        right = left + width
        top = bottom + height
        ax_text_position = (left, width, bottom, height, right, top, loc)

        if 'yz' in pid:
            if self.config_manager.print_basic_stats:
                # plt.rc('text', usetex=True)
                fmt = self._basic_stats(data)
                ax.text(right, top, fmt, transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=10)

            if self.config_manager.use_history:
                ax.set_title(self.config_manager.history_expid + " (" + self.config_manager.history_expdsc + ")")
            else:
                self._axes_title(ax, findex, fs=8)

            ax.text(0.5 * (left + right), bottom + top + 0.1,
                    name, fontweight='bold',
                    fontstyle='italic',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    transform=ax.transAxes)

        elif 'xy' in pid:
            loc = ''
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
                self._axes_title(ax, findex, fs=8)

            ax.text(0.5 * (left + right), bottom + top + 0.1,
                    name + level_text, fontweight='bold',
                    fontstyle='italic',
                    fontsize=14,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)

        elif 'tx' in pid:
            if self.config_manager.use_history:
                ax.set_title(self.config_manager.history_expid + " (" + self.config_manager.history_expdsc + ")", fontsize=10)
            else:
                self._axes_title(ax, findex, fs=8)

            ax.text(0.5 * (left + right), bottom + top + 0.5,
                    name,
                    fontstyle='italic',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12,
                    transform=ax.transAxes)
        elif 'po' in pid:
            pass
        else:  # 'xt' and others
            if self.config_manager.use_history:
                ax.set_title(self.config_manager.history_expid + " (" + self.config_manager.history_expdsc + ")")
            else:
                self._axes_title(ax, findex, fs=8)

            ax.text(0.5 * (left + right), bottom + top + 0.1,
                    name, fontweight='bold',
                    fontstyle='italic',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    transform=ax.transAxes)
        
    def _axes_title(self, ax, findex, fs=12):
        if self.config_manager.get_file_description(findex):
            title_string = self.config_manager.get_file_description(findex)
        elif self.config_manager.get_file_exp_name(findex):
            title_string = self.config_manager.get_file_exp_name(findex)
        else:
            title_string = ''
            if self.config_manager.ax_opts['custom_title']:
                title_string = self.config_manager.ax_opts['custom_title']
        ax.set_title(title_string, loc='left', fontsize=fs)

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

    def _get_field_name(self, field_name, sname, findex, ds_index):
        """Retrieve the appropriate field name from configuration."""
        if 'name' in self.config_manager.config.spec_data.get(field_name, {}):
            return self.config_manager.config.spec_data[field_name]['name']
        return self.config_manager.readers[sname].datasets[findex]['vars'][field_name].attrs.get(
            "long_name", field_name)
 
    # def _handle_comparison_text(self, ax, geom, findex, fontsize):
    #     """Handle text for comparison plots."""
    #     if geom and geom[0] == (3, 1):  # (3,1) subplot structure
    #         if geom[1:] == (0, 1, 1, 1):  # Bottom plot
    #             self._ax_opts['plot_title'] = "Difference (top - middle)"
    #         elif geom[1:] in [(1, 1, 0, 1), (0, 1, 0, 1)]:  # Top/Middle plots
    #             self._axes_title(ax, findex)
    #     elif self._subplots == (2, 2):  # (2,2) subplot structure
    #         if geom[1:] == (0, 1, 1, 0):
    #             self._ax_opts['plot_title'] = "Difference (left - right)"
    #         elif geom[1:] == (0, 0, 1, 1):  # Extra diff plot
    #             diff_labels = {
    #                 "percd": ("% Diff", "%"),
    #                 "percc": ("% Change", "%"),
    #                 "ratio": ("Ratio Diff", "ratio"),
    #             }
    #             diff_type = self.config_manager.extra_diff_plot
    #             self._ax_opts['plot_title'], self._ax_opts['clabel'] = diff_labels.get(
    #                 diff_type, ("Difference (left - right)", None))
    #             self._ax_opts['line_contours'] = False
    #         else:  # Default case
    #             self._axes_title(ax, findex)
    #     else:  # Default title for comparison
    #         plot_title = os.path.basename(
    #             self.config_manager.readers[self.config_manager.source_names[self.config_manager.ds_index]].datasets[
    #                 ax.get_subplotspec().colspan.start]['filename'])
    #         ax.set_title(plot_title, fontsize=fontsize)
    #     ax.set_title(self._ax_opts.get('plot_title', ""), fontsize=fontsize)

    # def _handle_non_comparison_text(self, ax, field_name, sname, findex, ds_index, pid, level, data, fontsize):
    #     """Handle text for non-comparison plots."""
    #     level_text = self._format_level_text(level)
    #     name = self._get_field_name(field_name, sname, findex, ds_index)

    #     if 'yz' in pid:
    #         self._add_basic_stats(ax, data, loc='right')
    #         self._set_history_or_title(ax, findex)
    #         self._add_text(ax, name, position=(0.5, 1.1), fontsize=fontsize)
    #     elif 'xy' in pid:
    #         self._add_basic_stats(ax, data, loc='left')
    #         self._set_history_or_title(ax, findex)
    #         self._add_text(ax, name + level_text, position=(0.5, 1.1), fontsize=fontsize)
    #     elif 'tx' in pid:
    #         self._set_history_or_title(ax, findex)
    #         self._add_text(ax, name, position=(0.5, 1.5), fontsize=fontsize)
    #     elif 'po' in pid:
    #         pass
    #     else:  # 'xt' and others
    #         self._set_history_or_title(ax, findex)
    #         self._add_text(ax, name, position=(0.5, 1.1), fontsize=fontsize)

    # def _axes_title(self, ax, findex, fs=12):
    #     """Set the title for the axes or list of axes.
    #         Note we need to handle both single axes and lists of axes
    #     """
    #     if isinstance(ax, list):  # Check if ax is a list
    #         for single_ax in ax:
    #             self._set_single_axes_title(single_ax, findex, fs)
    #     else:
    #         self._set_single_axes_title(ax, findex, fs)

    # def _set_single_axes_title(self, ax, findex, fs):
    #     """Set the title for a single axes."""
    #     if self.config_manager.get_file_description(findex):
    #         title_string = self.config_manager.get_file_description(findex)
    #     elif self.config_manager.get_file_exp_name(findex):
    #         title_string = self.config_manager.get_file_exp_name(findex)
    #     else:
    #         title_string = ''
    #         if self.config_manager.ax_opts['custom_title']:
    #             title_string = self.config_manager.ax_opts['custom_title']
    #     ax.set_title(title_string, loc='left', fontsize=fs)

    # def _add_basic_stats(self, ax, data, loc='right'):
    #     """Add basic statistics text to the plot."""
    #     if self.config_manager.print_basic_stats and data is not None:
    #         stats_text = self._basic_stats(data)
    #         ax.text(1.0 if loc == 'right' else 0.0, 1.0, stats_text, transform=ax.transAxes,
    #                 ha=loc, va='bottom', fontsize=10)

    # def _axes_title(self, ax, findex, fs=12):
    #     if self.config_manager.get_file_description(findex):
    #         title_string = self.config_manager.get_file_description(findex)
    #     elif self.config_manager.get_file_exp_name(findex):
    #         title_string = self.config_manager.get_file_exp_name(findex)
    #     else:
    #         title_string = ''
    #         if self.config_manager.ax_opts['custom_title']:
    #             title_string = self.config_manager.ax_opts['custom_title']
    #     ax.set_title(title_string, loc='left', fontsize=fs)

    # def _set_history_or_title(self, ax):
    #     """Set plot title based on history flag."""
    #     if self.config_manager.history_config.use_history:
    #         ax.set_title(f"{self.config_manager.history_config.history_expid} ({self.config_manager.history_config.history_expdsc})",
    #                      fontsize=10)
    #     else:
    #         self._axes_title(ax, self.config_manager.findex, fs=8)

    # def _add_text(self, ax, text, level_text="", position=(0.5, 1.1), fontsize=14):
    #     """Add styled text annotation to the plot."""
    #     ax.text(position[0], position[1], text, fontweight='bold', fontstyle='italic',
    #             horizontalalignment='center', verticalalignment='center',
    #             fontsize=fontsize, transform=ax.transAxes)

    # def _get_title_string(self, findex):
    #     """Retrieve the title string for the axes."""
    #     if self.config_manager.get_file_description(findex):
    #         return self.config_manager.get_file_description(findex)
    #     elif self.config_manager.get_file_exp_name(findex):
    #         return self.config_manager.get_file_exp_name(findex)
    #     elif self.config_manager.ax_opts.get('custom_title'):
    #         return self.config_manager.ax_opts['custom_title']
    #     return ""

    # def _set_history_or_title(self, ax, findex):
    #     """Set plot title based on history flag."""
    #     if self.config_manager.history_config.use_history:
    #         ax.set_title(f"{self.config_manager.history_config.history_expid} ({self.config_manager.history_config.history_expdsc})",
    #                      fontsize=10)
    #     else:
    #         self._axes_title(ax, findex)

 
    # def _plot_text_new(self, field_name, ax, pid, level=None, data=None):
    #     """Add text to a single axes."""
    #     fontsize = pu.subplot_title_font_size(self._subplots)
    #     findex = self.config_manager.findex
    #     sname = self.config_manager.config.map_params[findex]['source_name']
    #     ds_index = self.config_manager.ds_index

    #     # Dispatch dictionary for plot type handlers
    #     plot_type_handlers = {
    #         'yz': self._handle_yz_plot_text,
    #         'xy': self._handle_xy_plot_text,
    #         'tx': self._handle_tx_plot_text,
    #         'po': self._handle_po_plot_text,
    #         'xt': self._handle_default_plot_text,
    #     }

    #     # Get the appropriate handler for the plot type
    #     handler = plot_type_handlers.get(pid[:2], self._handle_default_plot_text)

    #     # Call the handler
    #     handler(field_name, ax, pid, level, data, fontsize, findex, sname, ds_index)

    # # Helper functions for specific plot types
    # def _handle_yz_plot_text(self, field_name, ax, pid, level, data, fontsize, findex, sname, ds_index):
    #     """Handle text for 'yz' plot type."""
    #     if self.config_manager.print_basic_stats:
    #         fmt = self._basic_stats(data)
    #         ax.text(1.0, 1.0, fmt, transform=ax.transAxes, ha='right', va='bottom', fontsize=10)

    #     if self.config_manager.use_history:
    #         ax.set_title(f"{self.config_manager.history_expid} ({self.config_manager.history_expdsc})", fontsize=10)
    #     else:
    #         self._axes_title(ax, findex, fs=8)

    #     name = self._get_field_name(field_name, sname, findex, ds_index)
    #     ax.text(0.5, 1.1, name, fontweight='bold', fontstyle='italic',
    #             horizontalalignment='center', verticalalignment='center',
    #             fontsize=14, transform=ax.transAxes)

    # def _handle_xy_plot_text(self, field_name, ax, pid, level, data, fontsize, findex, sname, ds_index):
    #     """Handle text for 'xy' plot type."""
    #     if self.config_manager.print_basic_stats:
    #         fmt = self._basic_stats(data)
    #         ax.text(1.0, 1.0, fmt, transform=ax.transAxes, ha='right', va='bottom', fontsize=10)

    #     if self.config_manager.real_time and not self.config_manager.print_basic_stats:
    #         ax.text(1.0, 1.0, self.config_manager.real_time, transform=ax.transAxes,
    #                 ha='right', va='bottom', fontsize=10)

    #     if self.config_manager.use_history:
    #         ax.set_title(f"{self.config_manager.history_expid} ({self.config_manager.history_expdsc})", fontsize=10)
    #     else:
    #         self._axes_title(ax, findex, fs=8)

    #     name = self._get_field_name(field_name, sname, findex, ds_index)
    #     level_text = self._format_level_text(level)
    #     ax.text(0.5, 1.1, name + level_text, fontweight='bold', fontstyle='italic',
    #             horizontalalignment='center', verticalalignment='center',
    #             fontsize=14, transform=ax.transAxes)

    # def _handle_tx_plot_text(self, field_name, ax, pid, level, data, fontsize, findex, sname, ds_index):
    #     """Handle text for 'tx' plot type."""
    #     if self.config_manager.use_history:
    #         ax.set_title(f"{self.config_manager.history_expid} ({self.config_manager.history_expdsc})", fontsize=10)
    #     else:
    #         self._axes_title(ax, findex, fs=8)

    #     name = self._get_field_name(field_name, sname, findex, ds_index)
    #     ax.text(0.5, 1.5, name, fontstyle='italic',
    #             horizontalalignment='center', verticalalignment='center',
    #             fontsize=12, transform=ax.transAxes)

    # def _handle_po_plot_text(self, field_name, ax, pid, level, data, fontsize, findex, sname, ds_index):
    #     """Handle text for 'po' plot type."""
    #     pass  # No specific text handling for 'po' plot type

    # def _handle_default_plot_text(self, field_name, ax, pid, level, data, fontsize, findex, sname, ds_index):
    #     """Handle text for default plot types."""
    #     if self.config_manager.use_history:
    #         ax.set_title(f"{self.config_manager.history_expid} ({self.config_manager.history_expdsc})", fontsize=10)
    #     else:
    #         self._axes_title(ax, findex, fs=8)

    #     name = self._get_field_name(field_name, sname, findex, ds_index)
    #     ax.text(0.5, 1.1, name, fontweight='bold', fontstyle='italic',
    #             horizontalalignment='center', verticalalignment='center',
    #             fontsize=14, transform=ax.transAxes)
