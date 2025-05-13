import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.transforms import BboxBase as bbase
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import numpy as np

import eviz.lib.xarray_utils as xu
import eviz.lib.autoviz.plot_utils as pu
from eviz.lib.data.data_utils import apply_conversion
from eviz.lib.autoviz.config import Config


# ===== Utility Classes =====

class Logger:
    """Centralized logging functionality"""
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


# ===== Abstract Base Classes =====

class PlotStrategy(ABC):
    """Strategy pattern for different plot types"""
    
    @abstractmethod
    def create_plot(self, figure_builder: 'FigureBuilder', field_name: str, level: Optional[int] = None) -> None:
        """Create a specific type of plot"""
        pass


class FigureBuilder(ABC):
    """Builder pattern for constructing figures"""
    
    @abstractmethod
    def create_figure(self) -> matplotlib.figure.Figure:
        """Create the matplotlib figure"""
        pass
    
    @abstractmethod
    def create_axes(self) -> Any:
        """Create the axes for the figure"""
        pass
    
    @abstractmethod
    def set_options(self, field_name: str) -> None:
        """Set options for the figure and axes"""
        pass
    
    @abstractmethod
    def add_features(self, axes: Any) -> Any:
        """Add features to the axes"""
        pass
    
    @abstractmethod
    def add_text(self, field_name: str, axes: Any, plot_id: str, level: Optional[int] = None, data: Any = None) -> None:
        """Add text to the plot"""
        pass


# ===== Concrete Implementations =====

class CartopyFeatureDecorator:
    """Decorator for adding Cartopy features to maps"""
    
    @staticmethod
    def add_features(axes: Axes, extent: Optional[Union[List, str]] = None) -> Axes:
        """Add standard Cartopy features to a map"""
        if extent:
            axes.coastlines(resolution='110m', color='black')
            if extent == 'conus':
                axes.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='brown')
                states_provinces = cfeature.NaturalEarthFeature(
                    category='cultural',
                    name='admin_1_states_provinces_lines',
                    scale='50m', 
                    facecolor='none'
                )
                axes.add_feature(states_provinces, edgecolor='grey', zorder=10)
            axes.add_feature(cfeature.BORDERS)
            axes.add_feature(cfeature.OCEAN)
            axes.add_feature(cfeature.LAND)
            axes.add_feature(cfeature.LAKES)
            axes.add_feature(cfeature.COASTLINE)
            axes.add_feature(cfeature.RIVERS)
        else:
            axes.coastlines(alpha=0.1)
            axes.add_feature(cfeature.LAND, facecolor='0.9')
            axes.add_feature(cfeature.LAKES, alpha=0.9)
            axes.add_feature(cfeature.BORDERS, zorder=10, linewidth=0.5, edgecolor='grey')
            axes.add_feature(cfeature.COASTLINE, zorder=10)
        return axes


class ProjectionFactory:
    """Factory for creating different map projections"""
    
    @staticmethod
    def create_projection(projection_type: Optional[str] = None, extent: Optional[List] = None) -> ccrs.Projection:
        """Create a Cartopy projection based on type and extent"""
        if not projection_type:
            return ccrs.PlateCarree()
            
        if not extent:
            extent = [-140, -40, 15, 65]  # default to CONUS
            
        central_lon = np.mean(extent[:2])
        central_lat = np.mean(extent[2:])
        
        projections = {
            'lambert': ccrs.LambertConformal(
                central_latitude=central_lat,
                central_longitude=central_lon
            ),
            'albers': ccrs.AlbersEqualArea(
                central_latitude=central_lat,
                central_longitude=central_lon
            ),
            'stereo': ccrs.Stereographic(
                central_latitude=central_lat,
                central_longitude=central_lon
            ),
            'ortho': ccrs.Orthographic(
                central_latitude=central_lat,
                central_longitude=central_lon
            ),
            'polar': ccrs.NorthPolarStereo(central_longitude=-100),
            'mercator': ccrs.Mercator()
        }
        
        return projections.get(projection_type, ccrs.PlateCarree())


class ColorbarFactory:
    """Factory for creating colorbars"""
    
    @staticmethod
    def create_colorbar(mappable: Any, ax: Axes) -> Any:
        """Create a colorbar for a mappable object"""
        last_axes = plt.gca()
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar


class StatsCalculator:
    """Utility class for calculating statistics on data"""
    
    @staticmethod
    def basic_stats(data: Any) -> str:
        """Calculate basic statistics for a dataset"""
        datamean = data.mean().values
        datastd = data.std().values
        return f"\nMean:{datamean:.2e}\nStd:{datastd:.2e}"
    
    @staticmethod
    def full_stats(config: Config, field_name: str, *dimensions) -> str:
        """Calculate full statistics over multiple dimensions"""
        dimension_names = []
        for dim in dimensions:
            dimension_names.append(config.get_model_dim_name(dim))
            
        m = xu.compute_mean_over_dim(
            config.readers[config.ds_index].datasets[config.findex]['vars'],
            tuple(dimension_names), 
            field_name=field_name
        )
        
        s = xu.compute_std_over_dim(
            config.readers[config.ds_index].datasets[config.findex]['vars'],
            tuple(dimension_names), 
            field_name=field_name
        )
        
        datamean = apply_conversion(config, np.nanmean(m), field_name)
        datastd = apply_conversion(config, np.nanstd(s), field_name)
        
        return f"\nMean:{datamean:.2e}\nStd:{datastd:.2e}"


class GridHelper:
    """Helper class for adding grids to plots"""
    
    @staticmethod
    def add_grid(ax: Axes, lines: bool = True, locations: Optional[Tuple] = None) -> None:
        """Add a grid to an axis"""
        if lines:
            ax.grid(lines, alpha=0.5, which="minor", ls=":")
            ax.grid(lines, alpha=0.7, which="major")

        if locations is not None:
            assert len(locations) == 4, "Invalid entry for the locations of the markers"
            xmin, xmaj, ymin, ymaj = locations

            ax.xaxis.set_minor_locator(MultipleLocator(xmin))
            ax.xaxis.set_major_locator(MultipleLocator(xmaj))
            ax.yaxis.set_minor_locator(MultipleLocator(ymin))
            ax.yaxis.set_major_locator(MultipleLocator(ymaj))
    
    @staticmethod
    def set_cartopy_latlon_opts(ax: Axes, extent: List) -> Axes:
        """Set latitude/longitude options for Cartopy maps"""
        gl = ax.grid(color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

        if extent[0] < -179.0:
            xgrid = np.array([-180, -120, -60, 0, 60, 120, 180])
            ygrid = np.array([-90, -60, -30, 0, 30, 60])
        else:
            xgrid = np.linspace(extent[0], extent[1], 8)
            ygrid = np.linspace(extent[2], extent[3], 8)
            
        return ax


class AxesOptionsManager:
    """Manages options for plot axes"""
    
    def __init__(self, config: Config):
        self.config = config
        self.options = {}
        
    def init_options(self, field_name: str, plot_type: str) -> Dict:
        """Initialize options for a specific field and plot type"""
        if plot_type == "po":
            plot_type = "polar"
            
        options = {
            'boundary': None,
            'use_pole': 'north',
            'profile_dim': None,
            'zsum': None,
            'zave': None,
            'tave': None,
            'taverange': 'all',
            'cmap_set_over': None,
            'cmap_set_under': None,
            'use_cmap': self.config.cmap,
            'use_diff_cmap': self.config.cmap,
            'cscale': None,
            'zscale': 'linear',
            'cbar_sci_notation': False,
            'custom_title': False,
            'add_grid': False,
            'line_contours': True,
            'add_tropp_height': False,
            'torder': None,
            'add_trend': False,
            'use_cartopy': True,
            'projection': None,
            'extent': [-180, 180, -90, 90],
            'central_lon': 0.0,
            'central_lat': 0.0,
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
            'contour_linestyle': {'lines.linewidth': 0.5, 'lines.linestyle': 'solid'},
            'time_series_plot_linestyle': {'lines.linewidth': 1, 'lines.linestyle': 'solid'},
            'colorbar_fontsize': {'colorbar.fontsize': 8},
            'axes_fontsize': {'axes.fontsize': 10},
            'title_fontsize': {'title.fontsize': 10},
            'subplot_title_fontsize': {'subplot_title.fontsize': 12}
        }
        
        # Update options from config if available
        spec_data = self.config.spec_data.get(field_name, {}).get(plot_type + 'plot', {})
        
        for key in [
            'boundary', 'pole', 'profile_dim', 'zsum', 'zave', 'tave', 'taverange',
            'cmap_set_over', 'cmap_set_under', 'cmap', 'diff_cmap', 'cscale', 'zscale',
            'cbar_sci_notation', 'custom_title', 'grid', 'line_contours', 'add_tropp_height',
            'torder', 'add_trend', 'use_cartopy', 'projection', 'extent', 'num_clevs', 'time_lev'
        ]:
            if key in spec_data:
                if key == 'pole':
                    options['use_pole'] = spec_data[key]
                elif key == 'grid':
                    options['add_grid'] = spec_data[key]
                elif key == 'cmap':
                    options['use_cmap'] = spec_data[key]
                elif key == 'diff_cmap':
                    options['use_diff_cmap'] = spec_data[key]
                else:
                    options[key] = spec_data[key]
        
        self.options = options
        return options
    
    def update_options(self, field_name: str, ax: Axes, plot_id: str, level: Optional[int] = None) -> Dict:
        """Update options based on plot type and comparison settings"""
        if self.config.compare:
            geom = pu.get_subplot_geometry(ax)
            subplots = self.options.get('_subplots', (1, 1))
            
            if subplots == (3, 1):
                # Bottom plot (difference)
                if geom[1:] == (0, 1, 1, 1):
                    self.options['line_contours'] = False
                    if 'yz' in plot_id:
                        self._set_clevs(field_name, 'yzplot', 'diffcontours')
                    elif 'xy' in plot_id:
                        self._set_clevs(field_name, 'xyplot', f'diff_{level}')
                # Top and middle plots
                else:
                    if 'yz' in plot_id:
                        self._set_clevs(field_name, 'yzplot', 'contours')
                    elif 'xy' in plot_id:
                        self._set_clevs(field_name, 'xyplot', int(level))
            elif subplots == (2, 2):
                # Handle 2x2 subplot configurations
                if geom[1:] == (0, 1, 1, 0):  # Difference plot
                    if 'yz' in plot_id:
                        self._set_clevs(field_name, 'yzplot', 'diffcontours')
                    elif 'xy' in plot_id:
                        self._set_clevs(field_name, 'xyplot', f'diff_{level}')
                elif geom[1:] == (0, 0, 1, 1):  # Extra difference plot
                    self.options['line_contours'] = False
                    diff_opt = f'diff_{self.config.extra_diff_plot}'
                    if diff_opt in self.config.spec_data[field_name]:
                        self.options['clevs'] = self.config.spec_data[field_name][diff_opt]
                else:  # Regular plots
                    self._set_clevs_by_plot_type(field_name, plot_id, level)
        else:  # Single plot
            self._set_clevs_by_plot_type(field_name, plot_id, level)
            
        return self.options
    
    def _set_clevs_by_plot_type(self, field_name: str, plot_id: str, level: Optional[int] = None) -> None:
        """Set contour levels based on plot type"""
        if plot_id == 'yz':
            self._set_clevs(field_name, 'yzplot', 'contours')
        elif plot_id == 'yzave':
            self._set_clevs(field_name, 'yzaveplot', 'contours')
        elif plot_id == 'xy':
            self._set_clevs(field_name, 'xyplot', int(level))
        elif plot_id == 'xyave':
            self._set_clevs(field_name, 'xyaveplot', int(level))
        elif plot_id == 'tx':
            self._set_clevs(field_name, 'txplot', 'contours')
        elif plot_id == 'polar':
            self._set_clevs(field_name, 'polarplot', int(level))
    
    def _set_clevs(self, field_name: str, plot_type: str, contour_type: Union[str, int]) -> None:
        """Set contour levels for a specific field and plot type"""
        if isinstance(contour_type, int):
            if contour_type in self.config.spec_data[field_name][plot_type]['levels']:
                self.options['clevs'] = self.config.spec_data[field_name][plot_type]['levels'][contour_type]
                if not self.options['clevs']:
                    self.options['create_clevs'] = True
            else:
                self.options['create_clevs'] = True
        else:
            if contour_type in self.config.spec_data[field_name][plot_type]:
                self.options['clevs'] = self.config.spec_data[field_name][plot_type][contour_type]
                if not self.options['clevs']:
                    self.options['create_clevs'] = True
            else:
                self.options['create_clevs'] = True


class TextManager:
    """Manages text elements on plots"""
    
    def __init__(self, config: Config, options: Dict):
        self.config = config
        self.options = options
        self.stats_calculator = StatsCalculator()
        
    def add_text(self, field_name: str, ax: Axes, plot_id: str, level: Optional[int] = None, data: Any = None) -> None:
        """Add text elements to a plot"""
        if self.config.compare:
            self._add_comparison_text(field_name, ax, plot_id)
        else:
            self._add_standard_text(field_name, ax, plot_id, level, data)
    
    def _add_comparison_text(self, field_name: str, ax: Axes, plot_id: str) -> None:
        """Add text for comparison plots"""
        fontsize = pu.subplot_title_font_size(self.options.get('_subplots', (1, 1)))
        findex = self.config.findex
        geom = pu.get_subplot_geometry(ax)
        
        if geom[0] == (3, 1):
            # Bottom plot (difference)
            if geom[1:] == (0, 1, 1, 1):
                self.options['plot_title'] = "Difference (top - middle)"
                ax.set_title(self.options['plot_title'], fontsize=fontsize)
            # Top and middle plots
            elif geom[1:] == (1, 1, 0, 1) or geom[1:] == (0, 1, 0, 1):
                self._set_axes_title(ax, findex)
        elif self.options.get('_subplots', (1, 1)) == (2, 2):
            if geom[1:] == (0, 1, 1, 0):
                self.options['plot_title'] = "Difference (left - right)"
                ax.set_title(self.options['plot_title'], fontsize=fontsize)
            elif geom[1:] == (0, 0, 1, 1):
                self._set_diff_title(ax, fontsize)
            else:
                if ax.get_subplotspec().colspan.start == 0 and ax.get_subplotspec().rowspan.start == 0:
                    self._set_axes_title(ax, findex)
                if ax.get_subplotspec().colspan.start == 1 and ax.get_subplotspec().rowspan.start == 0:
                    self._set_axes_title(ax, findex)
        else:
            sname = self.config.map_params[findex]['source_name']
            plot_title = os.path.basename(
                self.config.readers[sname].datasets[ax.get_subplotspec().colspan.start]['filename'])
            ax.set_title(plot_title, fontsize=fontsize)
    
    def _set_diff_title(self, ax: Axes, fontsize: int) -> None:
        """Set title for difference plots"""
        if "percd" in self.config.extra_diff_plot:
            self.options['plot_title'] = "% Diff"
            self.options['clabel'] = "%"
        elif "percc" in self.config.extra_diff_plot:
            self.options['plot_title'] = "% Change"
            self.options['clabel'] = "%"
        elif "ratio" in self.config.extra_diff_plot:
            self.options['plot_title'] = "Ratio Diff"
            self.options['clabel'] = "ratio"
        else:
            self.options['plot_title'] = "Difference (left - right)"
        
        self.options['line_contours'] = False
        ax.set_title(self.options['plot_title'], fontsize=fontsize)
    
    def _add_standard_text(self, field_name: str, ax: Axes, plot_id: str, level: Optional[int] = None, data: Any = None) -> None:
        """Add text for standard (non-comparison) plots"""
        left, width = 0, 1.0
        bottom, height = 0, 1.0
        right = left + width
        top = bottom + height
        findex = self.config.findex
        sname = self.config.map_params[findex]['source_name']
        ds_index = self.config.map_params[findex]['source_index']
        
        # Get level text
        level_text = self._get_level_text(level)
        
        # Get field name
        if 'name' in self.config.spec_data[field_name]:
            name = self.config.spec_data[field_name]['name']
        else:
            name = self.config.readers[sname].datasets[findex]['vars'][field_name].attrs.get("long_name", field_name)
        
        # Add plot-specific text
        if 'yz' in plot_id:
            self._add_yz_text(ax, data, findex, name, right, top, left, bottom)
        elif 'xy' in plot_id:
            self._add_xy_text(ax, data, findex, name, level_text, right, top, left, bottom)
        elif 'tx' in plot_id:
            self._add_tx_text(ax, findex, name, left, right, bottom, top)
        elif 'po' in plot_id:
            pass  # No specific text for polar plots
        else:  # 'xt' and others
            self._add_default_text(ax, findex, name, left, right, bottom, top)
    
    def _get_level_text(self, level: Optional[int]) -> str:
        """Get text describing the vertical level"""
        if self.config.ax_opts['zave']:
            return ' (Column Mean)'
        elif self.config.ax_opts['zsum']:
            return ' (Total Column)'
        elif str(level) == '0':
            return ''
        elif level is not None:
            if level > 10000:
                return f'@ {level} Pa'
            else:
                return f'@ {level} mb'
        return ''
    
    