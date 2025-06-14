import os
import sys
from dataclasses import dataclass
import numpy as np
import xarray as xr
import logging
import warnings
from matplotlib import pyplot as plt
from eviz.lib.data.utils import apply_mean
from eviz.lib.data.utils import apply_conversion
from eviz.models.esm.nuwrf import NuWrf

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class Wrf(NuWrf):
    """ Define WRF specific model data and functions."""

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.comparison_plot = False
        self.source_name = 'wrf'

    def _init_wrf_domain(self, data_source):
        """WRF-specific initialization."""
        if not self.p_top:
            self._init_domain(data_source)

    def _init_domain(self, data_source):
        """ Approximate unknown fields """
        # Create sigma->pressure dictionary
        self.p_top = data_source['P_TOP'][0] / 1e5  # mb
        self.eta_full = np.array(data_source['ZNW'][0])
        self.eta_mid = np.array(data_source['ZNU'][0])
        self.levf = np.empty(len(self.eta_full))
        self.levs = np.empty(len(self.eta_mid))

        for i, s in enumerate(self.eta_full):
            if s > 0:
                self.levf[i] = int(self.p_top + s * (1000 - self.p_top))
            else:
                self.levf[i] = self.p_top + s * (1000 - self.p_top)

        for i, s in enumerate(self.eta_mid):
            if s > 0:
                self.levs[i] = int(self.p_top + s * (1000 - self.p_top))
            else:
                self.levs[i] = self.p_top + s * (1000 - self.p_top)

    def _process_coordinates(self, data2d, dim1, dim2, 
                             field_name, 
                             plot_type, file_index, figure):
        """Process coordinates for WRF plots"""
        dim1, dim2 = self.coord_names(self.source_name, data2d, plot_type)
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure
        elif 'yz' in plot_type:
            xs = np.array(self._get_wrf_coord(dim1[0], data2d)[0, :][:, 0])
            ys = self.levs
            latN = max(xs[:])
            latS = min(xs[:])
            self.config_manager.ax_opts['extent'] = [None, None, latS, latN]
            return data2d, xs, ys, field_name, plot_type, file_index, figure
        else:
            xs = np.array(self._get_wrf_coord(dim1[0], data2d)[0, :])
            ys = np.array(self._get_wrf_coord(dim2[0], data2d)[:, 0])
            latN = max(ys[:])
            latS = min(ys[:])
            lonW = min(xs[:])
            lonE = max(xs[:])
            self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
            self.config_manager.ax_opts['central_lon'] = np.mean([lonW, lonE])
            self.config_manager.ax_opts['central_lat'] = np.mean([latS, latN])
            return data2d, xs, ys, field_name, plot_type, file_index, figure

    def _get_wrf_coord(self, name, data):
        try:
            return data[name]
        except Exception as e:
            self.logger.error('key error: %s, not found' % str(e))
            return None

    def _get_field_for_simple_plot(self, field_name, plot_type):
        data2d = None
        d = self.source_data['vars'][field_name]
        dim1, dim2 = self.coord_names(self.source_name, self.source_data,
                                      field_name, plot_type)
        if 'yz' in plot_type:
            data2d = self.__get_yz(d, field_name)
        elif 'xy' in plot_type:
            data2d = self.__get_xy(d, field_name, level=0)
        else:
            pass

        xs, ys = None, None
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type
        elif 'yz' in plot_type:
            xs = np.array(self._get_field(dim1, d)[0, :][:, 0])
            ys = self.levs
            return data2d, xs, ys, field_name, plot_type
        else:
            xs = np.array(self._get_field(dim1, data2d)[0, :])
            ys = np.array(self._get_field(dim2, data2d)[:, 0])
            return data2d, xs, ys, field_name, plot_type

    def dim_names(self, field_name, pid):
        """ Get WRF dim names based on field and plot type

        Parameters:
            field_name(str) : Field name associated with this plot
            pid (str) : plot type

        """
        dims = []
        d = self.source_data[field_name]
        stag = d.stagger
        xsuf, ysuf, zsuf = "", "", ""
        if stag == "X":
            xsuf = "_stag"
        elif stag == "Y":
            ysuf = "_stag"
        elif stag == "Z":
            zsuf = "_stag"

        xc = self.get_dd(self.source_name, self.source_data, 'xc', field_name)
        if xc:
            dims.append(xc + xsuf)

        yc = self.get_dd(self.source_name, self.source_data, 'yc', field_name)
        if yc:
            dims.append(yc + ysuf)

        zc = self.get_dd(self.source_name, self.source_data, 'zc', field_name)
        if zc:
            dims.append(zc + zsuf)

        tc = self.get_dd(self.source_name, self.source_data, 'tc', field_name)
        if tc:
            dims.append(tc)

        # Maps are 2D plots, so we only need - at most - 2 dimensions, depending on plot type
        dim1, dim2 = None, None
        if 'yz' in pid:
            dim1 = dims[1]
            dim2 = dims[2]
        elif 'xt' in pid:
            dim1 = dims[3]
        elif 'tx' in pid:
            dim1 = dims[0]
            dim2 = dims[3]
        else:
            dim1 = dims[0]
            dim2 = dims[1]
        return dim1, dim2

    def _get_yz(self, d, time_lev=0):
        """ Create YZ slice from N-dim data field"""
        d = d.squeeze()
        if self.get_model_dim_name('tc') in d.dims:
            num_times = np.size(d.Time)
            if self.config_manager.ax_opts['tave'] and num_times > 1:
                self.logger.debug(f"Averaging over {num_times} time levels.")
                data2d = apply_mean(self.config_manager, d)
            else:
                data2d = d.isel(Time=time_lev)
        else:
            data2d = d
        # WRF specific:
        d = self.source_data['vars'][d.name]
        stag = d.stagger
        if stag == "X":
            data2d = data2d.mean(
                dim=self.get_model_dim_name('xc') + "_stag")
        else:
            data2d = data2d.mean(dim=self.get_model_dim_name('xc'))
        return apply_conversion(self.config_manager, data2d, d.name)

    def _get_xt(self, d, time_lev, level=None):
        """ Extract time-series from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 1D (time) series
        """
        d_temp = d
        if d_temp is None:
            return

        xtime = d.XTIME.values
        num_times = xtime.size
        self.logger.info(f"'{d.name}' field has {num_times} time levels")

        if isinstance(time_lev, list):
            self.logger.info(f"Computing time series on {time_lev} time range")
            data2d = d_temp.isel(Time=slice(time_lev))
        else:
            data2d = d_temp.squeeze()

        if 'mean_type' in self.config.spec_data[d.name]['xtplot']:
            mean_type = self.config.spec_data[d.name]['xtplot']['mean_type']
            self.logger.info(f"Averaging method: {mean_type}")
            # annual:
            if mean_type == 'point_sel':
                xc = self.config.spec_data[d.name]['xtplot']['point_sel'][0]
                yc = self.config.spec_data[d.name]['xtplot']['point_sel'][1]
                data2d = data2d.sel(lon=xc, lat=yc, method='nearest')
            elif mean_type == 'area_sel':
                x1 = self.config.spec_data[d.name]['xtplot']['area_sel'][0]
                x2 = self.config.spec_data[d.name]['xtplot']['area_sel'][1]
                y1 = self.config.spec_data[d.name]['xtplot']['area_sel'][2]
                y2 = self.config.spec_data[d.name]['xtplot']['area_sel'][3]
                data2d = data2d.sel(lon=np.arange(x1, x2, 0.5),
                                    lat=np.arange(y1, y2, 0.5), method='nearest')
                data2d = data2d.mean(dim=(self.find_matching_dimension(d.dims, 'xc'),
                                          self.find_matching_dimension(d.dims, 'yc')))
            elif mean_type in ['year', 'season', 'month']:
                data2d = data2d.groupby(
                    self.find_matching_dimension(d.dims, 'tc') + '.' + mean_type).mean(
                    dim=self.find_matching_dimension(d.dims, 'tc'), keep_attrs=True)
            else:
                data2d = data2d.groupby(self.find_matching_dimension(d.dims, 'tc')).mean(
                    dim=xr.ALL_DIMS, keep_attrs=True)
                if 'mean_type' in self.config.spec_data[d.name]['xtplot']:
                    if self.config.spec_data[d.name]['xtplot']['mean_type'] == 'rolling':
                        window_size = 5
                        if 'window_size' in self.config.spec_data[d.name]['xtplot']:
                            window_size = self.config.spec_data[d.name]['xtplot'][
                                'window_size']
                        self.logger.info(f" -- smoothing window size: {window_size}")
                        kernel = np.ones(window_size) / window_size
                        convolved_data = np.convolve(data2d, kernel, mode="same")
                        data2d = xr.DataArray(convolved_data,
                                              dims=self.find_matching_dimension(d.dims, 'tc'),
                                              coords=data2d.coords)

        else:
            data2d = data2d.groupby(self.find_matching_dimension(d.dims, 'tc')).mean(
                dim=xr.ALL_DIMS, keep_attrs=True)

        if 'level' in self.config.spec_data[d.name]['xtplot']:
            level = int(self.config.spec_data[d.name]['xtplot']['level'])
            lev_to_plot = int(np.where(
                data2d.coords[self.find_matching_dimension(d.dims, 'zc')].values == level)[0])
            data2d = data2d[:, lev_to_plot].squeeze()

        data2d.attrs = d.attrs.copy()
        return apply_conversion(self.config, data2d, d.name)

    def _select_yrange(self, data2d, name):
        """ Select a range of vertical levels"""
        if 'zrange' in self.config_manager.spec_data[name]['yzplot']:
            if not self.config_manager.spec_data[name]['yzplot']['zrange']:
                return data2d
            lo_z = self.config_manager.spec_data[name]['yzplot']['zrange'][0]
            hi_z = self.config_manager.spec_data[name]['yzplot']['zrange'][1]
            if hi_z >= lo_z:
                self.logger.error(
                    f"Upper level value ({hi_z}) must be less than low level value ({lo_z})")
                return
            lev = self.find_matching_dimension(data2d.dims, 'zc')
            min_index, max_index = 0, len(data2d.coords[lev].values) - 1
            for k, v in enumerate(data2d.coords[lev]):
                if data2d.coords[lev].values[k] == lo_z:
                    min_index = k
            for k, v in enumerate(data2d.coords[lev]):
                if data2d.coords[lev].values[k] == hi_z:
                    max_index = k
            return data2d[min_index:max_index + 1, :, :]
        else:
            return data2d

    def basic_plot(self):
        """
        Create a basic plot, i.e. one without specifications.
        """
        for k, field_names in self.config_manager.to_plot.items():
            for field_name in field_names:
                self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
                self._basic_plot(field_name, self.ax)
                self._plot_dest(field_name)

    def _basic_plot(self, field_name, fig, ax, level=0):
        """Helper function for basic_plot() """
        pid = self.config_manager.app_data['inputs'][0]['to_plot'][field_name]
        data2d, dim1, dim2 = self.__get_plot_data(field_name, pid=pid)
        if data2d is None:
            return
        cf = ax.contourf(dim1.values, dim2.values, data2d, cmap=self.config_manager.cmap)
        cbar = self.fig.colorbar(cf, ax=ax,
                                 orientation='vertical',
                                 pad=0.05,
                                 fraction=0.05)

        # Get the appropriate reader
        reader = self._get_reader(self.source_name)
        if not reader:
            self.logger.error(f"No reader found for source {self.source_name}")
            return

        d = reader.get_field(field_name, self.config_manager.findex)
        dvars = d['vars'][field_name]
        t_label = self.config_manager.meta_attrs['field_name'][self.source_name]
        if self.config_manager.source_names[self.config_manager.findex] in ['lis', 'wrf']:
            dim1_name = self.config_manager.meta_coords['xc'][self.source_name]
            dim2_name = self.config_manager.meta_coords['yc'][self.source_name]
        else:
            dim1_name = dim1.attrs[t_label]
            dim2_name = dim2.attrs[t_label]

        if pid == 'xy':
            ax.set_title(dvars.attrs[t_label])
            ax.set_xlabel(dim1_name)
            ax.set_ylabel(dim2_name)
            if 'units' in dvars.attrs:
                cbar.set_label(dvars.attrs['units'])
        fig.squeeze_fig_aspect(self.fig)

    def __get_plot_data(self, field_name, pid=None):
        dim1 = self.config_manager.meta_coords['xc'][self.source_name]
        dim2 = self.config_manager.meta_coords['yc'][self.source_name]
        data2d = None
        if 'yz' in pid:
            dim1 = self.config_manager.meta_coords['yc'][self.source_name]
            dim2 = self.config_manager.meta_coords['zc'][self.source_name]

        # Get the appropriate reader
        reader = self._get_reader(self.source_name)
        if not reader:
            self.logger.error(f"No reader found for source {self.source_name}")
            return None, None, None

        d = reader.get_field(field_name, self.config_manager.findex)['vars']

        if 'yz' in pid:
            data2d = self.__get_yz(d, field_name)
        elif 'xy' in pid:
            data2d = self.__get_xy(d, field_name, 0)
        else:
            self.logger.error(f'[{pid}] plot: Please create specifications file.')
            sys.exit()

        coords = data2d.coords

        return data2d, coords[dim1], coords[dim2]

    def _apply_vertical_level_selection(self, data2d, field_name, level):
        """
        Apply vertical level selection for WRF data, handling staggered grids and pressure levels.
        """
        # Get the vertical dimension name
        zname = self.get_field_dim_name(self.source_name, self.source_data, 'zc',
                                        field_name)

        # If no vertical dimension or it's not in the data, return as is
        if not zname or zname not in data2d.dims:
            return data2d

        # Handle soil layers differently
        if 'soil' in zname:
            soil_layer = 0  # Default to top soil layer
            return eval(f"data2d.isel({zname}=soil_layer)")

        # For atmospheric levels, convert to pressure level
        if level is not None:
            # Find the closest model level to the requested pressure level
            difference_array = np.absolute(self.levs - level)
            index = difference_array.argmin()
            lev_to_plot = self.levs[index]
            self.logger.debug(f'Level to plot: {lev_to_plot} at index {index}')
            return eval(f"data2d.isel({zname}=index)")

        return data2d

    def _get_time_dimension_name(self, d):
        """WRF uses 'Time' as the time dimension."""
        return 'Time' if 'Time' in d.dims else super()._get_time_dimension_name(d)

    @staticmethod
    def _apply_time_selection(original_data, data2d, time_dim, time_lev, field_name,
                              level):
        """WRF applies time selection directly."""
        if time_dim and time_dim in original_data.dims:
            return original_data.isel({time_dim: time_lev}).squeeze()
        return data2d

    def _load_comparison_data(self, map1, map2):
        """Load data from both WRF sources for comparison."""
        source_name1, source_name2 = map1['source_name'], map2['source_name']
        filename1, filename2 = map1['filename'], map2['filename']

        # Get readers for both sources
        reader1 = self._get_reader(source_name1)
        reader2 = self._get_reader(source_name2)

        if not reader1 or not reader2:
            self.logger.error("No suitable readers found for comparison")
            return None

        # Read data from both files
        sdat1 = reader1.read_data(filename1)
        sdat2 = reader2.read_data(filename2)

        if not sdat1 or not sdat2:
            self.logger.error("Failed to read data for comparison")
            return None

        return sdat1, sdat2

    def _get_file_indices(self, source_name1, source_name2, filename1, filename2):
        """Determine file indices for comparison."""
        if source_name1 == source_name2:
            return 0, 1  # Same data source
        else:
            file_index1 = self.config_manager.get_file_index(filename1)
            file_index2 = self.config_manager.get_file_index(filename2)
            return file_index1, file_index2

    def _get_source_name_for_file_index(self, file_index):
        """
        Get the source name for a given file index.
        This handles the case where multiple files come from the same source.
        """
        if hasattr(self.config_manager,
                   'file_list') and file_index in self.config_manager.file_list:
            return self.config_manager.file_list[file_index].get('source_name', 'wrf')

        if hasattr(self.config_manager, 'map_params'):
            for param_key, param_config in self.config_manager.map_params.items():
                if param_key == file_index or param_config.get(
                        'file_index') == file_index:
                    return param_config.get('source_name', 'wrf')

        # If we can't find the source name, default to 'wrf' since we're in the Wrf class!
        return 'wrf'

    def _plot_dest(self, name):
        if self.config_manager.print_to_file:
            output_fname = name + "." + self.config_manager.print_format
            filename = os.path.join(self.config_manager.output_dir, output_fname)
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()

