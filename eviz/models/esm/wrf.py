import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr
import logging
import warnings
from matplotlib import pyplot as plt
from eviz.lib.data.utils import apply_mean
from eviz.lib.data.utils import apply_conversion
from eviz.lib.autoviz.figure import Figure
from eviz.lib.autoviz.utils import print_map, create_gif
from eviz.models.esm.nuwrf import NuWrf

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class Wrf(NuWrf):
    """ Define NUWRF specific model data and functions.
    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.comparison_plot = False
        self.source_name = 'wrf'

    def _init_domain(self):
        """ Approximate unknown fields """
        # Create sigma->pressure dictionary
        # model_top + sigma * (surf_pressure - model_top)
        self.p_top = self.source_data['vars']['P_TOP'][0] / 1e5  # mb
        self.eta_full = np.array(self.source_data['vars']['ZNW'][0])
        self.eta_mid = np.array(self.source_data['vars']['ZNU'][0])
        self.levf = np.empty(len(self.eta_full))
        self.levs = np.empty(len(self.eta_mid))
        i = 0
        for s in self.eta_full:
            if s > 0:
                self.levf[i] = int(self.p_top + s * (1000 - self.p_top))
            else:
                self.levf[i] = self.p_top + s * (1000 - self.p_top)
            i += 1
        i = 0
        for s in self.eta_mid:
            if s > 0:
                self.levs[i] = int(self.p_top + s * (1000 - self.p_top))
            else:
                self.levs[i] = self.p_top + s * (1000 - self.p_top)
            i += 1

    @property
    def global_attrs(self):
        return self._global_attrs

    def _simple_plots(self, plotter):
        map_params = self.config_manager.map_params
        field_num = 0
        self.config_manager.findex = 0
        for i in map_params.keys():
            field_name = map_params[i]['field']
            source_name = map_params[i]['source_name']
            self.source_name = source_name
            filename = map_params[i]['filename']
            file_index = self.config_manager.get_file_index(filename)
            
            # Get the appropriate reader
            reader = self._get_reader(source_name)
            if not reader:
                self.logger.error(f"No reader found for source {source_name}")
                continue
                
            self.source_data = reader.read_data(filename)
            if not self.source_data:
                self.logger.error(f"Failed to read data from {filename}")
                continue
                
            self._global_attrs = self.set_global_attrs(source_name, self.source_data['attrs'])
            if not self.p_top:
                self._init_domain()
            self.config_manager.findex = file_index
            self.config_manager.pindex = field_num
            self.config_manager.axindex = 0
            for pt in map_params[i]['to_plot']:
                self.logger.info(f"Plotting {field_name}, {pt} plot")
                field_to_plot = self._get_field_for_simple_plot(field_name, pt)
                plotter.simple_plot(self.config_manager, field_to_plot)
            field_num += 1
    
    def _single_plots(self, plotter):
        for s in range(len(self.config_manager.source_names)):
            map_params = self.config_manager.map_params
            field_num = 0
            for i in map_params.keys():
                source_name = map_params[i]['source_name']
                if source_name == self.config_manager.source_names[s]:
                    field_name = map_params[i]['field']
                    self.source_name = source_name
                    filename = map_params[i]['filename']
                    file_index = field_num  # self.config_manager.get_file_index(filename)
                    
                    # Get the appropriate reader
                    reader = self._get_reader(source_name)
                    if not reader:
                        self.logger.error(f"No reader found for source {source_name}")
                        continue
                        
                    self.source_data = reader.read_data(filename)
                    if not self.source_data:
                        self.logger.error(f"Failed to read data from {filename}")
                        continue
                        
                    self._global_attrs = self.source_data['attrs']
                    if not self.p_top:
                        self._init_domain()

                    self.config_manager.findex = file_index
                    self.config_manager.pindex = field_num
                    self.config_manager.axindex = 0
                    for pt in map_params[i]['to_plot']:
                        self.logger.info(f"Plotting {field_name}, {pt} plot")
                        figure = Figure(self.config_manager, pt)
                        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
                        time_level = self.config_manager.ax_opts['time_lev']
                        num_times = 1
                        d = self.source_data['vars'][field_name]
                        
                        # Get the time dimension name - WRF uses 'Time', not 'time'
                        time_dim = 'Time' if 'Time' in d.dims else 'time'
                        
                        if time_level == 'all':
                            num_times = d.dims[time_dim] if time_dim in d.dims else 1
                            time_levels = range(num_times)
                        else:  # one value
                            time_levels = [time_level]
                        
                        if 'xy' in pt:
                            levels = self.config_manager.get_levels(field_name, pt + 'plot')
                            if not levels:
                                self.logger.warning(f' -> No levels specified for {field_name}')
                                continue
                            for level in levels:
                                self.logger.info(f' -> Processing {num_times} time levels')
                                for t in time_levels:
                                    self.config_manager.time_level = t
                                    # Handle WRF time variable
                                    if hasattr(d, 'XTIME'):
                                        if time_dim in d.XTIME.dims:
                                            real_time = d.XTIME.isel({time_dim: t}).values
                                        else:
                                            # If XTIME doesn't have the time dimension, create a dummy time
                                            real_time = pd.Timestamp('2000-01-01')
                                    else:
                                        # If XTIME is not available, check for other time variables
                                        time_var = None
                                        for var_name in ['Times', 'time', 'Time']:
                                            if var_name in self.source_data['vars']:
                                                time_var = self.source_data['vars'][var_name]
                                                break
                                        
                                        if time_var is not None and time_dim in time_var.dims:
                                            real_time = time_var.isel({time_dim: t}).values
                                        else:
                                            # Last resort - create a dummy time
                                            real_time = pd.Timestamp('2000-01-01')
                                    
                                    # Convert time to a readable format
                                    try:
                                        real_time_readable = pd.to_datetime(real_time).strftime('%Y-%m-%d %H')
                                    except:
                                        real_time_readable = f"Time step {t}"
                                        
                                    self.config_manager.real_time = real_time_readable
                                    field_to_plot = self._get_field_to_plot_wrf(field_name, file_index, pt, figure, t,
                                                                            level=level)
                                    plotter.single_plots(self.config_manager, field_to_plot=field_to_plot, level=level)
                                    print_map(self.config_manager, pt, self.config_manager.findex, figure, level=level)

                        else:
                            for t in time_levels:
                                self.config_manager.time_level = t
                                # Handle WRF time variable (same as above)
                                if hasattr(d, 'XTIME'):
                                    if time_dim in d.XTIME.dims:
                                        real_time = d.XTIME.isel({time_dim: t}).values
                                    else:
                                        real_time = pd.Timestamp('2000-01-01')
                                else:
                                    time_var = None
                                    for var_name in ['Times', 'time', 'Time']:
                                        if var_name in self.source_data['vars']:
                                            time_var = self.source_data['vars'][var_name]
                                            break
                                    
                                    if time_var is not None and time_dim in time_var.dims:
                                        real_time = time_var.isel({time_dim: t}).values
                                    else:
                                        real_time = pd.Timestamp('2000-01-01')
                                
                                try:
                                    real_time_readable = pd.to_datetime(real_time).strftime('%Y-%m-%d %H')
                                except:
                                    real_time_readable = f"Time step {t}"
                                    
                                self.config_manager.real_time = real_time_readable
                                field_to_plot = self._get_field_to_plot_wrf(field_name, file_index, pt, figure, t,
                                                                        level=None)
                                plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
                                print_map(self.config_manager, pt, self.config_manager.findex, figure)

                    field_num += 1
        if self.config_manager.make_gif:
            create_gif(self.config_manager)

    def _process_coordinates(self, data2d, dim1, dim2, field_name, plot_type, file_index, figure, ax):
        """
        Process coordinates for WRF plots
        """
        # TODO: Use the config_manager to get the coordinate names
        if plot_type == 'xy':
            dim1 = 'XLONG'
            dim2 = 'XLAT'
        elif plot_type == 'yz': 
            dim1 = 'XLAT'

        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        elif 'yz' in plot_type:
            # For YZ plots, use latitude and pressure levels
            xs = np.array(self._get_field(dim1[0], data2d)[0, :][:, 0])
            ys = self.levs  # Pressure levels   
            latN = max(xs[:])
            latS = min(xs[:])
            self.config_manager.ax_opts['extent'] = [None, None, latS, latN]
            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax
        else:
            # For XY plots, use longitude and latitude
            xs = np.array(self._get_field(dim1, data2d)[0, :])
            ys = np.array(self._get_field(dim2, data2d)[:, 0])
            latN = max(ys[:])
            latS = min(ys[:])
            lonW = min(xs[:])
            lonE = max(xs[:])
            self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
            self.config_manager.ax_opts['central_lon'] = np.mean(self.config_manager.ax_opts['extent'][:2])
            self.config_manager.ax_opts['central_lat'] = np.mean(self.config_manager.ax_opts['extent'][2:])
            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax


    def _get_field_to_plot_wrf(self, field_name, file_index, plot_type, figure, time_level, level=None):
        ax = figure.get_axes()
        dim1, dim2 = self.coord_names(self.source_name, self.source_data,
                                         field_name, plot_type)
        data2d = None
        d = self.source_data['vars'][field_name]
        if 'yz' in plot_type:
            data2d = self._get_yz(d, time_lev=time_level)
        elif 'xt' in plot_type:
            pass  # TODO!
        elif 'tx' in plot_type:
            pass  # TODO!
        elif 'xy' in plot_type or 'polar' in plot_type:
            data2d = self._get_xy(d, level=level, time_lev=time_level)
        else:
            pass

        xs, ys = None, None
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        elif 'yz' in plot_type:
            xs = np.array(self._get_field(dim1[0], d)[0, :][:, 0])
            ys = self.levs
            latN = max(xs[:])
            latS = min(xs[:])
            self.config_manager.ax_opts['extent'] = [None, None, latS, latN]
            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax
        else:
            xs = np.array(self._get_field(dim1[0], data2d)[0, :])
            ys = np.array(self._get_field(dim2[0], data2d)[:, 0])
            latN = max(ys[:])
            latS = min(ys[:])
            lonW = min(xs[:])
            lonE = max(xs[:])
            self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
            self.config_manager.ax_opts['central_lon'] = np.mean(self.config_manager.ax_opts['extent'][:2])
            self.config_manager.ax_opts['central_lat'] = np.mean(self.config_manager.ax_opts['extent'][2:])
            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax

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


    def _get_xy(self, d, level, time_lev):
        """ Extract XY slice from N-dim data field"""
        if d is None:
            return
        if level:
            level = int(level)

        self.logger.debug(f"Selecting time level: {time_lev}")
        data2d = eval(f"d.isel({self.get_model_dim_name(self.source_name, 'tc')}=time_lev)")
        data2d = data2d.squeeze()
        zname = self.get_field_dim_name('wrf', d, 'zc')
        if zname in data2d.dims:
            # TODO: Make soil_layer configurable
            soil_layer = 0
            if 'soil' in zname:
                data2d = eval(f"data2d.isel({zname}=soil_layer)")
            else:
                difference_array = np.absolute(self.levs - level)
                index = difference_array.argmin()
                lev_to_plot = self.levs[index]
                self.logger.debug(f'Level to plot: {lev_to_plot} at index {index}')
                data2d = eval(f"data2d.isel({zname}=index)")
        return apply_conversion(self.config_manager, data2d, d.name)

    def _get_yz(self, d, time_lev=0):
        """ Create YZ slice from N-dim data field"""
        d = d.squeeze()
        if self.get_model_dim_name(self.source_name, 'tc') in d.dims:
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
            data2d = data2d.mean(dim=self.get_model_dim_name(self.source_name, 'xc') + "_stag")
        else:
            data2d = data2d.mean(dim=self.get_model_dim_name(self.source_name, 'xc'))
        return apply_conversion(self.config_manager, data2d, d.name)

    def _get_xt(self, d, name, time_lev, level=None):
        """ Extract time-series from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 1D (time) series
        """
        d_temp = d
        if d_temp is None:
            return
        
        xtime = d.XTIME
        
        num_times = xtime.size
        self.logger.info(f"'{name}' field has {num_times} time levels")
        print(xr.ALL_DIMS)
        if isinstance(time_lev, list):
            self.logger.info(f"Computing time series on {time_lev} time range")
            data2d = eval(f"d_temp.isel({self.config.get_model_dim_name('tc')}=slice(time_lev))")
        else:
            data2d = d_temp.squeeze()

        if 'mean_type' in self.config.spec_data[name]['xtplot']:
            mean_type = self.config.spec_data[name]['xtplot']['mean_type']
            self.logger.info(f"Averaging method: {mean_type}")
            # annual:
            if mean_type == 'point_sel':
                xc = self.config.spec_data[name]['xtplot']['point_sel'][0]
                yc = self.config.spec_data[name]['xtplot']['point_sel'][1]
                data2d = data2d.sel(lon=xc, lat=yc, method='nearest')
            elif mean_type == 'area_sel':
                x1 = self.config.spec_data[name]['xtplot']['area_sel'][0]
                x2 = self.config.spec_data[name]['xtplot']['area_sel'][1]
                y1 = self.config.spec_data[name]['xtplot']['area_sel'][2]
                y2 = self.config.spec_data[name]['xtplot']['area_sel'][3]
                data2d = data2d.sel(lon=np.arange(x1, x2, 0.5), lat=np.arange(y1, y2, 0.5), method='nearest')
                data2d = data2d.mean(dim=(self.config.get_model_dim_name('xc'), self.config.get_model_dim_name('yc')))
            elif mean_type in ['year', 'season', 'month']:
                data2d = data2d.groupby(self.config.get_model_dim_name('tc') + '.' + mean_type).mean(
                    dim=self.config.get_model_dim_name('tc'), keep_attrs=True)
            else:
                data2d = data2d.groupby(self.config.get_model_dim_name('tc')).mean(dim=xr.ALL_DIMS, keep_attrs=True)
                if 'mean_type' in self.config.spec_data[name]['xtplot']:
                    if self.config.spec_data[name]['xtplot']['mean_type'] == 'rolling':
                        window_size = 5
                        if 'window_size' in self.config.spec_data[name]['xtplot']:
                            window_size = self.config.spec_data[name]['xtplot']['window_size']
                        self.logger.info(f" -- smoothing window size: {window_size}")
                        kernel = np.ones(window_size) / window_size
                        convolved_data = np.convolve(data2d, kernel, mode="same")
                        data2d = xr.DataArray(convolved_data, dims=self.config.get_model_dim_name('tc'),
                                              coords=data2d.coords)

        else:
            data2d = data2d.groupby(self.config.get_model_dim_name('tc')).mean(dim=xr.ALL_DIMS, keep_attrs=True)

        if 'level' in self.config.spec_data[name]['xtplot']:
            level = int(self.config.spec_data[name]['xtplot']['level'])
            lev_to_plot = int(np.where(data2d.coords[self.config.get_model_dim_name('zc')].values == level)[0])
            data2d = data2d[:, lev_to_plot].squeeze()

        return apply_conversion(self.config, data2d, name)
    
    def _select_yrange(self, data2d, name):
        """ Select a range of vertical levels"""
        if 'zrange' in self.config_manager.spec_data[name]['yzplot']:
            if not self.config_manager.spec_data[name]['yzplot']['zrange']:
                return data2d
            lo_z = self.config_manager.spec_data[name]['yzplot']['zrange'][0]
            hi_z = self.config_manager.spec_data[name]['yzplot']['zrange'][1]
            if hi_z >= lo_z:
                self.logger.error(f"Upper level value ({hi_z}) must be less than low level value ({lo_z})")
                return
            lev = self.get_model_dim_name(self.source_name, 'zc')
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
        zname = self.get_field_dim_name(self.source_name, self.source_data, 'zc', field_name)
        
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

    def _apply_time_selection(self, original_data, data2d, time_dim, time_lev, field_name, level):
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
        # Try to get source name from file_list
        if hasattr(self.config_manager, 'file_list') and file_index in self.config_manager.file_list:
            return self.config_manager.file_list[file_index].get('source_name', 'wrf')
        
        # Try to get source name from map_params
        if hasattr(self.config_manager, 'map_params'):
            for param_key, param_config in self.config_manager.map_params.items():
                if param_key == file_index or param_config.get('file_index') == file_index:
                    return param_config.get('source_name', 'wrf')
        
        # If we can't find the source name, default to 'wrf' since we're in the Wrf class
        return 'wrf'
    