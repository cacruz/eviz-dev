import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
import warnings
from matplotlib import pyplot as plt

from eviz.lib.data.utils import apply_mean
from eviz.lib.data.utils import apply_conversion
from eviz.lib.autoviz.figure import Figure
from eviz.lib.autoviz.utils import print_map, create_gif
from eviz.models.esm.nuwrf import NuWrf
from eviz.lib.data.pipeline.processor import DataProcessor


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

    def _get_field_to_plot_wrf(self, field_name, file_index, plot_type, figure, time_level, level=None):
        ax = figure.get_axes()
        # self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        dim1, dim2 = self.coord_names(self.source_name, self.source_data,
                                         field_name, plot_type)
        data2d = None
        d = self.source_data['vars'][field_name]
        if 'yz' in plot_type:
            data2d = self._get_yz(d, field_name, time_lev=time_level)
        elif 'xt' in plot_type:
            pass  # TODO!
        elif 'tx' in plot_type:
            pass  # TODO!
        elif 'xy' in plot_type or 'polar' in plot_type:
            data2d = self._get_xy(d, field_name, level=level, time_lev=time_level)
        else:
            pass

        xs, ys, extent, central_lon, central_lat = None, None, [], 0.0, 0.0
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

        xs, ys, extent, central_lon, central_lat = None, None, [], 0.0, 0.0
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
        d = self.source_data['vars'][field_name]
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

    def _get_yz(self, d, name, time_lev=0):
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
        d = self.source_data['vars'][name]
        stag = d.stagger
        if stag == "X":
            data2d = data2d.mean(dim=self.get_model_dim_name(self.source_name, 'xc') + "_stag")
        else:
            data2d = data2d.mean(dim=self.get_model_dim_name(self.source_name, 'xc'))
        return apply_conversion(self.config_manager, data2d, name)

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

    # TODO: put in nuwrf_utils.py
    def _get_field(self, name, data):
        try:
            return data[name]
        except Exception as e:
            self.logger.error('key error: %s, not found' % str(e))
            return None

    def _process_coordinates(self, data2d, dim1, dim2, field_name, plot_type, file_index, figure, ax):
        """
        Process coordinates for WRF plots, handling staggered grids.
        """
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


    def _comparison_plots(self, plotter):
        """Generate comparison plots for paired WRF data sources according to configuration."""
        current_field_index = 0
        self.data2d_list = []  # Initialize list to store data for comparison

        # Process each pair of indices from the comparison configuration
        for idx1, idx2 in zip(self.config_manager.a_list, self.config_manager.b_list):
            # Get map parameters for these indices
            map1 = self.config_manager.config.map_params[idx1]
            map2 = self.config_manager.config.map_params[idx2]

            # Load data from both sources
            source_data_pair = self._load_comparison_data(map1, map2)
            if not source_data_pair:
                continue

            sdat1, sdat2 = source_data_pair
            
            # Initialize p_top for both datasets if needed
            self._init_domain_for_comparison(sdat1, sdat2)
            
            # Create a tuple of both datasets for difference calculations
            sdat = (sdat1, sdat2)

            # Determine file indices
            source_name1, source_name2 = map1['source_name'], map2['source_name']
            filename1, filename2 = map1['filename'], map2['filename']
            file_indices = self._get_file_indices(source_name1, source_name2, filename1, filename2)

            # Process each plot type
            field_name1, field_name2 = map1['field'], map2['field']
            self.field_names = (field_name1, field_name2)

            for pt1, pt2 in zip(map1['to_plot'], map2['to_plot']):
                plot_type = pt1  # Using the first plot type
                self.logger.info(f"Plotting {field_name1} vs {field_name2}, {plot_type} plot")
                self.data2d_list = []  # Reset for each plot type

                if 'xy' in plot_type or 'polar' in plot_type:
                    self._process_xy_comparison_plots(plotter, file_indices,
                                                    current_field_index,
                                                    field_name1, field_name2, plot_type,
                                                    sdat1, sdat2, sdat)
                else:
                    self._process_other_comparison_plots(plotter, file_indices,
                                                    current_field_index,
                                                    field_name1, field_name2,
                                                    plot_type, sdat1, sdat2, sdat)

            current_field_index += 1

    def _init_domain_for_comparison(self, sdat1, sdat2):
        """Initialize domain information for both datasets in a comparison."""
        # Save current source_data
        original_source_data = self.source_data
        
        # Initialize domain for first dataset
        self.source_data = sdat1
        if not hasattr(self, 'p_top') or self.p_top is None:
            try:
                self._init_domain()
            except Exception as e:
                self.logger.warning(f"Could not initialize domain for first dataset: {e}")
        
        # Save first dataset's domain info
        p_top1 = self.p_top
        levs1 = self.levs.copy() if hasattr(self, 'levs') else None
        
        # Initialize domain for second dataset
        self.source_data = sdat2
        if not hasattr(self, 'p_top') or self.p_top is None:
            try:
                self._init_domain()
            except Exception as e:
                self.logger.warning(f"Could not initialize domain for second dataset: {e}")
        
        # Restore original source_data
        self.source_data = original_source_data

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

    def _process_xy_comparison_plots(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, plot_type, sdat1, sdat2, sdat):
        """Process comparison plots for xy or polar plot types."""
        file_index1, file_index2 = file_indices

        # Get levels for the plots
        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            self.logger.warning(f' -> No levels specified for {field_name1}')
            return

        for level in levels:
            figure = Figure(self.config_manager, plot_type)
            ax = figure.get_axes()
            axes_shape = figure.get_gs_geometry()
            self.config_manager.level = level

            if axes_shape == (3, 1):
                self._create_3x1_comparison_plot(plotter, file_indices,
                                                current_field_index,
                                                field_name1, field_name2, figure, ax,
                                                plot_type, sdat1, sdat2, sdat, level)
            elif axes_shape == (2, 2):
                self._create_2x2_comparison_plot(plotter, file_indices,
                                                current_field_index,
                                                field_name1, field_name2, figure,
                                                plot_type, sdat1, sdat2, sdat, level)

            print_map(self.config_manager, plot_type, self.config_manager.findex, figure, level=level)
            self.comparison_plot = False

    def _process_other_comparison_plots(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, plot_type, sdat1, sdat2, sdat):
        """Process comparison plots for other plot types."""
        file_index1, file_index2 = file_indices

        figure = Figure(self.config_manager, plot_type)
        ax = figure.get_axes()
        axes_shape = figure.get_gs_geometry()
        self.config_manager.level = None

        if axes_shape == (3, 1):
            self._create_3x1_comparison_plot(plotter, file_indices, current_field_index,
                                            field_name1, field_name2, figure, ax,
                                            plot_type, sdat1, sdat2, sdat)
        elif axes_shape == (2, 2):
            self._create_2x2_comparison_plot(plotter, file_indices, current_field_index,
                                            field_name1, field_name2, figure,
                                            plot_type, sdat1, sdat2, sdat)

        print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def _create_3x1_comparison_plot(self, plotter, file_indices, current_field_index,
                                field_name1, field_name2, figure, ax,
                                plot_type, sdat1, sdat2, sdat, level=None):
        """Create a 3x1 comparison plot for WRF data."""
        file_index1, file_index2 = file_indices

        # Plot the first dataset
        self._process_comparison_plot(plotter, file_index1, current_field_index,
                                    field_name1,
                                    figure, ax, 0, sdat1, plot_type, level=level)

        # Plot the second dataset
        self._process_comparison_plot(plotter, file_index2, current_field_index,
                                    field_name2,
                                    figure, ax, 1, sdat2, plot_type, level=level)

        # Plot the comparison (difference)
        self.comparison_plot = True
        self._process_comparison_plot(plotter, file_index1, current_field_index,
                                    field_name1,
                                    figure, ax, 2, sdat, plot_type, level=level)

    def _create_2x2_comparison_plot(self, plotter, file_indices, current_field_index,
                                field_name1, field_name2, figure,
                                plot_type, sdat1, sdat2, sdat, level=None):
        """Create a 2x2 comparison plot for WRF data."""
        file_index1, file_index2 = file_indices

        # Plot the first dataset in the top-left
        self._process_comparison_plot_2x2(plotter, file_index1, current_field_index,
                                        field_name1,
                                        figure, [0, 0], 0, sdat1, plot_type,
                                        level=level)

        # Plot the second dataset in the top-right
        self._process_comparison_plot_2x2(plotter, file_index2, current_field_index,
                                        field_name2,
                                        figure, [0, 1], 1, sdat2, plot_type,
                                        level=level)

        # Plot comparison in the bottom row
        self.comparison_plot = True
        self._process_comparison_plot_2x2(plotter, file_index1, current_field_index,
                                        field_name1,
                                        figure, [1, 0], 2, sdat, plot_type, level=level)
        self._process_comparison_plot_2x2(plotter, file_index1, current_field_index,
                                        field_name1,
                                        figure, [1, 1], 2, sdat, plot_type, level=level)

    def _process_comparison_plot(self, plotter, file_index, current_field_index, field_name, 
                                figure, ax, ax_index, source_data, plot_type, level=None):
        """Process a comparison plot."""
        self.config_manager.config._findex = file_index
        
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        figure.set_ax_opts_diff_field(ax[ax_index])
        
        # Get field to plot
        field_to_plot = self._get_field_to_plot_compare(source_data, field_name, file_index,
                                                    plot_type, figure, ax=ax[ax_index], level=level)
        
        # Plot the field
        plotter.comparison_plots(self.config_manager, field_to_plot, level=level)

    def _process_comparison_plot_2x2(self, plotter, file_index, current_field_index, field_name, 
                                    figure, gsi, ax_index, source_data, plot_type, level=None):
        """Process a 2x2 comparison plot for WRF data."""
        fig, axes = figure.get_fig_ax()
        ax1 = axes[gsi[0], gsi[1]] if isinstance(axes, np.ndarray) else plt.subplot(figure.gs[gsi[0], gsi[1]])
        figure.set_ax_opts_diff_field(ax1)
        
        # Get field to plot
        field_to_plot = self._get_field_to_plot_compare(source_data, field_name, file_index,
                                                    plot_type, figure, ax=ax1, level=level)
        
        # Plot the field
        plotter.comparison_plots(self.config_manager, field_to_plot, level=level)

    def _get_field_to_plot_compare(self, source_data, field_name, file_index, 
                                plot_type, figure, ax=None, level=None):
        """Get field data for comparison plots with WRF-specific handling."""
        if ax is None:
            ax = figure.get_axes()
        
        # For difference field
        if self.comparison_plot and self.config_manager.ax_opts['is_diff_field']:
            proc = DataProcessor(self.config_manager, self.data2d_list)
            data2d, xx, yy = proc.regrid(plot_type)
            return data2d, xx, yy, self.field_names[0], plot_type, file_index, figure, ax
        
        # Get source name safely - don't use source_names[file_index] directly
        source_name = self._get_source_name_for_file_index(file_index)
        
        # For regular fields
        if isinstance(source_data, tuple):
            # This is a tuple of two datasets for comparison
            sdat1, sdat2 = source_data
            self.source_data = sdat1  # Temporarily set source_data for dimension handling
            
            source_name = self._get_source_name_for_file_index(file_index)
            dim1, dim2 = self.coord_names(source_name, source_data, field_name, plot_type)
  
            # Get time level
            time_level = self.config_manager.ax_opts['time_lev']
            if isinstance(time_level, str) and time_level == 'all':
                time_level = 0  # Default to first time level for comparison
            
            # Extract data based on plot type
            d1 = sdat1['vars'][field_name]
            d2 = sdat2['vars'][field_name]
            
            # Process data based on plot type
            if 'yz' in plot_type:
                data2d1 = self._get_yz(d1, field_name, time_lev=time_level)
                
                # Temporarily switch source_data for second dataset
                temp_source_data = self.source_data
                self.source_data = sdat2
                data2d2 = self._get_yz(d2, field_name, time_lev=time_level)
                self.source_data = temp_source_data
                
                # Calculate difference
                data2d = data2d1 - data2d2
                
            elif 'xy' in plot_type or 'polar' in plot_type:
                data2d1 = self._get_xy(d1, field_name, level=level, time_lev=time_level)
                
                # Temporarily switch source_data for second dataset
                temp_source_data = self.source_data
                self.source_data = sdat2
                data2d2 = self._get_xy(d2, field_name, level=level, time_lev=time_level)
                self.source_data = temp_source_data
                
                # Calculate difference
                data2d = data2d1 - data2d2
                
            else:
                # Other plot types (xt, tx)
                self.logger.warning(f"Plot type {plot_type} not fully supported for comparison")
                return None
            
            # Process coordinates
            if 'xt' in plot_type or 'tx' in plot_type:
                return data2d, None, None, field_name, plot_type, file_index, figure, ax
            elif 'yz' in plot_type:
                # For YZ plots, use latitude and pressure levels
                xs = np.array(self._get_field(dim1[0], d1)[0, :][:, 0])
                ys = self.levs  # Pressure levels
                return data2d, xs, ys, field_name, plot_type, file_index, figure, ax
            else:
                # For XY plots, use longitude and latitude
                xs = np.array(self._get_field(dim1[0], data2d1)[0, :])
                ys = np.array(self._get_field(dim2[0], data2d1)[:, 0])
                
                # Store data for difference calculation
                self.data2d_list.append(data2d)
                
                return data2d, xs, ys, field_name, plot_type, file_index, figure, ax
        else:
            # Single dataset
            self.source_data = source_data  # Set source_data for dimension handling
            
            # Get dimension names using source_name instead of indexing into source_names
            dim1, dim2 = self.coord_names(source_name, source_data, field_name, plot_type)
            
            # Get time level
            time_level = self.config_manager.ax_opts['time_lev']
            if isinstance(time_level, str) and time_level == 'all':
                time_level = 0  # Default to first time level for comparison
            
            # Extract data based on plot type
            d = source_data['vars'][field_name]
            
            # Process data based on plot type
            if 'yz' in plot_type:
                data2d = self._get_yz(d, field_name, time_lev=time_level)
            elif 'xy' in plot_type or 'polar' in plot_type:
                data2d = self._get_xy(d, field_name, level=level, time_lev=time_level)
            else:
                # Other plot types (xt, tx)
                self.logger.warning(f"Plot type {plot_type} not fully supported for comparison")
                return None
            
            # Store data for difference calculation
            self.data2d_list.append(data2d)
            
            # Process coordinates
            if 'xt' in plot_type or 'tx' in plot_type:
                return data2d, None, None, field_name, plot_type, file_index, figure, ax
            elif 'yz' in plot_type:
                # For YZ plots, use latitude and pressure levels
                xs = np.array(self._get_field(dim1[0], d)[0, :][:, 0])
                ys = self.levs  # Pressure levels
                return data2d, xs, ys, field_name, plot_type, file_index, figure, ax
            else:
                # For XY plots, use longitude and latitude
                xs = np.array(self._get_field(dim1[0], data2d)[0, :])
                ys = np.array(self._get_field(dim2[0], data2d)[:, 0])
                return data2d, xs, ys, field_name, plot_type, file_index, figure, ax

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
    

    def _side_by_side_plots(self, plotter):
        """
        Generate side-by-side comparison plots (2x1 subplots) without difference.
        """
        current_field_index = 0
        self.data2d_list = []  # Initialize list to store data for comparison

        # Process each pair of indices from the comparison configuration
        for idx1, idx2 in zip(self.config_manager.a_list, self.config_manager.b_list):
            # Get map parameters for these indices
            map1 = self.config_manager.config.map_params[idx1]
            map2 = self.config_manager.config.map_params[idx2]

            # Load data from both sources
            source_data_pair = self._load_comparison_data(map1, map2)
            if not source_data_pair:
                continue

            sdat1, sdat2 = source_data_pair
            
            # Initialize p_top for both datasets if needed
            self._init_domain_for_comparison(sdat1, sdat2)

            # Determine file indices
            source_name1, source_name2 = map1['source_name'], map2['source_name']
            filename1, filename2 = map1['filename'], map2['filename']
            file_indices = self._get_file_indices(source_name1, source_name2, filename1, filename2)

            # Process each plot type
            field_name1, field_name2 = map1['field'], map2['field']
            self.field_names = (field_name1, field_name2)

            for pt1, pt2 in zip(map1['to_plot'], map2['to_plot']):
                plot_type = pt1  # Using the first plot type
                self.logger.info(f"Plotting {field_name1} vs {field_name2} side by side, {plot_type} plot")
                self.data2d_list = []  # Reset for each plot type

                if 'xy' in plot_type or 'polar' in plot_type:
                    self._process_xy_side_by_side_plots(plotter, file_indices,
                                                    current_field_index,
                                                    field_name1, field_name2, plot_type,
                                                    sdat1, sdat2)
                else:
                    self._process_other_side_by_side_plots(plotter, file_indices,
                                                        current_field_index,
                                                        field_name1, field_name2,
                                                        plot_type, sdat1, sdat2)

            current_field_index += 1

    def _process_xy_side_by_side_plots(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, plot_type, sdat1, sdat2):
        """Process side-by-side comparison plots for xy or polar plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels
        
        # Get levels for the plots
        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            self.logger.warning(f' -> No levels specified for {field_name1}')
            return
        
        for level in levels:
            # Create a figure with 2x1 subplots (side by side)
            figure = Figure(self.config_manager, plot_type, nrows=nrows, ncols=ncols)
            ax = figure.get_fig_ax()
            self.config_manager.level = level
            
            # Create the 2x1 side-by-side comparison plot
            self._create_2x1_side_by_side_plot(plotter, file_indices,
                                            current_field_index,
                                            field_name1, field_name2, figure, ax,
                                            plot_type, sdat1, sdat2, level)
            
    def _process_other_side_by_side_plots(self, plotter, file_indices, current_field_index,
                                        field_name1, field_name2, plot_type, sdat1, sdat2):
        """Process side-by-side comparison plots for other plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels
        
        # Create a figure with 2x1 subplots (side by side)
        figure = Figure(self.config_manager, plot_type, nrows=nrows, ncols=ncols)
        ax = figure.get_axes()
        self.config_manager.level = None
        
        # Create the 2x1 side-by-side comparison plot
        self._create_2x1_side_by_side_plot(plotter, file_indices, current_field_index,
                                        field_name1, field_name2, figure, ax,
                                        plot_type, sdat1, sdat2)
        

    def _create_2x1_side_by_side_plot(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, figure, ax,
                                    plot_type, sdat1, sdat2, level=None):
        """
        Create a 2x1 side-by-side comparison plot for WRF data.
        
        The layout is:
        - Left subplot: First dataset
        - Right subplot: Second dataset
        """
        file_index1, file_index2 = file_indices
        
        # Plot the first dataset in the left subplot
        self.comparison_plot = False
        self._process_side_by_side_plot(plotter, file_index1, current_field_index,
                                    field_name1,
                                    figure, ax, 0, sdat1, plot_type, level=level)
        
        # Plot the second dataset in the right subplot
        self._process_side_by_side_plot(plotter, file_index2, current_field_index,
                                    field_name2,
                                    figure, ax, 1, sdat2, plot_type, level=level)

    def _process_side_by_side_plot(self, plotter, file_index, current_field_index, field_name, 
                                figure, ax, ax_index, source_data, plot_type, level=None):
        """Process a side-by-side plot for WRF data."""
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        # figure.set_ax_opts_diff_field(ax[ax_index])
        
        # # Set up the axis
        # if isinstance(ax, list):
        #     current_ax = ax[ax_index]
        # else:
        #     current_ax = ax
        
        # Get field to plot
        field_to_plot = self._get_field_to_plot_side_by_side(source_data, field_name, file_index,
                                                        plot_type, figure, level=level)
        
        # Check which type of plotter we're using and call the appropriate method
        if hasattr(plotter, 'single_plots'):
            # SinglePlotter
            plotter.single_plots(self.config_manager, field_to_plot, level=level)
        elif hasattr(plotter, 'comparison_plots'):
            # ComparisonPlotter
            plotter.comparison_plots(self.config_manager, field_to_plot, level=level)
        else:
            # Fallback - try to call plot directly
            self.logger.warning(f"Unknown plotter type: {type(plotter).__name__}. Trying to call plot method.")
            if hasattr(plotter, 'plot'):
                plotter.plot(self.config_manager, field_to_plot, level=level)
            else:
                self.logger.error(f"Plotter {type(plotter).__name__} has no plot method.")

        # Save the plot
        print_map(self.config_manager, plot_type, self.config_manager.findex, figure, level=level)

    def _get_field_to_plot_side_by_side(self, source_data, field_name, file_index, 
                                    plot_type, figure, ax=None, level=None):
        """Get field data for side-by-side plots with WRF-specific handling."""
        if ax is None:
            ax = figure.get_axes()
        
        # Get source name safely
        source_name = self._get_source_name_for_file_index(file_index)
        
        # Set source_data for dimension handling
        self.source_data = source_data
        
        # Get dimension names
        dim1, dim2 = self.coord_names(source_name, source_data, field_name, plot_type)
        
        # Get time level
        time_level = self.config_manager.ax_opts['time_lev']
        if isinstance(time_level, str) and time_level == 'all':
            time_level = 0  # Default to first time level for comparison
        
        # Extract data based on plot type
        d = source_data['vars'][field_name]
        
        # Process data based on plot type
        if 'yz' in plot_type:
            data2d = self._get_yz(d, field_name, time_lev=time_level)
        elif 'xy' in plot_type or 'polar' in plot_type:
            data2d = self._get_xy(d, field_name, level=level, time_lev=time_level)
        else:
            # Other plot types (xt, tx)
            self.logger.warning(f"Plot type {plot_type} not fully supported for side-by-side comparison")
            return None
        
        # Process coordinates
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        elif 'yz' in plot_type:
            # For YZ plots, use latitude and pressure levels
            xs = np.array(self._get_field(dim1[0], d)[0, :][:, 0])
            ys = self.levs  # Pressure levels
            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax
        else:
            # For XY plots, use longitude and latitude
            xs = np.array(self._get_field(dim1[0], data2d)[0, :])
            ys = np.array(self._get_field(dim2[0], data2d)[:, 0])
            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax

    def get_field_dim_name(self, source_name: str, source_data: dict, dim_name: str, field_name: str):
        d = source_data['vars'][field_name]
        field_dims = list(d.dims)   # use dims only!?
        names = self.get_model_dim_name(source_name, dim_name).split(',')
        common = list(set(names).intersection(field_dims))
        dim = list(common)[0] if common else None
        return dim

    def get_model_dim_name(self, source_name: str, dim_name: str):
        try:
            dim = self.config_manager.meta_coords[dim_name][source_name]['dim']
            return dim
        except KeyError:
            return None
