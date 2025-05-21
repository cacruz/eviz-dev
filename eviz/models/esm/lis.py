import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eviz.lib.data.utils import apply_mean
from eviz.lib.data.utils import apply_conversion
from eviz.lib.autoviz.figure import Figure
from eviz.models.esm.nuwrf import NuWrf
from eviz.lib.autoviz.utils import print_map

warnings.filterwarnings("ignore")


@dataclass
class Lis(NuWrf):
    """ Define LIS specific model data and functions.
    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.comparison_plot = False
        self.source_name = 'lis'
        
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
                    self.config_manager.findex = file_index
                    self.config_manager.pindex = field_num
                    self.config_manager.axindex = 0
                    for pt in map_params[i]['to_plot']:
                        self.logger.info(f"Plotting {field_name}, {pt} plot")
                        figure = Figure(self.config_manager, pt)
                        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
                        time_level = self.config_manager.ax_opts['time_lev']
                        d = self.source_data['vars'][field_name]
                        
                        # Get the time dimension name - try several common variants
                        time_dim = None
                        for dim_name in ['time', 'Time', 't', 'T']:
                            if dim_name in d.dims:
                                time_dim = dim_name
                                break
                        
                        # If no time dimension found, create a dummy
                        if time_dim is None:
                            self.logger.warning(f"No time dimension found in {field_name}")
                            num_times = 1
                        else:
                            num_times = d.dims[time_dim] if time_level == 'all' else 1
                        
                        time_levels = range(num_times)

                        # vertical levels (soil moisture or soil temperature)
                        levels = self.config_manager.get_levels(field_name, pt + 'plot')

                        if not levels:
                            self.logger.warning(f' -> No levels specified for {field_name}')
                            for t in time_levels:
                                self.config_manager.time_level = t
                                
                                # Get time value safely
                                real_time = self._get_time_value(d, t, time_dim)
                                
                                # Convert time to a readable format
                                try:
                                    real_time_readable = pd.to_datetime(real_time).strftime('%Y-%m-%d %H')
                                except:
                                    real_time_readable = f"Time step {t}"
                                    
                                self.config_manager.real_time = real_time_readable
                                
                                field_to_plot = self._get_field_to_plot(field_name, file_index, pt, figure)
                                plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
                                print_map(self.config_manager, pt, self.config_manager.findex, figure)
                        else:
                            for level in levels:
                                self.logger.info(f' -> Processing {num_times} time levels')
                                for t in time_levels:
                                    self.config_manager.time_level = t
                                    
                                    # Get time value safely
                                    real_time = self._get_time_value(d, t, time_dim)
                                    
                                    # Convert time to a readable format
                                    try:
                                        real_time_readable = pd.to_datetime(real_time).strftime('%Y-%m-%d %H')
                                    except:
                                        real_time_readable = f"Time step {t}"
                                        
                                    self.config_manager.real_time = real_time_readable
                                    
                                    field_to_plot = self._get_field_to_plot(field_name, file_index, pt, figure, level=level)
                                    plotter.single_plots(self.config_manager, field_to_plot=field_to_plot, level=level)
                                    print_map(self.config_manager, pt, self.config_manager.findex, figure, level=level)

                    field_num += 1

    def _get_time_value(self, data_array, time_index, time_dim=None):
        """
        Get the time value for the given index.
        
        Args:
            data_array: The data array containing time values
            time_index: The index to extract
            time_dim: The name of the time dimension
            
        Returns:
            The time value or a default timestamp if unavailable
        """
        # If no time dimension, return a default timestamp
        if time_dim is None:
            return pd.Timestamp('2000-01-01')
        
        # Try different approaches to get the time value
        try:
            # First try: Check if there's a time coordinate with the time_dim
            if time_dim in data_array.coords:
                return data_array.coords[time_dim].values[time_index]
            
            # Second try: Check for a time variable attribute
            if hasattr(data_array, 'time') and hasattr(data_array.time, 'isel'):
                return data_array.time.isel({time_dim: time_index}).values
            
            # Third try: Check for time-related variables in source_data
            for var_name in ['time', 'Time', 'times', 'Times']:
                if var_name in self.source_data['vars']:
                    time_var = self.source_data['vars'][var_name]
                    if time_dim in time_var.dims:
                        return time_var.isel({time_dim: time_index}).values
            
            # Last resort: create a dummy timestamp
            return pd.Timestamp('2000-01-01') + pd.Timedelta(days=time_index)
        
        except Exception as e:
            self.logger.warning(f"Error getting time value: {e}")
            return pd.Timestamp('2000-01-01') + pd.Timedelta(days=time_index)
        
    def _get_field_to_plot(self, field_name, file_index, plot_type, figure, level=None):
        ax = figure.get_axes()
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        data2d = None
        d = self.source_data['vars'][field_name]
        
        # Get the time level from config
        time_lev = self.config_manager.time_level if hasattr(self.config_manager, 'time_level') else 0
        
        if 'xt' in plot_type:
            data2d = self._get_xt(d, field_name, time_lev=self.ax_opts['time_lev'])
        elif 'xy' in plot_type:
            # Pass all required arguments to _get_xy
            data2d = self._get_xy(d, level=level,
                        time_lev=self.config_manager.ax_opts.get('time_lev',
                                                                0))
        else:
            pass

        xs, ys, extent, central_lon, central_lat = None, None, [], 0.0, 0.0
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        else:
            lon = self.source_data['vars'][self.get_model_coord_name(self.source_name, 'xc')]
            lat = self.source_data['vars'][self.get_model_coord_name(self.source_name, 'yc')]
            xs = np.array(lon[0, :])
            ys = np.array(lat[:, 0])
            # Some LIS coordinates are NaN. The following workaround fills out those elements
            # with reasonable values:
            idx = np.argwhere(np.isnan(xs))
            for i in idx:
                xs[i] = xs[i - 1] + self._global_attrs["DX"] / 1000.0 / 100.0
            idx = np.argwhere(np.isnan(ys))
            for i in idx:
                ys[i] = ys[i - 1] + self._global_attrs["DY"] / 1000.0 / 100.0

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
        
        # Default values for level and time_lev
        level = 0
        time_lev = 0
        
        if 'xt' in plot_type:
            data2d = self._get_xt(d, field_name, time_lev=self.ax_opts['time_lev'])
        elif 'tx' in plot_type:
            data2d = self._get_tx(d, field_name, level=None, time_lev=self.ax_opts['time_lev'])
        elif 'xy' in plot_type:
            # Pass all required arguments to _get_xy
            data2d = self._get_xy(d, field_name, level, time_lev)
        else:
            pass

        xs, ys = None, None
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, xs, ys, field_name, plot_type
        else:
            lon = self._get_field('east_west', data2d)
            lat = self._get_field('north_south', data2d)
            xs = np.array(lon)
            ys = np.array(lat)
            # Some LIS coordinates are NaN. The following workaround fills out those elements
            # with reasonable values:
            idx = np.argwhere(np.isnan(xs))
            for i in idx:
                xs[i] = xs[i - 1] + self._global_attrs["DX"] / 1000.0 / 100.0
            idx = np.argwhere(np.isnan(ys))
            for i in idx:
                ys[i] = ys[i - 1] + self._global_attrs["DY"] / 1000.0 / 100.0

            return data2d, xs, ys, field_name, plot_type


    def _calculate_diff(self, name1, name2, ax_opts):
        """ Helper method for get_diff_data """
        d1 = self._get_data(name1, ax_opts, 0)
        d2 = self._get_data(name2, ax_opts, 1)
        d1 = apply_conversion(self.config_manager, d1, name1).squeeze()
        d2 = apply_conversion(self.config_manager, d2, name2).squeeze()
        return d1 - d2

    def _get_data(self, field_name, ax_opts, pid):
        d = self.config_manager.readers[0].get_field(field_name, self.config_manager.findex)
        return self._get_xy(d, field_name, level=0, time_lev=ax_opts['time_lev'])

    @staticmethod
    def __get_xy(d, name):
        """ Extract XY slice from N-dim data field"""
        if d is None:
            return
        dlength = len(d.shape)
        if dlength == 2:
            return d[:, :]
        if dlength == 3:
            return d[0, :, :]
        if dlength == 4:
            return d[0, 0, :, :]

    def _process_coordinates(self, data2d, dim1, dim2, field_name, plot_type, file_index, figure, ax):
        """
        Process coordinates for LIS plots, handling NaN values in coordinates.
        """
        xr = np.array(self.lon[0, :])
        yr = np.array(self.lat[:, 0])        
        latN = max(yr[:])
        latS = min(yr[:])
        lonW = min(xr[:])
        lonE = max(xr[:])
        self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
        self.config_manager.ax_opts['central_lon'] = np.mean([lonW, lonE])
        self.config_manager.ax_opts['central_lat'] = np.mean([latS, latN])
        return data2d, xr, yr, field_name, plot_type, file_index, figure, ax

    def _apply_vertical_level_selection(self, data2d, field_name, level):
        """
        Apply vertical level selection for LIS data.
        """
        # Get the vertical dimension name
        dim = self.get_dd(self.source_name, self.source_data, 'zc', field_name)
        
        # If vertical dimension exists and level is specified, select it
        if dim and level is not None:
            try:
                data2d = eval(f"data2d.isel({dim}=level)")
            except Exception as e:
                self.logger.warning(f"Error selecting vertical level: {e}")
        
        return data2d

    def _apply_time_selection(self, original_data, data2d, time_dim, time_lev, field_name, level):
        """
        LIS has more complex time handling with optional time averaging.
        """
        if time_dim and time_dim in original_data.dims:
            num_times = original_data.dims[time_dim]
            
            # Check if time averaging is enabled
            if hasattr(self, 'ax_opts') and self.ax_opts.get('tave', False) and num_times > 1:
                self.logger.debug(f"Averaging over {num_times} time levels.")
                return apply_mean(self.config_manager, data2d, level)
            else:
                try:
                    return original_data.isel({time_dim: time_lev}).squeeze()
                except Exception as e:
                    self.logger.warning(f"Error selecting time level: {e}")
                    # If time selection fails, use the data as is
        
        return data2d

    def get_field_dim_name(self, source_data, dim_name, field_name):
        d = source_data['vars'][field_name]
        field_dims = list(d.dims) 
        names = self.get_model_dim_name(self.source_name, dim_name).split(',')
        common = list(set(names).intersection(field_dims))
        dim = list(common)[0] if common else None
        return dim

    def _get_xy(self, data_array, level, time_lev):
        """ Extract XY slice from N-dim data field"""
        if data_array is None:
            return None

        data2d = data_array.squeeze()
        tc_dim = self.config_manager.get_model_dim_name('tc')
        zc_dim = self.config_manager.get_model_dim_name('zc')
        
        if zc_dim in data2d.dims:
            data2d = data2d.isel({zc_dim: level})
        if tc_dim in data2d.dims:
            num_times = data_array[tc_dim].size
            if self.ax_opts['tave'] and num_times > 1:
                self.logger.debug(f"Averaging over {num_times} time levels.")
                data2d = apply_mean(self.config_manager, data2d, level)
                return apply_conversion(self.config_manager, data2d, data_array.name)
            else:
                data2d = data2d.isel({tc_dim: time_lev})

        return apply_conversion(self.config_manager, data2d, data_array.name)

