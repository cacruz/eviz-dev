import warnings
from dataclasses import dataclass

import numpy as np

from eviz.lib.data.utils import apply_mean
from eviz.lib.data.utils import apply_conversion
from eviz.models.esm.nuwrf import NuWrf

warnings.filterwarnings("ignore")


@dataclass
class Lis(NuWrf):
    """ Define LIS specific model data and functions."""

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.comparison_plot = False
        self.source_name = 'lis'

    def _get_field_for_simple_plot(self, field_name, plot_type):
        """LIS-specific simple plot field processing."""
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
            data2d = self._get_xy(d, field_name, level, time_lev)

        xs, ys = None, None
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, xs, ys, field_name, plot_type
        else:
            lon = self._get_field('east_west', data2d)
            lat = self._get_field('north_south', data2d)
            xs = np.array(lon)
            ys = np.array(lat)
            
            # Handle NaN coordinates
            self._fix_nan_coordinates(xs, ys)
            return data2d, xs, ys, field_name, plot_type

    def _get_field_to_plot(self, field_name, file_index, plot_type, figure, time_level, level=None):
        """LIS-specific field processing."""
        ax = figure.get_axes()
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        data2d = None
        d = self.source_data['vars'][field_name]
        
        if 'xt' in plot_type:
            data2d = self._get_xt(d, field_name, time_lev=time_level)
        elif 'xy' in plot_type:
            data2d = self._get_xy(d, level=level, time_lev=time_level)

        xs, ys = None, None
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        else:
            lon = self.source_data['vars'][self.get_model_coord_name(self.source_name, 'xc')]
            lat = self.source_data['vars'][self.get_model_coord_name(self.source_name, 'yc')]
            xs = np.array(lon[0, :])
            ys = np.array(lat[:, 0])
            
            # Handle NaN coordinates and set extents
            self._fix_nan_coordinates(xs, ys)
            self._set_lis_extents(xs, ys)
            
            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax

    def _fix_nan_coordinates(self, xs, ys):
        """Fix NaN values in LIS coordinates."""
        idx = np.argwhere(np.isnan(xs))
        for i in idx:
            xs[i] = xs[i - 1] + self._global_attrs["DX"] / 1000.0 / 100.0
        
        idx = np.argwhere(np.isnan(ys))
        for i in idx:
            ys[i] = ys[i - 1] + self._global_attrs["DY"] / 1000.0 / 100.0

    def _set_lis_extents(self, xs, ys):
        """Set LIS-specific map extents."""
        latN = max(ys[:])
        latS = min(ys[:])
        lonW = min(xs[:])
        lonE = max(xs[:])
        self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
        self.config_manager.ax_opts['central_lon'] = np.mean([lonW, lonE])
        self.config_manager.ax_opts['central_lat'] = np.mean([latS, latN])

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

