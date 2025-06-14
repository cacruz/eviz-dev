import warnings
from dataclasses import dataclass
import numpy as np
import xarray as xr
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

    def _init_lis_domain(self, data_source):
        """LIS-specific initialization."""
        self.source_data = data_source

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
        
        level = 0
        time_lev = 0
        
        if 'xt' in plot_type:
            data2d = self._get_xt(d, time_lev=self.ax_opts['time_lev'], level=None)
        elif 'tx' in plot_type:
            data2d = self._get_tx(d, field_name, level=None, time_lev=self.ax_opts['time_lev'])
        elif 'xy' in plot_type:
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

    def _process_coordinates(self, data2d, dim1, dim2, field_name, plot_type, file_index, figure):
        """
        Process coordinates for LIS plots, handling NaN values in coordinates.
        """
        dim1, dim2 = self.coord_names(self.source_name, data2d, plot_type)
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure
        else:
            lon = self.source_data[self.get_model_coord_name(self.source_name, 'xc')]
            lat = self.source_data[self.get_model_coord_name(self.source_name, 'yc')]
            xs = np.array(lon[0, :])
            ys = np.array(lat[:, 0])
            # Handle NaN coordinates and set extents
            self._fix_nan_coordinates(xs, ys)
            self._set_lis_extents(xs, ys)
           
            return data2d, xs, ys, field_name, plot_type, file_index, figure        

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
        names = self.get_model_dim_name(dim_name).split(',')
        common = list(set(names).intersection(field_dims))
        dim = list(common)[0] if common else None
        return dim

    def _get_xt(self, d, time_lev, level=None):
        """ Extract time-series from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 1D (time) series
        """
        d_temp = d
        if d_temp is None:
            return

        xtime = d.time.values
        num_times = xtime.size
        self.logger.info(f"'{d.name}' field has {num_times} time levels")

        if isinstance(time_lev, list):
            self.logger.info(f"Computing time series on {time_lev} time range")
            data2d = d_temp.isel(time=slice(time_lev))
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

