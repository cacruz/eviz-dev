import sys
import logging
import numpy as np

from dataclasses import dataclass
import logging
from eviz.models.root import Root


@dataclass
class Crest(Root):
    """ The Crest class contains definitions for handling CREST data. This is data
        produced by the Coupled Reusable Earth System Tensor-framework.
    """
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

    def plot(self):
        self.logger.info("Plotting data for Crest model")

    def _simple_plots(self, plotter):
        map_params = self.config.map_params
        field_num = 0
        self.config.findex = 0
        for i in map_params.keys():
            field_name = map_params[i]['field']
            source_name = map_params[i]['source_name']
            filename = map_params[i]['filename']
            file_index = self.config.get_file_index(filename)
            source_data = self.config.readers[source_name].read_data(filename)
            if field_name not in source_data['vars']:
                continue
            self.config.findex = file_index
            self.config.pindex = field_num
            self.config.axindex = 0
            for pt in map_params[i]['to_plot']:
                self.logger.info(f"Plotting {field_name}, {pt} plot")
                field_to_plot = self._get_field_for_simple_plot(source_data, field_name, pt)
                plotter.simple_plot(self.config, field_to_plot)
            field_num += 1

    def _single_plots(self, plotter):
        pass

    def _get_field_for_simple_plot(self, model_data, field_name, plot_type):
        d = model_data['vars']
        if 'xy' in plot_type:
            dim1 = self.config.get_model_dim_name('xc')
            dim2 = self.config.get_model_dim_name('yc')
            data2d = self._get_xy_simple(d, field_name, 0)
        elif 'yz' in plot_type:
            dim1 = self.config.get_model_dim_name('yc')
            dim2 = self.config.get_model_dim_name('zc')
            data2d = self._get_yz_simple(d, field_name)
        elif 'xt' in plot_type:
            data2d = self._get_xt(d, field_name, time_lev=time_level)
        else:
            self.logger.error(f'Plot type [{plot_type}] error: Either specify in SPECS file or create plot type.')
            sys.exit()
        coords = data2d.coords
        return data2d, coords[dim1], coords[dim2], field_name, plot_type

    def _get_xy_simple(self, d, name, level):
        """ Extract XY slice from N-dim data field"""
        if d is None:
            return
        data2d = d[name].squeeze()
        # Hackish
        if len(data2d.shape) == 4:
            data2d = data2d.isel(time=0)
        if len(data2d.shape) == 3:
            if self.config.get_model_dim_name('tc') in data2d.dims:
                data2d = data2d.isel(time=0)
            else:
                data2d = data2d.isel(lev=0)
        return data2d

    def _get_yz_simple(self, d, name):
        """ Create YZ slice from N-dim data field"""
        if d is None:
            return
        data2d = d[name].squeeze()
        if len(data2d.shape) == 4:
            data2d = data2d.isel(time=0)
        data2d = data2d.mean(dim=self.config.get_model_dim_name('xc'))
        return data2d

    def _get_xt(self, d, name, time_lev):
        """ Extract time-series from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 1D (time) series
        """
        d_temp = d['vars'][name]
        if d_temp is None:
            return
        # num_times = eval(f"np.size(d_temp.{self.config.get_model_dim_name('tc')})")
        num_times = np.size(d_temp.time)
        self.logger.info(f"'{name}' field has {num_times} time levels")

        if isinstance(time_lev, list):
            self.logger.info(f"Computing time series on {time_lev} time range")
            data2d = d_temp.isel(time=slice(time_lev))
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
            # data2d = data2d.groupby(self.config.get_model_dim_name('tc')).mean(dim=xr.ALL_DIMS, keep_attrs=True)
            data2d = data2d.groupby('time').mean(dim=xr.ALL_DIMS, keep_attrs=True)

    def _get_model_dim_name(self, source_name: str, dim_name: str):
        try:
            dim = self.config.meta_coords[dim_name][source_name]
            return dim
        except KeyError:
            return None

