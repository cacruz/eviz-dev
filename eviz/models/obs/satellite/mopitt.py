import logging
import multiprocessing
import sys
import warnings
from typing import Any

import numpy as np
import xarray as xr

from dataclasses import dataclass, field

from eviz.lib.autoviz.figure import Figure
from eviz.lib.autoviz.utils import print_map
from eviz.models.source_base import GenericSource
from eviz.lib.data.utils import apply_conversion

warnings.filterwarnings("ignore")


@dataclass
class Mopitt(GenericSource):
    """ Define MOPITT satellite data and functions.
    """
    source_data: Any = None
    _ds_attrs: dict = field(default_factory=dict)
    maps_params: dict = field(default_factory=dict)
    frame_params: Any = None
    mopitt_dataset: Any = None
    ds_lon: Any = None
    ds_lat: Any = None

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

    def prepare_data(self, fid) -> dict:
        ds = xr.Dataset()
        map_params = self.config.map_params
        for i in map_params.keys():
            field_name = map_params[i]['field']
            self.logger.info(f"Plotting {field_name}")
            # for field_name in field_names:
            da = fid["/HDFEOS/GRIDS/MOP03/Data Fields/"+field_name][:]
            da_lat = fid["/HDFEOS/GRIDS/MOP03/Data Fields/Latitude"][:]
            da_lon = fid["/HDFEOS/GRIDS/MOP03/Data Fields/Longitude"][:]
            da_new = xr.DataArray(da, dims=["lon", "lat"], coords=[da_lon, da_lat])
            da_new = da_new.where(da_new != -9999.)
            ds[field_name] = da_new
            source_name = map_params[i]['source_name']

        data_group = self.config.readers[source_name].get_data_group(fid)
        fid_attrs = self.config.readers[source_name].get_fid_attrs(fid)
        ds.attrs = fid_attrs

        unzipped_data = {}
        unzipped_data['id'] = self.findex
        unzipped_data['ptr'] = ds
        unzipped_data['regrid'] = False
        unzipped_data['vars'] = ds.data_vars
        unzipped_data['attrs'] = ds.attrs
        unzipped_data['dims'] = ds.dims
        unzipped_data['coords'] = ds.coords
        unzipped_data['filename'] = "".join(self.config.file_list[self.config.findex]['filename'])
        self.config.readers[source_name].datasets.append(unzipped_data)
        if '*empty*' in repr(ds.data_vars):   # If the dataset is empty
            return {}
        else:
            return unzipped_data

    def process_simple_plots(self, plotter):
        map_params = self.config.map_params
        field_num = 0
        self.config.findex = 0
        for i in map_params.keys():
            field_name = map_params[i]['field']
            source_name = map_params[i]['source_name']
            filename = map_params[i]['filename']
            file_index = self.config.get_file_index(filename)
            mopitt_data = self.config.readers[source_name].read_data(filename)
            # Additional HDF5 processing
            source_data = self.prepare_data(mopitt_data.get_fid())
            self.config.findex = file_index
            self.config.pindex = field_num
            self.config.axindex = 0
            for pt in map_params[i]['to_plot']:
                self.logger.info(f"Plotting {field_name}, {pt} plot")
                field_to_plot = self._get_field_for_simple_plot(source_data, field_name, pt)
                plotter.simple_plot(self.config, field_to_plot)
            field_num += 1

    def process_single_plots(self, plotter):
        for s in range(len(self.config.source_names)):
            map_params = self.config.map_params
            field_num = 0
            for i in map_params.keys():
                source_name = map_params[i]['source_name']
                if source_name == self.config.source_names[s]:
                    field_name = map_params[i]['field']
                    source_name = map_params[i]['source_name']
                    filename = map_params[i]['filename']
                    file_index = field_num
                    mopitt_data = self.config.readers[source_name].read_data(filename)
                    # Additional HDF5 processing
                    source_data = self.prepare_data(mopitt_data)
                    self.config.findex = file_index
                    self.config.pindex = field_num
                    self.config.axindex = 0
                    for pt in map_params[i]['to_plot']:
                        self.logger.info(f"Plotting {field_name}, {pt} plot")
                        figure = Figure(self.config, pt)
                        if 'xy' in pt:
                            levels = self.config.get_levels(field_name, pt + 'plot')
                            if not levels:
                                self.logger.warning(f' -> No levels specified for {field_name}')
                                continue
                            for level in levels:
                                field_to_plot = self._prepare_field_to_plot(source_data,
                                                                            field_name, file_index, pt, figure, level=level)
                                if self.use_mp_pool:
                                    p = multiprocessing.Process(target=plotter.single_plots,
                                                                args=(self.config, field_to_plot, level))
                                    self.logger.info(f"  start " + p.name)
                                    self.procs.append(p)
                                    p.start()
                                else:
                                    plotter.single_plots(self.config, field_to_plot=field_to_plot, level=level)
                                    print_map(self.config, pt, self.config.findex, figure, level=level)

                        else:
                            field_to_plot = self._prepare_field_to_plot(source_data, field_name, file_index, pt, figure)
                            if self.use_mp_pool:
                                p = multiprocessing.Process(target=plotter.single_plots,
                                                            args=(self.config, field_to_plot))
                                self.logger.info(f"  start " + p.name)
                                self.procs.append(p)
                                p.start()

                            else:
                                plotter.single_plots(self.config, field_to_plot=field_to_plot)
                                print_map(self.config, pt, self.config.findex, figure)

                    field_num += 1

        if self.use_mp_pool:
            for p in self.procs:
                self.logger.info(f"process{p.name} is done")
                p.join()

    def _prepare_field_to_plot(self, source_data, field_name, file_index, plot_type, figure, level=None):
        _, ax = figure.get_fig_ax()
        self.config.ax_opts = figure.init_ax_opts(field_name)
        dim1, dim2 = self.config.get_dim_names(plot_type)
        data2d = None
        if 'xy' in plot_type:
            data2d = self._extract_xy_data(source_data, field_name, level=level, time_lev=self.config.ax_opts['time_lev'])
        else:
            pass

        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        return data2d, data2d[dim1].values, data2d[dim2].values, field_name, plot_type, file_index, figure, ax

    def _get_field_for_simple_plot(self, model_data, field_name, plot_type):
        name = self.config.source_names[self.config.ds_index]
        dim1, dim2 = self.config.get_dim_names(plot_type)
        # dim1 = self.config.meta_coords['xc'][name]
        # dim2 = self.config.meta_coords['yc'][name]
        if 'yz' in plot_type:
            dim1 = self.config.meta_coords['yc'][name]
            dim2 = self.config.meta_coords['zc'][name]
        d = model_data['vars']

        if 'xy' in plot_type:
            data2d = self._get_xy_simple(d, field_name, 0)
        else:
            self.logger.error(f'[{plot_type}] plot: Please create specifications file.')
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

    def _extract_xy_data(self, d, name, level, time_lev):
        """ Extract XY slice from N-dim data field"""
        if d is None:
            return
        data2d = d['vars'][name].squeeze()
        # Hackish
        if len(data2d.shape) == 4:
            data2d = data2d.isel(time=0)
        if len(data2d.shape) == 3:
            if self.config.get_model_dim_name('tc') in data2d.dims:
                data2d = data2d.isel(time=0)
            else:
                data2d = data2d.isel(lev=0)
        return apply_conversion(self.config, data2d, name)
