import logging
import sys
import warnings
from typing import Any
import pandas as pd
from dataclasses import dataclass, field

from eviz.lib.autoviz.plotter import SimplePlotter, ComparisonPlotter, SinglePlotter

from eviz.lib.autoviz.figure import Figure
from eviz.lib.autoviz.plot_utils import print_map
from eviz.models.root import Root
from eviz.lib import const as constants
import eviz.lib.utils as u

warnings.filterwarnings("ignore")


@dataclass
class Airnow(Root):
    """ Define Airnow inventory data and functions.
    """
    source_data: Any = None
    _ds_attrs: dict = field(default_factory=dict)
    _maps_params: dict = field(default_factory=dict)
    frame_params: Any = None

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

    def process_data(self, filename, field_name):
        """ Prepare data for plotting """
        # Get the model data
        model_data = self.config.readers[self.source_name].read_data(filename)
        # create time column from ValidDate and ValidTime
        model_data['time'] = pd.to_datetime(
            (model_data.ValidDate + ' ' + model_data.ValidTime),
            format='%m/%d/%y %H:%M')

        # Extract selected columns
        selected_columns = model_data[['time', 'lat', 'lon', field_name]]
        selected_columns = selected_columns.set_index('time')
        ds = selected_columns.to_xarray()
        ds = ds.dropna(dim='time')
        return ds

    def _simple_plots(self, plotter):
        map_params = self.config.map_params
        field_num = 0
        self.config.findex = 0
        for i in map_params.keys():
            field_name = map_params[i]['field']
            source_name = map_params[i]['source_name']
            self.source_name = source_name
            filename = map_params[i]['filename']
            file_index = self.config.get_file_index(filename)
            self.source_data = self.process_data(filename, field_name)
            self.config.findex = file_index
            self.config.pindex = field_num
            self.config.axindex = 0
            for pt in map_params[i]['to_plot']:
                self.logger.info(f"Plotting {field_name}, {pt} plot")
                field_to_plot = self._get_field_for_simple_plot(field_name, pt)
                plotter.simple_plot(self.config, field_to_plot)
            field_num += 1

    def _single_plots(self, plotter):
        for s in range(len(self.config.source_names)):
            map_params = self.config.map_params
            field_num = 0
            for i in map_params.keys():
                source_name = map_params[i]['source_name']
                if source_name == self.config.source_names[s]:
                    field_name = map_params[i]['field']
                    self.source_name = source_name
                    filename = map_params[i]['filename']
                    file_index = field_num  # self.config.get_file_index(filename)
                    self.source_data = self.process_data(filename, field_name)
                    # TODO: Is ds_index really necessary?
                    self.config.ds_index = s
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
                                field_to_plot = self._get_field_to_plot(field_name, file_index, pt, figure,
                                                                        level=level)
                                plotter.single_plots(self.config, field_to_plot=field_to_plot, level=level)
                                print_map(self.config, pt, self.config.findex, figure, level=level)

                        else:
                            field_to_plot = self._get_field_to_plot(field_name, file_index, pt, figure)
                            plotter.single_plots(self.config, field_to_plot=field_to_plot)
                            print_map(self.config, pt, self.config.findex, figure)

                    field_num += 1

    def _get_field_to_plot(self, field_name, file_index, plot_type, figure, level=None) -> tuple:
        self.config.ax_opts = figure.init_ax_opts(field_name)
        _, ax = figure.get_fig_ax()
        dim1, dim2 = self.config.get_dim_names(plot_type)
        d = self.source_data[field_name]

        data2d = None
        if 'xy' in plot_type:
            data2d = self._get_xy_simple(d, field_name, 0)
        elif 'sc' in plot_type:
            lon = self.source_data['lon'].data
            lat = self.source_data['lat'].data
            data2d = d.data
            return data2d, lon, lat, field_name, plot_type, file_index, figure, ax
        else:
            pass

        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        return data2d, data2d[dim1].values, data2d[dim2].values, field_name, plot_type, file_index, figure, ax

    def _get_field_for_simple_plot(self, field_name, plot_type):
        name = self.config.source_names[self.config.ds_index]
        dim1, dim2 = self.config.get_dim_names(plot_type)
        d = self.source_data[field_name]

        if 'xy' in plot_type:
            data2d = self._get_xy_simple(d, field_name, 0)
        elif 'sc' in plot_type:
            lon = self.source_data['lon'].data
            lat = self.source_data['lat'].data
            return d.data, lon, lat, field_name, plot_type
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
