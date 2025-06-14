import logging
import sys
import warnings
from typing import Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import eviz.lib.autoviz.utils as pu
from eviz.lib.autoviz.figure import Figure
from eviz.models.obs_source import ObsSource

warnings.filterwarnings("ignore")


@dataclass
class Airnow(ObsSource):
    """ Define Airnow inventory data and functions.
    """
    source_data: Any = None
    _ds_attrs: dict = field(default_factory=dict)
    _maps_params: dict = field(default_factory=dict)
    frame_params: Any = None

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        return logging.getLogger(__name__)
    
    def add_data_source(self, *args, **kwargs):
        # Implement as needed, or just pass if not used
        pass

    def get_data_source(self, *args, **kwargs):
        # Implement as needed, or just return None if not used
        return None

    def load_data_sources(self, *args, **kwargs):
        # Implement as needed, or just pass if not used
        pass

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
            self.source_data = self.process_data(filename, field_name)
            self.config_manager.findex = file_index
            self.config_manager.pindex = field_num
            self.config_manager.axindex = 0
            for pt in map_params[i]['to_plot']:
                self.logger.info(f"Plotting {field_name}, {pt} plot")
                field_to_plot = self._get_field_for_simple_plot(field_name, pt)
                plotter.simple_plot(self.config_manager, field_to_plot)
            field_num += 1

    def _process_scatter_plot(self, data_array, field_name, file_index, plot_type, figure):
    # def _process_plot(self, data_array: pd.DataFrame, field_name: str, file_index: int,
                    #   plot_type: str):
        """Process a single plot type for a given field."""
        self.logger.info(f"Plotting {field_name}, {plot_type} plot")
        figure = Figure.create_eviz_figure(self.config_manager, plot_type)
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        self._process_obs_plot(data_array, field_name, file_index, plot_type, figure)

    def _process_obs_plot(self, data_array: pd.DataFrame, field_name: str,
                            file_index: int, plot_type: str, figure):
        """Process non-xy and non-polar plot types."""
        self.config_manager.level = None
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)

        field_to_plot = self._get_field_to_plot(data_array, field_name, file_index,
                                                plot_type, figure,
                                                time_level=time_level_config)
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex, plot_result)

    def _get_field_to_plot(self, data_array: pd.DataFrame, field_name: str, file_index: int,
                           plot_type: str, figure, time_level=None,) -> tuple:
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        data2d = None

        dim1, dim2 = self.config_manager.get_dim_names(plot_type)
        lon = data_array.lon
        lat = data_array.lat
   
        self.config_manager.ax_opts['extent'] = [-120, -70, 24, 50.5]
        self.config_manager.ax_opts['central_lon'] = np.mean(self.config_manager.ax_opts['extent'][:2])
        self.config_manager.ax_opts['central_lat'] = np.mean(self.config_manager.ax_opts['extent'][2:])

        if 'xy' in plot_type:
            data2d = self._get_xy_simple(data_array, field_name, 0)
        elif 'sc' in plot_type:
            data2d = data_array.data
            return data2d, lon, lat, field_name, plot_type, file_index, figure
        else:
            pass

        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure
        return data2d, data2d[dim1].values, data2d[dim2].values, field_name, plot_type, file_index, figure
    
    def _get_field_for_simple_plot(self, field_name, plot_type):
        dim1, dim2 = self.config_manager.get_dim_names(plot_type)
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
            if self.config_manager.get_model_dim_name('tc') in data2d.dims:
                data2d = data2d.isel(time=0)
            else:
                data2d = data2d.isel(lev=0)
        return data2d
