import logging
import sys
import warnings
from typing import Any
from dataclasses import dataclass, field

from eviz.models.root import Root
from eviz.lib.data.data_utils import apply_conversion

warnings.filterwarnings("ignore")


@dataclass
class Omi(Root):
    """ Define OMI satellite data and functions.
    """
    source_data: Any = None
    _ds_attrs: dict = field(default_factory=dict)
    maps_params: dict = field(default_factory=dict)
    frame_params: Any = None

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

    def _get_field_to_plot(self, model_data, field_name, file_index, plot_type, figure, level=None):
        _, ax = figure.get_fig_ax()
        self.config.ax_opts = figure.init_ax_opts(field_name)
        dim1, dim2 = self.config.get_dim_names(plot_type)
        d = model_data['vars']

        if 'xy' in plot_type:
            data2d = self._get_xy(d, field_name, 0)
        else:
            self.logger.error(f'[{plot_type}] plot: Please create specifications file.')
            sys.exit()

        return data2d, data2d[dim1].values, data2d[dim2].values, field_name, plot_type, file_index, figure, ax

    def _get_field_for_simple_plot(self, model_data, field_name, plot_type):
        name = self.config.source_names[self.config.ds_index]
        dim1, dim2 = self.config.get_dim_names(plot_type)

        d = model_data['vars']

        if 'xy' in plot_type:
            data2d = self._get_xy(d, field_name, 0)
        else:
            self.logger.error(f'[{plot_type}] plot: Please create specifications file.')
            sys.exit()

        coords = data2d.coords
        return data2d, coords[dim1], coords[dim2], field_name, plot_type

    def _get_xy(self, d, name, level):
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
        return apply_conversion(self.config, data2d, name)
