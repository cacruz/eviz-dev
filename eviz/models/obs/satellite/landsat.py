import logging
import warnings
from typing import Any

from matplotlib import pyplot as plt
import xarray as xr

from eviz.lib.config.config import Config
from dataclasses import dataclass, field

from eviz.models.source_base import GenericSource

warnings.filterwarnings("ignore")


@dataclass
class Landsat(GenericSource):
    """ Define LandSAT data and functions.

    Parameters:
        config (Config) : Config object associated with this data source
    """
    config: Config = None
    source_data: Any = None
    _ds_attrs: dict = field(default_factory=dict)
    maps_params: dict = field(default_factory=dict)
    frame_params: Any = None

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        self.model_data = None
        super().__post_init__()

    def prepare_data(self) -> None:
        ds = xr.Dataset()
        fid = self.config.readers[0].read_data(self.config._file_list[0]['filename'])
        self.model_data = self.config.readers[0].process_file(fid, ds)
        self.config.readers[0].datasets.append(self.model_data)

        self.field_name = 'toa_band6'
        self.toa_band6 = self.model_data[self.field_name][0, :, :]

    def plot(self):
        self.prepare_data()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # plot
        units = self.toa_band6.attrs['units']
        lon = self.toa_band6.lon.values
        lat = self.toa_band6.lat.values
        c = ax.contourf(lon, lat, self.toa_band6)
        ax.set_title(self.field_name, fontsize=12)
        _ = fig.colorbar(c, ax=ax, orientation='vertical', pad=0.05, fraction=0.05, label=units)
        plt.show()

