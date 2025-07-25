import logging
import warnings
from typing import Any

import pandas as pd

from eviz.lib.config.config import Config
from dataclasses import dataclass, field

from eviz.models.source_base import GenericSource

warnings.filterwarnings("ignore")


@dataclass
class Fluxnet(GenericSource):
    """ Define Fluxnet inventory data and functions.

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
        super().__post_init__()
        self.source_name = 'fluxnet'

    def process_data(self, filename) -> pd.DataFrame:
        """ Prepare data for plotting """
        # Get the model data
        model_data = self.config.readers[self.source_name].read_data(filename)
        return model_data

    def process_single_plots(self, plotter):
        pass

    def _prepare_field_to_plot(self):
        pass
