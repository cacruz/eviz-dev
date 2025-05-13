from dataclasses import dataclass, field
from eviz.lib.autoviz.config import Config

import logging
import warnings

warnings.filterwarnings("ignore")


@dataclass
class Plottable:
    config: Config
    plottable: dict = field(default_factory=dict)

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        self.plottable['name'] = None
        self.plottable['data'] = None
        self.plottable['fig'] = None
        self.plottable['ax'] = None
        self.plottable['ax_opts'] = None
        self.plottable['filename'] = None
        self.plottable['findex'] = None
        self.plottable['compare'] = False
        self.plottable['to_plot'] = None
