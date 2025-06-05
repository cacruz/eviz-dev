import warnings
import logging
from dataclasses import dataclass
from eviz.models.gridded_source import GriddedSource


warnings.filterwarnings("ignore")

@dataclass
class Crest(GriddedSource):
    """ The Crest class contains definitions for handling CREST data. This is data
        produced by the Coupled Reusable Earth System Tensor-framework.
    """
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.comparison_plot = False
        self.source_name = 'crest'

