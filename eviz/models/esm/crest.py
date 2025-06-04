import logging
from dataclasses import dataclass
from eviz.models.source_base import GenericSource


@dataclass
class Crest(GenericSource):
    """ The Crest class contains definitions for handling CREST data. This is data
        produced by the Coupled Reusable Earth System Tensor-framework.
    """
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

