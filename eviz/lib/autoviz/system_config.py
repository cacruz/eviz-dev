from dataclasses import dataclass, field
import logging
from time import strftime
from typing import Any
from eviz.lib.utils import log_method
from eviz.lib.autoviz.app_data import AppData


@dataclass
class SystemConfig:
    app_data: AppData = field(default_factory=AppData)
    use_mp_pool: bool = False
    archive_web_results: bool = False
    collection: str = None
    event_stamp: str = None

    @log_method
    def initialize(self):
        """Initialize system configuration."""
        system_opts = self.app_data.system_opts

        self.use_mp_pool = system_opts.get('use_mp_pool', False)

        self.archive_web_results = system_opts.get('archive_web_results', False)
        if self.archive_web_results:
            self.event_stamp = strftime("%Y%m%d-%H%M%S")

        self.collection = system_opts.get('collection', '')

        self.logger.debug(f"SystemConfig initialized with: "
                          f"use_mp_pool={self.use_mp_pool}, "
                          f"archive_web_results={self.archive_web_results}, "
                          f"collection={self.collection}, "
                          f"event_stamp={self.event_stamp}")
        
    @property
    def logger(self):
        """Return the logger for this class."""
        return logging.getLogger(__name__)

    def to_dict(self) -> dict:
        """Return a dictionary representation of the SystemConfig."""
        return {
            "use_mp_pool": self.use_mp_pool,
            "archive_web_results": self.archive_web_results,
            "collection": self.collection,
            "event_stamp": self.event_stamp,
        }   