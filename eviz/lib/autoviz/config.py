import logging
import matplotlib as mpl

from dataclasses import dataclass, field
from typing import List, Dict, Any
from eviz.lib.utils import log_method
from eviz.lib.autoviz.input_config import InputConfig
from eviz.lib.autoviz.output_config import OutputConfig
from eviz.lib.autoviz.system_config import SystemConfig
from eviz.lib.autoviz.history_config import HistoryConfig
from eviz.lib.autoviz.yaml_parser import YAMLParser
from eviz.lib.autoviz.app_data import AppData

rc_matplotlib = mpl.rcParams  # PEP8 4 lyfe


@dataclass
class Config:
    """ Main configuration class that delegates responsibilities to sub-configurations. """
    source_names: List[str]
    config_files: List[str]
    app_data: AppData = field(default_factory=AppData)
    spec_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.logger.info("Start init")
        self.yaml_parser = YAMLParser(config_files=self.config_files, source_names=self.source_names)
        self.yaml_parser.parse()
        # We need to convert parsed app_data to AppData object
        self.app_data = AppData(**self.yaml_parser.app_data)  
        self.spec_data = self.yaml_parser.spec_data  
        self.map_params = self.yaml_parser.map_params
        self._ds_index = self.yaml_parser._ds_index
        self._specs_yaml_exists = self.yaml_parser._specs_yaml_exists
        self.meta_coords = self.yaml_parser.meta_coords
        self.meta_attrs = self.yaml_parser.meta_attrs
        self.species_db = self.yaml_parser.species_db

        self.input_config = InputConfig(self.source_names, self.config_files)
        self.output_config = OutputConfig()
        self.system_config = SystemConfig()
        self.history_config = HistoryConfig()

        self._assign_app_data_to_subconfigs()

        self.initialize()

    def _assign_app_data_to_subconfigs(self):
        """Assign app_data to all sub-configurations."""
        self.input_config.app_data = self.app_data
        self.output_config.app_data = self.app_data
        self.system_config.app_data = self.app_data
        self.history_config.app_data = self.app_data

    @log_method
    def initialize(self):
        """Initialize all configurations."""
        self.input_config.initialize()
        self.output_config.initialize()
        self.system_config.initialize()
        self.history_config.initialize()

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @property
    def ds_index(self) -> int:
        return self._ds_index

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Config object."""
        return {
            "input_config": self.input_config.to_dict(),
            "output_config": self.output_config.to_dict(),
            "system_config": self.system_config.to_dict(),
            "history_config": self.history_config.to_dict(),
            }
    
