import logging
import matplotlib as mpl

from dataclasses import dataclass, field
from typing import List, Dict, Any
from eviz.lib.config.input_config import InputConfig
from eviz.lib.config.output_config import OutputConfig
from eviz.lib.config.system_config import SystemConfig
from eviz.lib.config.history_config import HistoryConfig
from eviz.lib.config.yaml_parser import YAMLParser
from eviz.lib.config.app_data import AppData
from eviz.lib.utils import log_method

rc_matplotlib = mpl.rcParams  # PEP8 4 lyfe

@dataclass
class Config:
    """ 
    Main configuration class that delegates responsibilities to sub-configurations.
    
    This class serves as the central hub for all configuration data in the eViz application.
    It loads and parses configuration files, initializes specialized sub-configuration objects,
    and provides access to configuration data through a unified interface.
    
    The class follows a delegation pattern, where specific configuration domains are managed
    by dedicated sub-configuration classes. This approach provides separation of concerns
    while maintaining a cohesive configuration API.
    
    Attributes:
        source_names: List of source identifiers used in configuration
        config_files: List of YAML configuration file paths to load
        app_data: Application-wide settings and parameters
        spec_data: Specification data for variables and visualization
        
    Sub-configurations:
        input_config: Manages data source specifications and input parameters
        output_config: Controls visualization output settings and file generation
        system_config: Handles system-level settings and environment configuration
        history_config: Tracks configuration history and provides versioning
        
    Additional attributes (populated during initialization):
        yaml_parser: Parser for YAML configuration files
        map_params: Mapping parameters for visualization
        meta_coords: Metadata for coordinate systems
        meta_attrs: Metadata for attributes
        species_db: Database of chemical species information
        
    Note:
        The Config class automatically initializes all sub-configurations during __post_init__,
        ensuring that the entire configuration system is ready to use after instantiation.
    """
    source_names: List[str]
    config_files: List[str]
    app_data: AppData = field(default_factory=AppData)
    spec_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.yaml_parser = YAMLParser(config_files=self.config_files, source_names=self.source_names)
        self.yaml_parser.parse()
        # Convert parsed app_data to AppData object
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
    
