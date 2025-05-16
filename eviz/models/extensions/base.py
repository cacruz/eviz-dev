"""
Base class for model-specific extensions.
"""

import logging

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.lib.data.sources import DataSource


class ModelExtension:
    """Base class for model-specific extensions.
    
    This class provides a framework for model-specific data processing
    that can be applied to data sources.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize a new ModelExtension.
        
        Args:
            config_manager: The configuration manager
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def process_data_source(self, data_source: DataSource) -> DataSource:
        """Process a data source with model-specific logic.
        
        Args:
            data_source: The data source to process
            
        Returns:
            The processed data source
        """
        # Base implementation does nothing
        return data_source
