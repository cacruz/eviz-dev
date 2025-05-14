"""
Adapter between YAML configuration and data source architecture.
"""

import os
import logging
from typing import Dict, List, Optional, Any

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.lib.data.pipeline import DataPipeline  # Import for backward compatibility with tests

class ConfigurationAdapter:
    """Adapter for configuration management."""
    
    def __init__(self, config_manager):
        """Initialize a new ConfigurationAdapter.
        
        Args:
            config_manager: The configuration manager
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        # Instead of directly accessing data_sources, we'll use the pipeline
        # self.data_sources = self.config_manager.data_sources  # This line causes the error
        # We don't need to store data_sources locally, as they're managed by the pipeline
        
    def process_configuration(self):
        """Process the configuration and load data."""
        self.logger.info("Processing configuration")
        
        # Load data for each input file
        for i, file_entry in enumerate(self.config_manager.app_data.inputs):
            file_path = self._get_file_path(file_entry)
            source_name = file_entry.get('source_name', self.config_manager.source_names[0])
            
            self.logger.info(f"Loading data for file {i+1}: {file_path}")
            
            # Use the pipeline to process the file
            try:
                # This will create a DataSource and store it in the pipeline's data_sources dict
                data_source = self.config_manager.pipeline.process_file(
                    file_path, 
                    model_name=source_name
                )
                
                if data_source is None:
                    self.logger.warning(f"Failed to load data from {file_path}")
                else:
                    self.logger.info(f"Successfully loaded data from {file_path}")
                    
            except Exception as e:
                self.logger.error(f"Error loading data from {file_path}: {e}")
        
        # If integration is enabled, integrate the datasets
        if hasattr(self.config_manager.input_config, '_enable_integration') and self.config_manager.input_config._enable_integration:
            self.logger.info("Integrating datasets")
            try:
                # Use the pipeline's integrator to integrate the datasets
                integrated_dataset = self.config_manager.pipeline.integrate_data_sources()
                if integrated_dataset is not None:
                    self.logger.info("Successfully integrated datasets")
                else:
                    self.logger.warning("Failed to integrate datasets")
            except Exception as e:
                self.logger.error(f"Error integrating datasets: {e}")
    
    def _get_file_path(self, file_entry: Dict[str, Any]) -> str:
        """Get the full file path from a file entry.
        
        Args:
            file_entry: The file entry from app_data.inputs
            
        Returns:
            The full file path
        """
        location = file_entry.get('location', '')
        name = file_entry.get('name', '')
        
        if location:
            return f"{location}/{name}"
        else:
            return name
    
    def close(self):
        """Close resources used by the adapter."""
        self.logger.debug("Closing ConfigurationAdapter resources")
        # Close the pipeline, which will close all data sources
        if hasattr(self.config_manager, 'pipeline'):
            self.config_manager.pipeline.close()
