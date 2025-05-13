"""
Adapter between YAML configuration and data source architecture.
"""

import os
import logging
from typing import Dict, List, Optional, Any

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.lib.data.pipeline import DataPipeline  # Import for backward compatibility with tests


class ConfigurationAdapter:
    """Adapter between YAML configuration and data source architecture.
    
    This class bridges the gap between the YAML configuration system and the
    data source architecture, allowing them to work together seamlessly.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize a new ConfigurationAdapter.
        
        Args:
            config_manager: The configuration manager
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        # Use the pipeline from the config_manager instead of creating a new one
        self.pipeline = self.config_manager.pipeline
        # Use the data_sources from the config_manager
        self.data_sources = self.config_manager.data_sources
        
    def process_configuration(self):
        """Process the configuration and set up the data pipeline."""
        # Process each input file
        for file_index, file_info in enumerate(self.config_manager.app_data.inputs):
            # Extract file information
            file_path = os.path.join(file_info.get('location', ''), file_info.get('name', ''))
            model_name = file_info.get('exp_name')
            
            # Set up processing options
            processing_options = file_info.get('processing', {})
            
            # Set up transformation options
            transformation_options = file_info.get('transformations', {})
            transform_params = {}
            for transform_type, transform_config in transformation_options.items():
                if transform_config.get('enabled', False):
                    transform_params[transform_type] = True
                    # Add specific parameters for this transformation
                    for param, value in transform_config.items():
                        if param != 'enabled':
                            transform_params[param] = value
            
            # Process the file through the pipeline
            try:
                self.logger.info(f"Processing file: {file_path}")
                data_source = self.pipeline.process_file(
                    file_path, 
                    model_name=model_name,
                    process=True,
                    transform=bool(transform_params),
                    transform_params=transform_params
                )
                
                # Store the data source for later use
                self.data_sources[file_path] = data_source
                
                # Also store in config_manager for backward compatibility
                if not hasattr(self.config_manager, 'data_sources'):
                    self.config_manager.data_sources = {}
                self.config_manager.data_sources[file_path] = data_source
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
        
        # Handle integration if specified
        if hasattr(self.config_manager.input_config, '_integrate') and self.config_manager.input_config._integrate:
            self.logger.info("Integrating data sources")
            self.pipeline.integrate_data_sources()
        
        # Handle composite fields if specified
        if hasattr(self.config_manager.input_config, '_composite') and self.config_manager.input_config._composite:
            composite_config = self.config_manager.input_config._composite
            if isinstance(composite_config, dict):
                for output_name, config in composite_config.items():
                    variables = config.get('variables', [])
                    operation = config.get('operation', 'add')
                    if variables and len(variables) >= 2:
                        self.logger.info(f"Creating composite field {output_name} using {operation} on {variables}")
                        self.pipeline.integrate_variables(variables, operation, output_name)
    
    def get_data_source(self, file_path: str):
        """Get a data source by file path.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            The data source, or None if not found
        """
        return self.data_sources.get(file_path)
    
    def get_all_data_sources(self) -> Dict[str, Any]:
        """Get all data sources.
        
        Returns:
            Dictionary mapping file paths to data sources
        """
        return self.data_sources
    
    def get_dataset(self):
        """Get the integrated dataset.
        
        Returns:
            The integrated dataset, or None if not available
        """
        return self.pipeline.get_dataset()
    
    def close(self):
        """Close all data sources and free resources."""
        self.pipeline.close()
