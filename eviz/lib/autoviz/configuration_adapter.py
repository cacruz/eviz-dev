"""
Adapter between YAML configuration and data source architecture.
"""
import logging
import os
from typing import Dict, Any, List, Optional

import xarray as xr

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.lib.data.pipeline import DataPipeline
from eviz.lib.data.sources import DataSource


class ConfigurationAdapter:
    """Adapter for configuration management."""
    
    def __init__(self, config_manager):
        """Initialize a new ConfigurationAdapter.
        
        Args:
            config_manager: The configuration manager
        """
        self.logger.info("Start init")
        self.config_manager = config_manager
        # For backward compatibility with tests
        self.data_sources = {}

    @property
    def logger(self):
        """Return the logger for this class."""
        return logging.getLogger(__name__)

    def process_configuration(self):
        """Process the configuration and load data."""
        self.logger.info("Processing configuration")
        
        # Load data for each input file
        for i, file_entry in enumerate(self.config_manager.app_data.inputs):
            file_path = self._get_file_path(file_entry)
            
            # Use source_name from the command line arguments, not exp_name or exp_id from the file entry
            # This ensures we're using the correct data source type
            try:
                source_name = self.config_manager.source_names[self.config_manager.ds_index]
            except (IndexError, AttributeError):
                # Fallback for tests: use source_name from file_entry or default to 'generic'
                source_name = file_entry.get('source_name', 'generic')
            
            # Store exp_name or exp_id as metadata if available
            exp_metadata = {}
            if 'exp_name' in file_entry:
                exp_metadata['exp_name'] = file_entry['exp_name']
            if 'exp_id' in file_entry:
                exp_metadata['exp_id'] = file_entry['exp_id']
            
            self.logger.info(f"Loading data for file {i+1}: {file_path} with source_name: {source_name}")
            
            # Use the pipeline to process the file
            try:
                # This will create a DataSource and store it in the pipeline's data_sources dict
                data_source = self.config_manager.pipeline.process_file(
                    file_path, 
                    model_name=source_name,
                    metadata=exp_metadata  # Pass experiment metadata
                )
                
                if data_source is None:
                    self.logger.warning(f"Failed to load data from {file_path}")
                else:
                    self.logger.info(f"Successfully loaded data from {file_path}")
                    # For backward compatibility with tests
                    self.data_sources[file_path] = data_source
                    
            except Exception as e:
                self.logger.error(f"Error loading data from {file_path}: {e}")
        
        # If integration is enabled, integrate the datasets
        if hasattr(self.config_manager.input_config, '_integrate') and self.config_manager.input_config._integrate:
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
                
        # If composite is enabled, integrate variables
        if hasattr(self.config_manager.input_config, '_composite') and self.config_manager.input_config._composite:
            self.logger.info("Creating composite field")
            try:
                # Use the pipeline's integrator to integrate variables
                composite_config = self.config_manager.input_config._composite
                variables = composite_config.get('variables', [])
                operation = composite_config.get('operation', 'add')
                output_name = composite_config.get('output_name', 'composite')
                
                self.config_manager.pipeline.integrate_variables(variables, operation, output_name)
                self.logger.info(f"Successfully created composite field {output_name}")
            except Exception as e:
                self.logger.error(f"Error creating composite field: {e}")

    def process_configuration2(self):
        """Process the configuration and load data."""
        self.logger.info("Processing configuration")
        
        # Load data for each input file
        for i, file_entry in enumerate(self.config_manager.app_data.inputs):
            file_path = self._get_file_path(file_entry)
            
            # Use source_name from the command line arguments, not exp_name or exp_id from the file entry
            # This ensures we're using the correct data source type
            source_name = self.config_manager.source_names[self.config_manager.ds_index]
            
            # Store exp_name or exp_id as metadata if available
            exp_metadata = {}
            if 'exp_name' in file_entry:
                exp_metadata['exp_name'] = file_entry['exp_name']
            if 'exp_id' in file_entry:
                exp_metadata['exp_id'] = file_entry['exp_id']
            
            self.logger.info(f"Loading data for file {i+1}: {file_path} with source_name: {source_name}")
            
            # Use the pipeline to process the file
            try:
                # This will create a DataSource and store it in the pipeline's data_sources dict
                data_source = self.config_manager.pipeline.process_file(
                    file_path, 
                    model_name=source_name,
                    metadata=exp_metadata  # Pass experiment metadata
                )
                
                if data_source is None:
                    self.logger.warning(f"Failed to load data from {file_path}")
                else:
                    self.logger.info(f"Successfully loaded data from {file_path}")
                    # For backward compatibility with tests
                    self.data_sources[file_path] = data_source
                    
            except Exception as e:
                self.logger.error(f"Error loading data from {file_path}: {e}")
    
        # If integration is enabled, integrate the datasets
        if hasattr(self.config_manager.input_config, '_integrate') and self.config_manager.input_config._integrate:
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
                
        # If composite is enabled, integrate variables
        if hasattr(self.config_manager.input_config, '_composite') and self.config_manager.input_config._composite:
            self.logger.info("Creating composite field")
            try:
                # Use the pipeline's integrator to integrate variables
                composite_config = self.config_manager.input_config._composite
                variables = composite_config.get('variables', [])
                operation = composite_config.get('operation', 'add')
                output_name = composite_config.get('output_name', 'composite')
                
                self.config_manager.pipeline.integrate_variables(variables, operation, output_name)
                self.logger.info(f"Successfully created composite field {output_name}")
            except Exception as e:
                self.logger.error(f"Error creating composite field: {e}")
    
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
            return os.path.join(location, name)
        else:
            return name
    
    def get_data_source(self, file_path: str) -> Optional[DataSource]:
        """Get a data source from the adapter.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            The data source for the file path, or None if not found
        """
        # For backward compatibility with tests
        return self.data_sources.get(file_path)
    
    def get_all_data_sources(self) -> Dict[str, DataSource]:
        """Get all data sources from the adapter.
        
        Returns:
            A dictionary mapping file paths to data sources
        """
        # For backward compatibility with tests
        return self.data_sources
    
    def get_dataset(self) -> Optional[xr.Dataset]:
        """Get the integrated dataset from the adapter.
        
        Returns:
            The integrated dataset, or None if not available
        """
        # Delegate to the pipeline
        return self.config_manager.pipeline.get_dataset()
    
    def close(self) -> None:
        """Close resources used by the adapter."""
        self.logger.debug("Closing ConfigurationAdapter resources")
        # Close the pipeline, which will close all data sources
        if hasattr(self.config_manager, 'pipeline'):
            self.config_manager.pipeline.close()
