import logging
import os
from typing import Dict, Any, Optional
import xarray as xr

from eviz.lib.data.pipeline import DataPipeline
from eviz.lib.data.sources import DataSource


class ConfigurationAdapter:
    """
    Adapter between YAML configuration and data source architecture.
    
    This class serves as a mediator between the configuration system and the data processing
    pipeline. It interprets configuration settings, loads and processes data according to
    those settings, and provides access to the resulting data sources and datasets.
    
    The adapter handles:
    - Loading data from file paths specified in the configuration
    - Associating metadata with data sources (e.g., experiment names/IDs)
    - Managing the data pipeline for processing and transforming data
    - Integrating multiple datasets when specified in the configuration
    - Creating composite fields by combining variables with operations
    - Providing access to individual data sources and the integrated dataset
    
    Attributes:
        config_manager: The configuration manager containing app_data, input_config, etc.
        data_sources: Dictionary mapping file paths to their corresponding DataSource objects
        
    Note:
        The adapter initializes a DataPipeline instance and attaches it to the config_manager
        as _pipeline. This pipeline is used for all data processing operations.
    """
    def __init__(self, config_manager):
        """Initialize a new ConfigurationAdapter.
        
        Args:
            config_manager: The configuration manager
        """
        self.logger.info("Start init")
        self.config_manager = config_manager
        self.data_sources = {}
        self.config_manager._pipeline = DataPipeline(self.config_manager)

    @property
    def logger(self):
        """Return the logger for this class."""
        return logging.getLogger(__name__)

    def process_configuration(self):
        """Process the configuration and load data."""        
        for i, file_entry in enumerate(self.config_manager.app_data.inputs):
            file_path = self._get_file_path(file_entry)
            
            try:
                source_name = self.config_manager.source_names[self.config_manager.ds_index]
            except (IndexError, AttributeError):
                source_name = file_entry.get('source_name', 'gridded')
            
            # Store exp_name or exp_id as metadata if available
            exp_metadata = {}
            if 'exp_name' in file_entry:
                exp_metadata['exp_name'] = file_entry['exp_name']
            if 'exp_id' in file_entry:
                exp_metadata['exp_id'] = file_entry['exp_id']
            
            self.logger.debug(f"Loading data for file {i+1}: {file_path} with source_name: {source_name}")
            
            try:
                data_source = self.config_manager._pipeline.process_file(
                    file_path, 
                    model_name=source_name,
                    metadata=exp_metadata 
                )
                
                if data_source is None:
                    self.logger.warning(f"Failed to load data from {file_path}")
                else:
                    self.logger.debug(f"Successfully loaded data from {file_path}")
                    self.data_sources[file_path] = data_source
                    
            except Exception as e:
                self.logger.error(f"Error loading data from {file_path}: {e}")
        
        # If integration is enabled, integrate the datasets
        if hasattr(self.config_manager.input_config, '_integrate') and self.config_manager.input_config._integrate:
            self.logger.debug("Integrating datasets")
            try:
                integrated_dataset = self.config_manager._pipeline.integrate_data_sources()
                if integrated_dataset is not None:
                    self.logger.debug("Successfully integrated datasets")
                else:
                    self.logger.warning("Failed to integrate datasets")
            except Exception as e:
                self.logger.error(f"Error integrating datasets: {e}")
                
        # If composite is enabled, integrate variables
        if hasattr(self.config_manager.input_config, '_composite') and self.config_manager.input_config._composite:
            self.logger.debug("Creating composite field")
            try:
                composite_config = self.config_manager.input_config._composite
                variables = composite_config.get('variables', [])
                operation = composite_config.get('operation', 'add')
                output_name = composite_config.get('output_name', 'composite')
                
                self.config_manager._pipeline.integrate_variables(variables, operation, output_name)
                self.logger.debug(f"Successfully created composite field {output_name}")
            except Exception as e:
                self.logger.error(f"Error creating composite field: {e}")

    @staticmethod
    def _get_file_path(file_entry: Dict[str, Any]) -> str:
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
        return self.data_sources.get(file_path)
    
    def get_all_data_sources(self) -> Dict[str, DataSource]:
        """Get all data sources from the adapter.
        
        Returns:
            A dictionary mapping file paths to data sources
        """
        return self.data_sources
    
    def get_dataset(self) -> Optional[xr.Dataset]:
        """Get the integrated dataset from the adapter.
        
        Returns:
            The integrated dataset, or None if not available
        """
        if hasattr(self.config_manager, '_pipeline') and self.config_manager._pipeline:
            return self.config_manager._pipeline.get_dataset()
        return None
    
    def close(self) -> None:
        """Close resources used by the adapter."""
        self.logger.debug("Closing ConfigurationAdapter resources")
        if hasattr(self.config_manager, '_pipeline') and self.config_manager._pipeline:
            self.config_manager._pipeline.close()
