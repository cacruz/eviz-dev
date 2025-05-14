"""
Data reading stage of the pipeline.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any

import xarray as xr

from eviz.lib.data.factory import DataSourceFactory
from eviz.lib.data.sources import DataSource


class DataReader:
    """Data reading stage of the pipeline.
    
    This class handles reading data from files using the appropriate data source.
    """
    
    def __init__(self):
        """Initialize a new DataReader."""
        self.factory = DataSourceFactory()
        self.logger = logging.getLogger(__name__)
        self.data_sources = {}  # Maps file paths to data source instances
    
    def read_file(self, file_path: str, model_name: Optional[str] = None) -> DataSource:
        """Read data from a file.
        
        Args:
            file_path: Path to the data file
            model_name: Optional name of the model this data source belongs to
            
        Returns:
            A data source instance containing the loaded data
            
        Raises:
            ValueError: If the file extension is not supported
            FileNotFoundError: If the file does not exist
        """
        self.logger.debug(f"Reading file: {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path in self.data_sources:
            self.logger.debug(f"Using cached data source for file: {file_path}")
            return self.data_sources[file_path]
        
        try:
            data_source = self.factory.create_data_source(file_path, model_name)
            data_source.load_data(file_path)
            # Caching
            self.data_sources[file_path] = data_source
            
            return data_source
            
        except Exception as e:
            self.logger.error(f"Error reading file: {file_path}. Exception: {e}")
            raise
    
    def read_files(self, file_paths: List[str], model_name: Optional[str] = None) -> Dict[str, DataSource]:
        """Read data from multiple files.
        
        Args:
            file_paths: List of paths to data files
            model_name: Optional name of the model these data sources belong to
            
        Returns:
            A dictionary mapping file paths to data source instances
        """
        self.logger.debug(f"Reading {len(file_paths)} files")
        
        result = {}
        for file_path in file_paths:
            try:
                data_source = self.read_file(file_path, model_name)
                result[file_path] = data_source
            except Exception as e:
                self.logger.error(f"Error reading file: {file_path}. Exception: {e}")
                # Continue with the next file
        
        return result
    
    def get_data_source(self, file_path: str) -> Optional[DataSource]:
        """Get the data source for a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            The data source for the file, or None if the file has not been read
        """
        return self.data_sources.get(file_path)
    
    def get_all_data_sources(self) -> Dict[str, DataSource]:
        """Get all data sources.
        
        Returns:
            A dictionary mapping file paths to data source instances
        """
        return self.data_sources.copy()
    
    def close(self) -> None:
        """Close all data sources and free resources."""
        for data_source in self.data_sources.values():
            data_source.close()
        self.data_sources.clear()
