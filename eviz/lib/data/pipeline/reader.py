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
    """Data reading stage of the pipeline."""

    def __init__(self, config_manager=None):
        """Initialize a new DataReader.

        Args:
            config_manager: Configuration manager instance
        """
        self.factory = DataSourceFactory(config_manager)
        self.logger = logging.getLogger(__name__)
        # For backward compatibility with tests
        self.data_sources = {}
        self.config_manager = config_manager

    def read_file(self, file_path: str, model_name: Optional[str] = None) -> DataSource:
        """Read data from a file."""
        self.logger.debug(f"Reading file: {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # For backward compatibility with tests
        if file_path in self.data_sources:
            self.logger.debug(f"Using cached data source for {file_path}")
            return self.data_sources[file_path]

        try:
            data_source = self.factory.create_data_source(file_path, model_name)
            data_source.load_data(file_path)
            
            # For backward compatibility with tests
            self.data_sources[file_path] = data_source

            return data_source

        except Exception as e:
            self.logger.error(f"Error reading file: {file_path}. Exception: {e}")
            raise

    def read_files(self, file_paths: List[str], model_name: Optional[str] = None) -> Dict[str, DataSource]:
        """Read data from multiple files."""
        self.logger.debug(f"Reading {len(file_paths)} files")

        result = {}
        for file_path in file_paths:
            try:
                data_source = self.read_file(file_path, model_name)
                result[file_path] = data_source
            except Exception as e:
                self.logger.error(f"Error reading file: {file_path}. Exception: {e}")

        return result

    # For backward compatibility with tests
    def get_data_source(self, file_path: str) -> Optional[DataSource]:
        """Get a data source.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            The data source, or None if not found
        """
        return self.data_sources.get(file_path)
    
    # For backward compatibility with tests
    def get_all_data_sources(self) -> Dict[str, DataSource]:
        """Get all data sources.
        
        Returns:
            A dictionary mapping file paths to data sources
        """
        return self.data_sources.copy()

    def close(self) -> None:
        """Close resources used by the reader."""
        self.logger.debug("Closing DataReader resources")
        # Close each data source
        for data_source in self.data_sources.values():
            if hasattr(data_source, 'close'):
                data_source.close()
        # Clear the data sources dictionary
        self.data_sources.clear()
