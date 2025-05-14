# File: eviz/lib/data/pipeline/reader.py
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

    def __init__(self, config_manager=None):
        """Initialize a new DataReader.

        Args:
            config_manager: Configuration manager instance
        """
        # The factory will now be responsible for creating the data source instance
        # The reader will load the data into it.
        self.factory = DataSourceFactory(config_manager)
        self.logger = logging.getLogger(__name__)
        # Removed self.data_sources = {} - DataPipeline will store them
        self.config_manager = config_manager

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

        # Removed caching check - DataPipeline handles caching

        try:
            # Create the data source instance using the factory
            data_source = self.factory.create_data_source(file_path, model_name)
            # Load data into the instance
            data_source.load_data(file_path)

            # Removed caching line - DataPipeline handles caching

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
                # Call read_file for each path
                data_source = self.read_file(file_path, model_name)
                result[file_path] = data_source
            except Exception as e:
                self.logger.error(f"Error reading file: {file_path}. Exception: {e}")
                # Continue with the next file

        return result

    # Removed get_data_source and get_all_data_sources methods

    def close(self) -> None:
        """Close resources used by the reader."""
        # The reader itself doesn't hold data sources anymore,
        # so this method might become less critical here,
        # but we keep it for potential future resource management.
        self.logger.debug("Closing DataReader resources")
        pass # No specific resources to close in the reader itself now
