"""
Data reading stage of the pipeline.
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

import numpy as np
from eviz.lib.data.factory import DataSourceFactory
from eviz.lib.data.sources import DataSource


@dataclass
class DataReader:
    """Data reading stage of the pipeline."""
    config_manager: Optional[object] = None  # Replace 'object' with actual type if known
    data_sources: Dict = field(default_factory=dict, init=False)
    factory: object = field(init=False)  # Replace 'object' with actual type if known

    def __post_init__(self):
        """Post-initialization to set up factory and logger."""
        self.factory = DataSourceFactory(self.config_manager)

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def read_file(self, file_path: str, model_name: Optional[str] = None) -> DataSource:
        """Read data from a file."""
        self.logger.debug(f"Reading file: {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path in self.data_sources:
            self.logger.debug(f"Using cached data source for {file_path}")
            return self.data_sources[file_path]

        try:
            data_source = self.factory.create_data_source(file_path, model_name)
            data_source.load_data(file_path)
            
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

    def get_data_source(self, file_path: str) -> Optional[DataSource]:
        """Get a data source.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            The data source, or None if not found
        """
        return self.data_sources.get(file_path)
    
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


def get_data_coords(data_array, attribute_name):
    """
    Get coordinates for a data array attribute.

    Args:
        data_array: The xarray DataArray
        attribute_name: The name of the attribute to get coordinates for

    Returns:
        The coordinates for the attribute, or a fallback if the attribute is not found
    """
    if attribute_name is None:
        # If attribute_name is None, try to find an appropriate dimension
        if hasattr(data_array, 'dims'):
            dim_candidates = ['lon', 'longitude', 'x', 'lon_rho', 'x_rho']
            for dim in dim_candidates:
                if dim in data_array.dims:
                    return data_array[dim].values

            # If no candidate dimension is found, just return the first dimension
            if data_array.dims:
                return data_array[data_array.dims[0]].values

        # If all else fails, create a dummy coordinate
        return np.arange(data_array.shape[0])

    # Original implementation for when attribute_name is provided
    attribute_mapping = {
        'time': ['time', 't', 'TIME'],
        'lon': ['lon', 'longitude', 'x', 'lon_rho', 'x_rho'],
        'lat': ['lat', 'latitude', 'y', 'lat_rho', 'y_rho'],
        'lev': ['lev', 'level', 'z', 'altitude', 'height', 'depth', 'plev'],
    }


    # Check if attribute_name is a generic name present in the mapping
    for generic, specific_list in attribute_mapping.items():
        if attribute_name in specific_list:
            attribute_name = generic
            break

    # Check if we have a mapping for this generic name
    if attribute_name in attribute_mapping:
        # Try each specific name in the mapping
        for specific_name in attribute_mapping[attribute_name]:
            if specific_name in data_array.dims:
                return data_array[specific_name].values
            elif specific_name in data_array.coords:
                return data_array.coords[specific_name].values

    # If no mapping worked, try the attribute name directly
    if attribute_name in data_array.dims:
        return data_array[attribute_name].values
    elif attribute_name in data_array.coords:
        return data_array.coords[attribute_name].values

    # If the attribute wasn't found after all attempts, raise an error
    raise ValueError(f"Generic name for {attribute_name} not found in attribute_mapping.")

