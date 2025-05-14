"""
Factory for creating data source instances.
"""

# File: eviz/lib/data/factory/source_factory.py
import os
import logging
from typing import Dict, Type, List, Optional

from eviz.lib.data.sources import (
    DataSource,
    NetCDFDataSource,
    HDF5DataSource,
    CSVDataSource,
    GRIBDataSource
)
from .registry import DataSourceRegistry


class DataSourceFactory:
    """Factory for creating data source instances."""
    
    def __init__(self, config_manager=None):
        """Initialize a new DataSourceFactory.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.registry = DataSourceRegistry()
        self._register_default_data_sources()
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    
    def _register_default_data_sources(self) -> None:
        """Register the default data source implementations."""
        self.registry.register(['nc', 'nc4', 'netcdf', 'netcdf4'], NetCDFDataSource)
        self.registry.register(['h5', 'hdf5', 'hdf'], HDF5DataSource)
        self.registry.register(['csv', 'dat', 'txt'], CSVDataSource)
        self.registry.register(['grib', 'grib2'], GRIBDataSource)
    
    def register_data_source(self, extensions: List[str], data_source_class: Type[DataSource]) -> None:
        """Register a custom data source class.
        
        Args:
            extensions: List of file extensions (without the dot)
            data_source_class: The data source class to register
        """
        self.registry.register(extensions, data_source_class)
    
    def create_data_source(self, file_path: str, model_name: Optional[str] = None) -> DataSource:
        """Create a data source instance for the specified file.
        
        Args:
            file_path: Path to the data file
            model_name: Optional name of the model this data source belongs to
            
        Returns:
            A data source instance for the specified file
            
        Raises:
            ValueError: If the file extension is not supported
        """
        _, ext = os.path.splitext(file_path)
        if not ext:
            if 'nc' in file_path.lower() or 'netcdf' in file_path.lower():
                ext = '.nc'
            elif 'h5' in file_path.lower() or 'hdf5' in file_path.lower():
                ext = '.h5'
            elif 'csv' in file_path.lower():
                ext = '.csv'
            elif 'grib' in file_path.lower():
                ext = '.grib'
            else:
                raise ValueError(f"Could not determine file extension for: {file_path}")
        
        ext = ext[1:] if ext.startswith('.') else ext
        try:
            data_source_class = self.registry.get_data_source_class(ext)
        except ValueError:
            self.logger.error(f"Unsupported file extension: {ext}")
            raise ValueError(f"Unsupported file extension: {ext}")
        
        return data_source_class(model_name, self.config_manager)
    
    def get_supported_extensions(self) -> List[str]:
        """Get the list of supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return sorted(list(self.registry.get_supported_extensions()))
    
    def is_supported(self, file_path: str) -> bool:
        """Check if the specified file is supported.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            True if the file is supported, False otherwise
        """
        _, ext = os.path.splitext(file_path)
        if not ext:
            return False
        
        ext = ext[1:] if ext.startswith('.') else ext
        
        return self.registry.is_supported(ext)
