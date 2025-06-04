import os
from typing import Type, List, Optional
from eviz.lib.data.sources import (
    DataSource,
    NetCDFDataSource,
    HDF5DataSource,
    CSVDataSource,
    GRIBDataSource,
    ZARRDataSource
)
from eviz.lib.data.url_validator import is_url, is_opendap_url
from .registry import DataSourceRegistry
from dataclasses import dataclass, field
import logging


@dataclass
class DataSourceFactory:
    """Factory for creating data source instances."""
    config_manager: Optional[object] = None
    registry: DataSourceRegistry = field(init=False)

    def __post_init__(self):
        """Post-initialization setup."""
        self.registry = DataSourceRegistry()
        self._register_default_data_sources()

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def _register_default_data_sources(self) -> None:
        """Register the default data source implementations."""
        self.registry.register(
            ['nc', 'nc4', 'netcdf', 'netcdf4', 'opendap', 'dods', 'dap'],
            NetCDFDataSource)
        self.registry.register(['h5', 'he5', 'hdf5', 'hdf'], HDF5DataSource)
        self.registry.register(['csv', 'dat', 'txt'], CSVDataSource)
        self.registry.register(['grib', 'grib2'], GRIBDataSource)
        self.registry.register(['zarr'], ZARRDataSource)

    def register_data_source(self, extensions: List[str],
                             data_source_class: Type[DataSource]) -> None:
        """Register a custom data source class."""
        self.registry.register(extensions, data_source_class)

    def create_data_source(self, file_path: str, model_name: Optional[str] = None,
                           reader_type: Optional[str] = None,
                           file_format: Optional[str] = None) -> DataSource:
        """ Create a data source instance for the specified file or URL, with optional
            explicit reader_type or format.
        
        Args:
            file_path: Path to the data file or URL
            model_name: Optional name of the model this data source belongs to
            reader_type: Optional explicit reader type (e.g., 'CSV', 'NetCDF')
            file_format: Optional explicit file format (e.g., 'netcdf', 'csv', 'grib')
            
        Returns:
            A data source instance for the specified file
            
        Raises:
            ValueError: If the file type is not supported
        """
        if reader_type is not None:
            reader_type = reader_type.strip().lower()
        elif file_format is not None:
            self.logger.info(
                f"Using explicit format '{file_format}' for file: {file_path}")
            file_format = file_format.strip().lower()
            if file_format in ['netcdf', 'nc', 'nc4']:
                reader_type = 'netcdf'
            elif file_format in ['csv', 'text', 'txt', 'dat']:
                reader_type = 'csv'
            elif file_format in ['hdf5', 'h5', 'he5']:
                reader_type = 'hdf5'
            elif file_format in ['zarr']:
                reader_type = 'zarr'
            elif file_format in ['grib', 'grib2']:
                reader_type = 'grib'
            else:
                self.logger.warning(f"Unknown format: {file_format}, "
                                    f"attempting to infer from file path")

        if reader_type is not None:
            if reader_type == 'csv':
                return CSVDataSource(model_name, self.config_manager)
            elif reader_type in ['netcdf', 'nc']:
                return NetCDFDataSource(model_name, self.config_manager)
            elif reader_type in ['hdf5', 'h5']:
                return HDF5DataSource(model_name, self.config_manager)
            elif reader_type in ['zarr']:
                return ZARRDataSource(model_name, self.config_manager)
            elif reader_type in ['grib', 'grib2']:
                return GRIBDataSource(model_name, self.config_manager)
            else:
                self.logger.error(f"Unsupported explicit reader type: {reader_type}")
                raise ValueError(f"Unsupported explicit reader type: {reader_type}")

        if is_opendap_url(file_path):
            return NetCDFDataSource(model_name, self.config_manager)

        # For regular URLs, try to determine the type from the path
        if is_url(file_path):
            path = file_path.split('?')[0]  # Remove query parameters
            _, ext = os.path.splitext(path)
        else:
            _, ext = os.path.splitext(file_path)

        # Hack to detect WRF files by name, treat as NetCDF
        base = os.path.basename(file_path).lower()
        if base.startswith('wrfout') or 'wrf' in base:
            ext = 'nc'

        if not ext:
            # Try to infer the type from the path
            path_lower = file_path.lower()
            if 'opendap' in path_lower or 'dods' in path_lower or 'dap' in path_lower:
                return NetCDFDataSource(model_name, self.config_manager)
            elif 'nc' in path_lower or 'netcdf' in path_lower:
                ext = '.nc'
            elif 'h5' in path_lower or 'hdf5' in path_lower:
                ext = '.h5'
            elif 'csv' in path_lower:
                ext = '.csv'
            elif 'zarr' in path_lower:
                ext = '.zarr'
            elif 'grib' in path_lower:
                ext = '.grib'
            else:
                raise ValueError(f"Could not determine file type for: {file_path}")

        ext = ext[1:] if ext.startswith('.') else ext
        try:
            data_source_class = self.registry.get_data_source_class(ext)
        except ValueError:
            self.logger.error(f"Unsupported file type: {ext}")
            raise ValueError(f"Unsupported file type: {ext}")

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
