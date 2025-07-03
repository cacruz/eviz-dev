import glob
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
import numpy as np
from eviz.lib.data.factory import DataSourceFactory
from eviz.lib.data.sources import DataSource
from eviz.lib.data.url_validator import is_url


@dataclass
class DataReader:
    """Data reading stage of the pipeline."""
    config_manager: Optional[object] = None 
    data_sources: Dict = field(default_factory=dict, init=False)
    factory: object = field(init=False) 

    def __post_init__(self):
        """Post-initialization to set up factory and logger."""
        self.factory = DataSourceFactory(self.config_manager)

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def read_file(self, file_path: str, model_name: Optional[str] = None, file_format: Optional[str] = None) -> DataSource:
        """Read data from a file or URL, supporting wildcards.
        
        Args:
            file_path: Path to the file or URL
            model_name: Optional name of the model this data source belongs to
            file_format: Optional explicit file format (e.g., 'netcdf', 'csv')
            
        Returns:
            A data source for the file
        """
        self.logger.debug(f"Reading file: {file_path}")

        is_remote = is_url(file_path)

        if not is_remote and ('*' in file_path or '?' in file_path or '[' in file_path):
            files = glob.glob(file_path)
            if not files:
                self.logger.error(f"No files found matching pattern: {file_path}")
                raise FileNotFoundError(f"No files found matching pattern: {file_path}")

            self.logger.info(f"Found {len(files)} files matching pattern: {file_path}")

            # Use the factory to create a data source for the first file (to get the right type)
            data_source = self.factory.create_data_source(files[0], model_name, file_format=file_format)
            # If the data source supports loading multiple files, do so
            if hasattr(data_source, "load_data"):
                data_source.load_data(files)
            else:
                datasets = []
                for f in files:
                    ds = self.factory.create_data_source(f, model_name, file_format=file_format)
                    ds.load_data(f)
                    datasets.append(ds.dataset)
                # Combine datasets as appropriate (e.g., pd.concat for CSV, xr.concat for NetCDF)
                # Here, we just assign the first for simplicity
                data_source.dataset = datasets[0]  # You may want to implement a real combine
            self.data_sources[file_path] = data_source
            return data_source

        if not is_remote and not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path in self.data_sources:
            self.logger.debug(f"Using cached data source for {file_path}")
            return self.data_sources[file_path]

        try:
            data_source = self.factory.create_data_source(file_path, model_name, file_format=file_format)
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
                file_format = None
                if self.config_manager and hasattr(self.config_manager, 'get_file_format'):
                    file_format = self.config_manager.get_file_format(file_path)
                
                data_source = self.read_file(file_path, model_name, file_format=file_format)
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
        for data_source in self.data_sources.values():
            if hasattr(data_source, 'close'):
                data_source.close()
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
        if hasattr(data_array, 'dims'):
            dim_candidates = ['lon', 'longitude', 'x', 'east_west', 'west_east']
            for dim in dim_candidates:
                if dim in data_array.dims:
                    return data_array[dim].values

            if data_array.dims:
                return data_array[data_array.dims[0]].values

        return np.arange(data_array.shape[0])

    # Original implementation for when attribute_name is provided
    attribute_mapping = {
        'time': ['time', 't', 'TIME'],
        'lon': ['lon', 'longitude', 'x', 'east_west', 'west_east'],
        'lat': ['lat', 'latitude', 'y', 'notrt_south', 'south_north'],
        'lev': ['lev', 'level', 'z', 'altitude', 'height', 'depth', 'plev'],
    }

    for gridded, specific_list in attribute_mapping.items():
        if attribute_name in specific_list:
            attribute_name = gridded
            break

    if attribute_name in attribute_mapping:
        for specific_name in attribute_mapping[attribute_name]:
            if specific_name in data_array.dims:
                return data_array[specific_name].values
            elif specific_name in data_array.coords:
                return data_array.coords[specific_name].values

    if attribute_name in data_array.dims:
        return data_array[attribute_name].values
    elif attribute_name in data_array.coords:
        return data_array.coords[attribute_name].values

    raise ValueError(f"GriddedSource name for {attribute_name} not found in attribute_mapping.")
