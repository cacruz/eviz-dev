from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod

import logging

import numpy as np
from eviz.lib import const as constants
import eviz.lib.utils as u


@dataclass
class DataReader(ABC):
    source_name: str

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        self.datasets = []
        self.meta_coords = u.read_meta_coords()
        self.meta_attrs = u.read_meta_attrs()

    @abstractmethod
    def read_data(self, file_path: str) -> Any:
        raise NotImplementedError("Subclasses must implement read_data method.")

    def get_field(self, name: str, ds_index: int):
        """ Extract field from xarray Dataset

        Parameters:
            name (str) : name of field to extract from dataset
            ds_index (int) : index of dataset to extract from

        Returns:
            DataArray containing field data
        """
        try:
            self.logger.debug(f" -> getting field {name}, from index {ds_index}")
            self.logger.debug(f" -> from  filename {self.datasets[ds_index]['filename']}")
            return self.datasets[ds_index]['vars'][name]
        except Exception as e:
            self.logger.error('key error: %s, not found' % str(e))
        return None

    @staticmethod
    def get_attrs(data, key):
        """ Get attributes associated with a key"""
        for attr in data.attrs:
            if key == attr:
                return data.attrs[key]
            else:
                continue
        return None

    def get_datasets(self):
        return self.datasets

    def get_dataset(self, i):
        return self.datasets[i]


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