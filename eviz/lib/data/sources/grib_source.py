"""
GRIB data source implementation.
"""

import os
import logging
from typing import Dict, List, Optional, Union

import xarray as xr
import numpy as np

from .base import DataSource


class GRIBDataSource(DataSource):
    """Data source implementation for GRIB files.
    
    This class handles loading and processing data from GRIB files.
    """
    
    def __init__(self, model_name: str = None):
        """Initialize a new GRIBDataSource.
        
        Args:
            model_name: Name of the model this data source belongs to
        """
        super().__init__(model_name)
    
    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        return logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from a GRIB file into an Xarray dataset.
        
        Args:
            file_path: Path to the GRIB file
            
        Returns:
            An Xarray dataset containing the loaded data
        """
        self.logger.debug(f"Loading GRIB data from {file_path}")
        
        try:
            # Try to open with xarray's cfgrib engine
            try:
                import cfgrib
                dataset = xr.open_dataset(file_path, engine="cfgrib")
                self.logger.info(f"Loaded GRIB file using cfgrib engine: {file_path}")
            except ImportError:
                self.logger.warning("cfgrib not installed. Trying to use pynio engine.")
                try:
                    dataset = xr.open_dataset(file_path, engine="pynio")
                    self.logger.info(f"Loaded GRIB file using pynio engine: {file_path}")
                except ImportError:
                    self.logger.error("Neither cfgrib nor pynio is installed. Cannot read GRIB files.")
                    raise ImportError("Neither cfgrib nor pynio is installed. Please install one of them to read GRIB files.")
            
            self.dataset = dataset
            dataset = self._process_data(dataset)
            self._extract_metadata(dataset)
            return dataset
            
        except Exception as exc:
            self.logger.error(f"Error loading GRIB file: {file_path}. Exception: {exc}")
            raise
    
    def _process_data(self, dataset: xr.Dataset) -> xr.Dataset:
        """Process the loaded GRIB data.
        
        Args:
            dataset: The dataset to process
            
        Returns:
            The processed dataset
        """
        self.logger.debug("Processing GRIB data")
        
        rename_dict = {}
        coord_mappings = {
            'latitude': 'lat',
            'longitude': 'lon',
            'isobaricInhPa': 'lev',
            'isobaricInPa': 'lev',
            'level': 'lev',
            'valid_time': 'time'
        }
        
        for old_name, new_name in coord_mappings.items():
            if old_name in dataset.coords and new_name not in dataset.coords:
                rename_dict[old_name] = new_name
        
        if rename_dict:
            dataset = dataset.rename(rename_dict)
            self.logger.debug(f"Renamed coordinates: {rename_dict}")
        
        # Convert pressure levels from hPa to Pa if needed
        if 'lev' in dataset.coords:
            lev_values = dataset.coords['lev'].values
            if np.max(lev_values) < 1100:  # Likely in hPa
                dataset = dataset.assign_coords(lev=lev_values * 100)
                self.logger.debug("Converted pressure levels from hPa to Pa")
        
        return dataset
    
    def _extract_metadata(self, dataset: xr.Dataset) -> None:
        """Extract metadata from the dataset.
        
        Args:
            dataset: The dataset to extract metadata from
        """
        if dataset is None:
            return
        
        self.metadata["global_attrs"] = dict(dataset.attrs)
        self.metadata["dimensions"] = {dim: dataset.dims[dim] for dim in dataset.dims}
        self.metadata["variables"] = {}
        for var_name, var in dataset.data_vars.items():
            self.metadata["variables"][var_name] = {
                "dims": var.dims,
                "attrs": dict(var.attrs),
                "dtype": str(var.dtype),
                "shape": var.shape
            }
            
            # Add GRIB-specific metadata if available
            grib_keys = ['shortName', 'name', 'units', 'paramId', 'stepType', 'typeOfLevel']
            for key in grib_keys:
                if key in var.attrs:
                    self.metadata["variables"][var_name][key] = var.attrs[key]
