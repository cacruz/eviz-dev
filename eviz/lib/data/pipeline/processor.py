"""
Data processing stage of the pipeline.
"""

import logging
from typing import Dict, List, Optional, Union, Any

import xarray as xr
import numpy as np

from eviz.lib.data.sources import DataSource


class DataProcessor:
    """Data processing stage of the pipeline.
    
    This class handles processing data from data sources.
    """
    
    def __init__(self):
        """Initialize a new DataProcessor."""
        self.logger = logging.getLogger(__name__)
    
    def process_data_source(self, data_source: DataSource) -> DataSource:
        """Process a data source.
        
        Args:
            data_source: The data source to process
            
        Returns:
            The processed data source
        """
        self.logger.debug("Processing data source")
        
        if not data_source.validate_data():
            self.logger.error("Data validation failed")
            return data_source
        data_source.dataset = self._process_dataset(data_source.dataset)
        
        return data_source
    
    def _process_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """Process an Xarray dataset.
        
        Args:
            dataset: The dataset to process
            
        Returns:
            The processed dataset
        """
        if dataset is None:
            return None
        
        dataset = self._standardize_coordinates(dataset)
        dataset = self._handle_missing_values(dataset)
        dataset = self._apply_unit_conversions(dataset)
        
        return dataset
    
    def _standardize_coordinates(self, dataset: xr.Dataset) -> xr.Dataset:
        """Standardize coordinate names and values.
        
        Args:
            dataset: The dataset to process
            
        Returns:
            The processed dataset
        """
        coord_mappings = {
            'latitude': 'lat',
            'longitude': 'lon',
            'level': 'lev',
            'height': 'lev',
            'depth': 'lev',
            'pressure': 'lev',
            'time_bnds': 'time_bounds',
            'lat_bnds': 'lat_bounds',
            'lon_bnds': 'lon_bounds',
        }
        
        rename_dict = {}
        for old_name, new_name in coord_mappings.items():
            if old_name in dataset.coords and new_name not in dataset.coords:
                rename_dict[old_name] = new_name
        
        if rename_dict:
            dataset = dataset.rename(rename_dict)
            self.logger.debug(f"Renamed coordinates: {rename_dict}")
        
        # Ensure latitude is in the range [-90, 90]
        if 'lat' in dataset.coords:
            lat_values = dataset.coords['lat'].values
            if np.any(lat_values > 90) or np.any(lat_values < -90):
                self.logger.warning("Latitude values outside the range [-90, 90]")
                # Attempt to fix by normalizing to [-90, 90]
                lat_values = np.clip(lat_values, -90, 90)
                dataset = dataset.assign_coords(lat=lat_values)
                self.logger.debug("Normalized latitude values to the range [-90, 90]")
        
        # Ensure longitude is in the range [-180, 180] or [0, 360]
        if 'lon' in dataset.coords:
            lon_values = dataset.coords['lon'].values
            if np.any(lon_values > 360) or np.any(lon_values < -180):
                self.logger.warning("Longitude values outside the range [-180, 180] or [0, 360]")
                # Attempt to fix by normalizing to [-180, 180]
                lon_values = ((lon_values + 180) % 360) - 180
                dataset = dataset.assign_coords(lon=lon_values)
                self.logger.debug("Normalized longitude values to the range [-180, 180]")
        
        return dataset
    
    def _handle_missing_values(self, dataset: xr.Dataset) -> xr.Dataset:
        """Handle missing values in the dataset.
        
        Args:
            dataset: The dataset to process
            
        Returns:
            The processed dataset
        """
        # Replace NaN values with the _FillValue attribute if available
        for var_name, var in dataset.data_vars.items():
            if '_FillValue' in var.attrs:
                fill_value = var.attrs['_FillValue']
                if np.isnan(fill_value):
                    continue  # Skip if the fill value is already NaN
                
                # Replace NaN values with the fill value
                var_data = var.values
                var_data[np.isnan(var_data)] = fill_value
                dataset[var_name] = xr.DataArray(var_data, dims=var.dims, coords=var.coords, attrs=var.attrs)
                self.logger.debug(f"Replaced NaN values with fill value {fill_value} for variable {var_name}")
        
        return dataset
    
    def _apply_unit_conversions(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply unit conversions to the dataset.
        
        Args:
            dataset: The dataset to process
            
        Returns:
            The processed dataset
        """
        # Apply common unit conversions
        for var_name, var in dataset.data_vars.items():
            if 'units' not in var.attrs:
                continue
            
            units = var.attrs['units'].lower()
            
            # Convert temperature from Kelvin to Celsius if needed
            if units == 'k' and var_name.lower() in ['temp', 'temperature', 'air_temperature']:
                var_data = var.values - 273.15
                dataset[var_name] = xr.DataArray(var_data, dims=var.dims, coords=var.coords, attrs=var.attrs)
                dataset[var_name].attrs['units'] = 'C'
                self.logger.debug(f"Converted temperature from Kelvin to Celsius for variable {var_name}")
            
            # Convert pressure from hPa to Pa if needed
            elif units == 'hpa' and var_name.lower() in ['pressure', 'air_pressure', 'surface_pressure']:
                var_data = var.values * 100
                dataset[var_name] = xr.DataArray(var_data, dims=var.dims, coords=var.coords, attrs=var.attrs)
                dataset[var_name].attrs['units'] = 'Pa'
                self.logger.debug(f"Converted pressure from hPa to Pa for variable {var_name}")
        
        return dataset
