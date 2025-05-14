"""
Data transformation stage of the pipeline.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import xarray as xr
import numpy as np
import scipy.interpolate as interp

from eviz.lib.data.sources import DataSource


class DataTransformer:
    """Data transformation stage of the pipeline.
    
    This class handles transforming data from data sources.
    """
    
    def __init__(self):
        """Initialize a new DataTransformer."""
        self.logger = logging.getLogger(__name__)
    
    def transform_data_source(self, data_source: DataSource, **kwargs) -> DataSource:
        """Transform a data source.
        
        Args:
            data_source: The data source to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            The transformed data source
        """
        self.logger.debug("Transforming data source")
        
        if not data_source.validate_data():
            self.logger.error("Data validation failed")
            return data_source
        
        data_source.dataset = self._transform_dataset(data_source.dataset, **kwargs)
        
        return data_source
    
    def _transform_dataset(self, dataset: xr.Dataset, **kwargs) -> xr.Dataset:
        """Transform an Xarray dataset.
        
        Args:
            dataset: The dataset to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            The transformed dataset
        """
        if dataset is None:
            return None
        
        if kwargs.get('regrid', False):
            dataset = self._regrid_dataset(dataset, **kwargs)
        
        if kwargs.get('subset', False):
            dataset = self._subset_dataset(dataset, **kwargs)
        
        if kwargs.get('time_average', False):
            dataset = self._time_average_dataset(dataset, **kwargs)
        
        if kwargs.get('vertical_average', False):
            dataset = self._vertical_average_dataset(dataset, **kwargs)
        
        if kwargs.get('vertical_sum', False):
            dataset = self._vertical_sum_dataset(dataset, **kwargs)
        
        return dataset
    
    def _regrid_dataset(self, dataset: xr.Dataset, **kwargs) -> xr.Dataset:
        """Regrid a dataset to a new grid.
        
        Args:
            dataset: The dataset to regrid
            **kwargs: Additional regridding parameters
            
        Returns:
            The regridded dataset
        """
        self.logger.debug("Regridding dataset")
        
        target_grid = kwargs.get('target_grid')
        method = kwargs.get('method', 'linear')
        
        if target_grid is None:
            self.logger.warning("No target grid specified for regridding")
            return dataset
        
        # Create the new grid
        if isinstance(target_grid, dict):
            # Create a new grid from the specified parameters
            lat_min = target_grid.get('lat_min', -90)
            lat_max = target_grid.get('lat_max', 90)
            lon_min = target_grid.get('lon_min', -180)
            lon_max = target_grid.get('lon_max', 180)
            lat_res = target_grid.get('lat_res', 1.0)
            lon_res = target_grid.get('lon_res', 1.0)
            
            new_lat = np.arange(lat_min, lat_max + lat_res, lat_res)
            new_lon = np.arange(lon_min, lon_max + lon_res, lon_res)
            
        elif isinstance(target_grid, tuple) and len(target_grid) == 2:
            # Use the provided lat/lon arrays
            new_lat, new_lon = target_grid
            
        else:
            self.logger.error("Invalid target grid specification")
            return dataset
        
        new_dataset = xr.Dataset(coords={'lat': new_lat, 'lon': new_lon})
        
        for var_name, var in dataset.data_vars.items():
            # Skip variables that don't have lat/lon dimensions
            if 'lat' not in var.dims or 'lon' not in var.dims:
                new_dataset[var_name] = var
                continue
            
            orig_lat = dataset.coords['lat'].values
            orig_lon = dataset.coords['lon'].values
            
            orig_lon_grid, orig_lat_grid = np.meshgrid(orig_lon, orig_lat)
            new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)
            
            if len(var.dims) == 2:  # 2D variable (lat, lon)
                regridded_data = self._regrid_2d(var.values, orig_lat_grid, orig_lon_grid,
                                                new_lat_grid, new_lon_grid, method)
                new_dataset[var_name] = xr.DataArray(regridded_data, dims=('lat', 'lon'),
                                                    coords={'lat': new_lat, 'lon': new_lon},
                                                    attrs=var.attrs)
                
            elif len(var.dims) == 3:  # 3D variable (time, lat, lon) or (lev, lat, lon)
                if var.dims[0] == 'time':
                    regridded_data = np.zeros((len(var.coords['time']), len(new_lat), len(new_lon)))
                    for t in range(len(var.coords['time'])):
                        regridded_data[t] = self._regrid_2d(var.values[t], orig_lat_grid, orig_lon_grid,
                                                        new_lat_grid, new_lon_grid, method)
                    new_dataset[var_name] = xr.DataArray(regridded_data, dims=('time', 'lat', 'lon'),
                                                        coords={'time': var.coords['time'], 'lat': new_lat, 'lon': new_lon},
                                                        attrs=var.attrs)
                else:
                    regridded_data = np.zeros((len(var.coords['lev']), len(new_lat), len(new_lon)))
                    for z in range(len(var.coords['lev'])):
                        regridded_data[z] = self._regrid_2d(var.values[z], orig_lat_grid, orig_lon_grid,
                                                        new_lat_grid, new_lon_grid, method)
                    new_dataset[var_name] = xr.DataArray(regridded_data, dims=('lev', 'lat', 'lon'),
                                                        coords={'lev': var.coords['lev'], 'lat': new_lat, 'lon': new_lon},
                                                        attrs=var.attrs)
                
            elif len(var.dims) == 4:  # 4D variable (time, lev, lat, lon)
                regridded_data = np.zeros((len(var.coords['time']), len(var.coords['lev']), len(new_lat), len(new_lon)))
                for t in range(len(var.coords['time'])):
                    for z in range(len(var.coords['lev'])):
                        regridded_data[t, z] = self._regrid_2d(var.values[t, z], orig_lat_grid, orig_lon_grid,
                                                            new_lat_grid, new_lon_grid, method)
                new_dataset[var_name] = xr.DataArray(regridded_data, dims=('time', 'lev', 'lat', 'lon'),
                                                    coords={'time': var.coords['time'], 'lev': var.coords['lev'],
                                                            'lat': new_lat, 'lon': new_lon},
                                                    attrs=var.attrs)
        
        return new_dataset
    
    def _regrid_2d(self, data: np.ndarray, orig_lat_grid: np.ndarray, orig_lon_grid: np.ndarray,
                  new_lat_grid: np.ndarray, new_lon_grid: np.ndarray, method: str = 'linear') -> np.ndarray:
        """Regrid a 2D array to a new grid.
        
        Args:
            data: The data to regrid
            orig_lat_grid: The original latitude grid
            orig_lon_grid: The original longitude grid
            new_lat_grid: The new latitude grid
            new_lon_grid: The new longitude grid
            method: The interpolation method
            
        Returns:
            The regridded data
        """
        orig_points = np.column_stack((orig_lat_grid.flatten(), orig_lon_grid.flatten()))
        orig_values = data.flatten()
        
        valid_indices = ~np.isnan(orig_values)
        orig_points = orig_points[valid_indices]
        orig_values = orig_values[valid_indices]
        
        if method == 'nearest':
            regridded_data = interp.griddata(orig_points, orig_values,
                                            (new_lat_grid, new_lon_grid),
                                            method='nearest')
        else:
            # Use linear interpolation with nearest extrapolation
            regridded_data = interp.griddata(orig_points, orig_values,
                                            (new_lat_grid, new_lon_grid),
                                            method='linear')
            
            # Fill NaN values with nearest neighbor interpolation
            if np.any(np.isnan(regridded_data)):
                nn_data = interp.griddata(orig_points, orig_values,
                                        (new_lat_grid, new_lon_grid),
                                        method='nearest')
                regridded_data = np.where(np.isnan(regridded_data), nn_data, regridded_data)
        
        return regridded_data
    
    def _subset_dataset(self, dataset: xr.Dataset, **kwargs) -> xr.Dataset:
        """Subset a dataset.
        
        Args:
            dataset: The dataset to subset
            **kwargs: Additional subsetting parameters
            
        Returns:
            The subsetted dataset
        """
        self.logger.debug("Subsetting dataset")
        
        lat_range = kwargs.get('lat_range')
        lon_range = kwargs.get('lon_range')
        time_range = kwargs.get('time_range')
        lev_range = kwargs.get('lev_range')
        
        if lat_range is not None and 'lat' in dataset.coords:
            dataset = dataset.sel(lat=slice(lat_range[0], lat_range[1]))
        
        if lon_range is not None and 'lon' in dataset.coords:
            dataset = dataset.sel(lon=slice(lon_range[0], lon_range[1]))
        
        if time_range is not None and 'time' in dataset.coords:
            dataset = dataset.sel(time=slice(time_range[0], time_range[1]))
        
        if lev_range is not None and 'lev' in dataset.coords:
            dataset = dataset.sel(lev=slice(lev_range[0], lev_range[1]))
        
        return dataset
    
    def _time_average_dataset(self, dataset: xr.Dataset, **kwargs) -> xr.Dataset:
        """Compute time average of a dataset.
        
        Args:
            dataset: The dataset to average
            **kwargs: Additional averaging parameters
            
        Returns:
            The time-averaged dataset
        """
        self.logger.debug("Computing time average of dataset")
        
        time_dim = kwargs.get('time_dim', 'time')
        
        if time_dim not in dataset.dims:
            self.logger.warning(f"Time dimension '{time_dim}' not found in dataset")
            return dataset
        
        return dataset.mean(dim=time_dim)
    
    def _vertical_average_dataset(self, dataset: xr.Dataset, **kwargs) -> xr.Dataset:
        """Compute vertical average of a dataset.
        
        Args:
            dataset: The dataset to average
            **kwargs: Additional averaging parameters
            
        Returns:
            The vertically-averaged dataset
        """
        self.logger.debug("Computing vertical average of dataset")
        
        lev_dim = kwargs.get('lev_dim', 'lev')
        
        if lev_dim not in dataset.dims:
            self.logger.warning(f"Vertical dimension '{lev_dim}' not found in dataset")
            return dataset
        
        return dataset.mean(dim=lev_dim)
    
    def _vertical_sum_dataset(self, dataset: xr.Dataset, **kwargs) -> xr.Dataset:
        """Compute vertical sum of a dataset.
        
        Args:
            dataset: The dataset to sum
            **kwargs: Additional summing parameters
            
        Returns:
            The vertically-summed dataset
        """
        self.logger.debug("Computing vertical sum of dataset")
        
        lev_dim = kwargs.get('lev_dim', 'lev')
        
        if lev_dim not in dataset.dims:
            self.logger.warning(f"Vertical dimension '{lev_dim}' not found in dataset")
            return dataset
        
        return dataset.sum(dim=lev_dim)
