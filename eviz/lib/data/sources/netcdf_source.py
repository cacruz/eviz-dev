"""
NetCDF data source implementation.
"""

import os
import logging
from typing import Dict, Optional

import xarray as xr
from dask.distributed import Client

from .base import DataSource


class NetCDFDataSource(DataSource):
    """Data source implementation for NetCDF files.
    
    This class handles loading and processing data from NetCDF files.
    """
    
    def __init__(self, model_name: str = None, config_manager=None):
        """Initialize a new NetCDFDataSource.
        
        Args:
            model_name: Name of the model this data source belongs to
            config_manager: Configuration manager instance
        """
        super().__init__(model_name, config_manager)
        self.datasets = {}  # Dictionary to store datasets by file name

    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        return logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from a NetCDF file into an Xarray dataset.
        
        Args:
            file_path: Path to the NetCDF file or a glob pattern
            
        Returns:
            An Xarray dataset containing the loaded data
        """
        self.logger.debug(f"Loading NetCDF data from {file_path}")    
        try:
            if "*" in file_path:
                # Handle multiple files using a glob pattern
                self._setup_dask_client()
                dataset = xr.open_mfdataset(file_path, decode_cf=True, combine="by_coords")
                self.logger.debug(f"Loaded multiple NetCDF files matching pattern: {file_path}")
            else:
                # Handle a single file
                dataset = xr.open_dataset(file_path, decode_cf=True)
                self.logger.debug(f"Loaded single NetCDF file: {file_path}")
            
            # Standardize dimension names
            dataset = self._rename_dims(dataset)
            
            self.dataset = dataset
            self._extract_metadata(dataset)
            
            file_name = os.path.basename(file_path)
            self.datasets[file_name] = dataset
            
            return dataset
            
        except FileNotFoundError as exc:
            self.logger.error(f"Error loading NetCDF file: {file_path}. Exception: {exc}")
            raise
    
    def _setup_dask_client(self) -> None:
        """Set up a Dask distributed client for parallel computation."""
        try:
            n_workers = max(1, os.cpu_count() - 2)
            client = Client(n_workers=n_workers, threads_per_worker=1)
            self.logger.info(f"Dask dashboard is available at: {client.dashboard_link}")
            self.logger.info(f"Using {n_workers} workers for parallel computation")
        except Exception as exc:
            self.logger.warning(f"Failed to set up Dask client: {exc}. Continuing without parallel computation.")
    
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
    
    def get_dataset(self, file_name: str) -> Optional[xr.Dataset]:
        """Get a specific dataset by file name.
        
        Args:
            file_name: Name of the file
            
        Returns:
            The dataset for the specified file, or None if not found
        """
        return self.datasets.get(file_name)
    
    def get_all_datasets(self) -> Dict[str, xr.Dataset]:
        """Get all loaded datasets.
        
        Returns:
            Dictionary of all loaded datasets
        """
        return self.datasets
    
    def _rename_dims(self, ds):
        """
        Standardize dimension names in the dataset.
        
        This method renames dimensions to standard names (lon, lat, lev, time)
        regardless of their original names in the source data.
        
        Args:
            ds: xarray Dataset to rename dimensions in
            
        Returns:
            xarray Dataset with standardized dimension names
        """
        if self.model_name in ['wrf', 'lis']:
            # Skip renaming for these special models
            return ds
        
        # Get available dimensions
        available_dims = list(ds.dims)
        
        # Get model-specific dimension names
        xc = self._get_model_dim_name('xc', available_dims)
        yc = self._get_model_dim_name('yc', available_dims)
        zc = self._get_model_dim_name('zc', available_dims)
        tc = self._get_model_dim_name('tc', available_dims)
        
        # Create rename mapping
        rename_dict = {}
        
        # Add mappings only for dimensions that exist and need renaming
        if xc and xc != 'lon' and xc in available_dims:
            print(f"Renaming {xc} to lon")
            rename_dict[xc] = 'lon'
        
        if yc and yc != 'lat' and yc in available_dims:
            print(f"Renaming {yc} to lat")
            rename_dict[yc] = 'lat'
        
        if zc and zc != 'lev' and zc in available_dims:
            print(f"Renaming {zc} to lev")
            rename_dict[zc] = 'lev'
        
        if tc and tc != 'time' and tc in available_dims:
            print(f"Renaming {tc} to time") 
            rename_dict[tc] = 'time'
        
        # Only rename if there's something to rename
        if rename_dict:
            self.logger.info(f"Renaming dimensions: {rename_dict}")
            try:
                ds = ds.rename(rename_dict)
            except Exception as e:
                self.logger.error(f"Error renaming dimensions: {e}")
        
        return ds
    
    def close(self) -> None:
        """Close all datasets and free resources."""
        super().close()
        for dataset in self.datasets.values():
            if hasattr(dataset, 'close'):
                dataset.close()
        self.datasets.clear()
