import os
import logging
import xarray as xr
from dask.distributed import Client
from dataclasses import dataclass, field
from typing import Optional, Dict
from .base import DataSource
from eviz.lib.data.url_validator import is_url, is_opendap_url


@dataclass
class NetCDFDataSource(DataSource):
    """Data source implementation for NetCDF files.

    This class handles loading and processing data from NetCDF files and OpenDAP URLs.
    """
    model_name: Optional[str] = None
    config_manager: Optional[object] = None 
    datasets: Dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Post-initialization to ensure base class is properly initialized."""
        super().__init__(self.model_name, self.config_manager)

    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        return logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> xr.Dataset:
        """
        Load data from a NetCDF file or OpenDAP URL into an Xarray dataset.
        """
        self.logger.debug(f"Loading NetCDF data from {file_path}")

        try:
            # Check if it's a URL
            is_remote = is_url(file_path)
            is_opendap = is_opendap_url(file_path)

            # If file_path is a list (from globbing), use open_mfdataset
            if isinstance(file_path, list):
                files = file_path
            elif "*" in file_path or "?" in file_path or "[" in file_path:
                import glob
                files = sorted(glob.glob(file_path))
            else:
                files = [file_path]

            if is_remote:
                self.logger.info(f"Loading remote data from URL: {file_path}")
                if is_opendap:
                    self.logger.info(f"Detected OpenDAP URL: {file_path}")
                dataset = xr.open_dataset(file_path, decode_cf=True, engine='netcdf4')
                self.logger.debug(f"Loaded remote NetCDF data: {file_path}")
            elif len(files) == 1:
                dataset = xr.open_dataset(files[0], decode_cf=True)
                self.logger.debug(f"Loaded single NetCDF file: {files[0]}")
            elif len(files) > 1:
                self._setup_dask_client()

                # Possible optimization for large datasets:
                vars_to_keep = set()
                for entry in self.config_manager.app_data.inputs:
                    if 'to_plot' in entry:
                        vars_to_keep.update(entry['to_plot'].keys())
                self.logger.info(f"Variables to keep: {vars_to_keep}")        
                def drop_unneeded_vars(ds):
                    available = set(ds.data_vars)
                    keep = [v for v in vars_to_keep if v in available]
                    return ds[keep]
                dataset = xr.open_mfdataset(
                    files,
                    decode_cf=True,
                    combine="by_coords",
                    parallel=True,
                    preprocess=drop_unneeded_vars
                )

                # No-op:
                # dataset = xr.open_mfdataset(file_path, decode_cf=True, engine='netcdf4')
                # self.logger.debug(f"Loaded multiple NetCDF files: {files}")

            else:
                raise FileNotFoundError(f"No files found for pattern: {file_path}")

            # Standardize dimension names
            dataset = self._rename_dims(dataset)

            self.dataset = dataset
            self._extract_metadata(dataset)

            # For URLs, use the last part of the path as the file name
            if is_remote:
                file_name = os.path.basename(file_path.split('?')[0])  # Remove query parameters
            else:
                file_name = os.path.basename(files[0])

            self.datasets[file_name] = dataset

            return dataset

        except FileNotFoundError as exc:
            self.logger.error(f"Error loading NetCDF file: {file_path}. Exception: {exc}")
            raise
        except Exception as exc:
            self.logger.error(f"Error loading NetCDF data: {file_path}. Exception: {exc}")
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
        
        available_dims = list(ds.dims)
        
        xc = self._get_model_dim_name('xc', available_dims)
        yc = self._get_model_dim_name('yc', available_dims)
        zc = self._get_model_dim_name('zc', available_dims)
        tc = self._get_model_dim_name('tc', available_dims)
        
        rename_dict = {}
        
        # Add mappings only for dimensions that exist and need renaming
        if xc and xc != 'lon' and xc in available_dims:
            rename_dict[xc] = 'lon'
        
        if yc and yc != 'lat' and yc in available_dims:
            rename_dict[yc] = 'lat'
        
        if zc and zc != 'lev' and zc in available_dims:
            rename_dict[zc] = 'lev'
        
        if tc and tc != 'time' and tc in available_dims:
            rename_dict[tc] = 'time'
        
        if rename_dict:
            self.logger.debug(f"Renaming dimensions: {rename_dict}")
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
