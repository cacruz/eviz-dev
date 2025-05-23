import logging
import xarray as xr
import h5py
from eviz.lib.data.sources.base import DataSource


class HDF5DataSource(DataSource):
    """Data source implementation for HDF5 files.
    
    This class handles loading and processing data from HDF5 files.
    """
    def __init__(self, model_name: str = None, config_manager=None):
        """Initialize a new HDF5DataSource.
        
        Args:
            model_name: Name of the model this data source belongs to
            config_manager: Configuration manager instance
        """
        super().__init__(model_name, config_manager)
        self.h5_file = None
    
    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        return logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from an HDF5 file into an Xarray dataset.
        
        Args:
            file_path: Path to the HDF5 file
            
        Returns:
            An Xarray dataset containing the loaded data
        """
        self.logger.debug(f"Loading HDF5 data from {file_path}")
        
        try:
            # First try to open with xarray's h5netcdf engine
            try:
                dataset = xr.open_dataset(file_path, engine="h5netcdf")
                self.logger.info(f"Loaded HDF5 file using h5netcdf engine: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to open HDF5 file with h5netcdf engine: {e}")
                # Fall back to manual loading with h5py
                dataset = self._load_with_h5py(file_path)
                self.logger.info(f"Loaded HDF5 file using h5py: {file_path}")
            
            self.dataset = dataset
            self._extract_metadata(dataset)
            return dataset
            
        except Exception as exc:
            self.logger.error(f"Error loading HDF5 file: {file_path}. Exception: {exc}")
            raise
    
    def _load_with_h5py(self, file_path: str) -> xr.Dataset:
        """Load an HDF5 file using h5py and convert to xarray.
        
        Args:
            file_path: Path to the HDF5 file
            
        Returns:
            An Xarray dataset containing the loaded data
        """
        self.h5_file = h5py.File(file_path, 'r')
        dataset_dict = {}
        coords_dict = {}
        self._process_h5_group(self.h5_file, dataset_dict, coords_dict)
        dataset = xr.Dataset(dataset_dict, coords=coords_dict)
        return dataset
    
    def _process_h5_group(self, group, dataset_dict, coords_dict, path=""):
        """Process an HDF5 group recursively.
        
        Args:
            group: The HDF5 group to process
            dataset_dict: Dictionary to store datasets
            coords_dict: Dictionary to store coordinates
            path: Current path in the HDF5 hierarchy
        """
        for key, item in group.items():
            item_path = f"{path}/{key}" if path else key
            
            if isinstance(item, h5py.Group):
                # Recursively process groups
                self._process_h5_group(item, dataset_dict, coords_dict, item_path)
            elif isinstance(item, h5py.Dataset):
                data = item[()]
                
                # Check if this looks like a coordinate variable
                if len(data.shape) == 1 and (key.lower() in ['lat', 'latitude', 'lon', 'longitude', 'time', 'lev', 'level']):
                    coords_dict[key] = data
                else:
                    dims = [f"dim_{i}" for i in range(len(data.shape))]
                    dataset_dict[item_path] = xr.DataArray(data, dims=dims)
                    for attr_key, attr_value in item.attrs.items():
                        dataset_dict[item_path].attrs[attr_key] = attr_value
    
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
    
    def close(self) -> None:
        """Close the HDF5 file and free resources."""
        super().close()
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
