import xarray as xr
from .base import DataSource


class ZARRDataSource(DataSource):
    """Data source implementation for Zarr files.
    
    This class handles loading and processing data from Zarr stores.
    """
    def __init__(self, model_name: str = None, config_manager=None):
        """Initialize a new ZARR DataSource.
        
        Args:
            model_name: Name of the model this data source belongs to
            config_manager: Configuration manager instance
        """
        super().__init__(model_name, config_manager)
    
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from a Zarr store into an Xarray dataset.
        
        Args:
            file_path: Path to the Zarr store
            
        Returns:
            The loaded dataset
        """
        self.logger.debug(f"Loading Zarr data from {file_path}")

        try:
            # Handle single file or list of files
            if isinstance(file_path, list):
                # For multiple Zarr stores, open and merge them
                datasets = []
                for f in file_path:
                    self.logger.debug(f"Reading Zarr store: {f}")
                    # Use the store engine instead of zarr directly
                    ds = xr.open_dataset(f, engine="zarr")
                    datasets.append(ds)
                
                # Combine all datasets
                if datasets:
                    dataset = xr.merge(datasets)
                else:
                    raise ValueError("No valid Zarr stores found in the provided list")
            else:
                # Handle a single Zarr store
                self.logger.debug(f"Reading Zarr store: {file_path}")
                # Use the store engine instead of zarr directly
                dataset = xr.open_dataset(file_path, engine="zarr")
            
            dataset = self._process_data(dataset)
            self.dataset = dataset
            self._extract_metadata(dataset)
            
            return dataset
            
        except Exception as exc:
            self.logger.error(f"Error loading Zarr store: {file_path}. Exception: {exc}")
            raise
    
    def _process_data(self, dataset: xr.Dataset) -> xr.Dataset:
        """Process the loaded Zarr data.
        
        Args:
            dataset: The dataset to process
            
        Returns:
            The processed dataset
        """
        self.logger.debug("Processing Zarr data")
        
        # Zarr files typically already have well-defined coordinates,
        # but we can add additional processing if needed
        
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
            
            # Add some basic statistics
            try:
                self.metadata["variables"][var_name]["stats"] = {
                    "min": float(var.min().values),
                    "max": float(var.max().values),
                    "mean": float(var.mean().values),
                    "std": float(var.std().values)
                }
            except Exception:
                # Skip statistics if they can't be computed
                pass
