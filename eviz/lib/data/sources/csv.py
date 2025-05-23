import pandas as pd
import xarray as xr
from .base import DataSource


class CSVDataSource(DataSource):
    """Data source implementation for CSV files.
    
    This class handles loading and processing data from CSV files.
    """
    def __init__(self, model_name: str = None, config_manager=None):
        """Initialize a new CSVDataSource.
        
        Args:
            model_name: Name of the model this data source belongs to
            config_manager: Configuration manager instance
        """
        super().__init__(model_name, config_manager)
    
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from a CSV file or a list of CSV files into an Xarray dataset."""
        self.logger.debug(f"Loading CSV data from {file_path}")

        try:
            combined_data = pd.DataFrame()

            if isinstance(file_path, list):
                # Handle multiple files
                for f in file_path:
                    self.logger.debug(f"Reading file: {f}")
                    this_data = pd.read_csv(f)
                    combined_data = pd.concat([combined_data, this_data], ignore_index=True)
            else:
                # Handle a single file
                self.logger.debug(f"Reading file: {file_path}")
                combined_data = pd.read_csv(file_path)

            dataset = combined_data.to_xarray()
            dataset = self._process_data(dataset)
            self.dataset = dataset
            self._extract_metadata(dataset)

            return dataset

        except Exception as exc:
            self.logger.error(f"Error loading CSV file: {file_path}. Exception: {exc}")
            raise
    
    def _process_data(self, dataset: xr.Dataset) -> xr.Dataset:
        """Process the loaded CSV data.
        
        Args:
            dataset: The dataset to process
            
        Returns:
            The processed dataset
        """
        self.logger.debug("Processing CSV data")
        
        for var_name in dataset.variables:
            # Skip coordinate variables
            if var_name in dataset.dims:
                continue
                
            var = dataset[var_name]
            # Is this a date/time column?
            if var_name.lower() in ['date', 'time', 'datetime', 'timestamp']:
                try:
                    # Convert to datetime and set as a coordinate
                    dates = pd.to_datetime(var.values)
                    dataset = dataset.assign_coords(time=dates)
                    self.logger.debug(f"Converted {var_name} to datetime coordinate")
                except Exception as e:
                    self.logger.warning(f"Failed to convert {var_name} to datetime: {e}")
        
        # Check for lat/lon columns and set as coordinates
        lat_names = ['lat', 'latitude', 'y']
        lon_names = ['lon', 'longitude', 'x']
        
        for var_name in dataset.variables:
            if var_name.lower() in lat_names:
                dataset = dataset.assign_coords(lat=dataset[var_name])
                self.logger.debug(f"Set {var_name} as latitude coordinate")
            elif var_name.lower() in lon_names:
                dataset = dataset.assign_coords(lon=dataset[var_name])
                self.logger.debug(f"Set {var_name} as longitude coordinate")
        
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
