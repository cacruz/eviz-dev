"""
Base DataSource class that defines the interface for all data sources.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import logging
import xarray as xr


@dataclass
class DataSource(ABC):
    """Abstract base class that defines the interface for all data sources.

    All data sources are represented as Xarray datasets internally.
    Subclasses must implement the `load_data` method to populate the dataset.
    """
    model_name: Optional[str] = None
    config_manager: Optional[
        object] = None  # Replace 'object' with actual config manager type
    dataset: Optional[xr.Dataset] = field(default=None, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        return logging.getLogger(__name__)

    @abstractmethod
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from the specified file path into an Xarray dataset.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            An Xarray dataset containing the loaded data
        """
        raise NotImplementedError("Subclasses must implement the load_data method.")
    
    def validate_data(self) -> bool:
        """Validate the loaded data.
        
        Returns:
            True if the data is valid, False otherwise
        """
        self.logger.debug("Validating data")
        if self.dataset is None:
            self.logger.error("No data has been loaded")
            return False
        return True
    
    def get_field(self, field_name: str) -> Optional[xr.DataArray]:
        """Get a specific field from the dataset.
        
        Args:
            field_name: Name of the field to retrieve
            
        Returns:
            DataArray containing the field data, or None if the field doesn't exist
        """
        if self.dataset is None:
            self.logger.error("No data has been loaded")
            return None
        
        try:
            return self.dataset[field_name]
        except KeyError:
            self.logger.error(f"Field '{field_name}' not found in dataset")
            return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the dataset.
        
        Returns:
            Dictionary containing metadata
        """
        return self.metadata
    
    def get_dimensions(self) -> List[str]:
        """Get the dimensions of the dataset.
        
        Returns:
            List of dimension names
        """
        if self.dataset is None:
            return []
        return list(self.dataset.dims)
    
    def get_variables(self) -> List[str]:
        """Get the variables in the dataset.
        
        Returns:
            List of variable names
        """
        if self.dataset is None:
            return []
        return list(self.dataset.data_vars)
    
    def close(self) -> None:
        """Close the dataset and free resources."""
        if hasattr(self.dataset, 'close'):
            self.dataset.close()
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying dataset.
        
        This allows users to call xarray.Dataset methods directly on DataSource objects
        without having to access the .dataset attribute explicitly.
        
        Args:
            name: Name of the attribute to access
            
        Returns:
            The attribute from the underlying dataset
            
        Raises:
            AttributeError: If the attribute doesn't exist in the dataset
        """
        if self.dataset is None:
            raise AttributeError(f"'{self.__class__.__name__}' has no dataset loaded")
        
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Delegate item access to the underlying dataset.
        
        This allows users to access variables using square brackets directly on DataSource objects
        without having to access the .dataset attribute explicitly.
        
        Args:
            key: Key to access (usually a variable name)
            
        Returns:
            The item from the underlying dataset
            
        Raises:
            KeyError: If the key doesn't exist in the dataset
            TypeError: If no dataset is loaded
        """
        if self.dataset is None:
            raise TypeError(f"'{self.__class__.__name__}' has no dataset loaded")
        
        return self.dataset[key]
        
    def _get_model_dim_name2(self, dim_name, dims):
        # Get the list of valid values from the meta_coords dictionary
        meta_coords = self.config_manager.meta_coords
        if isinstance(meta_coords[dim_name][self.model_name], list):
            for valid_value in meta_coords[dim_name][self.model_name]:
                for dim in dims:
                    if dim_name == 'xc' and dim in valid_value:
                        return dim
                    if dim_name == 'yc' and dim in valid_value:
                        return dim
                    if dim_name == 'zc' and dim in valid_value:
                        return dim
                    if dim_name == 'tc' and dim in valid_value:
                        return dim
                return None
        else:
            valid_values = meta_coords[dim_name][self.model_name].split(',')
            # Check if any entry in the dimensions list is in the valid values list
            for dim in dims:
                if dim_name == 'xc' and dim in valid_values:
                    return dim
                if dim_name == 'yc' and dim in valid_values:
                    return dim
                if dim_name == 'zc' and dim in valid_values:
                    return dim
                if dim_name == 'tc' and dim in valid_values:
                    return dim
            return None
        
    def _get_model_dim_name(self, gridded_dim_name, available_dims=None):
        """
        Get the model-specific dimension name for a gridded dimension.
        
        Args:
            gridded_dim_name: Gridded dimension name (e.g., 'xc', 'yc', 'zc', 'tc')
            available_dims: List of available dimensions in the dataset
            
        Returns:
            str: The model-specific dimension name if found, otherwise None
        """
        if not hasattr(self, 'config_manager') or not self.config_manager:
            self.logger.warning("No config_manager available to get dimension mappings")
            return None
            
        if not hasattr(self.config_manager, 'meta_coords'):
            self.logger.warning("No meta_coords available in config_manager")
            return None
            
        meta_coords = self.config_manager.meta_coords
        if gridded_dim_name not in meta_coords:
            self.logger.warning(f"No mapping found for dimension '{gridded_dim_name}'")
            return None
            
        model_name = self.model_name
        
        self.logger.debug(f"Looking for model '{model_name}' in meta_coords['{gridded_dim_name}']")
        self.logger.debug(f"Available models for {gridded_dim_name}: {list(meta_coords[gridded_dim_name].keys())}")
        
        if not model_name or model_name not in meta_coords[gridded_dim_name]:
            # Try to use a default model if available
            if 'gridded' in meta_coords[gridded_dim_name]:
                self.logger.debug(f"Using 'gridded' mapping for model '{model_name}' and dimension '{gridded_dim_name}'")
                model_name = 'gridded'
            else:
                self.logger.warning(f"No mapping found for model '{model_name}' and dimension '{gridded_dim_name}'")
                return None
        
        coords = meta_coords[gridded_dim_name][model_name]
            
        if isinstance(coords, list):
            for coord in coords:
                if available_dims and coord in available_dims:
                    return coord
            return coords[0] if coords else None
        
        elif isinstance(coords, dict):
            if 'dim' in coords:
                if available_dims:
                    if ',' in coords['dim']:
                        dim_candidates = coords['dim'].split(',')
                        for dim in dim_candidates:
                            if dim in available_dims:
                                return dim
                        return None
                    # Single dimension name
                    return coords['dim'] if coords['dim'] in available_dims else None
                return coords['dim']
            
            if 'coords' in coords:
                # For coordinate names
                return coords['coords']    
            return None
        
        elif isinstance(coords, str):
            # If coords is a string, handle comma-separated list of possible dimension names
            if ',' in coords:
                coord_candidates = coords.split(',')
                if available_dims:
                    for coord in coord_candidates:
                        if coord in available_dims:
                            return coord
                    # No matching dimension found
                    return None
                return coord_candidates[0]
            return coords
        
        self.logger.warning(f"Unexpected type for coords: {type(coords)}")
        return None
