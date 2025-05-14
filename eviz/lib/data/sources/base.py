"""
Base DataSource class that defines the interface for all data sources.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Union
import xarray as xr


class DataSource(ABC):
    """Abstract base class that defines the interface for all data sources.
    
    All data sources are represented as Xarray datasets internally.
    Subclasses must implement the `load_data` method to populate the dataset.
    """
    
    def __init__(self, model_name: str = None, config_manager=None):
        """Initialize a new DataSource.
        
        Args:
            model_name: Name of the model this data source belongs to
            config_manager: Configuration manager instance
        """
        self.model_name = model_name
        self.dataset = None
        self.metadata = {}
        self.config_manager = config_manager
    
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
        
    def _get_model_dim_name(self, generic_dim_name, available_dims=None):
        """
        Get the model-specific dimension name for a generic dimension.
        
        Args:
            generic_dim_name: Generic dimension name (e.g., 'xc', 'yc', 'zc', 'tc')
            available_dims: List of available dimensions in the dataset
            
        Returns:
            str: The model-specific dimension name if found, otherwise None
        """
        if not hasattr(self, 'config_manager') or not self.config_manager:
            self.logger.warning("No config_manager available to get dimension mappings")
            return None
            
        # Get the mapping from meta_coordinates.yaml
        if not hasattr(self.config_manager, 'meta_coords'):
            self.logger.warning("No meta_coords available in config_manager")
            return None
            
        meta_coords = self.config_manager.meta_coords
        if generic_dim_name not in meta_coords:
            self.logger.warning(f"No mapping found for dimension '{generic_dim_name}'")
            return None
            
        # Get the mapping for this model
        if not self.model_name or self.model_name not in meta_coords[generic_dim_name]:
            self.logger.warning(f"No mapping found for model '{self.model_name}' and dimension '{generic_dim_name}'")
            return None
            
        coords = meta_coords[generic_dim_name][self.model_name]
        
        # Handle comma-separated list of possible dimension names
        if ',' in coords:
            coord_candidates = coords.split(',')
            
            # If available_dims is provided, check which candidate exists
            if available_dims:
                for coord in coord_candidates:
                    if coord in available_dims:
                        return coord
                
                # No matching dimension found
                self.logger.warning(f"None of the candidate dimensions {coord_candidates} found in available dimensions {available_dims}")
                return None
            
            # If no available_dims provided, return the first candidate
            return coord_candidates[0]
        
        # Handle special case for WRF, LIS, etc. with nested structure
        if isinstance(coords, dict):
            if 'dim' in coords:
                # For dimension names
                if available_dims:
                    # If multiple dimensions are specified (comma-separated), check each one
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
        
        # Simple case: direct mapping
        return coords
