"""
Base DataSource class that defines the interface for all data sources.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional
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
            
        # Use the model_name (source_name) directly, not exp_name or exp_id
        model_name = self.model_name
        
        # Debug: Print model name and meta_coords for this dimension
        self.logger.debug(f"Looking for model '{model_name}' in meta_coords['{generic_dim_name}']")
        self.logger.debug(f"Available models for {generic_dim_name}: {list(meta_coords[generic_dim_name].keys())}")
        
        # Check if model name is in meta_coords
        if not model_name or model_name not in meta_coords[generic_dim_name]:
            # Try to use a default model if available
            if 'generic' in meta_coords[generic_dim_name]:
                # Only log at debug level to avoid unnecessary warnings
                self.logger.debug(f"Using 'generic' mapping for model '{model_name}' and dimension '{generic_dim_name}'")
                model_name = 'generic'
            else:
                self.logger.warning(f"No mapping found for model '{model_name}' and dimension '{generic_dim_name}'")
                return None
        
        # Get the coordinate mapping for this model
        coords = meta_coords[generic_dim_name][model_name]
            
            # Handle different types of coordinate specifications
        if isinstance(coords, list):
            # If coords is a list, check each entry against available dimensions
            for coord in coords:
                if available_dims and coord in available_dims:
                    return coord
            # If no match found, return the first entry
            return coords[0] if coords else None
        elif isinstance(coords, dict):
            # If coords is a dictionary, handle special cases
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
            
            # If neither 'dim' nor 'coords' is present, return None
            return None
        elif isinstance(coords, str):
            # If coords is a string, handle comma-separated list of possible dimension names
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
            
            # Simple case: direct mapping
            return coords
        
        # If we get here, coords is of an unexpected type
        self.logger.warning(f"Unexpected type for coords: {type(coords)}")
        return None
