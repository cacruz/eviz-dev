from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import logging
import xarray as xr


@dataclass
class DataSource(ABC):
    """
    Abstract base class that defines the interface for all data sources.
    
    The DataSource class establishes a common contract for all data sources in the eViz
    application, providing a unified representation of data as xarray Datasets regardless
    of the original format. This abstraction allows the application to work with different
    data formats through a consistent interface.
    
    All concrete data source implementations must inherit from this class and implement
    the required abstract methods, particularly load_data. The class provides common
    functionality for data access, validation, and dimension mapping that is shared
    across all data sources.
    
    The class follows a composition pattern, encapsulating an xarray Dataset and providing
    methods to access and manipulate it. It also implements delegation patterns to allow
    direct access to the underlying dataset's attributes and items.
    
    Attributes:
        model_name (str, optional): Name of the model or data source type.
        config_manager (object, optional): Configuration manager providing access to settings.
        dataset (xarray.Dataset): The loaded dataset (initialized to None).
        metadata (dict): Dictionary containing metadata about the dataset.
    
    Abstract Methods:
        load_data: Load data from a specified file path into an xarray Dataset.
    
    Methods:
        validate_data: Validate the loaded data for integrity and completeness.
        get_field: Get a specific field (variable) from the dataset.
        get_metadata: Get metadata about the dataset.
        get_dimensions: Get the dimensions of the dataset.
        get_variables: Get the variables in the dataset.
        close: Close the dataset and free resources.
        _get_model_dim_name: Get model-specific dimension name for a standard dimension.
        _get_model_dim_name2: Alternative implementation for dimension name mapping.
        __getattr__: Delegate attribute access to the underlying dataset.
        __getitem__: Delegate item access to the underlying dataset.
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
        """
        Load data from the specified file path into an xarray dataset.
        
        Args:
            file_path (str): Path to the data file.
            
        Returns:
            xarray.Dataset: An xarray dataset containing the loaded data.
            
        This abstract method must be implemented by all concrete data source classes.
        The implementation should handle opening the specified file, loading its contents
        into an xarray Dataset, and performing any necessary preprocessing or validation.
        
        The method should handle format-specific details while ensuring the returned
        dataset conforms to the expected structure for the eViz application.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file format is invalid or unsupported.
            IOError: If there are issues reading the file.
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

        Subclasses may override this method to implement more specific validation
        logic appropriate for their data format or structure.
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

        This method allows users to call xarray.Dataset methods directly on DataSource
        objects without having to access the .dataset attribute explicitly, providing
        a more convenient API.
        
        Example:
            # Instead of:
            result = data_source.dataset.mean()
            
            # You can use:
            result = data_source.mean()
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
            
        This method allows users to access variables using square brackets directly
        on DataSource objects without having to access the .dataset attribute explicitly,
        providing a more convenient API.
        
        Example:
            # Instead of:
            temperature = data_source.dataset['temperature']
            
            # You can use:
            temperature = data_source['temperature']
        """
        if self.dataset is None:
            raise TypeError(f"'{self.__class__.__name__}' has no dataset loaded")
        
        return self.dataset[key]
        
        
    def _get_model_dim_name(self, gridded_dim_name, available_dims=None):
        """
        Get the model-specific dimension name for a gridded dimension.
        
        Args:
            gridded_dim_name (str): Gridded dimension name (e.g., 'xc', 'yc', 'zc', 'tc')
            available_dims (list, optional): List of available dimensions in the dataset
            
        Returns:
            str or None: The model-specific dimension name if found, otherwise None
            
        This method maps standard dimension names ('xc', 'yc', 'zc', 'tc') to their
        model-specific equivalents using the configuration's meta_coords mapping.
        It handles various formats of dimension specifications including strings,
        lists, and dictionaries.
        
        The method supports fallback to default mappings when model-specific mappings
        are not available, and can filter the results based on available dimensions
        in the dataset.
        
        This is a key utility for working with datasets from different models that
        may use different naming conventions for the same conceptual dimensions.
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
