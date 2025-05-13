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
    
    def __init__(self, model_name: str = None):
        """Initialize a new DataSource.
        
        Args:
            model_name: Name of the model this data source belongs to
        """
        self.model_name = model_name
        self.dataset = None
        self.metadata = {}
    
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
