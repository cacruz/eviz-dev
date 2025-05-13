"""
Factory for creating model-specific extensions.
"""

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.models.extensions.base import ModelExtension
from eviz.models.esm.ccm_extension import CCMExtension
from eviz.models.esm.generic_extension import GenericExtension


class ModelExtensionFactory:
    """Factory for creating model-specific extensions."""
    
    @staticmethod
    def create_extension(model_name: str, config_manager: ConfigManager) -> ModelExtension:
        """Create a model-specific extension.
        
        Args:
            model_name: The model name
            config_manager: The configuration manager
            
        Returns:
            A model-specific extension
        """
        # Map model names to extension classes
        extension_map = {
            'ccm': CCMExtension,
            'geos': CCMExtension,  # GEOS uses the same extension as CCM
            'generic': GenericExtension,
            'cf': GenericExtension,  # CF uses the same extension as Generic
            # Add more mappings as needed
        }
        
        # Get the extension class
        extension_class = extension_map.get(model_name.lower(), GenericExtension)
        
        # Create and return the extension
        return extension_class(config_manager)
