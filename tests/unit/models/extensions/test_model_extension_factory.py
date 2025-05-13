"""
Unit tests for the ModelExtensionFactory class.
"""

import pytest
from unittest.mock import MagicMock, patch

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.models.extensions.base import ModelExtension
from eviz.models.extensions.factory import ModelExtensionFactory
from eviz.models.esm.ccm_extension import CCMExtension
from eviz.models.esm.generic_extension import GenericExtension


class TestModelExtensionFactory:
    """Test cases for the ModelExtensionFactory class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_config_manager = MagicMock(spec=ConfigManager)
    
    def test_create_extension_ccm(self):
        """Test creating a CCM extension."""
        extension = ModelExtensionFactory.create_extension('ccm', self.mock_config_manager)
        
        assert isinstance(extension, CCMExtension)
        assert extension.config_manager == self.mock_config_manager
    
    def test_create_extension_geos(self):
        """Test creating a GEOS extension (which uses CCM extension)."""
        extension = ModelExtensionFactory.create_extension('geos', self.mock_config_manager)
        
        assert isinstance(extension, CCMExtension)
        assert extension.config_manager == self.mock_config_manager
    
    def test_create_extension_generic(self):
        """Test creating a Generic extension."""
        extension = ModelExtensionFactory.create_extension('generic', self.mock_config_manager)
        
        assert isinstance(extension, GenericExtension)
        assert extension.config_manager == self.mock_config_manager
    
    def test_create_extension_cf(self):
        """Test creating a CF extension (which uses Generic extension)."""
        extension = ModelExtensionFactory.create_extension('cf', self.mock_config_manager)
        
        assert isinstance(extension, GenericExtension)
        assert extension.config_manager == self.mock_config_manager
    
    def test_create_extension_unknown(self):
        """Test creating an extension for an unknown model (should use Generic extension)."""
        extension = ModelExtensionFactory.create_extension('unknown_model', self.mock_config_manager)
        
        assert isinstance(extension, GenericExtension)
        assert extension.config_manager == self.mock_config_manager
