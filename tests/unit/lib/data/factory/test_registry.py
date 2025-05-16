import pytest
from unittest.mock import MagicMock
from eviz.lib.data.factory.registry import DataSourceRegistry
from eviz.lib.data.sources import DataSource


class TestDataSourceRegistry:
    """Test cases for the DataSourceRegistry class."""
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.registry = DataSourceRegistry()
        self.mock_data_source_class1 = MagicMock(spec=DataSource)
        self.mock_data_source_class2 = MagicMock(spec=DataSource)
    
    def test_init(self):
        """Test initialization of DataSourceRegistry."""
        assert self.registry is not None
        assert self.registry._registry == {}
    
    def test_register(self):
        """Test registering a data source class."""
        self.registry.register(['ext1', 'ext2'], self.mock_data_source_class1)
        assert 'ext1' in self.registry._registry
        assert 'ext2' in self.registry._registry
        assert self.registry._registry['ext1'] == self.mock_data_source_class1
        assert self.registry._registry['ext2'] == self.mock_data_source_class1
    
    def test_register_with_dot(self):
        """Test registering a data source class with extensions that have dots."""
        self.registry.register(['.ext1', '.ext2'], self.mock_data_source_class1)
        assert 'ext1' in self.registry._registry
        assert 'ext2' in self.registry._registry
        assert self.registry._registry['ext1'] == self.mock_data_source_class1
        assert self.registry._registry['ext2'] == self.mock_data_source_class1
    
    def test_register_with_uppercase(self):
        """Test registering a data source class with uppercase extensions."""
        self.registry.register(['EXT1', 'EXT2'], self.mock_data_source_class1)
        assert 'ext1' in self.registry._registry
        assert 'ext2' in self.registry._registry
        assert self.registry._registry['ext1'] == self.mock_data_source_class1
        assert self.registry._registry['ext2'] == self.mock_data_source_class1
    
    def test_get_data_source_class(self):
        """Test getting a data source class."""
        self.registry.register(['ext1', 'ext2'], self.mock_data_source_class1)
        result = self.registry.get_data_source_class('ext1')
        assert result == self.mock_data_source_class1
    
    def test_get_data_source_class_with_dot(self):
        """Test getting a data source class with an extension that has a dot."""
        self.registry.register(['ext1', 'ext2'], self.mock_data_source_class1)
        result = self.registry.get_data_source_class('.ext1')
        assert result == self.mock_data_source_class1
    
    def test_get_data_source_class_with_uppercase(self):
        """Test getting a data source class with an uppercase extension."""
        self.registry.register(['ext1', 'ext2'], self.mock_data_source_class1)
        result = self.registry.get_data_source_class('EXT1')
        assert result == self.mock_data_source_class1
    
    def test_get_data_source_class_not_found(self):
        """Test getting a data source class that doesn't exist."""
        with pytest.raises(ValueError):
            self.registry.get_data_source_class('ext1')
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        self.registry.register(['ext1', 'ext2'], self.mock_data_source_class1)
        self.registry.register(['ext3'], self.mock_data_source_class2)
        result = self.registry.get_supported_extensions()
        assert isinstance(result, set)
        assert len(result) == 3
        assert 'ext1' in result
        assert 'ext2' in result
        assert 'ext3' in result
    
    def test_is_supported(self):
        """Test checking if an extension is supported."""
        self.registry.register(['ext1', 'ext2'], self.mock_data_source_class1)
        assert self.registry.is_supported('ext1') is True
        assert self.registry.is_supported('ext2') is True
        assert self.registry.is_supported('ext3') is False
    
    def test_is_supported_with_dot(self):
        """Test checking if an extension with a dot is supported."""
        self.registry.register(['ext1', 'ext2'], self.mock_data_source_class1)
        assert self.registry.is_supported('.ext1') is True
        assert self.registry.is_supported('.ext2') is True
        assert self.registry.is_supported('.ext3') is False
    
    def test_is_supported_with_uppercase(self):
        """Test checking if an uppercase extension is supported."""
        self.registry.register(['ext1', 'ext2'], self.mock_data_source_class1)
        assert self.registry.is_supported('EXT1') is True
        assert self.registry.is_supported('EXT2') is True
        assert self.registry.is_supported('EXT3') is False
