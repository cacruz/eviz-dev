"""
Unit tests for the DataSourceFactory class.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from eviz.lib.data.factory.source_factory import DataSourceFactory
from eviz.lib.data.sources import (
    DataSource,
    NetCDFDataSource,
    HDF5DataSource,
    CSVDataSource,
    GRIBDataSource
)


class TestDataSourceFactory:
    """Test cases for the DataSourceFactory class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.factory = DataSourceFactory()
    
    def test_init(self):
        """Test initialization of DataSourceFactory."""
        assert self.factory is not None
        assert self.factory.registry is not None
    
    def test_register_default_data_sources(self):
        """Test that default data sources are registered."""
        # Get the supported extensions
        extensions = self.factory.get_supported_extensions()
        
        # Verify that the default extensions are supported
        assert 'nc' in extensions
        assert 'h5' in extensions
        assert 'csv' in extensions
        assert 'grib' in extensions
    
    def test_register_data_source(self):
        """Test registering a custom data source."""
        # Create a mock data source class
        mock_data_source_class = MagicMock(spec=DataSource)
        
        # Register the mock data source
        self.factory.register_data_source(['test'], mock_data_source_class)
        
        # Verify that the extension is supported
        extensions = self.factory.get_supported_extensions()
        assert 'test' in extensions
    
    @patch('os.path.exists')
    def test_create_data_source_nc(self, mock_exists):
        """Test creating a NetCDF data source."""
        # Setup mock
        mock_exists.return_value = True
        
        # Call the method
        data_source = self.factory.create_data_source('test.nc', 'test_model')
        
        # Verify the result
        assert isinstance(data_source, NetCDFDataSource)
        assert data_source.model_name == 'test_model'
    
    @patch('os.path.exists')
    def test_create_data_source_h5(self, mock_exists):
        """Test creating an HDF5 data source."""
        # Setup mock
        mock_exists.return_value = True
        
        # Call the method
        data_source = self.factory.create_data_source('test.h5', 'test_model')
        
        # Verify the result
        assert isinstance(data_source, HDF5DataSource)
        assert data_source.model_name == 'test_model'
    
    @patch('os.path.exists')
    def test_create_data_source_csv(self, mock_exists):
        """Test creating a CSV data source."""
        # Setup mock
        mock_exists.return_value = True
        
        # Call the method
        data_source = self.factory.create_data_source('test.csv', 'test_model')
        
        # Verify the result
        assert isinstance(data_source, CSVDataSource)
        assert data_source.model_name == 'test_model'
    
    @patch('os.path.exists')
    def test_create_data_source_grib(self, mock_exists):
        """Test creating a GRIB data source."""
        # Setup mock
        mock_exists.return_value = True
        
        # Call the method
        data_source = self.factory.create_data_source('test.grib', 'test_model')
        
        # Verify the result
        assert isinstance(data_source, GRIBDataSource)
        assert data_source.model_name == 'test_model'
    
    @patch('os.path.exists')
    def test_create_data_source_no_extension(self, mock_exists):
        """Test creating a data source with no extension."""
        # Setup mock
        mock_exists.return_value = True
        
        # Call the method with a file name containing 'nc'
        data_source = self.factory.create_data_source('test_nc_file', 'test_model')
        
        # Verify the result
        assert isinstance(data_source, NetCDFDataSource)
        assert data_source.model_name == 'test_model'
    
    @patch('os.path.exists')
    def test_create_data_source_unsupported_extension(self, mock_exists):
        """Test creating a data source with an unsupported extension."""
        # Setup mock
        mock_exists.return_value = True
        
        # Call the method and verify it raises an exception
        with pytest.raises(ValueError):
            self.factory.create_data_source('test.unsupported', 'test_model')
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        # Call the method
        extensions = self.factory.get_supported_extensions()
        
        # Verify the result
        assert isinstance(extensions, list)
        assert len(extensions) > 0
        assert 'nc' in extensions
        assert 'h5' in extensions
        assert 'csv' in extensions
        assert 'grib' in extensions
    
    def test_is_supported(self):
        """Test checking if a file extension is supported."""
        # Call the method
        assert self.factory.is_supported('test.nc') is True
        assert self.factory.is_supported('test.h5') is True
        assert self.factory.is_supported('test.csv') is True
        assert self.factory.is_supported('test.grib') is True
        assert self.factory.is_supported('test.unsupported') is False
