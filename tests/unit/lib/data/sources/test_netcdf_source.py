"""
Unit tests for the NetCDFDataSource class.
"""

import os
import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.data.sources.netcdf_source import NetCDFDataSource


class TestNetCDFDataSource:
    """Test cases for the NetCDFDataSource class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.data_source = NetCDFDataSource('test_model')
    
    def test_init(self):
        """Test initialization of NetCDFDataSource."""
        assert self.data_source.model_name == 'test_model'
        assert self.data_source.dataset is None
        assert self.data_source.metadata == {}
    
    @patch('xarray.open_dataset')
    def test_load_data(self, mock_open_dataset):
        """Test loading data from a NetCDF file."""
        # Setup mock
        mock_dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90]),
                        'lon': np.array([0, 90, 180, 270])
                    }
                )
            }
        )
        mock_open_dataset.return_value = mock_dataset
        
        # Call the method
        result = self.data_source.load_data('test_file.nc')
        
        # Verify the result
        assert result is not None
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        
        # Verify the mock was called correctly
        mock_open_dataset.assert_called_once_with('test_file.nc', decode_cf=True)
    
    @patch('xarray.open_dataset')
    def test_load_data_with_error(self, mock_open_dataset):
        """Test loading data with an error."""
        # Setup mock
        mock_open_dataset.side_effect = Exception("Test error")
        
        # Call the method and verify it raises an exception
        with pytest.raises(Exception):
            self.data_source.load_data('test_file.nc')
        
        # Verify the mock was called correctly
        mock_open_dataset.assert_called_once_with('test_file.nc', decode_cf=True)
    
    def test_validate_data(self, temp_netcdf_file):
        """Test validating data."""
        # Load data
        self.data_source.load_data(temp_netcdf_file)
        
        # Call the method
        result = self.data_source.validate_data()
        
        # Verify the result
        assert result is True
    
    def test_validate_data_no_dataset(self):
        """Test validating data with no dataset."""
        # Call the method
        result = self.data_source.validate_data()
        
        # Verify the result
        assert result is False
    
    def test_get_field(self, temp_netcdf_file):
        """Test getting a field."""
        # Load data
        self.data_source.load_data(temp_netcdf_file)
        
        # Call the method
        result = self.data_source.get_field('temperature')
        
        # Verify the result
        assert result is not None
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('time', 'lat', 'lon')
    
    def test_get_field_not_found(self, temp_netcdf_file):
        """Test getting a field that doesn't exist."""
        # Load data
        self.data_source.load_data(temp_netcdf_file)
        
        # Call the method
        result = self.data_source.get_field('non_existent')
        
        # Verify the result
        assert result is None
    
    def test_get_metadata(self, temp_netcdf_file):
        """Test getting metadata."""
        # Load data
        self.data_source.load_data(temp_netcdf_file)
        
        # Call the method
        result = self.data_source.get_metadata()
        
        # Verify the result
        assert isinstance(result, dict)
    
    def test_get_dimensions(self, temp_netcdf_file):
        """Test getting dimensions."""
        # Load data
        self.data_source.load_data(temp_netcdf_file)
        
        # Call the method
        result = self.data_source.get_dimensions()
        
        # Verify the result
        assert 'time' in result
        assert 'lat' in result
        assert 'lon' in result
    
    def test_get_variables(self, temp_netcdf_file):
        """Test getting variables."""
        # Load data
        self.data_source.load_data(temp_netcdf_file)
        
        # Call the method
        result = self.data_source.get_variables()
        
        # Verify the result
        assert 'temperature' in result
        assert 'pressure' in result
    
    def test_close(self):
        """Test closing the data source."""
        # Setup mock
        self.data_source.dataset = MagicMock()
        self.data_source.dataset.close = MagicMock()
        
        # Call the method
        self.data_source.close()
        
        # Verify the mock was called correctly
        self.data_source.dataset.close.assert_called_once()
