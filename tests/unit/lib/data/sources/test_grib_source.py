"""
Unit tests for the GRIBDataSource class.
"""

import os
import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.data.sources.grib_source import GRIBDataSource


class TestGRIBDataSource:
    """Test cases for the GRIBDataSource class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.data_source = GRIBDataSource('test_model')
    
    def test_init(self):
        """Test initialization of GRIBDataSource."""
        assert self.data_source.model_name == 'test_model'
        assert self.data_source.dataset is None
        assert self.data_source.metadata == {}
    
    @patch('xarray.open_dataset')
    def test_load_data(self, mock_open_dataset):
        """Test loading data from a GRIB file."""
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
        result = self.data_source.load_data('test_file.grib')
        
        # Verify the result
        assert result is not None
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        
        # Verify the mock was called correctly
        mock_open_dataset.assert_called_once_with('test_file.grib', engine='pynio')
    
    @patch('xarray.open_dataset')
    def test_load_data_with_error(self, mock_open_dataset):
        """Test loading data with an error."""
        # Setup mock
        mock_open_dataset.side_effect = Exception("Test error")
        
        # Call the method and verify it raises an exception
        with pytest.raises(Exception):
            self.data_source.load_data('test_file.grib')
        
        # Verify the mock was called correctly
        mock_open_dataset.assert_called_once_with('test_file.grib', engine='pynio')
    
    def test_validate_data(self):
        """Test validating data."""
        # Setup dataset
        self.data_source.dataset = xr.Dataset(
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
    
    def test_get_field(self):
        """Test getting a field."""
        # Setup dataset
        self.data_source.dataset = xr.Dataset(
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
        
        # Call the method
        result = self.data_source.get_field('temperature')
        
        # Verify the result
        assert result is not None
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('time', 'lat', 'lon')
    
    def test_get_field_not_found(self):
        """Test getting a field that doesn't exist."""
        # Setup dataset
        self.data_source.dataset = xr.Dataset(
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
        
        # Call the method
        result = self.data_source.get_field('non_existent')
        
        # Verify the result
        assert result is None
    
    def test_get_metadata(self):
        """Test getting metadata."""
        # Setup metadata
        self.data_source.metadata = {'key': 'value'}
        
        # Call the method
        result = self.data_source.get_metadata()
        
        # Verify the result
        assert result == {'key': 'value'}
    
    def test_close(self):
        """Test closing the data source."""
        # Setup mock
        self.data_source.dataset = MagicMock()
        self.data_source.dataset.close = MagicMock()
        
        # Call the method
        self.data_source.close()
        
        # Verify the mock was called correctly
        self.data_source.dataset.close.assert_called_once()
    
    @patch('xarray.open_dataset')
    def test_extract_metadata(self, mock_open_dataset):
        """Test extracting metadata from a GRIB file."""
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
                    },
                    attrs={
                        'long_name': 'Temperature',
                        'units': 'K'
                    }
                )
            },
            attrs={
                'Conventions': 'CF-1.6',
                'history': 'Created by GRIB to NetCDF converter'
            }
        )
        mock_open_dataset.return_value = mock_dataset
        
        # Call the method
        self.data_source.load_data('test_file.grib')
        
        # Verify the result
        assert self.data_source.metadata is not None
        assert 'global_attrs' in self.data_source.metadata
        assert 'Conventions' in self.data_source.metadata['global_attrs']
        assert self.data_source.metadata['global_attrs']['Conventions'] == 'CF-1.6'
