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
        
        result = self.data_source.load_data('test_file.grib')
        assert result is not None
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        
        mock_open_dataset.assert_called_once_with('test_file.grib', engine='pynio')
    
    @patch('xarray.open_dataset')
    def test_load_data_with_error(self, mock_open_dataset):
        """Test loading data with an error."""
        mock_open_dataset.side_effect = Exception("Test error")
        
        with pytest.raises(Exception):
            self.data_source.load_data('test_file.grib')
        
        mock_open_dataset.assert_called_once_with('test_file.grib', engine='pynio')
    
    def test_validate_data(self):
        """Test validating data."""
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
        
        result = self.data_source.validate_data()
        assert result is True
    
    def test_validate_data_no_dataset(self):
        """Test validating data with no dataset."""
        result = self.data_source.validate_data()
        assert result is False
    
    def test_get_field(self):
        """Test getting a field."""
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
        
        result = self.data_source.get_field('temperature')
        assert result is not None
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('time', 'lat', 'lon')
    
    def test_get_field_not_found(self):
        """Test getting a field that doesn't exist."""
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
        
        result = self.data_source.get_field('non_existent')
        assert result is None
    
    def test_get_metadata(self):
        """Test getting metadata."""
        self.data_source.metadata = {'key': 'value'}
        result = self.data_source.get_metadata()
        assert result == {'key': 'value'}
    
    def test_close(self):
        """Test closing the data source."""
        self.data_source.dataset = MagicMock()
        self.data_source.dataset.close = MagicMock()
        self.data_source.close()
        self.data_source.dataset.close.assert_called_once()
    
    @patch('xarray.open_dataset')
    def test_extract_metadata(self, mock_open_dataset):
        """Test extracting metadata from a GRIB file."""
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
        
        self.data_source.load_data('test_file.grib')
        assert self.data_source.metadata is not None
        assert 'global_attrs' in self.data_source.metadata
        assert 'Conventions' in self.data_source.metadata['global_attrs']
        assert self.data_source.metadata['global_attrs']['Conventions'] == 'CF-1.6'
