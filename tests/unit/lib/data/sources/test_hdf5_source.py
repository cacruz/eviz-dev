"""
Unit tests for the HDF5DataSource class.
"""

import os
import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.data.sources.hdf5 import HDF5DataSource


class TestHDF5DataSource:
    """Test cases for the HDF5DataSource class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.data_source = HDF5DataSource('test_model')
    
    def test_init(self):
        """Test initialization of HDF5DataSource."""
        assert self.data_source.model_name == 'test_model'
        assert self.data_source.dataset is None
        assert self.data_source.metadata == {}
    
    @patch('h5py.File')
    @patch('xarray.Dataset.from_dict')
    def test_load_data(self, mock_from_dict, mock_h5py_file):
        """Test loading data from an HDF5 file."""
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file
        
        mock_file.keys.return_value = ['temperature', 'pressure']
        mock_file['temperature'].attrs = {'units': 'K'}
        mock_file['pressure'].attrs = {'units': 'hPa'}
        mock_file['temperature'].shape = (2, 3, 4)
        mock_file['pressure'].shape = (2, 3, 4)
        mock_file['temperature'].dtype = np.float32
        mock_file['pressure'].dtype = np.float32
        
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
                ),
                'pressure': xr.DataArray(
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
        mock_from_dict.return_value = mock_dataset
        
        result = self.data_source.load_data('test_file.h5')
        assert result is not None
        assert isinstance(result, xr.Dataset)

        mock_h5py_file.assert_called_once_with('test_file.h5', 'r')
    
    @patch('h5py.File')
    def test_load_data_with_error(self, mock_h5py_file):
        """Test loading data with an error."""
        mock_h5py_file.side_effect = Exception("Test error")
        
        with pytest.raises(Exception):
            self.data_source.load_data('test_file.h5')
        mock_h5py_file.assert_called_once_with('test_file.h5', 'r')
    
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
