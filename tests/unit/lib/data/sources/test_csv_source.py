"""
Unit tests for the CSVDataSource class.
"""

import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.data.sources.csv_source import CSVDataSource


class TestCSVDataSource:
    """Test cases for the CSVDataSource class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.data_source = CSVDataSource('test_model')
    
    def test_init(self):
        """Test initialization of CSVDataSource."""
        assert self.data_source.model_name == 'test_model'
        assert self.data_source.dataset is None
        assert self.data_source.metadata == {}
    
    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        """Test loading data from a CSV file."""
        # Setup mock
        mock_df = pd.DataFrame({
            'time': ['2022-01-01', '2022-01-02'],
            'lat': [0, 45],
            'lon': [0, 90],
            'temperature': [25.0, 30.0],
            'pressure': [1000.0, 1010.0]
        })
        mock_read_csv.return_value = mock_df
        
        # Call the method
        result = self.data_source.load_data('test_file.csv')
        
        # Verify the result
        assert result is not None
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert 'pressure' in result.data_vars
        
        # Verify the mock was called correctly
        mock_read_csv.assert_called_once_with('test_file.csv')
    
    @patch('pandas.read_csv')
    def test_load_data_with_error(self, mock_read_csv):
        """Test loading data with an error."""
        # Setup mock
        mock_read_csv.side_effect = Exception("Test error")
        
        # Call the method and verify it raises an exception
        with pytest.raises(Exception):
            self.data_source.load_data('test_file.csv')
        
        # Verify the mock was called correctly
        mock_read_csv.assert_called_once_with('test_file.csv')
    
    def test_validate_data(self):
        """Test validating data."""
        # Setup dataset
        self.data_source.dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 2),
                    dims=['time', 'station'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'station': np.array([1, 2])
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
                    data=np.random.rand(2, 2),
                    dims=['time', 'station'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'station': np.array([1, 2])
                    }
                )
            }
        )
        
        # Call the method
        result = self.data_source.get_field('temperature')
        
        # Verify the result
        assert result is not None
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('time', 'station')
    
    def test_get_field_not_found(self):
        """Test getting a field that doesn't exist."""
        # Setup dataset
        self.data_source.dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 2),
                    dims=['time', 'station'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'station': np.array([1, 2])
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
    
    @patch('pandas.read_csv')
    def test_convert_dataframe_to_dataset(self, mock_read_csv):
        """Test converting a DataFrame to a Dataset."""
        # Setup mock
        mock_df = pd.DataFrame({
            'time': ['2022-01-01', '2022-01-02'],
            'lat': [0, 45],
            'lon': [0, 90],
            'temperature': [25.0, 30.0],
            'pressure': [1000.0, 1010.0]
        })
        mock_read_csv.return_value = mock_df
        
        # Call the method
        self.data_source.load_data('test_file.csv')
        
        # Verify the result
        assert self.data_source.dataset is not None
        assert isinstance(self.data_source.dataset, xr.Dataset)
        assert 'temperature' in self.data_source.dataset.data_vars
        assert 'pressure' in self.data_source.dataset.data_vars
        
        # Verify the dimensions
        assert 'time' in self.data_source.dataset.dims
        assert self.data_source.dataset.dims['time'] == 2
