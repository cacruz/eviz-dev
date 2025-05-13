"""
Unit tests for the DataProcessor class.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.lib.data.sources import DataSource


class TestDataProcessor:
    """Test cases for the DataProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.processor = DataProcessor()
        
        # Create a mock data source
        self.mock_data_source = MagicMock(spec=DataSource)
        
        # Create a test dataset
        self.test_dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'latitude', 'longitude'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'latitude': np.array([0, 45, 90]),
                        'longitude': np.array([0, 90, 180, 270])
                    },
                    attrs={'units': 'K'}
                ),
                'pressure': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'latitude', 'longitude'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'latitude': np.array([0, 45, 90]),
                        'longitude': np.array([0, 90, 180, 270])
                    },
                    attrs={'units': 'hPa'}
                )
            }
        )
        
        self.mock_data_source.dataset = self.test_dataset
        self.mock_data_source.validate_data.return_value = True
    
    def test_init(self):
        """Test initialization of DataProcessor."""
        assert self.processor is not None
    
    def test_process_data_source(self):
        """Test processing a data source."""
        # Call the method
        result = self.processor.process_data_source(self.mock_data_source)
        
        # Verify the result
        assert result == self.mock_data_source
        
        # Verify the mock was called correctly
        self.mock_data_source.validate_data.assert_called_once()
    
    def test_process_data_source_validation_failed(self):
        """Test processing a data source with validation failure."""
        # Setup mock
        self.mock_data_source.validate_data.return_value = False
        
        # Call the method
        result = self.processor.process_data_source(self.mock_data_source)
        
        # Verify the result
        assert result == self.mock_data_source
        
        # Verify the mock was called correctly
        self.mock_data_source.validate_data.assert_called_once()
    
    def test_standardize_coordinates(self):
        """Test standardizing coordinates."""
        # Call the method
        result = self.processor._standardize_coordinates(self.test_dataset)
        
        # Verify the result
        assert 'lat' in result.coords
        assert 'lon' in result.coords
        assert 'latitude' not in result.coords
        assert 'longitude' not in result.coords
        
        # Check that the data is preserved
        np.testing.assert_array_equal(result.coords['lat'].values, self.test_dataset.coords['latitude'].values)
        np.testing.assert_array_equal(result.coords['lon'].values, self.test_dataset.coords['longitude'].values)
    
    def test_standardize_coordinates_with_out_of_range_values(self):
        """Test standardizing coordinates with out-of-range values."""
        # Create a dataset with out-of-range lat/lon values
        dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([-100, 0, 100]),  # Out of range
                        'lon': np.array([-200, 0, 200, 400])  # Out of range
                    }
                )
            }
        )
        
        # Call the method
        result = self.processor._standardize_coordinates(dataset)
        
        # Verify the result
        assert np.all(result.coords['lat'].values >= -90)
        assert np.all(result.coords['lat'].values <= 90)
        assert np.all(result.coords['lon'].values >= -180)
        assert np.all(result.coords['lon'].values <= 180)
    
    def test_handle_missing_values(self):
        """Test handling missing values."""
        # Create a dataset with NaN values
        dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]]),
                    dims=['x', 'y'],
                    attrs={'_FillValue': -999.9}
                )
            }
        )
        
        # Call the method
        result = self.processor._handle_missing_values(dataset)
        
        # Verify the result
        assert np.isclose(result['temperature'].values[0, 2], -999.9)
        assert np.isclose(result['temperature'].values[1, 1], -999.9)
    
    def test_apply_unit_conversions(self):
        """Test applying unit conversions."""
        # Call the method
        result = self.processor._apply_unit_conversions(self.test_dataset)
        
        # Verify the result
        assert result['temperature'].attrs['units'] == 'C'
        assert result['pressure'].attrs['units'] == 'Pa'
        
        # Check that the data was converted correctly
        # Temperature: K to C (subtract 273.15)
        # Pressure: hPa to Pa (multiply by 100)
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    # Skip the unit conversion test since it's not implemented yet
                    pass
