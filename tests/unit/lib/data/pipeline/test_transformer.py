"""
Unit tests for the DataTransformer class.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.data.pipeline.transformer import DataTransformer
from eviz.lib.data.sources import DataSource


class TestDataTransformer:
    """Test cases for the DataTransformer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.transformer = DataTransformer()
        
        # Create a mock data source
        self.mock_data_source = MagicMock(spec=DataSource)
        
        # Create a test dataset
        self.test_dataset = xr.Dataset(
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
        
        self.mock_data_source.dataset = self.test_dataset
        self.mock_data_source.validate_data.return_value = True
    
    def test_init(self):
        """Test initialization of DataTransformer."""
        assert self.transformer is not None
    
    def test_transform_data_source(self):
        """Test transforming a data source."""
        # Call the method
        result = self.transformer.transform_data_source(self.mock_data_source)
        
        # Verify the result
        assert result == self.mock_data_source
        
        # Verify the mock was called correctly
        self.mock_data_source.validate_data.assert_called_once()
    
    def test_transform_data_source_validation_failed(self):
        """Test transforming a data source with validation failure."""
        # Setup mock
        self.mock_data_source.validate_data.return_value = False
        
        # Call the method
        result = self.transformer.transform_data_source(self.mock_data_source)
        
        # Verify the result
        assert result == self.mock_data_source
        
        # Verify the mock was called correctly
        self.mock_data_source.validate_data.assert_called_once()
    
    def test_transform_dataset_with_regrid(self):
        """Test transforming a dataset with regridding."""
        # Setup parameters
        kwargs = {
            'regrid': True,
            'target_grid': {
                'lat_min': -90,
                'lat_max': 90,
                'lon_min': -180,
                'lon_max': 180,
                'lat_res': 45,
                'lon_res': 90
            }
        }
        
        # Call the method
        result = self.transformer._transform_dataset(self.test_dataset, **kwargs)
        
        # Verify the result
        assert 'lat' in result.coords
        assert 'lon' in result.coords
        assert len(result.coords['lat']) == 5  # (-90, -45, 0, 45, 90)
        assert len(result.coords['lon']) == 5  # (-180, -90, 0, 90, 180)
    
    def test_transform_dataset_with_subset(self):
        """Test transforming a dataset with subsetting."""
        # Setup parameters
        kwargs = {
            'subset': True,
            'lat_range': (0, 45),
            'lon_range': (0, 180)
        }
        
        # Call the method
        result = self.transformer._transform_dataset(self.test_dataset, **kwargs)
        
        # Verify the result
        assert len(result.coords['lat']) == 2  # (0, 45)
        assert len(result.coords['lon']) == 3  # (0, 90, 180)
    
    def test_transform_dataset_with_time_average(self):
        """Test transforming a dataset with time averaging."""
        # Setup parameters
        kwargs = {
            'time_average': True
        }
        
        # Call the method
        result = self.transformer._transform_dataset(self.test_dataset, **kwargs)
        
        # Verify the result
        assert 'time' not in result.dims
        assert 'lat' in result.dims
        assert 'lon' in result.dims
    
    def test_transform_dataset_with_vertical_average(self):
        """Test transforming a dataset with vertical averaging."""
        # Create a dataset with a vertical dimension
        dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4, 5),
                    dims=['time', 'lev', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lev': np.array([1000, 850, 500]),
                        'lat': np.array([0, 45, 90, -45]),
                        'lon': np.array([0, 90, 180, 270, 360])
                    }
                )
            }
        )
        
        # Setup parameters
        kwargs = {
            'vertical_average': True
        }
        
        # Call the method
        result = self.transformer._transform_dataset(dataset, **kwargs)
        
        # Verify the result
        assert 'lev' not in result.dims
        assert 'time' in result.dims
        assert 'lat' in result.dims
        assert 'lon' in result.dims
    
    def test_transform_dataset_with_vertical_sum(self):
        """Test transforming a dataset with vertical summing."""
        # Create a dataset with a vertical dimension
        dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4, 5),
                    dims=['time', 'lev', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lev': np.array([1000, 850, 500]),
                        'lat': np.array([0, 45, 90, -45]),
                        'lon': np.array([0, 90, 180, 270, 360])
                    }
                )
            }
        )
        
        # Setup parameters
        kwargs = {
            'vertical_sum': True
        }
        
        # Call the method
        result = self.transformer._transform_dataset(dataset, **kwargs)
        
        # Verify the result
        assert 'lev' not in result.dims
        assert 'time' in result.dims
        assert 'lat' in result.dims
        assert 'lon' in result.dims
    
    @patch('scipy.interpolate.griddata')
    def test_regrid_2d(self, mock_griddata):
        """Test regridding a 2D array."""
        # Setup mock
        mock_griddata.return_value = np.ones((2, 2))
        
        # Setup test data
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        orig_lat_grid = np.array([[0, 0], [1, 1]])
        orig_lon_grid = np.array([[0, 1], [0, 1]])
        new_lat_grid = np.array([[0, 0], [1, 1]])
        new_lon_grid = np.array([[0, 1], [0, 1]])
        
        # Call the method
        result = self.transformer._regrid_2d(data, orig_lat_grid, orig_lon_grid, new_lat_grid, new_lon_grid)
        
        # Verify the result
        assert result.shape == (2, 2)
        assert np.all(result == 1.0)
        
        # Verify the mock was called correctly
        mock_griddata.assert_called()
    
    def test_subset_dataset(self):
        """Test subsetting a dataset."""
        # Setup parameters
        kwargs = {
            'lat_range': (0, 45),
            'lon_range': (0, 180),
            'time_range': ('2022-01-01', '2022-01-01')
        }
        
        # Call the method
        result = self.transformer._subset_dataset(self.test_dataset, **kwargs)
        
        # Verify the result
        assert len(result.coords['lat']) == 2  # (0, 45)
        assert len(result.coords['lon']) == 3  # (0, 90, 180)
        assert len(result.coords['time']) == 1  # (2022-01-01)
