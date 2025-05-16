import pandas as pd
import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch
from datetime import datetime
import eviz.lib.data.utils as utils
import eviz.lib.const as constants

# Fixtures
@pytest.fixture
def mock_xarray_dataarray():
    # Create a simple DataArray with some test data
    data = np.random.rand(4, 3, 2)  # (time, lat, lon)
    coords = {
        'time': pd.date_range('2020-01-01', periods=4),
        'latitude': np.linspace(-45, 45, 3),
        'longitude': np.linspace(-90, 90, 2)
    }
    return xr.DataArray(data, coords=coords, dims=['time', 'latitude', 'longitude'])

@pytest.fixture
def mock_config():
    config = Mock()
    config.spec_data = {
        'test_var': {
            'units': 'kg/m2',
            'unitconversion': 2.0
        }
    }
    return config

# Test apply_conversion function
def test_apply_conversion_with_units_and_conversion(mock_config, mock_xarray_dataarray):
    result = utils.apply_conversion(mock_config, mock_xarray_dataarray, 'test_var')
    # Check if conversion factor was applied
    assert np.allclose(result.values, mock_xarray_dataarray.values * 2.0)

def test_apply_conversion_without_spec_data():
    config = Mock()
    config.spec_data = None
    data = xr.DataArray(np.array([1.0, 2.0]))
    result = utils.apply_conversion(config, data, 'test_var')
    # Should return original data unchanged
    assert np.array_equal(result.values, data.values)

# Test apply_mean function
def test_apply_mean_3d():
    data = np.random.rand(4, 3, 2)  # (time, lat, lon)
    coords = {
        'time': range(4),
        'lat': range(3),
        'lon': range(2)
    }
    da = xr.DataArray(data, coords=coords, dims=['time', 'lat', 'lon'])
    config = Mock()
    config.get_model_dim_name = Mock(return_value='time')
    
    result = utils.apply_mean(config, da)
    assert result.dims == ('lat', 'lon')
    assert np.array_equal(result.values, data.mean(axis=0))

# Test grid_cell_areas function
def test_grid_cell_areas():
    lon1d = np.array([0, 90, 180])
    lat1d = np.array([-45, 0, 45])
    result = utils.grid_cell_areas(lon1d, lat1d)
    
    # Check shape
    assert result.shape == (3, 3)
    # Check positive values
    assert np.all(result > 0)
    # Check symmetry around equator
    assert np.allclose(result[0], result[-1])

# Test calc_spatial_mean function
def test_calc_spatial_mean(mock_xarray_dataarray):
    result = utils.calc_spatial_mean(mock_xarray_dataarray)
    assert isinstance(result, xr.DataArray)
    assert 'latitude' not in result.dims
    assert 'longitude' not in result.dims

# Test datetime utilities
def test_get_timestamp_string():
    date_array = [2020, 1, 1, 12, 0, 0]
    result = utils.get_timestamp_string(date_array)
    assert result == "2020-01-01T12:00:00Z"

def test_get_timestamp_string_year_only():
    result = utils.get_timestamp_string(2020)
    assert result == "2020-01-01T00:00:00Z"

def test_add_months():
    start_date = np.datetime64('2020-01-01')
    result = utils.add_months(start_date, 3)
    assert result == np.datetime64('2020-04-01')

def test_is_full_year():
    start_date = np.datetime64('2020-01-01')
    end_date = np.datetime64('2021-01-01')
    assert utils.is_full_year(start_date, end_date)

# Test NetCDF reading functions
@patch('xarray.open_dataset')
def test_read_netcdf(mock_open_dataset):
    mock_dataset = Mock()
    mock_open_dataset.return_value = mock_dataset
    
    result = utils.read_netcdf('test.nc')
    mock_open_dataset.assert_called_once_with('test.nc')
    assert result == mock_dataset

@patch('xarray.open_mfdataset')
def test_read_multiple_netcdf(mock_open_mfdataset):
    mock_dataset = Mock()
    mock_open_mfdataset.return_value = mock_dataset
    
    file_paths = ['test1.nc', 'test2.nc']
    result = utils.read_multiple_netcdf(file_paths)
    mock_open_mfdataset.assert_called_once_with(file_paths, combine='by_coords')
    assert result == mock_dataset

# Test xarray dataset attribute functions
def test_get_dst_attribute():
    data = xr.DataArray([1, 2, 3])
    data.attrs['test_attr'] = 'test_value'
    
    result = utils.get_dst_attribute(data, 'test_attr')
    assert result == 'test_value'
    
    result = utils.get_dst_attribute(data, 'non_existent')
    assert result is None

# Test compute means functions
def test_compute_means():
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    data = np.random.rand(365)
    da = xr.DataArray(data, coords=[dates], dims=['time'])
    
    # Test monthly means
    monthly = utils.compute_means(da, '1M')
    assert len(monthly) == 12

def test_compute_mean_over_dim():
    data = np.random.rand(4, 3, 2)
    coords = {
        'time': range(4),
        'lat': range(3),
        'lon': range(2)
    }
    da = xr.DataArray(data, coords=coords, dims=['time', 'lat', 'lon'])
    
    result = utils.compute_mean_over_dim(da, 'time')
    assert 'time' not in result.dims
    assert np.array_equal(result.values, data.mean(axis=0))
