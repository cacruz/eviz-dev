import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock
from eviz.models.esm.nuwrf import NuWrf

@pytest.fixture
def mock_config_manager():
    # This structure matches meta_coordinates.yaml
    meta_coords = {
        'xc': {
            'wrf': {'dim': 'west_east', 'coords': 'XLONG'},
            'lis': {'dim': 'lon', 'coords': 'lon'}
        },
        'yc': {
            'wrf': {'dim': 'south_north', 'coords': 'XLAT'},
            'lis': {'dim': 'lat', 'coords': 'lat'}
        },
        'zc': {
            'wrf': {'dim': 'bottom_top', 'coords': 'lev'},
            'lis': {'dim': 'lev', 'coords': 'lev'}
        },
        'tc': {
            'wrf': {'dim': 'Time', 'coords': 'Time'},
            'lis': {'dim': 'time', 'coords': 'time'}
        }
    }
    mock = MagicMock()
    mock.meta_coords = meta_coords
    mock.source_names = ['wrf', 'lis']
    mock.findex = 0
    mock.map_params = {}
    return mock

@pytest.fixture
def nuwrf(mock_config_manager):
    return NuWrf(config_manager=mock_config_manager)

def test_logger_property(mock_config_manager):
    nuwrf = NuWrf(config_manager=mock_config_manager)
    assert hasattr(nuwrf.logger, 'info')

def test_set_global_attrs_lis():
    attrs = {'DX': 1, 'DY': 2, 'other': 3}
    result = NuWrf.set_global_attrs('lis', attrs)
    assert result['DX'] == 1000.0
    assert result['DY'] == 2000.0
    assert result['other'] == 3

def test_set_global_attrs_wrf():
    attrs = {'DX': 1, 'DY': 2, 'other': 3}
    result = NuWrf.set_global_attrs('wrf', attrs)
    assert result['DX'] == 1
    assert result['DY'] == 2
    assert result['other'] == 3

def test_post_init(mock_config_manager):
    # Test post_init method
    nuwrf = NuWrf(config_manager=mock_config_manager)
    assert hasattr(nuwrf, 'p_top')
    assert nuwrf.p_top is None

def test_load_source_data(nuwrf):
    # Test _load_source_data method
    # First test when reader is None
    nuwrf._get_reader = MagicMock(return_value=None)
    data = nuwrf._load_source_data('wrf', 'test.nc')
    assert data is None
    
    # Now test with a mock reader that returns None
    mock_reader = MagicMock()
    mock_reader.read_data.return_value = None
    nuwrf._get_reader = MagicMock(return_value=mock_reader)
    data = nuwrf._load_source_data('wrf', 'test.nc')
    assert data is None
    
    # Now test with a mock reader that returns data
    mock_data = {'vars': {'temp': MagicMock()}, 'attrs': {'global_attr': 'value'}}
    mock_reader = MagicMock()
    mock_reader.read_data.return_value = mock_data
    nuwrf._get_reader = MagicMock(return_value=mock_reader)
    data = nuwrf._load_source_data('wrf', 'test.nc')
    assert data == mock_data


def test_get_field(nuwrf):
    # Test _get_field method
    # Create mock data
    mock_data = {'temperature': 'temp_data'}
    
    # Test with existing field
    field = nuwrf._get_field('temperature', mock_data)
    assert field == 'temp_data'
    
    # Test with non-existing field
    field = nuwrf._get_field('not_a_field', mock_data)
    assert field is None

def test_find_matching_dimension(nuwrf):
    # Test find_matching_dimension method
    # Create mock config_manager
    nuwrf.config_manager.meta_coords = {
        'xc': {'wrf': {'dim': 'west_east'}}
    }
    nuwrf.source_name = 'wrf'
    
    # Test with matching dimension
    field_dims = ('Time', 'bottom_top', 'south_north', 'west_east')
    dim = nuwrf.find_matching_dimension(field_dims, 'xc')
    assert dim == 'west_east'
    
    # Test with non-matching dimension
    field_dims = ('Time', 'bottom_top', 'south_north')
    dim = nuwrf.find_matching_dimension(field_dims, 'xc')
    assert dim is None

def test_set_global_attrs_with_none():
    # Test set_global_attrs with None attributes
    with pytest.raises(AttributeError):
        NuWrf.set_global_attrs('wrf', None)

def test_set_global_attrs_with_empty_dict():
    # Test set_global_attrs with empty dict
    result = NuWrf.set_global_attrs('wrf', {})
    assert result == {}

def test_set_global_attrs_with_unknown_source():
    # Test set_global_attrs with unknown source
    attrs = {'DX': 1, 'DY': 2}
    result = NuWrf.set_global_attrs('unknown', attrs)
    assert result == attrs  # Should return unchanged for unknown sources

@pytest.fixture
def mock_dataset():
    # Create a mock dataset for testing
    data = np.random.rand(2, 3, 4, 5)
    ds = xr.Dataset(
        data_vars={
            'temperature': xr.DataArray(
                data=data,
                dims=['Time', 'bottom_top', 'south_north', 'west_east'],
                coords={
                    'Time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                    'bottom_top': np.array([1, 2, 3]),
                    'south_north': np.array([10, 20, 30, 40]),
                    'west_east': np.array([100, 200, 300, 400, 500])
                }
            )
        }
    )
    return ds

def test_simple_plots(nuwrf):
    # Test _simple_plots method
    # Create mock plotter
    mock_plotter = MagicMock()
    
    # Create mock config_manager
    nuwrf.config_manager.map_params = {
        0: {
            'field': 'temperature',
            'source_name': 'wrf',
            'filename': 'test.nc',
            'to_plot': ['xy']
        }
    }
    nuwrf.config_manager.get_file_index = MagicMock(return_value=0)
    
    # Mock _load_source_data to return mock data
    mock_data = {'vars': {'temperature': MagicMock()}, 'attrs': {}}
    nuwrf._load_source_data = MagicMock(return_value=mock_data)
    
    # Mock set_global_attrs
    nuwrf.set_global_attrs = MagicMock(return_value={})
    
    # Mock _init_model_specific_data
    nuwrf._init_model_specific_data = MagicMock()
    
    # Mock _get_field_for_simple_plot
    nuwrf._get_field_for_simple_plot = MagicMock(return_value='field_data')
    
    # Call _simple_plots
    nuwrf._simple_plots(mock_plotter)
    
    # Check that plotter.simple_plot was called
    mock_plotter.simple_plot.assert_called_once_with(nuwrf.config_manager, 'field_data')
