import pytest
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

def test_get_model_dim_name(nuwrf):
    # Should return the correct dim for wrf and lis
    assert nuwrf.get_model_dim_name('wrf', 'xc') == 'west_east'
    assert nuwrf.get_model_dim_name('lis', 'xc') == 'lon'
    assert nuwrf.get_model_dim_name('wrf', 'zc') == 'bottom_top'
    assert nuwrf.get_model_dim_name('lis', 'zc') == 'lev'
    # Nonexistent
    assert nuwrf.get_model_dim_name('wrf', 'not_a_dim') is None

def test_get_model_coord_name(nuwrf):
    # Should return the correct coords for wrf and lis
    assert nuwrf.get_model_coord_name('wrf', 'xc') == 'XLONG'
    assert nuwrf.get_model_coord_name('lis', 'xc') == 'lon'
    assert nuwrf.get_model_coord_name('wrf', 'zc') == 'lev'
    assert nuwrf.get_model_coord_name('lis', 'zc') == 'lev'
    # Nonexistent
    assert nuwrf.get_model_coord_name('wrf', 'not_a_dim') is None

def test_get_field_dim_name(nuwrf):
    # Simulate a WRF source_data with dims
    wrf_source_data = MagicMock()
    wrf_source_data.dims = ['west_east', 'south_north', 'bottom_top', 'Time']
    # Should find the correct dim for WRF
    assert nuwrf.get_field_dim_name('wrf', wrf_source_data, 'xc') == 'west_east'
    assert nuwrf.get_field_dim_name('wrf', wrf_source_data, 'yc') == 'south_north'
    assert nuwrf.get_field_dim_name('wrf', wrf_source_data, 'zc') == 'bottom_top'
    assert nuwrf.get_field_dim_name('wrf', wrf_source_data, 'tc') == 'Time'
    # Should return None for missing
    assert nuwrf.get_field_dim_name('wrf', wrf_source_data, 'not_a_dim') is None

    # Simulate a LIS source_data with dims
    lis_source_data = MagicMock()
    lis_source_data.dims = ['lon', 'lat', 'lev', 'time']
    # Should find the correct dim for LIS
    assert nuwrf.get_field_dim_name('lis', lis_source_data, 'xc') == 'lon'
    assert nuwrf.get_field_dim_name('lis', lis_source_data, 'yc') == 'lat'
    assert nuwrf.get_field_dim_name('lis', lis_source_data, 'zc') == 'lev'
    assert nuwrf.get_field_dim_name('lis', lis_source_data, 'tc') == 'time'
    # Should return None for missing
    assert nuwrf.get_field_dim_name('lis', lis_source_data, 'not_a_dim') is None

def test_coord_names_wrf(nuwrf):
    # Simulate a WRF field with stagger and coords
    field = MagicMock()
    field.dims = ['Time', 'bottom_top', 'south_north', 'west_east']
    field.coords = {'XLONG': MagicMock(), 'XLAT': MagicMock()}
    field.stagger = ""
    source_data = {'vars': {'T': field}}
    # Should return correct dim1, dim2 for 'xy' plot
    dim1, dim2 = nuwrf.coord_names('wrf', source_data, 'T', 'xy')
    assert dim1 == ('XLONG', 'west_east')
    assert dim2 == ('XLAT', 'south_north')
    # For 'xt' plot
    dim1, dim2 = nuwrf.coord_names('wrf', source_data, 'T', 'xt')
    assert dim1 == 'Time'
    # For 'yz' plot
    dim1, dim2 = nuwrf.coord_names('wrf', source_data, 'T', 'yz')
    assert dim1 == ('XLAT', 'south_north')
    assert dim2 == 'bottom_top'


def test_coord_names_lis(nuwrf):
    # Simulate a LIS field with all expected coords
    field = MagicMock()
    field.dims = ['time', 'lev', 'lat', 'lon']
    field.coords = {
        'lon': MagicMock(),
        'lat': MagicMock(),
        'lev': MagicMock(),
        'time': MagicMock()
    }
    source_data = {'vars': {'soil_moisture': field}}
    # Should return correct dim1, dim2 for 'xy' plot
    dim1, dim2 = nuwrf.coord_names('lis', source_data, 'soil_moisture', 'xy')
    assert dim1 == 'lon'
    assert dim2 == 'lat'
    # For 'xt' plot
    dim1, dim2 = nuwrf.coord_names('lis', source_data, 'soil_moisture', 'xt')
    assert dim1 == 'time'
