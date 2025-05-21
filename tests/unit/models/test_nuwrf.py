import pytest
from unittest.mock import MagicMock
from eviz.models.esm.nuwrf import NuWrf

@pytest.fixture
def mock_config_manager():
    mock = MagicMock()
    mock.meta_coords = {'xc': {'wrf': 'lon'}, 'yc': {'wrf': 'lat'}}
    mock.source_names = ['wrf']
    mock.findex = 0
    return mock

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

def test_coord_names_wrf(mock_config_manager):
    nuwrf = NuWrf(config_manager=mock_config_manager)
    nuwrf.source_name = 'wrf'
    nuwrf.get_model_coord_name = MagicMock(return_value='lon')
    nuwrf.get_model_dim_name = MagicMock(return_value='lon')
    nuwrf.get_field_dim_name = MagicMock(return_value='lev')
    source_data = {'vars': {'T': MagicMock(stagger='X', coords={'lon': 1})}}
    dim1, dim2 = nuwrf.coord_names('wrf', source_data, 'T', 'xy')
    assert dim1 is not None
    assert dim2 is not None
