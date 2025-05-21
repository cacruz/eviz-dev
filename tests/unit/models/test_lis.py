import pytest
from unittest.mock import MagicMock
import numpy as np

from eviz.models.esm.lis import Lis

@pytest.fixture
def mock_config_manager():
    mock = MagicMock()
    mock.map_params = {
        0: {'field': 'soil_moisture', 'source_name': 'lis', 'filename': 'dummy.nc', 'to_plot': ['xy']}
    }
    mock.get_file_index.return_value = 0
    mock.get_levels.return_value = [0]
    mock.source_names = ['lis']
    mock.findex = 0
    mock.pindex = 0
    mock.axindex = 0
    mock.ax_opts = {'time_lev': 0, 'tave': False}
    mock.to_plot = {0: ['soil_moisture']}
    mock.meta_coords = {'xc': {'lis': 'lon'}, 'yc': {'lis': 'lat'}}
    mock.meta_attrs = {'field_name': {'lis': 'soil_moisture'}}
    mock.spec_data = {}
    return mock

def test_logger_property(mock_config_manager):
    lis = Lis(config_manager=mock_config_manager)
    assert hasattr(lis.logger, 'info')

def test_global_attrs_property(mock_config_manager):
    lis = Lis(config_manager=mock_config_manager)
    lis._global_attrs = {'DX': 1000, 'DY': 1000}
    assert lis.global_attrs['DX'] == 1000

def test_get_field_for_simple_plot(mock_config_manager):
    lis = Lis(config_manager=mock_config_manager)
    lis.source_data = {'vars': {'soil_moisture': MagicMock()}}
    lis._global_attrs = {'DX': 1000, 'DY': 1000}
    # Patch _get_xy to return a real array
    lis._get_xy = MagicMock(return_value=np.ones((2, 2)))
    # Patch _get_field to return arrays with NaN for testing NaN-filling logic
    lis._get_field = MagicMock(side_effect=lambda name, data: np.array([[np.nan, 2.0], [3.0, 4.0]]) if name == 'east_west' else np.array([[1.0, 2.0], [np.nan, 4.0]]))
    result = lis._get_field_for_simple_plot('soil_moisture', 'xy')
    assert isinstance(result, tuple)
    assert result[3] == 'soil_moisture'
