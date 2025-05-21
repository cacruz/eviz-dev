import pytest
from unittest.mock import MagicMock
import numpy as np

from eviz.models.esm.wrf import Wrf

@pytest.fixture
def mock_config_manager():
    mock = MagicMock()
    mock.map_params = {
        0: {'field': 'T', 'source_name': 'wrf', 'filename': 'dummy.nc', 'to_plot': ['xy']}
    }
    mock.get_file_index.return_value = 0
    mock.get_levels.return_value = [1000]
    mock.source_names = ['wrf']
    mock.findex = 0
    mock.pindex = 0
    mock.axindex = 0
    mock.ax_opts = {'time_lev': 0, 'tave': False}
    mock.to_plot = {0: ['T']}
    mock.meta_coords = {'xc': {'wrf': 'lon'}, 'yc': {'wrf': 'lat'}}
    mock.meta_attrs = {'field_name': {'wrf': 'T'}}
    mock.spec_data = {}
    return mock

def test_global_attrs_property(mock_config_manager):
    wrf = Wrf(config_manager=mock_config_manager)
    wrf._global_attrs = {'P_TOP': 10000}
    assert wrf.global_attrs['P_TOP'] == 10000

def test_init_domain_sets_levels(mock_config_manager):
    wrf = Wrf(config_manager=mock_config_manager)
    wrf.source_data = {
        'vars': {
            'P_TOP': np.array([10000]),
            'ZNW': np.array([[0.0, 0.5, 1.0]]),
            'ZNU': np.array([[0.25, 0.75]])
        }
    }
    wrf._init_domain()
    assert hasattr(wrf, 'levf')
    assert hasattr(wrf, 'levs')
    assert len(wrf.levf) == 3
    assert len(wrf.levs) == 2
