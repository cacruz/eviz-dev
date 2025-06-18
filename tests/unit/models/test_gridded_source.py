import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch
from eviz.models.gridded_source import GriddedSource


def make_config_manager(tc_name='time', zc_name='lev', xc_name='lon', yc_name='lat'):
    cm = MagicMock()
    cm.get_model_dim_name.side_effect = lambda x: {
        'tc': tc_name, 'zc': zc_name, 'xc': xc_name, 'yc': yc_name
    }.get(x)
    cm.ax_opts = {}
    cm.spec_data = {}
    cm.pipeline.get_data_source.return_value = MagicMock(dataset={'myfield': xr.DataArray(np.ones((2,2)), dims=('lat','lon'))})
    cm.pipeline.get_all_data_sources.return_value = [MagicMock(dataset={'myfield': xr.DataArray(np.ones((2,2)), dims=('lat','lon'))})]
    cm.map_params = {0: {'field': 'myfield', 'filename': 'file.nc', 'to_plot': ['xy']}}
    cm.findex = 0
    cm.pindex = 0
    cm.axindex = 0
    cm.make_gif = False
    cm.compare_exp_ids = [0, 1]
    cm.a_list = [0]
    cm.b_list = [1]
    cm.input_config._comp_panels = (3, 1)
    return cm


def test_get_xy_simple_basic():
    arr = xr.DataArray(np.random.rand(2, 3, 4), dims=('time', 'lev', 'lat'))
    g = GriddedSource(config_manager=make_config_manager())
    result = g._get_xy_simple(arr)
    assert isinstance(result, xr.DataArray)
    assert result.ndim <= 2


def test_get_xy_simple_none():
    g = GriddedSource(config_manager=make_config_manager())
    assert g._get_xy_simple(None) is None


def test_get_field_for_simple_plot_xy():
    arr = xr.DataArray(np.random.rand(2, 3, 4), dims=('time', 'lev', 'lat'))
    g = GriddedSource(config_manager=make_config_manager())
    tup = g._get_field_for_simple_plot(arr, 'myfield', 'xy')
    assert tup is None


def test_get_data_source_delegates():
    cm = make_config_manager()
    cm.pipeline.get_data_source.return_value = "ds"
    g = GriddedSource(config_manager=cm)
    assert g.get_data_source("foo") == "ds"


def test_get_yz_simple_basic():
    arr = xr.DataArray(np.random.rand(2, 3, 4), dims=('time', 'lev', 'lon'))
    g = GriddedSource(config_manager=make_config_manager())
    # Patch apply_conversion to just return its input
    with patch('eviz.models.source_base.apply_conversion', lambda cm, da, name: da):
        result = g._get_yz_simple(arr)
    assert isinstance(result, xr.DataArray)


def test_get_field_for_simple_plot_graph():
    arr = xr.DataArray(np.random.rand(2, 3), dims=('x', 'y'))
    g = GriddedSource(config_manager=make_config_manager())
    tup = g._get_field_for_simple_plot(arr, 'myfield', 'graph')
    assert tup is None


def test_get_field_for_simple_plot_none():
    g = GriddedSource(config_manager=make_config_manager())
    assert g._get_field_for_simple_plot(None, 'myfield', 'xy') is None


def test_get_field_for_simple_plot_badtype():
    arr = xr.DataArray(np.random.rand(2, 3), dims=('x', 'y'))
    g = GriddedSource(config_manager=make_config_manager())
    assert g._get_field_for_simple_plot(arr, 'myfield', 'badtype') is None


def test_get_yz_simple_none():
    g = GriddedSource(config_manager=make_config_manager())
    assert g._get_yz_simple(None) is None


def test_get_yz_simple_missing_dim():
    arr = xr.DataArray(np.random.rand(2, 3), dims=('time', 'lat'))
    g = GriddedSource(config_manager=make_config_manager())
    result = g._get_yz_simple(arr)
    # Should return a 1D array over 'lat'
    assert isinstance(result, xr.DataArray)
    assert result.dims == ('lat',) or result.ndim == 1


def test_get_yz_simple_zonal_mean_missing_dim():
    arr = xr.DataArray(np.random.rand(2, 3), dims=('lev', 'lat'))
    g = GriddedSource(config_manager=make_config_manager())
    g.config_manager.get_model_dim_name = lambda x: {'xc': 'lon', 'tc': 'time', 'zc': 'lev'}.get(x)
    result = g._get_yz_simple(arr)
    # Should return a 2D array over ('lev', 'lat')
    assert isinstance(result, xr.DataArray)
    assert result.dims == ('lev', 'lat') or result.ndim == 2


def test_logger_property():
    g = GriddedSource(config_manager=make_config_manager())
    assert hasattr(g.logger, 'info')
