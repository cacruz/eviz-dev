import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import xarray as xr # Import xarray
from eviz.lib.autoviz.plotter import (
    _single_xy_plot,
    _create_clevs,
    _time_series_plot,
    _determine_axes_shape,
)


# Fixtures
@pytest.fixture
def mock_config_manager():
    mock = MagicMock()
    mock.ax_opts = {
        'use_cmap': 'viridis',
        'clevs': np.linspace(0, 10, 11),
        'extend_value': 'both',
        'cscale': None,
        'clabel': None,
        'add_grid': True,
        'zave': False,
        'zsum': False,
        'extent': [],
        'time_series_plot_linestyle': {},
        'cbar_sci_notation': False,
        'clevs_prec': 1,
        'cmap_set_under': None,
        'cmap_set_over': None,
        'create_clevs': True,
        'num_clevs': 10,
        'is_diff_field': False, # for _select_axes
        'add_extra_field_type': False, # for _select_axes
        'use_diff_cmap': 'coolwarm_r', # for _plot_xy_data via _single_xy_plot
    }
    mock.source_names = ['gridded']
    mock.ds_index = 0
    mock.findex = 0 # Used in _simple_xy_plot for meta_attrs
    mock.compare = False
    mock.compare_diff = False # for _plot_xy_data
    mock.axindex = 0
    mock.spec_data = {
        'test_field': {
            'units': 'K',
            'xyplot': {},
            'yzplot': {'zrange': [1000, 100]},
            'xtplot': {'mean_type': None}
        }
    }
    mock.meta_attrs = { # for _simple_xy_plot
        'field_name': {
            'gridded': 'Test Field Name From Meta'
        }
    }
    mock.get_dim_names = MagicMock(return_value=('dim1_name', 'dim2_name'))
    mock.cmap = 'viridis' 
    mock.add_logo = False
    mock.EVIZ_LOGO = 'dummy_logo_path' # For pu.add_logo_fig
    return mock


@pytest.fixture
def test_data_2d():
    # Create a simple 2D array for testing
    x_np = np.linspace(-5, 5, 20)
    y_np = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x_np, y_np)
    Z = np.sin(X) * np.cos(Y)

    # Create DataArray
    data = xr.DataArray(
        Z,
        coords={'lat': y_np, 'lon': x_np},
        dims=['lat', 'lon'],
        attrs={'units': 'K', 'long_name': 'Test Field'}
    )

    # Create mock figure and axes
    fig = MagicMock()
    ax = MagicMock()
    # Mock get_axes() to return a list containing the mock ax
    fig.get_axes.return_value = [ax]
    fig.get_gs_geometry = MagicMock(return_value=(1,1)) # for _single_xy_plot

    # Add necessary methods to fig
    fig.update_ax_opts = MagicMock(return_value={
        'use_cmap': 'viridis',
        'clevs': np.linspace(-1, 1, 11),
        'extend_value': 'both',
        'cscale': None,
        'clabel': None,
        'add_grid': True,
        'extent': None, # Added for _plot_xy_data
        'use_diff_cmap': 'coolwarm_r',
        'cmap_set_under': None,
        'cmap_set_over': None,
        'clevs_prec': 1,
        'cbar_sci_notation': False,

    })
    fig.plot_text = MagicMock()
    fig.subplots = (1, 1)
    fig.EVIZ_LOGO = 'dummy_logo_path' # For pu.add_logo_fig

    return data, x_np, y_np, 'test_field', 'xy', 0, fig, ax

@pytest.fixture
def simple_plot_data():
    x_coord = xr.DataArray(np.linspace(0, 350, 36), dims='lon', attrs={'name': 'Longitude', 'units': 'degrees_east'})
    y_coord = xr.DataArray(np.linspace(-90, 90, 19), dims='lat', attrs={'name': 'Latitude', 'units': 'degrees_north'})
    data_values = np.random.rand(19, 36)
    data = xr.DataArray(
        data_values,
        coords={'lat': y_coord, 'lon': x_coord},
        dims=['lat', 'lon'],
        attrs={'units': 'm/s', 'long_name': 'Sample Data'}
    )
    return data, x_coord, y_coord, 'sample_field', 'xy'


# Tests
def test_create_clevs(mock_config_manager, test_data_2d):
    data2d = test_data_2d[0]
    ax_opts = {'num_clevs': 10, 'create_clevs': True} # Ensure create_clevs is True

    _create_clevs('test_field', ax_opts, data2d)

    assert 'clevs' in ax_opts
    assert len(ax_opts['clevs']) > 0
    # Allow for slight differences due to np.around and linspace behavior
    assert ax_opts['clevs'][0] >= data2d.min().values.item() - 1e-9
    assert ax_opts['clevs'][-1] <= data2d.max().values.item() + 1e-9


def test_xy_plot_creation(mock_config_manager, test_data_2d):
    # Fix: More comprehensive mocking
    with patch('eviz.lib.autoviz.plotter._plot_xy_data') as mock_plot_xy_data, \
            patch('eviz.lib.autoviz.plotter._determine_axes_shape', return_value=(1, 1)), \
            patch('eviz.lib.autoviz.plotter._select_axes', return_value=test_data_2d[7]):
        # Use non-interactive backend
        with plt.rc_context({'backend': 'Agg'}):
            # Should not raise an exception
            _single_xy_plot(mock_config_manager, test_data_2d, level=0)

            mock_plot_xy_data.assert_called_once()


def test_xy_plot_with_none_data(mock_config_manager):
    # Create test data with None for data2d
    test_data = (None, np.array([1, 2, 3]), np.array([4, 5, 6]), 'test_field', 'xy', 0, MagicMock(), MagicMock())

    # Should return early without error
    result = _single_xy_plot(mock_config_manager, test_data, level=0)
    assert result is None


def test_time_series_plot(mock_config_manager, test_data_2d):
    import pandas as pd

    times = pd.date_range('2020-01-01', periods=10)
    data = xr.DataArray(
        np.sin(np.linspace(0, 2 * np.pi, 10)),
        coords={'time': times},
        dims=['time'],
        attrs={'units': 'K'}
    )

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_fig.subplots = (1,1) # for pu.axis_tick_font_size

    # Add necessary attributes to mock_config_manager for this test
    mock_config_manager.get_model_dim_name = MagicMock(return_value='time')
    mock_config_manager.get_primary_reader = MagicMock(return_value=None)
    # Ensure xtplot key exists for spec_data
    mock_config_manager.spec_data['test_field']['xtplot'] = {
        'mean_type': 'none', # or 'rolling'
        # 'window_size': 5 # if rolling
        # 'add_trend': False # or True
        # 'trend_polyfit': 1 # if add_trend is True
    }


    with patch('matplotlib.pyplot.figure'): # Keep this if figure creation is part of the tested logic
        _time_series_plot(mock_config_manager, mock_ax, mock_config_manager.ax_opts,
                          mock_fig, data, 'test_field', 0)

    mock_ax.plot.assert_called_once()
    mock_ax.set_ylabel.assert_called_once()
    mock_ax.set_xlim.assert_called_once()


def test_plot_components(mock_config_manager, test_data_2d):
    data2d, x, y, field_name, plot_type, findex, fig, ax = test_data_2d
    ax_opts = mock_config_manager.ax_opts
    ax_opts.update(fig.update_ax_opts(field_name, ax, 'xy', level=0))


    with patch('eviz.lib.autoviz.plotter._filled_contours') as mock_filled_contours, \
            patch('eviz.lib.autoviz.plotter._set_colorbar') as mock_set_colorbar, \
            patch('eviz.lib.autoviz.plotter._line_contours') as mock_line_contours:
        # Mock the return value of _filled_contours
        mock_filled_contours.return_value = MagicMock()

        # Call the function under test
        from eviz.lib.autoviz.plotter import _plot_xy_data
        _plot_xy_data(mock_config_manager, ax, data2d, x, y, field_name, fig,
                      ax_opts, level=0, plot_type=plot_type, findex=findex)

        # Verify the helper functions were called
        mock_filled_contours.assert_called_once()
        mock_set_colorbar.assert_called_once()
        mock_line_contours.assert_called_once()

def test_determine_axes_shape():
    mock_fig = MagicMock()
    mock_ax_single = MagicMock(spec=plt.Axes) # Single axes
    mock_ax_list = [MagicMock(spec=plt.Axes), MagicMock(spec=plt.Axes)] # List of axes

    # Test with single GeoAxes (or regular Axes)
    mock_fig.get_gs_geometry = MagicMock(return_value=(1,1)) # Should not be called here
    shape = _determine_axes_shape(mock_fig, mock_ax_single)
    assert shape == (1, 1)
    mock_fig.get_gs_geometry.assert_not_called()

    # Test with list of axes
    mock_fig.get_gs_geometry = MagicMock(return_value=(1,2))
    shape = _determine_axes_shape(mock_fig, mock_ax_list)
    assert shape == (1, 2)
    mock_fig.get_gs_geometry.assert_called_once()

    # Test with a single item list
    mock_fig.get_gs_geometry = MagicMock(return_value=(1,1))
    shape = _determine_axes_shape(mock_fig, [mock_ax_single])
    assert shape == (1,1)


def test_select_axes(mock_config_manager):
    mock_axes_list_3 = [MagicMock(), MagicMock(), MagicMock()]
    mock_axes_list_4 = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    single_ax = MagicMock()

