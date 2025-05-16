import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import io
from PIL import Image

from eviz.lib.autoviz.plotter import _single_xy_plot, _create_clevs, _time_series_plot

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
        'time_series_plot_linestyle': {},
        'cbar_sci_notation': False,
        'clevs_prec': 1,
        'cmap_set_under': None,
        'cmap_set_over': None,
        'create_clevs': True,  # Add this to ensure clevs are created
        'num_clevs': 10,       # Add this to control number of levels
    }
    mock.source_names = ['generic']
    mock.ds_index = 0
    mock.findex = 0
    mock.compare = False
    mock.axindex = 0  # Add this to fix the index error
    mock.spec_data = {
        'test_field': {
            'units': 'K',
            'xyplot': {},
            'yzplot': {'zrange': [1000, 100]},
            'xtplot': {'mean_type': None}
        }
    }
    # Add any other required attributes
    mock.add_logo = False
    return mock

@pytest.fixture
def test_data_2d():
    # Create a simple 2D array for testing
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    
    # Create DataArray
    import xarray as xr
    data = xr.DataArray(
        Z,
        coords={'lat': y, 'lon': x},
        dims=['lat', 'lon'],
        attrs={'units': 'K', 'long_name': 'Test Field'}
    )
    
    # Create mock figure and axes
    fig = MagicMock()
    ax = MagicMock()
    fig.get_axes.return_value = [ax]
    
    # Add necessary methods to fig
    fig.update_ax_opts = MagicMock(return_value={
        'use_cmap': 'viridis',
        'clevs': np.linspace(-1, 1, 11),
        'extend_value': 'both',
        'cscale': None,
        'clabel': None,
        'add_grid': True,
    })
    fig.plot_text = MagicMock()
    fig.subplots = (1, 1)
    
    return (data, x, y, 'test_field', 'xy', 0, fig, ax)

# Tests
def test_create_clevs(mock_config_manager, test_data_2d):
    data2d = test_data_2d[0]
    ax_opts = {}
    
    _create_clevs('test_field', ax_opts, data2d)
    
    assert 'clevs' in ax_opts
    assert len(ax_opts['clevs']) > 0
    assert ax_opts['clevs'][0] >= data2d.min().values.item()
    assert ax_opts['clevs'][-1] <= data2d.max().values.item()

def test_xy_plot_creation(mock_config_manager, test_data_2d):
    # Fix: More comprehensive mocking
    with patch('eviz.lib.autoviz.plotter._plot_xy_data') as mock_plot_xy_data, \
         patch('eviz.lib.autoviz.plotter._determine_axes_shape', return_value=(1, 1)), \
         patch('eviz.lib.autoviz.plotter._select_axes', return_value=test_data_2d[7]):
        
        # Use non-interactive backend
        with plt.rc_context({'backend': 'Agg'}):
            # Should not raise an exception
            _single_xy_plot(mock_config_manager, test_data_2d, level=0)
            
            # Verify _plot_xy_data was called
            mock_plot_xy_data.assert_called_once()

def test_xy_plot_with_none_data(mock_config_manager):
    # Create test data with None for data2d
    test_data = (None, np.array([1, 2, 3]), np.array([4, 5, 6]), 'test_field', 'xy', 0, MagicMock(), MagicMock())
    
    # Should return early without error
    result = _single_xy_plot(mock_config_manager, test_data, level=0)
    assert result is None

def test_time_series_plot(mock_config_manager, test_data_2d):
    # Create time series data
    import pandas as pd
    import xarray as xr
    
    times = pd.date_range('2020-01-01', periods=10)
    data = xr.DataArray(
        np.sin(np.linspace(0, 2*np.pi, 10)),
        coords={'time': times},
        dims=['time'],
        attrs={'units': 'K'}
    )
    
    mock_ax = MagicMock()
    mock_fig = MagicMock()
    
    # Add necessary attributes to mock_config_manager for this test
    mock_config_manager.get_model_dim_name = MagicMock(return_value='time')
    mock_config_manager.get_primary_reader = MagicMock(return_value=None)
    
    with patch('matplotlib.pyplot.figure'):
        _time_series_plot(mock_config_manager, mock_ax, mock_config_manager.ax_opts, 
                         mock_fig, data, 'test_field', 0)
    
    # Verify ax.plot was called
    mock_ax.plot.assert_called_once()
    
    # Verify other methods were called
    mock_ax.set_ylabel.assert_called_once()
    mock_ax.set_xlim.assert_called_once()

def test_plot_components(mock_config_manager, test_data_2d):
    data2d, x, y, field_name, plot_type, findex, fig, ax = test_data_2d
    
    with patch('eviz.lib.autoviz.plotter._filled_contours') as mock_filled_contours, \
         patch('eviz.lib.autoviz.plotter._set_colorbar') as mock_set_colorbar, \
         patch('eviz.lib.autoviz.plotter._line_contours') as mock_line_contours:
        
        # Mock the return value of _filled_contours
        mock_filled_contours.return_value = MagicMock()
        
        # Call the function under test
        from eviz.lib.autoviz.plotter import _plot_xy_data
        _plot_xy_data(mock_config_manager, ax, data2d, x, y, field_name, fig, 
                     mock_config_manager.ax_opts, level=0, plot_type=plot_type, findex=findex)
        
        # Verify the helper functions were called
        mock_filled_contours.assert_called_once()
        mock_set_colorbar.assert_called_once()
        mock_line_contours.assert_called_once()
