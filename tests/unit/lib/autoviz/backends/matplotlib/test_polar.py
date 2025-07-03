import unittest
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../')))

from eviz.lib.autoviz.plotting.backends.matplotlib.polar_plot import MatplotlibPolarPlotter


class TestMatplotlibPolarPlotter(unittest.TestCase):
    """Test the MatplotlibPolarPlotter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.plotter = MatplotlibPolarPlotter()
        
        # Create a mock config
        self.config = MagicMock()
        self.config.ax_opts = {
            'use_cmap': 'viridis',
            'clevs': [-1, -0.5, 0, 0.5, 1],
            'use_pole': 'north',
            'boundary': True,
            'line_contours': True,
            'add_grid': True,
            'clabel': 'Test Units'
        }
        self.config.spec_data = {
            'test_field': {
                'name': 'Test Field',
                'units': 'Test Units'
            }
        }
        
        # Create test data
        lat = np.linspace(-90, 90, 73)
        lon = np.linspace(-180, 180, 144)
        data = np.random.rand(len(lat), len(lon))
        self.test_data = xr.DataArray(
            data,
            coords={'lat': lat, 'lon': lon},
            dims=['lat', 'lon'],
            name='test_field'
        )
        
        # Create a mock figure
        self.fig = MagicMock()
        self.fig.subplots = (1, 1)
        self.fig.get_axes.return_value = MagicMock()
        self.fig.update_ax_opts.return_value = self.config.ax_opts
        
        # Create data_to_plot tuple
        self.data_to_plot = (self.test_data, lon, lat, 'test_field', 'polar', 0, self.fig)

    @patch('matplotlib.pyplot.figure')
    @patch('cartopy.crs.NorthPolarStereo')
    @patch('cartopy.crs.SouthPolarStereo')
    @patch('cartopy.crs.PlateCarree')
    def test_plot(self, mock_plate_carree, mock_south_polar, mock_north_polar, mock_figure):
        """Test the plot method."""
        # Mock the necessary matplotlib and cartopy objects
        mock_figure.return_value = self.fig
        mock_north_polar.return_value = MagicMock()
        mock_south_polar.return_value = MagicMock()
        mock_plate_carree.return_value = MagicMock()
        
        # Call the plot method
        result = self.plotter.plot(self.config, self.data_to_plot)
        
        # Check that the result is the figure
        self.assertEqual(result, self.fig)
        
        # Check that the plot_object attribute is set
        self.assertEqual(self.plotter.plot_object, self.fig)

    def test_convert_to_polar(self):
        """Test the _convert_to_polar method."""
        # Create test data
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        
        # Call the method
        theta, r = self.plotter._convert_to_polar(x, y)
        
        # Check the results
        self.assertEqual(len(theta), len(x))
        self.assertEqual(len(r), len(y))


if __name__ == '__main__':
    unittest.main()
