import unittest
from unittest.mock import patch
import pytest
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from eviz.lib.autoviz.plotting.factory import PlotterFactory
from eviz.lib.autoviz.plotting.backends.matplotlib.xy_plot import MatplotlibXYPlotter
from eviz.lib.autoviz.plotting.backends.matplotlib.polar_plot import MatplotlibPolarPlotter
from eviz.lib.autoviz.plotting.backends.hvplot.metric_plot import HvplotMetricPlotter
from eviz.lib.autoviz.plotting.backends.hvplot.box_plot import HvplotBoxPlotter


class TestPlotterFactory(unittest.TestCase):
    """Test the PlotterFactory class."""

    def test_create_plotter_matplotlib_xy(self):
        """Test creating a matplotlib xy plotter."""
        plotter = PlotterFactory.create_plotter('xy', 'matplotlib')
        self.assertIsInstance(plotter, MatplotlibXYPlotter)

    def test_create_plotter_matplotlib_polar(self):
        """Test creating a matplotlib polar plotter."""
        plotter = PlotterFactory.create_plotter('polar', 'matplotlib')
        self.assertIsInstance(plotter, MatplotlibPolarPlotter)

    def test_create_plotter_hvplot_metric(self):
        """Test creating an hvplot metric plotter."""
        plotter = PlotterFactory.create_plotter('corr', 'hvplot')
        self.assertIsInstance(plotter, HvplotMetricPlotter)

    def test_create_plotter_hvplot_box(self):
        """Test creating an hvplot box plotter."""
        plotter = PlotterFactory.create_plotter('box', 'hvplot')
        self.assertIsInstance(plotter, HvplotBoxPlotter)

    def test_create_plotter_default_backend(self):
        """Test creating a plotter with the default backend."""
        plotter = PlotterFactory.create_plotter('xy')
        self.assertIsInstance(plotter, MatplotlibXYPlotter)

    def test_create_plotter_unsupported_type(self):
        """Test creating a plotter with an unsupported type."""
        with self.assertRaises(ValueError):
            PlotterFactory.create_plotter('unsupported_type')


if __name__ == '__main__':
    unittest.main()
