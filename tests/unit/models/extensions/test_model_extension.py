"""
Unit tests for the model extension system.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.models.extensions.base import ModelExtension
from eviz.lib.data.sources import DataSource


class TestModelExtension:
    """Test cases for the ModelExtension class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_config_manager = MagicMock(spec=ConfigManager)
        
        self.mock_data_source = MagicMock(spec=DataSource)
        self.mock_data_source.model_name = 'test_model'
        self.mock_data_source.dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90]),
                        'lon': np.array([0, 90, 180, 270])
                    }
                ),
                'pressure': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90]),
                        'lon': np.array([0, 90, 180, 270])
                    }
                )
            }
        )
        
        class ConcreteModelExtension(ModelExtension):
            def process_data_source(self, data_source):
                # Add a new variable to the dataset
                data_source.dataset['derived'] = data_source.dataset['temperature'] + data_source.dataset['pressure']
                return data_source
        
        self.model_extension_class = ConcreteModelExtension
        self.model_extension = ConcreteModelExtension(self.mock_config_manager)
    
    def test_init(self):
        """Test initialization of ModelExtension."""
        assert self.model_extension.config_manager == self.mock_config_manager
    
    def test_process_data_source(self):
        """Test processing a data source."""
        result = self.model_extension.process_data_source(self.mock_data_source)
        
        assert 'derived' in result.dataset.data_vars
        assert result.dataset['derived'].dims == ('time', 'lat', 'lon')
        expected = self.mock_data_source.dataset['temperature'] + self.mock_data_source.dataset['pressure']
        xr.testing.assert_equal(result.dataset['derived'], expected)
