"""
Unit tests for the ConfigurationAdapter class.
"""

import os
import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.lib.autoviz.configuration_adapter import ConfigurationAdapter
from eviz.lib.data.pipeline import DataPipeline
from eviz.lib.data.sources import DataSource


class TestConfigurationAdapter:
    """Test cases for the ConfigurationAdapter class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_config_manager = MagicMock(spec=ConfigManager)
        self.mock_config_manager.app_data = MagicMock()
        self.mock_config_manager.app_data.inputs = [
            {
                'name': 'test_file.nc',
                'location': '/test/path',
                'exp_name': 'test_model',
                'processing': {'normalize': True},
                'transformations': {
                    'regrid': {'enabled': True, 'target_grid': '1x1'},
                    'subset': {'enabled': False}
                }
            },
            {
                'name': 'test_file2.nc',
                'location': '/test/path',
                'exp_name': 'test_model',
                'processing': {},
                'transformations': {}
            }
        ]
        
        self.mock_pipeline = MagicMock(spec=DataPipeline)
        
        # Add data_sources and pipeline attributes to the mock config manager
        self.mock_config_manager.data_sources = {}
        self.mock_config_manager.pipeline = self.mock_pipeline
        
        self.mock_data_source = MagicMock(spec=DataSource)
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
        
        self.mock_pipeline.process_file.return_value = self.mock_data_source
        
        with patch('eviz.lib.autoviz.configuration_adapter.DataPipeline', return_value=self.mock_pipeline):
            self.adapter = ConfigurationAdapter(self.mock_config_manager)
    
    def test_init(self):
        """Test initialization of ConfigurationAdapter."""
        assert self.adapter.config_manager == self.mock_config_manager
        assert self.adapter.pipeline == self.mock_pipeline
        assert self.adapter.data_sources == {}
    
    def test_process_configuration(self):
        """Test processing the configuration."""
        self.mock_config_manager.input_config = MagicMock()
        self.mock_config_manager.input_config._integrate = False
        self.mock_config_manager.input_config._composite = None
        
        self.adapter.process_configuration()
        assert self.mock_pipeline.process_file.call_count == 2
        
        # Verify the first call
        args1, kwargs1 = self.mock_pipeline.process_file.call_args_list[0]
        assert args1[0] == '/test/path/test_file.nc'
        assert kwargs1['model_name'] == 'test_model'
        assert kwargs1['transform'] is True
        assert kwargs1['transform_params'] == {'regrid': True, 'target_grid': '1x1'}
        
        # Verify the second call
        args2, kwargs2 = self.mock_pipeline.process_file.call_args_list[1]
        assert args2[0] == '/test/path/test_file2.nc'
        assert kwargs2['model_name'] == 'test_model'
        assert kwargs2['transform'] is False
        assert kwargs2['transform_params'] == {}
        
        # Verify that the data sources were stored
        assert '/test/path/test_file.nc' in self.adapter.data_sources
        assert '/test/path/test_file2.nc' in self.adapter.data_sources
        assert self.adapter.data_sources['/test/path/test_file.nc'] == self.mock_data_source
        assert self.adapter.data_sources['/test/path/test_file2.nc'] == self.mock_data_source
        
        # Verify that the data sources were also stored in the config manager
        assert hasattr(self.mock_config_manager, 'data_sources')
        assert '/test/path/test_file.nc' in self.mock_config_manager.data_sources
        assert '/test/path/test_file2.nc' in self.mock_config_manager.data_sources
    
    def test_process_configuration_with_integration(self):
        """Test processing the configuration with integration."""
        self.mock_config_manager.input_config = MagicMock()
        self.mock_config_manager.input_config._integrate = True
        
        mock_dataset = xr.Dataset()
        self.mock_pipeline.integrate_data_sources.return_value = mock_dataset
        
        self.adapter.process_configuration()
        
        self.mock_pipeline.integrate_data_sources.assert_called_once()
    
    def test_process_configuration_with_composite(self):
        """Test processing the configuration with composite fields."""
        self.mock_config_manager.input_config = MagicMock()
        self.mock_config_manager.input_config._composite = {
            'total': {
                'variables': ['temperature', 'pressure'],
                'operation': 'add'
            }
        }
        
        mock_dataset = xr.Dataset()
        self.mock_pipeline.integrate_variables.return_value = mock_dataset
        
        self.adapter.process_configuration()
        
        self.mock_pipeline.integrate_variables.assert_called_once_with(
            ['temperature', 'pressure'], 'add', 'total'
        )
    
    def test_process_configuration_with_error(self):
        """Test processing the configuration with an error."""
        self.mock_config_manager.input_config = MagicMock()
        self.mock_config_manager.input_config._integrate = False
        self.mock_config_manager.input_config._composite = None
        
        # Set up the mock pipeline to raise an exception for the first file
        self.mock_pipeline.process_file.side_effect = [Exception("Test error"), self.mock_data_source]
        
        self.adapter.process_configuration()
        
        # Verify that process_file was called for each input file
        assert self.mock_pipeline.process_file.call_count == 2
        
        # Verify that only the second data source was stored
        assert '/test/path/test_file.nc' not in self.adapter.data_sources
        assert '/test/path/test_file2.nc' in self.adapter.data_sources
    
    def test_get_data_source(self):
        """Test getting a data source."""
        # Set up the data sources
        self.adapter.data_sources = {
            '/test/path/test_file.nc': self.mock_data_source
        }
        
        # Get an existing data source
        result = self.adapter.get_data_source('/test/path/test_file.nc')
        assert result == self.mock_data_source
        
        # Get a non-existent data source
        result = self.adapter.get_data_source('/test/path/non_existent.nc')
        assert result is None
    
    def test_get_all_data_sources(self):
        """Test getting all data sources."""
        self.adapter.data_sources = {
            '/test/path/test_file.nc': self.mock_data_source,
            '/test/path/test_file2.nc': self.mock_data_source
        }
        
        result = self.adapter.get_all_data_sources()
        assert len(result) == 2
        assert '/test/path/test_file.nc' in result
        assert '/test/path/test_file2.nc' in result
        assert result['/test/path/test_file.nc'] == self.mock_data_source
        assert result['/test/path/test_file2.nc'] == self.mock_data_source
    
    def test_get_dataset(self):
        """Test getting the dataset."""
        mock_dataset = xr.Dataset()
        self.mock_pipeline.get_dataset.return_value = mock_dataset
        
        result = self.adapter.get_dataset()
        assert result is mock_dataset
        self.mock_pipeline.get_dataset.assert_called_once()
    
    def test_close(self):
        """Test closing the adapter."""
        self.adapter.close()
        self.mock_pipeline.close.assert_called_once()
