import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch
from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.data.pipeline import DataPipeline
from eviz.lib.data.sources import DataSource
from eviz.lib.config.configuration_adapter import ConfigurationAdapter


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
        
        # Patch the DataPipeline class to return our mock
        with patch('eviz.lib.config.configuration_adapter.DataPipeline', return_value=self.mock_pipeline):
            self.adapter = ConfigurationAdapter(self.mock_config_manager)
            
    def test_init(self):
        config_manager = MagicMock()
        adapter = ConfigurationAdapter(config_manager)
        assert adapter.config_manager == config_manager
        assert adapter.data_sources == {}  # Check for empty dictionary instead of None

    def test_process_configuration(self):
        config_manager = MagicMock()
        config_manager.app_data.inputs = [{'name': 'test.nc', 'source_name': 'test_model'}]
        config_manager.source_names = ['default_model']
        config_manager.ds_index = 0
        
        # Create a mock pipeline
        pipeline_mock = MagicMock(spec=DataPipeline)
        pipeline_mock.process_file.return_value = MagicMock()
        
        # Patch the DataPipeline class to return our mock
        with patch('eviz.lib.config.configuration_adapter.DataPipeline', return_value=pipeline_mock):
            adapter = ConfigurationAdapter(config_manager)
            adapter.process_configuration()
        
        pipeline_mock.process_file.assert_called_once()
        # Check that the first positional argument is 'test.nc'
        assert pipeline_mock.process_file.call_args[0][0] == 'test.nc'
        # Check that model_name is 'test_model' or 'default_model'
        model_name = pipeline_mock.process_file.call_args[1].get('model_name')
        assert model_name in ['test_model', 'default_model']
        assert len(adapter.data_sources) == 1

    def test_process_configuration_with_composite(self):
        config_manager = MagicMock()
        config_manager.app_data.inputs = [
            {'name': 'test1.nc', 'source_name': 'test_model'},
            {'name': 'test2.nc', 'source_name': 'test_model'}
        ]
        config_manager.source_names = ['default_model']
        config_manager.ds_index = 0
        config_manager.input_config._composite = {
            'variables': ['var1', 'var2'],
            'operation': 'add',
            'output_name': 'composite'
        }
        
        # Create a mock pipeline
        pipeline_mock = MagicMock(spec=DataPipeline)
        pipeline_mock.process_file.return_value = MagicMock()
        
        # Patch the DataPipeline class to return our mock
        with patch('eviz.lib.config.configuration_adapter.DataPipeline', return_value=pipeline_mock):
            adapter = ConfigurationAdapter(config_manager)
            adapter.process_configuration()
        
        assert pipeline_mock.process_file.call_count == 2
        pipeline_mock.integrate_variables.assert_called_once_with(
            ['var1', 'var2'], 'add', 'composite'
        )

    def test_process_configuration_with_integration(self):
        """Test processing the configuration with integration."""
        self.mock_config_manager.input_config = MagicMock()
        self.mock_config_manager.input_config._integrate = True
        
        mock_dataset = xr.Dataset()
        self.mock_pipeline.integrate_data_sources.return_value = mock_dataset
        
        self.adapter.process_configuration()
        
        self.mock_pipeline.integrate_data_sources.assert_called_once()

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
    