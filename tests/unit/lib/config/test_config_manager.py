import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.config.config import Config
from eviz.lib.config.input_config import InputConfig
from eviz.lib.config.output_config import OutputConfig
from eviz.lib.config.system_config import SystemConfig
from eviz.lib.config.history_config import HistoryConfig
from eviz.lib.config.app_data import AppData
import os
import numpy as np


@pytest.fixture
def mock_input_config():
    """Create a mock InputConfig."""
    input_config = MagicMock(spec=InputConfig)
    
    # Set up file_list for testing
    input_config.file_list = [
        {
            'description': 'File 1 description',
            'exp_name': 'Experiment 1',
            'exp_id': 'exp1'
        },
        {
            'description': 'File 2 description',
            'exp_name': 'Experiment 2',
            'exp_id': 'exp2'
        }
    ]
    
    # Add attributes needed for setup_comparison
    input_config._compare_exp_ids = ['exp1', 'exp2']
    input_config._overlay_exp_ids = None
    input_config._compare = False
    input_config._compare_diff = False
    input_config._overlay = False
    
    return input_config

@pytest.fixture
def mock_output_config():
    """Create a mock OutputConfig."""
    output_config = MagicMock(spec=OutputConfig)
    output_config.output_dir = '/test/output'
    output_config.print_to_file = False
    output_config.add_logo = False
    return output_config


@pytest.fixture
def mock_system_config():
    """Create a mock SystemConfig."""
    system_config = MagicMock(spec=SystemConfig)
    system_config.use_mp_pool = False
    system_config.archive_web_results = False
    return system_config


@pytest.fixture
def mock_history_config():
    """Create a mock HistoryConfig."""
    return MagicMock(spec=HistoryConfig)


@pytest.fixture
def mock_config():
    """Create a mock Config."""
    config = MagicMock(spec=Config)
    
    # Set up app_data
    app_data = MagicMock(spec=AppData)
    app_data.inputs = [
        {
            'source_name': 'source1',
            'location': '/path/to',
            'name': 'file1.nc',
            'filename': 'file1.nc'
        },
        {
            'source_name': 'source2',
            'location': '/path/to',
            'name': 'file2.nc',
            'filename': 'file2.nc'
        }
    ]
    config.app_data = app_data
    
    # Set up source_names
    config.source_names = ['source1', 'source2']
    
    # Set up meta_coords
    config.meta_coords = {
        'lat': {
            'source1': 'latitude',
            'source2': 'lat,latitude'
        },
        'lon': {
            'source1': 'longitude',
            'source2': {'dim': 'lon'}
        }
    }
    
    # Set up meta_attrs
    config.meta_attrs = {
        'attr1': {
            'source1': 'attribute1',
            'source2': 'attribute2'
        }
    }
    
    # Set up spec_data
    config.spec_data = {
        'field1': {
            'plot_type1': {
                'levels': [1, 2, 3]
            }
        }
    }
    
    return config


@pytest.fixture
def mock_data_source():
    """Create a mock data source with a dataset."""
    data_source = MagicMock()
    data_source.dataset = MagicMock()
    data_source.dataset.dims = {'lat': 180, 'lon': 360, 'time': 12}
    return data_source


@pytest.fixture
def config_manager(mock_input_config, mock_output_config, mock_system_config, mock_history_config, mock_config):
    """Create a ConfigManager instance with mock dependencies."""
    cm = ConfigManager(
        input_config=mock_input_config,
        output_config=mock_output_config,
        system_config=mock_system_config,
        history_config=mock_history_config,
        config=mock_config
    )
    
    # Mock setup_comparison to avoid side effects
    cm.setup_comparison = MagicMock()
    
    return cm


class TestConfigManager:
    """Tests for the ConfigManager class."""
    
    # Your existing tests...
    
    def test_get_model_dim_name(self, config_manager, mock_config):
        """Test the get_model_dim_name method."""
        # Mock the pipeline and data source
        mock_pipeline = MagicMock()
        mock_data_source = MagicMock()
        mock_data_source.dataset = MagicMock()
        mock_data_source.dataset.dims = {'latitude': 180, 'longitude': 360, 'time': 12}
        
        mock_pipeline.get_data_source.return_value = mock_data_source
        config_manager._pipeline = mock_pipeline
        
        # Test with direct match
        result = config_manager.get_model_dim_name('lat')
        assert result == 'latitude'
        
        # Test with comma-separated options
        config_manager.ds_index = 1  # Use source2
        result = config_manager.get_model_dim_name('lat')
        assert result == 'latitude'  # Should match the first option in "lat,latitude"
        
        # Test with dictionary format
        result = config_manager.get_model_dim_name('lon')
        assert result == 'longitude'  # Should extract from {"dim": "lon"}
        
        # Test with non-existent dimension
        result = config_manager.get_model_dim_name('altitude')
        assert result is None
    
    def test_should_overlay_plots(self, config_manager, mock_input_config):
        """Test the should_overlay_plots method."""
        # Mock the spec_data
        config_manager.config.spec_data = {
            'temperature': {
                'yzplot': {'profile_dim': 'lev'},
                'xtplot': {}
            },
            'pressure': {
                'yzplot': {},
                'xtplot': {}
            }
        }
        
        # Test profile plot with overlay enabled
        mock_input_config._overlay = True
        assert config_manager.should_overlay_plots('temperature', 'yz') is True
        
        # Test non-profile plot with overlay enabled
        assert config_manager.should_overlay_plots('pressure', 'yz') is False
        
        # Test time series plot with overlay enabled
        assert config_manager.should_overlay_plots('temperature', 'xt') is True
        
        # Test box plot with overlay enabled
        assert config_manager.should_overlay_plots('temperature', 'bo') is True
        
        # Test with overlay disabled
        mock_input_config._overlay = False
        assert config_manager.should_overlay_plots('temperature', 'yz') is False
        assert config_manager.should_overlay_plots('temperature', 'xt') is False
    
    def test_get_file_format(self, config_manager, mock_input_config):
        """Test the get_file_format method."""
        # Mock the input_config.get_format_for_file method
        mock_input_config.get_format_for_file = MagicMock(return_value='netcdf')
        
        result = config_manager.get_file_format('/path/to/file.nc')
        assert result == 'netcdf'
        mock_input_config.get_format_for_file.assert_called_once_with('/path/to/file.nc')
        
        # Test when method doesn't exist
        delattr(mock_input_config, 'get_format_for_file')
        result = config_manager.get_file_format('/path/to/file.nc')
        assert result is None
    
    def test_file_formats_property(self, config_manager, mock_input_config):
        """Test the file_formats property."""
        # Mock the _file_format_mapping attribute
        mock_input_config._file_format_mapping = {
            '/path/to/file1.nc': 'netcdf',
            '/path/to/file2.zarr': 'zarr'
        }
        
        result = config_manager.file_formats
        assert result == {
            '/path/to/file1.nc': 'netcdf',
            '/path/to/file2.zarr': 'zarr'
        }
        
        # Test when attribute doesn't exist
        delattr(mock_input_config, '_file_format_mapping')
        result = config_manager.file_formats
        assert result == {}
    
    def test_get_current_file_path(self, config_manager, mock_config):
        """Test the _get_current_file_path method."""
        # Test with valid findex
        config_manager.findex = 0  # Explicitly set to valid index
        result = config_manager._get_current_file_path('source1')
        assert result == '/path/to/file1.nc'
        
        # Test with invalid findex
        config_manager.findex = 5  # Out of range
        
        # Modify app_data.inputs to ensure source1 isn't found in fallback search
        original_inputs = config_manager.app_data.inputs
        config_manager.app_data.inputs = [
            {
                'source_name': 'source2',
                'location': '/path/to',
                'name': 'file2.nc',
                'filename': 'file2.nc'
            }
        ]
        
        result = config_manager._get_current_file_path('source1')
        assert result is None
        
        # Restore original inputs
        config_manager.app_data.inputs = original_inputs
        
        # Test with source name fallback
        config_manager.app_data.inputs.append({
            'source_name': 'source3',
            'location': '/path/to',
            'name': 'file3.nc'
        })
        result = config_manager._get_current_file_path('source3')
        assert result == '/path/to/file3.nc'
    
    def test_get_data_source_for_file(self, config_manager):
        """Test the _get_data_source_for_file method."""
        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_data_source = MagicMock()
        mock_data_source.dataset = MagicMock()
        
        mock_pipeline.get_data_source.return_value = mock_data_source
        config_manager._pipeline = mock_pipeline
        
        # Test with valid file path
        result = config_manager._get_data_source_for_file('/path/to/file.nc')
        assert result == mock_data_source
        mock_pipeline.get_data_source.assert_called_once_with('/path/to/file.nc')
        
        # Test with None file path
        mock_pipeline.reset_mock()
        result = config_manager._get_data_source_for_file(None)
        assert result is None
        mock_pipeline.get_data_source.assert_not_called()
        
        # Test with data source without dataset
        mock_pipeline.reset_mock()
        mock_pipeline.get_data_source.return_value = MagicMock(spec=[])
        result = config_manager._get_data_source_for_file('/path/to/file.nc')
        assert result is None
    
    def test_get_available_dimensions(self, config_manager, mock_data_source):
        """Test the _get_available_dimensions method."""
        # Test with valid data source
        result = ConfigManager._get_available_dimensions(mock_data_source)
        assert result == ['lat', 'lon', 'time']
        
        # Test with None data source
        result = ConfigManager._get_available_dimensions(None)
        assert result is None
        
        # Test with data source without dataset
        result = ConfigManager._get_available_dimensions(MagicMock(spec=[]))
        assert result is None
    
    def test_get_file_index(self, config_manager, mock_config):
        """Test the get_file_index method."""
        # Test with exact filename match
        result = config_manager.get_file_index('file1.nc')
        assert result == 0
        
        # Test with path in filename
        result = config_manager.get_file_index('/some/other/path/file2.nc')
        assert result == 1
        
        # Test with non-existent filename
        result = config_manager.get_file_index('nonexistent.nc')
        assert result == 0
        
        # Test with None filename
        result = config_manager.get_file_index(None)
        assert result == 0
    
    def test_get_levels(self, config_manager, mock_config):
        """Test the get_levels method."""
        # Test with existing levels
        result = config_manager.get_levels('field1', 'plot_type1')
        assert result == [1, 2, 3]
        
        # Test with non-existent field
        result = config_manager.get_levels('nonexistent', 'plot_type1')
        assert result == []
        
        # Test with non-existent plot type
        result = config_manager.get_levels('field1', 'nonexistent')
        assert result == []
    
    def test_get_file_description(self, config_manager, mock_input_config):
        """Test the get_file_description method."""
        # Test with valid file index
        result = config_manager.get_file_description(0)
        assert result == "File 1 description"
        
        # Test with non-existent file index
        result = config_manager.get_file_description(5)
        assert result is None
    
    def test_get_file_exp_name(self, config_manager, mock_input_config):
        """Test the get_file_exp_name method."""
        # Test with valid file index
        result = config_manager.get_file_exp_name(0)
        assert result == "Experiment 1"
        
        # Test with non-existent file index
        result = config_manager.get_file_exp_name(5)
        assert result is None
    
    def test_get_file_exp_id(self, config_manager, mock_input_config):
        """Test the get_file_exp_id method."""
        # Test with valid file index
        result = config_manager.get_file_exp_id(0)
        assert result == "exp1"
        
        # Test with non-existent file index
        result = config_manager.get_file_exp_id(5)
        assert result is None
    
    def test_get_dim_names(self, config_manager):
        """Test the get_dim_names method."""
        # Mock the get_model_dim_name method
        config_manager.get_model_dim_name = MagicMock()
        config_manager.get_model_dim_name.side_effect = lambda dim: {
            'xc': 'lon', 'yc': 'lat', 'zc': 'lev', 'tc': 'time'
        }.get(dim)
        
        # Test with yz plot
        dim1, dim2 = config_manager.get_dim_names('yz')
        assert dim1 == 'lat'
        assert dim2 == 'lev'
        
        # Test with xt plot
        dim1, dim2 = config_manager.get_dim_names('xt')
        assert dim1 == 'time'
        assert dim2 is None
        
        # Test with tx plot
        dim1, dim2 = config_manager.get_dim_names('tx')
        assert dim1 == 'lon'
        assert dim2 == 'time'
        
        # Test with xy plot (default)
        dim1, dim2 = config_manager.get_dim_names('xy')
        assert dim1 == 'lon'
        assert dim2 == 'lat'
    
    def test_get_model_attr_name(self, config_manager, mock_config):
        """Test the get_model_attr_name method."""
        # Test with existing attribute
        result = config_manager.get_model_attr_name('attr1')
        assert result == 'attribute1'
        
        # Test with non-existent attribute
        result = config_manager.get_model_attr_name('nonexistent')
        assert result is None
        
        # Test with out of bounds ds_index
        config_manager.ds_index = 5
        result = config_manager.get_model_attr_name('attr1')
        assert result is None
    
    def test_register_and_get_plot_type(self, config_manager):
        """Test the register_plot_type and get_plot_type methods."""
        # Register a plot type
        config_manager.register_plot_type('temperature', 'yz')
        
        # Get the registered plot type
        result = config_manager.get_plot_type('temperature')
        assert result == 'yz'
        
        # Get a non-registered plot type (should return default)
        result = config_manager.get_plot_type('pressure')
        assert result == 'xy'
        
        # Get a non-registered plot type with custom default
        result = config_manager.get_plot_type('pressure', default='xt')
        assert result == 'xt'
    
    def test_get_file_index_by_filename(self, config_manager, mock_config):
        """Test the get_file_index_by_filename method."""
        # Add some map_params
        mock_config.map_params = {
            0: {'filename': 'file1.nc', 'file_index': 0},
            1: {'filename': 'file2.nc', 'file_index': 1}
        }
        
        # Test with existing filename
        result = config_manager.get_file_index_by_filename('file1.nc')
        assert result == 0
        
        # Test with non-existent filename
        result = config_manager.get_file_index_by_filename('nonexistent.nc')
        assert result == -1
    
    def test_units_lazy_initialization(self, config_manager):
        """Test lazy initialization of Units."""
        # Mock the Units class
        with patch('eviz.lib.data.units.Units') as mock_units_class:
            mock_units_instance = MagicMock()
            mock_units_class.return_value = mock_units_instance
            
            # First access should create the units
            units = config_manager.units
            assert units is mock_units_instance
            mock_units_class.assert_called_once_with(config_manager)
            
            # Second access should use the cached instance
            mock_units_class.reset_mock()
            units_cached = config_manager.units
            assert units_cached is mock_units_instance
            mock_units_class.assert_not_called()
    
    def test_integrator_lazy_initialization(self, config_manager):
        """Test lazy initialization of DataIntegrator."""
        # Mock the DataIntegrator class
        with patch('eviz.lib.config.config_manager.DataIntegrator') as mock_integrator_class:
            mock_integrator_instance = MagicMock()
            mock_integrator_class.return_value = mock_integrator_instance
            
            # First access should create the integrator
            integrator = config_manager.integrator
            assert integrator is mock_integrator_instance
            mock_integrator_class.assert_called_once()
            
            # Second access should use the cached instance
            mock_integrator_class.reset_mock()
            integrator_cached = config_manager.integrator
            assert integrator_cached is mock_integrator_instance
            mock_integrator_class.assert_not_called()
    
    def test_property_delegation_to_configs(self, config_manager):
        """Test property delegation to various config objects."""
        # Test delegation to input_config
        config_manager.input_config._correlation = True
        assert config_manager.correlation is True
        
        config_manager.input_config._overlay = True
        assert config_manager.overlay is True
        
        config_manager.input_config._compare = True
        assert config_manager.compare is True
        
        config_manager.input_config._compare_diff = True
        assert config_manager.compare_diff is True
        
        # Test delegation to output_config
        config_manager.output_config.add_logo = True
        assert config_manager.add_logo is True
        
        config_manager.output_config.print_to_file = True
        assert config_manager.print_to_file is True
        
        config_manager.output_config.output_dir = '/output/dir'
        assert config_manager.output_dir == '/output/dir'
        
        # Test delegation to system_config
        config_manager.system_config.use_mp_pool = True
        assert config_manager.use_mp_pool is True
        
        config_manager.system_config.archive_web_results = True
        assert config_manager.archive_web_results is True
    
    def test_state_variables(self, config_manager):
        """Test state variables used during plotting."""
        # Test pindex
        config_manager.pindex = 5
        assert config_manager.pindex == 5
        assert config_manager.config._pindex == 5
        
        # Test axindex
        config_manager.axindex = 3
        assert config_manager.axindex == 3
        assert config_manager.config._axindex == 3
        
        # Test ax_opts
        test_opts = {'grid': True, 'color': 'red'}
        config_manager.ax_opts = test_opts
        assert config_manager.ax_opts == test_opts
        assert config_manager.config._ax_opts == test_opts
        
        # Test level
        config_manager.level = 850
        assert config_manager.level == 850
        assert config_manager.config._level == 850
        
        # Test time_level
        config_manager.time_level = 10
        assert config_manager.time_level == 10
        assert config_manager.config._time_level == 10
        
        # Test real_time
        config_manager.real_time = '2023-01-01 12:00'
        assert config_manager.real_time == '2023-01-01 12:00'
        assert config_manager.config._real_time == '2023-01-01 12:00'
