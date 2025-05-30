import pytest
from unittest.mock import MagicMock, patch
from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.config.config import Config
from eviz.lib.config.input_config import InputConfig
from eviz.lib.config.output_config import OutputConfig
from eviz.lib.config.system_config import SystemConfig
from eviz.lib.config.history_config import HistoryConfig
from eviz.lib.config.app_data import AppData


@pytest.fixture
def mock_config():
    """Create a mock Config object."""
    config = MagicMock(spec=Config)
    config.source_names = ["source1", "source2"]
    config.app_data = MagicMock(spec=AppData)
    config.app_data.inputs = [
        {"exp_id": "exp1", "filename": "file1.nc", "location": "/path/to", "name": "file1.nc"},
        {"exp_id": "exp2", "filename": "file2.nc", "location": "/path/to", "name": "file2.nc"}
    ]
    config.spec_data = {"field1": {"plot_type1": {"levels": [1, 2, 3]}}}
    config.meta_coords = {
        "lat": {"source1": "latitude", "source2": "lat,latitude"},
        "lon": {"source1": "longitude", "source2": {"dim": "lon"}}
    }
    config.meta_attrs = {
        "attr1": {"source1": "attribute1", "source2": "attribute2"}
    }
    config._specs_yaml_exists = True
    config._ds_index = 0
    config._findex = 0
    config.map_params = {} 
    return config


@pytest.fixture
def mock_input_config():
    """Create a mock InputConfig object."""
    input_config = MagicMock(spec=InputConfig)
    input_config._overlay = False
    input_config._overlay_exp_ids = None
    input_config._compare = True
    input_config._compare_diff = False
    input_config._compare_exp_ids = ["exp1", "exp2"]
    input_config.file_list = {
        0: {"description": "File 1 description", "exp_name": "Experiment 1", "exp_id": "exp1"},
        1: {"description": "File 2 description", "exp_name": "Experiment 2", "exp_id": "exp2"}
    }
    return input_config


@pytest.fixture
def config_manager(mock_config, mock_input_config):
    """Create a ConfigManager with mock dependencies."""
    output_config = MagicMock(spec=OutputConfig)
    system_config = MagicMock(spec=SystemConfig)
    history_config = MagicMock(spec=HistoryConfig)
    
    return ConfigManager(
        input_config=mock_input_config,
        output_config=output_config,
        system_config=system_config,
        history_config=history_config,
        config=mock_config
    )


class TestConfigManager:
    """Tests for the ConfigManager class."""
    
    def test_initialization(self, config_manager):
        """Test that the ConfigManager initializes correctly."""
        assert config_manager.a_list == [0]
        assert config_manager.b_list == [1]
        assert config_manager._findex == 0
        assert config_manager._ds_index == 0
        
    def test_property_delegation(self, config_manager, mock_config, mock_input_config):
        """Test that properties are correctly delegated to the underlying config objects."""
        assert config_manager.source_names == mock_config.source_names
        assert config_manager.app_data == mock_config.app_data
        assert config_manager.spec_data == mock_config.spec_data
        assert config_manager.compare == mock_input_config._compare
        assert config_manager.compare_diff == mock_input_config._compare_diff
        
    def test_ds_index_property(self, config_manager):
        """Test the ds_index property getter and setter."""
        config_manager.ds_index = 1
        assert config_manager._ds_index == 1
        assert config_manager.ds_index == 1
        assert config_manager.config._ds_index == 1
        
    def test_findex_property(self, config_manager):
        """Test the findex property getter and setter."""
        config_manager.findex = 1
        assert config_manager._findex == 1
        assert config_manager.findex == 1
        assert config_manager.config._findex == 1
        
    def test_lazy_initialization(self, config_manager):
        """Test lazy initialization of pipeline."""
        # Patch DataPipeline where it's looked up (in the module of ConfigManager)
        with patch('eviz.lib.config.config_manager.DataPipeline') as mock_pipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance
            
            # First access should create the pipeline
            pipeline = config_manager.pipeline
            assert pipeline is mock_pipeline_instance 
            mock_pipeline.assert_called_once_with(config_manager)
            
            # Second access should use the cached instance
            mock_pipeline.reset_mock()
            pipeline_cached = config_manager.pipeline # Assign to a different variable
            assert pipeline_cached is mock_pipeline_instance 
            mock_pipeline.assert_not_called()
            
    def test_to_dict(self, config_manager):
        """Test the to_dict method."""
        config_manager.input_config.to_dict.return_value = {"input": "data"}
        config_manager.output_config.to_dict.return_value = {"output": "data"}
        config_manager.system_config.to_dict.return_value = {"system": "data"}
        config_manager.history_config.to_dict.return_value = {"history": "data"}
        
        result = config_manager.to_dict()
        
        assert result == {
            "input_config": {"input": "data"},
            "output_config": {"output": "data"},
            "system_config": {"system": "data"},
            "history_config": {"history": "data"},
            "app_data": config_manager.config.app_data,
            "spec_data": config_manager.config.spec_data,
            "map_params": config_manager.config.map_params,
        }
        
    def test_getattr_delegation(self, config_manager):
        """Test the __getattr__ method for attribute delegation."""
        config_manager.config.test_attr = "config_value"
        config_manager.input_config.test_attr_input = "input_value"
        config_manager.output_config.test_attr_output = "output_value"
        
        assert config_manager.test_attr == "config_value"
        assert config_manager.test_attr_input == "input_value"

        assert config_manager.test_attr_output == "output_value"

        with pytest.raises(AttributeError):
            _ = config_manager.non_existent_attr
            
    @pytest.mark.parametrize("compare, compare_diff, overlay, compare_exp_ids, overlay_exp_ids, expected_a, expected_b", [
        # Test with comparison enabled
        (True, False, False, ["exp1", "exp2", "non_existent"], None, [0], [1]),
        
        # Test with comparison_diff enabled
        (False, True, False, ["exp1", "exp2", "non_existent"], None, [0], [1]),
        
        # Test with overlay enabled
        (False, False, True, None, ["exp1", "exp2", "non_existent"], [0], [1]),
        
        # Test with overlay enabled but using compare_exp_ids as fallback
        (False, False, True, ["exp1", "exp2", "non_existent"], None, [0], [1]),
        
        # Test with both overlay and compare enabled (should use compare_exp_ids)
        (True, False, True, ["exp1", "exp2"], ["exp3", "exp4"], [0], [1]),
        
        # Test with both overlay and compare_diff enabled (should use compare_exp_ids)
        (False, True, True, ["exp1", "exp2"], ["exp3", "exp4"], [0], [1]),
        
        # Test with comparison disabled
        (False, False, False, None, None, [], []),
    ])
    def test_setup_comparison(self, config_manager, mock_input_config, 
                              compare, compare_diff, overlay, 
                              compare_exp_ids, overlay_exp_ids,
                              expected_a, expected_b):
        """Test the setup_comparison method with different configurations."""
        # Reset comparison settings
        config_manager.a_list = []
        config_manager.b_list = []
        
        # Configure the input config
        mock_input_config._compare = compare
        mock_input_config._compare_diff = compare_diff
        mock_input_config._overlay = overlay
        mock_input_config._compare_exp_ids = compare_exp_ids
        mock_input_config._overlay_exp_ids = overlay_exp_ids
        
        # Run the method
        config_manager.setup_comparison()
        
        # Check results
        assert config_manager.a_list == expected_a
        assert config_manager.b_list == expected_b
