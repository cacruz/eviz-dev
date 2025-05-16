import os
import pytest
from argparse import Namespace
from unittest.mock import patch, MagicMock, PropertyMock

from eviz.lib.autoviz.base import (
    get_config_path_from_env,
    create_config,
    get_factory_from_user_input,
    Autoviz
)
from eviz.models.root_factory import (
    GriddedFactory,
    WrfFactory,
    LisFactory,
    AirnowFactory,
    MopittFactory,
    LandsatFactory,
    OmiFactory,
    FluxnetFactory,
)

# Fixtures
@pytest.fixture
def mock_env_path():
    return "/mock/config/path"

@pytest.fixture
def basic_args():
    return Namespace(
        sources=["gridded"],
        compare=False,
        file=None,
        vars=None,
        configfile=None,
        config=None,
        data_dirs=None,
        output_dirs=None,
        verbose=1
    )

@pytest.fixture
def mock_config_manager():
    mock = MagicMock()
    mock.app_data.inputs = [
        {"location": "/path/to", "name": "file1.nc"},
        {"location": "/path/to", "name": "file2.nc"}
    ]
    # Mock the input_config and output_config
    mock.input_config = MagicMock()
    mock.output_config = MagicMock()
    return mock

@pytest.fixture
def mock_autoviz():
    # Create a mock logger that won't try to write to any files
    mock_logger = MagicMock()
    
    # Create the Autoviz instance with mocked components
    with patch('eviz.lib.autoviz.base.create_config') as mock_create_config:
        mock_config = MagicMock()
        mock_config.app_data.inputs = [
            {"location": "/path/to", "name": "file1.nc"},
            {"location": "/path/to", "name": "file2.nc"}
        ]
        mock_config.input_config = MagicMock()
        mock_config.output_config = MagicMock()
        mock_create_config.return_value = mock_config
        
        # Patch the logging.getLogger before creating the Autoviz instance
        with patch('logging.getLogger') as mock_get_logger:
            mock_get_logger.return_value = mock_logger
            autoviz = Autoviz(source_names=["gridded"])
            
            # Don't mock _check_input_files here, we'll do it in the specific tests
            yield autoviz

# Tests for get_config_path_from_env
def test_get_config_path_from_env_exists(mock_env_path):
    with patch.dict('os.environ', {'EVIZ_CONFIG_PATH': mock_env_path}):
        result = get_config_path_from_env()
        assert result == mock_env_path

def test_get_config_path_from_env_not_exists():
    with patch.dict('os.environ', {}, clear=True):
        result = get_config_path_from_env()
        assert result is None

# Tests for get_factory_from_user_input
@pytest.mark.parametrize("input_source,expected_factory_type", [
    (["gridded"], GriddedFactory),
    (["wrf"], WrfFactory),
    (["lis"], LisFactory),
    (["airnow"], AirnowFactory),
    (["mopitt"], MopittFactory),
    (["landsat"], LandsatFactory),
    (["omi"], OmiFactory),
    (["fluxnet"], FluxnetFactory),
])
def test_get_factory_from_user_input_single(input_source, expected_factory_type):
    factories = get_factory_from_user_input(input_source)
    assert len(factories) == 1
    assert isinstance(factories[0], expected_factory_type)

def test_get_factory_from_user_input_multiple():
    factories = get_factory_from_user_input(["gridded", "wrf", "lis"])
    assert len(factories) == 3
    assert isinstance(factories[0], GriddedFactory)
    assert isinstance(factories[1], WrfFactory)
    assert isinstance(factories[2], LisFactory)

def test_get_factory_from_user_input_invalid():
    with pytest.raises(KeyError):
        get_factory_from_user_input(["invalid_source"])

# Tests for Autoviz class
@patch('eviz.lib.autoviz.base.create_config')
@patch('eviz.lib.autoviz.base.get_factory_from_user_input')
def test_autoviz_initialization(mock_get_factory, mock_create_config):
    # Setup mocks
    mock_factory = MagicMock()
    mock_get_factory.return_value = [mock_factory]
    mock_config = MagicMock()
    mock_create_config.return_value = mock_config
    
    source_names = ["gridded"]
    autoviz = Autoviz(source_names=source_names)
    
    assert autoviz.source_names == source_names
    assert autoviz.args is not None
    mock_get_factory.assert_called_once_with(source_names)
    assert autoviz.factory_sources == [mock_factory]

@patch('eviz.lib.autoviz.base.create_config')
@patch('eviz.lib.autoviz.base.get_factory_from_user_input')
def test_autoviz_initialization_with_args(mock_get_factory, mock_create_config, basic_args):
    # Setup mocks
    mock_factory = MagicMock()
    mock_get_factory.return_value = [mock_factory]
    mock_config = MagicMock()
    mock_create_config.return_value = mock_config
    
    autoviz = Autoviz(source_names=["gridded"], args=basic_args)
    
    assert autoviz.args == basic_args
    mock_get_factory.assert_called_once_with(["gridded"])
    assert autoviz.factory_sources == [mock_factory]

@patch('eviz.lib.autoviz.base.get_factory_from_user_input')
def test_autoviz_initialization_no_factories(mock_get_factory):
    mock_get_factory.return_value = []
    with pytest.raises(ValueError):
        Autoviz(source_names=["gridded"])


# Tests for create_config
@patch('eviz.lib.autoviz.base.Config')
@patch('eviz.lib.autoviz.base.ConfigManager')
def test_create_config_with_config_file(mock_config_manager, mock_config, basic_args):
    # Setup
    basic_args.configfile = "test_config.yaml"
    basic_args.sources = ["gridded"]
    mock_config.return_value = MagicMock()
    
    # Execute
    create_config(basic_args)
    
    # Verify
    mock_config.assert_called_once()
    mock_config_manager.assert_called_once()

@patch('eviz.lib.autoviz.base.Config')
@patch('eviz.lib.autoviz.base.ConfigManager')
def test_create_config_with_config_dir(mock_config_manager, mock_config, basic_args):
    # Setup
    basic_args.config = ["/test/config/dir"]
    basic_args.configfile = None
    basic_args.sources = ["gridded"]
    mock_config.return_value = MagicMock()
    
    # Execute
    create_config(basic_args)
    
    # Verify
    mock_config.assert_called_once()
    mock_config_manager.assert_called_once()

@patch('eviz.lib.autoviz.base.Config')
@patch('eviz.lib.autoviz.base.ConfigManager')
@patch('eviz.lib.autoviz.base.get_config_path_from_env')
def test_create_config_with_env_path(mock_get_env_path, mock_config_manager, mock_config, basic_args, mock_env_path):
    # Setup
    basic_args.config = None
    basic_args.configfile = None
    basic_args.sources = ["gridded"]
    mock_get_env_path.return_value = mock_env_path
    mock_config.return_value = MagicMock()
    
    # Execute
    create_config(basic_args)
    
    # Verify
    mock_config.assert_called_once()
    mock_config_manager.assert_called_once()


def test_check_input_files_missing(mock_autoviz):
    with patch('os.path.exists', return_value=False):
        mock_autoviz._check_input_files()
        mock_autoviz.logger.warning.assert_called()

def test_check_input_files_exist(mock_autoviz):
    with patch('os.path.exists', return_value=True):
        mock_autoviz._check_input_files()
        mock_autoviz.logger.warning.assert_not_called()

def test_set_data(mock_autoviz):
    input_files = ["file1.nc", "file2.nc"]
    mock_autoviz.set_data(input_files)
    mock_autoviz._config_manager.input_config.set_input_files.assert_called_once_with(input_files)

def test_set_output(mock_autoviz):
    output_dir = "/path/to/output"
    mock_autoviz.set_output(output_dir)
    mock_autoviz._config_manager.output_config.set_output_dir.assert_called_once_with(output_dir)

@pytest.mark.skip(reason="Need to fix this test")
def test_autoviz_run_basic(mock_autoviz):
    # Setup
    mock_autoviz.config_adapter = MagicMock()
    mock_autoviz._config_manager._pipeline = MagicMock()
    mock_autoviz._config_manager._pipeline.get_all_data_sources.return_value = {"source1": "data1"}
    # mock_autoviz._check_input_files = MagicMock()

    # Create a mock model
    mock_model = MagicMock()
    mock_factory = MagicMock()
    mock_factory.create_root_instance.return_value = mock_model
    mock_autoviz.factory_sources = [mock_factory]
    
    # Execute
    mock_autoviz.run()
    
    # Verify
    mock_autoviz._check_input_files.assert_called_once()
    mock_autoviz.config_adapter.process_configuration.assert_called_once()
    mock_model.assert_called_once()

@pytest.mark.skip(reason="Need to fix this test")
def test_autoviz_run_with_composite(mock_autoviz):
    # Setup
    mock_autoviz.args = Namespace(
        sources=["gridded"],
        composite=["field1,field2,add"],
        integrate=False,
        compare=False,
        file=None,
        vars=None,
        configfile=None,
        config=None,
        data_dirs=None,
        output_dirs=None,
        verbose=1
    )
    
    mock_autoviz.config_adapter = MagicMock()
    mock_autoviz._config_manager._pipeline = MagicMock()
    mock_autoviz._config_manager._pipeline.get_all_data_sources.return_value = {"source1": "data1"}
    
    # Create a mock model
    mock_model = MagicMock()
    mock_factory = MagicMock()
    mock_factory.create_root_instance.return_value = mock_model
    mock_autoviz.factory_sources = [mock_factory]
    
    # Execute
    mock_autoviz.run()
    
    # Verify
    mock_autoviz._check_input_files.assert_called_once()
    mock_autoviz.config_adapter.process_configuration.assert_called_once()
    mock_model.plot_composite_field.assert_called_once_with("field1", "field2", "add")
