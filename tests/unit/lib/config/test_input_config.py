import pytest
import os
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Assuming AppData is defined in eviz.lib.config.app_data
# For testing purposes, we can define a minimal mock or use the actual if simple
try:
    from eviz.lib.config.app_data import AppData
except ImportError:
    @dataclass
    class AppData:
        inputs: List[Dict[str, Any]] = field(default_factory=list)
        for_inputs: Dict[str, Any] = field(default_factory=dict)
        # Add other fields if InputConfig directly accesses them during init
        # or if they are needed for the methods being tested.

from eviz.lib.config.input_config import InputConfig

@pytest.fixture
def minimal_app_data():
    """Provides a minimal AppData instance."""
    return AppData(inputs=[], for_inputs={})

@pytest.fixture
def app_data_with_inputs():
    """Provides AppData with some sample inputs."""
    return AppData(
        inputs=[
            {'name': 'file1.nc', 'location': '/path/to'},
            {'name': 'file2.csv', 'location': '/data'},
            {'name': 'wrfout_d01.nc', 'location': '/model/run'},
            {'name': 'archive.tar.gz.nc', 'location': '/complex/path'}, # Test complex names
            {'name': 'data.he5', 'location': '/hdf/files'},
            {'name': 'old_data.hdf', 'location': '/hdf/files'},
            {'name': 'wildcard_*.nc', 'location': '/wildcard/path'},
            {'name': 'explicit_reader.dat', 'location': '/custom', 'reader': 'CustomReader'},
        ],
        for_inputs={}
    )

@pytest.fixture
def mock_data_source_factory():
    """Mocks the DataSourceFactory and its created readers."""
    with patch('eviz.lib.config.input_config.DataSourceFactory') as mock_factory_class:
        mock_factory_instance = mock_factory_class.return_value
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.get_variables.return_value = {'var1': {'units': 'K'}}
        mock_reader_instance.get_metadata.return_value = {'global_attr': 'test_value'}

        mock_factory_instance.create_data_source.return_value = mock_reader_instance
        
        yield mock_factory_instance


# --- Test Cases ---

def test_input_config_initialization_defaults(minimal_app_data):
    config = InputConfig(source_names=['source1'], config_files=['conf.yaml'], app_data=minimal_app_data)
    assert config.source_names == ['source1']
    assert config.config_files == ['conf.yaml']
    assert isinstance(config.app_data, AppData)
    assert config.file_list == {}
    assert config.readers == {}
    assert not config.compare
    assert not config.compare_diff
    assert config._cmap == "rainbow" 
    assert config._comp_panels == (1,1) 
    assert not config._use_cartopy 


def test_input_config_initialize_calls_internal_methods(minimal_app_data):
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=minimal_app_data)
    with patch.object(config, '_get_file_list') as mock_get_files, \
         patch.object(config, '_init_file_list_to_plot') as mock_init_plot_list, \
         patch.object(config, '_init_readers') as mock_init_readers, \
         patch.object(config, '_init_for_inputs') as mock_init_for_inputs:
        
        config.initialize()
        
        mock_get_files.assert_called_once()
        mock_init_plot_list.assert_called_once()
        mock_init_readers.assert_called_once()
        mock_init_for_inputs.assert_called_once()

def test_get_file_list_success(app_data_with_inputs):
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=app_data_with_inputs)
    config._get_file_list() 

    assert len(config.file_list) == len(app_data_with_inputs.inputs)
    assert config.file_list[0]['name'] == 'file1.nc'
    assert config.file_list[0]['filename'] == os.path.join('/path/to', 'file1.nc')
    assert config.file_list[1]['name'] == 'file2.csv'
    assert config.file_list[1]['filename'] == os.path.join('/data', 'file2.csv')


def test_get_file_list_no_inputs_logs_error_and_exits(minimal_app_data, caplog):
    buggy_app_data = MagicMock()
    buggy_app_data.inputs = None 
    
    config_with_buggy_data = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=buggy_app_data)

    with pytest.raises(SystemExit):
        with patch('eviz.lib.config.input_config.logging.getLogger') as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance
            config_with_buggy_data._get_file_list() 
            mock_logger_instance.error.assert_called_once_with("The 'inputs' section in the AppData object is empty or missing.")


@pytest.mark.parametrize("file_path, explicit_reader, expected_type", [
    ("test.nc", None, "NetCDF"),
    ("test.nc4", None, "NetCDF"),
    ("test.csv", None, "CSV"),
    ("test.dat", None, "CSV"),
    ("test.h5", None, "HDF5"),
    ("test.he5", None, "HDF5"),
    ("test.hdf", None, "HDF4"),
    ("wrfout_d01_2023-10-26", None, "NetCDF"),
    ("path/to/wrf.output", None, "NetCDF"),
    ("something.wrf-arw", None, "NetCDF"),
    ("data_with_netcdf_in_name.txt", None, "NetCDF"), 
    ("data_with_csv_in_name.bin", None, "CSV"),
    ("data_with_hdf5_in_name.dat", None, "HDF5"),
    ("data_with_hdf4_in_name.log", None, "HDF4"),
    ("unknown.ext", None, "NetCDF"), 
    ("file_with_no_ext", None, "NetCDF"), 
    ("path/to/wildcard_*.nc", None, "NetCDF"), 
    ("custom.data", "MyReader", "MyReader"), 
    ("custom.nc", "ExplicitNetcdf", "ExplicitNetcdf"), 
])
def test_get_reader_type_for_extension(minimal_app_data, file_path, explicit_reader, expected_type):
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=minimal_app_data)
    with patch('eviz.lib.config.input_config.logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        reader_type = config._get_reader_type_for_extension(file_path, explicit_reader)
        assert reader_type == expected_type


def test_init_readers_success(app_data_with_inputs, mock_data_source_factory):
    config = InputConfig(source_names=['source1', 'source2'], config_files=['c1.yaml'], app_data=app_data_with_inputs)
    config._get_file_list()
    
    # Mock the _create_data_source method
    with patch('eviz.lib.config.input_config.InputConfig._create_data_source') as mock_create_data_source:
        # Create mock readers
        mock_netcdf_reader = MagicMock(name="NetCDFReader")
        mock_csv_reader = MagicMock(name="CSVReader")
        
        # Set up the mock to return appropriate readers
        def side_effect(file_path, source_name, reader_type=None):
            if reader_type == 'NetCDF':
                return mock_netcdf_reader
            elif reader_type == 'CSV':
                return mock_csv_reader
            return MagicMock()
            
        mock_create_data_source.side_effect = side_effect
        
        # Mock the logger
        with patch('eviz.lib.config.input_config.logging.getLogger') as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance
            
            # Initialize readers
            config._init_readers()
    
    # Check that readers were initialized correctly
    assert 'source1' in config.readers
    assert 'NetCDF' in config.readers['source1']
    assert config.readers['source1']['NetCDF'] is mock_netcdf_reader


def test_init_for_inputs_compare_diff(mock_data_source_factory):
    app_data = AppData(
        inputs=[{'name': 'f1.nc', 'location': '.'}, {'name': 'f2.nc', 'location': '.'}],
        for_inputs={
            'compare_diff': {
                'ids': 'exp1,exp2',
                'extra_diff': True,
                'profile': True,
                'cmap': 'coolwarm'
            },
            'use_cartopy': True,
            'subplot_specs': (2,2)
        }
    )
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=app_data)
    config.compare_diff = True 
    
    with patch('eviz.lib.config.input_config.logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        config.initialize() 

    assert config._compare_diff  
    assert config._compare_exp_ids == ['exp1', 'exp2']
    assert config._extra_diff_plot
    assert config._profile
    assert config._cmap == 'coolwarm'
    assert config._comp_panels == (2, 2) 
    assert config._use_cartopy
    assert config._subplot_specs == (2,2)

def test_init_for_inputs_compare(mock_data_source_factory):
    app_data = AppData(
        inputs=[{'name': 'f1.nc', 'location': '.'}, {'name': 'f2.nc', 'location': '.'}, {'name': 'f3.nc', 'location': '.'}],
        for_inputs={
            'compare': {
                'ids': 'runA,runB,runC',
                'profile': False,
                'cmap': 'viridis'
            }
        }
    )
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=app_data)
    config.compare = True 
    
    with patch('eviz.lib.config.input_config.get_subplot_shape', return_value=(1,3)) as mock_get_shape, \
         patch('eviz.lib.config.input_config.logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        config.initialize()

    assert config._compare
    assert config._compare_exp_ids == ['runA', 'runB', 'runC']
    assert not config._profile
    assert config._cmap == 'viridis'
    mock_get_shape.assert_called_once_with(3) 
    assert config._comp_panels == (1,3)


def test_get_primary_reader(app_data_with_inputs, mock_data_source_factory):
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=app_data_with_inputs)
    config._get_file_list()

    netcdf_reader = MagicMock(name="NetCDFReader")
    csv_reader = MagicMock(name="CSVReader")
    
    # Mock the _create_data_source method
    with patch('eviz.lib.config.input_config.InputConfig._create_data_source') as mock_create_data_source:
        def side_effect(file_path, source_name, reader_type=None):
            if source_name == 's1':
                if reader_type == 'NetCDF': return netcdf_reader
                if reader_type == 'CSV': return csv_reader
            return MagicMock()
            
        mock_create_data_source.side_effect = side_effect
        
        config.readers = {}
        with patch('eviz.lib.config.input_config.logging.getLogger') as mock_get_logger:
            mock_get_logger.return_value = MagicMock()
            config._init_readers()

    primary_reader_s1 = config.get_primary_reader('s1')
    assert primary_reader_s1 is netcdf_reader


def test_set_trop_height_file_list(mock_data_source_factory):
    app_data = AppData(
        inputs=[],
        for_inputs={
            'trop_height': [
                {'location': '/path/trop', 'name': 'trop1.nc', 'exp_id': 'expA', 'trop_field_name': 'TROPH'},
                {'name': '/abs/path/trop2.nc', 'exp_id': 'expB', 'trop_field_name': 'TPHGT'} 
            ]
        }
    )
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=app_data)
    
    with patch('eviz.lib.config.input_config.logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        config._init_for_inputs() 

    assert config._use_trop_height
    assert len(config.trop_height_file_list) == 2
    assert config.trop_height_file_list[0]['filename'] == '/path/trop/trop1.nc'
    assert config.trop_height_file_list[0]['exp_name'] == 'expA'
    assert config.trop_height_file_list[0]['trop_field_name'] == 'TROPH'
    assert config.trop_height_file_list[1]['filename'] == '/abs/path/trop2.nc' 
    assert config.trop_height_file_list[1]['exp_name'] == 'expB'


def test_to_dict_serialization(app_data_with_inputs, mock_data_source_factory):
    config = InputConfig(source_names=['s1'], config_files=['conf.yaml'], app_data=app_data_with_inputs)
    
    # Mock the _create_data_source method
    with patch('eviz.lib.config.input_config.InputConfig._create_data_source') as mock_create_data_source:
        # Create mock readers
        mock_netcdf_reader = MagicMock(name="NetCDFReader")
        mock_csv_reader = MagicMock(name="CSVReader")
        
        # Set up the mock to return appropriate readers
        def side_effect(file_path, source_name, reader_type=None):
            if reader_type == 'NetCDF':
                return mock_netcdf_reader
            elif reader_type == 'CSV':
                return mock_csv_reader
            return MagicMock()
            
        mock_create_data_source.side_effect = side_effect
        
        with patch('eviz.lib.config.input_config.logging.getLogger') as mock_get_logger:
            mock_get_logger.return_value = MagicMock()
            config.initialize()

    config._compare_exp_ids = ['exp1']
    config._extra_diff_plot = True
    config._profile = False
    config._cmap = "custom_map"
    config._comp_panels = (2,1)
    config._use_trop_height = True
    config._subplot_specs = (1,1)
    config._use_cartopy = True

    result_dict = config.to_dict()

    assert result_dict['source_names'] == ['s1']
    assert result_dict['config_files'] == ['conf.yaml']
    assert result_dict['app_data'] == app_data_with_inputs.__dict__
    assert len(result_dict['file_list']) == len(app_data_with_inputs.inputs)
    assert 'NetCDF' in result_dict['readers']['s1']
