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

def test_get_format_for_file_returns_format(app_data_with_inputs):
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=app_data_with_inputs)
    config._file_format_mapping['/path/to/file1.nc'] = 'netcdf'
    assert config.get_format_for_file('/path/to/file1.nc') == 'netcdf'
    assert config.get_format_for_file('/not/found.nc') is None

def test_get_reader_for_file_and_fallback(app_data_with_inputs, mock_data_source_factory):
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=app_data_with_inputs)
    config._get_file_list()
    config.readers = {'s1': {'NetCDF': 'reader1', 'CSV': 'reader2'}}
    config._file_reader_mapping = {'/path/to/file1.nc': 'NetCDF', '/data/file2.csv': 'CSV'}
    assert config.get_reader_for_file('s1', '/path/to/file1.nc') == 'reader1'
    assert config.get_reader_for_file('s1', '/data/file2.csv') == 'reader2'
    # fallback to first reader if mapping not found
    assert config.get_reader_for_file('s1', '/not/found.nc') == 'reader1'
    # fallback to None if no readers
    config.readers = {'s1': {}}
    assert config.get_reader_for_file('s1', '/not/found.nc') is None

def test_get_all_variables_merges_vars(app_data_with_inputs, mock_data_source_factory):
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=app_data_with_inputs)
    mock_reader1 = MagicMock()
    mock_reader1.get_variables.return_value = {'a': {'units': 'K'}}
    mock_reader2 = MagicMock()
    mock_reader2.get_variables.return_value = {'b': {'units': 'Pa'}}
    config.readers = {'s1': {'NetCDF': mock_reader1, 'CSV': mock_reader2}}
    all_vars = config.get_all_variables('s1')
    assert 'a' in all_vars and 'b' in all_vars
    assert all_vars['a']['source_reader'] == 'NetCDF'
    assert all_vars['b']['source_reader'] == 'CSV'

def test_get_metadata_success_and_error(app_data_with_inputs, mock_data_source_factory):
    config = InputConfig(source_names=['s1'], config_files=['c1.yaml'], app_data=app_data_with_inputs)
    mock_reader = MagicMock()
    mock_reader.get_metadata.return_value = {'meta': 1}
    config.readers = {'s1': {'NetCDF': mock_reader}}
    config._file_reader_mapping = {'/path/to/file1.nc': 'NetCDF'}
    assert config.get_metadata('s1', '/path/to/file1.nc') == {'meta': 1}
    # Simulate error
    mock_reader.get_metadata.side_effect = Exception('fail')
    assert config.get_metadata('s1', '/path/to/file1.nc') == {}

def test__init_reader_structure_creates_dicts():
    config = InputConfig(source_names=['A', 'B'], config_files=['c.yaml'])
    config._init_reader_structure()
    assert 'A' in config.readers and 'B' in config.readers
    assert config.readers['A'] == {} and config.readers['B'] == {}

def test__determine_reader_types_groups_by_source(app_data_with_inputs):
    config = InputConfig(source_names=['A', 'B'], config_files=['c.yaml'], app_data=app_data_with_inputs)
    config._get_file_list()
    result = config._determine_reader_types()
    assert 'A' in result and 'B' in result
    assert isinstance(result['A'], list)

def test__get_reader_type_from_format_warns(caplog):
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    with caplog.at_level('WARNING'):
        assert config._get_reader_type_from_format('unknown') == 'NetCDF'
        assert "Unknown format" in caplog.text

def test__get_reader_handles_wrf_fallback(monkeypatch):
    config = InputConfig(source_names=['wrf'], config_files=['c.yaml'])
    # Patch _create_data_source to raise ValueError first, then succeed
    called = {}
    def fake_create(file_path, source_name, reader_type=None):
        if file_path == 'dummy.nc':
            called['ok'] = True
            return 'fallback'
        raise ValueError('fail')
    config._create_data_source = fake_create
    assert config._get_reader('wrf', '.bad', 'BadType') == 'fallback'
    # Now test non-wrf raises
    config = InputConfig(source_names=['foo'], config_files=['c.yaml'])
    config._create_data_source = lambda *a, **k: (_ for _ in ()).throw(ValueError('fail'))
    with pytest.raises(ValueError):
        config._get_reader('foo', '.bad', 'BadType')

def test__set_sphum_conv_file_list_sets_and_unsets():
    app_data = AppData(inputs=[], for_inputs={'sphum_field': [
        {'location': '/a', 'name': 's1.nc', 'exp_id': 'e1', 'sphum_field_name': 'Q'}
    ]})
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data)
    config.use_sphum_conv = True
    config._set_sphum_conv_file_list()
    assert config.sphum_conv_file_list[0]['filename'] == '/a/s1.nc'
    # Now test disables if not present
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=AppData(inputs=[], for_inputs={}))
    config.use_sphum_conv = True
    config._set_sphum_conv_file_list()
    assert not config.use_sphum_conv

def test_properties_trop_and_sphum_lists():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    config._trop_height_file_list = {1: 'a'}
    config._sphum_conv_file_list = {2: 'b'}
    assert config.trop_height_file_list == {1: 'a'}
    assert config.sphum_conv_file_list == {2: 'b'}

# --- Additional Coverage Tests ---
def test__init_file_list_to_plot_history(monkeypatch, minimal_app_data):
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=minimal_app_data)
    config._use_history = True
    config._history_year = None
    config._history_month = None
    with pytest.raises(SystemExit):
        config._init_file_list_to_plot()

def test__init_file_list_to_plot_single_file(app_data_with_inputs):
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data_with_inputs)
    config.file_list = {0: app_data_with_inputs.inputs[0]}
    config._init_file_list_to_plot()
    assert not config.compare
    assert not config.compare_diff

def test__get_reader_type_for_extension_various_cases():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    # Test wrf in path
    assert config._get_reader_type_for_extension('foo/wrfout_d01', None) == 'NetCDF'
    # Test hdf5
    assert config._get_reader_type_for_extension('foo/file.h5', None) == 'HDF5'
    # Test hdf4
    assert config._get_reader_type_for_extension('foo/file.hdf', None) == 'HDF4'
    # Test zarr
    assert config._get_reader_type_for_extension('foo/file.zarr', None) == 'ZARR'
    # Test grib
    assert config._get_reader_type_for_extension('foo/file.grib2', None) == 'GRIB'
    # Test fallback
    assert config._get_reader_type_for_extension('foo/file.unknown', None) == 'NetCDF'
    # Test explicit reader
    assert config._get_reader_type_for_extension('foo/file.nc', 'Explicit') == 'Explicit'

def test__parse_for_inputs_all_branches():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    # Correlation
    for_inputs = {'correlation': {'ids': 'a,b', 'method': 'spearman', 'space_corr': True, 'time_corr': True}}
    config._parse_for_inputs(for_inputs)
    assert config.correlation
    assert config.compare_exp_ids == ['a', 'b']
    assert config.correlation_method == 'spearman'
    assert config.space_corr
    assert config.time_corr
    # Overlay
    for_inputs = {'overlay': {'ids': 'x,y', 'box_colors': 'red,blue', 'add_legend': True}}
    config._parse_for_inputs(for_inputs)
    assert config.overlay
    assert config.overlay_exp_ids == ['x', 'y']
    assert config.box_colors == ['red', 'blue']
    assert config.add_legend
    # Compare
    for_inputs = {'compare': {'ids': 'foo,bar'}}
    config._parse_for_inputs(for_inputs)
    assert config.compare
    assert config.compare_exp_ids == ['foo', 'bar']
    # Compare diff
    for_inputs = {'compare_diff': {'ids': 'baz,qux'}}
    config._parse_for_inputs(for_inputs)
    assert config.compare_diff
    assert config.compare_exp_ids == ['baz', 'qux']

def test_to_dict_includes_all_fields(app_data_with_inputs):
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data_with_inputs)
    config.readers = {'A': {'NetCDF': 'reader'}}
    config.compare = True
    config.compare_diff = False
    config.overlay_exp_ids = ['id1']
    config.compare_exp_ids = ['id2']
    config.extra_diff_plot = True
    config.profile = True
    config.cmap = 'cmap'
    config.comp_panels = (2, 2)
    config.use_trop_height = True
    config.subplot_specs = (1, 2)
    config.use_cartopy = True
    config._file_format_mapping = {'/path/to/file1.nc': 'netcdf'}
    d = config.to_dict()
    assert d['source_names'] == ['A']
    assert d['compare']
    assert d['overlay_exp_ids'] == ['id1']
    assert d['compare_exp_ids'] == ['id2']
    assert d['extra_diff_plot']
    assert d['profile']
    assert d['cmap'] == 'cmap'
    assert d['comp_panels'] == (2, 2)
    assert d['use_trop_height']
    assert d['subplot_specs'] == (1, 2)
    assert d['use_cartopy']
    assert d['file_formats'] == {'/path/to/file1.nc': 'netcdf'}

def test__set_trop_height_file_list_handles_empty():
    app_data = AppData(inputs=[], for_inputs={'trop_height': []})
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data)
    config.use_trop_height = True
    config._set_trop_height_file_list()
    assert not config.use_trop_height

def test__set_trop_height_file_list_handles_entries():
    app_data = AppData(inputs=[], for_inputs={'trop_height': [
        {'location': '/a', 'name': 't1.nc', 'exp_id': 'e1', 'trop_field_name': 'T1'},
        {'name': '/b/t2.nc', 'exp_id': 'e2', 'trop_field_name': 'T2'}
    ]})
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data)
    config._set_trop_height_file_list()
    assert config._trop_height_file_list[0]['filename'] == '/a/t1.nc'
    assert config._trop_height_file_list[1]['filename'] == '/b/t2.nc'

def test_get_primary_reader_prefers_netcdf(app_data_with_inputs, mock_data_source_factory):
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data_with_inputs)
    config.readers = {'A': {'NetCDF': 'reader', 'CSV': 'csvreader'}}
    assert config.get_primary_reader('A') == 'reader'
    config.readers = {'A': {'CSV': 'csvreader'}}
    assert config.get_primary_reader('A') == 'csvreader'
    config.readers = {'A': {}}
    assert config.get_primary_reader('A') is None

def test_get_all_variables_handles_exception():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    bad_reader = MagicMock()
    bad_reader.get_variables.side_effect = Exception('fail')
    config.readers = {'A': {'NetCDF': bad_reader}}
    # Should not raise
    assert config.get_all_variables('A') == {}

def test_get_metadata_reader_none():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    config._file_reader_mapping = {'/file.nc': 'NetCDF'}
    config.readers = {'A': {}}
    assert config.get_metadata('A', '/file.nc') == {}

def test_get_metadata_reader_not_found():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    config._file_reader_mapping = {}
    config.readers = {'A': {}}
    assert config.get_metadata('A', '/file.nc') == {}

def test__get_reader_type_from_format_all_types():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    assert config._get_reader_type_from_format('netcdf') == 'NetCDF'
    assert config._get_reader_type_from_format('csv') == 'CSV'
    assert config._get_reader_type_from_format('hdf5') == 'HDF5'
    assert config._get_reader_type_from_format('hdf4') == 'HDF4'
    assert config._get_reader_type_from_format('zarr') == 'ZARR'
    assert config._get_reader_type_from_format('grib') == 'GRIB'
    # Unknown triggers warning and returns NetCDF
    assert config._get_reader_type_from_format('unknown') == 'NetCDF'

def test__get_reader_handles_valueerror(monkeypatch):
    config = InputConfig(source_names=['wrf'], config_files=['c.yaml'])
    def raise_valueerror(*a, **k):
        raise ValueError('fail')
    config._create_data_source = raise_valueerror
    with pytest.raises(ValueError):
        config._get_reader('notwrf', '.bad', 'BadType')

def test__init_for_inputs_overlay_compare_logic():
    app_data = AppData(inputs=[], for_inputs={'overlay': {'ids': 'a,b'}, 'compare': {'ids': 'a,b'}})
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data)
    config.overlay = True
    config.compare = True
    config.compare_diff = True
    config._init_for_inputs()
    assert not config.compare
    assert not config.compare_diff

def test__init_for_inputs_compare_diff_extra_fields():
    app_data = AppData(inputs=[], for_inputs={'compare_diff': {'ids': 'a,b', 'extra_diff_plot': True, 'profile': True, 'cmap': 'hot'}})
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data)
    config.compare_diff = True
    config._init_for_inputs()
    assert config.extra_diff_plot
    assert config.profile
    assert config.cmap == 'hot'

# --- New tests for additional coverage ---
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

    config.compare_exp_ids = ['exp1']
    config.extra_diff_plot = True
    config.profile = False
    config.cmap = "custom_map"
    config.comp_panels = (2,1)
    config.use_trop_height = True
    config.subplot_specs = (1,1)
    config.use_cartopy = True

    result_dict = config.to_dict()

    assert result_dict['source_names'] == ['s1']
    assert result_dict['config_files'] == ['conf.yaml']
    assert result_dict['app_data'] == app_data_with_inputs.__dict__
    assert len(result_dict['file_list']) == len(app_data_with_inputs.inputs)
    assert 'NetCDF' in result_dict['readers']['s1']

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

    assert config.use_trop_height
    assert len(config.trop_height_file_list) == 2
    assert config.trop_height_file_list[0]['filename'] == '/path/trop/trop1.nc'
    assert config.trop_height_file_list[0]['exp_name'] == 'expA'
    assert config.trop_height_file_list[0]['trop_field_name'] == 'TROPH'
    assert config.trop_height_file_list[1]['filename'] == '/abs/path/trop2.nc' 
    assert config.trop_height_file_list[1]['exp_name'] == 'expB'

# --- End new tests ---

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

# --- Extra edge case: _get_file_list logs error and exits if no inputs ---
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

def test_get_reader_for_file_no_reader_and_no_fallback():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    config.readers = {'A': {}}
    config._file_reader_mapping = {}
    assert config.get_reader_for_file('A', '/notfound.nc') is None

def test_get_primary_reader_empty():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    config.readers = {'A': {}}
    assert config.get_primary_reader('A') is None
    config.readers = {}
    assert config.get_primary_reader('A') is None

def test__get_reader_type_for_extension_empty_and_weird_cases():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    # No extension
    assert config._get_reader_type_for_extension('file', None) == 'NetCDF'
    # .wrf extension
    assert config._get_reader_type_for_extension('foo/file.wrf', None) == 'NetCDF'
    # .nc4 extension
    assert config._get_reader_type_for_extension('foo/file.nc4', None) == 'NetCDF'
    # .dat extension
    assert config._get_reader_type_for_extension('foo/file.dat', None) == 'CSV'
    # .txt extension
    assert config._get_reader_type_for_extension('foo/file.txt', None) == 'CSV'
    # .hdf extension with .h5 in name
    assert config._get_reader_type_for_extension('foo/file.hdf5.hdf', None) == 'HDF5'
    # .hdf extension with .he5 in name
    assert config._get_reader_type_for_extension('foo/file.he5.hdf', None) == 'HDF5'
    # .hdf extension with no .h5 or .he5
    assert config._get_reader_type_for_extension('foo/file.hdf', None) == 'HDF4'
    # .hdf4 in name
    assert config._get_reader_type_for_extension('foo/file.hdf4', None) == 'HDF4'
    # .grib in name
    assert config._get_reader_type_for_extension('foo/file.grib', None) == 'GRIB'
    # .grib2 in name
    assert config._get_reader_type_for_extension('foo/file.grib2', None) == 'GRIB'
    # .zarr in name
    assert config._get_reader_type_for_extension('foo/file.zarr', None) == 'ZARR'
    # netcdf in name
    assert config._get_reader_type_for_extension('foo/netcdf_file', None) == 'NetCDF'

def test__get_reader_type_for_extension_case_insensitivity():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    # Should be case-insensitive
    assert config._get_reader_type_for_extension('FOO/WRFOUT_D01', None) == 'NetCDF'
    assert config._get_reader_type_for_extension('foo/FILE.H5', None) == 'HDF5'
    assert config._get_reader_type_for_extension('foo/FILE.HDF', None) == 'HDF4'
    assert config._get_reader_type_for_extension('foo/FILE.ZARR', None) == 'ZARR'
    assert config._get_reader_type_for_extension('foo/FILE.GRIB2', None) == 'GRIB'
    assert config._get_reader_type_for_extension('foo/FILE.UNKNOWN', None) == 'NetCDF'

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

def test_init_readers_empty_file_list():
    config = InputConfig(source_names=['A'], config_files=['c.yaml'])
    config.file_list = {}
    config._init_readers()  # Should not fail or throw
    assert 'A' in config.readers
    assert config.readers['A'] == {}

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

    assert config.compare_diff  
    assert config.compare_exp_ids == ['exp1', 'exp2']
    assert config.extra_diff_plot
    assert config.profile
    assert config.cmap == 'coolwarm'
    assert config.comp_panels == (2, 2) 
    assert config.use_cartopy
    assert config.subplot_specs == (2,2)

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

    assert config.compare
    assert config.compare_exp_ids == ['runA', 'runB', 'runC']
    assert not config.profile
    assert config.cmap == 'viridis'
    mock_get_shape.assert_called_once_with(3) 
    assert config.comp_panels == (1,3)

def test_init_for_inputs_compare_and_overlay_priority():
    app_data = AppData(inputs=[], for_inputs={'overlay': {'ids': 'a,b'}, 'compare': {'ids': 'a,b'}})
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data)
    config.overlay = True
    config.compare = True
    config.compare_diff = True
    config._init_for_inputs()
    assert not config.compare
    assert not config.compare_diff


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

# --- Edge case: test _set_sphum_conv_file_list disables if not present ---
def test_set_sphum_conv_file_list_disables():
    app_data = AppData(inputs=[], for_inputs={})
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data)
    config.use_sphum_conv = True
    config._set_sphum_conv_file_list()
    assert not config.use_sphum_conv

# --- Edge case: test _set_trop_height_file_list disables if not present ---
def test_set_trop_height_file_list_disables():
    app_data = AppData(inputs=[], for_inputs={'trop_height': []})
    config = InputConfig(source_names=['A'], config_files=['c.yaml'], app_data=app_data)
    config.use_trop_height = True
    config._set_trop_height_file_list()
    assert not config.use_trop_height
