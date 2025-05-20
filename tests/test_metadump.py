import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import xarray as xr
import json

# Add parent directory to path to import metadump
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import metadump
from metadump import MetadumpConfig, MetadataExtractor


@pytest.fixture
def test_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_dataset():
    """Create a test dataset with 2D, 3D, and 4D variables."""
    times = pd.date_range('2020-01-01', periods=3)
    lats = np.linspace(-90, 90, 5)
    lons = np.linspace(-180, 180, 6)
    levs = np.array([1000, 850, 500, 200])

    # Create test data arrays
    data_2d = np.random.rand(5, 6)  # lat, lon
    data_3d_time = np.random.rand(3, 5, 6)  # time, lat, lon
    data_3d_lev = np.random.rand(5, 6, 4)  # lat, lon, lev
    data_4d = np.random.rand(3, 5, 6, 4)  # time, lat, lon, lev

    return xr.Dataset(
        data_vars={
            'var_2d': (['latitude', 'longitude'], data_2d,
                       {'units': 'K', 'long_name': '2D Variable'}),
            'var_3d_time': (['time', 'latitude', 'longitude'], data_3d_time,
                            {'units': 'm/s', 'long_name': '3D Time Variable'}),
            'var_3d_lev': (['latitude', 'longitude', 'level'], data_3d_lev,
                           {'units': 'Pa', 'long_name': '3D Level Variable'}),
            'var_4d': (['time', 'latitude', 'longitude', 'level'], data_4d,
                       {'units': 'kg/kg', 'long_name': '4D Variable'}),
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
            'level': levs
        },
        attrs={'title': 'Test Dataset', 'source': 'Unit Test'}
    )


@pytest.fixture
def test_files(test_dir, test_dataset):
    """Create test NetCDF files."""
    file1 = Path(test_dir) / 'test_data.nc'
    file2 = Path(test_dir) / 'test_data2.nc'

    # Save first dataset
    test_dataset.to_netcdf(file1)

    # Create and save second dataset (scaled by 1.1)
    ds2 = test_dataset.copy(deep=True)
    for var in ds2.data_vars:
        ds2[var] = ds2[var] * 1.1
    ds2.to_netcdf(file2)

    return {'file1': str(file1), 'file2': str(file2)}


@pytest.fixture
def mock_meta_coords():
    """Create mock meta_coords dictionary."""
    return {
        'tc': {'gridded': 'time'},
        'xc': {'gridded': 'longitude'},
        'yc': {'gridded': 'latitude'},
        'zc': {'gridded': 'level'}
    }


@pytest.fixture
def metadata_extractor(test_files, mock_meta_coords):
    """Create a MetadataExtractor instance with mocked meta_coords."""
    with patch('eviz.lib.utils.read_meta_coords', return_value=mock_meta_coords):
        config = MetadumpConfig(filepath_1=test_files['file1'])
        return MetadataExtractor(config)


def test_metadata_extractor_initialization(metadata_extractor):
    """Test initialization of MetadataExtractor."""
    assert metadata_extractor.dataset is not None
    assert metadata_extractor.dataset.dims['latitude'] == 5
    assert metadata_extractor.dataset.dims['longitude'] == 6
    assert metadata_extractor.dataset.dims['time'] == 3
    assert metadata_extractor.dataset.dims['level'] == 4

    assert metadata_extractor.tc == 'time'
    assert metadata_extractor.xc == 'longitude'
    assert metadata_extractor.yc == 'latitude'
    assert metadata_extractor.zc == 'level'
    assert metadata_extractor.space_coords == {'longitude', 'latitude'}


@pytest.mark.parametrize("config_params,expected_vars", [
    ({}, {'var_2d', 'var_3d_time', 'var_3d_lev', 'var_4d'}),
    ({'ignore_vars': ['var_2d', 'var_3d_time']}, {'var_3d_lev', 'var_4d'}),
    ({'vars': ['var_2d', 'var_4d']}, {'var_2d', 'var_4d'}),
])
def test_get_plottable_vars(test_files, mock_meta_coords, config_params, expected_vars):
    """Test getting plottable variables with different configurations."""
    with patch('eviz.lib.utils.read_meta_coords', return_value=mock_meta_coords):
        config = MetadumpConfig(filepath_1=test_files['file1'], **config_params)
        extractor = MetadataExtractor(config)
        assert set(extractor.get_plottable_vars()) == expected_vars


def test_process_variable(metadata_extractor):
    """Test processing variables."""
    # Test 2D variable
    var_2d_dict = metadata_extractor._process_variable(
        'var_2d', metadata_extractor.dataset['var_2d'])
    assert var_2d_dict['units'] == 'K'
    assert var_2d_dict['name'] == '2D Variable'
    assert 'xyplot' in var_2d_dict

    # Test 4D variable
    var_4d_dict = metadata_extractor._process_variable(
        'var_4d', metadata_extractor.dataset['var_4d'])
    assert var_4d_dict['units'] == 'kg/kg'
    assert var_4d_dict['name'] == '4D Variable'
    assert 'xyplot' in var_4d_dict
    assert 'yzplot' in var_4d_dict
    assert 'xtplot' in var_4d_dict


def test_generate_specs_dict(metadata_extractor):
    """Test generating specs dictionary."""
    specs_dict = metadata_extractor._generate_specs_dict()

    assert set(specs_dict.keys()) == {'var_2d', 'var_3d_time', 'var_3d_lev', 'var_4d'}

    var_4d_dict = specs_dict['var_4d']
    assert var_4d_dict['units'] == 'kg/kg'
    assert var_4d_dict['name'] == '4D Variable'
    assert all(key in var_4d_dict for key in ['xyplot', 'yzplot', 'xtplot'])


@pytest.mark.parametrize("with_second_file", [False, True])
def test_generate_app_dict(test_files, mock_meta_coords, with_second_file):
    """Test generating app dictionary with and without second file."""
    with patch('eviz.lib.utils.read_meta_coords', return_value=mock_meta_coords):
        config_params = {
            'filepath_1': test_files['file1'],
            'filepath_2': test_files['file2'] if with_second_file else None
        }
        config = MetadumpConfig(**config_params)
        extractor = MetadataExtractor(config)

        app_dict = extractor._generate_app_dict()

        assert len(app_dict['inputs']) == (2 if with_second_file else 1)
        assert app_dict['inputs'][0]['name'] == test_files['file1']
        assert 'to_plot' in app_dict['inputs'][0]

        if with_second_file:
            assert app_dict['inputs'][1]['name'] == test_files['file2']
            assert 'for_inputs' in app_dict
            assert 'compare' in app_dict['for_inputs']


def test_write_specs_yaml(test_dir, metadata_extractor):
    """Test writing specs YAML file."""
    specs_file = Path(test_dir) / 'test_specs.yaml'
    metadata_extractor.config.specs_output = str(specs_file)

    specs_dict = {
        'var_1': {'units': 'K', 'name': 'Temperature',
                  'xyplot': {'levels': {1000.0: []}}},
        'var_2': {'units': 'm/s', 'name': 'Wind Speed',
                  'xyplot': {'levels': {850.0: []}}}
    }

    metadata_extractor._write_specs_yaml(specs_dict)

    assert specs_file.exists()
    content = specs_file.read_text()
    assert all(text in content for text in
               ['var_1:', 'var_2:', 'units: K', 'units: m/s'])


def test_write_app_yaml(test_dir, metadata_extractor):
    """Test writing app YAML file."""
    app_file = Path(test_dir) / 'test_app.yaml'
    metadata_extractor.config.app_output = str(app_file)

    app_dict = {
        'inputs': [{
            'name': metadata_extractor.config.filepath_1,
            'to_plot': {'var_1': 'xy', 'var_2': 'xy,yz'}
        }],
        'outputs': {'print_to_file': 'yes', 'output_dir': None}
    }

    metadata_extractor._write_app_yaml(app_dict)

    assert app_file.exists()
    content = app_file.read_text()
    assert all(text in content for text in
               ['inputs:', 'to_plot:', 'print_to_file: yes'])


def test_generate_json_metadata(test_dir, metadata_extractor):
    """Test generating JSON metadata."""
    json_file = Path(test_dir) / 'test_metadata.json'
    metadata_extractor.config.json_output = str(json_file)

    metadata_extractor._generate_json_metadata()

    assert json_file.exists()
    with json_file.open() as f:
        metadata = json.load(f)

    assert 'global_attributes' in metadata
    assert 'variables' in metadata
    assert metadata['global_attributes']['title'] == 'Test Dataset'
    assert 'var_2d' in metadata['variables']
    assert metadata['variables']['var_2d']['attributes']['units'] == 'K'


@pytest.mark.parametrize("var_name,dims,expected", [
    ('var_2d', ['lat', 'lon'], True),
    ('var_3d_time', ['time', 'lat', 'lon'], True),
    ('var_3d_lev', ['lat', 'lon', 'lev'], True),
    ('var_4d', ['time', 'lat', 'lon', 'lev'], True),
    ('var_1d', ['time'], False),
])
def test_is_plottable(var_name, dims, expected):
    """Test is_plottable function with different variable configurations."""
    ds = xr.Dataset(
        data_vars={
            var_name: (dims, np.random.rand(*[2] * len(dims)))
        }
    )

    assert metadump.is_plottable(
        ds, var_name, {'lat', 'lon'}, 'lev', 'time') == expected


@pytest.mark.parametrize("var_name,shape,expected", [
    ('var_single_time', (1, 3, 4), False),
    ('var_multi_time', (3, 3, 4), True),
    ('var_no_time', (3, 4), False),
])
def test_has_multiple_time_levels(var_name, shape, expected):
    """Test has_multiple_time_levels function."""
    dims = ['time', 'lat', 'lon'] if len(shape) == 3 else ['lat', 'lon']
    ds = xr.Dataset(
        data_vars={
            var_name: (dims, np.random.rand(*shape))
        }
    )

    assert metadump.has_multiple_time_levels(ds, var_name, 'time') == expected


def test_get_model_dim_name():
    """Test get_model_dim_name function."""
    dims = ['time', 'latitude', 'longitude', 'level']
    meta_coords = {
        'tc': {'gridded': 'time', 'wrf': 'Time'},
        'xc': {'gridded': 'longitude', 'wrf': 'west_east'},
        'yc': {'gridded': 'latitude', 'wrf': 'south_north'},
        'zc': {'gridded': 'level', 'wrf': 'bottom_top'}
    }

    # Test with gridded source
    assert metadump.get_model_dim_name(dims, 'tc', meta_coords, 'gridded') == 'time'
    assert metadump.get_model_dim_name(dims, 'xc', meta_coords, 'gridded') == 'longitude'

    # Test with non-matching source
    assert metadump.get_model_dim_name(dims, 'xc', meta_coords, 'wrf') != 'longitude'

    # Test with non-existent dimension
    assert metadump.get_model_dim_name(dims, 'non_existent', meta_coords, 'gridded') is None


@pytest.mark.parametrize("test_input,expected", [
    (np.float32(1.5), 1.5),
    (np.int32(5), 5),
    (np.array([1, 2, 3]), [1, 2, 3]),
    ({'a': np.float64(1.5), 'b': np.array([4, 5, 6])},
     {'a': 1.5, 'b': [4, 5, 6]}),
])
def test_json_compatible(test_input, expected):
    """Test json_compatible function with different input types."""
    assert metadump.json_compatible(test_input) == expected


def test_command_line_parsing():
    """Test command line argument parsing."""
    test_args = ['test.nc', '--specs', 'test_specs.yaml',
                 '--app', 'test_app.yaml', '--ignore', 'var_1']

    with patch('sys.argv', ['metadump.py'] + test_args):
        args = metadump.parse_command_line()

        assert args.filepaths == ['test.nc']
        assert args.specs == 'test_specs.yaml'
        assert args.app == 'test_app.yaml'
        assert args.ignore == ['var_1']
        assert args.source == 'gridded'  # default value


def test_main_function(test_files):
    """Test main function execution."""
    with patch('metadump.parse_command_line') as mock_parse_args:
        mock_parse_args.return_value = MagicMock(
            filepaths=[test_files['file1']],
            specs='test_specs.yaml',
            app='test_app.yaml',
            json=None,
            ignore=['var_1'],
            vars=None,
            source='gridded'
        )

        with patch('metadump.MetadataExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_extractor.return_value = mock_instance

            metadump.main()

            mock_extractor.assert_called_once()
            config = mock_extractor.call_args[0][0]
            assert config.filepath_1 == test_files['file1']
            assert config.specs_output == 'test_specs.yaml'
            assert config.app_output == 'test_app.yaml'
            assert config.ignore_vars == ['var_1']

            mock_instance.process.assert_called_once()
