import os
import pytest
import eviz.lib.const as constants
import eviz.lib.utils as u
from eviz.lib.autoviz.config import Config


def test_config_type(get_config_instance):
    config = get_config_instance
    assert isinstance(config, Config)


def test_config_source_names_len(get_config_instance):
    config = get_config_instance
    assert len(config.source_names) == 1


def test_source_name(get_config_instance):
    config = get_config_instance
    assert config.source_names[0] == 'test'


def test_config_file_path(get_config_instance):
    config = get_config_instance
    assert os.path.basename(config.config_files[0]) == 'test.yaml'


def test_load_meta_attrs_file():
    top = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    file = os.path.join(top, 'config', 'meta_coordinates.yaml')
    assert isinstance(u.load_yaml(file), dict)


def test_load_meta_coords_file():
    top = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    file = os.path.join(top, 'config', 'meta_coordinates.yaml')
    assert isinstance(u.load_yaml(file), dict)


# Already in test_utils?
def test_load_yaml(get_config_instance):
    config = get_config_instance
    u.load_yaml(config.config_files[0])
    assert config.app_data['inputs'][0]['name'] == 'spacetime.nc'


def test_app_yaml_inputs_len(get_config_instance):
    config = get_config_instance
    # One input is the minimum required
    assert len(config.app_data['inputs']) > 0


def test_app_yaml_inputs_to_plot_len(get_config_instance):
    config = get_config_instance
    # Need to plot at least one field
    assert len(config.app_data['inputs']) > 0


def test_app_yaml_inputs_to_plot_is_dict(get_config_instance):
    config = get_config_instance
    # Need to plot at least one field
    assert isinstance(config.app_data['inputs'][0]['to_plot'], dict)


def test_app_yaml_inputs_to_plot_values(get_config_instance):
    config = get_config_instance
    # Need to plot at least one field
    assert config.app_data['inputs'][0]['to_plot'] == {'air_temp': 'xy'}


def test_app_yaml_inputs_to_plot_items(get_config_instance):
    config = get_config_instance
    # to_plot is a dict()
    # The key, for the first file [0], is 'to_plot'' and the value is a string of fields to plot
    assert config.app_data['inputs'][0]['to_plot']['air_temp'] == 'xy'


def test_app_yaml_outputs():
    if 'EVIZ_OUTPUT_PATH' in os.environ:
        assert constants.output_path is not None
    else:
        assert constants.output_path == './output_plots'


def test_app_yaml_output_dir(get_config_instance):
    config = get_config_instance
    if 'EVIZ_OUTPUT_PATH' not in os.environ:
        assert config.app_data['outputs']['output_dir'] == 'test/data'


# sdat1 = self.config.readers[s1].read_data(fn1)

def test_spec_yaml_exists(get_config_instance):
    config = get_config_instance
    assert config.have_specs_yaml_file is True


def test_app_yaml_inputs_exp_id(get_config_instance):
    config = get_config_instance
    assert config.app_data['inputs'][0]['exp_id'] == 'eviz_4D_data'


def test_app_yaml_inputs_exp_name(get_config_instance):
    config = get_config_instance
    assert 'exp_name' not in config.app_data['inputs'][0]


def test_app_yaml_inputs_description(get_config_instance):
    config = get_config_instance
    assert config.app_data['inputs'][0]['description'] == 'Sample 4D test data'


def test_app_yaml_inputs_name2(get_config_instance):
    config = get_config_instance
    assert config.app_data['inputs'][1]['name'] == 'timeseries.nc'


def test_app_yaml_inputs_exp_id2(get_config_instance):
    config = get_config_instance
    assert config.app_data['inputs'][1]['exp_id'] == 'eviz_TS_data'


def test_app_yaml_inputs_description2(get_config_instance):
    config = get_config_instance
    assert config.app_data['inputs'][1]['description'] == 'Sample timeseries test data'


def test_app_yaml_for_inputs_compare_exp_lists(get_config_instance):
    config = get_config_instance
    # Check for comma-separated string
    assert 'ids' in config.app_data['for_inputs']['compare']


def test_app_yaml_for_inputs_get_airmass_file_name(get_config_instance):
    config = get_config_instance
    # Check for comma-separated string
    assert config.app_data['for_inputs']['airmass_file_name'] == 'airmass.clim.*.nc4'


def test_app_yaml_for_inputs_get_airmass_field_name(get_config_instance):
    config = get_config_instance
    # Check for comma-separated string
    assert config.app_data['for_inputs']['airmass_field_name'] == 'AIRMASS'


def test_app_yaml_for_inputs_get_airmass_season(get_config_instance):
    config = get_config_instance
    # Check for comma-separated string
    assert config.app_data['for_inputs']['airmass_season'] == 'DJF'


def test_app_yaml_for_inputs_compare_extra_diff(get_config_instance):
    config = get_config_instance
    # extra_diff options limited to 3 choices
    assert config.app_data['for_inputs']['compare']['extra_diff'] in ['ratio', 'percc', 'percd']


def test_app_yaml_for_inputs_compare_cmap(get_config_instance):
    config = get_config_instance
    assert isinstance(config.app_data['for_inputs']['compare']['cmap'], str)


def test_app_yaml_outputs_print_to_file(get_config_instance):
    config = get_config_instance
    assert config.app_data['outputs']['print_to_file'] in ['yes', 'no', 1, 0, 'true', 'false']


def test_app_yaml_outputs_print_format(get_config_instance):
    config = get_config_instance
    assert config.app_data['outputs']['print_format'] in ['png', 'pdf', 'ps', 'eps', 'tiff', 'jpg']


def test_app_system_opts_archive_web_results_false(get_config_instance):
    config = get_config_instance
    assert config.app_data['system_opts']['archive_web_results'] is False


def test_get_plot_specs(get_config_instance):
    config = get_config_instance
    assert config.get_plot_specs('air_temp') == ['xyplot', 'yzplot', 'xtplot', 'txplot']


def test_read_meta_coords_is_dict(get_config_instance):
    config = get_config_instance
    assert isinstance(config.meta_coords, dict)


def test_read_meta_coords_size(get_config_instance):
    """" Test for updates in meta coords YAML file"""
    config = get_config_instance
    assert len(config.meta_coords) == 5


@pytest.mark.parametrize(
    ('key', 'model', 'expected'),
    (
            ('xc', 'generic', 'lon,longitude,im'),
            ('yc', 'generic', 'lat,latitude,jm'),
            ('zc', 'generic', 'lev,level,levels,plev,lm,eta_dim'),
            ('tc', 'generic', 'time,rec_dim,ntimemax'),
            ('xc', 'geos', 'lon'),
            ('yc', 'geos', 'lat'),
            ('zc', 'geos', 'lev'),
            ('tc', 'geos', 'time'),
            ('xc', 'ccm', 'lon'),
            ('yc', 'ccm', 'lat'),
            ('zc', 'ccm', 'lev'),
            ('tc', 'ccm', 'time'),
            ('xc', 'airnow', 'Longitude'),
            ('yc', 'airnow', 'Latitude'),
            ('zc', 'airnow', 'NA'),
            ('tc', 'airnow', 'time'),
            ('xc', 'omi', 'longitude,Longitude,Long,long,lon'),
            ('yc', 'omi', 'Latitude,latitude,lat'),
            ('zc', 'omi', 'NA'),
            ('tc', 'omi', 'Time'),
            ('xc', 'wrf', {'dim': 'west_east', 'coords': 'XLONG,XLONG_U,XLONG_V'}),
            ('yc', 'wrf', {'dim': 'south_north', 'coords': 'XLAT,XLAT_U,XLAT_V'}),
            ('zc', 'wrf', {'dim': 'bottom_top,bottom_top_stag,soil_layers,soil_layers_stag', 'coords': 'VERT'}),
            ('tc', 'wrf', {'dim': 'Time', 'coords': 'XTIME'}),
            ('xc', 'lis', {'dim': 'east_west', 'coords': 'lon'}),
            ('yc', 'lis', {'dim': 'north_south', 'coords': 'lat'}),
            ('zc', 'lis', {'dim': 'RelSMC_profiles,SoilTemp_profiles,SoilMoist_profiles,SmLiqFrac_profiles'}),
            ('tc', 'lis', {'dim': 'time', 'coords': 'time'}),
    )
)
def test_read_meta_coords_get_values_from_keys(key, model, expected, get_config_instance):
    """" Test for updates in meta coords YAML file"""
    config = get_config_instance
    assert config.meta_coords[key][model] == expected


def test_get_meta_coord_xc_dim_key(get_config_instance):
    config = get_config_instance
    assert 'dim' in config.get_meta_coord('xc').keys()


def test_get_meta_coord_xc_coords_key(get_config_instance):
    config = get_config_instance
    assert 'coords' in config.get_meta_coord('xc').keys()


def test_get_meta_coord_xc_coord_value_from_test_model_key(get_config_instance):
    config = get_config_instance
    assert config.get_meta_coord('xc')['coords'] == 'lon'


def test_get_meta_coord_zc_value_from_model_key(get_config_instance):
    config = get_config_instance
    assert config.get_meta_coord('zc') == 'lev'


@pytest.mark.parametrize(
    ('key', 'model', 'expected'),
    (
            ('field_name', 'generic', 'long_name'),
            ('field_name', 'geos', 'long_name'),
            ('field_name', 'lis', 'long_name'),
            ('field_name', 'wrf', 'description'),
            ('field_name', 'test', 'long_name'),
            ('xc', 'lis', 'east_west'),
            ('xc', 'wrf', 'west_east'),
            ('xc', 'test', 'lon'),
            ('yc', 'lis', 'north_south'),
            ('yc', 'wrf', 'south_north'),
            ('yc', 'test', 'lat'),
            ('projection', 'lis', 'MAP_PROJECTION'),
            ('projection', 'wrf', 'MAP_PROJ_CHAR'),
            ('projection', 'test', 'Lambert Conformal'),
    )
)
def test_read_meta_attrs_get_values_from_keys(key, model, expected, get_config_instance):
    """" Test for updates in meta attrs YAML file"""
    config = get_config_instance
    assert config.meta_attrs[key][model] == expected


def test_read_meta_attrs_is_dict(get_config_instance):
    config = get_config_instance
    assert isinstance(config.meta_attrs, dict)


def test_read_meta_attrs_size(get_config_instance):
    config = get_config_instance
    assert len(config.meta_attrs) == 10


def test_read_meta_attrs_get_value_from_key(get_config_instance):
    config = get_config_instance
    assert config.meta_attrs['units'] == 'units'
