import os
import pytest
import eviz.lib.const as constants


def test_root_file_path():
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    assert constants.ROOT_FILEPATH == here


def test_cartopy_data_dir():
    discover_only = '/discover/nobackup/projects/jh_tutorials/JH_examples/JH_datafiles/Cartopy'
    assert constants.CARTOPY_DATA_DIR == discover_only


def test_eviz_config_path_default():
    default = os.path.join(constants.ROOT_FILEPATH, 'config')
    assert constants.config_path == default


def test_default_output_path():
    assert constants.output_path == './output_plots'


def test_default_data_path():
    assert constants.data_path == 'data'


def test_meta_attrs_path():
    top = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    here = os.path.join(top, 'config', 'meta_attributes.yaml')
    assert constants.meta_attrs_path == here


def test_meta_coords_path():
    top = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    here = os.path.join(top, 'config', 'meta_coordinates.yaml')
    assert constants.meta_coords_path == here


def test_meta_attrs_name():
    assert constants.meta_attrs_name == 'meta_attributes.yaml'


def test_meta_coords_name():
    assert constants.meta_coords_name == 'meta_coordinates.yaml'


@pytest.mark.parametrize(
    ('value', 'expected'),
    (
            ('AVOGADRO', 6.022140857e+23),
            ('BOLTZ', 1.38064852e-23),
            ('G', 9.80665),
            ('R_EARTH_m', 6371.0072e+3),
            ('R_EARTH_km', 6371.0072),
            ('MW_AIR_g', 28.9644),
            ('MW_AIR_kg', 28.9644e-3),
            ('MW_H2O_g', 18.016),
            ('MW_H2O_kg', 18.016e-3),
            ('RD', 287.0),
            ('RSTARG', 8.3144598),
            ('RV', 461.0),
    )
)
def test_const_attrs(value, expected):
    assert constants.__getattribute__(value) == expected


def test_supported_models():
    assert constants.__getattribute__('supported_models') == ['geos', 'ccm', 'cf', 'wrf', 'lis', 'gridded',
                                                              'fluxnet', 'airnow', 'test', 'omi', 'landsat', 'mopitt']


def test_supported_plot_types():
    assert constants.__getattribute__('plot_types') == ['xy', 'yz', 'xt', 'tx', 'polar', 'sc']
