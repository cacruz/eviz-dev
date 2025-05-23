import pytest
import eviz.lib.constants as constants

def test_cartopy_data_dir():
    discover_only = '/discover/nobackup/projects/jh_tutorials/JH_examples/JH_datafiles/Cartopy'
    assert constants.CARTOPY_DATA_DIR == discover_only

@pytest.mark.parametrize(
    ('value', 'expected'),
    (
        ('AVOGADRO', 6.022140857e+23),
        ('BOLTZ', 1.38064852e-23),
        ('G', 9.80665),
        ('R_EARTH_M', 6371.0072e+3),
        ('R_EARTH_KM', 6371.0072),
        ('MW_AIR_G', 28.9644),
        ('MW_AIR_KG', 28.9644e-3),
        ('MW_H2O_G', 18.016),
        ('MW_H2O_KG', 18.016e-3),
        ('RD', 287.0),
        ('RSTARG', 8.3144598),
        ('RV', 461.0),
    )
)
def test_const_attrs(value, expected):
    assert getattr(constants, value) == expected

def test_supported_models():
    assert constants.SUPPORTED_MODELS == [
        'geos', 'ccm', 'cf', 'wrf', 'lis', 'gridded',
        'fluxnet', 'airnow', 'test', 'omi', 'landsat', 'mopitt'
    ]

def test_supported_plot_types():
    assert constants.PLOT_TYPES == ['xy', 'yz', 'xt', 'tx', 'polar', 'sc']

def test_format_png():
    assert constants.FORMAT_PNG == 'png'

def test_meta_attrs_name():
    assert constants.META_ATTRS_NAME == 'meta_attributes.yaml'

def test_meta_coords_name():
    assert constants.META_COORDS_NAME == 'meta_coordinates.yaml'

def test_species_db_name():
    assert constants.SPECIES_DB_NAME == 'species_database.yaml'

def test_xp_const():
    # XP_CONST = (AVOGADRO * 10) / (MW_AIR_G * G) * 1e-09
    expected = (constants.AVOGADRO * 10) / (constants.MW_AIR_G * constants.G) * 1e-09
    assert constants.XP_CONST == expected
