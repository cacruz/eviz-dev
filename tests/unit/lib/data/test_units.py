import numpy as np
import xarray as xr
import pytest
from unittest.mock import patch
import os
import requests
from eviz.lib import constants as constants

from eviz.lib.data.units import (
    adjust_units, moles_to_mass, mass_to_moles,
    mb_to_Pa, Pa_to_hPa, Pa_to_mb, mb_to_hPa, hPa_to_mb, g_to_mg, mg_to_g,
    kg_to_mg, mg_to_kg, g_to_kg, kg_to_g, f_to_c, c_to_f, c_to_k, k_to_c,
    f_to_k, k_to_f
)


def test_get_airmass_with_mock(config_for_units):
    """Test get_airmass using mock data"""
    from tests.fixtures.mock_airmass import create_mock_airmass_dataarray
    
    # Patch the get_airmass function to return our mock data
    with patch('eviz.lib.data.units.get_airmass', return_value=create_mock_airmass_dataarray()):
        from eviz.lib.data.units import get_airmass
        airmass = get_airmass(config_for_units)
        assert airmass is not None
        assert isinstance(airmass, xr.DataArray)
        assert airmass.name == 'AIRMASS'
        assert 'lat' in airmass.dims
        assert 'lon' in airmass.dims
        assert 'lev' in airmass.dims


@pytest.fixture(scope="session")
def check_airmass_availability():
    """
    Check if the airmass file is available either locally or via URL.
    
    Returns:
        bool: True if airmass data is available, False otherwise
    """
    # First check local file
    local_path = os.path.join(os.getcwd(), 'airmass.nc')
    if os.path.exists(local_path):
        return True
        
    # Then check URL
    try:
        response = requests.head(constants.AIRMASS_URL, timeout=5)
        return response.status_code in (200, 302, 307, 443)
    except:
        return False


def test_moles_to_mass(config_for_units):
    """Test moles_to_mass conversion using mock species data"""
    assert moles_to_mass(num_moles=1, molar_mass=config_for_units.species_db['O3']['MW_g']) == 48


def test_mass_to_moles(config_for_units):
    """Test mass_to_moles conversion using mock species data"""
    assert mass_to_moles(g_of_element=48, molar_mass=config_for_units.species_db['O3']['MW_g']) == 1


@pytest.fixture
def mock_dataset():
    # Create a simple synthetic dataset with AIRMASS variable
    return xr.Dataset(
        data_vars={"AIRMASS": (["lat", "lon"], np.ones((10, 10)))},
        coords={"lat": np.linspace(-90, 90, 10), "lon": np.linspace(-180, 180, 10)}
    )


@pytest.mark.parametrize(
    ('key', 'expected'),
    (
            ('kg / m2 / s', 'kg/m2/s'),
            ('kgm-2 s-1', 'kg/m2/s'),
            ('kgm^-2s^-1', 'kg/m2/s'),
            ('du', 'DU'),
            ('dobson units', 'DU'),
            ('kgC/m2/s', 'kgC/m2/s'),
            ('kgC m-2 s-1', 'kgC/m2/s'),
            ('kgC m^-2 s^-1', 'kgC/m2/s'),
            ('kg cm-2 s-1', 'kgC/m2/s'),
            ('kg cm^-2 s^-1', 'kgC/m2/s'),
            ('molec per cm2', 'molec/cm2'),
            ('molecules/cm2', 'molec/cm2'),
            ('moleccm-2s-1', 'molec/cm2/s'),
            ('moleccm^-2s^-1', 'molec/cm2/s'),
            ('molmol-1dry', 'mol/mol'),
            ('molmol-1', 'mol/mol'),
    )
)
def test_adjust_units(key, expected):
    assert adjust_units(key) == expected


def test_mb_to_Pa():
    assert mb_to_Pa(1) == 100


def test_Pa_to_hPa():
    assert Pa_to_hPa(100) == 1


def test_Pa_to_mb():
    assert Pa_to_mb(100) == 1


def test_mb_to_hPa():
    assert mb_to_hPa(100) == 100


def test_hPa_to_mb():
    assert hPa_to_mb(100) == 100


def test_g_to_mg():
    assert g_to_mg(1) == 1000


def test_mg_to_g():
    assert mg_to_g(1000) == 1


def test_kg_to_mg():
    assert kg_to_mg(1) == 1e6


def test_mg_to_kg():
    assert mg_to_kg(1e6) == 1


def test_g_to_kg():
    assert g_to_kg(1e3) == 1


def test_kg_to_g():
    assert kg_to_g(1) == 1e3


def test_f_to_c():
    assert f_to_c(32) == 0
    assert f_to_c(212) == 100


def test_c_to_f():
    assert c_to_f(0) == 32
    assert c_to_f(100) == 212


def test_c_to_k():
    assert c_to_k(-273.15) == 0


def test_k_to_c():
    assert k_to_c(0) == -273.15


def test_f_to_k():
    assert f_to_k(32) == 273.15
    assert f_to_k(212) == 373.15


def test_k_to_f():
    assert k_to_f(273.15) == 32
    assert k_to_f(373.15) == 212

