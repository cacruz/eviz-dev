import sys
from dataclasses import dataclass, field
import xarray as xr
import numpy as np
import requests
import logging
import os
from eviz.lib import constants as constants
import eviz.lib.utils as u
from eviz.lib.data.pipeline.processor import DataProcessor

logger = logging.getLogger(__name__)

"""
Contains methods for converting the units of data.
Some functions are adopted from GCpy - with minor modifications
"""

AVOGADRO = constants.AVOGADRO
MOLAR_MASS_AIR = constants.MW_AIR_KG
DU_CONVERSION_CGS = 2.6867e16  # molecules/cm² for 1 DU
DU_CONVERSION = 2.6867e20  # molecules/cm² for 1 DU
PPB_CONVERSION = 1e9


# Cache for airmass data to avoid repeated reads
_airmass_cache = {}

def get_airmass(config, dry_run=False):
    """
    Retrieves airmass field stored in a file or URL

    Parameters:
        config: eviz config object
        dry_run: bool

    Returns:
        airmass field: xArray
    """
    global _airmass_cache
    
    # Get configuration values
    airmass_file_name = u.get_nested_key_value(config.app_data.for_inputs, ['airmass_file_name']) if (
        config.app_data is not None and hasattr(config.app_data, 'for_inputs')
    ) else None
    
    airmass_field_name = u.get_nested_key_value(config.app_data.for_inputs, ['airmass_field_name']) if (
        config.app_data is not None and hasattr(config.app_data, 'for_inputs')
    ) else 'AIRMASS'
    
    # If no local file specified, use the URL
    if not airmass_file_name:
        airmass_file_name = constants.AIRMASS_URL
    
    # Expand environment variables in the file path if it's a local file
    if airmass_file_name and 'https' not in airmass_file_name:
        airmass_file_name = os.path.expandvars(airmass_file_name)
    
    # Create a cache key based on file name and field name
    cache_key = f"{airmass_file_name}:{airmass_field_name}"
    
    # For dry run, just check if the file exists
    if dry_run:
        if 'https' in airmass_file_name:
            import requests
            response = requests.head(airmass_file_name, timeout=5)
            return response.status_code in (200, 302, 307, 443)
        else:
            return os.path.exists(airmass_file_name)
    
    # Check if we have this data in the cache
    if cache_key in _airmass_cache:
        logger.debug(f"Using cached airmass data for {airmass_file_name}")
        return _airmass_cache[cache_key]
    
    logger.info(f"Loading airmass data from {airmass_file_name}")
    
    # Try local file first if it's not a URL
    if 'https' not in airmass_file_name:
        try:
            ds = xr.open_dataset(airmass_file_name, decode_cf=True)
            airmass_data = ds[airmass_field_name]
            # Cache the result
            _airmass_cache[cache_key] = airmass_data
            return airmass_data
        except FileNotFoundError as e:
            logger.warning(f"Local airmass file not found: {e}")
            logger.info(f"Falling back to URL: {constants.AIRMASS_URL}")
            # Fall back to URL if local file not found
            try:
                ds = download_airmass(constants.AIRMASS_URL)
                airmass_data = ds[airmass_field_name]
                # Cache the result
                _airmass_cache[cache_key] = airmass_data
                return airmass_data
            except Exception as e:
                logger.error(f"Failed to download airmass data: {e}")
                sys.exit(1)
    else:
        # Direct URL access
        try:
            ds = download_airmass(airmass_file_name)
            airmass_data = ds[airmass_field_name]
            # Cache the result
            _airmass_cache[cache_key] = airmass_data
            return airmass_data
        except Exception as e:
            logger.error(f"Failed to download airmass data: {e}")
            sys.exit(1)


def download_airmass(url):
    """
    Downloads airmass file

    Parameters:
        url (str): URL of the file

    Returns:
        xArray dataset
    """
    filename = os.path.basename(url)
    downloaded_file = os.path.join("./", filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(downloaded_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return xr.open_dataset(downloaded_file)


def calculate_total_mass(airmass):
    """
    Calculates the total mass given the airmass per square meter field.

    Parameters:
        airmass (xarray.DataArray): The input data array representing mass per square meter.

    Returns:
        float: The total mass in kilograms.
    """
    area_expanded = calculate_total_area(airmass)
    return (airmass * area_expanded).sum().item()


def calculate_total_area(data):
    """
    Calculates the total surface area for a regular lat-lon grid.

    Parameters:
        data (xarray.DataArray): The input data array.

    Returns:
        xarray.DataArray: An array with the same shape as data containing cell areas in m^2.
    """
    dlat = np.deg2rad(data.coords['lat'].diff('lat').mean().item())
    dlon = np.deg2rad(data.coords['lon'].diff('lon').mean().item())

    area = constants.R_EARTH_M ** 2 * dlat * dlon * np.cos(np.deg2rad(data.lat))

    return area.broadcast_like(data)


def adjust_units(units):
    """
    Creates a consistent unit string that will be used in the unit
    conversion routines below.

    Parameters:
        units: str
            Input unit string.

    Returns:
        adjusted_units: str
            Output unit string, adjusted to a consistent value.

    Remarks:
        Unit list is incomplete -- currently is geared to units from
        common model diagnostics (e.g. kg/m2/s, kg, and variants).
    """
    # Error check arguments
    if not isinstance(units, str):
        raise TypeError("Units must be of type str!")

    # Strip all spaces in the unit string
    units_squeezed = units.replace(" ", "")

    if units_squeezed in [
        "kg/m2/s",
        "kgm-2s-1",
        "kgm^-2s^-1"
    ]:
        unit_desc = "kg/m2/s"

    elif units_squeezed in [
        "du",
        "dobsonunits",
        "dob",
    ]:
        unit_desc = "DU"

    elif units_squeezed in [
        "kgC/m2/s",
        "kgCm-2s-1",
        "kgCm^-2s^-1",
        "kgc/m2/s",
        "kgcm-2s-1",
        "kgcm^-2s^-1",
    ]:
        unit_desc = "kgC/m2/s"

    elif units_squeezed in [
        "molec/cm2",
        "molecules/cm2",
        "moleccm-2",
        "molecpercm2",
    ]:
        unit_desc = "molec/cm2"

    elif units_squeezed in [
        "molec/cm2/s",
        "moleccm-2s-1",
        "moleccm^-2s^-1"
    ]:
        unit_desc = "molec/cm2/s"
    elif units_squeezed in [
        "molmol-1dry",
        "mol/mol",
        "molmol-1"
    ]:
        unit_desc = "mol/mol"
    elif units_squeezed in [
        "kg/kg",
        "kgkg-1",
    ]:
        unit_desc = "kg/kg"
    else:
        unit_desc = units_squeezed

    return unit_desc


#  Mass conversions
def moles_to_mass(num_moles, molar_mass):
    """Convert moles to mass"""
    grams = num_moles * molar_mass
    return grams


def mass_to_moles(g_of_element, molar_mass):
    """Convert mass of an element to moles"""
    moles = g_of_element / molar_mass
    return moles


def mb_to_hPa(mb):
    """  Convert mb to hPa """
    return mb


def mb_to_Pa(mb):
    """  Convert mb to Pa """
    return mb * 100

def Pa_to_hPa(Pa):
    """  Convert Pa to hPa """
    return Pa / 100


def Pa_to_mb(Pa):
    """  Convert Pa to mb """
    return Pa / 100


def hPa_to_Pa(hPa):
    """  Convert hPa to Pa """
    return hPa / 100


def hPa_to_mb(hPa):
    """  Convert hPa to mb """
    return hPa


def g_to_mg(gram):
    """  Convert grams to mg """
    return gram * 1000


def g_to_kg(gram):
    """ Convert grams to kilograms """
    return gram / 1000


def mg_to_g(mgram):
    """ Convert mg to grams """
    return mgram / 1000


def mg_to_kg(mgram):
    """Convert mg to kilograms"""
    return mgram / 1e6


def kg_to_g(kilogram):
    """Convert kilograms to grams"""
    return kilogram * 1000


def kg_to_mg(kilogram):
    """Convert kilograms to mg"""
    return kilogram * 1e6


#  Temperature conversions
def f_to_c(f):
    """Convert fahrenheit to celsius"""
    return (f - 32) * 5 / 9


def f_to_k(f):
    """Convert fahrenheit to kelvin"""
    return (f - 32) * 5 / 9 + 273.15


def c_to_f(c):
    """Convert celsius to fahrenheit"""
    return (c * 9 / 5) + 32


def c_to_k(c):
    """Convert celsius to kelvin"""
    return c + 273.15


def k_to_c(k):
    """Convert kelvin to celsius"""
    return k - 273.15


def k_to_f(k):
    """Convert kelvin to fahrenheit"""
    return (k - 273.15) * 9 / 5 + 32


def mol_to_kg(mol_frac, molar_mass_species):
    """Convert molar fraction to kg"""
    return mol_frac * (molar_mass_species / MOLAR_MASS_AIR)


def kg_to_mol(kg_frac, molar_mass_species):
    """Convert kg to molar fraction"""
    return kg_frac * (MOLAR_MASS_AIR / molar_mass_species)


def mol_to_molecules_cm2(mol_frac, air_column_density):
    """
    Convert from mol mol⁻¹ to molecules cm⁻².
    Parameters:
        mol_frac: molar fraction of the species (mol mol⁻¹)
        air_column_density: air column density (molecules cm⁻²)
    Returns:
        number of molecules per cm²
    """
    return mol_frac * air_column_density


def compute_column_DU(
    mixing_ratio: xr.DataArray,
    airmass: xr.DataArray,
    molar_mass_species: float,
    mixing_ratio_units: str = "mol/mol",
    vertical_dim: str = "lev"
) -> xr.DataArray:
    """
    Convert 3D mixing ratio of a chemical species to column Dobson Units (DU).

    Parameters
    ----------
    mixing_ratio : xr.DataArray
        Mixing ratio of the species (mol/mol or kg/kg). Shape must match airmass.
    airmass : xr.DataArray
        Airmass per model layer [kg/m²]. Same shape as mixing_ratio.
    molar_mass_species : float
        Molar mass of the chemical species [kg/mol] (e.g., 0.048 for O3).
    mixing_ratio_units : str, optional
        Units of the mixing ratio ("mol/mol" or "kg/kg"). Default is "mol/mol".
    vertical_dim : str, optional
        Name of the vertical dimension. Default is "lev".

    Returns
    -------
    xr.DataArray
        2D field of total column DU [DU] over latitude/longitude.
    """

    if mixing_ratio.shape != airmass.shape:
        raise ValueError("mixing_ratio and airmass must have the same shape")

    if mixing_ratio_units == "mol/mol":
        DU_factor = AVOGADRO / (DU_CONVERSION * MOLAR_MASS_AIR)  # ≈ 77552
    elif mixing_ratio_units == "kg/kg":
        DU_factor = AVOGADRO / (DU_CONVERSION * molar_mass_species)  # ≈ 46584 for O3, varies for other species
    else:
        raise ValueError(f"Unsupported mixing_ratio_units: {mixing_ratio_units}. Use 'mol/mol' or 'kg/kg'.")

    du_per_level = mixing_ratio * airmass * DU_factor

    return du_per_level.sum(dim=vertical_dim)


def convert_DU_to_mixing_ratio(
    total_DU: xr.DataArray,
    airmass: xr.DataArray,
    molar_mass_species: float,
    output_units: str = "mol/mol"
) -> xr.DataArray:
    """
    Convert total column Dobson Units (DU) to a 3D mixing ratio field.

    Assumes the species is uniformly distributed over the vertical profile
    according to airmass weight (i.e., mass-weighted uniform scaling).

    Parameters
    ----------
    total_DU : xr.DataArray
        2D field of total column Dobson Units [DU].
    airmass : xr.DataArray
        3D field of air mass per layer [kg/m²].
    molar_mass_species : float
        Molar mass of the species [kg/mol] (e.g., 0.048 for ozone).
    output_units : str, optional
        Desired units of mixing ratio: "mol/mol" or "kg/kg".

    Returns
    -------
    xr.DataArray
        3D mixing ratio field in requested units.
    """

    # Compute total airmass column (sum over vertical levels)
    total_airmass = airmass.sum(dim='lev') 

    if output_units == "mol/mol":
        DU_to_molmol = (DU_CONVERSION * MOLAR_MASS_AIR) / AVOGADRO
        # Distribute total DU across levels using normalized airmass weights
        weight = airmass / total_airmass
        mixing_ratio = total_DU * weight * DU_to_molmol  # broadcasted
    elif output_units == "kg/kg":
        DU_to_kgkg = (DU_CONVERSION * molar_mass_species) / AVOGADRO
        weight = airmass / total_airmass
        mixing_ratio = total_DU * weight * DU_to_kgkg
    else:
        raise ValueError(f"Unsupported output_units: {output_units}. Use 'mol/mol' or 'kg/kg'.")

    return mixing_ratio


def mixing_ratio_to_ppb(
    mixing_ratio,
    units="mol/mol",
    molar_mass_species=None
):
    """
    Convert mixing ratio to parts per billion (ppb).

    Supports both scalar and xarray.DataArray inputs.

    Parameters
    ----------
    mixing_ratio : float, np.ndarray, or xr.DataArray
        Mixing ratio in mol/mol or kg/kg.
    units : str
        Units of input mixing ratio. One of "mol/mol" or "kg/kg".
    molar_mass_species : float, optional
        Required if units is "kg/kg". Molar mass of species in kg/mol.

    Returns
    -------
    Same type as input (xr.DataArray or float), with values in ppb.
    """

    # Helper function for scalar/numpy inputs
    def _convert_scalar(value):
        if units == "mol/mol":
            return value * 1e9
        elif units == "kg/kg":
            if molar_mass_species is None:
                raise ValueError("molar_mass_species must be provided for 'kg/kg' input.")
            molmol = (value / molar_mass_species) * MOLAR_MASS_AIR
            return molmol * 1e9
        else:
            raise ValueError("Unsupported units: use 'mol/mol' or 'kg/kg'.")

    # If input is xarray.DataArray
    if isinstance(mixing_ratio, xr.DataArray):
        result = mixing_ratio.copy()

        if units == "mol/mol":
            result.data = mixing_ratio.data * 1e9
        elif units == "kg/kg":
            if molar_mass_species is None:
                raise ValueError("molar_mass_species must be provided for 'kg/kg' input.")
            molmol = (mixing_ratio / molar_mass_species) * MOLAR_MASS_AIR
            result.data = molmol.data * 1e9
        else:
            raise ValueError("Unsupported units: use 'mol/mol' or 'kg/kg'.")

        result.attrs["units"] = "ppb"
        result.name = mixing_ratio.name or "mixing_ratio_ppb"
        return result

    else:  # scalar or ndarray
        return _convert_scalar(mixing_ratio)


def ppb_to_mixing_ratio(
    ppb,
    output_units="mol/mol",
    molar_mass_species=None
):
    """
    Convert from parts per billion (ppb) to mixing ratio.

    Supports scalar, NumPy, and xarray.DataArray inputs.

    Parameters
    ----------
    ppb : float, np.ndarray, or xr.DataArray
        Mixing ratio in ppb.
    output_units : str
        Desired output units: "mol/mol" or "kg/kg".
    molar_mass_species : float, optional
        Required if output_units is "kg/kg". Molar mass of species in kg/mol.

    Returns
    -------
    Same type as input (xr.DataArray or float), with converted mixing ratio.
    """

    # Internal helper for scalar inputs
    def _convert_scalar(val):
        if output_units == "mol/mol":
            return val * 1e-9
        elif output_units == "kg/kg":
            if molar_mass_species is None:
                raise ValueError("molar_mass_species must be provided for 'kg/kg' output.")
            molmol = val * 1e-9
            return (molmol * molar_mass_species) / MOLAR_MASS_AIR
        else:
            raise ValueError("Unsupported output_units. Use 'mol/mol' or 'kg/kg'.")

    # xarray support
    if isinstance(ppb, xr.DataArray):
        result = ppb.copy()

        if output_units == "mol/mol":
            result.data = ppb.data * 1e-9
            result.attrs["units"] = "mol/mol"
        elif output_units == "kg/kg":
            if molar_mass_species is None:
                raise ValueError("molar_mass_species must be provided for 'kg/kg' output.")
            molmol = ppb.data * 1e-9
            result.data = (molmol * molar_mass_species) / MOLAR_MASS_AIR
            result.attrs["units"] = "kg/kg"
        else:
            raise ValueError("Unsupported output_units. Use 'mol/mol' or 'kg/kg'.")

        result.name = ppb.name or "mixing_ratio"
        return result

    else:
        return _convert_scalar(ppb)


@dataclass
class Units:
    """ This class defines attributes and methods to perform unit conversions of xarray data arrays.
        The conversion will be automatic if the fields are registered in eviz's species database and
        the units are supported. Otherwise, the conversion specification can be made in eviz's config
        files (APP and SPECS YAML files). Please see user's guide for more information.

    Parameters:

    config (ConfigManager) :
        Representation of the model configuration used to specify data sources and
        user choices for the map generation.
    """
    config: 'ConfigManager'
    species_db: dict = field(init=False)
    airmass: float = field(init=False)

    def __post_init__(self):
        self.logger.debug("Create units converter")
        self.species_db = self.config.species_db

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def convert_chem(self, data, species_name, to_unit,
                     air_column_density=None,
                     airmass=None):
        """ Conversion method for chemical species

        Parameters:
            data (xArray): data to undergo unit conversion
            to_unit (str): data destination unit
            species_name (str): species name of the data
            air_column_density (xArray)
            airmass (xArray)
        """
        if species_name not in self.species_db:
            self.logger.warning(f"Species {species_name} not found in data.")
            return data

        # Get a consistent value for the units string
        # (ignoring minor differences in formatting)
        if not data.attrs.get('units'):
            msg = f"{species_name} has no units!"
            self.logger.warning(msg)
            return data

        self.airmass = get_airmass(self.config).squeeze()

        from_unit = adjust_units(data.attrs.get('units'))
        to_unit = adjust_units(to_unit)
        if from_unit == to_unit:
            return data

        species_info = self.species_db[species_name]
        self.logger.info(f"Convert {species_name} units from {from_unit} to {to_unit}")

        molar_mass_species = species_info['MW_kg']
        if "MW_g" in species_info:
            mw_g = species_info.get("MW_g")
        else:
            msg = "Cannot find molecular weight MW_g for species {}".format(
                species_name)
            msg += "!\nPlease add the MW_g field for {}".format(species_name)
            msg += " to the species_database.yaml file."
            raise ValueError(msg)

        # Mass of dry air in kg (required when converting from v/v)
        area_m2 = 1.0
        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = DataProcessor(self.config)

        if from_unit in ['mol/mol', 'kg/kg'] and to_unit in ['DU'] :
            new_data = compute_column_DU(
                    mixing_ratio=data,
                    airmass=self.airmass,
                    molar_mass_species=molar_mass_species,
                    mixing_ratio_units=from_unit
            )

        elif from_unit in ['DU'] and to_unit in ['mol/mol', 'kg/kg']:
            new_data = convert_DU_to_mixing_ratio(
                    total_DU=data,
                    airmass=self.airmass,
                    molar_mass_species=molar_mass_species, 
                    output_units=from_unit
            )
        elif from_unit in ['mol/mol'] and to_unit in ['ppb'] :
            new_data = mixing_ratio_to_ppb(data, units="mol/mol")

        elif from_unit in ['kg/kg'] and to_unit in ['ppb'] :
            new_data = mixing_ratio_to_ppb(data, units="kg/kg", 
                                           molar_mass_species=molar_mass_species)

        elif from_unit in ['ppb'] and to_unit in ['mol/mol'] :
            new_data = ppb_to_mixing_ratio(data, output_units="mol/mol")

        elif from_unit in ['ppb'] and to_unit in ['kg/kg'] :
            new_data = mixing_ratio_to_ppb(data, units="kg/kg", 
                                           molar_mass_species=molar_mass_species)
        else:

            # Number of seconds should be passed via the interval argument, but...
            interval = [2678400.0]
            numsec = interval
            # Special handling is required if multiple times in interval (for
            # broadcast)
            if len(interval) > 1:
                if 'time' in data.dims:
                    # Need to right pad the interval array with new axes up to the
                    # time dim of the dataset to enable broadcasting
                    numnewdims = len(data.dims) - (data.dims.index('time') + 1)
                    for _ in range(numnewdims):
                        numsec = numsec[:, np.newaxis]
                else:
                    # Raise an error if no time in dataset but interval has length > 1
                    raise ValueError(
                        'Interval passed to convert_units has length greater than one but data array has no time dimension')

            # TODO: needs testing
            if to_unit == "kg/m2/s":
                data = data * area_m2
                data = data * numsec

            if from_unit in ['mol/mol', 'kg/kg'] and to_unit in ['DU'] :
                new_data = compute_column_DU(
                        mixing_ratio=data,
                        airmass=self.airmass,
                        molar_mass_species=molar_mass_species,
                        mixing_ratio_units=from_unit
                )

            elif from_unit in ['mol/mol'] and to_unit in ['ppb'] :
                new_data = mixing_ratio_to_ppb(data, units="mol/mol")

            elif from_unit in ['kg/kg'] and to_unit in ['ppb'] :
                new_data = mixing_ratio_to_ppb(data, units="kg/kg", 
                                            molar_mass_species=molar_mass_species)

            else:
                raise ValueError(
                    f"Units ({to_unit}) in variable {species_name} are not supported")

            conversion_functions = {
                'moles': {
                    'g': lambda d: moles_to_mass(d, self.species_db['MW_g']),
                },
                'g': {
                    'mg': lambda d: g_to_mg(d),
                    'kg': lambda d: g_to_kg(d),
                    'moles': lambda d: mass_to_moles(d, self.species_db['MW_g']),
                },
                'mg': {
                    'g': lambda d: mg_to_g(d),
                    'kg': lambda d: mg_to_kg(d),
                },
                'kg': {
                    'g': lambda d: kg_to_g(d),
                    'mg': lambda d: kg_to_mg(d),
                },
                'mol/mol': {
                    'kg/kg': lambda d: mol_to_kg(d, molar_mass_species),
                    'molecules/cm2': lambda d: mol_to_molecules_cm2(d, air_column_density)
                },
                'kg/kg': {
                    'mol/mol': lambda d: kg_to_mol(d, molar_mass_species),
                },
                'mol': {
                    'mol/mol': lambda d: moles_to_mass(d, molar_mass_species)
                },
            }

            if from_unit not in conversion_functions or to_unit not in conversion_functions[
                from_unit]:
                raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported.")

            new_data = conversion_functions[from_unit][to_unit](data)

        new_data.attrs["units"] = to_unit
        return new_data

    def convert(self, data, species_name, to_unit):
        """ Conversion method for non-chemical species (e.g. atmospheric fields)

        Parameters:
            to_unit (str): data destination unit
            species_name (str): species name of the data
            data (xArray): data to undergo unit conversion
        """
        # Get a consistent value for the units string
        # (ignoring minor differences in formatting)
        if not data.attrs.get('units'):
            msg = f"{species_name} has no units!"
            raise ValueError(msg)

        from_unit = adjust_units(data.attrs.get('units'))
        to_unit = adjust_units(to_unit)
        if from_unit == to_unit:
            self.logger.warning(
                f"Units are identical {from_unit} == {to_unit}...returning.")
            return data
        self.logger.debug(f"Convert {species_name} units from {from_unit} to {to_unit}")

        conversion_functions = {
            'mb': {
                'hPa': lambda d: mb_to_hPa(d),
                'Pa': lambda d: mb_to_Pa(d),
            },
            'hPa': {
                'mb': lambda d: hPa_to_mb(d),
                'Pa': lambda d: hPa_to_Pa(d),
            },
            'Pa': {
                'hPa': lambda d: Pa_to_hPa(d),
                'mb': lambda d: Pa_to_mb(d),
            },
            'g': {
                'mg': lambda d: g_to_mg(d),
                'kg': lambda d: g_to_kg(d),
            },
            'mg': {
                'g': lambda d: mg_to_g(d),
                'kg': lambda d: mg_to_kg(d),
            },
            'kg': {
                'g': lambda d: kg_to_g(d),
                'mg': lambda d: kg_to_mg(d),
            },
            'F': {
                'C': lambda d: f_to_c(d),
                'K': lambda d: f_to_k(d),
            },
            'C': {
                'F': lambda d: c_to_f(d),
                'K': lambda d: c_to_k(d),
            },
            'K': {
                'F': lambda d: k_to_f(d),
                'C': lambda d: k_to_c(d),
            },
        }

        if from_unit not in conversion_functions or to_unit not in conversion_functions[
            from_unit]:
            raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported.")

        new_data = conversion_functions[from_unit][to_unit](data)
        new_data.attrs["units"] = to_unit
        return new_data
