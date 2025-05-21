import sys
from dataclasses import dataclass, field
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import requests
import logging
import os
from eviz.lib import const as constants
import eviz.lib.utils as u


logger = logging.getLogger(__name__)

"""
Contains methods for converting the units of data.
Some functions are adopted from GCpy - with minor modifications
"""

AVOGADRO = constants.AVOGADRO
MOLAR_MASS_AIR = constants.MW_AIR_kg
DU_CONVERSION = 2.6867e16  # molecules/cm² for 1 DU
PPB_CONVERSION = 1e9


def get_airmass(config, dry_run=False):
    """
    Retrieves airmass field stored in a file or URL

    Parameters:
        config: eviz config object
        dry_run: bool

    Returns:
        airmass field: xArray
    """
    airmass_file_name = constants.AIRMASS_URL
    airmass_field_name = 'AIRMASS'
    if config.app_data is not None:
        if hasattr(config.app_data, 'for_inputs'):
            airmass_file_name = result if (
                result := u.get_nested_key_value(config.app_data.for_inputs, ['airmass_file_name'])) else constants.AIRMASS_URL
            airmass_field_name = result if (
                result := u.get_nested_key_value(config.app_data.for_inputs, ['airmass_field_name'])) else 'AIRMASS'

    logger.debug(f"Loading airmass data from {airmass_file_name}")

    if dry_run:
        if 'https' in airmass_file_name:
            import requests
            response = requests.head(airmass_file_name, timeout=5)
            return response.status_code in (200, 302, 307, 443)
        else:
            import os
            return os.path.exists(airmass_file_name)

    if 'https' in airmass_file_name:
        ds = download_airmass(airmass_file_name)
        return ds[airmass_field_name]
    else:
        try:
            ds = xr.open_dataset(airmass_file_name, decode_cf=True)
        except FileNotFoundError as e:
            logger.error(f"Cannot continue: {e}")
            sys.exit()
        return ds[airmass_field_name]


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


def calculate_total_area(field):
    """
    Calculates the total surface area

    Parameters:
        field (xarray.DataArray): The input data array.

    Returns:
        float: The total area over which the data array is defined.
    """
    R = constants.R_EARTH_m

    # Calculate the latitudinal and longitudinal spacing
    # Assumes 'lat' and 'lon' dim names!
    dlat = np.deg2rad(field.lat.diff('lat').mean())
    dlon = np.deg2rad(field.lon.diff('lon').mean())

    # Calculate the area of each grid cell
    area = R**2 * dlat * dlon * np.cos(np.deg2rad(field.lat))

    # Broadcast area to match the shape of field
    area_expanded = area.broadcast_like(field)

    return area_expanded


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


def check_units(ref_da, dev_da, enforce_units=True) -> bool:
    """ Ensures the units of two xarray DataArrays are the same.

    Parameters:
        ref_da: xarray DataArray
            First data array containing a units attribute.
        dev_da: xarray DataArray
            Second data array containing a units attribute.
    Keyword Args (optional):
        enforce_units: bool
            Whether to stop program if ref and dev units do not match (default: True)
    """
    units_ref = ref_da.units.strip()
    units_dev = dev_da.units.strip()
    if units_ref != units_dev:
        units_match = False
        logger.warning("WARNING: ref and dev concentration units do not match!")
        logger.warning("Ref units: {}".format(units_ref))
        logger.warning("Dev units: {}".format(units_dev))
        if enforce_units:
            # if enforcing units, stop the program if
            # units do not match
            assert units_ref == units_dev, \
                "Units do not match: ref {} and dev {}!".format(
                    units_ref, units_dev)
    else:
        units_match = True
    return units_match


def data_unit_is_mol_per_mol(da):
    """
    Check if the units of an xarray DataArray are mol/mol based on a set
    list of unit strings mol/mol may be.

    Parameters:
        da: xarray DataArray
            Data array containing a units attribute

    Returns:
        is_molmol: bool
            Whether input units are mol/mol
    """
    conc_units = ["mol mol-1 dry", "mol/mol", "mol mol-1"]
    is_molmol = False
    if da.units.strip() in conc_units:
        is_molmol = True
    return is_molmol


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
    return mb


def mb_to_Pa(mb):
    return mb * 100


def Pa_to_hPa(Pa):
    return Pa / 100


def Pa_to_mb(Pa):
    return Pa / 100


def hPa_to_Pa(hPa):
    return hPa / 100


def hPa_to_mb(hPa):
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


def kg_to_du(kg_frac, molar_mass_species, airmass):
    """Convert kg to DU"""
    mol_frac = kg_to_mol(kg_frac, molar_mass_species)
    return mol_to_du(mol_frac, molar_mass_species, airmass)


def mol_to_du(mol_frac, molar_mass_species, airmass):
    """Convert molar fraction to DU"""
    return mol_frac * (airmass / molar_mass_species) * (1 / DU_CONVERSION)


def du_to_mol(du, molar_mass_species, airmass):
    """Convert DU to molar fraction"""
    return du * molar_mass_species * DU_CONVERSION / airmass


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


def kg_to_ppb(kg_frac, molar_mass_species):
    """
    Convert from kg kg⁻¹ to parts per billion (ppb).

    Parameters:
        kg_frac (dataarray values): mass fraction of the species (kg kg⁻¹)
        molar_mass_species: molar mass (mol/mol)
    Returns:
        concentration in parts per billion (ppb)
    """
    return kg_frac * (MOLAR_MASS_AIR / molar_mass_species) * PPB_CONVERSION


def mol_to_ppb(mol_frac):
    """
    Convert from mol mol⁻¹ to parts per billion (ppb).

    Parameters:
        mol_frac (dataarray values): molar fraction of the species (mol mol⁻¹)
    Returns:
        concentration in parts per billion (ppb)
    """
    return mol_frac * PPB_CONVERSION


def ppb_to_mol(ppb):
    """
    Convert from parts per billion (ppb) to mol mol⁻¹.

    Parameters:
        ppb (dataarray values): concentration in parts per billion
    Returns:
        mol_frac: molar fraction of the species (mol mol⁻¹)
    """
    return ppb / PPB_CONVERSION


def calculate_total_column(species, airmass, species_name):
    """
    Calculate the total column of a given species in mol/m², molecules/cm², and Dobson Units (DU).

    Parameters:
        species (xarray.DataArray): Mixing ratio of the species (in mol/mol)
        airmass (xarray.DataArray): Airmass (in kg/m²)
        species_name (str): Name of the species (for DU conversion)

    Returns:
        total_column_mol_m2 (xarray.DataArray): Total column of the species in mol/m²
        total_column_molecules_cm2 (xarray.DataArray): Total column of the species in molecules/cm²
        total_column_du (xarray.DataArray): Total column of the species in Dobson Units (DU)
    """

    # Ensure the dimensions match
    assert species.shape == airmass.shape, "Species and airmass fields must have the same shape"

    # Calculate total column in mol/m²
    total_column_mol_m2 = (species * airmass).sum(dim='lev')

    # Convert total column from mol/m² to molecules/cm²
    total_column_molecules_cm2 = total_column_mol_m2 * AVOGADRO * 1e-4

    # Convert total column from mol/m² to Dobson Units (DU)
    # Note: DU is typically used for ozone; ensure correct usage for other species
    if species_name.lower() == "ozone":
        total_column_du = total_column_mol_m2 * (DU_CONVERSION / AVOGADRO)
    else:
        total_column_du = xr.full_like(total_column_mol_m2, np.nan)  # DU conversion not typically applicable

    return total_column_mol_m2, total_column_molecules_cm2, total_column_du


def _interp(y_src, x_src, x_dest, **kwargs):
    """ Wrapper for SciPy's interp1d """
    return interp1d(x_src, y_src, **kwargs)(x_dest)


def _regrid(ref_arr, in_arr, dim1_name, dim2_name, regrid_dims=(0, 0)):
    """ Main regrid function used in eviz

    The regridding uses SciPy's interp1d function and interpolates
    a 2D field one row at a time.

    Parameters:
       ref_arr (ndarray) : the reference array
        in_arr (ndarray) : the input array
        dim1_name (str) : name of the input dimension
        dim2_name (str) : name of the output dimension
    """
    new_arr = ref_arr

    if regrid_dims[0]:
        new_arr = xr.apply_ufunc(_interp, new_arr,
                                 input_core_dims=[[dim2_name]],
                                 output_core_dims=[[dim2_name]],
                                 exclude_dims={dim2_name},
                                 kwargs={'x_src': ref_arr[dim2_name],
                                         'x_dest': in_arr.coords[dim2_name].values,
                                         'fill_value': "extrapolate"},
                                 dask='allowed', vectorize=True)
        new_arr.coords[dim2_name] = in_arr.coords[dim2_name]
    elif regrid_dims[1]:
        new_arr = xr.apply_ufunc(_interp, new_arr,
                                 input_core_dims=[[dim1_name]],
                                 output_core_dims=[[dim1_name]],
                                 exclude_dims={dim1_name},
                                 kwargs={'x_src': ref_arr[dim1_name],
                                         'x_dest': in_arr.coords[dim1_name].values,
                                         'fill_value': "extrapolate",},
                                 dask='allowed', vectorize=True)
        new_arr.coords[dim1_name] = in_arr.coords[dim1_name]

    return new_arr


def get_species_name(species_name):
    species_map = {
        'o3': 'O3',
        'no2': 'NO2',
        'so2': 'SO2',
        'so4': 'SO4',
        'nh3': 'NH3',
        'bc': 'BC'
    }

    f = str(species_name).lower()
    for key in species_map:
        if key in f:
            return species_map[key]

    print("Field not found in known fields:", f)


@dataclass
class Units:
    """ This class defines attributes and methods to perform unit conversions of xarray data arrays.
        The conversion will be automatic if the fields are registered in eviz's species database and
        the units are supported. Otherwise, the conversion specification can be made in eviz's config
        files (APP and SPECS YAML files). Please see user's guide for more information.

    Parameters:

    config (Config) :
        Representation of the model configuration used to specify data sources and
        user choices for the map generation. The config instance is created at the
        application level.
    """
    config: 'Config'
    species_db: dict = field(init=False)
    airmass: float = field(init=False)

    def __post_init__(self):
        self.logger.debug("Create units converter")
        self.species_db = self.config.species_db
                

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def convert_chem(self, data, species_name, to_unit, air_column_density=None, airmass=None):
        """ Conversion method for chemical species

        Parameters:
            to_unit (str): data destination unit
            species_name (str): species name of the data
            data (xArray): data to undergo unit conversion
        """
        species_name = get_species_name(species_name)

        if species_name not in self.species_db:
            self.logger.warning(f"Species {species_name} not found in data.")
            return data
            # raise ValueError(f"Species {species_name} not found in data.")

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
        vv_to_kg = 1.0
        if 'mol/mol' in from_unit:
            if self.config.map_params[self.config.findex]['to_plot'][self.config.pindex] == 'xy':
                lev_to_plot = int(np.where(self.airmass.coords[self.config.get_model_dim_name('zc')].values == self.config.level)[0])
                self.airmass = self.airmass.isel(lev=lev_to_plot)
                self.airmass = _regrid(self.airmass, data, 'lon', 'lat', regrid_dims=(1, 0))
                self.airmass = _regrid(self.airmass, data, 'lon', 'lat', regrid_dims=(0, 1))

            # Conversion factor for v/v to kg
            # v/v * kg dry air / g/mol dry air * g/mol species = kg species
            if "DU" in to_unit:
                vv_to_kg = self.airmass.values # / MOLAR_MASS_AIR * mw_g

                # Conversion factor for v/v to kg
                # v/v * kg dry air / g/mol dry air * g/mol species = kg species
            elif "g" in to_unit:
                vv_to_kg = self.airmass

                # Conversion factor for v/v to molec/cm3
                # v/v * kg dry air * mol/g dry air * molec/mol dry air /
                #  (area_m2 * box_height ) * 1m3/10^6cm3 = molec/cm3
                area_m2 = calculate_total_area(data)
                box_height = 1.0  # TODO
                if "molec" in to_unit:
                    vv_to_MND = self.airmass / MOLAR_MASS_AIR * AVOGADRO / (area_m2 * box_height) / 1e6

        elif 'kg/kg' in from_unit:
            if "DU" in to_unit:
                self.airmass = _regrid(self.airmass, data, 'lon', 'lat', regrid_dims=(1, 0))
                self.airmass = _regrid(self.airmass, data, 'lon', 'lat', regrid_dims=(0, 1))
                vv_to_kg = self.airmass  # / MOLAR_MASS_AIR * mw_g

                # Conversion factor for v/v to kg
                # v/v * kg dry air / g/mol dry air * g/mol species = kg species
            if "ppb" in to_unit:
                pass

                # Conversion factor for v/v to kg
                # v/v * kg dry air / g/mol dry air * g/mol species = kg species

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

        # TODO: needs testing
        elif to_unit == 'kg/kg':
            # Calculate total column in mol/m²
            total_column_mol_m2 = data * vv_to_kg
            # Convert total column from mol/m² to molecules/cm²
            data.values = total_column_mol_m2 * AVOGADRO * 1e-4

            # kgm2 = data * get_airmass(self.config)
            # molm2 = kgm2 / self.config.species_db[sp_name]['MW_kg']
            # moleculem2 = molm2 * AVOGADRO
            # data.values = moleculem2 * 1e-4

        elif 'ppb' in to_unit:
            pass

        # TODO: needs testing
        elif 'molmol-1' in to_unit or 'mol/mol' in to_unit:

            if "DU" in to_unit:
                data = data * vv_to_kg

            elif "g" in to_unit:
                data = data * vv_to_kg
            else:
                pass

        # TODO: needs testing
        elif to_unit == 'DU':
            # Calculate kg/kg
            kgkg = data * (molar_mass_species/MOLAR_MASS_AIR)
            # Calculate kgm2 
            kgm2 = kgkg * vv_to_kg
            # Calculate molm2
            molm2 = kgm2 / molar_mass_species
            # Calculate moleculecm2
            moleculecm2 = (molm2 * AVOGADRO) * 1e-4
            # Calculate DU
            total_column_du = moleculecm2 / DU_CONVERSION
            total_column_du.attrs["units"] = to_unit
            return total_column_du

        else:
            raise ValueError(
                f"Units ({to_unit}) in variable {species_name} are not supported")

        conversion_functions = {
            'moles': {
                'g': lambda data: moles_to_mass(data, self.species_db['MW_g']),
            },
            'g': {
                'mg': lambda data: g_to_mg(data),
                'kg': lambda data: g_to_kg(data),
                'moles': lambda data: mass_to_moles(data, self.species_db['MW_g']),
            },
            'mg': {
                'g': lambda data: mg_to_g(data),
                'kg': lambda data: mg_to_kg(data),
            },
            'kg': {
                'g': lambda data: kg_to_g(data),
                'mg': lambda data: kg_to_mg(data),
            },
            'mol/mol': {
                'kg/kg': lambda data: mol_to_kg(data, molar_mass_species),
                # 'DU': lambda data: mol_to_du(data, molar_mass_species, airmass),
                # 'DU': total_column_du,
                'ppb': lambda data: mol_to_ppb(data),
                'molecules/cm2': lambda data: mol_to_molecules_cm2(data, air_column_density)
            },
            'kg/kg': {
                'mol/mol': lambda data: kg_to_mol(data, molar_mass_species),
                'DU': lambda data: kg_to_du(data, molar_mass_species, airmass),
                'ppb': lambda data: kg_to_ppb(data, molar_mass_species)
            },
            'DU': {
                'mol/mol': lambda data: du_to_mol(data, molar_mass_species, airmass)
            },
            'ppb': {
                'mol/mol': lambda data: ppb_to_mol(data)
            },
            'mol': {
                'mol/mol': lambda data: moles_to_mass(data, molar_mass_species)
            },
        }

        if from_unit not in conversion_functions or to_unit not in conversion_functions[from_unit]:
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
            self.logger.warning(f"Units are identical {from_unit} == {to_unit}...returning.")
            return data
        self.logger.debug(f"Convert {species_name} units from {from_unit} to {to_unit}")

        conversion_functions = {
            'mb': {
                'hPa': lambda data: mb_to_hPa(data),
                'Pa': lambda data: mb_to_Pa(data),
            },
            'hPa': {
                'mb': lambda data: hPa_to_mb(data),
                'Pa': lambda data: hPa_to_Pa(data),
            },
            'Pa': {
                'hPa': lambda data: Pa_to_hPa(data),
                'mb': lambda data: Pa_to_mb(data),
            },
            'g': {
                'mg': lambda data: g_to_mg(data),
                'kg': lambda data: g_to_kg(data),
            },
            'mg': {
                'g': lambda data: mg_to_g(data),
                'kg': lambda data: mg_to_kg(data),
            },
            'kg': {
                'g': lambda data: kg_to_g(data),
                'mg': lambda data: kg_to_mg(data),
            },
            'F': {
                'C': lambda data: f_to_c(data),
                'K': lambda data: f_to_k(data),
            },
            'C': {
                'F': lambda data: c_to_f(data),
                'K': lambda data: c_to_k(data),
            },
            'K': {
                'F': lambda data: k_to_f(data),
                'C': lambda data: k_to_c(data),
            },
        }

        if from_unit not in conversion_functions or to_unit not in conversion_functions[from_unit]:
            raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported.")

        new_data = conversion_functions[from_unit][to_unit](data)
        new_data.attrs["units"] = to_unit
        return new_data
