import logging
import os
from abc import ABC, abstractmethod
from glob import glob

import pandas as pd
import xarray as xr
import numpy as np
import eviz.lib.const as constants
import eviz.lib.utils as u
from dataclasses import dataclass, field
from typing import Any

FILE_EXTENSIONS = (
    'nc',
    'nc4',
    'hdf',
    'hdf4',
    'h5',
    'hdf5',
    'csv',
    'dat',
    'grib',
    'grib2',
)


@dataclass
class DataSource(xr.Dataset, ABC):
    """Abstract class that defines a data source object.

    All data sources are represented as Xarray datasets internally.
    Subclasses must implement the `load_data` method to populate the dataset.
    """
    model_name: str = None  # Name of the model this data source belongs to

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @abstractmethod
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data into an Xarray dataset."""
        raise NotImplementedError("Subclasses must implement the load_data method.")

    def validate_data(self):
        """Validate the loaded data."""
        self.logger.debug("Validating data")
        # Add validation logic here
        
    def get_variable(self, variable_name):
        """Get a variable from the dataset by name.
        
        Args:
            variable_name (str): The name of the variable to retrieve
            
        Returns:
            xarray.DataArray or None: The requested variable, or None if not found
        """
        self.logger.debug(f"Getting variable: {variable_name}")
        try:
            if variable_name in self:
                return self[variable_name]
            else:
                self.logger.warning(f"Variable {variable_name} not found in dataset")
                return None
        except Exception as e:
            self.logger.error(f"Error retrieving variable {variable_name}: {e}")
            return None


@dataclass
class CSVDataSource(DataSource):
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load CSV data into an Xarray dataset."""
        self.logger.debug(f"Loading CSV data from {file_path}")
        files = glob(file_path)
        combined_data = pd.DataFrame()

        if "*" in file_path:
            for f in files:
                this_data = pd.read_csv(f)
                combined_data = pd.concat([combined_data, this_data], ignore_index=True)
        else:
            this_data = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, this_data], ignore_index=True)

        # Convert the Pandas DataFrame to an Xarray dataset
        dataset = combined_data.to_xarray()
        self._process_data(dataset)
        return dataset

    def _process_data(self, dataset: xr.Dataset):
        """Process the loaded CSV data."""
        self.logger.debug("Processing CSV data")
        # Add any additional processing logic here


@dataclass
class HDF5DataSource(DataSource):
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load HDF5 data into an Xarray dataset."""
        self.logger.debug(f"Loading HDF5 data from {file_path}")
        dataset = xr.open_dataset(file_path, engine="h5netcdf")
        self._process_data(dataset)
        return dataset

    def _process_data(self, dataset: xr.Dataset):
        """Process the loaded HDF5 data."""
        self.logger.debug("Processing HDF5 data")
        # Add any additional processing logic here


@dataclass
class NetCDFDataSource(DataSource):
    datasets: dict = field(default_factory=dict)  # Dictionary to store datasets by file name

    def load_data(self, file_paths: list) -> dict:
        """Load one or more NetCDF files into a dictionary of Xarray datasets.

        Parameters:
            file_paths (list or str): A list of file paths or a single file path (string).

        Returns:
            dict: A dictionary where keys are file names and values are Xarray datasets.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]  # Convert single file path to a list

        self.logger.debug(f"Loading NetCDF data from: {file_paths}")

        for file_path in file_paths:
            try:
                if "*" in file_path:
                    # Handle multiple files using a glob pattern
                    dataset = xr.open_mfdataset(file_path, decode_cf=True, combine="by_coords")
                    self.logger.info(f"Loaded multiple NetCDF files matching pattern: {file_path}")
                else:
                    # Handle a single file
                    dataset = xr.open_dataset(file_path, decode_cf=True)
                    self.logger.info(f"Loaded single NetCDF file: {file_path}")

                # Store the dataset in the dictionary using the file name as the key
                file_name = os.path.basename(file_path)
                self.datasets[file_name] = dataset

            except FileNotFoundError as exc:
                self.logger.error(f"Error loading NetCDF file: {file_path}. Exception: {exc}")
                raise

        self.logger.info(f"Successfully loaded {len(self.datasets)} NetCDF datasets.")
        return self.datasets

    def _process_data(self, dataset: xr.Dataset):
        """Process the loaded NetCDF data."""
        self.logger.debug("Processing NetCDF data")
        # Add any additional processing logic here


class DataSourceFactory:
    """Factory class to create DataSource instances based on file extensions."""

    @staticmethod
    def get_data_class(file_extension: str) -> DataSource:
        """Return the appropriate DataSource class based on the file extension."""
        if file_extension in ["csv", "dat"]:
            return CSVDataSource()
        elif file_extension in ["hdf5", "h5"]:
            return HDF5DataSource()
        elif file_extension in ["nc", "nc4"]:
            return NetCDFDataSource()
        else:
            raise ValueError(f"No data source specified for file extension: {file_extension}")


@dataclass
class DataProcessor(DataSource):
    """ This class provides methods to access and process EVIZ data sources.

    An instance of DataProcessor is created for each model and its associated file list.
    To maintain model agnosticism, the names for the model's coordinates are represented
    by generic names as xc, yc, zc, and tc. These names are mapped to the actual model
    coordinate names in the YAML file meta_coordinates.yaml. Likewise, the data attributes
    are stored and mapped in a dictionary defined in meta_attributes.yaml.

    Parameters:
        model_name (str) : The name of the `supported` model.
        file_list (list) : The list of data file names.
        meta_coords (dict) : A dictionary of metadata coordinate names from the file list.
        meta_attrs (dict) : A dictionary of metadata attribute names from the file list.

    """
    model_name: str = None
    file_list: dict = field(default_factory=dict)
    meta_coords: dict = field(default_factory=dict)
    meta_attrs: dict = field(default_factory=dict)
    season: Any = None

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        self.datasets = []

    def load_data(self, file_path):
        pass

    def process_data(self):
        factory = DataSourceFactory()
        for fid, filename in self.file_list.items():
            file_path = filename['name']
            if 'location' in filename:
                file_path = os.path.join(filename['location'], filename['name'])
            file_extension = file_path.split(".")[-1]
            if file_extension not in FILE_EXTENSIONS:
                self.logger.error(f"File extension '{file_extension}' not found. Will assume NetCDF4.")
                file_extension = 'nc4'
            data_class = factory.get_data_class(file_extension)
            self.datasets.append(data_class.load_data(file_path))

        self.logger.info(f"Processing data for model: {self.model_name}")

    def get_field(self, name, ds_index=0):
        """ Extract field from xarray Dataset

        Parameters:
            name (str) : name of field to extract from dataset
            ds_index (int) : fid index associated with dataset containing field name

        Returns:
            DataArray containing field data
        """
        try:
            self.logger.debug(f" -> getting field {name}")
            return self.datasets[ds_index][name]
        except Exception as e:
            self.logger.error('key error: %s, not found' % str(e))
        return None

    def get_meta_attrs(self, data, key):
        """ Get attributes associated with a key"""
        if self.model_name in self.meta_attrs[key]:
            return self.meta_attrs[key][self.model_name]
        return None

    @staticmethod
    def get_attrs(data, key):
        """ Get attributes associated with a key"""
        for attr in data.attrs:
            if key == attr:
                return data.attrs[key]
            else:
                continue
        return None

    @staticmethod
    def adjust_units(units):

        # Error check arguments
        if not isinstance(units, str):
            raise TypeError("Units must be of type str!")

        # Strip all spaces in the unit string
        units_squeezed = units.replace(" ", "")

        if units_squeezed in ["kg/m2/s", "kgm-2s-1", "kgm^-2s^-1"]:
            unit_desc = "kg/m2/s"

        elif units_squeezed in [
            "kgC/m2/s",
            "kgCm-2s-1",
            "kgCm^-2s^-1",
            "kgc/m2/s",
            "kgcm-2s-1",
            "kgcm^-2s^-1",
        ]:
            unit_desc = "kgC/m2/s"

        elif units_squeezed in ["molec/cm2/s", "moleccm-2s-1", "moleccm^-2s^-1"]:
            unit_desc = "molec/cm2/s"

        else:
            unit_desc = units_squeezed

        return unit_desc

    @staticmethod
    def convert_kg_to_target_units(data_kg, target_units, kg_to_kgC):

        # Convert to target unit
        if target_units == "Tg":
            data = data_kg * 1e-9

        elif target_units == "Tg C":
            data = data_kg * kg_to_kgC * 1.0e-9

        elif target_units == "Gg":
            data = data_kg * 1e-6

        elif target_units == "Gg C":
            data = data_kg * kg_to_kgC * 1.0e-6

        elif target_units == "Mg":
            data = data_kg * 1e-3

        elif target_units == "Mg C":
            data = data_kg * kg_to_kgC * 1.0e-3

        elif target_units == "kg":
            data = data_kg

        elif target_units == "kg C":
            data = data_kg * kg_to_kgC

        elif target_units == "g":
            data = data_kg * 1e3

        elif target_units == "g C":
            data = data_kg * kg_to_kgC * 1.0e3

        else:
            msg = "Target units {} are not yet supported!".format(target_units)
            raise ValueError(msg)

        # Return converted data
        return data

    @staticmethod
    def convert_units(
            dr,
            species_name,
            species_properties,
            target_units,
            interval=2678400.0,
            area_m2=None,
            delta_p=None,
            box_height=None):

        # Get species molecular weight information
        if "MW_g" in species_properties.keys():
            mw_g = species_properties.get("MW_g")
        else:
            msg = "Cannot find molecular weight MW_g for species {}".format(
                species_name)
            msg += "!\nPlease add the MW_g field for {}".format(species_name)
            msg += " to the species_database.yml file."
            raise ValueError(msg)

        # If the species metadata does not contain EmMW_g, use MW_g instead
        if "EmMW_g" in species_properties.keys():
            emitted_mw_g = species_properties.get("EmMW_g")
        else:
            emitted_mw_g = mw_g

        # If the species metadata does not containe MolecRatio, use 1.0 instead
        if "MolecRatio" in species_properties.keys():
            moles_C_per_mole_species = species_properties.get("MolecRatio")
        else:
            moles_C_per_mole_species = 1.0

        # ==============================
        # Compute conversion factors
        # ==============================

        # Physical constants
        avo = constants.AVOGADRO  # molecules/mole
        mw_air = constants.MW_AIR_g  # g/mole
        g0 = constants.G  # m/s2

        # Get a consistent value for the units string
        # (ignoring minor differences in formatting)
        units = DataSource.adjust_units(dr.units)

        # Error checks
        if units == "molmol-1dry" and area_m2 is None:
            raise ValueError(
                "Conversion from {} to {} for {} requires area_m2 as input".format(
                    units, target_units, species_name
                )
            )
        if units == "molmol-1dry" and delta_p is None:
            raise ValueError(
                "Conversion from {} to {} for {} requires delta_p as input".format(
                    units, target_units, species_name
                )
            )
        if "g" in target_units and mw_g is None:
            raise ValueError(
                "Conversion from {} to {} for {} requires MW_g definition in species_database.yml".format(
                    units, target_units, species_name))

        # Conversion factor for kg species to kg C
        kg_to_kgC = (emitted_mw_g * moles_C_per_mole_species) / mw_g

        # Mass of dry air in kg (required when converting from v/v)
        if 'molmol-1' in units:
            air_mass = delta_p * 100.0 / g0 * area_m2

            # Conversion factor for v/v to kg
            # v/v * kg dry air / g/mol dry air * g/mol species = kg species
            if "g" in target_units:
                vv_to_kg = air_mass / mw_air * mw_g

            # Conversion factor for v/v to molec/cm3
            # v/v * kg dry air * mol/g dry air * molec/mol dry air /
            #  (area_m2 * box_height ) * 1m3/10^6cm3 = molec/cm3
            if "molec" in target_units:
                vv_to_MND = air_mass / mw_air * avo / (area_m2 * box_height) / 1e6

        # ================================================
        # Get number of seconds per time in dataset
        # ================================================

        # Number of seconds is passed via the interval argument
        numsec = interval

        # Special handling is required if multiple times in interval (for
        # broadcast)
        if len([interval]) > 1:
            if 'time' in dr.dims:
                # Need to right pad the interval array with new axes up to the
                # time dim of the dataset to enable broadcasting
                numnewdims = len(dr.dims) - (dr.dims.index('time') + 1)
                for _ in range(numnewdims):
                    numsec = numsec[:, np.newaxis]
            else:
                # Raise an error if no time in dataset but interval has length > 1
                raise ValueError(
                    'Interval passed to convert_units has length greater than one but data array has no time dimension')

        # ==============================
        # Compute target units
        # ==============================

        if units == "kg/m2/s":
            data_kg = dr * area_m2
            data_kg = data_kg.values * numsec
            data = DataSource.convert_kg_to_target_units(data_kg, target_units, kg_to_kgC)

        elif units == "kgC/m2/s":
            data_kg = dr * area_m2 / kg_to_kgC
            data_kg = data_kg.values * numsec
            data = DataSource.convert_kg_to_target_units(data_kg, target_units, kg_to_kgC)

        elif units == "kg":
            data_kg = dr.values
            data = DataSource.convert_kg_to_target_units(data_kg, target_units, kg_to_kgC)

        elif units == "kgC":
            data_kg = dr.values / kg_to_kgC
            data = DataSource.convert_kg_to_target_units(data_kg, target_units, kg_to_kgC)

        #    elif units == 'molec/cm2/s':
        #        # Implement later

        #    elif units == 'atomsC/cm2/s':
        #         implement later

        elif 'molmol-1' in units:

            if "g" in target_units:
                data_kg = dr.values * vv_to_kg
                data = DataSource.convert_kg_to_target_units(data_kg, target_units, kg_to_kgC)

            elif "molec" in target_units:
                data = dr.values * vv_to_MND

        else:
            raise ValueError(
                "Units ({}) in variable {} are not supported".format(
                    units, species_name))

        # ==============================
        # Return result
        # ==============================

        # Create a new DataArray.  This will be exactly the same as the old
        # DataArray, except that the data will have been converted to the
        # target_units, and the units string will have been adjusted accordingly.
        dr_new = xr.DataArray(
            data, name=dr.name, coords=dr.coords, dims=dr.dims, attrs=dr.attrs
        )
        dr_new.attrs["units"] = target_units

        return dr_new

    @staticmethod
    def check_units(ref_da, dev_da, enforce_units=True):
        units_ref = ref_da.units.strip()
        units_dev = dev_da.units.strip()
        if units_ref != units_dev:
            units_match = False
            print("WARNING: ref and dev concentration units do not match!")
            print("Ref units: {}".format(units_ref))
            print("Dev units: {}".format(units_dev))
            if enforce_units:
                # if enforcing units, stop the program if
                # units do not match
                assert units_ref == units_dev, \
                    "Units do not match: ref {} and dev {}!".format(
                        units_ref, units_dev)
        else:
            units_match = True
        return units_match

    @staticmethod
    def data_unit_is_mol_per_mol(da):
        conc_units = ["mol mol-1 dry", "mol/mol", "mol mol-1"]
        is_molmol = False
        if da.units.strip() in conc_units:
            is_molmol = True
        return is_molmol

    def get_ds_index(self):
        return self.ds_index

    def get_datasets(self):
        return self.datasets

    def get_dataset(self, i):
        return self.datasets[i]


def get_season_from_file(file_name):
    if "ANN" in file_name:
        return "ANN"
    elif "JJA" in file_name:
        return "JJA"
    elif "DJF" in file_name:
        return "DJF"
    elif "SON" in file_name:
        return "SON"
    elif "MAM" in file_name:
        return "MAM"
    else:
        return None


def make_fake_timeseries_dataset(path=None):
    np.random.seed(123)
    times = pd.date_range("2000-01-01", "2001-12-31", name="time")
    annual_cycle = np.sin(2 * np.pi * (times.dayofyear.values / 365.25 - 0.28))

    base = 10 + 15 * annual_cycle.reshape(-1, 1)
    tmin_values = base + 3 * np.random.randn(annual_cycle.size, 3)
    tmax_values = base + 10 + 3 * np.random.randn(annual_cycle.size, 3)

    ds = xr.Dataset(
        {
            "tmin": (("time", "location"), tmin_values),
            "tmax": (("time", "location"), tmax_values),
        },
        {"time": times, "location": ["DC", "MD", "VA"]},
    )
    if path is None:
        path = os.path.join(constants.ROOT_FILEPATH, 'test/data')
    u.mkdir_p(path)
    ds.to_netcdf(os.path.join(path, 'timeseries.nc'), format='NETCDF4')
    return ds


def make_fake_4D_dataset(nt=366, path=None):
    nt = nt
    nx = 20
    ny = 15
    nz = 10

    temp = np.zeros((nt, nz, ny, nx))
    rh = np.zeros((nt, nz, ny, nx))
    o3 = np.zeros((nt, nz, ny, nx))
    no2 = np.zeros((nt, nz, ny, nx))
    qv = np.zeros((nt, nz, ny, nx))
    ps = np.zeros((nt, ny, nx))

    times = pd.date_range("2000-01-01",
                          periods=nt,
                          freq=pd.DateOffset(days=1),
                          name="time")
    t0 = 273.
    q0 = 50.
    p0 = 1e5
    o30 = 1e-6
    no20 = 1e-6
    qv0 = 0.01

    lon = np.zeros((ny, nx))
    lat = np.zeros((ny, nx))
    lev = np.linspace(nz, 1., 10) * 1e4
    for y in range(ny):
        for x in range(nx):
            lon[y, x] = 100 + y * 10 + x / 10
            lat[y, x] = 200 + x * 10 + y / 10

    for t in range(nt):
        tt = (nt - t) * p0 / 100
        for y in range(ny):
            for x in range(nx):
                ps[t, y, x] = np.random.normal(p0, 1000) + \
                              np.sin(x + tt * np.pi / 4) + \
                              np.cos(y + tt * np.pi / 6)
    for t in range(nt):
        tt = (nt - t) * 10
        for z in range(nz):
            zz = nz - z
            for y in range(ny):
                for x in range(nx):
                    temp[t, z, y, x] = np.random.normal(t0, 20) + \
                                       np.sin(x + tt * np.pi / 4) + \
                                       np.cos(y + tt * np.pi / 6) + \
                                       np.exp(zz / 10.)
                    rh[t, z, y, x] = np.random.normal(q0, 20) + \
                                     np.sin(x + tt * np.pi / 4) + \
                                     np.cos(y + tt * np.pi / 6) + \
                                     np.exp(zz / 20.)
                    o3[t, z, y, x] = np.random.normal(o30, 1e-7) + \
                                     np.sin(x + tt * np.pi / 4) + \
                                     np.cos(y + tt * np.pi / 6) + \
                                     np.exp(zz / 20.)
                    no2[t, z, y, x] = np.random.normal(no20, 1e-7) + \
                                     np.sin(x + tt * np.pi / 4) + \
                                     np.cos(y + tt * np.pi / 6) + \
                                     np.exp(zz / 20.)
                    qv[t, z, y, x] = np.random.normal(qv0, 0.001) + \
                                     np.sin(x + tt * np.pi / 4) + \
                                     np.cos(y + tt * np.pi / 6) + \
                                     np.exp(zz / 20.)

    ds = xr.Dataset({
        'sfc_press': xr.DataArray(
            data=ps,
            dims=['time', 'lat', 'lon'],
            coords={'lons': (['lat', 'lon'], lon),
                    'lats': (['lat', 'lon'], lat),
                    'time': times
                    },
            attrs={'long_name': 'Surface Pressure', 'units': 'Pa'}
        ),
        'air_temp': xr.DataArray(
            data=temp,
            dims=['time', 'lev', 'lat', 'lon'],
            coords={'lons': (['lat', 'lon'], lon),
                    'lats': (['lat', 'lon'], lat),
                    'lev': (['lev'], lev),
                    'time': times,
                    },
            attrs={'long_name': 'Air Temperature', 'units': 'K'}
        ),
        'rel_humid': xr.DataArray(
            data=rh,
            dims=['time', 'lev', 'lat', 'lon'],
            coords={'lons': (['lat', 'lon'], lon),
                    'lats': (['lat', 'lon'], lat),
                    'lev': (['lev'], lev),
                    'time': times,
                    },
            attrs={'long_name': 'Relative Humidity', 'units': '%'}
        ),
        'spc_humid': xr.DataArray(
            data=qv,
            dims=['time', 'lev', 'lat', 'lon'],
            coords={'lons': (['lat', 'lon'], lon),
                    'lats': (['lat', 'lon'], lat),
                    'lev': (['lev'], lev),
                    'time': times,
                    },
            attrs={'long_name': 'Specific Humidity', 'units': 'kg kg-1'}
        ),
        'ozone': xr.DataArray(
            data=o3,
            dims=['time', 'lev', 'lat', 'lon'],
            coords={'lons': (['lat', 'lon'], lon),
                    'lats': (['lat', 'lon'], lat),
                    'lev': (['lev'], lev),
                    'time': times,
                    },
            attrs={'long_name': 'Ozone dry mixing ratio', 'units': 'mol mol-1 dry'}
        ),
        'nitrogen_dioxide': xr.DataArray(
            data=no2,
            dims=['time', 'lev', 'lat', 'lon'],
            coords={'lons': (['lat', 'lon'], lon),
                    'lats': (['lat', 'lon'], lat),
                    'lev': (['lev'], lev),
                    'time': times,
                    },
            attrs={'long_name': 'NO2 dry mixing ratio', 'units': 'mol mol-1 dry'}
        ),
    },
        attrs={'Title': 'EViz test data',
               'Start_date': '2022-01-01',
               'MAP_PROJECTION': 'Lambert Conformal',
               'SOUTH_WEST_CORNER_LAT': 35.,
               'SOUTH_WEST_CORNER_LON': -105.,
               'TRUELAT1': 40.,
               'TRUELAT2': 35.,
               'STANDARD_LON': -99.,
               }
    )
    if path is None:
        path = os.path.join(constants.ROOT_FILEPATH, 'test/data')
    u.mkdir_p(path)
    ds.to_netcdf(os.path.join(path, 'spacetime.nc'), format='NETCDF4')
    return ds


def make_fake_column_dataset(path=None):
    np.random.seed(123)
    times = pd.date_range(start='2000-01-01',
                          freq=pd.DateOffset(months=1),
                          periods=12)
    ds = xr.Dataset({
        'SWdown': xr.DataArray(
            data=np.random.random(12),  # enter data here
            dims=['time'],
            coords={'time': times},
            attrs={
                '_FillValue': -999.9,
                'units': 'W/m2'
            }
        ),
        'LWdown': xr.DataArray(
            data=np.random.random(12),  # enter data here
            dims=['time'],
            coords={'time': times},
            attrs={
                '_FillValue': -999.9,
                'units': 'W/m2'
            }
        )
    },
        attrs={'example_attr': 'this is a global attribute'}
    )
    if path is None:
        path = os.path.join(constants.ROOT_FILEPATH, 'test/data')
    u.mkdir_p(path)
    ds.to_netcdf(os.path.join(path, 'column.nc'), format='NETCDF4')
    return ds
