from dataclasses import dataclass
from typing import Any, List
import logging
import numpy as np
from scipy.interpolate import interp1d
import xarray as xr
from xarray import DataArray

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.lib.xarray_utils import get_dst_attribute
from eviz.lib import const as constants
from eviz.lib.data.data_utils import apply_conversion
from eviz.lib.data.factory import DataSourceFactory


@dataclass
class Overlays:
    """ Class that define overlays

    Example of overlays include:
        - specialized contours
        - specialized line plots

    Parameters:
        plot_type (str) : type of plot
        config_manager (ConfigManager) : ConfigManager object

    """
    config: ConfigManager
    plot_type: str
    # data: list = field(default_factory=list)

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        # self.data = []
        self.tropp = None
        self.tropp_conversion = None
        self.trop_ok = False

    def process_data(self):
        """ Get tropopause field and apply to a given experiment _name

        Parameters:
            ds_meta (dict) : Dataset metadata
            findex (int) : Dataset index (default=0, i.e. just one dataset)
       """
        if self.config.use_trop_height:
            if self.config.findex not in self.config.trop_height_file_list:
                return
            try:
                findex = self.config.findex
                # The experiment id that we want to apply the overlay to:
                exp_id = self.config.trop_height_file_list[findex]['exp_id']
                field_exp_id = self.config.file_list[findex]['exp_id']
                if exp_id != field_exp_id:
                    self.trop_ok = False
                    return None
                trop_filename = self.config.trop_height_file_list[findex]['filename']
                self.logger.debug(f"Processing {trop_filename}...")

                tropp = None
                with xr.open_dataset(trop_filename) as f:
                    trop_field = self.config.trop_height_file_list[findex]['trop_field_name']
                    tropp = f.data_vars.get(trop_field)
                    # TODO: remove assumption that 'time=0' is used
                    tropp = tropp.isel(time=0)

                    # TODO: remove units stuff from here
                    units = get_dst_attribute(tropp, 'units')
                    if units == 'Pa':
                        self.tropp_conversion = 1 / 100.0
                    elif units == 'hPa':
                        self.tropp_conversion = 1.0
                    else:
                        self.tropp_conversion = 1.0

                self.trop_ok = True
                if 'yz' in self.plot_type:
                    return tropp.mean(dim='lon') * self.tropp_conversion
                else:  # not supported
                    return None

            except FileNotFoundError:
                self.logger.warning('The given tropopause name was not found')
                return None
        else:
            self.trop_ok = False
            return None

    def sphum_field(self, ds_meta, findex=0):
        """ Get specific humidity field and apply to a given experiment _name

        Parameters:
            ds_meta (dict) : Dateset metadata
            findex (int) : Dataset index (default=0, i.e. just one dataset)
        """
        if not self.config.use_sphum_conv:
            return
        radionuclides = ['Be10', 'Be10s', 'Be7', 'Be7s', 'Pb210', 'Rn222']

        to_convert = set(self.config.to_plot).intersection(set(radionuclides))
        if to_convert:
            if ds_meta['sphum_conv_meta'][findex]['exp_name'] == ds_meta['exp_name']:
                try:
                    sphum_filename = ds_meta['sphum_conv_meta'][findex]['filename']
                    self.logger.debug(f"Processing {sphum_filename}...")

                    with xr.open_dataset(sphum_filename) as f:
                        sphum_field = ds_meta['sphum_conv_meta'][findex]['sphum_field_name']
                        specific_hum = f.data_vars.get(sphum_field)
                        # Assume 'time=0' is used
                        self.specific_hum = specific_hum.isel(time=0)

                except FileNotFoundError:
                    self.logger.warning(
                        self.config.get_sphum_filename()+" file could not be opened. Setting QV=0.0")
                    self.specific_hum = 0.0  # assume dry air conditions
                else:
                    for name in to_convert:
                        self._convert_radionuclide_units(name, 'mol mol-1')
            else:
                self.logger.warning(
                    "QV file was not specified. Setting QV=0.0")
                self.specific_hum = 0.0  # assume dry air conditions

    def _convert_radionuclide_units(self, species_name, target_units):
        # TODO: move the actual conversions to units.py
        """ CCM-specific conversion function for radionuclides """
        ds_index = self.config.data_source.get_ds_index()
        if not self.config.use_sphum_field:
            return
        if self.config.data_source.data_unit_is_mol_per_mol(self.config.data_source.datasets[ds_index]['vars'][species_name]):
            return
        self.logger.debug(f"Converting {species_name} units to {target_units}")

        # Get species molecular weight information
        if "MW_g" in self.config.species_db[species_name].keys():
            mw_g = self.config.species_db[species_name].get("MW_g")
        else:
            msg = "Cannot find molecular weight MW_g for species {}".format(
                species_name)
            msg += "!\nPlease add the MW_g field for {}".format(species_name)
            msg += " to the species_database.yaml file."
            raise ValueError(msg)

        mw_air = constants.MW_AIR_g  # g/mole

        rn_arr = self.config.datasets[ds_index]['vars'][species_name]

        # TODO: regrid if different resolutions
        data = (rn_arr / (1. - self.specific_hum)) * (mw_air / mw_g)

        rn_new = xr.DataArray(
            data, name=rn_arr.name, coords=rn_arr.coords, dims=rn_arr.dims, attrs=rn_arr.attrs
        )
        rn_new.attrs["units"] = target_units

        self.config.data_source.datasets[ds_index]['ptr'][rn_arr.name] = rn_new

    def get_processed_data(self, field: str) -> Any | None:
        for processed_field, data_array in self.data:
            if processed_field == field:
                return data_array
        return None

    @staticmethod
    def process_difference(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        return data1 - data2

    @staticmethod
    def process_xy(data: Any, field: str) -> np.ndarray:
        data_variable = data[field]
        data_array = data_variable.values

        if data_array.ndim == 3:
            data_array = data_array[0]

        return data_array

    @staticmethod
    def process_yz(data: Any, field: str) -> np.ndarray:
        data_variable = data[field]
        data_array = data_variable.values

        if data_array.ndim == 3:
            data_array = data_array[0]

        data_array = np.mean(data_array, axis=0)

        return data_array

    @staticmethod
    def process_scat(data: Any, field: str) -> np.ndarray:
        data_variable = data[field]
        data_array = data_variable.values

        return data_array

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
            return self.config.readers[ds_index].datasets[ds_index]['vars'][name]
        except Exception as e:
            self.logger.error('key error: %s, not found' % str(e))
        return None

    def get_meta_attrs(self, data, key):
        """ Get attributes associated with a key"""
        if self.config.source_names[key] in self.config.meta_attrs[key]:
            return self.config.meta_attrs[key][self.config.source_names[key]]
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


@dataclass
class Interp:
    config_manager: ConfigManager
    data: List[Any]

    def __post_init__(self):
        self.logger.info("Start init")
        
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def regrid(self, pid):
        """ Wrapper for regrid method

        This function regrids two fields (if necessary)

        Parameters:
            pid (str) : a plot identifier

        Returns:
            Regridded fields
        """
        # TODO: get rid of ax/ax_opts dependency
        return self._regrid_check(pid)

    # Interpolation (AKA regrid) methods
    def _regrid_check(self, pid, compute_diff=True):
        """ Main regrid method """
        xc = self.config_manager.get_model_dim_name('xc')
        yc = self.config_manager.get_model_dim_name('yc')
        zc = self.config_manager.get_model_dim_name('zc')
        dim1, dim2 = xc, yc
        if 'yz' in pid:
            dim1, dim2 = yc, zc
        d1 = self.data[0]
        d2 = self.data[1]

        da1_size = d1.size
        da2_size = d2.size
        if da1_size < da2_size:
            d2 = self._regrid(d2, d1, dim1, dim2, regrid_dims=(1, 0))
            d2 = self._regrid(d2, d1, dim1, dim2, regrid_dims=(0, 1))
        elif da1_size > da2_size:
            d1 = self._regrid(d1, d2, dim1, dim2, regrid_dims=(1, 0))
            d1 = self._regrid(d1, d2, dim1, dim2, regrid_dims=(0, 1))
        elif da1_size == da2_size:
            d1 = self._regrid(d1, d2, dim1, dim2, regrid_dims=(1, 0))
            d1 = self._regrid(d1, d2, dim1, dim2, regrid_dims=(0, 1))
            d2 = self._regrid(d2, d1, dim1, dim2, regrid_dims=(1, 0))
            d2 = self._regrid(d2, d1, dim1, dim2, regrid_dims=(0, 1))

        if compute_diff:
            if self.config_manager.ax_opts['add_extra_field_type']:
                data_diff = self._compute_diff_type(d1, d2).squeeze()
            else:
                data_diff = (d1 - d2).squeeze()
            coords = data_diff.coords
            return data_diff, coords[dim1].values, coords[dim2].values
        else:
            return d1, d2

    def _select_yrange(self, data2d, name):
        """ For 3D fields, select vertical level range to use

        Parameters:
            data2d (ndarray) : A 2D array of an ESM field
            name (str) : field name

        Returns:
            sliced data array
        """
        if 'zrange' in self.config_manager.spec_data[name]['yzplot']:
            if self.config_manager.spec_data[name]['yzplot']['zrange']:
                lo_z = self.config_manager.spec_data[name]['yzplot']['zrange'][0]
                hi_z = self.config_manager.spec_data[name]['yzplot']['zrange'][1]
                if hi_z >= lo_z:
                    self.logger.error(f"Upper level value ({hi_z}) must be less than low level value ({lo_z})")
                    return
                lev = self.config_manager.get_model_dim_name('zc')
                min_index, max_index = 0, len(data2d.coords[lev].values) - 1
                for k, v in enumerate(data2d.coords[lev]):
                    if data2d.coords[lev].values[k] == lo_z:
                        min_index = k
                for k, v in enumerate(data2d.coords[lev]):
                    if data2d.coords[lev].values[k] == hi_z:
                        max_index = k
                return data2d[min_index:max_index + 1, :]
            else:
                return data2d
        else:
            return data2d

    def _compute_diff_type(self, d1, d2):
        """ Compute difference between two fields based on specified type

        Difference is specified in ``app`` file. It can be a percent difference, a percent change
        or a ratio difference.

        Parameters:
            d1 (ndarray) : A 2D array of an ESM field
            d2 (ndarray) : A 2D array of an ESM field

        Returns:
            Difference of the two fields
        """
        field_diff = None
        if self.config_manager.extra_diff_plot == "percd":  # percent diff
            num = abs(d1 - d2)
            den = (d1 + d2) / 2.0
            field_diff = (num / den) * 100.
        elif self.config_manager.extra_diff_plot == "percc":  # percent change
            field_diff = d1 - d2
            field_diff = field_diff / d2
            field_diff = field_diff * 100
        elif self.config_manager.extra_diff_plot == "ratio":
            field_diff = d1 / d2

        return field_diff

    @staticmethod
    def _interp(y_src, x_src, x_dest, **kwargs):
        """ Wrapper for SciPy's interp1d """
        return interp1d(x_src, y_src, **kwargs)(x_dest)

    def _regrid(self, ref_arr, in_arr, dim1_name, dim2_name, regrid_dims=(0, 0)):
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
            new_arr = xr.apply_ufunc(self._interp, new_arr,
                                     input_core_dims=[[dim2_name]],
                                     output_core_dims=[[dim2_name]],
                                     exclude_dims={dim2_name},
                                     kwargs={'x_src': ref_arr[dim2_name],
                                             'x_dest': in_arr.coords[dim2_name].values,
                                             'fill_value': "extrapolate"},
                                     dask='allowed', vectorize=True)
            new_arr.coords[dim2_name] = in_arr.coords[dim2_name]
        elif regrid_dims[1]:
            new_arr = xr.apply_ufunc(self._interp, new_arr,
                                     input_core_dims=[[dim1_name]],
                                     output_core_dims=[[dim1_name]],
                                     exclude_dims={dim1_name},
                                     kwargs={'x_src': ref_arr[dim1_name],
                                             'x_dest': in_arr.coords[dim1_name].values,
                                             'fill_value': "extrapolate",},
                                     dask='allowed', vectorize=True)
            new_arr.coords[dim1_name] = in_arr.coords[dim1_name]

        return new_arr


class DataProcessor:
    def __init__(self, file_list):
        self.file_list = file_list
        self.datasets = []

    def process_files(self):
        for file_path in self.file_list:
            file_extension = file_path.split('.')[-1]
            data_source = DataSourceFactory.get_data_source(file_extension)
            self.datasets.append(data_source.load_data(file_path))
