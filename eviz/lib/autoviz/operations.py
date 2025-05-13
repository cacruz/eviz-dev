import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import logging
import xesmf as xe

logger = logging.getLogger(__name__)


def obs_to_grid():
    ds_unstructured = None
    ds_target_grid = None
    regridder = xe.Regridder(ds_unstructured, ds_target_grid, method='bilinear')
    return regridder(ds_unstructured)


def regrid(config, field_names, ax, ax_opts, level, pid):
        """ Wrapper for regrid method

        This function takes in two fields and, if necessary, regrids them to a common resolution.

        Parameters:
            field_names (tuple) : Names of fields
            ax (Axes) : Axes object
            ax_opts (dict) : Axes object
            level (int) : index of vertical level to interpolate (level=0 for 2D fields)
            pid (str) : a plot identifier

        Returns:
            Regridded fields
        """
        # TODO: get rid of ax/ax_opts dependency
        return regrid_check(config, field_names, ax, ax_opts, level, pid)


    # Interpolation (AKA regrid) methods
def regrid_check(config, field_names, ax, ax_opts, level, pid):
    """ Main regrid method """
    name1, name2 = field_names
    dim1 = config.data_source.meta_coords['xc'][config.model_name]
    dim2 = config.data_source.meta_coords['yc'][config.model_name]
    if 'yz' in pid:
        dim1 = config.data_source.meta_coords['yc'][config.model_name]
        dim2 = config.data_source.meta_coords['zc'][config.model_name]

    d1 = config.data_source.get_field(name1, ds_index=0)
    d2 = config.data_source.get_field(name2, ds_index=1)

    if 'yz' in pid:
        d1 = d1.isel(time=ax_opts['time_lev']).mean(dim=_get_model_dim_name('xc'))
        d2 = d2.isel(time=ax_opts['time_lev']).mean(dim=_get_model_dim_name('xc'))
    elif 'xy' in pid:
        if _get_model_dim_name('zc') not in d1.coords:
            d1 = d1.isel(time=ax_opts['time_lev'])
            d2 = d2.isel(time=ax_opts['time_lev'])
        else:
            if len(d1.coords[_get_model_dim_name('zc')]) == 1:
                d1 = d1.squeeze()
                d2 = d2.squeeze()
            else:
                lev_to_plot = np.where(d1.coords[_get_model_dim_name('zc')].values == level)[0]
                d1 = d1.isel(time=ax_opts['time_lev'], lev=lev_to_plot)
                d2 = d2.isel(time=ax_opts['time_lev'], lev=lev_to_plot)

    d1 = _select_yrange(d1, name1)
    d2 = _select_yrange(d2, name2)
    d1 = _apply_conversion(d1, name1)
    d2 = _apply_conversion(d2, name2)

    da1_size = d1.size
    da2_size = d2.size
    if da1_size < da2_size:
        d2 = _regrid(d2, d1, dim1, dim2, regrid_dims=(1, 0))
        d2 = _regrid(d2, d1, dim1, dim2, regrid_dims=(0, 1))
    elif da1_size > da2_size:
        d1 = _regrid(d1, d2, dim1, dim2, regrid_dims=(1, 0))
        d1 = _regrid(d1, d2, dim1, dim2, regrid_dims=(0, 1))
    elif da1_size == da2_size:
        d1 = _regrid(d1, d2, dim1, dim2, regrid_dims=(1, 0))
        d1 = _regrid(d1, d2, dim1, dim2, regrid_dims=(0, 1))
        d2 = _regrid(d2, d1, dim1, dim2, regrid_dims=(1, 0))
        d2 = _regrid(d2, d1, dim1, dim2, regrid_dims=(0, 1))

    if config.extra_diff_plot and ax.get_subplotspec().colspan.start == 1 and ax.get_subplotspec().rowspan.start == 1:
        data_diff = _compute_diff_type(d1, d2).squeeze()
    else:
        data_diff = (d1 - d2).squeeze()
    coords = data_diff.coords
    return data_diff, coords[dim1].values, coords[dim2].values


def _apply_conversion(config, data2d, name):
    """ Apply conversion factor to data

    The conversion factor is specified in the ``specs`` file.

    Parameters:
        data2d (ndarray) : A 2D array of an ESM field
        name (str) : field name
    Returns:
        Pre-processed data array
    """
    if 'unitconversion' in config.spec_data[name]:
        if "AOA" in name.upper():
            data2d = data2d / np.timedelta64(1, 'ns') / 1000000000 / 86400
        else:
            data2d = data2d * float(config.spec_data[name]['unitconversion'])
    return data2d


def _select_yrange(config, data2d, name):
    """ For 3D fields, select vertical level range to use

    Parameters:
        data2d (ndarray) : A 2D array of an ESM field
        name (str) : field name

    Returns:
        sliced data array
    """
    if 'zrange' in config.spec_data[name]['yzplot']:
        lo_z = config.spec_data[name]['yzplot']['zrange'][0]
        hi_z = config.spec_data[name]['yzplot']['zrange'][1]
        if hi_z >= lo_z:
            logger.error(f"Upper level value ({hi_z}) must be less than low level value ({lo_z})")
            return
        lev = _get_model_dim_name('zc')
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


def _compute_diff_type(config, d1, d2):
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
    if config.extra_diff_plot == "percd":  # percent diff
        num = abs(d1 - d2)
        den = (d1 + d2) / 2.0
        field_diff = (num / den) * 100.
    elif config.extra_diff_plot == "percc":  # percent change
        field_diff = d1 - d2
        field_diff = field_diff / d2
        field_diff = field_diff * 100
    elif config.extra_diff_plot == "ratio":
        field_diff = d1 / d2

    return field_diff


@staticmethod
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
                                 kwargs={'x_src': ref_arr[dim2_name], 'x_dest': in_arr.coords[dim2_name].values},
                                 dask='allowed', vectorize=True)
        new_arr.coords[dim2_name] = in_arr.coords[dim2_name]
    elif regrid_dims[1]:
        new_arr = xr.apply_ufunc(_interp, new_arr,
                                 input_core_dims=[[dim1_name]],
                                 output_core_dims=[[dim1_name]],
                                 exclude_dims={dim1_name},
                                 kwargs={'x_src': ref_arr[dim1_name], 'x_dest': in_arr.coords[dim1_name].values},
                                 dask='allowed', vectorize=True)
        new_arr.coords[dim1_name] = in_arr.coords[dim1_name]

    return new_arr


# TODO: Move to utils - also repeated in root.py
def _get_model_dim_name(config, dim_name):
    return config.meta_coords[dim_name][config.model_name]
