import xarray as xr
import os
import glob
import numpy as np
import dask
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
import eviz.lib.constants as constants

dask.config.set({"array.slicing.split_large_chunks": False})

logger = logging.getLogger(__name__)


def apply_conversion(config, data2d, name):
    """ Apply a unit conversion based on SPECS file entries

    Parameters:
        config (ConfigManager) configuration object
        data2d (DataArray) with original units
        name (str)

    Returns:
        data2d (DataArray) with target units

    For single-plots, we rely on specs file to determine the units and unit conversion factor
    For comparison plots, we rely on the "target" units specified in the specs file and the unit
    conversion is provided by the Units conversion module.
    """
    # Check if spec_data exists and contains the field name
    if not hasattr(config, 'spec_data') or config.spec_data is None:
        logger.warning(f"No spec_data found in config for {name}")
        return data2d

    if name not in config.spec_data:
        logger.warning(f"Field {name} not found in spec_data")
        return data2d
    
    # A user specifies units AND unitconversion factor:
    if 'units' in config.spec_data[name] and 'unitconversion' in config.spec_data[name]:
        if "AOA" in name.upper():
            data2d = data2d / np.timedelta64(1, 'ns') / 1000000000 / 86400
        else:
            data2d = data2d * float(config.spec_data[name]['unitconversion'])
    # A user specifies units AND no unitconversion factor, in that case we use units module
    elif 'units' in config.spec_data[name] and 'unitconversion' not in config.spec_data[name]:
        # If field name is a chemical species...
        if hasattr(config,
                   'species_db') and config.species_db and name in config.species_db.keys():
            data2d = config.units.convert_chem(data2d, name, config.spec_data[name]['units'])
        else:
            if hasattr(config, 'units') and config.units:
                try:
                    data2d = config.units.convert(data2d, name, config.spec_data[name]['units'])
                except Exception as e:
                    logger.debug(f"Error converting units for {name}: {e}")
                    logger.debug(f"Returning original data for {name} without unit conversion")
            else:
                logger.warning(f"No units module found in config for {name}")
    else:
        # Special hack for AOA metadump specs
        if "AOA" in name.upper():
            data2d = data2d / np.timedelta64(1, 'ns') / 1000000000 / 86400
        msg = f"No units found for {name}. Will use the given 'dataset' units."
        logger.debug(msg)

    return data2d


def apply_mean(config, d, level=None):
    """ Compute various averages over coordinates """
    if level:
        if level == 'all':
            data2d = d.mean(dim=config.get_model_dim_name('zc'))
        else:
            if len(d.dims) == 3:
                data2d = d.mean(dim=config.get_model_dim_name('tc'))
            else:  # 4D array - we need to select a level
                lev_to_plot = int(
                    np.where(d.coords[config.get_model_dim_name('zc')].values == level)[
                        0])
                logger.debug("Level to plot:" + str(lev_to_plot))
                # select level
                data2d = d.isel(lev=lev_to_plot)
                data2d = data2d.mean(dim=config.get_model_dim_name('tc'))
    else:
        if len(d.dims) == 3:
            data2d = d.mean(dim=config.get_model_dim_name('tc'))
        else:
            d = d.mean(dim=config.get_model_dim_name('xc'))
            data2d = d.mean(dim=config.get_model_dim_name('tc'))

    data2d.attrs = d.attrs.copy()  # retain units
    return data2d.squeeze()

def apply_zsum(config, data3d, name):
    """ Sum over vertical levels (column sum)"""
    try:
        is_chem = hasattr(config, 'species_db') and name in config.species_db
        if is_chem:
            data2d_zsum = config.units.convert_chem(data3d, name, config.spec_data[name]['units'])
        else:
            data2d_zsum = data3d.sum(dim='lev')
    except:
        logger.error(f"Could not apply zsum for {name}")
        return None
    data2d_zsum.attrs = data2d_zsum.attrs.copy()
    return data2d_zsum.squeeze()


def grid_cell_areas(lon1d, lat1d, radius=constants.R_EARTH_M):
    """ Calculate grid cell areas given 1D arrays of longitudes and latitudes
    for a planet with the given radius.

    Args:
        lon1d (ndarray): Array of longitude points [degrees] of shape (M,)
        lat1d (ndarray): Array of latitude points [degrees] of shape (M,)
        radius (float, optional): Radius of the planet [metres] (currently assumed spherical)

    Returns:
        Array of grid cell areas [metres**2] of shape (M, N).
    """
    lon_bounds_radian = np.deg2rad(_guess_bounds(lon1d))
    lat_bounds_radian = np.deg2rad(_guess_bounds(lat1d))
    area = _quadrant_area(lat_bounds_radian, lon_bounds_radian, radius)
    return area


def _quadrant_area(radian_lat_bounds, radian_lon_bounds, radius_of_earth):
    """
    Calculate spherical segment areas.

    Taken from SciTools iris library.

    Area weights are calculated for each lat/lon cell as:
        .. math::
            r^2 (lon_1 - lon_0) ( sin(lat_1) - sin(lat_0))

    The resulting array will have a shape of
    *(radian_lat_bounds.shape[0], radian_lon_bounds.shape[0])*
    The calculations are done at 64 bit precision and the returned array
    will be of type numpy.float64.

    Args:
        radian_lat_bounds:  Array of latitude bounds (radians) of shape (M, 2)
        radian_lon_bounds:  Array of longitude bounds (radians) of shape (N, 2)
        radius_of_earth: Radius of the Earth (currently assumed spherical)

    Returns:
        Array of grid cell areas of shape (M, N).
    """
    # ensure pairs of bounds
    if (
            radian_lat_bounds.shape[-1] != 2
            or radian_lon_bounds.shape[-1] != 2
            or radian_lat_bounds.ndim != 2
            or radian_lon_bounds.ndim != 2
    ):
        raise ValueError("Bounds must be [n,2] array")

    # fill in a new array of areas
    radius_sqr = radius_of_earth ** 2
    radian_lat_64 = radian_lat_bounds.astype(np.float64)
    radian_lon_64 = radian_lon_bounds.astype(np.float64)

    ylen = np.sin(radian_lat_64[:, 1]) - np.sin(radian_lat_64[:, 0])
    xlen = radian_lon_64[:, 1] - radian_lon_64[:, 0]
    areas = radius_sqr * np.outer(ylen, xlen)

    # we use abs because backwards bounds (min > max) give negative areas.
    return np.abs(areas)


def _guess_bounds(points, bound_position=0.5):
    """ Guess bounds of grid cells.

    Simplified function from iris.coord.Coord.

    Args:
        points (ndarray): Array of grid points of shape (N,).
        bound_position (float): Bounds offset relative to the grid cell centre.

    Returns:
        Array of shape (N, 2).
    """
    diffs = np.diff(points)
    diffs = np.insert(diffs, 0, diffs[0])
    diffs = np.append(diffs, diffs[-1])

    min_bounds = points - diffs[:-1] * bound_position
    max_bounds = points + diffs[1:] * (1 - bound_position)

    return np.array([min_bounds, max_bounds]).transpose()


def calc_spatial_mean(xr_da, lon_name="longitude", lat_name="latitude",
                      radius=constants.R_EARTH_M):
    """ Calculate spatial mean of xarray.DataArray with grid cell weighting.

    Args:
        xr_da (xarray.DataArray): Data to average
        lon_name (str, optional): Name of x-coordinate
        lat_name (str, optional): Name of y-coordinate
        radius (float):  Radius of the planet (in meters)

    Returns:
        Spatially averaged xarray.DataArray.
    """
    lon = xr_da[lon_name].values
    lat = xr_da[lat_name].values

    area_weights = grid_cell_areas(lon, lat, radius=radius)
    aw_factor = area_weights / area_weights.max()

    return (xr_da * aw_factor).mean(dim=[lon_name, lat_name])


def calc_spatial_integral(xr_da, lon_name="longitude", lat_name="latitude",
                          radius=constants.R_EARTH_M):
    """ Calculate spatial integral of xarray.DataArray with grid cell weighting.

    Args:
        xr_da: xarray.DataArray Data to average
        lon_name: str, optional Name of x-coordinate
        lat_name: str, optional Name of y-coordinate
        radius: float Radius of the planet [metres]

    Returns:
        Spatially averaged xarray.DataArray.
    """
    lon = xr_da[lon_name].values
    lat = xr_da[lat_name].values

    area_weights = grid_cell_areas(lon, lat, radius=radius)

    return (xr_da * area_weights).sum(dim=[lon_name, lat_name])


def get_file_ptr(data_dir, file_pat=None):
    """ Use xarray.open_mfdataset to read multiple files
    """
    if file_pat:
        pattern = os.path.join(data_dir, file_pat)
        print("Opening ", pattern)
        return xr.open_mfdataset(pattern)
    else:
        try:
            return xr.open_mfdataset(data_dir)
        except:
            return None


def read_multiple_netcdf_in_directory(directory_path):
    """
    Reads all NetCDF files in a specified directory using xarray and returns a combined
    xarray Dataset.

    Parameters:
    directory_path (str): The path to the directory containing NetCDF files to read.

    Returns:
    xarray.Dataset: The combined dataset contained in the NetCDF files.
    """
    try:
        file_paths = glob.glob(os.path.join(directory_path, '*.nc'))
        if not file_paths:
            raise FileNotFoundError("No NetCDF files found in the directory.")

        dataset = xr.open_mfdataset(file_paths, combine='by_coords')
        return dataset
    except Exception as e:
        print(f"An error occurred while reading the NetCDF files: {e}")
        return None


def read_multiple_netcdf(file_paths):
    """
    Reads multiple NetCDF files using xarray and returns a combined xarray Dataset.

    Parameters:
    file_paths (list of str): A list of paths to the NetCDF files to read.

    Returns:
    xarray.Dataset: The combined dataset contained in the NetCDF files.
    """
    try:
        dataset = xr.open_mfdataset(file_paths, combine='by_coords')
        return dataset
    except Exception as e:
        print(f"An error occurred while reading the NetCDF files: {e}")
        return None


def read_netcdf(file_path):
    """
    Reads a NetCDF file using xarray and returns an xarray Dataset.

    Parameters:
    file_path (str): The path to the NetCDF file to read.

    Returns:
    xarray.Dataset: The dataset contained in the NetCDF file.
    """
    try:
        dataset = xr.open_dataset(file_path)
        return dataset
    except Exception as e:
        print(f"An error occurred while reading the NetCDF file: {e}")
        return None


def get_dst_attribute(xr_dst, attr_name):
    """ Get an attribute value from a Xarray Dataset or DataArray

      Args:
         xr_dst: Xarray Dataset or DataArray
         attr_name: attribute name
      Returns:
         Attribute value or None if the attribute does not exist.
    """
    try:
        return xr_dst.attrs[attr_name]
    except:
        return None


def compute_means(xr_dst, means):
    """ Computer average over a dataArray (or dataset)

        means can be
            '1D' = daily
            '1M' = monthly
            'QS-JAN' = seasonal (JFM, AMJ, JAS and OND)
            'DS-DEC' = seasonal (DJF, MAM, JJA and SON)
            '1A' = annual

      Returns:
           the time average of a Xarray Dataset or DataArray.

    """
    return xr_dst.resample(time=means).mean()


def compute_mean_over_dim(xr_dst, mean_dim, field_name=None):
    """ Computer average over a dataArray (or dataset) dimension
        mean_dim can be 'Time', 'x', 'y',  or a tuple of dimensions etc.

    Returns:
        the average of a Xarray Dataset or DataArray over a specified dimension
    """
    if field_name is not None:
        return xr_dst[field_name].mean(dim=mean_dim)
    else:
        return xr_dst.mean(dim=mean_dim)


def compute_std_over_dim(xr_dst, std_dim, field_name=None):
    """ Computer standard deviation over a dataArray (or dataset) dimension
        std_dim can be 'Time', 'x', 'y',  or a tuple of dimensions etc.

      Returns:
           the standard deviation of a Xarray Dataset or DataArray over a specified dimension
    """
    if field_name is not None:
        return xr_dst[field_name].std(dim=std_dim)
    else:
        return xr_dst.std(dim=std_dim)


def sum_over_lev(data_array):
    """
    Sums over all lev layers in the given xarray DataArray to get a 2D lat-lon result.

    Parameters:
    data_array (xarray.DataArray): The input data array defined on lon, lat, and lev dimensions.

    Returns:
    xarray.DataArray: The resulting 2D data array after summing over the lev dimension.
    """
    # Sum over the 'lev' dimension
    result_array = data_array.sum(dim='lev')
    return result_array


"""
Internal utilities for managing datetime objects and strings
Adopted from GCpy - with minor modifications

"""


def get_timestamp_string(date_array):
    """
    Convenience function returning the datetime timestamp based on the given input

    Parameters:
        date_array: array
            Array of integers corresponding to [year, month, day, hour, minute, second].
            Any integers not provided will be padded accordingly
    Returns:
        date_str: string
            string in datetime format (eg. 2019-01-01T00:00:00Z)
    """
    # converts single integer to array for cases when only year is given
    date_array = [date_array] if isinstance(date_array, int) else date_array

    # datetime function must receive at least three arguments
    while len(date_array) < 3:
        date_array.append(None)

    # set default values for month and day if not present
    date_array[1] = date_array[1] or 1
    date_array[2] = date_array[2] or 1

    date_str = str(datetime(*date_array)).replace(" ", "T") + "Z"

    return date_str


def add_months(start_date, n_months):
    """

    Parameters:
        start_date: numpy.datetime64
            numpy datetime64 object
        n_months: integer
    Returns:
        new_date: numpy.datetime64
            numpy datetime64 object with exactly n_months added to the date
    """
    new_date = start_date.astype(datetime) + relativedelta(months=n_months)
    return np.datetime64(new_date)


def is_full_year(start_date, end_date):
    """
    Verifies if two dates are a full year starting Jan 1.

    Parameters:
        start_date: numpy.datetime64
            numpy datetime64 object
        end_date: numpy.datetime64
            numpy datetime64 object
    Returns: boolean
    """
    return (
            add_months(start_date, 12) == end_date
            and start_date.astype(datetime).month == 1
            and start_date.astype(datetime).day == 1
    )
