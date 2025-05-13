import numpy as np
import eviz.lib.const as constants
import dask
dask.config.set({"array.slicing.split_large_chunks": False})


def grid_cell_areas(lon1d, lat1d, radius=constants.R_EARTH_m):
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


def calc_spatial_mean(xr_da, lon_name="longitude", lat_name="latitude", radius=constants.R_EARTH_m):
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


def calc_spatial_integral(xr_da, lon_name="longitude", lat_name="latitude", radius=constants.R_EARTH_m):
    """ Calculate spatial integral of xarray.DataArray with grid cell weighting.

    Args:
        xr_da: xarray.DataArray Data to average
        lon_name: str, optional Name of x-coordinate
        lat_name: str, optional Name of y-coordinate
        radius: float Radius of the planet [metres], currently assumed spherical (not important anyway)

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


import xarray as xr
import os
import glob


def read_multiple_netcdf_in_directory(directory_path):
    """
    Reads all NetCDF files in a specified directory using xarray and returns a combined xarray Dataset.

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
