from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import xarray as xr

from eviz.lib.data.reader import DataReader


@dataclass
class HDF5DataReader(DataReader):
    """ Class definitions for reading HDF5 files."""
    def __post_init__(self):
        super().__post_init__()
        self.file_path = None
        self.findex = 0

    def get_findex(self, data_to_plot):
        try:
            return data_to_plot[5]
        except IndexError:
            return data_to_plot[3]

    def read_data(self, file_path: str) -> Any:
        """ Given an HDF5 file name, convert the data into an Xarray Dataset.

        Parameters
            file_path : str HDF5 file name containing OMI data

        Returns
        -------
            xr_ds : Xarray Dataset
        """
        self.logger.debug(f"Loading HDF5 data from {file_path}")
        self.file_path = file_path

        try:
            self.findex += 1
            return self.get_fid()
        except Exception as e:
            self.logger.error(f"An error occurred while reading the data: {str(e)}")
            return None

    def process_file(self, fid):
        ds = xr.Dataset()
        data_group = self.get_data_group(fid)
        fid_attrs = self.get_fid_attrs(fid)
        fid_coords = self.get_coords(fid)
        self.logger.info(f'Processing {len(data_group.keys())} HDF5 groups...')

        for var in data_group.keys():
            self.var = var
            ds[var] = self.get_array(data_group, fid_coords)

        ds.attrs = fid_attrs

        fid.close()

        unzipped_data = {}
        unzipped_data['id'] = self.findex
        unzipped_data['ptr'] = ds
        unzipped_data['regrid'] = False
        unzipped_data['vars'] = ds.data_vars
        unzipped_data['attrs'] = ds.attrs
        unzipped_data['dims'] = ds.dims
        unzipped_data['coords'] = ds.coords
        unzipped_data['filename'] = "".join(self.file_path)

        self.datasets.append(unzipped_data)

        if '*empty*' in repr(ds.data_vars):   # If the dataset is empty
            return None
        else:
            return unzipped_data

    def get_fid(self):
        """ Access the file reader object for a given HDF5 file

        Returns:
            an h5py file reader object
        """
        fid = h5py.File(self.file_path, 'r')
        return fid

    def get_data_group(self, fid):
        """
        Finds and returns the contents of the file data field subgroup in dictionary format

        Parameters:
            fid: a file identifier object
        Returns:
             a Python dictionary of dataset name String keys and dataset object values
        """
        parent_contents = dict(fid['HDFEOS']['GRIDS'])  # contents of our parent group
        sub = list(parent_contents.values())[0]  # our sub-parent group object
        sub_contents = dict(sub)  # contents of our sub-parent group
        data_group = list(sub_contents.values())[0]  # our data group object

        return dict(data_group)

    def convert_dict_dtype(self, sample_dict):
        """
        Converts a dictionary of attributes from NumPy data types to general Python data types

        Parameters:
            sample_dict: a Python dictionary of attributes
        Returns:
             a Python dictionary of attributes
        """
        for key, item in sample_dict.items():
            if isinstance(item, np.ndarray):  # Converts np arrays to a list to, if applicable, an int or float
                item = list(item)

                if len(item) == 1:
                    item = item[0]
            elif isinstance(item, np.bytes_):  # Converts np bytes to np string to a Python string
                item = str(item.astype('str'))

                if item[0] == '(' or item[0] == '{':  # Converts to tuple or dict if applicable
                    item = eval(item)
                # **eval() reliability??**

            sample_dict[key] = item  # Updates any changes to the key value

        return sample_dict

    def get_fid_attrs(self, fid):
        """
        Returns the file-level attributes (in Python data types)

        Parameters:
            fid: a file reader object
        Returns:
            a Python dictionary of attributes
        """
        fid_attrs = dict(fid['HDFEOS']['ADDITIONAL']['FILE_ATTRIBUTES'].attrs)
        fid_attrs = self.convert_dict_dtype(fid_attrs)

        fid_attrs.update(self.get_plot_attrs(fid))

        return fid_attrs

    def get_plot_attrs(self, fid):
        """
        Returns the file plotting attributes (in Python data types)

        Parameters:
            fid: a file reader object
        Returns:
            a Python dictionary of attributes
        """
        parent_contents = dict(fid['HDFEOS']['GRIDS'])
        subgroup = list(parent_contents.values())[0]

        plot_attrs = dict(subgroup.attrs)
        plot_attrs = self.convert_dict_dtype(plot_attrs)

        return plot_attrs

    def get_ds_attrs(self, ds):
        """
        Returns the attributes of an HDF5 dataset

        Parameters:
            ds: an HDF5 dataset object
        Returns:
            a Python dictionary of attributes
        """
        ds_attrs = dict(ds.attrs)
        ds_attrs = self.convert_dict_dtype(ds_attrs)

        return ds_attrs

    def get_fill(self, ds_attrs):
        """
        Returns the fill value of a dataset

        Parameters:
            ds_attrs: a Python dictionary of dataset attributes
        Returns:
            an integer, float, or 'None'
        """
        for key, value in ds_attrs.items():
            if key == '_FillValue':
                return value
        return None

    def get_scale(self, ds_attrs):
        """
        Returns the scale factor of a dataset

        Parameters:
            ds_attrs: a Python dictionary of dataset attributes
        Returns:
            an integer, float, or 'None'
        """
        for key, value in ds_attrs.items():
            if key == 'ScaleFactor':
                return value
        return 1

    def get_offset(self, ds_attrs):
        """
        Returns the offset value of a dataset

        Parameters:
            ds_attrs: a Python dictionary of dataset attributes
        Returns:
            an integer, float, or &/or 0
        """
        for key, value in ds_attrs.items():
            if key == 'Offset':
                return value
        return 0

    def restore_data(self, ds):
        """
        Restores the data of a given dataset object

        Parameters:
            ds: an HDF5 dataset object
        Returns:
            a NumPy array
        """
        ds_attrs = self.get_ds_attrs(ds)

        fill = self.get_fill(ds_attrs)
        scale = self.get_scale(ds_attrs)
        offset = self.get_offset(ds_attrs)

        data = ds[()]  # .astype('float')

        data = np.where(data != fill, data, np.nan)
        data *= scale
        data += offset

        data = np.expand_dims(data, axis = 0)

        return data

    def get_time(self, fid):
        """
        Returns the time at which the data was measured/acquired

        Parameters:
            fid: a file reader object
        Returns:
             a list of a DateTime object
        """
        fid_attrs = self.get_fid_attrs(fid)

        year = month = day = None
        for key, value in fid_attrs.items():
            if 'year' in key.lower():
                year = str(value)
            elif 'month' in key.lower():
                month = str(value)
                if len(month) == 1:
                    month = '0'+month
            elif 'day' in key.lower():
                day = str(value)
                if len(day) == 1:
                    day = '0'+day

        if year and month and day:
            time = year + '-' + month + '-' + day
        else:
            return None
        times = [time]
        # times = [datetime.strptime(time, '%Y %b %d')]

        return times

    def get_coords(self, fid):
        """
        Returns the coordinates of a file

        Parameters:
            fid: a file reader object
        Returns:
             a Python dictionary of String keys and NumPy array values
        """
        plot_attrs = self.get_plot_attrs(fid)

        lonW = plot_attrs['GridSpan'][0]
        lonE = plot_attrs['GridSpan'][1]
        latS = plot_attrs['GridSpan'][2]
        latN = plot_attrs['GridSpan'][3]

        lon_size = plot_attrs['NumberOfLongitudesInGrid']
        lat_size = plot_attrs['NumberOfLatitudesInGrid']

        lons = np.linspace(lonW, lonE, lon_size)
        lats = np.linspace(latS, latN, lat_size)
        times = self.get_time(fid)

        return {'times': times, 'lons': lons, 'lats': lats}

    def get_ds_dims(self, ds, coords):
        """
        Returns the dimension names of a dataset

        Parameters:
            ds: a dataset object
            coords: a Python dictionary of file coordinates (NumPy arrays)
        Returns:
            a Python dictionary of dimension name String keys and dimension size integer values
        """
        dims = ds.dims
        ds_dims = {'time': 1}

        for i in range(len(dims)):
            if dims[i].label == '':
                if ds.shape[i] == coords['lons'].size:
                    ds_dims['lon'] = ds.shape[i]
                elif ds.shape[i] == coords['lats'].size:
                    ds_dims['lat'] = ds.shape[i]
            else:
                if 'time' in dims[i].label.lower():
                    ds_dims['time'] = ds.shape[i]
                else:
                    ds_dims[dims[i].label] = ds.shape[i]

        return ds_dims

    def check_coords(self, dims, coords):
        """
        Rearranges order of the coordinates list to match the dimension shapes

        Parameters:
            dims: a Python dictionary of dimension name String keys and dimension size integer values
            coords: a Python dictionary of file coordinates (NumPy arrays)
        Returns:
            a Python dictionary of rearranged file coordinates (NumPy arrays)
        """
        if list(dims.values())[1] != list(coords.values())[1].size:
            temp = coords
            coords = {list(coords.keys())[0]: list(coords.values())[0],
                      list(coords.keys())[2]: list(coords.values())[2],
                      list(coords.keys())[1]: list(coords.values())[1]}
        return coords

    def get_array(self, data_group, fid_coords):
        """
        Returns an XArray DataArray of an HDF5 dataset given the data field subgroup contents and
        file-level coordinates.

        Parameters:
            data_group: a Python dictionary of String keys and HDF5 dataset object values
            fid_coords: a Python dictionary of file coordinates (NumPy arrays)
        Returns:
            an XArray DataArray
        """
        try:
            hdf_ds = data_group[self.var]
        except KeyError:   # If variable doesn't exist
            xr_arr = None
        else:
            data = self.restore_data(hdf_ds)
            ds_attrs = self.get_ds_attrs(hdf_ds)

            ds_dims = self.get_ds_dims(hdf_ds, fid_coords)
            ds_coords = self.check_coords(ds_dims, fid_coords)

            xr_arr = xr.DataArray(data, dims=list(ds_dims.keys()), coords=list(ds_coords.values()))
            xr_arr.attrs = ds_attrs

        return xr_arr

    def _process_data(self, data):
        self.logger.debug(f"Preparing HDF5 data")
        return data
