import os
from dataclasses import dataclass, field
from typing import Any

import xarray as xr

from eviz.lib.data.reader import DataReader
from dask.distributed import Client


@dataclass
class NetCDFDataReader(DataReader):
    datasets: list = field(default_factory=list)
    findex: int = 0

    def __post_init__(self):
        print(30*"!")
        print(30*"!")
        print(30*"!")
        super().__post_init__()

    def read_data(self, file_path: str) -> Any:
        """ Helper function to open and define a dataset

        Parameters:
            fid (int) : file id (starts at 0)
            file_path (str) : name of file associated with fid

        Returns:
            unzipped_data (xarray.Dataset) : dict with xarray dataset information
        """
        self.logger.debug(f"Loading NetCDF data from {file_path} , fid: {self.findex}")
        unzipped_data = {}
        if "*" in file_path:
            # set up a Dask distributed cluster for parallel computation
            c = Client(n_workers=os.cpu_count() - 2, threads_per_worker=1)
            _ = c.dashboard_link
            print("Dask dashboard is available at:", c.dashboard_link)
            self.logger.info(f"Num cores: {os.cpu_count()}")
            try:
                with xr.open_mfdataset(file_path, decode_cf=True) as ds:
                    f = self._rename_dims(ds)
                    unzipped_data['id'] = self.findex
                    unzipped_data['ptr'] = f
                    unzipped_data['regrid'] = False
                    unzipped_data['vars'] = f.data_vars
                    unzipped_data['attrs'] = f.attrs
                    unzipped_data['dims'] = f.dims
                    unzipped_data['coords'] = f.coords
                    unzipped_data['filename'] = "".join(file_path)
                    # TODO: get_season_from_file(file_name)
                    unzipped_data['season'] = None
                    self.findex += 1
            except FileNotFoundError as exc:
                self.logger.error(f"Error opening files: {exc}")
                return None

        else:
            try:
                with xr.open_dataset(file_path, decode_cf=True) as ds:
                    f = self._rename_dims(ds)
                    unzipped_data['id'] = self.findex
                    unzipped_data['ptr'] = f
                    unzipped_data['regrid'] = False
                    unzipped_data['vars'] = f.data_vars
                    unzipped_data['attrs'] = f.attrs
                    unzipped_data['dims'] = f.dims
                    unzipped_data['coords'] = f.coords
                    unzipped_data['filename'] = "".join(file_path)
                    unzipped_data['season'] = None
                    self.findex += 1
            except FileNotFoundError as exc:
                self.logger.error(f"Error opening file: {exc}")
                return None

        processed_data = self._process_data(unzipped_data)
        self.datasets.append(processed_data)
        return processed_data

    def load_data(self, file_path: str) -> xr.Dataset:
        """Load NetCDF data into an Xarray dataset."""
        self.logger.debug(f"Loading NetCDF data from {file_path}")
        if "*" in file_path:
            dataset = xr.open_mfdataset(file_path, decode_cf=True)
        else:
            dataset = xr.open_dataset(file_path, decode_cf=True)

        self._process_data(dataset)
        return dataset

    def _process_data(self, data):
        self.logger.debug("Processing NetCDF data")
        return data

    def get_findex(self, data_to_plot):
        try:
            return data_to_plot[5]
        except IndexError:
            return data_to_plot[3]

    def _get_model_dim_name2(self, dim_name):
        return self.meta_coords[dim_name][self.source_name]

    def _rename_dims(self, ds):
        """ Set Eviz recognized dims """
        # Avoid renaming for wrf and lis
        if self.source_name in ['wrf', 'lis']:
            return ds

        # Get model dimension names
        xc = self._get_model_dim_name('xc', ds.dims)
        yc = self._get_model_dim_name('yc', ds.dims)
        zc = self._get_model_dim_name('zc', ds.dims)
        tc = self._get_model_dim_name('tc', ds.dims)

        # Rename only if valid dimension names are found
        rename_dict = {}
        if xc:
            rename_dict[xc] = 'lon'
        if yc:
            rename_dict[yc] = 'lat'
        if zc:
            rename_dict[zc] = 'lev'
        if tc:
            rename_dict[tc] = 'time'

        return ds.rename(rename_dict) if rename_dict else ds

    def _get_model_dim_name(self, dim_name, dims):
        """Retrieve the correct dimension name from meta_coords based on source_name."""
        if dim_name not in self.meta_coords:
            return None  # Ensure dim_name exists in meta_coords

        dim_data = self.meta_coords[dim_name]

        if self.source_name not in dim_data:
            return None  # Source not found in meta_coords[dim_name]

        source_data = dim_data[self.source_name]

        # Handle cases where the dimension data is a list or a string
        valid_values = []
        if isinstance(source_data, dict):
            valid_values = source_data.get('dim', '')  # Use 'dim' key if nested
        elif isinstance(source_data, list):
            valid_values = source_data
        else:
            valid_values = source_data.split(',')

        # Return first matching dimension found
        for dim in dims:
            if dim in valid_values:
                return dim

        return None  # No match found
