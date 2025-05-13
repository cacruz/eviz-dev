from eviz.lib.data.sources.netcdf4 import NetCDFDataSource
from .sources.hdf5 import HDF5DataSource
from .sources.csv import CSVDataSource
from .sources.grib import GRIBDataSource

class DataSourceFactory:
    @staticmethod
    def get_data_source(file_extension: str):
        if file_extension in ['nc', 'nc4']:
            return NetCDFDataSource()
        elif file_extension in ['h5', 'hdf5']:
            return HDF5DataSource()
        elif file_extension == 'csv':
            return CSVDataSource()
        elif file_extension in ['grib', 'grib2']:
            return GRIBDataSource()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")