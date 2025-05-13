from dataclasses import dataclass
from glob import glob
from typing import Any
import pandas as pd

from eviz.lib.data.reader import DataReader


@dataclass
class CSVDataReader(DataReader):
    """ Class definitions for reading CSV data files.

    """
    file_path: str = None

    def __post_init__(self):
        super().__post_init__()
        self.findex = 0

    def read_data(self, file_path: str) -> Any:
        """ Reads CSV data files and returns its data as Pandas dataframe

        Returns:
            a Pandas dataframe
        """
        self.logger.info(f"Reading CSV data from {file_path}")
        self.file_path = file_path
        files = glob(self.file_path)
        alldata = pd.DataFrame()
        try:
            if "*" in self.file_path:
                for f in files:
                    this_data = pd.read_csv(f)
                    alldata = pd.concat([alldata, this_data], ignore_index=True)
            else:
                this_data = pd.read_csv(self.file_path)
                alldata = pd.concat([alldata, this_data], ignore_index=True)
        except Exception as e:
            self.logger.error(f"An error occurred while reading the data: {str(e)}")
            return None

        self.findex += 1
        self._rename_dims(alldata)
        dataframe = self._process_data(alldata)
        self.datasets.append(dataframe)
        return dataframe

    def _process_data(self, data):
        self.logger.debug(f"Preparing CSV data")
        return data

    def _rename_dims(self, df):
        """ Set Eviz recognized dims """
        xc = self._get_model_dim_name('xc')
        yc = self._get_model_dim_name('yc')
        df.rename(columns={xc: 'lon', yc: 'lat'}, inplace=True)
        return df

    def _get_model_dim_name(self, dim_name):
        return self.meta_coords[dim_name][self.source_name]
