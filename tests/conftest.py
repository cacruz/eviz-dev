"""
Pytest fixtures for the Eviz tests.
"""

import os
import pytest
import numpy as np
import xarray as xr
import tempfile
from unittest.mock import MagicMock

from eviz.lib.data.pipeline.pipeline import DataPipeline
from eviz.lib.data.pipeline.reader import DataReader
from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.lib.data.pipeline.transformer import DataTransformer
from eviz.lib.data.pipeline.integrator import DataIntegrator
from eviz.lib.data.sources import DataSource


class MockDataSource(DataSource):
    """Mock data source for testing."""
    
    def load_data(self, file_path):
        """Load data from the specified file path."""
        self.dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90]),
                        'lon': np.array([0, 90, 180, 270])
                    }
                ),
                'pressure': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90]),
                        'lon': np.array([0, 90, 180, 270])
                    }
                )
            }
        )
        return self.dataset


@pytest.fixture
def mock_data_source():
    """Create a mock data source."""
    return MockDataSource('test_model')


@pytest.fixture
def test_dataset():
    """Create a test dataset."""
    return xr.Dataset(
        data_vars={
            'temperature': xr.DataArray(
                data=np.random.rand(2, 3, 4),
                dims=['time', 'lat', 'lon'],
                coords={
                    'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                    'lat': np.array([0, 45, 90]),
                    'lon': np.array([0, 90, 180, 270])
                }
            ),
            'pressure': xr.DataArray(
                data=np.random.rand(2, 3, 4),
                dims=['time', 'lat', 'lon'],
                coords={
                    'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                    'lat': np.array([0, 45, 90]),
                    'lon': np.array([0, 90, 180, 270])
                }
            )
        }
    )


@pytest.fixture
def pipeline():
    """Create a DataPipeline instance."""
    return DataPipeline()


@pytest.fixture
def reader():
    """Create a DataReader instance."""
    return DataReader()


@pytest.fixture
def processor():
    """Create a DataProcessor instance."""
    return DataProcessor()


@pytest.fixture
def transformer():
    """Create a DataTransformer instance."""
    return DataTransformer()


@pytest.fixture
def integrator():
    """Create a DataIntegrator instance."""
    return DataIntegrator()


@pytest.fixture
def temp_netcdf_file():
    """Create a temporary NetCDF file for testing."""
    fd, path = tempfile.mkstemp(suffix='.nc')
    os.close(fd)    
    ds = xr.Dataset(
        data_vars={
            'temperature': xr.DataArray(
                data=np.random.rand(2, 3, 4),
                dims=['time', 'lat', 'lon'],
                coords={
                    'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                    'lat': np.array([0, 45, 90]),
                    'lon': np.array([0, 90, 180, 270])
                }
            ),
            'pressure': xr.DataArray(
                data=np.random.rand(2, 3, 4),
                dims=['time', 'lat', 'lon'],
                coords={
                    'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                    'lat': np.array([0, 45, 90]),
                    'lon': np.array([0, 90, 180, 270])
                }
            )
        }
    )
    ds.to_netcdf(path)    
    yield path    
    os.remove(path)
