"""
Pytest fixtures for the Eviz tests.
"""
import xarray as xr
import numpy as np
import yaml
import os
import pytest
import tempfile
import requests
from eviz.lib import constants as constants
from eviz.lib.config.config import Config
from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.config.input_config import InputConfig
from eviz.lib.config.output_config import OutputConfig
from eviz.lib.config.system_config import SystemConfig
from eviz.lib.config.history_config import HistoryConfig
from eviz.lib.data.pipeline.pipeline import DataPipeline
from eviz.lib.data.pipeline.reader import DataReader
from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.lib.data.pipeline.transformer import DataTransformer
from eviz.lib.data.pipeline.integrator import DataIntegrator
from eviz.lib.data.sources import DataSource
from tests.fixtures.mock_airmass import create_mock_airmass_dataset


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--run-airmass", 
        action="store_true", 
        default=False, 
        help="run tests that require airmass data"
    )


@pytest.fixture
def mock_species_db():
    """Create a minimal species database for testing."""
    return {
        'O3': {
            'Formula': 'O3',
            'FullName': 'Ozone',
            'MW_kg': 0.048,
            'MW_g': 48,
            'Unitconversion': 1E9
        },
        'NO2': {
            'Formula': 'NO2',
            'FullName': 'Nitrogen dioxide',
            'MW_kg': 0.04601,
            'MW_g': 46.01,
            'Unitconversion': 1
        }
    }


@pytest.fixture
def config_for_units(mock_species_db, tmp_path):
    """
    Create a ConfigManager instance suitable for units tests.
    
    Args:
        mock_species_db: The mock species database fixture
        tmp_path: pytest's temporary directory fixture
    
    Returns:
        ConfigManager: A configured instance with necessary attributes for units tests
    """
    # Create a minimal config structure
    airmass_file_path = str(tmp_path / 'airmass.nc4')
    
    config_data = {
        'for_inputs': {
            'airmass_file_name': airmass_file_path,
            'airmass_field_name': 'AIRMASS'
        },
        'inputs': [
            {
                'name': 'airmass.nc4',
                'location': str(tmp_path),
                'source_name': 'test',
                'description': 'Mock airmass data for testing'
            }
        ]
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    config = Config(source_names=['test'], config_files=[str(config_file)])
    config.species_db = mock_species_db
    
    # Make sure app_data.inputs is a list
    if not hasattr(config.app_data, 'inputs') or not isinstance(config.app_data.inputs, list):
        config.app_data.inputs = config_data['inputs']
    
    input_config = InputConfig(['test'], [str(config_file)])
    output_config = OutputConfig()
    system_config = SystemConfig()
    history_config = HistoryConfig()
    
    config_manager = ConfigManager(
        input_config=input_config,
        output_config=output_config,
        system_config=system_config,
        history_config=history_config,
        config=config
    )
    
    ds = create_mock_airmass_dataset()
    ds.to_netcdf(airmass_file_path)
    
    config_manager.app_data.for_inputs = config_data['for_inputs']
    
    return config_manager


@pytest.fixture
def check_airmass_availability():
    """
    Check if the airmass file is available either locally or via URL.
    
    Returns:
        bool: True if airmass data is available, False otherwise
    """
    # First check local file
    local_path = os.path.join(os.getcwd(), 'airmass.nc4')
    if os.path.exists(local_path):
        return True
        
    # Then check URL
    try:
        response = requests.head(constants.AIRMASS_URL, timeout=5)
        return response.status_code in (200, 302, 307, 443)
    except:
        return False


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
