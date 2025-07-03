import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch
from eviz.lib.data.pipeline.transformer import DataTransformer
from eviz.lib.data.sources import DataSource


class TestDataTransformer:
    """Test cases for the DataTransformer class."""
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.transformer = DataTransformer()        
        self.mock_data_source = MagicMock(spec=DataSource)
        
        self.test_dataset = xr.Dataset(
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
        
        self.mock_data_source.dataset = self.test_dataset
        self.mock_data_source.validate_data.return_value = True
    
    def test_init(self):
        """Test initialization of DataTransformer."""
        assert self.transformer is not None
    
    def test_transform_data_source(self):
        """Test transforming a data source."""
        result = self.transformer.transform_data_source(self.mock_data_source)
        assert result == self.mock_data_source
        
    
