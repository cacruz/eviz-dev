import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock
from eviz.lib.data.sources.base import DataSource


class ConcreteDataSource(DataSource):
    """Concrete implementation of DataSource for testing."""
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
                )
            }
        )
        return self.dataset


class TestDataSource:
    """Test cases for the DataSource class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.data_source = ConcreteDataSource('test_model')
        
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
        
        self.data_source.dataset = self.test_dataset
    
    def test_init(self):
        """Test initialization of DataSource."""
        assert self.data_source.model_name == 'test_model'
        assert self.data_source.dataset is not None
        assert self.data_source.metadata == {}
    
    def test_load_data(self):
        """Test loading data."""
        data_source = ConcreteDataSource('test_model')
        
        result = data_source.load_data('test_file.nc')
        assert result is not None
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
    
    def test_validate_data(self):
        """Test validating data."""
        result = self.data_source.validate_data()
        assert result is True
    
    def test_validate_data_no_dataset(self):
        """Test validating data with no dataset."""
        data_source = ConcreteDataSource('test_model')
        
        result = data_source.validate_data()
        assert result is False
    
    def test_get_field(self):
        """Test getting a field."""
        result = self.data_source.get_field('temperature')
        assert result is not None
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('time', 'lat', 'lon')
    
    def test_get_field_not_found(self):
        """Test getting a field that doesn't exist."""
        result = self.data_source.get_field('non_existent')
        assert result is None
    
    def test_get_field_no_dataset(self):
        """Test getting a field with no dataset."""
        data_source = ConcreteDataSource('test_model')
        result = data_source.get_field('temperature')
        assert result is None
    
    def test_get_metadata(self):
        """Test getting metadata."""
        self.data_source.metadata = {'key': 'value'}
        result = self.data_source.get_metadata()
        assert result == {'key': 'value'}
    
    def test_get_dimensions(self):
        """Test getting dimensions."""
        result = self.data_source.get_dimensions()
        assert result == ['time', 'lat', 'lon']
    
    def test_get_dimensions_no_dataset(self):
        """Test getting dimensions with no dataset."""
        data_source = ConcreteDataSource('test_model')
        result = data_source.get_dimensions()
        assert result == []
    
    def test_get_variables(self):
        """Test getting variables."""
        result = self.data_source.get_variables()
        assert result == ['temperature', 'pressure']
    
    def test_get_variables_no_dataset(self):
        """Test getting variables with no dataset."""
        data_source = ConcreteDataSource('test_model')
        result = data_source.get_variables()
        assert result == []
    
    def test_close(self):
        """Test closing the data source."""
        self.data_source.dataset = MagicMock()
        self.data_source.dataset.close = MagicMock()        
        self.data_source.close()
        self.data_source.dataset.close.assert_called_once()
    
    def test_getattr(self):
        """Test getting an attribute from the dataset."""
        self.data_source.dataset = MagicMock()
        self.data_source.dataset.dims = ('time', 'lat', 'lon')
        
        result = self.data_source.dims
        assert result == ('time', 'lat', 'lon')
    
    def test_getattr_no_dataset(self):
        """Test getting an attribute with no dataset."""
        data_source = ConcreteDataSource('test_model')        
        with pytest.raises(AttributeError):
            data_source.dims
    
    def test_getattr_not_found(self):
        """Test getting an attribute that doesn't exist."""
        # Call the method and verify it raises an exception
        with pytest.raises(AttributeError):
            self.data_source.non_existent
    
    def test_getitem(self):
        """Test getting an item from the dataset."""
        result = self.data_source['temperature']
        
        assert result is not None
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('time', 'lat', 'lon')
    
    def test_getitem_no_dataset(self):
        """Test getting an item with no dataset."""
        data_source = ConcreteDataSource('test_model')
        with pytest.raises(TypeError):
            data_source['temperature']
