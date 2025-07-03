import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch
from eviz.lib.data.sources.zarr import ZARRDataSource


class TestZarrDataSource:
    """Test cases for the ZarrDataSource class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.data_source = ZARRDataSource('test_model')
    
    def test_init(self):
        """Test initialization of ZarrDataSource."""
        assert self.data_source.model_name == 'test_model'
        assert self.data_source.dataset is None
        assert self.data_source.metadata == {}
    
    @patch('xarray.open_dataset')
    def test_load_data(self, mock_open_dataset):
        """Test loading data from a Zarr store."""
        mock_dataset = xr.Dataset(
            data_vars={
                'soil_moisture': xr.DataArray(
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
        mock_open_dataset.return_value = mock_dataset

        result = self.data_source.load_data('test_store.zarr')
        assert result is not None
        assert isinstance(result, xr.Dataset)
        assert 'soil_moisture' in result.data_vars

        # Verify the correct engine was used
        mock_open_dataset.assert_called_once_with('test_store.zarr', engine='zarr')

    @patch('xarray.open_dataset')
    def test_load_data_with_error(self, mock_open_dataset):
        """Test loading data with an error."""
        mock_open_dataset.side_effect = Exception("Test error")

        with pytest.raises(Exception):
            self.data_source.load_data('test_store.zarr')

        mock_open_dataset.assert_called_once_with('test_store.zarr', engine='zarr')
    
    @patch('xarray.open_dataset')
    def test_load_multiple_zarr_stores(self, mock_open_dataset):
        """Test loading data from multiple Zarr stores."""
        mock_dataset1 = xr.Dataset(
            data_vars={
                'soil_moisture': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon']
                )
            }
        )
        mock_dataset2 = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon']
                )
            }
        )
        
        # Set up the mock to return different datasets for different calls
        mock_open_dataset.side_effect = [mock_dataset1, mock_dataset2]
        
        # Mock the merge function
        with patch('xarray.merge') as mock_merge:
            merged_dataset = xr.Dataset(
                data_vars={
                    'soil_moisture': mock_dataset1.soil_moisture,
                    'temperature': mock_dataset2.temperature
                }
            )
            mock_merge.return_value = merged_dataset
            
            result = self.data_source.load_data(['store1.zarr', 'store2.zarr'])
            
            assert result is not None
            assert isinstance(result, xr.Dataset)
            assert mock_open_dataset.call_count == 2
            mock_merge.assert_called_once()
    
    def test_validate_data(self):
        """Test validating data."""
        self.data_source.dataset = xr.Dataset(
            data_vars={
                'soil_moisture': xr.DataArray(
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
        
        result = self.data_source.validate_data()
        assert result is True
    
    def test_validate_data_no_dataset(self):
        """Test validating data with no dataset."""
        result = self.data_source.validate_data()
        assert result is False
    
    def test_get_field(self):
        """Test getting a field."""
        self.data_source.dataset = xr.Dataset(
            data_vars={
                'soil_moisture': xr.DataArray(
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
        
        result = self.data_source.get_field('soil_moisture')
        assert result is not None
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('time', 'lat', 'lon')
    
    def test_get_field_not_found(self):
        """Test getting a field that doesn't exist."""
        self.data_source.dataset = xr.Dataset(
            data_vars={
                'soil_moisture': xr.DataArray(
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
        
        result = self.data_source.get_field('non_existent')
        assert result is None
    
    def test_get_metadata(self):
        """Test getting metadata."""
        self.data_source.metadata = {'key': 'value'}
        result = self.data_source.get_metadata()
        assert result == {'key': 'value'}
    
    def test_close(self):
        """Test closing the data source."""
        self.data_source.dataset = MagicMock()
        self.data_source.dataset.close = MagicMock()
        self.data_source.close()
        self.data_source.dataset.close.assert_called_once()
    
    @patch('xarray.open_dataset')
    def test_extract_metadata(self, mock_open_dataset):
        """Test extracting metadata from a Zarr store."""
        mock_dataset = xr.Dataset(
            data_vars={
                'soil_moisture': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90]),
                        'lon': np.array([0, 90, 180, 270])
                    },
                    attrs={
                        'long_name': 'Soil Moisture',
                        'units': 'm^3/m^3'
                    }
                )
            },
            attrs={
                'Conventions': 'CF-1.6',
                'history': 'Created by CREST model'
            }
        )
        mock_open_dataset.return_value = mock_dataset
        
        self.data_source.load_data('test_store.zarr')
        assert self.data_source.metadata is not None
        assert 'global_attrs' in self.data_source.metadata
        assert 'Conventions' in self.data_source.metadata['global_attrs']
        assert self.data_source.metadata['global_attrs']['Conventions'] == 'CF-1.6'
        
        # Check variable metadata
        assert 'variables' in self.data_source.metadata
        assert 'soil_moisture' in self.data_source.metadata['variables']
        assert 'attrs' in self.data_source.metadata['variables']['soil_moisture']
        assert self.data_source.metadata['variables']['soil_moisture']['attrs']['units'] == 'm^3/m^3'
        
        # Check dimensions
        assert 'dimensions' in self.data_source.metadata
        assert 'time' in self.data_source.metadata['dimensions']
        assert self.data_source.metadata['dimensions']['time'] == 2
    
    @patch('xarray.open_dataset')
    def test_process_data(self, mock_open_dataset):
        """Test processing data."""
        mock_dataset = xr.Dataset(
            data_vars={
                'soil_moisture': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon']
                )
            }
        )
        mock_open_dataset.return_value = mock_dataset
        
        # Create a spy on _process_data
        with patch.object(ZARRDataSource, '_process_data', wraps=self.data_source._process_data) as mock_process:
            self.data_source.load_data('test_store.zarr')
            mock_process.assert_called_once()
            # Verify the dataset was passed to _process_data
            assert mock_process.call_args[0][0] is mock_dataset
