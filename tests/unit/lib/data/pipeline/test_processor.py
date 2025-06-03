import numpy as np
import xarray as xr
from unittest.mock import MagicMock
from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.lib.data.sources import DataSource


class TestDataProcessor:
    """Test cases for the DataProcessor class."""
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.processor = DataProcessor()
        
        # Create a mock data source with metadata attribute initialized
        self.mock_data_source = MagicMock(spec=DataSource)
        self.mock_data_source.metadata = {}  # Initialize empty metadata dict
        
        self.test_dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'latitude', 'longitude'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'latitude': np.array([0, 45, 90]),
                        'longitude': np.array([0, 90, 180, 270])
                    },
                    attrs={'units': 'K'}
                ),
                'pressure': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'latitude', 'longitude'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'latitude': np.array([0, 45, 90]),
                        'longitude': np.array([0, 90, 180, 270])
                    },
                    attrs={'units': 'hPa'}
                )
            },
            attrs={'global_attr': 'test_value'}  # Add global attributes for testing
        )
        
        self.mock_data_source.dataset = self.test_dataset
        self.mock_data_source.validate_data.return_value = True
    
    def test_init(self):
        """Test initialization of DataProcessor."""
        assert self.processor is not None
    
    def test_process_data_source(self):
        """Test processing a data source."""
        # Process the data source
        result = self.processor.process_data_source(self.mock_data_source)
        
        # Verify the result
        assert result is not None
        assert self.mock_data_source.validate_data.called
        
       
    def test_process_data_source_validation_failed(self):
        """Test processing a data source with validation failure."""
        self.mock_data_source.validate_data.return_value = False
        
        result = self.processor.process_data_source(self.mock_data_source)
        assert result == self.mock_data_source

        self.mock_data_source.validate_data.assert_called_once()

    def test_process_data_source_invalid_data(self):
        """Test processing a data source with invalid data."""
        self.mock_data_source.validate_data.return_value = False
        result = self.processor.process_data_source(self.mock_data_source)
        assert result == self.mock_data_source

    def test_process_dataset(self):
        """Test processing a dataset."""
        processed_dataset = self.processor._process_dataset(self.test_dataset)
        assert processed_dataset is not None
        assert isinstance(processed_dataset, xr.Dataset)
   
    def test_standardize_coordinates_with_different_models(self):
        """Test coordinate standardization with different model names."""
        # Create a dataset with non-standard coordinate names
        test_dataset = xr.Dataset(
            coords={
                'lat': ('lat', np.array([-90, 0, 90])),
                'lon': ('lon', np.array([-180, 0, 180])),
                'lev': ('lev', np.array([1000, 500, 100])),
            }
        )
        
        # Create a mock config_manager with meta_coords for different models
        mock_config_manager = MagicMock()
        mock_config_manager.meta_coords = {
            'xc': {
                'gridded': ['longitude', 'lon', 'x'],
                'wrf': ['west_east', 'x'],
                'lis': ['east_west', 'x']
            },
            'yc': {
                'gridded': ['latitude', 'lat', 'y'],
                'wrf': ['south_north', 'y'],
                'lis': ['north_south', 'y']
            },
            'zc': {
                'gridded': ['lev', 'level', 'z'],
                'wrf': ['bottom_top', 'z'],
                'lis': ['vertical', 'z']
            },
            'tc': {
                'gridded': ['time', 't'],
                'wrf': ['Time', 't'],
                'lis': ['time', 't']
            }
        }
        
        # Set the mock config_manager on the processor
        self.processor.config_manager = mock_config_manager
        
        # Test with 'wrf' model - should skip renaming
        processed_wrf = self.processor._standardize_coordinates(test_dataset, model_name='wrf')
        assert 'lat' in processed_wrf.coords  # Should remain unchanged
        assert 'lon' in processed_wrf.coords  # Should remain unchanged
        
        # Test with 'lis' model - should skip renaming
        processed_lis = self.processor._standardize_coordinates(test_dataset, model_name='lis')
        assert 'lat' in processed_lis.coords  # Should remain unchanged
        assert 'lon' in processed_lis.coords  # Should remain unchanged
        
        # Test with a model that doesn't exist in meta_coords - should use 'gridded'
        processed_unknown = self.processor._standardize_coordinates(test_dataset, model_name='unknown')
        assert 'lat' in processed_unknown.coords  # Should remain unchanged
        assert 'lon' in processed_unknown.coords  # Should remain unchanged

    def test_standardize_coordinates_with_dimensions_to_rename(self):
        """Test coordinate standardization with dimensions that need renaming."""
        # Create a dataset with non-standard coordinate names
        test_dataset = xr.Dataset(
            coords={
                'latitude': ('latitude', np.array([-90, 0, 90])),
                'longitude': ('longitude', np.array([-180, 0, 180])),
                'level': ('level', np.array([1000, 500, 100])),
                'time': ('time', np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]')),
            },
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 3, 3),
                    dims=['time', 'level', 'latitude', 'longitude']
                )
            }
        )
        
        # Create a mock config_manager with meta_coords
        mock_config_manager = MagicMock()
        mock_config_manager.meta_coords = {
            'xc': {
                'gridded': ['longitude', 'lon', 'x']
            },
            'yc': {
                'gridded': ['latitude', 'lat', 'y']
            },
            'zc': {
                'gridded': ['level', 'lev', 'z']
            },
            'tc': {
                'gridded': ['time', 't']
            }
        }
        
        # Set the mock config_manager on the processor
        self.processor.config_manager = mock_config_manager
        
        # Now call _standardize_coordinates
        processed = self.processor._standardize_coordinates(test_dataset, model_name='gridded')
        
        # Check that the coordinates were renamed
        assert 'lat' in processed.coords
        assert 'lon' in processed.coords
        assert 'lev' in processed.coords
        assert 'time' in processed.coords  # Should remain unchanged
        
        # Check that the data variable dimensions were updated
        assert processed['temperature'].dims == ('time', 'lev', 'lat', 'lon')

    
    def test_handle_missing_values(self):
        """Test handling missing values."""
        dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]]),
                    dims=['x', 'y'],
                    attrs={'_FillValue': -999.9}
                )
            }
        )
        
        result = self.processor._handle_missing_values(dataset)
        assert np.isclose(result['temperature'].values[0, 2], -999.9)
        assert np.isclose(result['temperature'].values[1, 1], -999.9)
    
    def test_apply_unit_conversions(self):
        """Test applying unit conversions."""
        result = self.processor._apply_unit_conversions(self.test_dataset)
        
        assert result['temperature'].attrs['units'] == 'C'
        assert result['pressure'].attrs['units'] == 'Pa'
        
        # Check that the data was converted correctly
        # Temperature: K to C (subtract 273.15)
        # Pressure: hPa to Pa (multiply by 100)
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    # Not implemented yet
                    pass
