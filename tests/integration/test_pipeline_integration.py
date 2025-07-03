import os
import tempfile
import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.data.pipeline.pipeline import DataPipeline
from eviz.lib.data.pipeline.reader import DataReader
from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.lib.data.pipeline.transformer import DataTransformer
from eviz.lib.data.pipeline.integrator import DataIntegrator


class TestPipelineIntegration:
    """Integration tests for the data pipeline components."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock config_manager with meta_coords
        self.config_manager = MagicMock()
        self.config_manager.meta_coords = {
            'xc': {'gridded': ['longitude', 'lon', 'x'], 'test_model': ['longitude', 'lon', 'x']},
            'yc': {'gridded': ['latitude', 'lat', 'y'], 'test_model': ['latitude', 'lat', 'y']},
            'zc': {'gridded': ['lev', 'level', 'z'], 'test_model': ['lev', 'level', 'z']},
            'tc': {'gridded': ['time', 't'], 'test_model': ['time', 't']}
        }
        
        # Create individual components with the config_manager
        self.reader = DataReader(config_manager=self.config_manager)
        self.processor = DataProcessor(config_manager=self.config_manager)
        self.transformer = DataTransformer()
        self.integrator = DataIntegrator()
        
        self.pipeline = DataPipeline(config_manager=self.config_manager)

    
    @pytest.fixture
    def temp_netcdf_file(self):
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
                    },
                    attrs={'units': 'K'}
                ),
                'pressure': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90]),
                        'lon': np.array([0, 90, 180, 270])
                    },
                    attrs={'units': 'hPa'}
                )
            },
            attrs={'global_attr': 'test_value'}
        )
        ds.to_netcdf(path)    
        yield path    
        os.remove(path)
    
    @pytest.fixture
    def multiple_netcdf_files(self):
        """Create multiple temporary NetCDF files for testing."""
        paths = []
        
        # Create first file
        fd1, path1 = tempfile.mkstemp(suffix='.nc')
        os.close(fd1)
        ds1 = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90]),
                        'lon': np.array([0, 90, 180, 270])
                    },
                    attrs={'units': 'K'}
                )
            }
        )
        ds1.to_netcdf(path1)
        paths.append(path1)
        
        # Create second file
        fd2, path2 = tempfile.mkstemp(suffix='.nc')
        os.close(fd2)
        ds2 = xr.Dataset(
            data_vars={
                'humidity': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90]),
                        'lon': np.array([0, 90, 180, 270])
                    },
                    attrs={'units': '%'}
                )
            }
        )
        ds2.to_netcdf(path2)
        paths.append(path2)
        
        yield paths
        
        for path in paths:
            os.remove(path)
    
    def test_pipeline_initialization(self):
        """Test that the pipeline initializes correctly."""
        assert self.pipeline is not None
        assert isinstance(self.pipeline.reader, DataReader)
        assert isinstance(self.pipeline.processor, DataProcessor)
        assert isinstance(self.pipeline.transformer, DataTransformer)
        assert isinstance(self.pipeline.integrator, DataIntegrator)
        assert self.pipeline.data_sources == {}
        assert self.pipeline.dataset is None
    
    @patch('eviz.lib.data.sources.netcdf.NetCDFDataSource')
    def test_integrate_variables(self, mock_netcdf_source, temp_netcdf_file):
        """Test integrating variables."""
        # Create a mock data source
        mock_source = MagicMock()
        mock_source.dataset = xr.open_dataset(temp_netcdf_file)
        mock_source.validate_data.return_value = True
        mock_source.metadata = {}
        
        # Configure the reader to return our mock source
        self.reader.read_file = MagicMock(return_value=mock_source)
        
        # Configure the integrator to perform the variable integration
        def mock_integrate_variables(dataset, variables, operation, output_name):
            if operation == 'add':
                result = dataset.copy()
                result[output_name] = dataset[variables[0]] + dataset[variables[1]]
                result[output_name].attrs['operation'] = operation
                return result
            return dataset
        
        self.integrator.integrate_variables = MagicMock(side_effect=mock_integrate_variables)
        
        # Process the file
        self.pipeline.process_file(temp_netcdf_file, model_name='test_model')
        
        # Set the dataset directly (since we're mocking the integrate_data_sources method)
        self.pipeline.dataset = mock_source.dataset
        
        # Integrate variables
        result = self.pipeline.integrate_variables(
            ['temperature', 'pressure'],
            'add',
            'combined'
        )
        
        assert result is not None
        assert 'combined' in result.data_vars
        assert 'temperature' in result.data_vars
        assert 'pressure' in result.data_vars
        
        # Check that the operation was recorded in the attributes
        assert 'operation' in result['combined'].attrs
        assert result['combined'].attrs['operation'] == 'add'
    
    @patch('eviz.lib.data.sources.netcdf.NetCDFDataSource')
    def test_get_all_data_sources(self, mock_netcdf_source, multiple_netcdf_files):
        """Test getting all data sources."""
        # Create mock data sources for each file
        mock_sources = {}
        for i, path in enumerate(multiple_netcdf_files):
            mock_source = MagicMock()
            mock_source.dataset = xr.open_dataset(path)
            mock_source.validate_data.return_value = True
            mock_source.metadata = {}
            mock_sources[path] = mock_source
        
        # Configure the reader to return the appropriate mock source for each file
        def mock_read_file(file_path, model_name=None):
            return mock_sources[file_path]
        
        self.reader.read_file = MagicMock(side_effect=mock_read_file)
        
        # Process multiple files
        self.pipeline.process_files(multiple_netcdf_files, model_name='test_model')
        
        # Get all data sources
        data_sources = self.pipeline.get_all_data_sources()
        
        assert len(data_sources) == 2
        assert multiple_netcdf_files[0] in data_sources
        assert multiple_netcdf_files[1] in data_sources
    
    def test_error_handling(self):
        """Test error handling for non-existent files."""
        # Configure the reader to raise FileNotFoundError
        self.reader.read_file = MagicMock(side_effect=FileNotFoundError)
        
        # Try processing a non-existent file
        with pytest.raises(FileNotFoundError):
            self.pipeline.process_file('non_existent_file.nc')
