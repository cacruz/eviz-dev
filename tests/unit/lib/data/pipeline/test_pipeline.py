"""
Unit tests for the DataPipeline class.
"""

import os
import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.data.pipeline.pipeline import DataPipeline
from eviz.lib.data.sources import DataSource


class TestDataPipeline:
    """Test cases for the DataPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.pipeline = DataPipeline()
        
        # Create a mock data source
        self.mock_data_source = MagicMock(spec=DataSource)
        self.mock_data_source.dataset = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 1, 2]),
                        'lon': np.array([0, 1, 2, 3])
                    }
                ),
                'pressure': xr.DataArray(
                    data=np.random.rand(2, 3, 4),
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 1, 2]),
                        'lon': np.array([0, 1, 2, 3])
                    }
                )
            }
        )
        
    def test_init(self):
        """Test initialization of DataPipeline."""
        assert self.pipeline.reader is not None
        assert self.pipeline.processor is not None
        assert self.pipeline.transformer is not None
        assert self.pipeline.integrator is not None
        assert self.pipeline.data_sources == {}
        assert self.pipeline.dataset is None
    
    @patch('eviz.lib.data.pipeline.reader.DataReader.read_file')
    @patch('eviz.lib.data.pipeline.processor.DataProcessor.process_data_source')
    def test_process_file(self, mock_process, mock_read):
        """Test processing a single file."""
        # Setup mocks
        mock_read.return_value = self.mock_data_source
        mock_process.return_value = self.mock_data_source
        
        # Call the method
        result = self.pipeline.process_file('test_file.nc', 'test_model')
        
        # Verify the result
        assert result == self.mock_data_source
        assert self.pipeline.data_sources['test_file.nc'] == self.mock_data_source
        
        # Verify the mocks were called correctly
        mock_read.assert_called_once_with('test_file.nc', 'test_model')
        mock_process.assert_called_once_with(self.mock_data_source)
    
    @patch('eviz.lib.data.pipeline.reader.DataReader.read_file')
    @patch('eviz.lib.data.pipeline.processor.DataProcessor.process_data_source')
    @patch('eviz.lib.data.pipeline.transformer.DataTransformer.transform_data_source')
    def test_process_file_with_transform(self, mock_transform, mock_process, mock_read):
        """Test processing a single file with transformation."""
        # Setup mocks
        mock_read.return_value = self.mock_data_source
        mock_process.return_value = self.mock_data_source
        mock_transform.return_value = self.mock_data_source
        
        # Call the method
        transform_params = {'regrid': True}
        result = self.pipeline.process_file('test_file.nc', 'test_model', 
                                          transform=True, transform_params=transform_params)
        
        # Verify the result
        assert result == self.mock_data_source
        assert self.pipeline.data_sources['test_file.nc'] == self.mock_data_source
        
        # Verify the mocks were called correctly
        mock_read.assert_called_once_with('test_file.nc', 'test_model')
        mock_process.assert_called_once_with(self.mock_data_source)
        mock_transform.assert_called_once_with(self.mock_data_source, **transform_params)
    
    @patch('eviz.lib.data.pipeline.pipeline.DataPipeline.process_file')
    def test_process_files(self, mock_process_file):
        """Test processing multiple files."""
        # Setup mock
        mock_process_file.side_effect = [self.mock_data_source, self.mock_data_source]
        
        # Call the method
        file_paths = ['file1.nc', 'file2.nc']
        result = self.pipeline.process_files(file_paths, 'test_model')
        
        # Verify the result
        assert len(result) == 2
        assert result['file1.nc'] == self.mock_data_source
        assert result['file2.nc'] == self.mock_data_source
        
        # Verify the mock was called correctly
        assert mock_process_file.call_count == 2
        mock_process_file.assert_any_call('file1.nc', 'test_model', True, False, None)
        mock_process_file.assert_any_call('file2.nc', 'test_model', True, False, None)
    
    @patch('eviz.lib.data.pipeline.pipeline.DataPipeline.process_file')
    def test_process_files_with_error(self, mock_process_file):
        """Test processing multiple files with an error."""
        # Setup mock
        mock_process_file.side_effect = [self.mock_data_source, Exception("Test error")]
        
        # Call the method
        file_paths = ['file1.nc', 'file2.nc']
        result = self.pipeline.process_files(file_paths, 'test_model')
        
        # Verify the result
        assert len(result) == 1
        assert result['file1.nc'] == self.mock_data_source
        assert 'file2.nc' not in result
        
        # Verify the mock was called correctly
        assert mock_process_file.call_count == 2
    
    @patch('eviz.lib.data.pipeline.integrator.DataIntegrator.integrate_data_sources')
    def test_integrate_data_sources(self, mock_integrate):
        """Test integrating data sources."""
        # Setup mock
        expected_dataset = xr.Dataset()
        mock_integrate.return_value = expected_dataset
        
        # Setup data sources
        self.pipeline.data_sources = {
            'file1.nc': self.mock_data_source,
            'file2.nc': self.mock_data_source
        }
        
        # Call the method
        result = self.pipeline.integrate_data_sources()
        
        # Verify the result
        assert isinstance(result, xr.Dataset)
        assert self.pipeline.dataset is expected_dataset
        
        # Verify the mock was called correctly
        mock_integrate.assert_called_once()
        args, kwargs = mock_integrate.call_args
        assert len(args[0]) == 2  # Two data sources
        assert args[0][0] == self.mock_data_source
        assert args[0][1] == self.mock_data_source
    
    @patch('eviz.lib.data.pipeline.integrator.DataIntegrator.integrate_data_sources')
    def test_integrate_data_sources_with_file_paths(self, mock_integrate):
        """Test integrating specific data sources."""
        # Setup mock
        expected_dataset = xr.Dataset()
        mock_integrate.return_value = expected_dataset
        
        # Setup data sources
        self.pipeline.data_sources = {
            'file1.nc': self.mock_data_source,
            'file2.nc': self.mock_data_source,
            'file3.nc': self.mock_data_source
        }
        
        # Call the method
        result = self.pipeline.integrate_data_sources(['file1.nc', 'file3.nc'])
        
        # Verify the result
        assert isinstance(result, xr.Dataset)
        assert self.pipeline.dataset is expected_dataset
        
        # Verify the mock was called correctly
        mock_integrate.assert_called_once()
        args, kwargs = mock_integrate.call_args
        assert len(args[0]) == 2  # Two data sources
        assert args[0][0] == self.mock_data_source
        assert args[0][1] == self.mock_data_source
    
    @patch('eviz.lib.data.pipeline.integrator.DataIntegrator.integrate_variables')
    def test_integrate_variables(self, mock_integrate):
        """Test integrating variables."""
        # Setup mock
        expected_dataset = xr.Dataset()
        mock_integrate.return_value = expected_dataset
        
        # Setup dataset
        self.pipeline.dataset = xr.Dataset()
        
        # Call the method
        variables = ['temperature', 'pressure']
        operation = 'add'
        output_name = 'total'
        result = self.pipeline.integrate_variables(variables, operation, output_name)
        
        # Verify the result
        assert isinstance(result, xr.Dataset)
        assert self.pipeline.dataset is expected_dataset
        
        # Verify the mock was called correctly
        assert mock_integrate.call_count == 1
        args, kwargs = mock_integrate.call_args
        assert args[1] == variables
        assert args[2] == operation
        assert args[3] == output_name
    
    def test_get_data_source(self):
        """Test getting a data source."""
        # Setup data sources
        self.pipeline.data_sources = {
            'file1.nc': self.mock_data_source
        }
        
        # Call the method
        result = self.pipeline.get_data_source('file1.nc')
        
        # Verify the result
        assert result == self.mock_data_source
        
        # Test getting a non-existent data source
        result = self.pipeline.get_data_source('non_existent.nc')
        assert result is None
    
    def test_get_all_data_sources(self):
        """Test getting all data sources."""
        # Setup data sources
        self.pipeline.data_sources = {
            'file1.nc': self.mock_data_source,
            'file2.nc': self.mock_data_source
        }
        
        # Call the method
        result = self.pipeline.get_all_data_sources()
        
        # Verify the result
        assert len(result) == 2
        assert result['file1.nc'] == self.mock_data_source
        assert result['file2.nc'] == self.mock_data_source
        
        # Verify it's a copy
        result['file3.nc'] = self.mock_data_source
        assert 'file3.nc' not in self.pipeline.data_sources
    
    def test_get_dataset(self):
        """Test getting the dataset."""
        # Setup dataset
        expected_dataset = xr.Dataset()
        self.pipeline.dataset = expected_dataset
        
        # Call the method
        result = self.pipeline.get_dataset()
        
        # Verify the result
        assert isinstance(result, xr.Dataset)
    
    @patch('eviz.lib.data.pipeline.reader.DataReader.close')
    def test_close(self, mock_close):
        """Test closing the pipeline."""
        # Setup data sources and dataset
        self.pipeline.data_sources = {
            'file1.nc': self.mock_data_source,
            'file2.nc': self.mock_data_source
        }
        self.pipeline.dataset = xr.Dataset()
        
        # Call the method
        self.pipeline.close()
        
        # Verify the result
        assert self.pipeline.data_sources == {}
        assert self.pipeline.dataset is None
        
        # Verify the mock was called correctly
        mock_close.assert_called_once()
