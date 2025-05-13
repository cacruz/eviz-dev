"""
Unit tests for the DataReader class.
"""

import os
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch, mock_open

from eviz.lib.data.pipeline.reader import DataReader
from eviz.lib.data.sources import DataSource
from eviz.lib.data.factory import DataSourceFactory


class TestDataReader:
    """Test cases for the DataReader class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.reader = DataReader()
        
        # Create a mock data source
        self.mock_data_source = MagicMock(spec=DataSource)
        self.mock_data_source.dataset = xr.Dataset()
    
    def test_init(self):
        """Test initialization of DataReader."""
        assert isinstance(self.reader.factory, DataSourceFactory)
        assert self.reader.data_sources == {}
    
    @patch('os.path.exists')
    @patch('eviz.lib.data.factory.DataSourceFactory.create_data_source')
    def test_read_file(self, mock_create_source, mock_exists):
        """Test reading a file."""
        # Setup mocks
        mock_exists.return_value = True
        mock_create_source.return_value = self.mock_data_source
        
        # Call the method
        result = self.reader.read_file('test_file.nc', 'test_model')
        
        # Verify the result
        assert result == self.mock_data_source
        assert self.reader.data_sources['test_file.nc'] == self.mock_data_source
        
        # Verify the mocks were called correctly
        mock_exists.assert_called_once_with('test_file.nc')
        mock_create_source.assert_called_once_with('test_file.nc', 'test_model')
        self.mock_data_source.load_data.assert_called_once_with('test_file.nc')
    
    @patch('os.path.exists')
    def test_read_file_not_found(self, mock_exists):
        """Test reading a file that doesn't exist."""
        # Setup mock
        mock_exists.return_value = False
        
        # Call the method and verify it raises an exception
        with pytest.raises(FileNotFoundError):
            self.reader.read_file('non_existent_file.nc')
        
        # Verify the mock was called correctly
        mock_exists.assert_called_once_with('non_existent_file.nc')
    
    @patch('os.path.exists')
    @patch('eviz.lib.data.factory.DataSourceFactory.create_data_source')
    def test_read_file_cached(self, mock_create_source, mock_exists):
        """Test reading a file that's already cached."""
        # Setup mocks
        mock_exists.return_value = True
        
        # Setup cache
        self.reader.data_sources['test_file.nc'] = self.mock_data_source
        
        # Call the method
        result = self.reader.read_file('test_file.nc')
        
        # Verify the result
        assert result == self.mock_data_source
        
        # Verify the mocks were called correctly
        mock_exists.assert_called_once_with('test_file.nc')
        mock_create_source.assert_not_called()
        self.mock_data_source.load_data.assert_not_called()
    
    @patch('eviz.lib.data.pipeline.reader.DataReader.read_file')
    def test_read_files(self, mock_read_file):
        """Test reading multiple files."""
        # Setup mock
        mock_read_file.side_effect = [self.mock_data_source, self.mock_data_source]
        
        # Call the method
        file_paths = ['file1.nc', 'file2.nc']
        result = self.reader.read_files(file_paths, 'test_model')
        
        # Verify the result
        assert len(result) == 2
        assert result['file1.nc'] == self.mock_data_source
        assert result['file2.nc'] == self.mock_data_source
        
        # Verify the mock was called correctly
        assert mock_read_file.call_count == 2
        mock_read_file.assert_any_call('file1.nc', 'test_model')
        mock_read_file.assert_any_call('file2.nc', 'test_model')
    
    @patch('eviz.lib.data.pipeline.reader.DataReader.read_file')
    def test_read_files_with_error(self, mock_read_file):
        """Test reading multiple files with an error."""
        # Setup mock
        mock_read_file.side_effect = [self.mock_data_source, Exception("Test error")]
        
        # Call the method
        file_paths = ['file1.nc', 'file2.nc']
        result = self.reader.read_files(file_paths, 'test_model')
        
        # Verify the result
        assert len(result) == 1
        assert result['file1.nc'] == self.mock_data_source
        assert 'file2.nc' not in result
        
        # Verify the mock was called correctly
        assert mock_read_file.call_count == 2
    
    def test_get_data_source(self):
        """Test getting a data source."""
        # Setup data sources
        self.reader.data_sources = {
            'file1.nc': self.mock_data_source
        }
        
        # Call the method
        result = self.reader.get_data_source('file1.nc')
        
        # Verify the result
        assert result == self.mock_data_source
        
        # Test getting a non-existent data source
        result = self.reader.get_data_source('non_existent.nc')
        assert result is None
    
    def test_get_all_data_sources(self):
        """Test getting all data sources."""
        # Setup data sources
        self.reader.data_sources = {
            'file1.nc': self.mock_data_source,
            'file2.nc': self.mock_data_source
        }
        
        # Call the method
        result = self.reader.get_all_data_sources()
        
        # Verify the result
        assert len(result) == 2
        assert result['file1.nc'] == self.mock_data_source
        assert result['file2.nc'] == self.mock_data_source
        
        # Verify it's a copy
        result['file3.nc'] = self.mock_data_source
        assert 'file3.nc' not in self.reader.data_sources
    
    def test_close(self):
        """Test closing the reader."""
        # Setup data sources
        mock_data_source1 = MagicMock(spec=DataSource)
        mock_data_source2 = MagicMock(spec=DataSource)
        self.reader.data_sources = {
            'file1.nc': mock_data_source1,
            'file2.nc': mock_data_source2
        }
        
        # Call the method
        self.reader.close()
        
        # Verify the result
        assert self.reader.data_sources == {}
        
        # Verify the mocks were called correctly
        mock_data_source1.close.assert_called_once()
        mock_data_source2.close.assert_called_once()
