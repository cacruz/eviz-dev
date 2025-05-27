import unittest
from unittest.mock import patch, MagicMock

import pytest

from eviz.lib.data.pipeline.reader import DataReader


class TestDataReader(unittest.TestCase):
    """Test cases for the DataReader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.reader = DataReader()
        self.mock_data_source = MagicMock()

    def test_init(self):
        """Test initialization."""
        assert hasattr(self.reader, 'factory')
        assert hasattr(self.reader, 'data_sources')

    @patch('os.path.exists')
    @patch('eviz.lib.data.factory.source_factory.DataSourceFactory.create_data_source')
    def test_read_file(self, mock_create_data_source, mock_exists):
        # Set up mocks
        mock_exists.return_value = True
        mock_data_source = MagicMock()
        mock_create_data_source.return_value = mock_data_source

        # Create reader and call read_file
        reader = DataReader()
        result = reader.read_file('file1.nc')

        # Assertions
        assert result == mock_data_source
        mock_create_data_source.assert_called_once_with('file1.nc', None, file_format=None)
        mock_data_source.load_data.assert_called_once_with('file1.nc')

    @patch('os.path.exists')
    @patch('eviz.lib.data.factory.source_factory.DataSourceFactory.create_data_source')
    def test_read_file_with_model_name(self, mock_create_data_source, mock_exists):
        # Set up mocks
        mock_exists.return_value = True
        mock_data_source = MagicMock()
        mock_create_data_source.return_value = mock_data_source

        # Create reader and call read_file
        reader = DataReader()
        result = reader.read_file('file1.nc', 'model1')

        # Assertions
        assert result == mock_data_source
        mock_create_data_source.assert_called_once_with('file1.nc', 'model1', file_format=None)
        mock_data_source.load_data.assert_called_once_with('file1.nc')

    @patch('os.path.exists')
    def test_read_file_not_found(self, mock_exists):
        # Set up mocks
        mock_exists.return_value = False

        # Create reader and call read_file
        reader = DataReader()
        with pytest.raises(FileNotFoundError):
            reader.read_file('file1.nc')

    @patch('os.path.exists')
    @patch('eviz.lib.data.factory.source_factory.DataSourceFactory.create_data_source')
    def test_read_file_with_error(self, mock_create_data_source, mock_exists):
        # Set up mocks
        mock_exists.return_value = True
        mock_create_data_source.side_effect = Exception("Test error")

        # Create reader and call read_file
        reader = DataReader()
        with pytest.raises(Exception):
            reader.read_file('file1.nc')

    @patch('eviz.lib.data.pipeline.reader.DataReader.read_file')
    def test_read_files(self, mock_read_file):
        """Test reading multiple files."""
        mock_read_file.side_effect = [self.mock_data_source, self.mock_data_source]

        file_paths = ['file1.nc', 'file2.nc']
        result = self.reader.read_files(file_paths, 'test_model')

        assert len(result) == 2
        assert result['file1.nc'] == self.mock_data_source
        assert result['file2.nc'] == self.mock_data_source

        assert mock_read_file.call_count == 2
        mock_read_file.assert_any_call('file1.nc', 'test_model', file_format=None)
        mock_read_file.assert_any_call('file2.nc', 'test_model', file_format=None)

    def test_get_data_source(self):
        """Test getting a data source."""
        self.reader.data_sources = {'file1.nc': self.mock_data_source}
        result = self.reader.get_data_source('file1.nc')
        assert result == self.mock_data_source

    def test_get_all_data_sources(self):
        """Test getting all data sources."""
        self.reader.data_sources = {'file1.nc': self.mock_data_source, 'file2.nc': self.mock_data_source}
        result = self.reader.get_all_data_sources()
        assert len(result) == 2
        assert result['file1.nc'] == self.mock_data_source
        assert result['file2.nc'] == self.mock_data_source

    def test_close(self):
        """Test closing the reader."""
        mock_data_source1 = MagicMock()
        mock_data_source2 = MagicMock()
        self.reader.data_sources = {'file1.nc': mock_data_source1, 'file2.nc': mock_data_source2}

        self.reader.close()

        mock_data_source1.close.assert_called_once()
        mock_data_source2.close.assert_called_once()
        assert len(self.reader.data_sources) == 0
