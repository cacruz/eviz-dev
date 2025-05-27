import unittest
from unittest.mock import patch, MagicMock

import pytest

from eviz.lib.data.pipeline.pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    """Test cases for the DataPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = DataPipeline()
        self.mock_data_source = MagicMock()

    def test_init(self):
        """Test initialization."""
        assert hasattr(self.pipeline, 'reader')
        assert hasattr(self.pipeline, 'processor')
        assert hasattr(self.pipeline, 'transformer')
        assert hasattr(self.pipeline, 'integrator')
        assert hasattr(self.pipeline, 'data_sources')
        assert hasattr(self.pipeline, 'dataset')
        assert hasattr(self.pipeline, 'config_manager')

    @patch('eviz.lib.data.pipeline.reader.DataReader.read_file')
    @patch('eviz.lib.data.pipeline.processor.DataProcessor.process_data_source')
    def test_process_file(self, mock_process, mock_read):
        """Test processing a single file."""
        mock_read.return_value = self.mock_data_source
        mock_process.return_value = self.mock_data_source

        result = self.pipeline.process_file('test_file.nc', 'test_model')
        assert result == self.mock_data_source
        assert self.pipeline.data_sources['test_file.nc'] == self.mock_data_source

        # Verify the mocks were called correctly
        mock_read.assert_called_once_with('test_file.nc', 'test_model', file_format=None)
        mock_process.assert_called_once_with(self.mock_data_source)

    @patch('eviz.lib.data.pipeline.reader.DataReader.read_file')
    @patch('eviz.lib.data.pipeline.processor.DataProcessor.process_data_source')
    @patch('eviz.lib.data.pipeline.transformer.DataTransformer.transform_data_source')
    def test_process_file_with_transform(self, mock_transform, mock_process, mock_read):
        """Test processing a single file with transformation."""
        mock_read.return_value = self.mock_data_source
        mock_process.return_value = self.mock_data_source
        mock_transform.return_value = self.mock_data_source

        transform_params = {'regrid': True}
        result = self.pipeline.process_file('test_file.nc', 'test_model',
                                          transform=True, transform_params=transform_params)

        assert result == self.mock_data_source
        assert self.pipeline.data_sources['test_file.nc'] == self.mock_data_source

        mock_read.assert_called_once_with('test_file.nc', 'test_model', file_format=None)
        mock_process.assert_called_once_with(self.mock_data_source)
        mock_transform.assert_called_once_with(self.mock_data_source, **transform_params)

    @patch('eviz.lib.data.pipeline.reader.DataReader.read_file')
    def test_process_files(self, mock_read):
        """Test processing multiple files."""
        mock_read.side_effect = [self.mock_data_source, self.mock_data_source]

        file_paths = ['file1.nc', 'file2.nc']
        result = self.pipeline.process_files(file_paths, 'test_model')

        assert len(result) == 2
        assert result['file1.nc'] == self.mock_data_source
        assert result['file2.nc'] == self.mock_data_source

        assert mock_read.call_count == 2

    @patch('eviz.lib.data.pipeline.reader.DataReader.read_file')
    def test_process_file_with_error(self, mock_read):
        """Test processing a file with an error."""
        mock_read.side_effect = Exception("Test error")

        with pytest.raises(Exception):
            self.pipeline.process_file('test_file.nc', 'test_model')

    @patch('eviz.lib.data.pipeline.integrator.DataIntegrator.integrate_data_sources')
    def test_integrate_data_sources(self, mock_integrate):
        """Test integrating data sources."""
        mock_dataset = MagicMock()
        mock_integrate.return_value = mock_dataset

        self.pipeline.data_sources = {'file1.nc': self.mock_data_source, 'file2.nc': self.mock_data_source}
        result = self.pipeline.integrate_data_sources()

        assert result == mock_dataset
        assert self.pipeline.dataset == mock_dataset
        mock_integrate.assert_called_once_with([self.mock_data_source, self.mock_data_source])

    @patch('eviz.lib.data.pipeline.integrator.DataIntegrator.integrate_variables')
    def test_integrate_variables(self, mock_integrate):
        """Test integrating variables."""
        mock_dataset = MagicMock()
        self.pipeline.dataset = mock_dataset
        mock_integrate.return_value = mock_dataset

        variables = ['var1', 'var2']
        operation = 'add'
        output_name = 'var_sum'

        result = self.pipeline.integrate_variables(variables, operation, output_name)

        assert result == mock_dataset
        mock_integrate.assert_called_once_with(mock_dataset, variables, operation, output_name)

    def test_get_data_source(self):
        """Test getting a data source."""
        self.pipeline.data_sources = {'file1.nc': self.mock_data_source}
        result = self.pipeline.get_data_source('file1.nc')
        assert result == self.mock_data_source

    def test_get_all_data_sources(self):
        """Test getting all data sources."""
        self.pipeline.data_sources = {'file1.nc': self.mock_data_source, 'file2.nc': self.mock_data_source}
        result = self.pipeline.get_all_data_sources()
        assert len(result) == 2
        assert result['file1.nc'] == self.mock_data_source
        assert result['file2.nc'] == self.mock_data_source

    def test_get_dataset(self):
        """Test getting the dataset."""
        mock_dataset = MagicMock()
        self.pipeline.dataset = mock_dataset
        result = self.pipeline.get_dataset()
        assert result == mock_dataset

    @patch('eviz.lib.data.pipeline.reader.DataReader.close')
    def test_close(self, mock_close):
        """Test closing the pipeline."""
        mock_dataset = MagicMock()
        self.pipeline.dataset = mock_dataset
        self.pipeline.data_sources = {'file1.nc': self.mock_data_source}

        self.pipeline.close()

        mock_close.assert_called_once()
        assert len(self.pipeline.data_sources) == 0
        assert self.pipeline.dataset is None
