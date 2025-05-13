"""
Unit tests for the DataIntegrator class.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from eviz.lib.data.pipeline.integrator import DataIntegrator
from eviz.lib.data.sources import DataSource


class TestDataIntegrator:
    """Test cases for the DataIntegrator class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.integrator = DataIntegrator()
        
        # Create mock data sources
        self.mock_data_source1 = MagicMock(spec=DataSource)
        self.mock_data_source2 = MagicMock(spec=DataSource)
        
        # Create test datasets
        self.test_dataset1 = xr.Dataset(
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
        
        self.test_dataset2 = xr.Dataset(
            data_vars={
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
        
        self.mock_data_source1.dataset = self.test_dataset1
        self.mock_data_source2.dataset = self.test_dataset2
    
    def test_init(self):
        """Test initialization of DataIntegrator."""
        assert self.integrator is not None
    
    def test_integrate_data_sources_empty(self):
        """Test integrating empty data sources."""
        # Call the method
        result = self.integrator.integrate_data_sources([])
        
        # Verify the result
        assert result is None
    
    @patch('xarray.merge')
    def test_integrate_data_sources_merge(self, mock_merge):
        """Test integrating data sources with merge method."""
        # Setup mock
        expected_dataset = xr.Dataset()
        mock_merge.return_value = expected_dataset
        
        # Call the method
        result = self.integrator.integrate_data_sources(
            [self.mock_data_source1, self.mock_data_source2],
            method='merge'
        )
        
        # Verify the result
        assert isinstance(result, xr.Dataset)
        
        # Verify the mock was called correctly
        mock_merge.assert_called_once_with(
            [self.test_dataset1, self.test_dataset2],
            join='outer',
            compat='override'
        )
    
    @patch('xarray.concat')
    def test_integrate_data_sources_concatenate(self, mock_concat):
        """Test integrating data sources with concatenate method."""
        # Setup mock
        expected_dataset = xr.Dataset()
        mock_concat.return_value = expected_dataset
        
        # Call the method
        result = self.integrator.integrate_data_sources(
            [self.mock_data_source1, self.mock_data_source2],
            method='concatenate',
            dim='time'
        )
        
        # Verify the result
        assert isinstance(result, xr.Dataset)
        
        # Verify the mock was called correctly
        mock_concat.assert_called_once_with(
            [self.test_dataset1, self.test_dataset2],
            dim='time'
        )
    
    def test_integrate_data_sources_unknown_method(self):
        """Test integrating data sources with unknown method."""
        # Call the method
        result = self.integrator.integrate_data_sources(
            [self.mock_data_source1, self.mock_data_source2],
            method='unknown'
        )
        
        # Verify the result
        assert result is None
    
    @patch('xarray.merge')
    def test_merge_datasets(self, mock_merge):
        """Test merging datasets."""
        # Setup mock
        expected_dataset = xr.Dataset()
        mock_merge.return_value = expected_dataset
        
        # Call the method
        result = self.integrator._merge_datasets(
            [self.test_dataset1, self.test_dataset2],
            join='inner',
            compat='identical'
        )
        
        # Verify the result
        assert isinstance(result, xr.Dataset)
        
        # Verify the mock was called correctly
        mock_merge.assert_called_once_with(
            [self.test_dataset1, self.test_dataset2],
            join='inner',
            compat='identical'
        )
    
    @patch('xarray.merge')
    def test_merge_datasets_error(self, mock_merge):
        """Test merging datasets with an error."""
        # Setup mock
        mock_merge.side_effect = Exception("Test error")
        
        # Call the method
        result = self.integrator._merge_datasets(
            [self.test_dataset1, self.test_dataset2]
        )
        
        # Verify the result
        assert result == self.test_dataset1  # Should return the first dataset as fallback
    
    @patch('xarray.concat')
    def test_concatenate_datasets(self, mock_concat):
        """Test concatenating datasets."""
        # Setup mock
        expected_dataset = xr.Dataset()
        mock_concat.return_value = expected_dataset
        
        # Call the method
        result = self.integrator._concatenate_datasets(
            [self.test_dataset1, self.test_dataset2],
            dim='time'
        )
        
        # Verify the result
        assert isinstance(result, xr.Dataset)
        
        # Verify the mock was called correctly
        mock_concat.assert_called_once_with(
            [self.test_dataset1, self.test_dataset2],
            dim='time'
        )
    
    def test_concatenate_datasets_missing_dim(self):
        """Test concatenating datasets with missing dimension."""
        # Create a dataset without the time dimension
        dataset_no_time = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(3, 4),
                    dims=['lat', 'lon'],
                    coords={
                        'lat': np.array([0, 45, 90]),
                        'lon': np.array([0, 90, 180, 270])
                    }
                )
            }
        )
        
        # Call the method
        result = self.integrator._concatenate_datasets(
            [self.test_dataset1, dataset_no_time],
            dim='time'
        )
        
        # Verify the result
        assert result == self.test_dataset1  # Should return the first dataset as fallback
    
    @patch('xarray.concat')
    def test_concatenate_datasets_error(self, mock_concat):
        """Test concatenating datasets with an error."""
        # Setup mock
        mock_concat.side_effect = Exception("Test error")
        
        # Call the method
        result = self.integrator._concatenate_datasets(
            [self.test_dataset1, self.test_dataset2],
            dim='time'
        )
        
        # Verify the result
        assert result == self.test_dataset1  # Should return the first dataset as fallback
    
    def test_integrate_variables_add(self):
        """Test integrating variables with add operation."""
        # Create a dataset with multiple variables
        dataset = xr.Dataset(
            data_vars={
                'var1': xr.DataArray(
                    data=np.ones((2, 3)),
                    dims=['x', 'y']
                ),
                'var2': xr.DataArray(
                    data=np.ones((2, 3)) * 2,
                    dims=['x', 'y']
                )
            }
        )
        
        # Call the method
        result = self.integrator.integrate_variables(
            dataset,
            ['var1', 'var2'],
            'add',
            'sum'
        )
        
        # Verify the result
        assert 'sum' in result.data_vars
        assert np.all(result['sum'].values == 3.0)  # 1 + 2 = 3
        assert result['sum'].attrs['operation'] == 'add'
    
    def test_integrate_variables_subtract(self):
        """Test integrating variables with subtract operation."""
        # Create a dataset with multiple variables
        dataset = xr.Dataset(
            data_vars={
                'var1': xr.DataArray(
                    data=np.ones((2, 3)) * 5,
                    dims=['x', 'y']
                ),
                'var2': xr.DataArray(
                    data=np.ones((2, 3)) * 2,
                    dims=['x', 'y']
                )
            }
        )
        
        # Call the method
        result = self.integrator.integrate_variables(
            dataset,
            ['var1', 'var2'],
            'subtract',
            'diff'
        )
        
        # Verify the result
        assert 'diff' in result.data_vars
        assert np.all(result['diff'].values == 3.0)  # 5 - 2 = 3
        assert result['diff'].attrs['operation'] == 'subtract'
    
    def test_integrate_variables_multiply(self):
        """Test integrating variables with multiply operation."""
        # Create a dataset with multiple variables
        dataset = xr.Dataset(
            data_vars={
                'var1': xr.DataArray(
                    data=np.ones((2, 3)) * 2,
                    dims=['x', 'y']
                ),
                'var2': xr.DataArray(
                    data=np.ones((2, 3)) * 3,
                    dims=['x', 'y']
                )
            }
        )
        
        # Call the method
        result = self.integrator.integrate_variables(
            dataset,
            ['var1', 'var2'],
            'multiply',
            'product'
        )
        
        # Verify the result
        assert 'product' in result.data_vars
        assert np.all(result['product'].values == 6.0)  # 2 * 3 = 6
        assert result['product'].attrs['operation'] == 'multiply'
    
    def test_integrate_variables_divide(self):
        """Test integrating variables with divide operation."""
        # Create a dataset with multiple variables
        dataset = xr.Dataset(
            data_vars={
                'var1': xr.DataArray(
                    data=np.ones((2, 3)) * 6,
                    dims=['x', 'y']
                ),
                'var2': xr.DataArray(
                    data=np.ones((2, 3)) * 2,
                    dims=['x', 'y']
                )
            }
        )
        
        # Call the method
        result = self.integrator.integrate_variables(
            dataset,
            ['var1', 'var2'],
            'divide',
            'ratio'
        )
        
        # Verify the result
        assert 'ratio' in result.data_vars
        assert np.all(result['ratio'].values == 3.0)  # 6 / 2 = 3
        assert result['ratio'].attrs['operation'] == 'divide'
    
    def test_integrate_variables_mean(self):
        """Test integrating variables with mean operation."""
        # Create a dataset with multiple variables
        dataset = xr.Dataset(
            data_vars={
                'var1': xr.DataArray(
                    data=np.ones((2, 3)) * 2,
                    dims=['x', 'y']
                ),
                'var2': xr.DataArray(
                    data=np.ones((2, 3)) * 4,
                    dims=['x', 'y']
                )
            }
        )
        
        # Call the method
        result = self.integrator.integrate_variables(
            dataset,
            ['var1', 'var2'],
            'mean',
            'avg'
        )
        
        # Verify the result
        assert 'avg' in result.data_vars
        assert np.all(result['avg'].values == 3.0)  # (2 + 4) / 2 = 3
        assert result['avg'].attrs['operation'] == 'mean'
    
    def test_integrate_variables_max(self):
        """Test integrating variables with max operation."""
        # Create a dataset with multiple variables
        dataset = xr.Dataset(
            data_vars={
                'var1': xr.DataArray(
                    data=np.ones((2, 3)) * 2,
                    dims=['x', 'y']
                ),
                'var2': xr.DataArray(
                    data=np.ones((2, 3)) * 4,
                    dims=['x', 'y']
                )
            }
        )
        
        # Call the method
        result = self.integrator.integrate_variables(
            dataset,
            ['var1', 'var2'],
            'max',
            'maximum'
        )
        
        # Verify the result
        assert 'maximum' in result.data_vars
        assert np.all(result['maximum'].values == 4.0)  # max(2, 4) = 4
        assert result['maximum'].attrs['operation'] == 'max'
    
    def test_integrate_variables_min(self):
        """Test integrating variables with min operation."""
        # Create a dataset with multiple variables
        dataset = xr.Dataset(
            data_vars={
                'var1': xr.DataArray(
                    data=np.ones((2, 3)) * 2,
                    dims=['x', 'y']
                ),
                'var2': xr.DataArray(
                    data=np.ones((2, 3)) * 4,
                    dims=['x', 'y']
                )
            }
        )
        
        # Call the method
        result = self.integrator.integrate_variables(
            dataset,
            ['var1', 'var2'],
            'min',
            'minimum'
        )
        
        # Verify the result
        assert 'minimum' in result.data_vars
        assert np.all(result['minimum'].values == 2.0)  # min(2, 4) = 2
        assert result['minimum'].attrs['operation'] == 'min'
    
    def test_integrate_variables_unknown_operation(self):
        """Test integrating variables with unknown operation."""
        # Create a dataset with multiple variables
        dataset = xr.Dataset(
            data_vars={
                'var1': xr.DataArray(
                    data=np.ones((2, 3)),
                    dims=['x', 'y']
                ),
                'var2': xr.DataArray(
                    data=np.ones((2, 3)) * 2,
                    dims=['x', 'y']
                )
            }
        )
        
        # Call the method
        result = self.integrator.integrate_variables(
            dataset,
            ['var1', 'var2'],
            'unknown',
            'result'
        )
        
        # Verify the result
        assert 'result' not in result.data_vars
        assert result == dataset  # Should return the original dataset
    
    def test_integrate_variables_missing_variable(self):
        """Test integrating variables with a missing variable."""
        # Create a dataset with one variable
        dataset = xr.Dataset(
            data_vars={
                'var1': xr.DataArray(
                    data=np.ones((2, 3)),
                    dims=['x', 'y']
                )
            }
        )
        
        # Call the method
        result = self.integrator.integrate_variables(
            dataset,
            ['var1', 'var2'],  # var2 doesn't exist
            'add',
            'sum'
        )
        
        # Verify the result
        assert 'sum' not in result.data_vars
        assert result == dataset  # Should return the original dataset
    
    @patch('numpy.unique')
    def test_integrate_datasets_by_time(self, mock_unique):
        """Test integrating datasets by time."""
        # Setup mock
        mock_unique.return_value = (np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'), np.array([0, 1]))
        
        # Create datasets with different time ranges
        dataset1 = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3),
                    dims=['time', 'lat'],
                    coords={
                        'time': np.array(['2022-01-01', '2022-01-02'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90])
                    }
                )
            }
        )
        
        dataset2 = xr.Dataset(
            data_vars={
                'temperature': xr.DataArray(
                    data=np.random.rand(2, 3),
                    dims=['time', 'lat'],
                    coords={
                        'time': np.array(['2022-01-03', '2022-01-04'], dtype='datetime64[D]'),
                        'lat': np.array([0, 45, 90])
                    }
                )
            }
        )
        
        # Call the method
        with patch('xarray.concat') as mock_concat:
            mock_concat.return_value = xr.Dataset()
            result = self.integrator.integrate_datasets_by_time([dataset1, dataset2])
            
            # Verify the mock was called correctly
            mock_concat.assert_called_once()
