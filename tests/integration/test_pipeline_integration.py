from eviz.lib.data.pipeline.pipeline import DataPipeline


class TestPipelineIntegration:
    """Integration tests for the data pipeline components."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.pipeline = DataPipeline()
    
    def test_process_file_to_integration(self, temp_netcdf_file):
        """Test the full pipeline from processing a file to integration."""
        data_source = self.pipeline.process_file(temp_netcdf_file, 'test_model')
        assert data_source is not None
        assert data_source.dataset is not None
        assert 'temperature' in data_source.dataset.data_vars
        assert 'pressure' in data_source.dataset.data_vars
        
        assert temp_netcdf_file in self.pipeline.data_sources
        assert self.pipeline.data_sources[temp_netcdf_file] == data_source
        
        dataset = self.pipeline.integrate_data_sources()
        assert dataset is not None
        assert 'temperature' in dataset.data_vars
        assert 'pressure' in dataset.data_vars
        assert self.pipeline.dataset == dataset
    
    def test_process_files_to_integration(self, temp_netcdf_file):
        """Test the full pipeline from processing multiple files to integration."""
        data_sources = self.pipeline.process_files([temp_netcdf_file], 'test_model')
        
        assert data_sources is not None
        assert len(data_sources) == 1
        assert temp_netcdf_file in data_sources
        assert data_sources[temp_netcdf_file].dataset is not None
        assert 'temperature' in data_sources[temp_netcdf_file].dataset.data_vars
        assert 'pressure' in data_sources[temp_netcdf_file].dataset.data_vars
        
        assert temp_netcdf_file in self.pipeline.data_sources
        assert self.pipeline.data_sources[temp_netcdf_file] == data_sources[temp_netcdf_file]
        
        dataset = self.pipeline.integrate_data_sources()
        assert dataset is not None
        assert 'temperature' in dataset.data_vars
        assert 'pressure' in dataset.data_vars
        assert self.pipeline.dataset == dataset
    
    def test_process_file_with_transform(self, temp_netcdf_file):
        """Test processing a file with transformation."""
        transform_params = {
            'subset': True,
            'lat_range': (0, 45),
            'lon_range': (0, 180)
        }
        data_source = self.pipeline.process_file(
            temp_netcdf_file,
            'test_model',
            transform=True,
            transform_params=transform_params
        )
        
        assert data_source is not None
        assert data_source.dataset is not None
        assert 'temperature' in data_source.dataset.data_vars
        assert 'pressure' in data_source.dataset.data_vars
        
        assert len(data_source.dataset.coords['lat']) == 2  # (0, 45)
        assert len(data_source.dataset.coords['lon']) == 3  # (0, 90, 180)
    
    def test_integrate_variables(self, temp_netcdf_file):
        """Test integrating variables."""
        self.pipeline.process_file(temp_netcdf_file, 'test_model')
        
        self.pipeline.integrate_data_sources()
        result = self.pipeline.integrate_variables(
            ['temperature', 'pressure'],
            'add',
            'total'
        )
        assert result is not None
        assert 'total' in result.data_vars
        assert result['total'].attrs['operation'] == 'add'
    
    def test_full_pipeline_workflow(self, temp_netcdf_file):
        """Test the full pipeline workflow."""
        transform_params = {
            'subset': True,
            'lat_range': (0, 45),
            'lon_range': (0, 180)
        }
        self.pipeline.process_file(
            temp_netcdf_file,
            'test_model',
            transform=True,
            transform_params=transform_params
        )
        
        self.pipeline.integrate_data_sources()
        result = self.pipeline.integrate_variables(
            ['temperature', 'pressure'],
            'add',
            'total'
        )
        assert result is not None
        assert 'total' in result.data_vars
        assert result['total'].attrs['operation'] == 'add'
        
        dataset = self.pipeline.get_dataset()
        
        assert dataset is not None
        assert 'temperature' in dataset.data_vars
        assert 'pressure' in dataset.data_vars
        assert 'total' in dataset.data_vars
        
        self.pipeline.close()
        
        assert self.pipeline.data_sources == {}
        assert self.pipeline.dataset is None
