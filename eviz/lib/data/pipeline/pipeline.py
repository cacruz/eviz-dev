import logging
from typing import Dict, List, Optional, Any
import xarray as xr
from eviz.lib.data.sources import DataSource
from eviz.lib.data.pipeline.reader import DataReader
from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.lib.data.pipeline.transformer import DataTransformer
from eviz.lib.data.pipeline.integrator import DataIntegrator


class DataPipeline:
    """
    Orchestrates the data processing workflow for the eViz application.
    
    The DataPipeline class implements a comprehensive data processing system that handles
    loading, processing, transforming, and integrating data from various sources. It serves
    as the central coordinator for all data operations, managing the flow of data through
    different processing stages while maintaining a registry of processed data sources.
    
    The pipeline follows a modular architecture with four main components:
    1. Reader: Loads data from files in various formats (NetCDF, GRIB, HDF5, CSV)
    2. Processor: Performs operations on loaded data (filtering, aggregation, etc.)
    3. Transformer: Converts data between different representations
    4. Integrator: Combines data from multiple sources
    
    This design allows for flexible data processing workflows while maintaining separation
    of concerns between different processing stages.
    
    Attributes:
        config_manager (ConfigManager): Configuration manager containing application settings
        reader (DataReader): Component for reading data from files
        processor (DataProcessor): Component for processing data
        transformer (DataTransformer): Component for transforming data
        integrator (DataIntegrator): Component for integrating data from multiple sources
        data_sources (dict): Dictionary mapping file paths to their corresponding DataSource objects
        logger (logging.Logger): Logger instance for this class
    
    Methods:
        process_file: Process a single file and return the resulting data source
        process_files: Process multiple files and return a list of data sources
        get_data_source: Retrieve a data source by file path
        get_all_data_sources: Get all registered data sources
        create_composite_field: Create a composite field from multiple variables
        integrate_data: Integrate data from multiple sources
    """
    def __init__(self, config_manager=None):
        """Initialize a new DataPipeline.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.reader = DataReader(config_manager)
        self.processor = DataProcessor()
        self.transformer = DataTransformer()
        self.integrator = DataIntegrator()
        self.logger = logging.getLogger(__name__)
        self.data_sources = {}
        self.dataset = None
        self.config_manager = config_manager

    def process_file(self, file_path: str, model_name: Optional[str] = None,
                    process: bool = True, transform: bool = False,
                    transform_params: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> DataSource:
        """Process a single file through the pipeline.
        
        Args:
            file_path: Path to the data file
            model_name: Optional name of the model this data source belongs to
            process: Whether to apply data processing
            transform: Whether to apply data transformation
            transform_params: Parameters for data transformation
            metadata: Optional metadata to attach to the data source
            
        Returns:
            A processed data source
        """
        self.logger.debug(f"Processing file: {file_path}")
        
        data_source = self.reader.read_file(file_path, model_name)

        # Attach metadata if provided
        if metadata and hasattr(data_source, 'metadata'):
            data_source.metadata.update(metadata)

        if process:
            data_source = self.processor.process_data_source(data_source)
        if transform and transform_params:
            data_source = self.transformer.transform_data_source(data_source, **transform_params)
        self.data_sources[file_path] = data_source
        
        return data_source
    
    def process_files(self, file_paths: List[str], model_name: Optional[str] = None,
                     process: bool = True, transform: bool = False,
                     transform_params: Optional[Dict[str, Any]] = None) -> Dict[str, DataSource]:
        """Process multiple files through the pipeline.
        
        Args:
            file_paths: List of paths to data files
            model_name: Optional name of the model these data sources belong to
            process: Whether to apply data processing
            transform: Whether to apply data transformation
            transform_params: Parameters for data transformation
            
        Returns:
            A dictionary mapping file paths to processed data sources
        """
        self.logger.debug(f"Processing {len(file_paths)} files")
        
        result = {}
        for file_path in file_paths:
            try:
                data_source = self.process_file(file_path, model_name, process, transform, transform_params)
                result[file_path] = data_source
            except Exception as e:
                self.logger.error(f"Error processing file: {file_path}. Exception: {e}")
        
        return result
    
    def integrate_data_sources(self, file_paths: Optional[List[str]] = None,
                              integration_params: Optional[Dict[str, Any]] = None) -> xr.Dataset:
        """Integrate data sources into a single dataset.
        
        Args:
            file_paths: List of paths to data files to integrate. If None, all data sources will be integrated.
            integration_params: Parameters for data integration
            
        Returns:
            An integrated dataset
        """
        self.logger.debug("Integrating data sources")
        
        if file_paths:
            data_sources = [self.data_sources[fp] for fp in file_paths if fp in self.data_sources]
        else:
            data_sources = list(self.data_sources.values())
        
        if not data_sources:
            self.logger.warning("No data sources to integrate")
            return None
        
        integration_params = integration_params or {}
        self.dataset = self.integrator.integrate_data_sources(data_sources, **integration_params)
        
        return self.dataset
    
    def integrate_variables(self, variables: List[str], operation: str, output_name: str) -> xr.Dataset:
        """Integrate multiple variables within the dataset.
        
        Args:
            variables: The variables to integrate
            operation: The operation to apply ('add', 'subtract', 'multiply', 'divide', 'mean', 'max', 'min')
            output_name: The name of the output variable
            
        Returns:
            The dataset with the integrated variable added
        """
        self.logger.debug(f"Integrating variables {variables} with operation '{operation}'")
        
        if self.dataset is None:
            self.logger.warning("No dataset available for variable integration")
            return None
        
        self.dataset = self.integrator.integrate_variables(self.dataset, variables, operation, output_name)
        
        return self.dataset
    
    def get_data_source(self, file_path: str) -> Optional[DataSource]:
        """Get a processed data source.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            The processed data source, or None if not found
        """
        return self.data_sources.get(file_path)
    
    def get_all_data_sources(self) -> Dict[str, DataSource]:
        """Get all processed data sources.
        
        Returns:
            A dictionary mapping file paths to processed data sources
        """
        return self.data_sources.copy()
    
    def get_dataset(self) -> Optional[xr.Dataset]:
        """Get the integrated dataset.
        
        Returns:
            The integrated dataset, or None if not available
        """
        return self.dataset
    
    def close(self) -> None:
        """Close all data sources and free resources."""
        self.reader.close()
        self.data_sources.clear()
        self.dataset = None
