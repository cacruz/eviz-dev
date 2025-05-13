from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import xarray as xr
import pandas as pd
import numpy as np
import logging

@dataclass
class DataIntegrator:
    """
    Class for integrating data from multiple sources and readers.
    """
    config_manager: 'ConfigManager'
    
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
    
    def __post_init__(self):
        self.logger.info("Initializing DataIntegrator")
    
    def integrate_datasets(self, source_name: str, file_paths: List[str]) -> Optional[xr.Dataset]:
        """
        Integrate multiple datasets into a single dataset.
        
        Args:
            source_name (str): The name of the data source
            file_paths (List[str]): List of file paths to integrate
            
        Returns:
            Optional[xr.Dataset]: The integrated dataset or None if integration failed
        """
        datasets = []
        
        for file_path in file_paths:
            reader = self.config_manager.get_reader_for_file(source_name, file_path)
            if not reader:
                self.logger.warning(f"No reader found for {file_path}")
                continue
                
            try:
                data = reader.read_data(file_path)
                if data:
                    # Convert to xarray Dataset if it's not already
                    if isinstance(data, dict) and 'vars' in data:
                        # Assuming 'vars' contains the actual data variables
                        ds = xr.Dataset(data['vars'])
                        datasets.append(ds)
                    elif isinstance(data, xr.Dataset):
                        datasets.append(data)
                    elif isinstance(data, pd.DataFrame):
                        ds = xr.Dataset.from_dataframe(data)
                        datasets.append(ds)
                    else:
                        self.logger.warning(f"Unsupported data type from {file_path}")
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {str(e)}")
        
        if not datasets:
            return None
            
        # Merge all datasets
        try:
            integrated_dataset = xr.merge(datasets)
            return integrated_dataset
        except Exception as e:
            self.logger.error(f"Error merging datasets: {str(e)}")
            return None
    
    def get_variable_from_any_source(self, source_name: str, variable_name: str) -> Optional[xr.DataArray]:
        """
        Get a variable from any available reader.
        
        Args:
            source_name (str): The name of the data source
            variable_name (str): The name of the variable
            
        Returns:
            Optional[xr.DataArray]: The variable data or None if not found
        """
        if source_name in self.config_manager.input_config.readers:
            for reader_type, reader in self.config_manager.input_config.readers[source_name].items():
                try:
                    for file_idx, file_entry in self.config_manager.file_list.items():
                        if file_entry.get('source_name') == source_name:
                            file_path = file_entry['filename']
                            data = reader.read_data(file_path)
                            if data and 'vars' in data and variable_name in data['vars']:
                                return data['vars'][variable_name]
                except Exception as e:
                    self.logger.error(f"Error reading variable {variable_name}: {str(e)}")
        
        return None