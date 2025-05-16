import logging
from typing import List
import xarray as xr
import numpy as np
from eviz.lib.data.sources import DataSource
from dataclasses import dataclass


@dataclass()
class DataIntegrator:
    """Data integration stage of the pipeline.
    
    This class handles integrating data from multiple data sources.
    """
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        """Post-initialization setup."""
        self.logger.info("Start init")
    
    def integrate_data_sources(self, data_sources: List[DataSource], **kwargs) -> xr.Dataset:
        """Integrate multiple data sources into a single dataset.
        
        Args:
            data_sources: The data sources to integrate
            **kwargs: Additional integration parameters
            
        Returns:
            An integrated dataset
        """
        self.logger.debug(f"Integrating {len(data_sources)} data sources")
        
        if not data_sources:
            self.logger.warning("No data sources to integrate")
            return None
        
        method = kwargs.get('method', 'merge')
        
        if method == 'merge':
            return self._merge_datasets([ds.dataset for ds in data_sources], **kwargs)
        elif method == 'concatenate':
            return self._concatenate_datasets([ds.dataset for ds in data_sources], **kwargs)
        else:
            self.logger.error(f"Unknown integration method: {method}")
            return None
    
    def _merge_datasets(self, datasets: List[xr.Dataset], **kwargs) -> xr.Dataset:
        """Merge multiple datasets along shared dimensions.
        
        Args:
            datasets: The datasets to merge
            **kwargs: Additional merging parameters
            
        Returns:
            A merged dataset
        """
        self.logger.debug("Merging datasets")
        
        if not datasets:
            return None
        
        join = kwargs.get('join', 'outer')
        compat = kwargs.get('compat', 'override')
        
        try:
            result = xr.merge(datasets, join=join, compat=compat)
            self.logger.info(f"Successfully merged {len(datasets)} datasets")
            return result
        except Exception as e:
            self.logger.error(f"Error merging datasets: {e}")
            return datasets[0]  # Return the first dataset as a fallback
    
    def _concatenate_datasets(self, datasets: List[xr.Dataset], **kwargs) -> xr.Dataset:
        """Concatenate multiple datasets along a specified dimension.
        
        Args:
            datasets: The datasets to concatenate
            **kwargs: Additional concatenation parameters
            
        Returns:
            A concatenated dataset
        """
        self.logger.debug("Concatenating datasets")
        
        if not datasets:
            return None
        
        dim = kwargs.get('dim', 'time')
        
        for i, ds in enumerate(datasets):
            if dim not in ds.dims:
                self.logger.warning(f"Dimension '{dim}' not found in dataset {i}")
                return datasets[0]  # Return the first dataset as a fallback
        
        try:
            result = xr.concat(datasets, dim=dim)
            self.logger.info(f"Successfully concatenated {len(datasets)} datasets along dimension '{dim}'")
            return result
        except Exception as e:
            self.logger.error(f"Error concatenating datasets: {e}")
            return datasets[0]  # Return the first dataset as a fallback
    
    def integrate_variables(self, dataset: xr.Dataset, variables: List[str], operation: str, output_name: str) -> xr.Dataset:
        """Integrate multiple variables within a dataset.
        
        Args:
            dataset: The dataset containing the variables
            variables: The variables to integrate
            operation: The operation to apply ('add', 'subtract', 'multiply', 'divide', 'mean', 'max', 'min')
            output_name: The name of the output variable
            
        Returns:
            The dataset with the integrated variable added
        """
        self.logger.debug(f"Integrating variables {variables} with operation '{operation}'")
        
        if not dataset or not variables:
            return dataset
        
        for var in variables:
            if var not in dataset.data_vars:
                self.logger.warning(f"Variable '{var}' not found in dataset")
                return dataset
        
        try:
            if operation == 'add':
                result = sum(dataset[var] for var in variables)
            elif operation == 'subtract':
                if len(variables) != 2:
                    self.logger.error("Subtract operation requires exactly 2 variables")
                    return dataset
                result = dataset[variables[0]] - dataset[variables[1]]
            elif operation == 'multiply':
                result = dataset[variables[0]].copy()
                for var in variables[1:]:
                    result *= dataset[var]
            elif operation == 'divide':
                if len(variables) != 2:
                    self.logger.error("Divide operation requires exactly 2 variables")
                    return dataset
                result = dataset[variables[0]] / dataset[variables[1]]
            elif operation == 'mean':
                result = sum(dataset[var] for var in variables) / len(variables)
            elif operation == 'max':
                result = dataset[variables[0]].copy()
                for var in variables[1:]:
                    result = xr.where(dataset[var] > result, dataset[var], result)
            elif operation == 'min':
                result = dataset[variables[0]].copy()
                for var in variables[1:]:
                    result = xr.where(dataset[var] < result, dataset[var], result)
            else:
                self.logger.error(f"Unknown operation: {operation}")
                return dataset
            
            dataset[output_name] = result
            dataset[output_name].attrs['long_name'] = f"{operation.capitalize()} of {', '.join(variables)}"
            dataset[output_name].attrs['operation'] = operation
            dataset[output_name].attrs['source_variables'] = variables
            
            self.logger.info(f"Successfully integrated variables {variables} with operation '{operation}'")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error integrating variables: {e}")
            return dataset
    
    def integrate_datasets_by_time(self, datasets: List[xr.Dataset], **kwargs) -> xr.Dataset:
        """Integrate multiple datasets by time.
        
        This method is useful for combining datasets with different time ranges.
        
        Args:
            datasets: The datasets to integrate
            **kwargs: Additional integration parameters
            
        Returns:
            An integrated dataset
        """
        self.logger.debug("Integrating datasets by time")
        
        if not datasets:
            return None
        
        time_dim = kwargs.get('time_dim', 'time')
        
        for i, ds in enumerate(datasets):
            if time_dim not in ds.dims:
                self.logger.warning(f"Time dimension '{time_dim}' not found in dataset {i}")
                return datasets[0]  # Return the first dataset as a fallback
        
        sorted_datasets = sorted(datasets, key=lambda ds: ds[time_dim].values[0])
        
        try:
            result = xr.concat(sorted_datasets, dim=time_dim)
            
            # Remove duplicate time steps if any
            _, index = np.unique(result[time_dim].values, return_index=True)
            result = result.isel({time_dim: index})
            
            self.logger.info(f"Successfully integrated {len(datasets)} datasets by time")
            return result
        except Exception as e:
            self.logger.error(f"Error integrating datasets by time: {e}")
            return datasets[0]  # Return the first dataset as a fallback
