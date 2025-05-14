"""
Extension for generic NetCDF models.
"""

import logging
from typing import Dict, Any, Optional

from eviz.models.extensions.base import ModelExtension
from eviz.lib.data.sources import DataSource


class GenericExtension(ModelExtension):
    """Extension for generic NetCDF models."""
    
    def process_data_source(self, data_source: DataSource) -> DataSource:
        """Apply generic processing.
        
        Args:
            data_source: The data source to process
            
        Returns:
            The processed data source
        """
        # Apply standard coordinate naming - DISABLED to avoid conflict with NetCDFDataSource._rename_dims
        # self._standardize_coordinates(data_source)
        
        # Apply unit conversions if needed
        self._apply_unit_conversions(data_source)
        
        return data_source
    
    def _standardize_coordinates(self, data_source: DataSource) -> None:
        """Standardize coordinate names.
        
        Args:
            data_source: The data source to process
        """
        # This method is disabled to avoid conflict with NetCDFDataSource._rename_dims
        self.logger.info("Coordinate standardization is now handled by NetCDFDataSource._rename_dims")
        return
        
        if data_source.dataset is None:
            return
        
        # Get coordinate mappings from meta_coords
        meta_coords = getattr(self.config_manager, 'meta_coords', {})
        if not meta_coords:
            return
        
        # Get the model name
        model_name = data_source.model_name
        if not model_name:
            return
        
        # Apply coordinate mappings
        rename_dict = {}
        for generic_name, model_dict in meta_coords.items():
            if model_name in model_dict:
                model_coords = model_dict[model_name]
                
                # Handle comma-separated coordinate list
                if ',' in model_coords:
                    coord_candidates = model_coords.split(',')
                    for coord in coord_candidates:
                        if coord in data_source.dataset.coords:
                            rename_dict[coord] = generic_name
                            break
                elif model_coords in data_source.dataset.coords:
                    rename_dict[model_coords] = generic_name
        
        # Rename coordinates
        if rename_dict:
            try:
                data_source.dataset = data_source.dataset.rename(rename_dict)
                self.logger.info(f"Renamed coordinates: {rename_dict}")
            except Exception as e:
                self.logger.error(f"Error renaming coordinates: {e}")
    
    def _apply_unit_conversions(self, data_source: DataSource) -> None:
        """Apply unit conversions.
        
        Args:
            data_source: The data source to process
        """
        if data_source.dataset is None:
            return
        
        # Get unit conversion specifications
        unit_conversions = getattr(self.config_manager, 'unit_conversions', {})
        if not unit_conversions:
            return
        
        # Apply conversions to each variable
        for var_name, var in data_source.dataset.data_vars.items():
            if 'units' in var.attrs:
                units = var.attrs['units'].lower()
                
                # Check if we have a conversion for these units
                if units in unit_conversions:
                    conversion = unit_conversions[units]
                    target_units = conversion.get('target_units')
                    factor = conversion.get('factor')
                    offset = conversion.get('offset', 0)
                    
                    if target_units and factor:
                        try:
                            # Apply the conversion
                            data_source.dataset[var_name] = var * factor + offset
                            data_source.dataset[var_name].attrs['units'] = target_units
                            data_source.dataset[var_name].attrs['original_units'] = units
                            self.logger.info(f"Converted {var_name} from {units} to {target_units}")
                        except Exception as e:
                            self.logger.error(f"Error converting {var_name}: {e}")
