"""
Extension for CCM model.
"""

import os
import logging
from typing import Dict, Any, Optional

from eviz.models.extensions.base import ModelExtension
from eviz.lib.data.sources import DataSource


class CCMExtension(ModelExtension):
    """Extension for CCM model."""
    
    def process_data_source(self, data_source: DataSource) -> DataSource:
        """Apply CCM-specific processing.
        
        Args:
            data_source: The data source to process
            
        Returns:
            The processed data source
        """
        # Handle tropopause height
        if hasattr(self.config_manager, 'use_trop_height') and self.config_manager.use_trop_height:
            self._apply_tropopause_height(data_source)
        
        # Handle specific humidity conversion
        if hasattr(self.config_manager, 'use_sphum_conv') and self.config_manager.use_sphum_conv:
            self._apply_sphum_conversion(data_source)
        
        return data_source
    
    def _apply_tropopause_height(self, data_source: DataSource) -> None:
        """Apply tropopause height to the data source.
        
        Args:
            data_source: The data source to process
        """
        self.logger.info("Applying tropopause height")
        
        # Get tropopause height file list
        trop_height_files = getattr(self.config_manager, 'trop_height_file_list', {})
        if not trop_height_files:
            self.logger.warning("No tropopause height files specified")
            return
        
        # Get the model name
        model_name = data_source.model_name
        if not model_name:
            self.logger.warning("No model name specified for data source")
            return
        
        # Find the tropopause height file for this model
        trop_file = None
        trop_field_name = None
        for file_info in trop_height_files.values():
            if file_info.get('exp_name') == model_name:
                trop_file = os.path.join(file_info.get('location', ''), file_info.get('name', ''))
                trop_field_name = file_info.get('trop_field_name')
                break
        
        if not trop_file or not trop_field_name:
            self.logger.warning(f"No tropopause height file found for model {model_name}")
            return
        
        # Load the tropopause height data
        try:
            from eviz.lib.data.pipeline import DataReader
            reader = DataReader()
            trop_source = reader.read_file(trop_file)
            
            # Extract the tropopause height field
            if trop_field_name in trop_source.dataset:
                trop_height = trop_source.dataset[trop_field_name]
                
                # Add the tropopause height to the data source's dataset
                data_source.dataset['tropopause_height'] = trop_height
                self.logger.info(f"Added tropopause height from {trop_file}")
            else:
                self.logger.warning(f"Tropopause field {trop_field_name} not found in {trop_file}")
        except Exception as e:
            self.logger.error(f"Error loading tropopause height: {e}")
    
    def _apply_sphum_conversion(self, data_source: DataSource) -> None:
        """Apply specific humidity conversion to the data source.
        
        Args:
            data_source: The data source to process
        """
        self.logger.info("Applying specific humidity conversion")
        
        # Get specific humidity file list
        sphum_files = getattr(self.config_manager, 'sphum_conv_file_list', {})
        if not sphum_files:
            self.logger.warning("No specific humidity files specified")
            return
        
        # Get the model name
        model_name = data_source.model_name
        if not model_name:
            self.logger.warning("No model name specified for data source")
            return
        
        # Find the specific humidity file for this model
        sphum_file = None
        sphum_field_name = None
        for file_info in sphum_files.values():
            if file_info.get('exp_name') == model_name:
                sphum_file = os.path.join(file_info.get('location', ''), file_info.get('name', ''))
                sphum_field_name = file_info.get('sphum_field_name')
                break
        
        if not sphum_file or not sphum_field_name:
            self.logger.warning(f"No specific humidity file found for model {model_name}")
            return
        
        # Load the specific humidity data
        try:
            from eviz.lib.data.pipeline import DataReader
            reader = DataReader()
            sphum_source = reader.read_file(sphum_file)
            
            # Extract the specific humidity field
            if sphum_field_name in sphum_source.dataset:
                sphum = sphum_source.dataset[sphum_field_name]
                
                # Add the specific humidity to the data source's dataset
                data_source.dataset['specific_humidity'] = sphum
                self.logger.info(f"Added specific humidity from {sphum_file}")
                
                # Apply conversion to all variables that need it
                self._convert_with_sphum(data_source, sphum)
            else:
                self.logger.warning(f"Specific humidity field {sphum_field_name} not found in {sphum_file}")
        except Exception as e:
            self.logger.error(f"Error loading specific humidity: {e}")
    
    def _convert_with_sphum(self, data_source: DataSource, sphum) -> None:
        """Convert variables using specific humidity.
        
        Args:
            data_source: The data source to process
            sphum: The specific humidity data
        """
        # Get the list of variables that need conversion
        # This would typically be defined in the configuration
        variables_to_convert = []
        
        # Check if we have a list of variables to convert in the config
        if hasattr(self.config_manager, 'sphum_conv_vars'):
            variables_to_convert = self.config_manager.sphum_conv_vars
        
        # If not specified, try to infer from variable attributes
        if not variables_to_convert:
            for var_name, var in data_source.dataset.data_vars.items():
                # Check if the variable has units that suggest it needs conversion
                if 'units' in var.attrs:
                    units = var.attrs['units'].lower()
                    if 'kg/kg' in units or 'kg kg-1' in units:
                        variables_to_convert.append(var_name)
        
        # Apply conversion to each variable
        for var_name in variables_to_convert:
            if var_name in data_source.dataset:
                try:
                    # Convert from kg/kg to mol/mol (ppmv)
                    # Formula: ppmv = (kg/kg) * (MW_air / MW_gas) * 1e6
                    # where MW_air = 28.97 g/mol and MW_gas depends on the species
                    
                    # Get the molecular weight of the gas
                    mw_gas = self._get_molecular_weight(var_name)
                    if not mw_gas:
                        continue
                    
                    # MW of air is approximately 28.97 g/mol
                    mw_air = 28.97
                    
                    # Apply the conversion
                    var = data_source.dataset[var_name]
                    converted = var * (mw_air / mw_gas) * 1e6
                    
                    # Update the variable
                    data_source.dataset[var_name] = converted
                    
                    # Update the units attribute
                    data_source.dataset[var_name].attrs['units'] = 'ppmv'
                    data_source.dataset[var_name].attrs['original_units'] = var.attrs.get('units', 'kg/kg')
                    
                    self.logger.info(f"Converted {var_name} from kg/kg to ppmv")
                except Exception as e:
                    self.logger.error(f"Error converting {var_name}: {e}")
    
    def _get_molecular_weight(self, var_name: str) -> Optional[float]:
        """Get the molecular weight for a variable.
        
        Args:
            var_name: The variable name
            
        Returns:
            The molecular weight in g/mol, or None if not found
        """
        # Try to get from species database
        species_db = getattr(self.config_manager, 'species_db', {})
        
        # Check if we have a direct match
        if var_name in species_db:
            return species_db[var_name].get('molecular_weight')
        
        # Check for partial matches
        for species, info in species_db.items():
            if species in var_name:
                return info.get('molecular_weight')
        
        # Default molecular weights for common species
        default_weights = {
            'o3': 48.0,  # Ozone
            'co': 28.01,  # Carbon monoxide
            'co2': 44.01,  # Carbon dioxide
            'ch4': 16.04,  # Methane
            'no': 30.01,  # Nitric oxide
            'no2': 46.01,  # Nitrogen dioxide
            'h2o': 18.02,  # Water
            'so2': 64.07,  # Sulfur dioxide
        }
        
        # Check for matches in default weights
        for species, weight in default_weights.items():
            if species in var_name.lower():
                return weight
        
        self.logger.warning(f"No molecular weight found for {var_name}")
        return None
