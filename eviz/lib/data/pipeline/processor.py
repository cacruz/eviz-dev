import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
import xarray as xr

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.data.utils import get_dst_attribute
from eviz.lib import const as constants
from eviz.lib.data.sources import DataSource

logger = logging.getLogger(__name__)


@dataclass
class DataProcessor:
    """Data processing stage of the pipeline.
    
    This class handles all data processing operations including:
    - Basic data validation and standardization
    - Coordinate standardization
    - Missing value handling
    - Unit conversions
    - Overlays (tropopause height, specific humidity)
    - Data interpolation and regridding
    """
    config_manager: Optional['ConfigManager'] = None

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        """Post-initialization setup."""
        self.logger.info("Start init")
        self.tropp = None
        self.tropp_conversion = None
        self.trop_ok = False

    def process_data_source(self, data_source: DataSource) -> DataSource:
        """Process a data source.
        
        Args:
            data_source: The data source to process
            
        Returns:
            The processed data source
        """
        self.logger.info("Processing data source")
        
        if not data_source.validate_data():
            self.logger.error("Data validation failed")
            return data_source

        # Basic processing
        data_source.dataset = self._process_dataset(data_source.dataset)
        
        # Apply advanced processing if config_manager is available
        if self.config_manager:
            data_source = self._apply_geos_processing(data_source)
        
        return data_source
    
    def _apply_geos_processing(self, data_source: DataSource) -> DataSource:
        """Apply GEOS processing operations if requested.
        
        Args:
            data_source: The data source to process
            
        Returns:
            The processed data source
        """
        if not self.config_manager:
            return data_source

        # Apply tropopause height processing if configured
        if hasattr(self.config_manager, 'use_trop_height') and self.config_manager.use_trop_height:
            data_source = self._apply_tropopause_height(data_source)

        # Apply specific humidity conversion if configured
        if hasattr(self.config_manager, 'use_sphum_conv') and self.config_manager.use_sphum_conv:
            data_source = self._apply_sphum_conversion(data_source)

        return data_source

    def _apply_tropopause_height(self, data_source: DataSource):
        """Apply tropopause height processing.
        
        Args:
            data_source: The data source to process
            
        Returns:
            The processed data source
        """
        self.logger.info("Applying tropopause height overlay")
        data_source = None
        if not hasattr(self.config_manager, 'trop_height_file_list'):
            return data_source

        findex = getattr(self.config_manager, 'findex', 0)
        if findex not in self.config_manager.trop_height_file_list:
            return data_source

        try:
            # Get tropopause configuration
            trop_config = self.config_manager.trop_height_file_list[findex]
            exp_id = trop_config['exp_id']
            field_exp_id = self.config_manager.file_list[findex]['exp_id']
            
            if exp_id != field_exp_id:
                return data_source

            # Process tropopause data
            trop_filename = trop_config['filename']
            self.logger.debug(f"Processing {trop_filename}...")

            with xr.open_dataset(trop_filename) as f:
                trop_field = trop_config['trop_field_name']
                tropp = f.data_vars.get(trop_field)
                if tropp is None:
                    return data_source

                # Select first time step (TODO: make this configurable)
                tropp = tropp.isel(time=0)

                # Handle units conversion
                units = get_dst_attribute(tropp, 'units')
                conversion_factor = 1.0
                if units == 'Pa':
                    conversion_factor = 1 / 100.0
                elif units == 'hPa':
                    conversion_factor = 1.0

                data_source = tropp * conversion_factor

        except Exception as e:
            self.logger.error(f"Error processing tropopause height: {e}")

        return data_source

    def _apply_sphum_conversion(self, data_source: DataSource) -> DataSource:
        """Apply specific humidity conversion.
        
        Args:
            data_source: The data source to process
            
        Returns:
            The processed data source
        """
        radionuclides = ['Be10', 'Be10s', 'Be7', 'Be7s', 'Pb210', 'Rn222']
        to_convert = set(self.config_manager.to_plot).intersection(set(radionuclides))
        
        if not to_convert:
            return data_source

        ds_index = getattr(self.config_manager, 'ds_index', 0)
        ds_meta = getattr(self.config_manager, 'data_source', {})

        if not ds_meta.get('sphum_conv_meta'):
            return data_source

        try:
            # Get specific humidity data
            sphum_meta = ds_meta['sphum_conv_meta'][ds_index]
            if sphum_meta['exp_name'] != ds_meta['exp_name']:
                return data_source

            sphum_filename = sphum_meta['filename']
            self.logger.debug(f"Processing {sphum_filename}...")

            with xr.open_dataset(sphum_filename) as f:
                sphum_field = sphum_meta['sphum_field_name']
                specific_hum = f.data_vars.get(sphum_field)
                if specific_hum is None:
                    specific_hum = 0.0  # assume dry air conditions
                else:
                    specific_hum = specific_hum.isel(time=0)

            # Convert each radionuclide
            for species_name in to_convert:
                self._convert_radionuclide_units(data_source, species_name, specific_hum, 'mol mol-1')

        except Exception as e:
            self.logger.error(f"Error processing specific humidity: {e}")
            
        return data_source

    def _convert_radionuclide_units(self, data_source: DataSource, species_name: str, 
                                  specific_hum: xr.DataArray, target_units: str) -> None:
        """Convert radionuclide units using specific humidity.
        
        Args:
            data_source: The data source containing the species data
            species_name: Name of the species to convert
            specific_hum: Specific humidity data
            target_units: Target units for conversion
        """
        try:
            ds_index = self.config_manager.data_source.get_ds_index()
            
            # Skip if already in target units
            if self.config_manager.data_source.data_unit_is_mol_per_mol(
                self.config_manager.data_source.datasets[ds_index]['vars'][species_name]):
                return

            self.logger.debug(f"Converting {species_name} units to {target_units}")

            # Get molecular weight
            if species_name not in self.config_manager.species_db:
                self.logger.error(f"Species {species_name} not found in species database")
                return
                
            mw_g = self.config_manager.species_db[species_name].get("MW_g")
            if not mw_g:
                self.logger.error(f"Molecular weight not found for species {species_name}")
                return

            # Perform conversion
            mw_air = constants.MW_AIR_g  # g/mole
            rn_arr = self.config_manager.datasets[ds_index]['vars'][species_name]
            
            # Convert using specific humidity
            data = (rn_arr / (1. - specific_hum)) * (mw_air / mw_g)

            # Create new DataArray with converted data
            rn_new = xr.DataArray(
                data, name=rn_arr.name, coords=rn_arr.coords, 
                dims=rn_arr.dims, attrs=rn_arr.attrs
            )
            rn_new.attrs["units"] = target_units

            # Update the dataset
            self.config_manager.data_source.datasets[ds_index]['ptr'][rn_arr.name] = rn_new

        except Exception as e:
            self.logger.error(f"Error converting {species_name}: {e}")

    def _process_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """Process a Xarray dataset.

        Args:
            dataset: The dataset to process

        Returns:
            The processed dataset
        """
        if dataset is None:
            return None

        dataset = self._standardize_coordinates(dataset)
        dataset = self._handle_missing_values(dataset)
        dataset = self._apply_unit_conversions(dataset)

        return dataset

    def _standardize_coordinates(self, dataset: xr.Dataset) -> xr.Dataset:
        """Standardize coordinate names and values.

        Args:
            dataset: The dataset to process

        Returns:
            The processed dataset
        """
        coord_mappings = {
            'latitude': 'lat',
            'longitude': 'lon',
            'level': 'lev',
            'height': 'lev',
            'depth': 'lev',
            'pressure': 'lev',
            'time_bnds': 'time_bounds',
            'lat_bnds': 'lat_bounds',
            'lon_bnds': 'lon_bounds',
        }

        rename_dict = {}
        for old_name, new_name in coord_mappings.items():
            if old_name in dataset.coords and new_name not in dataset.coords:
                rename_dict[old_name] = new_name

        if rename_dict:
            dataset = dataset.rename(rename_dict)
            self.logger.debug(f"Renamed coordinates: {rename_dict}")

        # Ensure latitude is in the range [-90, 90]
        if 'lat' in dataset.coords:
            lat_values = dataset.coords['lat'].values
            if np.any(lat_values > 90) or np.any(lat_values < -90):
                self.logger.warning("Latitude values outside the range [-90, 90]")
                lat_values = np.clip(lat_values, -90, 90)
                dataset = dataset.assign_coords(lat=lat_values)
                self.logger.debug("Normalized latitude values to the range [-90, 90]")

        # Ensure longitude is in the range [-180, 180] or [0, 360]
        if 'lon' in dataset.coords:
            lon_values = dataset.coords['lon'].values
            if np.any(lon_values > 360) or np.any(lon_values < -180):
                self.logger.warning(
                    "Longitude values outside the range [-180, 180] or [0, 360]")
                lon_values = ((lon_values + 180) % 360) - 180
                dataset = dataset.assign_coords(lon=lon_values)
                self.logger.debug("Normalized longitude values to the range [-180, 180]")

        return dataset

    def _handle_missing_values(self, dataset: xr.Dataset) -> xr.Dataset:
        """Handle missing values in the dataset.

        Args:
            dataset: The dataset to process

        Returns:
            The processed dataset
        """
        # Replace NaN values with the _FillValue attribute if available
        for var_name, var in dataset.data_vars.items():
            if '_FillValue' in var.attrs:
                fill_value = var.attrs['_FillValue']
                if np.isnan(fill_value):
                    continue  # Skip if the fill value is already NaN

                # Replace NaN values with the fill value
                var_data = var.values
                var_data[np.isnan(var_data)] = fill_value
                dataset[var_name] = xr.DataArray(var_data, dims=var.dims,
                                                 coords=var.coords, attrs=var.attrs)
                self.logger.debug(
                    f"Replaced NaN values with fill value {fill_value} for variable {var_name}")

        return dataset

    def _apply_unit_conversions(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply unit conversions to the dataset.

        Args:
            dataset: The dataset to process

        Returns:
            The processed dataset
        """
        # Apply common unit conversions
        for var_name, var in dataset.data_vars.items():
            if 'units' not in var.attrs:
                continue

            units = var.attrs['units'].lower()

            # Convert temperature from Kelvin to Celsius if needed
            if units == 'k' and var_name.lower() in ['temp', 'temperature',
                                                     'air_temperature']:
                var_data = var.values - 273.15
                dataset[var_name] = xr.DataArray(var_data, dims=var.dims,
                                                 coords=var.coords, attrs=var.attrs)
                dataset[var_name].attrs['units'] = 'C'
                self.logger.debug(
                    f"Converted temperature from Kelvin to Celsius for variable {var_name}")

            # Convert pressure from hPa to Pa if needed
            elif units == 'hpa' and var_name.lower() in ['pressure', 'air_pressure',
                                                         'surface_pressure']:
                var_data = var.values * 100
                dataset[var_name] = xr.DataArray(var_data, dims=var.dims,
                                                 coords=var.coords, attrs=var.attrs)
                dataset[var_name].attrs['units'] = 'Pa'
                self.logger.debug(
                    f"Converted pressure from hPa to Pa for variable {var_name}")

        return dataset

    def regrid(self, d1: xr.DataArray, d2: xr.DataArray,
               dim1: str, dim2: str) -> Tuple[xr.DataArray, xr.DataArray]:
        """Regrid two data arrays to a common grid.
        
        Args:
            data1: First data array
            data2: Second data array
            dim1_name: Name of first dimension
            dim2_name: Name of second dimension
            
        Returns:
            Tuple of regridded data arrays
        """
        da1_size = d1.size
        da2_size = d2.size
        if da1_size < da2_size:
            d2 = self._regrid(d2, d1, dim1, dim2, regrid_dims=(1, 0))
            d2 = self._regrid(d2, d1, dim1, dim2, regrid_dims=(0, 1))
        elif da1_size > da2_size:
            d1 = self._regrid(d1, d2, dim1, dim2, regrid_dims=(1, 0))
            d1 = self._regrid(d1, d2, dim1, dim2, regrid_dims=(0, 1))
        elif da1_size == da2_size:
            d1 = self._regrid(d1, d2, dim1, dim2, regrid_dims=(1, 0))
            d1 = self._regrid(d1, d2, dim1, dim2, regrid_dims=(0, 1))
            d2 = self._regrid(d2, d1, dim1, dim2, regrid_dims=(1, 0))
            d2 = self._regrid(d2, d1, dim1, dim2, regrid_dims=(0, 1))

        compute_diff = True
        if compute_diff:
            if self.config_manager.ax_opts['add_extra_field_type']:
                data_diff = self._compute_diff_type(d1, d2).squeeze()
            else:
                data_diff = (d1 - d2).squeeze()
            coords = data_diff.coords
            return data_diff, coords[dim1].values, coords[dim2].values
        else:
            return d1, d2
        
    def _regrid(self, ref_arr: xr.DataArray, target: xr.DataArray, 
                dim1_name: str, dim2_name: str, regrid_dims: Tuple[int, int]) -> xr.DataArray:
        """Regrid a data array to match a target grid along specified dimensions.
        
        Args:
            ref_arr: Array to regrid
            target: Target grid
            dim1_name: Name of first dimension
            dim2_name: Name of second dimension
            regrid_dims: Tuple indicating which dimensions to regrid
            
        Returns:
            Regridded data array
        """
        new_arr = ref_arr

        if regrid_dims[0]:
            new_arr = xr.apply_ufunc(
                self._interp, new_arr,
                input_core_dims=[[dim2_name]],
                output_core_dims=[[dim2_name]],
                exclude_dims={dim2_name},
                kwargs={'x_src': ref_arr[dim2_name],
                       'x_dest': target.coords[dim2_name].values,
                       'fill_value': "extrapolate"},
                dask='allowed', 
                vectorize=True
            )
            new_arr.coords[dim2_name] = target.coords[dim2_name]

        elif regrid_dims[1]:
            new_arr = xr.apply_ufunc(
                self._interp, new_arr,
                input_core_dims=[[dim1_name]],
                output_core_dims=[[dim1_name]],
                exclude_dims={dim1_name},
                kwargs={'x_src': ref_arr[dim1_name],
                       'x_dest': target.coords[dim1_name].values,
                       'fill_value': "extrapolate"},
                dask='allowed', 
                vectorize=True
            )
            new_arr.coords[dim1_name] = target.coords[dim1_name]

        return new_arr

    def _compute_diff_type(self, d1, d2):
        """ Compute difference between two fields based on specified type

        Difference is specified in ``app`` file. It can be a percent difference, a percent change
        or a ratio difference.

        Parameters:
            d1 (ndarray) : A 2D array of an ESM field
            d2 (ndarray) : A 2D array of an ESM field

        Returns:
            Difference of the two fields
        """
        field_diff = None
        if self.config_manager.extra_diff_plot == "percd":  # percent diff
            num = abs(d1 - d2)
            den = (d1 + d2) / 2.0
            field_diff = (num / den) * 100.
        elif self.config_manager.extra_diff_plot == "percc":  # percent change
            field_diff = d1 - d2
            field_diff = field_diff / d2
            field_diff = field_diff * 100
        elif self.config_manager.extra_diff_plot == "ratio":
            field_diff = d1 / d2

        return field_diff

    @staticmethod
    def _interp(y_src: np.ndarray, x_src: np.ndarray, x_dest: np.ndarray, **kwargs) -> np.ndarray:
        """Interpolate data to new coordinates.
        
        Args:
            y_src: Source data values
            x_src: Source coordinates
            x_dest: Target coordinates
            **kwargs: Additional arguments for interp1d
            
        Returns:
            Interpolated data
        """
        return interp1d(x_src, y_src, **kwargs)(x_dest)
