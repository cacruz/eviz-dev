import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.interpolate import interp1d
import xarray as xr

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.data.utils import get_dst_attribute
from eviz.lib import constants as constants
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
        self.logger.debug("Start init")
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
        if not data_source.validate_data():
            self.logger.error("Data validation failed")
            return data_source
        
        # Process the dataset with the model_name from the data source
        data_source.dataset = self._process_dataset(data_source.dataset, data_source.model_name)
        
        # Extract metadata after processing
        # self._extract_metadata(data_source.dataset, data_source)
        
        # TODO: remove this
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
        if hasattr(self.config_manager,
                   'use_trop_height') and self.config_manager.use_trop_height:
            data_source = self._apply_tropopause_height(data_source)

        # Apply specific humidity conversion if configured
        if hasattr(self.config_manager,
                   'use_sphum_conv') and self.config_manager.use_sphum_conv:
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
                self._convert_radionuclide_units(data_source, species_name, specific_hum,
                                                 'mol mol-1')
        except Exception as e:
            self.logger.error(f"Error processing specific humidity: {e}")

        return data_source

    def _convert_radionuclide_units(self, data_source: DataSource, species_name: str,
                                    specific_hum: xr.DataArray,
                                    target_units: str) -> None:
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
                    self.config_manager.data_source.datasets[ds_index]['vars'][
                        species_name]):
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
            mw_air = constants.MW_AIR_G  # g/mole
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
            self.config_manager.data_source.datasets[ds_index]['ptr'][
                rn_arr.name] = rn_new

        except Exception as e:
            self.logger.error(f"Error converting {species_name}: {e}")

    def _process_dataset(self, dataset: xr.Dataset, model_name: str = None) -> Optional[xr.Dataset]:
        """Process a Xarray dataset.

        Args:
            dataset: The dataset to process

        Returns:
            The processed dataset
        """
        if dataset is None:
            return None

        dataset = self._standardize_coordinates(dataset, model_name)
        dataset = self._handle_missing_values(dataset)
        dataset = self._apply_unit_conversions(dataset)

        return dataset


    def _extract_metadata(self, dataset: xr.Dataset, data_source: DataSource) -> None:
        """Extract metadata from the dataset and store it in the data source.
        
        Args:
            dataset: The dataset to extract metadata from
            data_source: The data source to store metadata in
        """
        if dataset is None:
            return
                
        # Extract global attributes
        data_source.metadata["global_attrs"] = dict(dataset.attrs)
        
        # Extract dimension information
        data_source.metadata["dimensions"] = {dim: dataset.dims[dim] for dim in dataset.dims}
        
        # Extract variable information
        data_source.metadata["variables"] = {}
        for var_name, var in dataset.data_vars.items():
            data_source.metadata["variables"][var_name] = {
                "dims": var.dims,
                "attrs": dict(var.attrs),
                "dtype": str(var.dtype),
                "shape": var.shape
            }
            
            # Add some basic statistics for numerical variables
            try:
                if hasattr(var, 'dtype') and np.issubdtype(var.dtype, np.number):
                    data_source.metadata["variables"][var_name]["stats"] = {
                        "min": float(var.min().values),
                        "max": float(var.max().values),
                        "mean": float(var.mean().values),
                        "std": float(var.std().values)
                    }
            except Exception as e:
                self.logger.debug(f"Could not compute statistics for {var_name}: {e}")

    def _standardize_coordinates(self, dataset: xr.Dataset, model_name: str = None) -> xr.Dataset:
        """
        Standardize dimension names in the dataset.
        
        This method renames dimensions to standard names (lon, lat, lev, time)
        regardless of their original names in the source data.
        
        Args:
            dataset: xarray Dataset to rename dimensions in
            
        Returns:
            xarray Dataset with standardized dimension names
        """
        self.logger.debug(f"Standardizing coordinates for model name {model_name}")

        if model_name in ['wrf', 'lis']:
            # Skip renaming for these special models
            return dataset

        available_dims = list(dataset.dims)

        xc = self._get_model_dim_name('xc', available_dims, model_name)
        yc = self._get_model_dim_name('yc', available_dims, model_name)
        zc = self._get_model_dim_name('zc', available_dims, model_name)
        tc = self._get_model_dim_name('tc', available_dims, model_name)

        rename_dict = {}

        # Add mappings only for dimensions that exist and need renaming
        if xc and xc != 'lon' and xc in available_dims:
            rename_dict[xc] = 'lon'

        if yc and yc != 'lat' and yc in available_dims:
            rename_dict[yc] = 'lat'

        if zc and zc != 'lev' and zc in available_dims:
            rename_dict[zc] = 'lev'

        if tc and tc != 'time' and tc in available_dims:
            rename_dict[tc] = 'time'

        if rename_dict:
            self.logger.debug(f"Renaming dimensions: {rename_dict}")
            try:
                dataset = dataset.rename(rename_dict)
            except Exception as e:
                self.logger.error(f"Error renaming dimensions: {e}")

        return dataset

    def _get_model_dim_name(self, gridded_dim_name, available_dims=None, model_name=None, config_manager=None):
        """
        Get the model-specific dimension name for a gridded dimension.
        
        Args:
            gridded_dim_name (str): GriddedSource dimension name (e.g., 'xc', 'yc', 'zc', 'tc')
            available_dims (list, optional): List of available dimensions in the dataset
            model_name (str, optional): Name of the model
            config_manager (ConfigManager, optional): Configuration manager to use
            
        Returns:
            str or None: The model-specific dimension name if found, otherwise None
        """        
        cm = config_manager or self.config_manager
        
        if not cm:
            self.logger.warning("No config_manager available to get dimension mappings")
            return None
            
        if not hasattr(cm, 'meta_coords'):
            self.logger.warning("No meta_coords available in config_manager")
            return None
            
        meta_coords = cm.meta_coords

        if gridded_dim_name not in meta_coords:
            self.logger.warning(f"No mapping found for dimension '{gridded_dim_name}'")
            return None
        
        self.logger.debug(f"Looking for model '{model_name}' in meta_coords['{gridded_dim_name}']")
        self.logger.debug(f"Available models for {gridded_dim_name}: {list(meta_coords[gridded_dim_name].keys())}")
        
        if not model_name or model_name not in meta_coords[gridded_dim_name]:
            # Try to use a default model if available
            if 'gridded' in meta_coords[gridded_dim_name]:
                self.logger.debug(f"Using 'gridded' mapping for model '{model_name}' and dimension '{gridded_dim_name}'")
                model_name = 'gridded'
            else:
                self.logger.warning(f"No mapping found for model '{model_name}' and dimension '{gridded_dim_name}'")
                return None
        
        coords = meta_coords[gridded_dim_name][model_name]
            
        if isinstance(coords, list):
            for coord in coords:
                if available_dims and coord in available_dims:
                    return coord
            return coords[0] if coords else None
        
        elif isinstance(coords, dict):
            if 'dim' in coords:
                if available_dims:
                    if ',' in coords['dim']:
                        dim_candidates = coords['dim'].split(',')
                        for dim in dim_candidates:
                            if dim in available_dims:
                                return dim
                        return None
                    # Single dimension name
                    return coords['dim'] if coords['dim'] in available_dims else None
                return coords['dim']
            
            if 'coords' in coords:
                # For coordinate names
                return coords['coords']    
            return None
        
        elif isinstance(coords, str):
            # If coords is a string, handle comma-separated list of possible dimension names
            if ',' in coords:
                coord_candidates = coords.split(',')
                if available_dims:
                    for coord in coord_candidates:
                        if coord in available_dims:
                            return coord
                    # No matching dimension found
                    return None
                return coord_candidates[0]
            return coords
        
        self.logger.warning(f"Unexpected type for coords: {type(coords)}")
        return None

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

    def regrid(self,
        d1: xr.DataArray,
        d2: xr.DataArray,
        dims: tuple = None,
        method: str = "linear",
        extrapolate: bool = True
    ) -> xr.DataArray:
        """
        Regrid one of the two input arrays to match the other's grid, based on resolution.

        Args:
            d1: First data array.
            d2: Second data array.
            dims: Tuple of (dim1, dim2), the coordinate dimension names.
            method: Interpolation method ('linear', 'nearest').
            extrapolate: Whether to allow extrapolation.

        Returns:
            Regridded version of d2 that matches d1's grid.
        """
        if dims is None:

            common_dims = set(d1.dims).intersection(set(d2.dims))
            if len(common_dims) >= 2:
                dims = list(common_dims)[:2]
            else:
                dims = (d1.dims[0], d1.dims[1])
        
        dim1, dim2 = dims
        
        self.logger.debug(f"Regridding with dimensions {dim1}, {dim2}")
        self.logger.debug(f"d1 shape: {d1.shape}, dims: {d1.dims}")
        self.logger.debug(f"d2 shape: {d2.shape}, dims: {d2.dims}")
        
        if len(d1.dims) != len(d2.dims):
            self.logger.debug("Arrays have different number of dimensions")
            
            if len(d1.dims) < len(d2.dims):
                self.logger.debug(f"d1 has fewer dimensions ({len(d1.dims)}) than d2 ({len(d2.dims)})")
                
                extra_dims = [dim for dim in d2.dims if dim not in d1.dims]
                
                for dim in extra_dims:
                    if dim in d2.dims and d2[dim].size > 0:
                        d2 = d2.isel({dim: 0})
                
                d2 = d2.squeeze()
                
            else:
                self.logger.debug(f"d2 has fewer dimensions ({len(d2.dims)}) than d1 ({len(d1.dims)})")
                
                extra_dims = [dim for dim in d1.dims if dim not in d2.dims]
                
                for dim in extra_dims:
                    if dim in d1.dims and d1[dim].size > 0:
                        d1 = d1.isel({dim: 0})
                
                d1 = d1.squeeze()
        
        self.logger.debug("After dimension adjustment:")
        self.logger.debug(f"d1 shape: {d1.shape}, dims: {d1.dims}")
        self.logger.debug(f"d2 shape: {d2.shape}, dims: {d2.dims}")
        
        # Compute resolution for each dimension to determine which grid to use as target
        def mean_resolution(da, dim):
            coords = da.coords[dim].values
            return np.mean(np.abs(np.diff(coords)))
        
        try:
            d1_res = mean_resolution(d1, dim1) * mean_resolution(d1, dim2)
            d2_res = mean_resolution(d2, dim1) * mean_resolution(d2, dim2)
            
            # Regrid d2 to match d1's grid (we always want to keep d1's grid)
            d2_on_d1 = self._regrid(d2, d1, dims=(dim1, dim2), method=method, extrapolate=extrapolate)
            
            self.logger.debug(f"Regridded d2 shape: {d2_on_d1.shape}, dims: {d2_on_d1.dims}")
            
            if d1.shape != d2_on_d1.shape:
                self.logger.warning(f"Shape mismatch after regridding: d1 {d1.shape} vs d2_on_d1 {d2_on_d1.shape}")
                # Try to align the arrays
                d1, d2_on_d1 = xr.align(d1, d2_on_d1, join='inner')                
                self.logger.debug(f"After alignment: d1 {d1.shape}, d2_on_d1 {d2_on_d1.shape}")
            
            return d2_on_d1
            
        except Exception as e:
            self.logger.error(f"Error during regridding: {e}")
            self.logger.warning("Returning dummy array with zeros")
            return xr.zeros_like(d1)

    def _regrid(self,
        source: xr.DataArray,
        target: xr.DataArray,
        dims: tuple,
        method: str = "linear",
        extrapolate: bool = True
    ) -> xr.DataArray:
        """
        Regrid a data array to match the grid of another.

        Args:
            source: The data array to regrid.
            target: The target grid (another data array).
            dims: Tuple of (dim1, dim2) representing the coordinate names (e.g., ('lat', 'lon')).
            method: Interpolation method ('linear', 'nearest').
            extrapolate: Whether to extrapolate beyond source bounds.

        Returns:
            Regridded DataArray.
        """
        dim1, dim2 = dims
        
        if dim1 not in source.dims or dim2 not in source.dims:
            self.logger.error(f"Source array missing required dimensions {dim1} or {dim2}")
            return xr.zeros_like(target)
            
        if dim1 not in target.dims or dim2 not in target.dims:
            self.logger.error(f"Target array missing required dimensions {dim1} or {dim2}")
            return xr.zeros_like(target)
        
        new_coords = {
            dim1: target.coords[dim1].values,
            dim2: target.coords[dim2].values
        }
        
        for coord_name, coord_values in source.coords.items():
            if coord_name not in [dim1, dim2] and coord_name not in new_coords:
                new_coords[coord_name] = coord_values
        
        try:
            temp = source.interp({dim2: target.coords[dim2]}, method=method)
            result = temp.interp({dim1: target.coords[dim1]}, method=method)
            result = result.transpose(*target.dims)
            return result
            
        except Exception as e:
            self.logger.error(f"Error during interpolation: {e}")
            return xr.zeros_like(target)

    @staticmethod
    def _interp_1d(
        y_src: np.ndarray,
        x_src: np.ndarray,
        x_dest: np.ndarray,
        method: str = "linear",
        extrapolate: bool = True
    ) -> np.ndarray:
        fill = "extrapolate" if extrapolate else None
        return interp1d(x_src, y_src, kind=method, fill_value=fill, bounds_error=not extrapolate)(x_dest)

    def compute_difference(self,
        d1: xr.DataArray,
        d2: xr.DataArray,
        method: str = "difference"
    ) -> xr.DataArray:
        """
        Compute the difference between two data arrays.
        
        Args:
            d1: First data array
            d2: Second data array
            method: Method to compute difference ('difference', 'percd', 'percc', 'ratio')
            
        Returns:
            DataArray containing the computed difference
        """
        self.logger.debug(f"Computing difference using method: {method}")
        self.logger.debug(f"d1 shape: {d1.shape}, dims: {d1.dims}")
        self.logger.debug(f"d2 shape: {d2.shape}, dims: {d2.dims}")
        
        if d1.shape != d2.shape:
            self.logger.warning(f"Shape mismatch in compute_difference: d1 {d1.shape} vs d2 {d2.shape}")
            
            try:
                d1, d2 = xr.align(d1, d2, join='inner')
                self.logger.debug(f"After alignment: d1 {d1.shape}, d2 {d2.shape}")
            except Exception as e:
                self.logger.error(f"Error aligning arrays: {e}")
                return xr.zeros_like(d1)
        
        try:
            if method == "percd":
                # Percent difference
                return abs(d1 - d2) / ((d1 + d2) / 2.0) * 100
            elif method == "percc":
                # Percent change
                return ((d1 - d2) / d2) * 100
            elif method == "ratio":
                # Ratio
                return d1 / d2
            else:
                # Simple difference
                return d1 - d2
        except Exception as e:
            self.logger.error(f"Error computing difference: {e}")
            return xr.zeros_like(d1)
