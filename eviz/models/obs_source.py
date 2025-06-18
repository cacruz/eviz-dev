from dataclasses import dataclass
import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from eviz.lib.autoviz.figure import Figure
from eviz.models.source_base import GenericSource
import eviz.lib.autoviz.utils as pu

warnings.filterwarnings("ignore")


@dataclass
class ObsSource(GenericSource):
    """
    The ObsSource class provides specialized functionality for handling 
    observational data, which may be in gridded or swath format.
    
    This class extends the GriddedSource implementation to work with 
    observational datasets that may have irregular grids, swath formats,
    or other specialized structures common in satellite and in-situ observations.
    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

    def get_data_extent(self, data_array: xr.DataArray) -> list:
        """
        Extract the geographical extent (bounding box) from an xarray DataArray.
        
        This method determines the geographical boundaries of observational data,
        which is particularly useful for swath data that covers specific regions.
        The extent is returned as [lon_min, lon_max, lat_min, lat_max].
        
        Args:
            data_array (xr.DataArray): The data array to extract extent from
            
        Returns:
            list: The geographical extent as [lon_min, lon_max, lat_min, lat_max]
        """
        # Default extent (global)
        default_extent = [-180, 180, -90, 90]
        
        if data_array is None:
            self.logger.warning("Cannot extract extent from None data_array")
            return default_extent
            
        try:
            # Try to get coordinate names
            xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
            yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'
            
            # Check if coordinates exist in the DataArray
            if xc_dim in data_array.coords and yc_dim in data_array.coords:
                # Get coordinate values
                lon_vals = data_array[xc_dim].values
                lat_vals = data_array[yc_dim].values
                
                # Calculate extent
                lon_min = np.nanmin(lon_vals)
                lon_max = np.nanmax(lon_vals)
                lat_min = np.nanmin(lat_vals)
                lat_max = np.nanmax(lat_vals)
                
                # Add a small buffer (5% of range) around the extent for better visualization
                lon_buffer = (lon_max - lon_min) * 0.05
                lat_buffer = (lat_max - lat_min) * 0.05
                
                extent = [
                    lon_min - lon_buffer,
                    lon_max + lon_buffer,
                    lat_min - lat_buffer,
                    lat_max + lat_buffer
                ]
                
                self.logger.debug(f"Extracted extent: {extent}")
                return extent
            
            # If standard coords not found, try alternative approaches
            
            # Check for bounds attributes
            for attr_name in ['bounds', 'spatial_bounds', 'geospatial_bounds']:
                if hasattr(data_array, attr_name):
                    bounds = getattr(data_array, attr_name)
                    if isinstance(bounds, list) and len(bounds) == 4:
                        self.logger.debug(f"Using bounds from {attr_name} attribute: {bounds}")
                        return bounds
            
            # Check for explicit min/max attributes
            lon_min = getattr(data_array, 'geospatial_lon_min', None)
            lon_max = getattr(data_array, 'geospatial_lon_max', None)
            lat_min = getattr(data_array, 'geospatial_lat_min', None)
            lat_max = getattr(data_array, 'geospatial_lat_max', None)
            
            if all(x is not None for x in [lon_min, lon_max, lat_min, lat_max]):
                extent = [lon_min, lon_max, lat_min, lat_max]
                self.logger.debug(f"Using extent from geospatial attributes: {extent}")
                return extent
            
            # For 2D coordinate arrays (common in swath data)
            for coord_name in data_array.coords:
                if 'lon' in coord_name.lower() and len(data_array[coord_name].shape) == 2:
                    lon_vals = data_array[coord_name].values
                    lon_min, lon_max = np.nanmin(lon_vals), np.nanmax(lon_vals)
                
                if 'lat' in coord_name.lower() and len(data_array[coord_name].shape) == 2:
                    lat_vals = data_array[coord_name].values
                    lat_min, lat_max = np.nanmin(lat_vals), np.nanmax(lat_vals)
            
            if all(var in locals() for var in ['lon_min', 'lon_max', 'lat_min', 'lat_max']):
                # Add buffer
                lon_buffer = (lon_max - lon_min) * 0.05
                lat_buffer = (lat_max - lat_min) * 0.05
                
                extent = [
                    lon_min - lon_buffer,
                    lon_max + lon_buffer,
                    lat_min - lat_buffer,
                    lat_max + lat_buffer
                ]
                self.logger.debug(f"Using extent from 2D coordinates: {extent}")
                return extent
                
        except Exception as e:
            self.logger.error(f"Error extracting extent: {e}")
        
        self.logger.warning("Could not determine extent, using default global extent")
        return default_extent
    
    def apply_extent_to_config(self, data_array: xr.DataArray, field_name: str = None):
        """
        Extract extent from data_array and apply it to the configuration.
        
        This method extracts the geographical extent from the data array and
        updates the configuration manager's ax_opts with the extent information.
        
        Args:
            data_array (xr.DataArray): The data array to extract extent from
            field_name (str, optional): Field name for logging purposes
        """
        extent = self.get_data_extent(data_array)
        
        # Update configuration with the extent
        self.config_manager.ax_opts['extent'] = extent
        
        # Also set central longitude and latitude for projections
        central_lon = (extent[0] + extent[1]) / 2
        central_lat = (extent[2] + extent[3]) / 2
        self.config_manager.ax_opts['central_lon'] = central_lon
        self.config_manager.ax_opts['central_lat'] = central_lat
        
        if field_name:
            self.logger.info(f"Applied extent {extent} to field {field_name}")
        else:
            self.logger.info(f"Applied extent {extent} to configuration")
        
        return extent
    
    def _get_xy(self, data_array, level=None, time_lev=None):
        """
        Extract XY slice from a DataArray and apply extent information.
        
        This method overrides the parent class method to add automatic
        extent detection for observational data.
        
        Args:
            data_array: The data array to process
            level: Vertical level to extract
            time_lev: Time level to extract
            
        Returns:
            xr.DataArray: The processed 2D data array
        """
        # Call the parent method to get the 2D slice
        data2d = super()._get_xy(data_array, level, time_lev)
        
        if data2d is not None:
            # Extract and apply extent if not already set
            if 'extent' not in self.config_manager.ax_opts:
                self.apply_extent_to_config(data2d, data_array.name if hasattr(data_array, 'name') else None)
        
        return data2d
    
    def _process_xy_plot(self, data_array, field_name, file_index, plot_type, figure):
        """Process an XY plot."""
        levels = self.config_manager.get_levels(field_name, plot_type + 'plot')
        do_zsum = self.config_manager.ax_opts.get('zsum', False)

        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        num_times = data_array[tc_dim].size if tc_dim in data_array.dims else 1
        time_levels = range(num_times) if time_level_config == 'all' else [time_level_config]

        if not levels and not do_zsum:
            return

        self._process_level_plot(data_array, field_name, file_index, plot_type, figure, time_levels, levels)

    def _process_level_plot(self, data_array, field_name, file_index, plot_type, figure, time_levels, levels):
        """Process plots for specific vertical levels."""
        self.logger.debug("Processing XY level plots")
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        has_vertical_dim = zc_dim and zc_dim in data_array.dims

        for level_val in levels.keys():
            self.config_manager.level = level_val
            for t in time_levels:
                if tc_dim in data_array.dims:
                    data_at_time = data_array.isel({tc_dim: t})
                else:
                    data_at_time = data_array.squeeze()  # Assume single time if no time dim
                
                # Check if data at this time level is all NaN
                if np.isnan(data_at_time).all():
                    self.logger.warning(f"Skipping time level {t} for {field_name} - all values are NaN")
                    continue
                    
                self._set_time_config(t, data_at_time)
                # Create a new figure for each level to avoid reusing axes
                figure = Figure.create_eviz_figure(self.config_manager, plot_type)
                self.config_manager.ax_opts = figure.init_ax_opts(field_name)

                # If the data doesn't have a vertical dimension, we can't select a level
                # In this case, we'll just use the data as is
                if not has_vertical_dim:
                    field_to_plot = self._get_field_to_plot(data_at_time, field_name, file_index, plot_type, figure, t)
                else:
                    field_to_plot = self._get_field_to_plot(data_at_time, field_name, file_index, plot_type, figure, t, level=level_val)

                if field_to_plot and not np.isnan(field_to_plot[0]).all():
                    plot_result = self.create_plot(field_name, field_to_plot)                    
                    pu.print_map(self.config_manager, plot_type, self.config_manager.findex, plot_result, level=level_val)
                else:
                    self.logger.warning(f"Skipping plot for time level {t} - no valid data after processing")

    def _process_xt_plot(self, data_array, field_name, file_index, plot_type, figure):
        """Process an XT plot."""
        self.config_manager.level = None
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        if tc_dim in data_array.dims:
            num_times = data_array[tc_dim].size
            time_levels = range(num_times) if time_level_config == 'all' else [time_level_config]
        else:
            time_levels = [0]

        field_to_plot = self._get_field_to_plot(data_array, field_name, file_index, plot_type, figure, time_level=time_level_config)
        
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex, plot_result)

    def _process_box_plot(self, data_array, field_name, file_index, plot_type, figure):
        """Process plots for specific time or vertical levels."""
        self.config_manager.level = None
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        if tc_dim in data_array.dims:
            num_times = data_array[tc_dim].size
            time_levels = range(num_times) if time_level_config == 'all' else [time_level_config]
        else:
            time_levels = [0]

        field_to_plot = self._get_field_to_plot(data_array, field_name, file_index, plot_type, figure, time_level=time_level_config)
        
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex, plot_result)

    def _process_line_plot(self, data_array, field_name, file_index, plot_type, figure):
        """Process a LINE plot."""
        self.config_manager.level = None
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        if tc_dim in data_array.dims:
            num_times = data_array[tc_dim].size
            time_levels = range(num_times) if time_level_config == 'all' else [time_level_config]
        else:
            time_levels = [0]

        field_to_plot = self._get_field_to_plot(data_array, field_name, file_index, plot_type, figure, time_level=time_level_config)
        
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex, plot_result)

    def _get_xy(self, data_array, level=None, time_lev=None):
        """
        Extract XY slice from a DataArray and apply extent information.
        """
        # Check if data is all NaN
        if np.isnan(data_array).all():
            self.logger.warning(f"All values are NaN for {data_array.name if hasattr(data_array, 'name') else 'unnamed field'}")
            return None
            
        # Call the parent method to get the 2D slice
        data2d = super()._get_xy(data_array, level, time_lev)
        
        if data2d is not None:
            # Check if the result is all NaN
            if np.isnan(data2d).all():
                self.logger.warning("All values are NaN in the extracted 2D slice")
                return None
                
            # Extract and apply extent if not already set
            if 'extent' not in self.config_manager.ax_opts:
                self.apply_extent_to_config(data2d, data_array.name if hasattr(data_array, 'name') else None)
        
        return data2d

    def _get_xt(self, data_array, time_lev=None):
        """
        Extract XT data.
        """
        # Call the parent method to get the data
        data2d = super()._get_xt(data_array, time_lev)
                
        return data2d


