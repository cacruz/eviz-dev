from dataclasses import dataclass
import logging
import warnings
import matplotlib
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
        default_extent = [-180, 180, -90, 90]
        
        if data_array is None:
            self.logger.warning("Cannot extract extent from None data_array")
            return default_extent
            
        try:
            xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
            yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'
            
            # Check if coordinates exist in the DataArray
            if xc_dim in data_array.coords and yc_dim in data_array.coords:
                lon_vals = data_array[xc_dim].values
                lat_vals = data_array[yc_dim].values
                
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
            for attr_name in ['bounds', 'spatial_bounds', 'geospatial_bounds']:
                if hasattr(data_array, attr_name):
                    bounds = getattr(data_array, attr_name)
                    if isinstance(bounds, list) and len(bounds) == 4:
                        self.logger.debug(f"Using bounds from {attr_name} attribute: {bounds}")
                        return bounds
            
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
        
        self.config_manager.ax_opts['extent'] = extent
        
        central_lon = (extent[0] + extent[1]) / 2
        central_lat = (extent[2] + extent[3]) / 2
        self.config_manager.ax_opts['central_lon'] = central_lon
        self.config_manager.ax_opts['central_lat'] = central_lat
                
        return extent
    
    def _process_xy_plot(self, 
                         data_array, 
                         field_name, 
                         file_index, 
                         plot_type, 
                         figure):
        """Process an XY plot."""
        levels = self.config_manager.get_levels(field_name, plot_type + 'plot')
        do_zsum = self.config_manager.ax_opts.get('zsum', False)

        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        num_times = data_array[tc_dim].size if tc_dim in data_array.dims else 1
        time_levels = range(num_times) if time_level_config == 'all' else [time_level_config]

        if not levels and not do_zsum:
            return

        self._process_level_plot(data_array, 
                                 field_name, 
                                 file_index, 
                                 plot_type, 
                                 figure, 
                                 time_levels, 
                                 levels)

    def _process_level_plot(self, 
                            data_array, 
                            field_name, 
                            file_index, 
                            plot_type, 
                            figure, 
                            time_levels, 
                            levels):
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
                    field_to_plot = self._prepare_field_to_plot(data_at_time, 
                                                                field_name, 
                                                                file_index, 
                                                                plot_type, 
                                                                figure, 
                                                                t)
                else:
                    field_to_plot = self._prepare_field_to_plot(data_at_time, 
                                                                field_name, 
                                                                file_index, 
                                                                plot_type, 
                                                                figure, 
                                                                t, 
                                                                level=level_val)

                if field_to_plot and not np.isnan(field_to_plot[0]).all():
                    plot_result = self.create_plot(field_name, field_to_plot)                    
                    pu.print_map(self.config_manager, 
                                 plot_type, 
                                 self.config_manager.findex, 
                                 plot_result, 
                                 level=level_val)
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

        field_to_plot = self._prepare_field_to_plot(data_array, 
                                                    field_name, 
                                                    file_index, 
                                                    plot_type, 
                                                    figure, 
                                                    time_level=time_level_config)
        
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, 
                         plot_type, 
                         self.config_manager.findex, 
                         plot_result)

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

        field_to_plot = self._prepare_field_to_plot(data_array, 
                                                    field_name, 
                                                    file_index, 
                                                    plot_type, 
                                                    figure, 
                                                    time_level=time_level_config)
        
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, 
                         plot_type, 
                         self.config_manager.findex, 
                         plot_result)

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

        field_to_plot = self._prepare_field_to_plot(data_array, 
                                                    field_name, 
                                                    file_index, 
                                                    plot_type, 
                                                    figure, 
                                                    time_level=time_level_config)
        
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, 
                         plot_type, 
                         self.config_manager.findex, 
                         plot_result)

    def _extract_xy_data(self, data_array, level=None, time_lev=None):
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
        if np.isnan(data_array).all():
            self.logger.warning(f"All values are NaN for {data_array.name if hasattr(data_array, 'name') else 'unnamed field'}")
            return None
            
        data2d = super()._extract_xy_data(data_array, level, time_lev)
        
        if data2d is not None:
            if np.isnan(data2d).all():
                self.logger.warning("All values are NaN in the extracted 2D slice")
                return None
                
            if 'extent' not in self.config_manager.ax_opts:
                self.apply_extent_to_config(data2d, data_array.name if hasattr(data_array, 'name') else None)
        
        return data2d

    def _extract_xt_data(self, data_array, time_lev=None):
        """
        Extract XT data.
        """
        data2d = super()._extract_xt_data(data_array, time_lev)
                
        return data2d

    def _process_xy_side_by_side_plots(self, 
                                       current_field_index, 
                                       field_name1, 
                                       field_name2, 
                                       plot_type, 
                                       sdat1_dataset, 
                                       sdat2_dataset):
        """Process side-by-side comparison plots for xy or polar plot types."""
        num_plots = len(self.config_manager.compare_exp_ids)
        nrows = 1
        ncols = num_plots

        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            return

        for level_val in levels:
            figure = Figure.create_eviz_figure(self.config_manager, plot_type,
                                            nrows=nrows, ncols=ncols)
            figure.set_axes()
            self.config_manager.level = level_val

            self.config_manager.is_regional = False  # Observational data is typically regional
            
            extent1 = self.get_data_extent(sdat1_dataset[field_name1])
            extent2 = self.get_data_extent(sdat2_dataset[field_name2])
            
            combined_extent = [
                min(extent1[0], extent2[0]), 
                max(extent1[1], extent2[1]),
                min(extent1[2], extent2[2]),
                max(extent1[3], extent2[3]) 
            ]
            
            self.config_manager.ax_opts['extent'] = combined_extent
            self.config_manager.ax_opts['central_lon'] = (combined_extent[0] + combined_extent[1]) / 2
            self.config_manager.ax_opts['central_lat'] = (combined_extent[2] + combined_extent[3]) / 2

            self._create_xy_side_by_side_plot(current_field_index,
                                              field_name1, 
                                              field_name2, 
                                              figure,
                                              plot_type, 
                                              sdat1_dataset, 
                                              sdat2_dataset,
                                              level_val)

            pu.print_map(self.config_manager, 
                         plot_type, 
                         self.config_manager.findex, 
                         self.plot_result)

        self.data2d_list = []

    def _create_xy_side_by_side_plot(self, 
                                     current_field_index,
                                     field_name1, 
                                     field_name2, 
                                     figure,
                                     plot_type, 
                                     sdat1_dataset, 
                                     sdat2_dataset, 
                                     level=None):
        """
        Create a side-by-side comparison plot for the given data with a shared colorbar.
        
        This method creates multiple plots side by side for comparison. The colorbar
        handling is delegated to the plotter in the library code.
        """
        num_plots = len(self.config_manager.compare_exp_ids)
        self.comparison_plot = False
        # self.data2d_list = []  # Reset the list

        # Plot first dataset (from a_list)
        if self.config_manager.a_list:
            self._process_single_side_by_side_plot(self.config_manager.a_list[0],
                                                   current_field_index,
                                                   field_name1, 
                                                   figure, 
                                                   0,
                                                   sdat1_dataset[field_name1], 
                                                   plot_type,
                                                   level=level)

        # Plot remaining datasets (from b_list)
        for i, file_idx in enumerate(self.config_manager.b_list, start=1):
            if i < num_plots:  # Only plot if we have a corresponding axis
                self.logger.debug(f"Plotting dataset {i} to axis {i}")               
                map_params = self.config_manager.map_params.get(file_idx)
                filename = map_params.get('filename')
                data_source = self.config_manager.pipeline.get_data_source(filename)
                dataset = data_source.dataset
                
                self._process_single_side_by_side_plot(file_idx,
                                                       current_field_index,
                                                       field_name2, 
                                                       figure, 
                                                       i,
                                                       dataset[field_name2], 
                                                       plot_type,
                                                       level=level)

    def _process_other_side_by_side_plots(self, 
                                          current_field_index,
                                          field_name1, 
                                          field_name2, 
                                          plot_type, 
                                          sdat1_dataset,
                                          sdat2_dataset):
        """Process side-by-side comparison plots for xy or polar plot types."""
        # self.data2d_list = []
        num_plots = len(self.config_manager.compare_exp_ids)
        nrows = 1
        ncols = num_plots

        use_overlay = self.config_manager.should_overlay_plots(field_name1, plot_type[:2])
        if use_overlay:
            ncols = 1  # Use a single plot for overlay


        figure = Figure.create_eviz_figure(self.config_manager, plot_type,
                                            nrows=nrows, ncols=ncols)
        figure.set_axes()
        self.config_manager.level = None

        # Plot first dataset (from a_list)
        if self.config_manager.a_list:
            self._process_single_side_by_side_plot(self.config_manager.a_list[0],
                                                   current_field_index,
                                                   field_name1, 
                                                   figure, 
                                                   0,
                                                   sdat1_dataset[field_name1], 
                                                   plot_type)

        # Plot remaining datasets (from b_list)
        for i, file_idx in enumerate(self.config_manager.b_list, start=1):
            if i < num_plots:  # Only plot if we have a corresponding axis
                map_params = self.config_manager.map_params.get(file_idx)
                if not map_params:
                    continue
                    
                filename = map_params.get('filename')
                if not filename:
                    continue
                data_source = self.config_manager.pipeline.get_data_source(filename)
                if not data_source or not hasattr(data_source, 'dataset') or data_source.dataset is None:
                    continue
                    
                dataset = data_source.dataset
                
                # Use the correct axis index based on overlay mode
                axis_index = 0 if use_overlay else i
                
                self._process_single_side_by_side_plot(file_idx,
                                                       current_field_index,
                                                       field_name2, 
                                                       figure, 
                                                       axis_index,
                                                       dataset[field_name2], 
                                                       plot_type)

        pu.print_map(self.config_manager, 
                     plot_type, 
                     self.config_manager.findex, 
                     self.plot_result)

    def _process_single_side_by_side_plot(self, 
                                          file_index, 
                                          current_field_index,
                                          field_name, 
                                          figure, 
                                          ax_index, 
                                          data_array, 
                                          plot_type,
                                          level=None):
        """Process a single plot for side-by-side comparison."""
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        time_level_config = self.config_manager.ax_opts.get('time_lev', -1)

        # Register the plot type
        self.register_plot_type(field_name, plot_type)

        # Track which dataset we're currently plotting and how many total
        if self.config_manager.should_overlay_plots(field_name, plot_type[:2]):
            if file_index in self.config_manager.a_list:
                dataset_index = self.config_manager.a_list.index(file_index)
            elif file_index in self.config_manager.b_list:
                dataset_index = len(self.config_manager.a_list) + self.config_manager.b_list.index(file_index)
            else:
                dataset_index = 0
                
            self.config_manager.current_dataset_index = dataset_index
            self.config_manager.total_datasets = len(self.config_manager.a_list) + len(self.config_manager.b_list)
        
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        
        # Apply extent information for this specific dataset
        _ = self.get_data_extent(data_array)
        self.apply_extent_to_config(data_array)
        
        field_to_plot = self._prepare_field_to_plot(data_array, 
                                                    field_name,
                                                    file_index,
                                                    plot_type, 
                                                    figure,
                                                    time_level=time_level_config,
                                                    level=level)

        if field_to_plot and field_to_plot[0] is not None:
            self.data2d_list.append(field_to_plot[0])

        if field_to_plot:
            self.plot_result = self.create_plot(field_name, field_to_plot)

    def _process_pearson_plot(self, data_array, field_name, file_index, plot_type, figure):
        """Process a Pearson correlation plot for observational data."""
        self.config_manager.level = None
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        pearson_settings = {}
        if hasattr(self.config_manager, 'input_config') and hasattr(self.config_manager.input_config, '_pearsonplot'):
            pearson_settings = self.config_manager.input_config._pearsonplot
            self.logger.debug(f"Found pearsonplot settings: {pearson_settings}")
        else:
            self.logger.debug("No pearsonplot settings found in input_config")
        
        # Determine correlation type (time or space)
        do_time_corr = pearson_settings.get('time_corr', True)
        do_space_corr = pearson_settings.get('space_corr', False)
        
        if do_time_corr:
            time_level_config = 'all'
            self.logger.debug("Using all time points for time correlation")
        
        # Get the fields to correlate
        fields_str = pearson_settings.get('fields', '')
        corr_fields = [f.strip() for f in fields_str.split(',') if f.strip()]
        self.logger.debug(f"Correlation fields from config: {corr_fields}")
        
        # Find the reference field (the other field in the correlation pair)
        reference_field = None
        if len(corr_fields) == 2:
            if corr_fields[0] == field_name:
                reference_field = corr_fields[1]
            elif corr_fields[1] == field_name:
                reference_field = corr_fields[0]
            self.logger.debug(f"Selected reference field: {reference_field}")
        
        # If no reference field found in global settings, try to get from ax_opts
        if not reference_field:
            reference_field = self.config_manager.ax_opts.get('reference_field')
            self.logger.debug(f"Using reference field from ax_opts: {reference_field}")
        
        if not reference_field:
            self.logger.error("No reference field specified for Pearson correlation plot")
            return
        
        reference_data = None
        if hasattr(self.config_manager, 'pipeline'):
            all_data_sources = self.config_manager.pipeline.get_all_data_sources()
            for source_name, data_source in all_data_sources.items():
                if hasattr(data_source, 'dataset') and reference_field in data_source.dataset:
                    reference_data = data_source.dataset[reference_field]
                    self.logger.debug(f"Found reference field '{reference_field}' in data source '{source_name}'")
                    break
            
            if reference_data is None:
                try:
                    reference_data = self.config_manager.pipeline.get_variable(reference_field)
                    if reference_data is not None:
                        self.logger.debug(f"Found reference field '{reference_field}' using get_variable")
                except (AttributeError, KeyError) as e:
                    self.logger.debug(f"Could not get reference field using get_variable: {e}")
                    
            if reference_data is None:
                try:
                    all_vars = self.config_manager.pipeline.get_all_variables()
                    if reference_field in all_vars:
                        reference_data = all_vars[reference_field]
                        self.logger.debug(f"Found reference field '{reference_field}' in all variables")
                except (AttributeError, KeyError) as e:
                    self.logger.debug(f"Could not get reference field from all variables: {e}")
        
        if reference_data is None:
            self.logger.error(f"Reference field '{reference_field}' not found in any data source")
            return
        
        # For observational data, extract and apply extent information
        extent = self.get_data_extent(data_array)
        self.config_manager.ax_opts['extent'] = extent
        self.config_manager.ax_opts['central_lon'] = (extent[0] + extent[1]) / 2
        self.config_manager.ax_opts['central_lat'] = (extent[2] + extent[3]) / 2
        
        # Prepare both datasets
        field_to_plot = self._prepare_field_to_plot(data_array, 
                                                field_name, 
                                                file_index, 
                                                plot_type, 
                                                figure,
                                                time_level=time_level_config)
                                                
        reference_to_plot = self._prepare_field_to_plot(reference_data, 
                                                    reference_field, 
                                                    file_index, 
                                                    plot_type, 
                                                    figure,
                                                    time_level=time_level_config)
                                                    
        if field_to_plot and reference_to_plot:
            data_tuple = (field_to_plot[0], reference_to_plot[0])
            
            correlation_to_plot = (data_tuple,) + field_to_plot[1:]
            
            plot_result = self.create_plot(field_name, correlation_to_plot)
            if isinstance(plot_result, tuple) and len(plot_result) >= 1:
                fig = plot_result[0]  # Extract the figure from the tuple
                pu.print_map(self.config_manager, 
                            plot_type, 
                            self.config_manager.findex, 
                            fig)  # Pass just the figure
            else:
                # If it's not a tuple, pass it directly
                pu.print_map(self.config_manager, 
                            plot_type, 
                            self.config_manager.findex, 
                            plot_result)