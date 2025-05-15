# File: eviz/models/esm/generic.py
from dataclasses import dataclass
import logging
import warnings
import sys
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from collections.abc import Iterable

from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.models.root import Root
from eviz.lib.data.data_utils import apply_conversion, apply_mean, apply_zsum
import eviz.lib.autoviz.plot_utils as pu
from eviz.lib.autoviz.figure import Figure
import multiprocessing

warnings.filterwarnings("ignore")


@dataclass
class Generic(Root):
    """ The generic class contains definitions for handling generic ESM data, that is 2D, 3D, and 4D
     field data. This is typically not the case for observational data which may be unstructured and very
     non-standard in its internal arrangement.
     Specific model functionality should be overridden in subclasses.
    """
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

    def add_data_source(self, file_path, data_source):
        """
        Add a data source to the model.
        
        This method is required by AbstractRoot but is now a no-op since data sources
        are managed by the pipeline. It's kept for backward compatibility.
        
        Args:
            file_path: Path to the data file
            data_source: The data source to add
        """
        self.logger.warning("add_data_source is deprecated. Data sources are now managed by the pipeline.")
        # No need to do anything, as data sources are managed by the pipeline

    def get_data_source(self, file_path):
        """
        Get a data source from the model.
        
        This method is required by AbstractRoot but now delegates to the pipeline.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            The data source for the file path, or None if not found
        """
        return self.config_manager.pipeline.get_data_source(file_path)

    def load_data_sources(self):
        """
        Load data sources for the model.
        
        This method is required by AbstractRoot but is now a no-op since data sources
        are loaded by the ConfigurationAdapter. It's kept for backward compatibility.
        """
        self.logger.warning("load_data_sources is deprecated. Data sources are now loaded by the ConfigurationAdapter.")
        # No need to do anything, as data sources are loaded by the ConfigurationAdapter


    # Removed _load_source_data and _integrate_datasets methods
    # Data loading and integration are now handled by the DataPipeline
    # and ConfigurationAdapter before the model's plot() method is called.


    # SIMPLE PLOTS METHODS (no SPECS file)
    #--------------------------------------------------------------------------
    def _simple_plots(self, plotter):
        """Generate simple plots."""
        self.logger.info("Generating simple plots")
        map_params = self.config_manager.map_params
        field_num = 0
        # self.config_manager.findex = 0 # findex is set in _single_plots loop

        # Access data sources via the pipeline
        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for simple plotting.")
            return

        for i, params in map_params.items():
            field_name = params.get('field')
            if not field_name:
                continue

            filename = params.get('filename')
            data_source = self.config_manager.pipeline.get_data_source(filename)

            if not data_source or not hasattr(data_source, 'dataset') or data_source.dataset is None:
                self.logger.warning(f"No data source or dataset found in pipeline for {filename}")
                continue

            if field_name not in data_source.dataset:
                self.logger.warning(f"Field {field_name} not found in dataset for {filename}")
                continue

            # Update config_manager state variables before plotting
            # Use the index from map_params as findex
            self.config_manager.findex = i
            self.config_manager.pindex = field_num
            self.config_manager.axindex = 0

            field_data_array = data_source.dataset[field_name]

            for pt in params.get('to_plot', ['xy']): # Default to 'xy' if not specified
                self.logger.info(f"Plotting {field_name}, {pt} plot")
                # Pass the DataArray directly to the helper
                field_to_plot = self._get_field_for_simple_plot(field_data_array, field_name, pt)
                if field_to_plot:
                    # The simple_plot method in SimplePlotter expects a tuple
                    # (data2d, dim1, dim2, field_name, plot_type)
                    plotter.simple_plot(self.config_manager, (*field_to_plot, pt))
            field_num += 1


    # Simple plots do not use configurations in SPECS file
    def _get_field_for_simple_plot(self, data_array: xr.DataArray, field_name: str, plot_type: str) -> tuple:
        """Prepare data for simple plots."""
        if data_array is None:
            return None

        data2d = None
        dim1_name, dim2_name = None, None

        if 'xy' in plot_type:
            dim1_name = self.config_manager.get_model_dim_name('xc')
            dim2_name = self.config_manager.get_model_dim_name('yc')
            data2d = self._get_xy_simple(data_array, 0) # Assuming time_lev=0 for simple plots
        elif 'yz' in plot_type:
            dim1_name = self.config_manager.get_model_dim_name('yc')
            dim2_name = self.config_manager.get_model_dim_name('zc')
            data2d = self._get_yz_simple(data_array)
        elif 'sc' in plot_type:
             # Assuming scatter plots use lat/lon
            dim1_name = self.config_manager.get_model_dim_name('xc')
            dim2_name = self.config_manager.get_model_dim_name('yc')
            data2d = data_array.squeeze() # Scatter plots usually don't need slicing
        elif 'graph' in plot_type:
             # Assuming graph data is the DataArray itself
             data2d = data_array
             dim1_name, dim2_name = None, None # Graph plots don't have standard dims
        else:
            self.logger.error(f'Plot type [{plot_type}] error: Either specify in SPECS file or create plot type.')
            return None

        if data2d is None:
            return None

        # Get coordinate DataArrays if dimension names were found
        dim1_coords = data2d[dim1_name] if dim1_name and dim1_name in data2d.coords else None
        dim2_coords = data2d[dim2_name] if dim2_name and dim2_name in data2d.coords else None

        return data2d, dim1_coords, dim2_coords, field_name

    def _get_xy_simple(self, data_array: xr.DataArray, time_level: int) -> xr.DataArray:
        """ Extract XY slice from N-dim data field for simple plot"""
        if data_array is None:
            return None
        data2d = data_array.squeeze()
        # Hackish - select first time and level if they exist
        if self.config_manager.get_model_dim_name('tc') in data2d.dims:
             if data2d[self.config_manager.get_model_dim_name('tc')].size > time_level:
                data2d = data2d.isel({self.config_manager.get_model_dim_name('tc'): time_level})
             else:
                 self.logger.warning(f"Time level {time_level} out of bounds for {data_array.name}")
                 if data2d[self.config_manager.get_model_dim_name('tc')].size > 0:
                     data2d = data2d.isel({self.config_manager.get_model_dim_name('tc'): 0})
                 else:
                     self.logger.warning(f"No time dimension found for {data_array.name}")

        if self.config_manager.get_model_dim_name('zc') in data2d.dims:
             if data2d[self.config_manager.get_model_dim_name('zc')].size > 0:
                data2d = data2d.isel({self.config_manager.get_model_dim_name('zc'): 0})
             else:
                 self.logger.warning(f"No vertical dimension found for {data_array.name}")

        return data2d

    def _get_yz_simple(self, data_array: xr.DataArray) -> xr.DataArray:
        """ Create YZ slice from N-dim data field for simple plot"""
        if data_array is None:
            return None
        data2d = data_array.squeeze()
        # Hackish - select first time if it exists
        if self.config_manager.get_model_dim_name('tc') in data2d.dims:
             if data2d[self.config_manager.get_model_dim_name('tc')].size > 0:
                data2d = data2d.isel({self.config_manager.get_model_dim_name('tc'): 0})
             else:
                 self.logger.warning(f"No time dimension found for {data_array.name}")

        # Compute zonal mean if longitude dimension exists
        xc_dim = self.config_manager.get_model_dim_name('xc')
        if xc_dim and xc_dim in data2d.dims:
            data2d = data2d.mean(dim=xc_dim)
        else:
            self.logger.warning(f"Could not find longitude dimension '{xc_dim}' for zonal mean in {data_array.name}")

        return data2d

    # Removed _get_model_dim_name method - use config_manager.get_model_dim_name


    # SINGLE PLOTS METHODS (using SPECS file)
    #--------------------------------------------------------------------------
    def _single_plots(self, plotter):
        """Generate single plots for each source and field according to configuration."""
        self.logger.info("Generating single plots")

        # Access data sources via the pipeline
        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for single plotting.")
            return

        # Iterate through map_params to generate plots
        # Access map_params via config_manager
        for idx, params in self.config_manager.map_params.items():
            field_name = params.get('field')
            if not field_name:
                continue

            filename = params.get('filename')
            data_source = self.config_manager.pipeline.get_data_source(filename)

            if not data_source or not hasattr(data_source, 'dataset') or data_source.dataset is None:
                self.logger.warning(f"No data source or dataset found in pipeline for {filename}")
                continue

            if field_name not in data_source.dataset:
                self.logger.warning(f"Field {field_name} not found in dataset for {filename}")
                continue

            # Update config_manager state variables before plotting
            self.config_manager.findex = idx # Assuming idx corresponds to file index in map_params
            self.config_manager.pindex = idx # Assuming idx corresponds to plot index
            self.config_manager.axindex = 0 # Reset axis index for each plot

            field_data_array = data_source.dataset[field_name]
            plot_type = params.get('to_plot', ['xy'])[0] # Default to 'xy' if not specified

            self._process_plot(field_data_array, field_name, idx, plot_type, plotter)

        if self.config_manager.make_gif:
            pu.create_gif(self.config_manager.config) # Still needs config object? Check pu.create_gif

    # Removed _process_source_fields - logic moved into _single_plots

    def _process_plot(self, data_array: xr.DataArray, field_name: str, file_index: int, plot_type: str, plotter):
        """Process a single plot type for a given field."""
        self.logger.info(f"Plotting {field_name}, {plot_type} plot")
        figure = Figure.create_eviz_figure(self.config_manager, plot_type)
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        if 'xy' in plot_type or 'po' in plot_type:
            self._process_xy_plot(data_array, field_name, file_index, plot_type,
                                  figure, plotter)
        else:
            self._process_other_plot(data_array, field_name, file_index, plot_type,
                                     figure, plotter)

    def _process_xy_plot(self, data_array: xr.DataArray, field_name: str, file_index: int, plot_type: str, figure, plotter):
        """Process xy or polar plot types."""
        # Get vertical levels to process
        levels = self.config_manager.get_levels(field_name, plot_type + 'plot')
        do_zsum = self.config_manager.ax_opts.get('zsum', False) # Use .get with default

        # Get time levels to process
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0) # Default to 0
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time' # Default to 'time'
        num_times = data_array[tc_dim].size if tc_dim in data_array.dims else 1
        time_levels = range(num_times) if time_level_config == 'all' else [time_level_config]

        if not levels and not do_zsum:
            self.logger.warning(f' -> No levels specified for {field_name}')
            return

        if levels:
            self._process_level_plots(data_array, field_name, file_index, plot_type,
                                      figure, time_levels, levels, plotter)
        else:
            self._process_zsum_plots(data_array, field_name, file_index, plot_type,
                                     figure, time_levels, plotter)

    def _process_level_plots(self, data_array: xr.DataArray, field_name: str, file_index: int, plot_type: str, figure,
                            time_levels: list, levels: dict, plotter):
        """Process plots for specific vertical levels."""
        self.logger.info(f' -> Processing {len(time_levels)} time levels')
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev' # Default to 'lev'
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time' # Default to 'time'

        # Check if the vertical dimension exists in the data
        has_vertical_dim = zc_dim and zc_dim in data_array.dims
        
        for level_val in levels.keys(): # Iterate through level values
            self.config_manager.level = level_val
            for t in time_levels:
                # Select time level
                if tc_dim in data_array.dims:
                    data_at_time = data_array.isel({tc_dim: t})
                else:
                    data_at_time = data_array.squeeze() # Assume single time if no time dim

                self._set_time_config(t, data_at_time)

                # Create a new figure for each level to avoid reusing axes
                # Pass config_manager to Figure
                figure = Figure.create_eviz_figure(self.config_manager, plot_type)
                self.config_manager.ax_opts = figure.init_ax_opts(field_name)

                ax = figure.get_axes()
                
                # If the data doesn't have a vertical dimension, we can't select a level
                # In this case, we'll just use the data as is
                if not has_vertical_dim:
                    self.logger.warning(f"Data for {field_name} doesn't have a vertical dimension. Using data as is.")
                    field_to_plot = self._get_field_to_plot(ax, data_at_time, field_name,
                                                        file_index, plot_type, figure, t)
                else:
                    # Pass the data array slice and level value
                    field_to_plot = self._get_field_to_plot(ax, data_at_time, field_name,
                                                        file_index, plot_type, figure, t,
                                                        level=level_val)

                if field_to_plot:
                    plotter.single_plots(self.config_manager, field_to_plot=field_to_plot,
                                        level=level_val)

                    pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure,
                                level=level_val)

    def _process_other_plot(self, data_array: xr.DataArray, field_name: str, file_index: int, plot_type: str, figure,
                            plotter):
        """Process non-xy and non-po plot types."""
        self.config_manager.level = None
        # Get time levels to process
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0) # Default to 0
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time' # Default to 'time'
        
        # Check if time dimension exists
        if tc_dim in data_array.dims:
            num_times = data_array[tc_dim].size
            # TODO: Handle yx_plot Gifs
            time_levels = range(num_times) if time_level_config == 'all' else [time_level_config]
        else:
            self.logger.warning(f"No time dimension found in data for {field_name}. Using single time level.")
            time_levels = [0]

        ax = figure.get_axes()
        # Assuming these plot types (xt, tx) might not need time slicing here,
        # or slicing is handled within _get_field_to_plot
        # Pass the full data_array and let _get_field_to_plot handle slicing if needed
        field_to_plot = self._get_field_to_plot(ax, data_array, field_name, file_index,
                                                plot_type, figure, time_level=time_level_config)
        if field_to_plot:
            plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)


    def _process_zsum_plots(self, data_array: xr.DataArray, field_name: str, file_index: int, plot_type: str, figure,
                            time_levels: list, plotter):
        """Process plots with vertical summation."""
        self.config_manager.level = None
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time' # Default to 'time'
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev' # Default to 'lev'

        # Check if vertical dimension exists
        if not zc_dim or zc_dim not in data_array.dims:
            self.logger.warning(f"Cannot perform vertical summation: no vertical dimension found in data for {field_name}")
            # Just use the data as is
            data_array = data_array.squeeze()
        
        for t in time_levels:
            # Select time level if time dimension exists
            if tc_dim in data_array.dims:
                data_at_time = data_array.isel({tc_dim: t})
            else:
                self.logger.warning(f"No time dimension found in data for {field_name}. Using data as is.")
                data_at_time = data_array.squeeze() # Assume single time if no time dim

            self._set_time_config(t, data_at_time)
            field_to_plot = self._get_field_to_plot(None, data_at_time, field_name, # Pass None for ax initially
                                                    file_index, plot_type, figure, t)
            if field_to_plot:
                plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
                pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def _get_field_to_plot(self, ax, data_array: xr.DataArray, field_name: str,
                        file_index: int, plot_type: str, figure, time_level, level=None) -> tuple:
        """Prepare the data array and coordinates for plotting."""
        if data_array is None:
            self.logger.error(f"No data array provided for field {field_name}")
            return None

        dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
        data2d = None

        # Apply slicing and processing based on plot type
        if 'yz' in plot_type:
            data2d = self._get_yz(data_array, time_lev=time_level)
        elif 'xt' in plot_type:
            data2d = self._get_xt(data_array, time_lev=time_level)
        elif 'tx' in plot_type:
            data2d = self._get_tx(data_array, level=level, time_lev=time_level)
        elif 'xy' in plot_type or 'polar' in plot_type:
            data2d = self._get_xy(data_array, level=level, time_lev=time_level)
        else:
            self.logger.warning(f"Unsupported plot type for _get_field_to_plot: {plot_type}")
            return None

        if data2d is None:
            self.logger.error(f"Failed to prepare 2D data for field {field_name}, plot type {plot_type}")
            return None

        x_values = None
        y_values = None
        if 'xt' in plot_type or 'tx' in plot_type:
            # For time-series or Hovmoller plots, coordinates are handled differently
            # The plotter functions for these types will need to extract them from data2d
            pass
        else:
            # For 2D spatial plots (xy, yz, polar, sc), get the coordinate DataArrays
            try:
                # Use the determined dimension names
                if dim1_name and dim1_name in data2d.coords:
                    x_values = data2d[dim1_name]
                else:
                    self.logger.warning(f"Dimension '{dim1_name}' not found in data coordinates for {field_name}")

                if dim2_name and dim2_name in data2d.coords:
                    y_values = data2d[dim2_name]
                else:
                    self.logger.warning(f"Dimension '{dim2_name}' not found in data coordinates for {field_name}")

            except KeyError as e:
                self.logger.error(f"Error getting coordinates for {field_name}: {e}")
                # Fallback: try to use the first two dimensions as coordinates
                dims = list(data2d.dims)
                if len(dims) >= 2:
                    self.logger.warning(f"Falling back to using dimensions {dims[0]} and {dims[1]} as coordinates")
                    x_values = data2d[dims[0]]
                    y_values = data2d[dims[1]]
                else:
                    self.logger.error("Dataset has fewer than 2 dimensions, cannot plot spatial data")
                    return None
                    
            # check for and handle NaN values
            if np.isnan(data2d.values).all():
                self.logger.error(f"All values are NaN for {field_name}. Using original data.")
                data2d = data_array.squeeze()
            elif np.isnan(data2d.values).any():
                self.logger.warning(f"Note: Some NaN values present ({np.sum(np.isnan(data2d.values))} NaNs).")
                # data2d = data2d.fillna(0)

        # Return the prepared data and coordinates in the expected tuple format
        return data2d, x_values, y_values, field_name, plot_type, file_index, figure, ax

    # COMPARE_DIFF METHODS (always need SPECS file)
    #--------------------------------------------------------------------------
    def _comparison_plots(self, plotter):
        """Generate comparison plots for paired data sources according to configuration."""
        self.logger.info("Generating comparison plots")
        current_field_index = 0

        # Access data sources via the pipeline
        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for comparison plotting.")
            return

        # map1 and map2 are indices, not maps
        for idx1, idx2 in zip(self.config_manager.a_list, self.config_manager.b_list):
            # Get map parameters for these indices
            map1_params = self.config_manager.map_params.get(idx1)
            map2_params = self.config_manager.map_params.get(idx2)

            if not map1_params or not map2_params:
                self.logger.warning(f"Could not find map parameters for indices {idx1} or {idx2}. Skipping comparison.")
                continue

            # Get data sources from the pipeline
            filename1 = map1_params.get('filename')
            filename2 = map2_params.get('filename')

            data_source1 = self.config_manager.pipeline.get_data_source(filename1)
            data_source2 = self.config_manager.pipeline.get_data_source(filename2)

            if not data_source1 or not data_source2:
                self.logger.warning(f"Could not find data sources for {filename1} or {filename2}. Skipping comparison.")
                continue

            # Get the actual datasets
            sdat1_dataset = data_source1.dataset if hasattr(data_source1, 'dataset') else None
            sdat2_dataset = data_source2.dataset if hasattr(data_source2, 'dataset') else None

            if sdat1_dataset is None or sdat2_dataset is None:
                 self.logger.warning(f"Datasets not loaded for {filename1} or {filename2}. Skipping comparison.")
                 continue

            # Determine file indices (these are the indices in app_data.inputs)
            # The config_manager already provides a_list and b_list which are these indices
            file_indices = (idx1, idx2)

            # Process each plot type
            field_name1 = map1_params.get('field')
            field_name2 = map2_params.get('field')

            if not field_name1 or not field_name2:
                self.logger.warning(f"Field names not specified for comparison indices {idx1} or {idx2}. Skipping.")
                continue

            self.field_names = (field_name1, field_name2)

            # Assuming plot types are the same for comparison
            plot_types = map1_params.get('to_plot', ['xy'])
            for plot_type in plot_types:
                self.logger.info(
                    f"Plotting {field_name1} vs {field_name2}, {plot_type} plot")
                self.data2d_list = [] # Reset for each plot type

                # Pass the datasets to the processing method
                if 'xy' in plot_type or 'po' in plot_type:
                    self._process_xy_comparison_plots(plotter, file_indices,
                                                    current_field_index,
                                                    field_name1, field_name2, plot_type,
                                                    sdat1_dataset, sdat2_dataset)
                else:
                    self._process_other_comparison_plots(plotter, file_indices,
                                                    current_field_index,\
                                                    field_name1, field_name2,\
                                                    plot_type, sdat1_dataset, sdat2_dataset)

            current_field_index += 1

    # Removed _load_comparison_data - data is accessed from the pipeline
    # Removed _get_file_indices_compare - indices are provided by config_manager.a_list and b_list

    def _process_xy_comparison_plots(self, plotter, file_indices: tuple, current_field_index: int,\
                                     field_name1: str, field_name2: str, plot_type: str,\
                                     sdat1_dataset: xr.Dataset, sdat2_dataset: xr.Dataset):
        """Process comparison plots for xy or polar plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels

        # Get levels for the plots (assuming levels are the same for both fields in comparison)
        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            self.logger.warning(f' -> No levels specified for {field_name1}')
            return

        for level_val in levels.keys(): # Iterate through level values
            # Create a figure with appropriate subplots
            figure = Figure.create_eviz_figure(self.config_manager, plot_type, nrows=nrows, ncols=ncols)
            ax = figure.get_axes() # Get the axes array
            axes_shape = figure.get_gs_geometry() # Get the grid spec geometry
            self.config_manager.level = level_val

            # Pass the datasets and level value to the creation methods
            if axes_shape == (3, 1):
                self._create_3x1_comparison_plot(plotter, file_indices,
                                                 current_field_index,
                                                 field_name1, field_name2, figure, ax,
                                                 plot_type, sdat1_dataset, sdat2_dataset, level_val)
            elif axes_shape == (2, 2):
                self._create_2x2_comparison_plot(plotter, file_indices,
                                                 current_field_index,
                                                 field_name1, field_name2, figure, ax, # Pass ax here
                                                 plot_type, sdat1_dataset, sdat2_dataset, level_val)

            # Save the plot
            # Set findex to file_index1 before calling print_map
            self.config_manager.findex = file_index1
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure, level=level_val)
            self.comparison_plot = False # Reset comparison flag


    def _process_other_comparison_plots(self, plotter, file_indices: tuple, current_field_index: int,
                                        field_name1: str, field_name2: str, plot_type: str,
                                        sdat1_dataset: xr.Dataset, sdat2_dataset: xr.Dataset):
        """Process comparison plots for other plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels

        figure = Figure.create_eviz_figure(self.config_manager, plot_type, nrows=nrows, ncols=ncols)  
        ax = figure.get_axes()
        axes_shape = figure.get_gs_geometry()
        self.config_manager.level = None

        if axes_shape == (3, 1):
            self._create_3x1_comparison_plot(plotter, file_indices, current_field_index,
                                            field_name1, field_name2, figure, ax,
                                            plot_type, sdat1_dataset, sdat2_dataset)
        elif axes_shape == (2, 2):
            self._create_2x2_comparison_plot(plotter, file_indices, current_field_index,
                                            field_name1, field_name2, figure, ax,
                                            plot_type, sdat1_dataset, sdat2_dataset)

        # Save the plot
        # Set findex to file_index1 before calling print_map
        self.config_manager.findex = file_index1
        pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)
        self.comparison_plot = False # Reset comparison flag


    def _create_3x1_comparison_plot(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, figure, ax,
                                    plot_type, sdat1_dataset, sdat2_dataset, level=None):
        """Create a 3x1 comparison plot."""
        file_index1, file_index2 = file_indices

        # Plot the first dataset
        self._process_3x1_comparison_plot(plotter, file_index1, current_field_index,
                                    field_name1, figure, ax, 0, 
                                    sdat1_dataset[field_name1], plot_type, level=level)

        # Plot the second dataset
        self._process_3x1_comparison_plot(plotter, file_index2, current_field_index,
                                    field_name2, figure, ax, 1, 
                                    sdat2_dataset[field_name2], plot_type, level=level)

        # Debug logging for difference calculation
        self.logger.info(f"Calculating difference between {field_name1} and {field_name2}")
        self.logger.info(f"Data1 shape: {sdat1_dataset[field_name1].shape}, Data2 shape: {sdat2_dataset[field_name2].shape}")
        self.logger.info(f"Data1 min/max: {sdat1_dataset[field_name1].min().values}/{sdat1_dataset[field_name1].max().values}")
        self.logger.info(f"Data2 min/max: {sdat2_dataset[field_name2].min().values}/{sdat2_dataset[field_name2].max().values}")
        
        # Reset data2d_list before processing the difference plot
        self.data2d_list = []
        
        # Plot the comparison (difference)
        self.comparison_plot = True
        # For the comparison, we need to pass both datasets
        # The _process_comparison_plot method will need to handle this special case
        self._process_3x1_comparison_plot(plotter, file_index1, current_field_index,
                                field_name1, figure, ax, 2, 
                                (sdat1_dataset[field_name1], sdat2_dataset[field_name2]), 
                                plot_type, level=level)


    def _create_2x2_comparison_plot(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, figure, ax,
                                    plot_type, sdat1_dataset, sdat2_dataset, level=None):
        """Create a 2x2 comparison plot."""
        file_index1, file_index2 = file_indices

        # Plot the first dataset in the top-left
        self._process_2x2_comparison_plot(plotter, file_index1, current_field_index,
                                        field_name1, figure, [0, 0], 0, 
                                        sdat1_dataset[field_name1], plot_type,
                                        level=level)

        # Plot the second dataset in the top-right
        self._process_2x2_comparison_plot(plotter, file_index2, current_field_index,
                                        field_name2, figure, [0, 1], 1, 
                                        sdat2_dataset[field_name2], plot_type,
                                        level=level)

        # Plot comparison in the bottom row
        self.comparison_plot = True
        # For the comparison, we need to pass both datasets
        self._process_2x2_comparison_plot(plotter, file_index1, current_field_index,
                                        field_name1, figure, [1, 0], 2, 
                                        (sdat1_dataset[field_name1], sdat2_dataset[field_name2]), 
                                        plot_type, level=level)
        
        # If extra field type is enabled, plot another comparison view
        if self.config_manager.ax_opts.get('add_extra_field_type', False):
            self._process_2x2_comparison_plot(plotter, file_index1, current_field_index,
                                            field_name1, figure, [1, 1], 3, 
                                            (sdat1_dataset[field_name1], sdat2_dataset[field_name2]), 
                                            plot_type, level=level)


    def _process_3x1_comparison_plot(self, plotter, file_index, current_field_index, field_name, 
                            figure, ax, ax_index, data_array, plot_type, level=None):
        """Process a comparison plot."""
        # Set state variables on config_manager
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        
        # Set difference field flag if this is the comparison panel
        if ax_index == 2:  # Third panel in 3x1 layout is the difference
            self.config_manager.ax_opts['is_diff_field'] = True
        
        # Set up the axis
        if isinstance(ax, tuple):
            fig, axes = ax
            current_ax = axes[ax_index]
        elif isinstance(ax, list) and len(ax) > ax_index:
            current_ax = ax[ax_index]
        else:
            # Fall back to using the figure's axes directly
            current_ax = figure.get_axes()[ax_index]
        
        figure.set_ax_opts_diff_field(current_ax)
        
        # Handle data_array differently based on its type
        if isinstance(data_array, tuple):
            # If data_array is a tuple (from comparison plots), it contains both datasets
            # We need to prepare the difference field
            data1, data2 = data_array
            
            # Prepare each field individually first
            field_to_plot1 = self._get_field_to_plot_compare(data1, field_name, file_index,
                                                        plot_type, figure, level=level)
            field_to_plot2 = self._get_field_to_plot_compare(data2, field_name, file_index,
                                                        plot_type, figure, level=level)
            
            # Now compute the difference
            # This assumes _get_field_to_plot_compare returns a tuple where the first element is the data array
            if field_to_plot1 and field_to_plot2:
                data2d1, x1, y1 = field_to_plot1[0], field_to_plot1[1], field_to_plot1[2]
                data2d2, x2, y2 = field_to_plot2[0], field_to_plot2[1], field_to_plot2[2]
                
                # Debug logging for difference calculation
                self.logger.info("Calculating difference between data arrays")
                
                # Check if data arrays are valid before accessing attributes
                if data2d1 is not None and data2d2 is not None:
                    self.logger.info(f"Data1 shape: {data2d1.shape}, Data2 shape: {data2d2.shape}")
                    
                    # Check if min/max operations are valid
                    try:
                        self.logger.info(f"Data1 min/max: {data2d1.min().values}/{data2d1.max().values}")
                        self.logger.info(f"Data2 min/max: {data2d2.min().values}/{data2d2.max().values}")
                    except Exception as e:
                        self.logger.warning(f"Could not get min/max values: {e}")
                else:
                    self.logger.error(f"One or both data arrays are None: data2d1={data2d1 is not None}, data2d2={data2d2 is not None}")
                    # Create dummy data arrays if needed
                    if data2d1 is None and data2d2 is not None:
                        self.logger.info("Creating dummy data array for data2d1")
                        data2d1 = xr.zeros_like(data2d2)
                    elif data2d2 is None and data2d1 is not None:
                        self.logger.info("Creating dummy data array for data2d2")
                        data2d2 = xr.zeros_like(data2d1)
                    elif data2d1 is None and data2d2 is None:
                        self.logger.error("Both data arrays are None, cannot create difference plot")
                        return
                    
                proc = DataProcessor(self.config_manager) 
                proc.data2d_list = [data2d1, data2d2] 
                dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
                diff_result = proc.regrid(data2d1, data2d2, dim1_name, dim2_name)
                
                if diff_result is None or diff_result[0] is None:
                    self.logger.error("Regridding failed, cannot create difference plot")
                    # Create a dummy field_to_plot with zeros
                    dummy_data = xr.zeros_like(data2d1)
                    field_to_plot = (dummy_data, x1, y1, field_name, plot_type, 
                                    file_index, figure, current_ax)
                    return
                
                target, regridded = diff_result
                # Compute the difference
                diff_data = target - regridded
                # Get the coordinates from the target grid
                diff_x = target[dim1_name].values if dim1_name in target.coords else None
                diff_y = target[dim2_name].values if dim2_name in target.coords else None

                field_to_plot = (diff_data, diff_x, diff_y, field_name, plot_type, 
                                file_index, figure, current_ax)
            else:
                self.logger.error("Could not prepare data for comparison plot")
                return
        else:
            # Regular single dataset case
            field_to_plot = self._get_field_to_plot_compare(data_array, field_name, file_index,
                                                        plot_type, figure, ax=current_ax, level=level)
            # Store the data for potential later use in difference calculation
            if field_to_plot:
                self.data2d_list.append(field_to_plot[0])
        
        # Call the plotter with the prepared data
        if field_to_plot:
            plotter.comparison_plots(self.config_manager, field_to_plot, level=level)


    def _process_2x2_comparison_plot(self, plotter, file_index, current_field_index, field_name, 
                            figure, gsi, ax_index, data_array, plot_type, level=None):
        # Get the axes array
        axes = figure.get_axes()
        
        # Get the correct axis based on grid spec indices
        if isinstance(axes, np.ndarray):
            current_ax = axes[gsi[0], gsi[1]]
        else:
            # Fall back to using subplot if axes is not a 2D array
            current_ax = plt.subplot(figure.gs[gsi[0], gsi[1]])
        
        # Set state variables on config_manager
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        
        # Initialize ax_opts BEFORE setting flags
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        
        # Set difference field flag if this is a comparison panel (bottom row)
        if gsi[0] == 1:  # Bottom row in 2x2 layout is for differences
            self.config_manager.ax_opts['is_diff_field'] = True
            # Set extra field type flag for the bottom-right panel
            if gsi[1] == 1:
                self.config_manager.ax_opts['add_extra_field_type'] = True
        
        figure.set_ax_opts_diff_field(current_ax)
        
        # Handle data_array differently based on its type
        if isinstance(data_array, tuple):
            # If data_array is a tuple (from comparison plots), it contains both datasets
            # We need to prepare the difference field
            data1, data2 = data_array
            
            # Prepare each field individually first
            field_to_plot1 = self._get_field_to_plot_compare(data1, field_name, file_index,
                                                        plot_type, figure, level=level)
            field_to_plot2 = self._get_field_to_plot_compare(data2, field_name, file_index,
                                                        plot_type, figure, level=level)
            
            # Now compute the difference
            if field_to_plot1 and field_to_plot2:
                data2d1, x1, y1 = field_to_plot1[0], field_to_plot1[1], field_to_plot1[2]
                data2d2, x2, y2 = field_to_plot2[0], field_to_plot2[1], field_to_plot2[2]
                
                proc = DataProcessor(self.config_manager) 
                proc.data2d_list = [data2d1, data2d2] 
                dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
                diff_result = proc.regrid(data2d1, data2d2, dim1_name, dim2_name)
                
                if diff_result is None or diff_result[0] is None:
                    self.logger.error("Regridding failed, cannot create difference plot")
                    # Create a dummy field_to_plot with zeros
                    dummy_data = xr.zeros_like(data2d1)
                    field_to_plot = (dummy_data, x1, y1, field_name, plot_type, 
                                    file_index, figure, current_ax)
                    return
                
                target, regridded = diff_result
                # Compute the difference
                diff_data = target - regridded
                # Get the coordinates from the target grid
                diff_x = target[dim1_name].values if dim1_name in target.coords else None
                diff_y = target[dim2_name].values if dim2_name in target.coords else None
                                
                field_to_plot = (diff_data, diff_x, diff_y, field_name, plot_type, 
                                file_index, figure, current_ax)
            else:
                self.logger.error("Could not prepare data for comparison plot")
                return
        else:
            # Regular single dataset case
            field_to_plot = self._get_field_to_plot_compare(data_array, field_name, file_index,
                                                        plot_type, figure, ax=current_ax, level=level)
            # Store the data for potential later use in difference calculation
            if field_to_plot:
                self.data2d_list.append(field_to_plot[0])
        
        # Call the plotter with the prepared data
        if field_to_plot:
            plotter.comparison_plots(self.config_manager, field_to_plot, level=level)


    def _get_field_to_plot_compare(self, data_array, field_name, file_index, 
                                plot_type, figure, ax=None, level=None) -> tuple:
        """Prepare data for comparison plots."""
        if ax is None:
            ax = figure.get_axes()
        
        dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
        data2d = None
        
        # Check if this is a difference field calculation
        if self.config_manager.ax_opts.get('is_diff_field', False) and len(self.data2d_list) >= 2:
            # If we already have two data arrays stored, use Interp to regrid and compute difference
            proc = DataProcessor(self.config_manager)
            data2d, x_values, y_values = proc.regrid(plot_type)
            return data2d, x_values, y_values, self.field_names[0], plot_type, file_index, figure, ax
        else:
            # Regular data preparation based on plot type
            if 'yz' in plot_type:
                data2d = self._get_yz(data_array, time_lev=self.config_manager.ax_opts.get('time_lev', 0))
            elif 'xt' in plot_type:
                data2d = self._get_xt(data_array, time_lev=self.config_manager.ax_opts.get('time_lev', 0))
            elif 'tx' in plot_type:
                data2d = self._get_tx(data_array, level=level, time_lev=self.config_manager.ax_opts.get('time_lev', 0))
            elif 'xy' in plot_type or 'polar' in plot_type:
                data2d = self._get_xy(data_array, level=level, time_lev=self.config_manager.ax_opts.get('time_lev', 0))
            else:
                self.logger.warning(f"Unsupported plot type for _get_field_to_plot_compare: {plot_type}")
                return None
        
        # Store the processed data for potential difference calculation
        self.data2d_list.append(data2d)
        
        # For time series plots, coordinates are handled differently
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        
        # For spatial plots, get the coordinate arrays
        try:
            x_values = data2d[dim1_name].values if dim1_name in data2d.coords else None
            y_values = data2d[dim2_name].values if dim2_name in data2d.coords else None
            return data2d, x_values, y_values, field_name, plot_type, file_index, figure, ax
        except KeyError as e:
            self.logger.error(f"Error getting coordinates for {field_name}: {e}")
            # Fallback to using the first two dimensions
            dims = list(data2d.dims)
            if len(dims) >= 2:
                x_values = data2d[dims[0]].values
                y_values = data2d[dims[1]].values
                return data2d, x_values, y_values, field_name, plot_type, file_index, figure, ax
            else:
                self.logger.error(f"Dataset has fewer than 2 dimensions, cannot plot")
                return None

    # SIDE-BY-SIDE COMPARE METHODS (always need SPECS file)
    #--------------------------------------------------------------------------
    def _side_by_side_plots(self, plotter):
        """
        Generate side-by-side comparison plots (2x1 subplots) without difference.
        """
        current_field_index = 0
        self.data2d_list = []  # Initialize list to store data for comparison

        # Process each pair of indices from the comparison configuration
        for idx1, idx2 in zip(self.config_manager.a_list, self.config_manager.b_list):
            # Get map parameters for these indices
            map1_params = self.config_manager.map_params.get(idx1)
            map2_params = self.config_manager.map_params.get(idx2)

            if not map1_params or not map2_params:
                self.logger.warning(f"Could not find map parameters for indices {idx1} or {idx2}. Skipping comparison.")
                continue

            # Get data sources from the pipeline
            filename1 = map1_params.get('filename')
            filename2 = map2_params.get('filename')

            data_source1 = self.config_manager.pipeline.get_data_source(filename1)
            data_source2 = self.config_manager.pipeline.get_data_source(filename2)

            if not data_source1 or not data_source2:
                self.logger.warning(f"Could not find data sources for {filename1} or {filename2}. Skipping comparison.")
                continue

            # Get the actual datasets
            sdat1_dataset = data_source1.dataset if hasattr(data_source1, 'dataset') else None
            sdat2_dataset = data_source2.dataset if hasattr(data_source2, 'dataset') else None

            if sdat1_dataset is None or sdat2_dataset is None:
                self.logger.warning(f"Datasets not loaded for {filename1} or {filename2}. Skipping comparison.")
                continue

            # Determine file indices
            file_indices = (idx1, idx2)

            # Process each plot type
            field_name1 = map1_params.get('field')
            field_name2 = map2_params.get('field')

            if not field_name1 or not field_name2:
                self.logger.warning(f"Field names not specified for comparison indices {idx1} or {idx2}. Skipping.")
                continue

            self.field_names = (field_name1, field_name2)

            # Assuming plot types are the same for comparison
            plot_types = map1_params.get('to_plot', ['xy'])
            for plot_type in plot_types:
                self.logger.info(f"Plotting {field_name1} vs {field_name2} side by side, {plot_type} plot")
                self.data2d_list = []  # Reset for each plot type

                if 'xy' in plot_type or 'polar' in plot_type:
                    self._process_xy_side_by_side_plots(plotter, file_indices,
                                                    current_field_index,
                                                    field_name1, field_name2, plot_type,
                                                    sdat1_dataset, sdat2_dataset)
                else:
                    self._process_other_side_by_side_plots(plotter, file_indices,
                                                        current_field_index,
                                                        field_name1, field_name2,
                                                        plot_type, sdat1_dataset, sdat2_dataset)

            current_field_index += 1

    def _process_xy_side_by_side_plots(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, plot_type, sdat1_dataset, sdat2_dataset):
        """Process side-by-side comparison plots for xy or polar plot types."""
        file_index1, file_index2 = file_indices
        
        # Get the number of plots from compare_exp_ids
        num_plots = len(self.config_manager.compare_exp_ids)
        
        # Set up the panel layout
        nrows = 1
        ncols = num_plots  # This will be 3 for three variables
        
        # Get levels for the plots
        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            self.logger.warning(f' -> No levels specified for {field_name1}')
            return
        
        for level_val in levels.keys():
            # Create a figure with 1xN subplots
            figure = Figure.create_eviz_figure(self.config_manager, plot_type, nrows=nrows, ncols=ncols)
            ax = figure.get_axes()
            self.config_manager.level = level_val
            
            # Create the side-by-side comparison plot
            self._create_xy_side_by_side_plot(plotter, file_indices,
                                        current_field_index,
                                        field_name1, field_name2, figure, ax,
                                        plot_type, sdat1_dataset, sdat2_dataset, level_val)
            
            # Save the plot
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure, level=level_val)

    def _create_xy_side_by_side_plot(self, plotter, file_indices, current_field_index,
                                field_name1, field_name2, figure, ax,
                                plot_type, sdat1_dataset, sdat2_dataset, level=None):
        """
        Create a side-by-side comparison plot for the given data.
        
        The layout is:
        - Left subplot: First dataset
        - Middle subplot: Second dataset
        - Right subplot: Third dataset (if present)
        """
        file_index1, file_index2 = file_indices
        
        # Ensure ax is a list with enough elements for all plots
        if not isinstance(ax, list):
            ax = [ax]
        num_plots = len(self.config_manager.compare_exp_ids)
        if len(ax) < num_plots:
            self.logger.warning(f"Not enough axes for {num_plots}-way comparison. Using the first axis.")
            ax = [ax[0]] * num_plots
        
        # Plot each dataset in its respective subplot
        self.comparison_plot = False
        
        # Plot first dataset (from a_list)
        if self.config_manager.a_list:
            self._process_side_by_side_plot(plotter, self.config_manager.a_list[0],
                                        current_field_index,
                                        field_name1, figure, ax, 0,
                                        sdat1_dataset[field_name1], plot_type, level=level)
        
        # Plot remaining datasets (from b_list)
        for i, file_idx in enumerate(self.config_manager.b_list, start=1):
            if i < num_plots:  # Only plot if we have a corresponding axis
                self._process_side_by_side_plot(plotter, file_idx,
                                            current_field_index,
                                            field_name2, figure, ax, i,
                                            sdat2_dataset[field_name2], plot_type, level=level)

    def _process_other_side_by_side_plots(self, plotter, file_indices, current_field_index,
                                        field_name1, field_name2, plot_type, sdat1_dataset, sdat2_dataset):
        """Process side-by-side comparison plots for other plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels
        
        # Create a figure with 2x1 subplots (side by side)
        figure = Figure.create_eviz_figure(self.config_manager, plot_type, nrows=nrows, ncols=ncols)
        axes_shape = figure.subplots
        ax = figure.get_axes()
        self.config_manager.level = None
        
        # Create the nx1 side-by-side comparison plot
        self._create_other_side_by_side_plot(plotter, file_indices, current_field_index,
                                        field_name1, field_name2, figure, ax,
                                        plot_type, sdat1_dataset, sdat2_dataset)
        
        # Save the plot
        pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def _create_other_side_by_side_plot(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, figure, ax,
                                    plot_type, sdat1_dataset, sdat2_dataset, level=None):
        """
        Create a nx1 side-by-side comparison plot for the given data.
        
        The layout is:
        - Left subplot: First dataset
        - Right subplot: Second dataset
        - etc... up to 3x1
        """
        file_index1, file_index2 = file_indices
        
        # Plot the first dataset in the left subplot
        self.comparison_plot = False
        # TODO: add loop for n in range(3)...
        self._process_side_by_side_plot(plotter, file_index1, current_field_index,
                                    field_name1,
                                    figure, ax, 0, sdat1_dataset[field_name1], plot_type, level=level)
        
        # Plot the second dataset in the right subplot
        self._process_side_by_side_plot(plotter, file_index2, current_field_index,
                                    field_name2,
                                    figure, ax, 1, sdat2_dataset[field_name2], plot_type, level=level)

    def _process_side_by_side_plot(self, plotter, file_index, current_field_index, field_name, 
                                figure, ax, ax_index, data_array, plot_type, level=None):
        """Process a single plot for side-by-side comparison."""
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        
        # Set up the axis
        if isinstance(ax, list) and len(ax) > ax_index:
            current_ax = ax[ax_index]
        else:
            self.logger.warning(f"Axis index {ax_index} out of range. Using the first axis.")
            current_ax = ax[0] if isinstance(ax, list) and len(ax) > 0 else ax
        
        # Get field to plot
        field_to_plot = self._get_field_to_plot_compare(data_array, field_name, file_index,
                                                    plot_type, figure, ax=current_ax, level=level)
        
        # Store the data for potential later use
        if field_to_plot and field_to_plot[0] is not None:
            self.data2d_list.append(field_to_plot[0])
        
        # Call the plotter with the prepared data
        if field_to_plot:
            if hasattr(plotter, 'single_plots'):
                plotter.single_plots(self.config_manager, field_to_plot, level=level)
            elif hasattr(plotter, 'comparison_plots'):
                plotter.comparison_plots(self.config_manager, field_to_plot, level=level)
            else:
                self.logger.warning(f"Unknown plotter type: {type(plotter).__name__}. Trying to call plot method.")
                if hasattr(plotter, 'plot'):
                    plotter.plot(self.config_manager, field_to_plot, level=level)
                else:
                    self.logger.error(f"Plotter {type(plotter).__name__} has no plot method.")

    # DATA SLICE PROCESSING METHODS
    #--------------------------------------------------------------------------    
    def _get_yz(self, data_array, time_lev):
        """ Extract YZ slice (zonal mean) from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 2D (lat, lev) slice
        """
        if data_array is None:
            return None

        # Get dimension names from config_manager
        xc_dim = self.config_manager.get_model_dim_name('xc')
        tc_dim = self.config_manager.get_model_dim_name('tc')
        zc_dim = self.config_manager.get_model_dim_name('zc')

        # Check if vertical dimension exists
        if not zc_dim or zc_dim not in data_array.dims:
            self.logger.error(f"Cannot create YZ plot: no vertical dimension found in data for {data_array.name}")
            return None

        # Compute zonal mean if longitude dimension exists
        if xc_dim and xc_dim in data_array.dims:
            zonal_mean = data_array.mean(dim=xc_dim)
        else:
            self.logger.error(f"Could not find any longitude dimension for zonal mean in {data_array.name}")
            return None

        # Copy attributes
        zonal_mean.attrs = data_array.attrs.copy()

        # Handle time dimension
        if tc_dim and tc_dim in zonal_mean.dims:
            num_times = zonal_mean[tc_dim].size
            if self.config_manager.ax_opts.get('tave', False) and num_times > 1:
                self.logger.debug(f"Averaging over {num_times} time levels.")
                zonal_mean = apply_mean(self.config_manager, zonal_mean)
            else:
                # Select the specified time level
                if isinstance(time_lev, int) and time_lev < num_times:
                    zonal_mean = zonal_mean.isel({tc_dim: time_lev})
                else:
                    self.logger.warning(f"Time level {time_lev} out of bounds, using first time level")
                    zonal_mean = zonal_mean.isel({tc_dim: 0})
        else:
            # No time dimension, just squeeze
            zonal_mean = zonal_mean.squeeze()

        # Apply y-range selection if specified
        zonal_mean = self._select_yrange(zonal_mean, data_array.name)
        
        return apply_conversion(self.config_manager, zonal_mean, data_array.name)

    def _get_xy(self, data_array, level, time_lev):
        """ Extract XY slice (latlon) from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 2D (lon, lat) slice
        """
        if data_array is None:
            return None

        # Debug: Log input data stats
        self.logger.info(f"_get_xy input: shape={data_array.shape}, dims={data_array.dims}")
        self.logger.info(f"_get_xy input stats: min={data_array.min().values}, max={data_array.max().values}")

        # Get dimension names from config_manager
        tc_dim = self.config_manager.get_model_dim_name('tc')
        zc_dim = self.config_manager.get_model_dim_name('zc')

        # Debug: Log dimension names
        self.logger.info(f"Dimension names: tc_dim={tc_dim}, zc_dim={zc_dim}")

        # Make a copy to avoid modifying the original
        d_temp = data_array.copy()

        # Handle time dimension
        if tc_dim and tc_dim in d_temp.dims:
            num_tc = d_temp[tc_dim].size
            self.logger.info(f"Time dimension found: {tc_dim}, size={num_tc}")
            if isinstance(time_lev, int) and time_lev < num_tc:
                d_temp = d_temp.isel({tc_dim: time_lev})
                self.logger.info(f"Selected time level {time_lev}")
            else:
                self.logger.warning(f"Time level {time_lev} out of bounds, using first time level")
                d_temp = d_temp.isel({tc_dim: 0})
        else:
            self.logger.info(f"No time dimension found matching {tc_dim}")

        # Handle vertical dimension
        has_vertical_dim = zc_dim and zc_dim in d_temp.dims
        if has_vertical_dim:
            self.logger.info(f"Vertical dimension found: {zc_dim}, size={d_temp[zc_dim].size}")
            
            # Handle level selection
            if level is not None:
                # Try to find the level in the vertical coordinate
                try:
                    # First try exact matching
                    if level in d_temp[zc_dim].values:
                        lev_idx = np.where(d_temp[zc_dim].values == level)[0][0]
                        d_temp = d_temp.isel({zc_dim: lev_idx})
                        self.logger.info(f"Selected exact level {level} at index {lev_idx}")
                    else:
                        # Try nearest neighbor
                        lev_idx = np.abs(d_temp[zc_dim].values - level).argmin()
                        self.logger.warning(f"Level {level} not found exactly, using nearest level {d_temp[zc_dim].values[lev_idx]}")
                        d_temp = d_temp.isel({zc_dim: lev_idx})
                except Exception as e:
                    self.logger.error(f"Error selecting level {level}: {e}")
                    # If level selection fails, use the first level
                    if d_temp[zc_dim].size > 0:
                        d_temp = d_temp.isel({zc_dim: 0})
                        self.logger.info("Falling back to first level")
            else:
                # No level specified, use the first level
                if d_temp[zc_dim].size > 0:
                    d_temp = d_temp.isel({zc_dim: 0})
                    self.logger.info("No level specified, using first level")
        elif level is not None:
            # If level is specified but there's no vertical dimension, log a warning
            self.logger.warning(f"Level {level} specified but no vertical dimension found in data. Using data as is.")

        # Squeeze to remove singleton dimensions
        data2d = d_temp.squeeze()
        self.logger.info(f"After squeeze: shape={data2d.shape}, dims={data2d.dims}")

        # Check if we still have more than 2 dimensions
        if len(data2d.dims) > 2:
            self.logger.warning(f"Data still has {len(data2d.dims)} dimensions after processing. Attempting to reduce to 2D.")
            
            # Try to identify the most likely 2D slice to use
            # Typically, we want lon-lat for xy plots
            xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
            yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'
            
            # Check if we have both lon and lat dimensions
            if xc_dim in data2d.dims and yc_dim in data2d.dims:
                # For each remaining dimension that's not lon or lat, take the first index
                for dim in data2d.dims:
                    if dim != xc_dim and dim != yc_dim:
                        self.logger.info(f"Selecting first index of extra dimension: {dim}")
                        data2d = data2d.isel({dim: 0})
            else:
                # If we don't have both lon and lat, take the first two dimensions
                dims = list(data2d.dims)
                self.logger.warning(f"Could not identify lon-lat dimensions. Using first two dimensions: {dims[:2]}")
                
                # For each dimension beyond the first two, take the first index
                for dim in dims[2:]:
                    self.logger.info(f"Selecting first index of extra dimension: {dim}")
                    data2d = data2d.isel({dim: 0})
            
            # Squeeze again to remove any singleton dimensions
            data2d = data2d.squeeze()
            self.logger.info(f"After dimension reduction: shape={data2d.shape}, dims={data2d.dims}")
            
            # Final check - if we still have more than 2 dimensions, we need to reshape
            if len(data2d.dims) > 2:
                self.logger.warning(f"Data still has {len(data2d.dims)} dimensions. Reshaping to 2D.")
                # Get the first two dimensions
                dims = list(data2d.dims)
                dim1, dim2 = dims[0], dims[1]
                
                # Reshape by taking the mean over all other dimensions
                for dim in dims[2:]:
                    data2d = data2d.mean(dim=dim)
                
                self.logger.info(f"After reshaping: shape={data2d.shape}, dims={data2d.dims}")

        # Handle time averaging if requested
        if tc_dim and tc_dim in data2d.dims and self.config_manager.ax_opts.get('tave', False):
            num_tc = data2d[tc_dim].size
            if num_tc > 1:
                self.logger.debug(f"Averaging over {num_tc} time levels.")
                data2d = apply_mean(self.config_manager, data2d, level)
                return apply_conversion(self.config_manager, data2d, data_array.name)

        # Handle vertical averaging if requested
        if self.config_manager.ax_opts.get('zave', False):
            self.logger.debug(f"Averaging over vertical levels.")
            data2d = apply_mean(self.config_manager, data2d, level='all')
            return apply_conversion(self.config_manager, data2d, data_array.name)

        # Handle vertical summation if requested
        if self.config_manager.ax_opts.get('zsum', False):
            self.logger.debug(f"Summing over vertical levels.")
            data2d_zsum = apply_zsum(self.config_manager, data2d)
            self.logger.debug(f"Min: {data2d_zsum.min().values}, Max: {data2d_zsum.max().values}")
            return apply_conversion(self.config_manager, data2d_zsum, data_array.name)

        # Debug: Log final data stats
        self.logger.info(f"_get_xy output: shape={data2d.shape}, dims={data2d.dims}")
        self.logger.info(f"_get_xy output stats: min={data2d.min().values}, max={data2d.max().values}")

        # Check for NaN values
        if np.isnan(data2d.values).any():
            self.logger.warning(f"Output contains NaN values: {np.sum(np.isnan(data2d.values))} NaNs")

        return apply_conversion(self.config_manager, data2d, data_array.name)

    def _get_xt(self, data_array, time_lev):
        """ Extract time-series from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 1D (time) series
        """
        if data_array is None:
            return None
                
        # Debug: Log input data stats
        self.logger.info(f"_get_xt input: shape={data_array.shape}, dims={data_array.dims}")
        self.logger.info(f"_get_xt input stats: min={data_array.min().values}, max={data_array.max().values}")
        
        # Get time dimension safely
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'
        xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
        yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'
        
        # Debug: Log dimension names
        self.logger.info(f"Dimension names: tc_dim={tc_dim}, zc_dim={zc_dim}, xc_dim={xc_dim}, yc_dim={yc_dim}")
        
        # Try to get the number of time steps safely
        try:
            if tc_dim in data_array.dims:
                num_times = data_array[tc_dim].size
                self.logger.info(f"Found time dimension '{tc_dim}' with {num_times} steps")
            else:
                # Fall back to 'time' if tc_dim not found in dimensions
                if 'time' in data_array.dims:
                    num_times = data_array.time.size
                    tc_dim = 'time'
                    self.logger.info(f"Using fallback time dimension 'time' with {num_times} steps")
                else:
                    # Try to find any time-like dimension
                    time_dims = [dim for dim in data_array.dims if 'time' in dim.lower()]
                    if time_dims:
                        tc_dim = time_dims[0]
                        num_times = data_array[tc_dim].size
                        self.logger.info(f"Using inferred time dimension '{tc_dim}' with {num_times} steps")
                    else:
                        # If all else fails, try to infer
                        if hasattr(data_array, 'shape') and len(data_array.shape) > 0:
                            num_times = data_array.shape[0]  # Assume time is the first dimension
                            self.logger.warning(f"No time dimension found, assuming first dimension with {num_times} steps")
                        else:
                            self.logger.error(f"Cannot determine time dimension for {data_array.name}")
                            return None
        except (AttributeError, KeyError) as e:
            self.logger.error(f"Error determining time dimension: {e}")
            # If all else fails, try to infer
            if hasattr(data_array, 'shape') and len(data_array.shape) > 0:
                num_times = data_array.shape[0]  # Assume time is the first dimension
                self.logger.warning(f"Error with time dimension, assuming first dimension with {num_times} steps")
            else:
                self.logger.error(f"Cannot determine time dimension for {data_array.name}")
                return None
        
        self.logger.info(f"'{data_array.name}' field has {num_times} time levels")

        # Make a copy to avoid modifying the original
        data2d = data_array.copy()

        # Handle time range selection
        if isinstance(time_lev, list):
            self.logger.info(f"Computing time series on {time_lev} time range")
            try:
                if tc_dim in data2d.dims:
                    data2d = data2d.isel({tc_dim: slice(*time_lev)})
                    self.logger.info(f"Selected time range {time_lev} from dimension {tc_dim}")
                else:
                    self.logger.warning(f"Time dimension {tc_dim} not found for slicing")
                    # Try with literal 'time'
                    if 'time' in data2d.dims:
                        data2d = data2d.isel(time=slice(*time_lev))
                        self.logger.info(f"Selected time range {time_lev} from dimension 'time'")
            except (AttributeError, KeyError, IndexError) as e:
                self.logger.error(f"Error slicing time dimension: {e}")
                # Just use the data as is
                self.logger.warning("Using data without time slicing")

        # Apply averaging or selection based on specs
        if self.config_manager.spec_data and data_array.name in self.config_manager.spec_data:
            spec = self.config_manager.spec_data[data_array.name]
            if 'xtplot' in spec and 'mean_type' in spec['xtplot']:
                mean_type = spec['xtplot']['mean_type']
                self.logger.info(f"Averaging method: {mean_type}")
                
                if mean_type == 'point_sel':
                    # Select a single point
                    try:
                        xc = spec['xtplot']['point_sel'][0]
                        yc = spec['xtplot']['point_sel'][1]
                        
                        # Try with model-specific dimension names first
                        if xc_dim in data2d.coords and yc_dim in data2d.coords:
                            data2d = data2d.sel({xc_dim: xc, yc_dim: yc}, method='nearest')
                            self.logger.info(f"Selected point ({xc}, {yc}) using dimensions {xc_dim}, {yc_dim}")
                        else:
                            # Try with literal 'lon' and 'lat'
                            if 'lon' in data2d.coords and 'lat' in data2d.coords:
                                data2d = data2d.sel(lon=xc, lat=yc, method='nearest')
                                self.logger.info(f"Selected point ({xc}, {yc}) using dimensions 'lon', 'lat'")
                            else:
                                self.logger.error(f"Could not find coordinates for point selection")
                    except (KeyError, ValueError) as e:
                        self.logger.error(f"Error in point selection: {e}")
                        
                elif mean_type == 'area_sel':
                    # Select an area and compute mean
                    try:
                        x1 = spec['xtplot']['area_sel'][0]
                        x2 = spec['xtplot']['area_sel'][1]
                        y1 = spec['xtplot']['area_sel'][2]
                        y2 = spec['xtplot']['area_sel'][3]
                        
                        # Try with model-specific dimension names first
                        if xc_dim in data2d.coords and yc_dim in data2d.coords:
                            data2d = data2d.sel({
                                xc_dim: slice(x1, x2),
                                yc_dim: slice(y1, y2)
                            })
                            self.logger.info(f"Selected area ({x1}, {y1}) to ({x2}, {y2}) using dimensions {xc_dim}, {yc_dim}")
                            
                            # Compute mean over spatial dimensions
                            if xc_dim in data2d.dims and yc_dim in data2d.dims:
                                data2d = data2d.mean(dim=(xc_dim, yc_dim))
                                self.logger.info(f"Computed mean over dimensions {xc_dim}, {yc_dim}")
                        else:
                            # Try with literal 'lon' and 'lat'
                            if 'lon' in data2d.coords and 'lat' in data2d.coords:
                                data2d = data2d.sel(lon=slice(x1, x2), lat=slice(y1, y2))
                                self.logger.info(f"Selected area ({x1}, {y1}) to ({x2}, {y2}) using dimensions 'lon', 'lat'")
                                
                                # Compute mean over spatial dimensions
                                if 'lon' in data2d.dims and 'lat' in data2d.dims:
                                    data2d = data2d.mean(dim=('lon', 'lat'))
                                    self.logger.info(f"Computed mean over dimensions 'lon', 'lat'")
                            else:
                                self.logger.error(f"Could not find coordinates for area selection")
                    except (KeyError, ValueError) as e:
                        self.logger.error(f"Error in area selection: {e}")
                        
                elif mean_type in ['year', 'season', 'month']:
                    # Group by time period
                    try:
                        if tc_dim in data2d.dims:
                            time_attr = f"{tc_dim}.{mean_type}"
                            data2d = data2d.groupby(time_attr).mean(dim=tc_dim, keep_attrs=True)
                            self.logger.info(f"Grouped by {mean_type} using dimension {tc_dim}")
                        else:
                            # Try with literal 'time'
                            if 'time' in data2d.dims:
                                time_attr = f"time.{mean_type}"
                                data2d = data2d.groupby(time_attr).mean(dim='time', keep_attrs=True)
                                self.logger.info(f"Grouped by {mean_type} using dimension 'time'")
                            else:
                                self.logger.error(f"Could not find time dimension for grouping")
                    except (AttributeError, KeyError) as e:
                        self.logger.error(f"Error in time grouping: {e}")
                        
                elif mean_type == 'rolling':
                    # Apply rolling mean
                    try:
                        window_size = spec['xtplot'].get('window_size', 5)
                        self.logger.info(f" -- smoothing window size: {window_size}")
                        
                        if tc_dim in data2d.dims:
                            data2d = data2d.rolling({tc_dim: window_size}, center=True).mean()
                            self.logger.info(f"Applied rolling mean with window size {window_size} on dimension {tc_dim}")
                        else:
                            # Try with literal 'time'
                            if 'time' in data2d.dims:
                                data2d = data2d.rolling(time=window_size, center=True).mean()
                                self.logger.info(f"Applied rolling mean with window size {window_size} on dimension 'time'")
                            else:
                                self.logger.error(f"Could not find time dimension for rolling mean")
                    except (AttributeError, KeyError) as e:
                        self.logger.error(f"Error in rolling mean: {e}")
                        
                else:
                    # General mean over all dimensions except time
                    try:
                        # Get all dimensions except time
                        if tc_dim in data2d.dims:
                            non_time_dims = [dim for dim in data2d.dims if dim != tc_dim]
                            if non_time_dims:
                                data2d = data2d.mean(dim=non_time_dims)
                                self.logger.info(f"Computed mean over all non-time dimensions: {non_time_dims}")
                        else:
                            # Try with literal 'time'
                            if 'time' in data2d.dims:
                                non_time_dims = [dim for dim in data2d.dims if dim != 'time']
                                if non_time_dims:
                                    data2d = data2d.mean(dim=non_time_dims)
                                    self.logger.info(f"Computed mean over all non-time dimensions: {non_time_dims}")
                            else:
                                self.logger.error(f"Could not find time dimension for general mean")
                    except (AttributeError, KeyError) as e:
                        self.logger.error(f"Error in general mean: {e}")

            # Handle level selection if specified
            if 'xtplot' in spec and 'level' in spec['xtplot']:
                level = int(spec['xtplot']['level'])
                self.logger.info(f"Selecting level {level}")
                
                # Get vertical dimension safely
                if zc_dim and zc_dim in data2d.dims:
                    try:
                        # Try exact matching
                        if level in data2d[zc_dim].values:
                            lev_idx = np.where(data2d[zc_dim].values == level)[0][0]
                            data2d = data2d.isel({zc_dim: lev_idx}).squeeze()
                            self.logger.info(f"Selected exact level {level} at index {lev_idx}")
                        else:
                            # Try nearest neighbor
                            lev_idx = np.abs(data2d[zc_dim].values - level).argmin()
                            self.logger.warning(f"Level {level} not found exactly, using nearest level {data2d[zc_dim].values[lev_idx]}")
                            data2d = data2d.isel({zc_dim: lev_idx}).squeeze()
                    except (AttributeError, KeyError, IndexError) as e:
                        self.logger.error(f"Error selecting level {level}: {e}")
                        # If level selection fails, use the first level
                        if data2d[zc_dim].size > 0:
                            data2d = data2d.isel({zc_dim: 0}).squeeze()
                            self.logger.info("Falling back to first level")
                else:
                    # Try with literal 'lev' or 'level'
                    for lev_name in ['lev', 'level', 'plev']:
                        if lev_name in data2d.dims:
                            try:
                                if level in data2d[lev_name].values:
                                    lev_idx = np.where(data2d[lev_name].values == level)[0][0]
                                    data2d = data2d.isel({lev_name: lev_idx}).squeeze()
                                    self.logger.info(f"Selected exact level {level} at index {lev_idx} from dimension {lev_name}")
                                    break
                                else:
                                    # Try nearest neighbor
                                    lev_idx = np.abs(data2d[lev_name].values - level).argmin()
                                    self.logger.warning(f"Level {level} not found exactly, using nearest level {data2d[lev_name].values[lev_idx]}")
                                    data2d = data2d.isel({lev_name: lev_idx}).squeeze()
                                    break
                            except (AttributeError, KeyError, IndexError) as e:
                                self.logger.error(f"Error selecting level {level} from dimension {lev_name}: {e}")
                                # If level selection fails, use the first level
                                if data2d[lev_name].size > 0:
                                    data2d = data2d.isel({lev_name: 0}).squeeze()
                                    self.logger.info(f"Falling back to first level from dimension {lev_name}")
                                    break
                    else:
                        self.logger.warning(f"Level {level} specified but no vertical dimension found")

        # Check if we still have more than 1 dimension (excluding time)
        dims = list(data2d.dims)
        if len(dims) > 1:
            self.logger.warning(f"Data still has {len(dims)} dimensions after processing. Attempting to reduce to 1D time series.")
            
            # Find the time dimension
            time_dim = None
            for dim in dims:
                if dim == tc_dim or 'time' in dim.lower():
                    time_dim = dim
                    break
            
            if time_dim:
                # Average over all non-time dimensions
                non_time_dims = [dim for dim in dims if dim != time_dim]
                if non_time_dims:
                    self.logger.info(f"Averaging over non-time dimensions: {non_time_dims}")
                    data2d = data2d.mean(dim=non_time_dims)
            else:
                # If we can't identify the time dimension, use the first dimension and average over the rest
                self.logger.warning(f"Could not identify time dimension. Using first dimension: {dims[0]}")
                other_dims = dims[1:]
                if other_dims:
                    self.logger.info(f"Averaging over dimensions: {other_dims}")
                    data2d = data2d.mean(dim=other_dims)

        # Squeeze to remove any singleton dimensions
        data2d = data2d.squeeze()
        
        # Debug: Log final data stats
        self.logger.info(f"_get_xt output: shape={data2d.shape}, dims={data2d.dims}")
        self.logger.info(f"_get_xt output stats: min={data2d.min().values}, max={data2d.max().values}")

        # Check for NaN values
        if np.isnan(data2d.values).any():
            self.logger.warning(f"Output contains NaN values: {np.sum(np.isnan(data2d.values))} NaNs")

        return apply_conversion(self.config_manager, data2d, data_array.name)

    def _get_tx(self, data_array, level=None, time_lev=0):
        """ Extract a time-series map from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 2D Hovmoller plot field where time is plotted on one axis (default y-axis)
            and the spatial dimension (either lon or lat)) is plotted on the other axis  (default x-axis)
        """
        if data_array is None:
            return None

        # Debug: Log input data stats
        self.logger.info(f"_get_tx input: shape={data_array.shape}, dims={data_array.dims}")
        self.logger.info(f"_get_tx input stats: min={data_array.min().values}, max={data_array.max().values}")

        # Make a copy to avoid modifying the original
        data2d = data_array.squeeze()
        
        # Get dimension names from config_manager
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'
        xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
        yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'
        
        # Debug: Log dimension names
        self.logger.info(f"Dimension names: tc_dim={tc_dim}, zc_dim={zc_dim}, xc_dim={xc_dim}, yc_dim={yc_dim}")
        
        # Handle level selection if vertical dimension exists
        if zc_dim in data2d.dims:
            self.logger.info(f"Vertical dimension found: {zc_dim}, size={data2d[zc_dim].size}")
            if level is not None:
                # Try to select the specified level
                try:
                    if level in data2d[zc_dim].values:
                        lev_idx = np.where(data2d[zc_dim].values == level)[0][0]
                        data2d = data2d.isel({zc_dim: lev_idx})
                        self.logger.info(f"Selected exact level {level} at index {lev_idx}")
                    else:
                        # Try nearest neighbor
                        lev_idx = np.abs(data2d[zc_dim].values - level).argmin()
                        self.logger.warning(f"Level {level} not found exactly, using nearest level {data2d[zc_dim].values[lev_idx]}")
                        data2d = data2d.isel({zc_dim: lev_idx})
                except Exception as e:
                    self.logger.error(f"Error selecting level {level}: {e}")
                    # If level selection fails, use the first level
                    if data2d[zc_dim].size > 0:
                        data2d = data2d.isel({zc_dim: 0})
                        self.logger.info("Falling back to first level")
            else:
                # No level specified, use the first level
                if data2d[zc_dim].size > 0:
                    data2d = data2d.isel({zc_dim: 0})
                    self.logger.info("No level specified, using first level")
        elif level is not None:
            # If level is specified but there's no vertical dimension, log a warning
            self.logger.warning(f"Level {level} specified but no vertical dimension found in data. Using data as is.")

        # Apply any range selections from the specs
        if self.config_manager.spec_data and data_array.name in self.config_manager.spec_data:
            spec = self.config_manager.spec_data[data_array.name]
            if 'txplot' in spec:
                # Apply time range selection if specified
                if 'trange' in spec['txplot']:
                    start_time = spec['txplot']['trange'][0]
                    end_time = spec['txplot']['trange'][1]
                    try:
                        data2d = data2d.sel({tc_dim: slice(start_time, end_time)})
                        self.logger.info(f"Applied time range selection: {start_time} to {end_time}")
                    except Exception as e:
                        self.logger.error(f"Error applying time range selection: {e}")
                
                # Apply latitude range selection if specified
                if 'yrange' in spec['txplot']:
                    lat_min = spec['txplot']['yrange'][0]
                    lat_max = spec['txplot']['yrange'][1]
                    try:
                        data2d = data2d.sel({yc_dim: slice(lat_min, lat_max)})
                        self.logger.info(f"Applied latitude range selection: {lat_min} to {lat_max}")
                    except Exception as e:
                        self.logger.error(f"Error applying latitude range selection: {e}")
                
                # Apply longitude range selection if specified
                if 'xrange' in spec['txplot']:
                    lon_min = spec['txplot']['xrange'][0]
                    lon_max = spec['txplot']['xrange'][1]
                    try:
                        data2d = data2d.sel({xc_dim: slice(lon_min, lon_max)})
                        self.logger.info(f"Applied longitude range selection: {lon_min} to {lon_max}")
                    except Exception as e:
                        self.logger.error(f"Error applying longitude range selection: {e}")

        # Squeeze again to remove any singleton dimensions
        data2d = data2d.squeeze()
        self.logger.info(f"After selections and squeeze: shape={data2d.shape}, dims={data2d.dims}")

        # Check if we still have more than 2 dimensions
        if len(data2d.dims) > 2:
            self.logger.warning(f"Data still has {len(data2d.dims)} dimensions. Attempting to reduce to 2D.")
            
            # For Hovmoller plots, we typically want time and longitude
            dims = list(data2d.dims)
            
            # Try to identify time and longitude dimensions
            time_dim = None
            lon_dim = None
            
            for dim in dims:
                if dim == tc_dim or 'time' in dim.lower():
                    time_dim = dim
                elif dim == xc_dim or 'lon' in dim.lower():
                    lon_dim = dim
            
            if time_dim and lon_dim:
                # We found time and longitude dimensions, average over other dimensions
                for dim in dims:
                    if dim != time_dim and dim != lon_dim:
                        self.logger.info(f"Averaging over dimension: {dim}")
                        data2d = data2d.mean(dim=dim)
            else:
                # If we can't identify time and longitude, use the first two dimensions
                self.logger.warning(f"Could not identify time and longitude dimensions. Using first two dimensions: {dims[:2]}")
                
                # For each dimension beyond the first two, take the mean
                for dim in dims[2:]:
                    self.logger.info(f"Averaging over dimension: {dim}")
                    data2d = data2d.mean(dim=dim)
            
            # Squeeze again to remove any singleton dimensions
            data2d = data2d.squeeze()
            self.logger.info(f"After dimension reduction: shape={data2d.shape}, dims={data2d.dims}")

        # Compute weighted mean over latitude if latitude dimension exists
        if yc_dim in data2d.dims:
            try:
                # Get latitude weights (cosine of latitude in radians)
                weights = np.cos(np.deg2rad(data2d[yc_dim].values))
                
                # Make sure weights have the right shape for broadcasting
                # Create a weights array with the same shape as the data
                weight_array = xr.ones_like(data2d)
                
                # Apply weights along the latitude dimension
                weighted_data = data2d * weight_array * weights
                
                # Sum over latitude and normalize by the sum of weights
                data2d = weighted_data.sum(dim=yc_dim) / weights.sum()
                
                self.logger.info(f"Applied latitude weighting. New shape: {data2d.shape}")
            except Exception as e:
                self.logger.error(f"Error applying latitude weighting: {e}")
                self.logger.warning("Falling back to simple mean over latitude")
                if yc_dim in data2d.dims:
                    data2d = data2d.mean(dim=yc_dim)

        # Debug: Log final data stats
        self.logger.info(f"_get_tx output: shape={data2d.shape}, dims={data2d.dims}")
        self.logger.info(f"_get_tx output stats: min={data2d.min().values}, max={data2d.max().values}")

        # Check for NaN values
        if np.isnan(data2d.values).any():
            self.logger.warning(f"Output contains NaN values: {np.sum(np.isnan(data2d.values))} NaNs")

        return apply_conversion(self.config_manager, data2d, data_array.name)

    def _select_yrange(self, data2d, name):
        """ Select a range of vertical levels"""
        if 'zrange' in self.config_manager.spec_data[name]['yzplot']:
            if not self.config_manager.spec_data[name]['yzplot']['zrange']:
                return data2d
            lo_z = self.config_manager.spec_data[name]['yzplot']['zrange'][0]
            hi_z = self.config_manager.spec_data[name]['yzplot']['zrange'][1]
            if hi_z >= lo_z:
                self.logger.error(f"Upper level value ({hi_z}) must be less than low level value ({lo_z})")
                return data2d
            lev = self.config_manager.get_model_dim_name('zc')
            min_index, max_index = 0, len(data2d.coords[lev].values) - 1
            for k, v in enumerate(data2d.coords[lev]):
                if data2d.coords[lev].values[k] == lo_z:
                    min_index = k
            for k, v in enumerate(data2d.coords[lev]):
                if data2d.coords[lev].values[k] == hi_z:
                    max_index = k
            return data2d[min_index:max_index + 1, :]
        else:
            return data2d

    def _set_time_config(self, time_index, data_var):
        """Set time-related configuration values."""
        self.config_manager.time_level = time_index
        
        # Try to get the time coordinate safely
        try:
            # Check if 'time' is a coordinate in the DataArray
            if 'time' in data_var.coords:
                # Get the time value at the specified index
                if isinstance(time_index, int) and time_index < len(data_var.coords['time']):
                    real_time = data_var.coords['time'].values[time_index]
                    real_time_readable = pd.to_datetime(real_time).strftime('%Y-%m-%d %H')
                    self.config_manager.real_time = real_time_readable
                else:
                    self.logger.warning(f"Time index {time_index} out of bounds for time coordinate")
                    self.config_manager.real_time = f"Time level {time_index}"
            else:
                # If 'time' is not a coordinate, try to find a time-like coordinate
                time_coords = [coord for coord in data_var.coords if 'time' in coord.lower()]
                if time_coords:
                    time_coord = time_coords[0]
                    if isinstance(time_index, int) and time_index < len(data_var.coords[time_coord]):
                        real_time = data_var.coords[time_coord].values[time_index]
                        real_time_readable = pd.to_datetime(real_time).strftime('%Y-%m-%d %H')
                        self.config_manager.real_time = real_time_readable
                    else:
                        self.logger.warning(f"Time index {time_index} out of bounds for {time_coord} coordinate")
                        self.config_manager.real_time = f"Time level {time_index}"
                else:
                    # If no time-like coordinate is found, use a generic label
                    self.logger.warning("No time coordinate found in data")
                    self.config_manager.real_time = f"Time level {time_index}"
        except Exception as e:
            self.logger.warning(f"Error setting time config: {e}")
            self.config_manager.real_time = f"Time level {time_index}"

