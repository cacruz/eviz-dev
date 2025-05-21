from dataclasses import dataclass
import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.models.root import Root
from eviz.lib.data.utils import apply_conversion, apply_mean, apply_zsum
import eviz.lib.autoviz.utils as pu
from eviz.lib.autoviz.figure import Figure

warnings.filterwarnings("ignore")


@dataclass
class Gridded(Root):
    """
    The Gridded class provides specialized functionality for handling gridded Earth System Model (ESM) data.

    This class extends the Root implementation to work specifically with structured grid data formats
    commonly used in ESMs, including 2D (lat-lon), 3D (lat-lon-time or lat-lon-level), and 4D 
    (lat-lon-level-time) datasets. It implements methods for extracting, processing, and visualizing
    various slices and projections of gridded data, such as:

    - Horizontal (XY) slices at specific vertical levels or times
    - Vertical (YZ) slices (zonal means)
    - Time series (XT) at points or averaged over regions
    - Hovmöller diagrams (TX) showing time-longitude evolution

    Unlike the observation modules which may handle both gridded and unstructured data formats,
    this class is optimized specifically for regular grid structures with consistent coordinate
    systems. It provides specialized grid-aware operations including:

    - Vertical level selection and averaging
    - Zonal and meridional means
    - Grid cell area-weighted averaging
    - Time averaging and selection
    - Comparison and differencing between gridded datasets

    This class serves as a base for more specific ESM implementations while providing
    comprehensive functionality for the most common gridded data operations.
    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.processor = DataProcessor(self.config_manager)


    def add_data_source(self, file_path, data_source):
        """
        Add a data source to the model.
        
        This method is required by AbstractRoot but is now a no-op since data sources
        are managed by the pipeline. It's kept for backward compatibility.
        
        Args:
            file_path: Path to the data file
            data_source: The data source to add
        """
        self.logger.warning(
            "add_data_source is deprecated. Data sources are now managed by the pipeline.")
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
        self.logger.warning(
            "load_data_sources is deprecated. Data sources are now loaded by the ConfigurationAdapter.")
        # No need to do anything, as data sources are loaded by the ConfigurationAdapter

    # SIMPLE PLOTS METHODS (no SPECS file)
    # --------------------------------------------------------------------------
    def _simple_plots(self, plotter):
        map_params = self.config_manager.map_params
        field_num = 0
        self.config_manager.findex = 0
        for i in map_params.keys():
            
            field_name = map_params[i]['field']
            filename = map_params[i]['filename']
            file_index = self.config_manager.get_file_index(filename)
            data_source =  self.config_manager.pipeline.get_data_source(filename)
            if field_name not in data_source.dataset:
                continue
            self.config_manager.findex = file_index
            self.config_manager.pindex = field_num
            self.config_manager.axindex = 0
            field_data_array = data_source.dataset[field_name]

            for plot_type in map_params[i]['to_plot']:
                self.logger.info(f"Plotting {field_name}, {plot_type} plot")
                field_to_plot = self._get_field_for_simple_plot(field_data_array, field_name, plot_type)
                plotter.simple_plot(self.config_manager, field_to_plot)
            field_num += 1

    # Simple plots do not use configurations in SPECS file
    def _get_field_for_simple_plot(self, data_array: xr.DataArray, field_name: str,
                                   plot_type: str) -> tuple:
        """Prepare data for simple plots."""
        if data_array is None:
            return None
        data2d = None
        dim1_name, dim2_name = None, None

        if 'xy' in plot_type:
            dim1_name = self.config_manager.get_model_dim_name('xc')
            dim2_name = self.config_manager.get_model_dim_name('yc')
            data2d = self._get_xy_simple(data_array)
        elif 'yz' in plot_type:
            dim1_name = self.config_manager.get_model_dim_name('yc')
            dim2_name = self.config_manager.get_model_dim_name('zc')
            data2d = self._get_yz_simple(data_array)
        elif 'sc' in plot_type:
            # Assuming scatter plots use lat/lon
            dim1_name = self.config_manager.get_model_dim_name('xc')
            dim2_name = self.config_manager.get_model_dim_name('yc')
            data2d = data_array.squeeze()  # Scatter plots usually don't need slicing
        elif 'graph' in plot_type:
            # Assuming graph data is the DataArray itself
            data2d = data_array
            dim1_name, dim2_name = None, None  # Graph plots don't have standard dims
        else:
            self.logger.error(
                f'Plot type [{plot_type}] error: Either specify in SPECS file or create plot type.')
            return None

        if data2d is None:
            return None

        dim1_coords = data2d[
            dim1_name] if dim1_name and dim1_name in data2d.coords else None
        dim2_coords = data2d[
            dim2_name] if dim2_name and dim2_name in data2d.coords else None
        if dim1_coords is None or dim2_coords is None:
            self.logger.error(
                f"Could not find coordinates for field '{field_name}' with plot type '{plot_type}'. "
                f"dim1: {dim1_name}, dim2: {dim2_name}, data2d.dims: {data2d.dims}"
            )
            return None
        
        return data2d, dim1_coords, dim2_coords, field_name, plot_type

    def _get_xy_simple(self, data_array: xr.DataArray) -> xr.DataArray:
        """ Extract XY slice from N-dim data field"""
        if data_array is None:
            return
        data2d = data_array.squeeze()

        tc_dim = self.config_manager.get_model_dim_name('tc')
        if tc_dim and tc_dim in data2d.dims:
            data2d = data2d.mean(dim=tc_dim)
        
        zc_dim = self.config_manager.get_model_dim_name('zc')
        if zc_dim and zc_dim in data2d.dims:
            data2d = data2d.mean(dim=zc_dim)
        
        if len(data2d.shape) == 4:
            data2d = data2d.isel({tc_dim: 0})
        if len(data2d.shape) == 3:
            if tc_dim in data2d.dims:
                data2d = data2d.isel({tc_dim: 0})
            else:
                data2d = data2d.isel({zc_dim: 0})
        return data2d

    def _get_yz_simple(self, data_array: xr.DataArray) -> xr.DataArray:
        if data_array is None:
            return None
        data2d = data_array.squeeze()
        tc_dim = self.config_manager.get_model_dim_name('tc')
        if tc_dim in data2d.dims and data2d[tc_dim].size > 0:
            data2d = data2d.isel({tc_dim: 0})
        xc_dim = self.config_manager.get_model_dim_name('xc')
        if xc_dim in data2d.dims:
            data2d = data2d.mean(dim=xc_dim)
        return data2d


    # SINGLE PLOTS METHODS (using SPECS file)
    # --------------------------------------------------------------------------
    def _single_plots(self, plotter):
        """Generate single plots for each source and field according to configuration."""
        self.logger.info("Generating single plots")

        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for single plotting.")
            return

        # Iterate through map_params to generate plots
        for idx, params in self.config_manager.map_params.items():
            field_name = params.get('field')
            if not field_name:
                continue

            filename = params.get('filename')
            data_source = self.config_manager.pipeline.get_data_source(filename)

            if not data_source or not hasattr(data_source,
                                              'dataset') or data_source.dataset is None:
                continue

            if field_name not in data_source.dataset:
                continue

            self.config_manager.findex = idx  
            self.config_manager.pindex = idx
            self.config_manager.axindex = 0

            field_data_array = data_source.dataset[field_name]
            plot_types = params.get('to_plot', ['xy'])
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
            for plot_type in plot_types:
                self._process_plot(field_data_array, field_name, idx, plot_type, plotter)

        if self.config_manager.make_gif:
            pu.create_gif(self.config_manager.config)

    def _process_plot(self, data_array: xr.DataArray, field_name: str, file_index: int,
                      plot_type: str, plotter):
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

    def _process_xy_plot(self, data_array: xr.DataArray, field_name: str, file_index: int,
                         plot_type: str, figure, plotter):
        """Process xy or polar plot types."""
        levels = self.config_manager.get_levels(field_name, plot_type + 'plot')
        do_zsum = self.config_manager.ax_opts.get('zsum', False)  # Use .get with default

        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        num_times = data_array[tc_dim].size if tc_dim in data_array.dims else 1
        time_levels = range(num_times) if time_level_config == 'all' else [
            time_level_config]

        if not levels and not do_zsum:
            return

        if levels:
            self._process_level_plots(data_array, field_name, file_index, plot_type,
                                      figure, time_levels, levels, plotter)
        else:
            self._process_zsum_plots(data_array, field_name, file_index, plot_type,
                                     figure, time_levels, plotter)

    def _process_level_plots(self, data_array: xr.DataArray, field_name: str,
                             file_index: int, plot_type: str, figure,
                             time_levels: list, levels: dict, plotter):
        """Process plots for specific vertical levels."""
        self.logger.debug(f' -> Processing {len(time_levels)} time levels')
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'  # Default to 'lev'
        tc_dim = self.config_manager.get_model_dim_name(
            'tc') or 'time'  # Default to 'time'

        has_vertical_dim = zc_dim and zc_dim in data_array.dims

        for level_val in levels.keys():  # Iterate through level values
            self.config_manager.level = level_val
            for t in time_levels:
                if tc_dim in data_array.dims:
                    data_at_time = data_array.isel({tc_dim: t})
                else:
                    data_at_time = data_array.squeeze()  # Assume single time if no time dim

                self._set_time_config(t, data_at_time)

                # Create a new figure for each level to avoid reusing axes
                figure = Figure.create_eviz_figure(self.config_manager, plot_type)
                self.config_manager.ax_opts = figure.init_ax_opts(field_name)

                ax = figure.get_axes()

                # If the data doesn't have a vertical dimension, we can't select a level
                # In this case, we'll just use the data as is
                if not has_vertical_dim:
                    field_to_plot = self._get_field_to_plot(ax, data_at_time, field_name,
                                                            file_index, plot_type, figure,
                                                            t)
                else:
                    field_to_plot = self._get_field_to_plot(ax, data_at_time, field_name,
                                                            file_index, plot_type, figure,
                                                            t,
                                                            level=level_val)

                if field_to_plot:
                    plotter.single_plots(self.config_manager, field_to_plot=field_to_plot,
                                         level=level_val)

                    pu.print_map(self.config_manager, plot_type,
                                 self.config_manager.findex, figure,
                                 level=level_val)

    def _process_other_plot(self, data_array: xr.DataArray, field_name: str,
                            file_index: int, plot_type: str, figure,
                            plotter):
        """Process non-xy and non-po plot types."""
        self.config_manager.level = None
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)  # Default to 0
        tc_dim = self.config_manager.get_model_dim_name(
            'tc') or 'time'  # Default to 'time'

        if tc_dim in data_array.dims:
            num_times = data_array[tc_dim].size
            # TODO: Handle yx_plot Gifs
            time_levels = range(num_times) if time_level_config == 'all' else [
                time_level_config]
        else:
            time_levels = [0]

        ax = figure.get_axes()
        # Assuming these plot types (xt, tx) might not need time slicing here,
        # or slicing is handled within _get_field_to_plot
        # Pass the full data_array and let _get_field_to_plot handle slicing if needed
        field_to_plot = self._get_field_to_plot(ax, data_array, field_name, file_index,
                                                plot_type, figure,
                                                time_level=time_level_config)
        if field_to_plot:
            plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex,
                         figure)

    def _process_zsum_plots(self, data_array: xr.DataArray, field_name: str,
                            file_index: int, plot_type: str, figure,
                            time_levels: list, plotter):
        """Process plots with vertical summation."""
        self.config_manager.level = None
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'  
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'

        if not zc_dim or zc_dim not in data_array.dims:
            data_array = data_array.squeeze()

        for t in time_levels:
            if tc_dim in data_array.dims:
                data_at_time = data_array.isel({tc_dim: t})
            else:
                data_at_time = data_array.squeeze()  # Assume single time if no time dim

            self._set_time_config(t, data_at_time)
            field_to_plot = self._get_field_to_plot(None, data_at_time, field_name,
                                                    # Pass None for ax initially
                                                    file_index, plot_type, figure, t)
            if field_to_plot:
                plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
                pu.print_map(self.config_manager, plot_type, self.config_manager.findex,
                             figure)

    def _get_field_to_plot(self, ax, data_array: xr.DataArray, field_name: str,
                           file_index: int, plot_type: str, figure, time_level,
                           level=None) -> tuple:
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
            return None

        if data2d is None:
            self.logger.error(
                f"Failed to prepare 2D data for field {field_name}, plot type {plot_type}")
            return None

        x_values = None
        y_values = None
        if 'xt' in plot_type or 'tx' in plot_type:
            # For time-series or Hovmoller plots, coordinates are handled differently
            # The plotter functions for these types will need to extract them from data2d
            pass
        else:
            try:
                # Use the determined dimension names
                if dim1_name and dim1_name in data2d.coords:
                    x_values = data2d[dim1_name]
                else:
                    self.logger.debug(
                        f"Dimension '{dim1_name}' not found in data coordinates for {field_name}")

                if dim2_name and dim2_name in data2d.coords:
                    y_values = data2d[dim2_name]
                else:
                    self.logger.debug(
                        f"Dimension '{dim2_name}' not found in data coordinates for {field_name}")

            except KeyError as e:
                self.logger.error(f"Error getting coordinates for {field_name}: {e}")
                dims = list(data2d.dims)
                if len(dims) >= 2:
                    self.logger.debug(
                        f"Falling back to using dimensions {dims[0]} and {dims[1]} as coordinates")
                    x_values = data2d[dims[0]]
                    y_values = data2d[dims[1]]
                else:
                    self.logger.error(
                        "Dataset has fewer than 2 dimensions, cannot plot spatial data")
                    return None

            if np.isnan(data2d.values).all():
                self.logger.error(
                    f"All values are NaN for {field_name}. Using original data.")
                data2d = data_array.squeeze()
            elif np.isnan(data2d.values).any():
                self.logger.debug(
                    f"Note: Some NaN values present ({np.sum(np.isnan(data2d.values))} NaNs).")
                # data2d = data2d.fillna(0)

        # Return the prepared data and coordinates in the expected tuple format
        return data2d, x_values, y_values, field_name, plot_type, file_index, figure, ax

    # COMPARE_DIFF METHODS (always need SPECS file)
    # --------------------------------------------------------------------------
    def _comparison_plots(self, plotter):
        """Generate comparison plots for paired data sources according to configuration."""
        self.logger.info("Generating comparison plots")
        current_field_index = 0

        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for comparison plotting.")
            return

        # Get the file indices for the two files being compared
        if not self.config_manager.a_list or not self.config_manager.b_list:
            self.logger.error("a_list or b_list is empty, cannot perform comparison.")
            return

        idx1 = self.config_manager.a_list[0]
        idx2 = self.config_manager.b_list[0]

        # Gather all unique field names from map_params for these files
        fields_file1 = [params['field'] for i, params in self.config_manager.map_params.items() if params['file_index'] == idx1]
        fields_file2 = [params['field'] for i, params in self.config_manager.map_params.items() if params['file_index'] == idx2]
        all_fields = set(fields_file1) & set(fields_file2)  # Only fields present in both

        self.logger.info(f"Comparing files {idx1} and {idx2}")
        self.logger.info(f"Fields in file 1: {fields_file1}")
        self.logger.info(f"Fields in file 2: {fields_file2}")
        self.logger.info(f"Fields to compare: {all_fields}")

        for field_name in all_fields:
            # Find map_params for this field in both files
            idx1_field = next((i for i, params in self.config_manager.map_params.items()
                            if params['file_index'] == idx1 and params['field'] == field_name), None)
            idx2_field = next((i for i, params in self.config_manager.map_params.items()
                            if params['file_index'] == idx2 and params['field'] == field_name), None)
            if idx1_field is None or idx2_field is None:
                continue

            map1_params = self.config_manager.map_params[idx1_field]
            map2_params = self.config_manager.map_params[idx2_field]

            filename1 = map1_params.get('filename')
            filename2 = map2_params.get('filename')

            data_source1 = self.config_manager.pipeline.get_data_source(filename1)
            data_source2 = self.config_manager.pipeline.get_data_source(filename2)

            if not data_source1 or not data_source2:
                continue

            sdat1_dataset = data_source1.dataset if hasattr(data_source1, 'dataset') else None
            sdat2_dataset = data_source2.dataset if hasattr(data_source2, 'dataset') else None

            if sdat1_dataset is None or sdat2_dataset is None:
                continue

            file_indices = (idx1_field, idx2_field)

            self.field_names = (field_name, field_name)

            # Assuming plot types are the same for comparison
            plot_types = map1_params.get('to_plot', ['xy'])
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
            for plot_type in plot_types:
                self.logger.info(
                    f"Plotting {field_name} vs {field_name}, {plot_type} plot")
                self.data2d_list = []  # Reset for each plot type

                if 'xy' in plot_type or 'po' in plot_type or 'polar' in plot_type:
                    self._process_xy_comparison_plots(plotter, file_indices,
                                                    current_field_index,
                                                    field_name, field_name, plot_type,
                                                    sdat1_dataset, sdat2_dataset)
                else:
                    self._process_other_comparison_plots(plotter, file_indices,
                                                        current_field_index,
                                                        field_name, field_name,
                                                        plot_type, sdat1_dataset,
                                                        sdat2_dataset)

            current_field_index += 1

    def _process_xy_comparison_plots(self, plotter, file_indices: tuple,
                                     current_field_index: int,
                                     field_name1: str, field_name2: str, plot_type: str,
                                     sdat1_dataset: xr.Dataset,
                                     sdat2_dataset: xr.Dataset):
        """Process comparison plots for xy or polar plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels

        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            return

        for level_val in levels.keys():
            figure = Figure.create_eviz_figure(self.config_manager, plot_type,
                                               nrows=nrows, ncols=ncols)
            ax = figure.get_axes()
            axes_shape = figure.get_gs_geometry()
            self.config_manager.level = level_val

            if axes_shape == (3, 1):
                self._create_3x1_comparison_plot(plotter, file_indices,
                                                 current_field_index,
                                                 field_name1, field_name2, figure, ax,
                                                 plot_type, sdat1_dataset, sdat2_dataset,
                                                 level_val)
            elif axes_shape == (2, 2):
                self._create_2x2_comparison_plot(plotter, file_indices,
                                                 current_field_index,
                                                 field_name1, field_name2, figure, ax,
                                                 plot_type, sdat1_dataset, sdat2_dataset,
                                                 level_val)

            self.config_manager.findex = file_index1
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex,
                         figure, level=level_val)
            self.comparison_plot = False  # Reset comparison flag

    def _process_other_comparison_plots(self, plotter, file_indices: tuple,
                                        current_field_index: int,
                                        field_name1: str, field_name2: str,
                                        plot_type: str,
                                        sdat1_dataset: xr.Dataset,
                                        sdat2_dataset: xr.Dataset):
        """Process comparison plots for other plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels

        figure = Figure.create_eviz_figure(self.config_manager, plot_type, nrows=nrows,
                                           ncols=ncols)
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

        self.config_manager.findex = file_index1
        pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)
        self.comparison_plot = False  # Reset comparison flag

    def _create_3x1_comparison_plot(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, figure, ax,
                                    plot_type, sdat1_dataset, sdat2_dataset, level=None):
        """Create a 3x1 comparison plot."""
        file_index1, file_index2 = file_indices

        # Plot the first dataset
        self._process_3x1_comparison_plot(plotter, file_index1, current_field_index,
                                          field_name1, figure, ax, 0,
                                          sdat1_dataset[field_name1], plot_type,
                                          level=level)

        # Plot the second dataset
        self._process_3x1_comparison_plot(plotter, file_index2, current_field_index,
                                          field_name2, figure, ax, 1,
                                          sdat2_dataset[field_name2], plot_type,
                                          level=level)

        # Plot the comparison (difference)
        self.comparison_plot = True
        # For the comparison, we need to pass both datasets
        # The _process_comparison_plot method will need to handle this special case
        self._process_3x1_comparison_plot(plotter, file_index1, current_field_index,
                                          field_name1, figure, ax, 2,
                                          (sdat1_dataset[field_name1],
                                           sdat2_dataset[field_name2]),
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
                                          (sdat1_dataset[field_name1],
                                           sdat2_dataset[field_name2]),
                                          plot_type, level=level)

        # If extra field type is enabled, plot another comparison view
        # if self.config_manager.ax_opts.get('add_extra_field_type', False):
        self._process_2x2_comparison_plot(plotter, file_index1, current_field_index,
                                            field_name1, figure, [1, 1], 2,
                                            (sdat1_dataset[field_name1],
                                            sdat2_dataset[field_name2]),
                                            plot_type, level=level)

    def _process_3x1_comparison_plot(self, plotter, file_index, current_field_index,
                                    field_name,
                                    figure, ax, ax_index, data_array, plot_type,
                                    level=None):
        """Process a comparison plot."""
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        if ax_index == 2:  # Third panel in 3x1 layout is the difference
            self.config_manager.ax_opts['is_diff_field'] = True

        figure.set_ax_opts_diff_field(ax[ax_index])
        
        if ax_index == 2:
            # Compute and plot the difference field
            if len(self.data2d_list) == 2:
                data2d1, data2d2 = self.data2d_list
                proc = self.processor
                dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
                self.logger.debug(f"Regridding {field_name} over {dim1_name} and {dim2_name} for difference plot")
                diff_result, diff_x, diff_y = proc.regrid(data2d1, data2d2, dim1_name, dim2_name)
                self.logger.debug(
                    f"Diff data min/max: {diff_result.min().values}/{diff_result.max().values}")
                if diff_result is None or diff_result[0] is None:
                    self.logger.error("Regridding failed, cannot create difference plot")
                    field_to_plot = None
                else:
                    field_to_plot = (diff_result, diff_x, diff_y, field_name, plot_type,
                                    file_index, figure, ax)
            else:
                self.logger.error("Not enough data for difference plot")
                field_to_plot = None
            # Reset for next plot
            self.data2d_list = []
        else:
            # For the first two panels, plot as usual and store data for diff
            field_to_plot = self._get_field_to_plot_compare(data_array, field_name,
                                                            file_index,
                                                            plot_type, figure,
                                                            level=level)
            if field_to_plot:
                self.data2d_list.append(field_to_plot[0])

        if field_to_plot:
            plotter.comparison_plots(self.config_manager, field_to_plot, level=level)


    def _process_2x2_comparison_plot(self, plotter, file_index, current_field_index,
                                     field_name,
                                     figure, gsi, ax_index, data_array, plot_type,
                                     level=None):
        ax = figure.get_axes()
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

        figure.set_ax_opts_diff_field(ax[ax_index])
        if isinstance(data_array, tuple):
            if len(self.data2d_list) == 2:
                data2d1, data2d2 = self.data2d_list
                proc = self.processor
                dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
                self.logger.debug(f"Regridding {field_name} over {dim1_name} and {dim2_name} for difference plot")
                diff_result, diff_x, diff_y = proc.regrid(data2d1, data2d2, dim1_name, dim2_name)
                self.logger.debug(
                    f"Diff data min/max: {diff_result.min().values}/{diff_result.max().values}")
                if diff_result is None or diff_result[0] is None:
                    self.logger.error("Regridding failed, cannot create difference plot")
                    field_to_plot = None
                else:
                    field_to_plot = (diff_result, diff_x, diff_y, field_name, plot_type,
                                    file_index, figure, ax)
            else:
                self.logger.error("Not enough data for difference plot")
                field_to_plot = None
        else:
            field_to_plot = self._get_field_to_plot_compare(data_array, field_name,
                                                            file_index,
                                                            plot_type, figure,
                                                            level=level)
            if field_to_plot:
                self.data2d_list.append(field_to_plot[0])

        if field_to_plot:
            plotter.comparison_plots(self.config_manager, field_to_plot, level=level)

    # SIDE-BY-SIDE COMPARE METHODS (always need SPECS file)
    # --------------------------------------------------------------------------
    def _side_by_side_plots(self, plotter):
        self.logger.info("Generating side-by-side comparison plots")
        current_field_index = 0
        self.data2d_list = []

        # Get the file indices for the two files being compared
        if not self.config_manager.a_list or not self.config_manager.b_list:
            self.logger.error("a_list or b_list is empty, cannot perform side-by-side comparison.")
            return

        idx1 = self.config_manager.a_list[0]
        idx2 = self.config_manager.b_list[0]

        # Gather all unique field names from map_params for these files
        fields_file1 = [params['field'] for i, params in self.config_manager.map_params.items() if params['file_index'] == idx1]
        fields_file2 = [params['field'] for i, params in self.config_manager.map_params.items() if params['file_index'] == idx2]
        all_fields = set(fields_file1) & set(fields_file2)  # Only fields present in both

        self.logger.debug(f"Comparing files {idx1} and {idx2}")
        self.logger.debug(f"Fields in file 1: {fields_file1}")
        self.logger.debug(f"Fields in file 2: {fields_file2}")
        self.logger.debug(f"Fields to compare: {all_fields}")

        for field_name in all_fields:
            # Find map_params for this field in both files
            idx1_field = next((i for i, params in self.config_manager.map_params.items()
                            if params['file_index'] == idx1 and params['field'] == field_name), None)
            idx2_field = next((i for i, params in self.config_manager.map_params.items()
                            if params['file_index'] == idx2 and params['field'] == field_name), None)
            if idx1_field is None or idx2_field is None:
                continue

            map1_params = self.config_manager.map_params[idx1_field]
            map2_params = self.config_manager.map_params[idx2_field]

            filename1 = map1_params.get('filename')
            filename2 = map2_params.get('filename')

            data_source1 = self.config_manager.pipeline.get_data_source(filename1)
            data_source2 = self.config_manager.pipeline.get_data_source(filename2)

            if not data_source1 or not data_source2:
                continue

            sdat1_dataset = data_source1.dataset if hasattr(data_source1, 'dataset') else None
            sdat2_dataset = data_source2.dataset if hasattr(data_source2, 'dataset') else None

            if sdat1_dataset is None or sdat2_dataset is None:
                continue

            file_indices = (idx1_field, idx2_field)

            self.field_names = (field_name, field_name)

            plot_types = map1_params.get('to_plot', ['xy'])
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
            for plot_type in plot_types:
                self.data2d_list = []
                self.logger.info(f"Plotting {field_name} vs {field_name} , {plot_type} plot")

                if 'xy' in plot_type or 'polar' in plot_type:
                    self._process_xy_side_by_side_plots(plotter, file_indices,
                                                        current_field_index,
                                                        field_name, field_name,
                                                        plot_type,
                                                        sdat1_dataset, sdat2_dataset)
                else:
                    self._process_other_side_by_side_plots(plotter, file_indices,
                                                        current_field_index,
                                                        field_name, field_name,
                                                        plot_type, sdat1_dataset,
                                                        sdat2_dataset)

            current_field_index += 1

    def _process_xy_side_by_side_plots(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, plot_type, sdat1_dataset,
                                    sdat2_dataset):
        """Process side-by-side comparison plots for xy or polar plot types."""
        num_plots = len(self.config_manager.compare_exp_ids)
        nrows = 1
        ncols = num_plots

        # Get levels for the plots
        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            return

        for level_val in levels.keys():
            # Create figure with appropriate number of subplots
            figure = Figure.create_eviz_figure(self.config_manager, plot_type,
                                            nrows=nrows, ncols=ncols)
            ax = figure.get_axes()
            self.config_manager.level = level_val

            # Store domain information for regional plots if available
            is_regional = hasattr(self, 'source_name') and self.source_name in ['lis', 'wrf']
            if is_regional:
                # For regional domains, store coordinates for consistent plotting
                if hasattr(sdat1_dataset, 'lon') and hasattr(sdat1_dataset, 'lat'):
                    self.lon = sdat1_dataset.lon
                    self.lat = sdat1_dataset.lat

            # Create the side-by-side plot
            self._create_xy_side_by_side_plot(plotter, file_indices,
                                            current_field_index,
                                            field_name1, field_name2, figure, ax,
                                            plot_type, sdat1_dataset, sdat2_dataset,
                                            level_val)

            # Print the map with appropriate level information
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex,
                    figure, level=level_val)

    def _process_xy_side_by_side_plots_old(self, plotter, file_indices, current_field_index,
                                       field_name1, field_name2, plot_type, sdat1_dataset,
                                       sdat2_dataset):
        """Process side-by-side comparison plots for xy or polar plot types."""
        num_plots = len(self.config_manager.compare_exp_ids)

        nrows = 1
        ncols = num_plots  # This will be 3 for three variables

        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            return

        for level_val in levels.keys():
            figure = Figure.create_eviz_figure(self.config_manager, plot_type,
                                               nrows=nrows, ncols=ncols)
            ax = figure.get_axes()
            self.config_manager.level = level_val

            # This is needed for regional plots later on
            self.lon = sdat1_dataset.lon
            self.lat = sdat1_dataset.lat

            self._create_xy_side_by_side_plot(plotter, file_indices,
                                              current_field_index,
                                              field_name1, field_name2, figure, ax,
                                              plot_type, sdat1_dataset, sdat2_dataset,
                                              level_val)

            pu.print_map(self.config_manager, plot_type, self.config_manager.findex,
                         figure, level=level_val)

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
        # Ensure ax is a list with enough elements for all plots
        if not isinstance(ax, list):
            ax = [ax]
        num_plots = len(self.config_manager.compare_exp_ids)
        if len(ax) < num_plots:
            self.logger.debug(
                f"Not enough axes for {num_plots}-way comparison. Using the first axis.")
            ax = [ax[0]] * num_plots

        self.comparison_plot = False

        # Plot first dataset (from a_list)
        if self.config_manager.a_list:
            self._process_side_by_side_plot(plotter, self.config_manager.a_list[0],
                                            current_field_index,
                                            field_name1, figure, 0,
                                            sdat1_dataset[field_name1], plot_type,
                                            level=level)

        # Plot remaining datasets (from b_list)
        for i, file_idx in enumerate(self.config_manager.b_list, start=1):
            if i < num_plots:  # Only plot if we have a corresponding axis
                self._process_side_by_side_plot(plotter, file_idx,
                                                current_field_index,
                                                field_name2, figure, i,
                                                sdat2_dataset[field_name2], plot_type,
                                                level=level)

    def _process_other_side_by_side_plots(self, plotter, file_indices,
                                          current_field_index,
                                          field_name1, field_name2, plot_type,
                                          sdat1_dataset, sdat2_dataset):
        """Process side-by-side comparison plots for other plot types."""
        nrows, ncols = self.config_manager.input_config._comp_panels

        figure = Figure.create_eviz_figure(self.config_manager, plot_type, nrows=nrows,
                                           ncols=ncols)
        self.config_manager.level = None

        self._create_other_side_by_side_plot(plotter, file_indices, current_field_index,
                                             field_name1, field_name2, figure,
                                             plot_type, sdat1_dataset, sdat2_dataset)

        pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def _create_other_side_by_side_plot(self, plotter, file_indices, current_field_index,
                                        field_name1, field_name2, figure,
                                        plot_type, sdat1_dataset, sdat2_dataset,
                                        level=None):
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
        self._process_side_by_side_plot(plotter, file_index1, current_field_index,
                                        field_name1,
                                        figure, 0, sdat1_dataset[field_name1],
                                        plot_type, level=level)

        # Plot the second dataset in the right subplot
        self._process_side_by_side_plot(plotter, file_index2, current_field_index,
                                        field_name2,
                                        figure, 1, sdat2_dataset[field_name2],
                                        plot_type, level=level)

    def _process_side_by_side_plot(self, plotter, file_index, current_field_index,
                                   field_name,
                                   figure, ax_index, data_array, plot_type,
                                   level=None):
        """Process a single plot for side-by-side comparison."""
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        field_to_plot = self._get_field_to_plot_compare(data_array, field_name,
                                                        file_index,
                                                        plot_type, figure,
                                                        level=level)

        if field_to_plot and field_to_plot[0] is not None:
            self.data2d_list.append(field_to_plot[0])

        if field_to_plot:
            if hasattr(plotter, 'single_plots'):
                plotter.single_plots(self.config_manager, field_to_plot, level=level)
            elif hasattr(plotter, 'comparison_plots'):
                plotter.comparison_plots(self.config_manager, field_to_plot, level=level)
            else:
                # Unknown plotter type
                if hasattr(plotter, 'plot'):
                    plotter.plot(self.config_manager, field_to_plot, level=level)
                else:
                    self.logger.error(
                        f"Plotter {type(plotter).__name__} has no plot method.")

    def _get_field_to_plot_compare(self, data_array, field_name, file_index, plot_type, figure, level=None) -> tuple:
        """Prepare data for comparison plots, handling both global and regional domains."""
        dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
        data2d = None

        ax = figure.get_axes()
        # Handle comparison plots layout
        if figure.get_gs_geometry() == (1, 2) or figure.get_gs_geometry() == (1, 3):
            ax = ax[self.config_manager.axindex]

        # Handle difference field for comparison plots
        if self.config_manager.ax_opts.get('is_diff_field', False) and len(self.data2d_list) >= 2:
            proc = self.processor
            data2d, x, y = proc.regrid(plot_type)
            return data2d, x, y, self.field_names[0], plot_type, file_index, figure, ax

        # Process single plots based on plot type
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

        # For time series plots, return without coordinates
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax

        # Process coordinates based on domain type
        try:
            # Determine if this is a regional domain
            is_regional = hasattr(self, 'source_name') and self.source_name in ['lis', 'wrf']
            if is_regional:
                # Handle regional domain coordinates (LIS/WRF specific)
                if hasattr(self, '_process_coordinates'):
                    # Use model-specific coordinate processing if available
                    return self._process_coordinates(data2d, dim1_name, dim2_name, field_name, 
                                                plot_type, file_index, figure, ax)
                else:
                    # Fallback for regional domains without specific processing
                    xs = np.array(self._get_field(dim1_name, data2d)[0, :])
                    ys = np.array(self._get_field(dim2_name, data2d)[:, 0])
                    
                    # Calculate domain extent
                    latN = max(ys[:])
                    latS = min(ys[:])
                    lonW = min(xs[:])
                    lonE = max(xs[:])
                    
                    # Set plot extent and central coordinates
                    self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
                    self.config_manager.ax_opts['central_lon'] = np.mean([lonW, lonE])
                    self.config_manager.ax_opts['central_lat'] = np.mean([latS, latN])
                    
                    return data2d, xs, ys, field_name, plot_type, file_index, figure, ax
            else:
                # Handle global domain coordinates
                x = data2d[dim1_name].values if dim1_name in data2d.coords else None
                y = data2d[dim2_name].values if dim2_name in data2d.coords else None
                
                if x is None or y is None:
                    # Fallback to using dimensions if coordinates are not available
                    dims = list(data2d.dims)
                    if len(dims) >= 2:
                        x = data2d[dims[0]].values
                        y = data2d[dims[1]].values
                    else:
                        self.logger.error("Dataset has fewer than 2 dimensions, cannot plot")
                        return None
                
                return data2d, x, y, field_name, plot_type, file_index, figure, ax

        except Exception as e:
            self.logger.error(f"Error processing coordinates for {field_name}: {e}")
            return None

    # DATA SLICE PROCESSING METHODS
    # --------------------------------------------------------------------------
    def _get_yz(self, data_array, time_lev):
        """ Extract YZ slice (zonal mean) from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 2D (lat, lev) slice
        """
        if data_array is None:
            return None

        xc_dim = self.config_manager.get_model_dim_name('xc')
        tc_dim = self.config_manager.get_model_dim_name('tc')
        zc_dim = self.config_manager.get_model_dim_name('zc')

        if not zc_dim or zc_dim not in data_array.dims:
            self.logger.error(
                f"Cannot create YZ plot: no vertical dimension found in data for {data_array.name}")
            return None

        if xc_dim and xc_dim in data_array.dims:
            zonal_mean = data_array.mean(dim=xc_dim)
        else:
            self.logger.error(
                f"Could not find any longitude dimension for zonal mean in {data_array.name}")
            return None

        zonal_mean.attrs = data_array.attrs.copy()

        if tc_dim and tc_dim in zonal_mean.dims:
            num_times = zonal_mean[tc_dim].size
            if self.config_manager.ax_opts.get('tave', False) and num_times > 1:
                self.logger.debug(f"Averaging over {num_times} time levels.")
                zonal_mean = apply_mean(self.config_manager, zonal_mean)
            else:
                if isinstance(time_lev, int) and time_lev < num_times:
                    zonal_mean = zonal_mean.isel({tc_dim: time_lev})
                else:
                    zonal_mean = zonal_mean.isel({tc_dim: 0})
        else:
            zonal_mean = zonal_mean.squeeze()

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

        tc_dim = self.config_manager.get_model_dim_name('tc')
        zc_dim = self.config_manager.get_model_dim_name('zc')

        d_temp = data_array.copy()

        if tc_dim and tc_dim in d_temp.dims:
            num_tc = d_temp[tc_dim].size
            if isinstance(time_lev, int) and time_lev < num_tc:
                d_temp = d_temp.isel({tc_dim: time_lev})
            else:
                d_temp = d_temp.isel({tc_dim: 0})
        else:
            self.logger.debug(f"No time dimension found matching {tc_dim}")

        has_vertical_dim = zc_dim and zc_dim in d_temp.dims
        if has_vertical_dim:
            if level is not None:
                try:
                    # First try exact matching
                    if level in d_temp[zc_dim].values:
                        lev_idx = np.where(d_temp[zc_dim].values == level)[0][0]
                        d_temp = d_temp.isel({zc_dim: lev_idx})
                        self.logger.debug(
                            f"Selected exact level {level} at index {lev_idx}")
                    else:
                        # Try nearest neighbor
                        lev_idx = np.abs(d_temp[zc_dim].values - level).argmin()
                        self.logger.debug(
                            f"Level {level} not found exactly, using nearest level {d_temp[zc_dim].values[lev_idx]}")
                        d_temp = d_temp.isel({zc_dim: lev_idx})
                except Exception as e:
                    self.logger.error(f"Error selecting level {level}: {e}")
                    if d_temp[zc_dim].size > 0:
                        d_temp = d_temp.isel({zc_dim: 0})
            else:
                # No level specified, use the first level
                if d_temp[zc_dim].size > 0:
                    d_temp = d_temp.isel({zc_dim: 0})
                    self.logger.debug("No level specified, using first level")
        elif level is not None:
            self.logger.debug(
                f"Level {level} specified but no vertical dimension found in data. Using data as is.")

        data2d = d_temp.squeeze()

        if len(data2d.dims) > 2:
            xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
            yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'

            if xc_dim in data2d.dims and yc_dim in data2d.dims:
                for dim in data2d.dims:
                    if dim != xc_dim and dim != yc_dim:
                        data2d = data2d.isel({dim: 0})
            else:
                dims = list(data2d.dims)
                for dim in dims[2:]:
                    data2d = data2d.isel({dim: 0})

            data2d = data2d.squeeze()

            if len(data2d.dims) > 2:
                self.logger.debug(
                    f"Data still has {len(data2d.dims)} dimensions. Reshaping to 2D.")
                dims = list(data2d.dims)
                dim1, dim2 = dims[0], dims[1]

                for dim in dims[2:]:
                    data2d = data2d.mean(dim=dim)

        if tc_dim and tc_dim in data2d.dims and self.config_manager.ax_opts.get('tave',
                                                                                False):
            num_tc = data2d[tc_dim].size
            if num_tc > 1:
                self.logger.debug(f"Averaging over {num_tc} time levels.")
                data2d = apply_mean(self.config_manager, data2d, level)
                return apply_conversion(self.config_manager, data2d, data_array.name)

        if self.config_manager.ax_opts.get('zave', False):
            self.logger.debug("Averaging over vertical levels.")
            data2d = apply_mean(self.config_manager, data2d, level='all')
            return apply_conversion(self.config_manager, data2d, data_array.name)

        if self.config_manager.ax_opts.get('zsum', False):
            self.logger.debug("Summing over vertical levels.")
            data2d_zsum = apply_zsum(self.config_manager, data2d)
            self.logger.debug(
                "Min: {data2d_zsum.min().values}, Max: {data2d_zsum.max().values}")
            return apply_conversion(self.config_manager, data2d_zsum, data_array.name)

        if np.isnan(data2d.values).any():
            self.logger.debug(
                f"Output contains NaN values: {np.sum(np.isnan(data2d.values))} NaNs")

        return apply_conversion(self.config_manager, data2d, data_array.name)

    def _get_xt(self, data_array, time_lev):
        """ Extract time-series from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 1D (time) series
        """
        if data_array is None:
            return None

        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'
        xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
        yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'

        try:
            if tc_dim in data_array.dims:
                num_times = data_array[tc_dim].size
            else:
                if 'time' in data_array.dims:
                    num_times = data_array.time.size
                    tc_dim = 'time'
                else:
                    time_dims = [dim for dim in data_array.dims if 'time' in dim.lower()]
                    if time_dims:
                        tc_dim = time_dims[0]
                        num_times = data_array[tc_dim].size
                    else:
                        if hasattr(data_array, 'shape') and len(data_array.shape) > 0:
                            num_times = data_array.shape[
                                0]  # Assume time is the first dimension
                        else:
                            self.logger.error(
                                f"Cannot determine time dimension for {data_array.name}")
                            return None
        except (AttributeError, KeyError) as e:
            self.logger.error(f"Error determining time dimension: {e}")
            if hasattr(data_array, 'shape') and len(data_array.shape) > 0:
                num_times = data_array.shape[0]  # Assume time is the first dimension
            else:
                self.logger.error(
                    f"Cannot determine time dimension for {data_array.name}")
                return None

        self.logger.debug(f"'{data_array.name}' field has {num_times} time levels")

        data2d = data_array.copy()

        if isinstance(time_lev, list):
            self.logger.debug(f"Computing time series on {time_lev} time range")
            try:
                if tc_dim in data2d.dims:
                    data2d = data2d.isel({tc_dim: slice(*time_lev)})
                else:
                    if 'time' in data2d.dims:
                        data2d = data2d.isel(time=slice(*time_lev))
            except (AttributeError, KeyError, IndexError) as e:
                self.logger.error(f"Error slicing time dimension: {e}")

        # Apply averaging or selection based on specs
        if self.config_manager.spec_data and data_array.name in self.config_manager.spec_data:
            spec = self.config_manager.spec_data[data_array.name]
            if 'xtplot' in spec and 'mean_type' in spec['xtplot']:
                mean_type = spec['xtplot']['mean_type']
                self.logger.debug(f"Averaging method: {mean_type}")

                if mean_type == 'point_sel':
                    # Select a single point
                    try:
                        xc = spec['xtplot']['point_sel'][0]
                        yc = spec['xtplot']['point_sel'][1]

                        if xc_dim in data2d.coords and yc_dim in data2d.coords:
                            data2d = data2d.sel({xc_dim: xc, yc_dim: yc},
                                                method='nearest')
                        else:
                            if 'lon' in data2d.coords and 'lat' in data2d.coords:
                                data2d = data2d.sel(lon=xc, lat=yc, method='nearest')
                            else:
                                self.logger.error(
                                    "Could not find coordinates for point selection")
                    except (KeyError, ValueError) as e:
                        self.logger.error(f"Error in point selection: {e}")

                elif mean_type == 'area_sel':
                    # Select an area and compute mean
                    try:
                        x1 = spec['xtplot']['area_sel'][0]
                        x2 = spec['xtplot']['area_sel'][1]
                        y1 = spec['xtplot']['area_sel'][2]
                        y2 = spec['xtplot']['area_sel'][3]

                        if xc_dim in data2d.coords and yc_dim in data2d.coords:
                            data2d = data2d.sel({
                                xc_dim: slice(x1, x2),
                                yc_dim: slice(y1, y2)
                            })

                            if xc_dim in data2d.dims and yc_dim in data2d.dims:
                                data2d = data2d.mean(dim=(xc_dim, yc_dim))
                        else:
                            if 'lon' in data2d.coords and 'lat' in data2d.coords:
                                data2d = data2d.sel(lon=slice(x1, x2), lat=slice(y1, y2))

                                # Compute mean over spatial dimensions
                                if 'lon' in data2d.dims and 'lat' in data2d.dims:
                                    data2d = data2d.mean(dim=('lon', 'lat'))
                            else:
                                self.logger.error(
                                    "Could not find coordinates for area selection")
                    except (KeyError, ValueError) as e:
                        self.logger.error(f"Error in area selection: {e}")

                elif mean_type in ['year', 'season', 'month']:
                    # Group by time period
                    try:
                        if tc_dim in data2d.dims:
                            time_attr = f"{tc_dim}.{mean_type}"
                            data2d = data2d.groupby(time_attr).mean(dim=tc_dim,
                                                                    keep_attrs=True)
                        else:
                            if 'time' in data2d.dims:
                                time_attr = f"time.{mean_type}"
                                data2d = data2d.groupby(time_attr).mean(dim='time',
                                                                        keep_attrs=True)
                            else:
                                self.logger.error(
                                    "Could not find time dimension for grouping")
                    except (AttributeError, KeyError) as e:
                        self.logger.error(f"Error in time grouping: {e}")

                elif mean_type == 'rolling':
                    # Apply rolling mean
                    try:
                        window_size = spec['xtplot'].get('window_size', 5)
                        self.logger.debug(f" -- smoothing window size: {window_size}")

                        if tc_dim in data2d.dims:
                            data2d = data2d.rolling({tc_dim: window_size},
                                                    center=True).mean()
                        else:
                            if 'time' in data2d.dims:
                                data2d = data2d.rolling(time=window_size,
                                                        center=True).mean()
                            else:
                                self.logger.error(
                                    "Could not find time dimension for rolling mean")
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
                        else:
                            if 'time' in data2d.dims:
                                non_time_dims = [dim for dim in data2d.dims if
                                                 dim != 'time']
                                if non_time_dims:
                                    data2d = data2d.mean(dim=non_time_dims)
                            else:
                                self.logger.error(
                                    "Could not find time dimension for general mean")
                    except (AttributeError, KeyError) as e:
                        self.logger.error(f"Error in general mean: {e}")

            if 'xtplot' in spec and 'level' in spec['xtplot']:
                level = int(spec['xtplot']['level'])
                self.logger.debug(f"Selecting level {level}")

                if zc_dim and zc_dim in data2d.dims:
                    try:
                        # Try exact matching
                        if level in data2d[zc_dim].values:
                            lev_idx = np.where(data2d[zc_dim].values == level)[0][0]
                            data2d = data2d.isel({zc_dim: lev_idx}).squeeze()
                            self.logger.debug(
                                f"Selected exact level {level} at index {lev_idx}")
                        else:
                            # Try nearest neighbor
                            lev_idx = np.abs(data2d[zc_dim].values - level).argmin()
                            self.logger.debug(
                                f"Level {level} not found exactly, using nearest level {data2d[zc_dim].values[lev_idx]}")
                            data2d = data2d.isel({zc_dim: lev_idx}).squeeze()
                    except (AttributeError, KeyError, IndexError) as e:
                        self.logger.error(f"Error selecting level {level}: {e}")
                        if data2d[zc_dim].size > 0:
                            data2d = data2d.isel({zc_dim: 0}).squeeze()
                else:
                    for lev_name in ['lev', 'level', 'plev']:
                        if lev_name in data2d.dims:
                            try:
                                if level in data2d[lev_name].values:
                                    lev_idx = \
                                    np.where(data2d[lev_name].values == level)[0][0]
                                    data2d = data2d.isel({lev_name: lev_idx}).squeeze()
                                    break
                                else:
                                    lev_idx = np.abs(
                                        data2d[lev_name].values - level).argmin()
                                    self.logger.debug(
                                        f"Level {level} not found exactly, using nearest level {data2d[lev_name].values[lev_idx]}")
                                    data2d = data2d.isel({lev_name: lev_idx}).squeeze()
                                    break
                            except (AttributeError, KeyError, IndexError) as e:
                                self.logger.error(
                                    f"Error selecting level {level} from dimension {lev_name}: {e}")
                                if data2d[lev_name].size > 0:
                                    data2d = data2d.isel({lev_name: 0}).squeeze()
                                    break
                    else:
                        self.logger.debug(
                            f"Level {level} specified but no vertical dimension found")

        dims = list(data2d.dims)
        if len(dims) > 1:
            time_dim = None
            for dim in dims:
                if dim == tc_dim or 'time' in dim.lower():
                    time_dim = dim
                    break

            if time_dim:
                non_time_dims = [dim for dim in dims if dim != time_dim]
                if non_time_dims:
                    self.logger.info(
                        f"Averaging over non-time dimensions: {non_time_dims}")
                    data2d = data2d.mean(dim=non_time_dims)
            else:
                # Could not identify time dimension. Using first dimension
                other_dims = dims[1:]
                if other_dims:
                    data2d = data2d.mean(dim=other_dims)

        data2d = data2d.squeeze()

        if np.isnan(data2d.values).any():
            self.logger.debug(
                f"Output contains NaN values: {np.sum(np.isnan(data2d.values))} NaNs")

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

        data2d = data_array.squeeze()

        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'
        xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
        yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'

        if zc_dim in data2d.dims:
            if level is not None:
                # Try to select the specified level
                try:
                    if level in data2d[zc_dim].values:
                        lev_idx = np.where(data2d[zc_dim].values == level)[0][0]
                        data2d = data2d.isel({zc_dim: lev_idx})
                    else:
                        # Try nearest neighbor
                        lev_idx = np.abs(data2d[zc_dim].values - level).argmin()
                        self.logger.debug(
                            f"Level {level} not found exactly, using nearest level {data2d[zc_dim].values[lev_idx]}")
                        data2d = data2d.isel({zc_dim: lev_idx})
                except Exception as e:
                    self.logger.error(f"Error selecting level {level}: {e}")
                    if data2d[zc_dim].size > 0:
                        data2d = data2d.isel({zc_dim: 0})
            else:
                if data2d[zc_dim].size > 0:
                    data2d = data2d.isel({zc_dim: 0})
        elif level is not None:
            self.logger.debug(
                f"Level {level} specified but no vertical dimension found in data. Using data as is.")

        if self.config_manager.spec_data and data_array.name in self.config_manager.spec_data:
            spec = self.config_manager.spec_data[data_array.name]
            if 'txplot' in spec:
                if 'trange' in spec['txplot']:
                    start_time = spec['txplot']['trange'][0]
                    end_time = spec['txplot']['trange'][1]
                    try:
                        data2d = data2d.sel({tc_dim: slice(start_time, end_time)})
                    except Exception as e:
                        self.logger.error(f"Error applying time range selection: {e}")

                if 'yrange' in spec['txplot']:
                    lat_min = spec['txplot']['yrange'][0]
                    lat_max = spec['txplot']['yrange'][1]
                    try:
                        data2d = data2d.sel({yc_dim: slice(lat_min, lat_max)})
                        self.logger.debug(
                            f"Applied latitude range selection: {lat_min} to {lat_max}")
                    except Exception as e:
                        self.logger.error(f"Error applying latitude range selection: {e}")

                if 'xrange' in spec['txplot']:
                    lon_min = spec['txplot']['xrange'][0]
                    lon_max = spec['txplot']['xrange'][1]
                    try:
                        data2d = data2d.sel({xc_dim: slice(lon_min, lon_max)})
                        self.logger.debug(
                            f"Applied longitude range selection: {lon_min} to {lon_max}")
                    except Exception as e:
                        self.logger.error(
                            f"Error applying longitude range selection: {e}")

        data2d = data2d.squeeze()

        if len(data2d.dims) > 2:
            # For Hovmoller plots, we typically want time and longitude
            dims = list(data2d.dims)
            time_dim = None
            lon_dim = None
            for dim in dims:
                if dim == tc_dim or 'time' in dim.lower():
                    time_dim = dim
                elif dim == xc_dim or 'lon' in dim.lower():
                    lon_dim = dim

            if time_dim and lon_dim:
                for dim in dims:
                    if dim != time_dim and dim != lon_dim:
                        data2d = data2d.mean(dim=dim)
            else:
                for dim in dims[2:]:
                    data2d = data2d.mean(dim=dim)

            data2d = data2d.squeeze()

        # Compute weighted mean over latitude if latitude dimension exists
        if yc_dim in data2d.dims:
            try:
                weights = np.cos(np.deg2rad(data2d[yc_dim].values))
                # Make sure weights have the right shape for broadcasting
                # Create a weights array with the same shape as the data
                weight_array = xr.ones_like(data2d)
                weighted_data = data2d * weight_array * weights
                # Sum over latitude and normalize by the sum of weights
                data2d = weighted_data.sum(dim=yc_dim) / weights.sum()
            except Exception as e:
                self.logger.error(f"Error applying latitude weighting: {e}")
                self.logger.debug("Falling back to simple mean over latitude")
                if yc_dim in data2d.dims:
                    data2d = data2d.mean(dim=yc_dim)

        if np.isnan(data2d.values).any():
            self.logger.debug(
                f"Output contains NaN values: {np.sum(np.isnan(data2d.values))} NaNs")

        return apply_conversion(self.config_manager, data2d, data_array.name)

    def _select_yrange(self, data2d, name):
        """ Select a range of vertical levels"""
        if 'zrange' in self.config_manager.spec_data[name]['yzplot']:
            if not self.config_manager.spec_data[name]['yzplot']['zrange']:
                return data2d
            lo_z = self.config_manager.spec_data[name]['yzplot']['zrange'][0]
            hi_z = self.config_manager.spec_data[name]['yzplot']['zrange'][1]
            if hi_z >= lo_z:
                self.logger.error(
                    f"Upper level value ({hi_z}) must be less than low level value ({lo_z})")
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

        try:
            if 'time' in data_var.coords:
                if isinstance(time_index, int) and time_index < len(
                        data_var.coords['time']):
                    real_time = data_var.coords['time'].values[time_index]
                    real_time_readable = pd.to_datetime(real_time).strftime('%Y-%m-%d %H')
                    self.config_manager.real_time = real_time_readable
                else:
                    self.config_manager.real_time = f"Time level {time_index}"
            else:
                # If 'time' is not a coordinate, try to find a time-like coordinate
                time_coords = [coord for coord in data_var.coords if
                               'time' in coord.lower()]
                if time_coords:
                    time_coord = time_coords[0]
                    if isinstance(time_index, int) and time_index < len(
                            data_var.coords[time_coord]):
                        real_time = data_var.coords[time_coord].values[time_index]
                        real_time_readable = pd.to_datetime(real_time).strftime(
                            '%Y-%m-%d %H')
                        self.config_manager.real_time = real_time_readable
                    else:
                        self.config_manager.real_time = f"Time level {time_index}"
                else:
                    self.config_manager.real_time = f"Time level {time_index}"
        except Exception as e:
            self.config_manager.real_time = f"Time level {time_index}"
