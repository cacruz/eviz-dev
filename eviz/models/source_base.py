import os
from dataclasses import dataclass
import logging
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd

from eviz.lib.autoviz.plotter import SimplePlotter
from eviz.lib.autoviz.plotting.factory import PlotterFactory
from eviz.lib.autoviz.figure import Figure
import eviz.lib.utils as u
import eviz.lib.autoviz.utils as pu
from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.data import DataSource
from eviz.models.base import BaseSource
from eviz.lib.data.utils import apply_conversion, apply_mean, apply_zsum


logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


@dataclass
class GenericSource(BaseSource):
    """This class defines gridded interfaces and plotting for all supported sources.
       These can be gridded or ungridded (e.g. observational data sources)

    Parameters
        config_manager :
            The ConfigManager instance that provides access to all configuration data.
    """
    config_manager: ConfigManager

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        self.config = self.config_manager.config
        self.app = self.config_manager.app_data
        self.specs = self.config_manager.spec_data

        self.use_mp_pool = self.app.system_opts.get('use_mp_pool', False)

        self.dims_name = None
        self.comparison_plot = False
        self.output_fname = None
        self.ax = None
        self.fig = None

        if self.use_mp_pool:
            # Set to avoid establishing a GUI in each sub-process:
            matplotlib.use('agg')
            self.procs = list()
        
        # Initialize plot type registry
        if not hasattr(self.config_manager, '_plot_type_registry'):
            self.config_manager._plot_type_registry = {}

    def load_data_sources(self, file_list: list):
        pass

    def get_data_source(self, name: str) -> DataSource:
        pass

    def add_data_source(self, name: str, data_source: DataSource):
        pass

    def set_map_params(self, map_params):
        """Set the map parameters for plotting.

        Args:
            map_params: Dictionary of map parameters from YAML parser
        """
        pass

    def __call__(self):
        self.plot()

    def plot(self):
        """
        Generate plots for gridded fields based on current configuration.

        This is the top-level interface for plotting spatial data using one of several
        supported modes. Plotting behavior is determined by the presence or absence of
        SPECS data and by the configuration options set in the `config_manager`.

        Plot Types
        ----------
        - **Simple Plot**: 
            A single-source plot that does not require SPECS data. Used when no 
            `spec_data` is provided.
        
        - **Single Plot**: 
            A standard plot showing one data source per figure. This is the most 
            common type of map.

        - **Comparison Plot**: 
            A plot that includes two or more data sources. These can take the form of:
            
            - *Side-by-side plots*: Multiple plots shown next to each other.
            - *Overlay plots*: All data sources are plotted on a single set of axes 
            (usually for line plots); can include more than two data sources.
            - *Difference plots*: Visualize the difference between datasets.

        Notes
        -----
        The selection of which plot type to generate is controlled by the internal
        state of the configuration manager. This function delegates to private 
        helper methods corresponding to each plot type.
        """
        self.logger.info("Generate plots.")

        if not self.config_manager.spec_data:
            plotter = SimplePlotter()
            self._simple_plots(plotter)
        else:
            if self.config_manager.compare and not self.config_manager.compare_diff:
                self._side_by_side_plots()
            elif self.config_manager.compare_diff:
                self._comparison_plots()
            elif self.config_manager.overlay:
                self._side_by_side_plots()
            else:
                self._single_plots()

        if self.config_manager.print_to_file:
            output_dirs = []
            for i in range(len(self.config_manager.map_params)):
                if self.config_manager.compare or self.config_manager.compare_diff:
                    entry = u.get_nested_key_value(self.config_manager.map_params[i],
                                                   ['outputs', 'output_dir'])
                    if entry:
                        output_dirs.append(entry)
                    break
                else:
                    entry = u.get_nested_key_value(self.config_manager.map_params[i],
                                                   ['outputs', 'output_dir'])
                    if entry:
                        output_dirs.append(entry)
            if not output_dirs:
                output_dirs = [self.config.paths.output_path]

            unique_dirs = set(output_dirs)
            for dir_path in unique_dirs:
                self.logger.info(f"Output files are in {dir_path}")

        self.logger.info("Done.")

    def register_plot_type(self, field_name, plot_type):
        """Register the plot type for a field."""
        self.config_manager._plot_type_registry[field_name] = plot_type
        
    def get_plot_type(self, field_name, default='xy'):
        """Get the plot type for a field."""
        return self.config_manager._plot_type_registry.get(field_name, default)
    
    def create_plotter(self, field_name: str, plot_type: str, backend=None):
        """Create a plotter for the given field.
        
        Args:
            field_name: Name of the field to plot
            plot_type: Type of plot to create
            backend: Backend to use (defaults to config_manager.plot_backend)
            
        Returns:
            An instance of the appropriate plotter
        """        
        try:
            return PlotterFactory.create_plotter(plot_type, backend)
        except ValueError as e:
            self.logger.error(f"Error creating plotter for {field_name}: {e}")
            return None
    
    def create_plot(self, field_name, data_to_plot):
        """Create a plot using the appropriate plotter.
        
        Args:
            field_name: Name of the field to plot
            data_to_plot: Tuple containing plot data
            
        Returns:
            The created plot object
        """
        # TODO: This gets a backend per plot, but we should probably get it once and pass it around
        # Does this degrade performance?
        backend = getattr(self.config_manager, 'plot_backend', 'matplotlib')
        
        plot_type = self.get_plot_type(field_name)

        plotter = self.create_plotter(field_name, plot_type, backend)
        if plotter is None:
            return None
        
        # Create and return the plot 
        return plotter.plot(self.config_manager, data_to_plot)
    
    def process_plot(self, data_array, field_name, file_index, plot_type):
        """Process a plot for the given field.
        
        This is a base implementation that delegates to subclass methods.
        Subclasses should implement the specific plot type methods.
        """
        self.register_plot_type(field_name, plot_type)
        
        figure = Figure.create_eviz_figure(self.config_manager, plot_type)
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        
        # Delegate to the appropriate method based on plot type
        if plot_type == 'xy':
            if hasattr(self, '_process_xy_plot'):
                self._process_xy_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_xy_plot not implemented for {self.__class__.__name__}")
        elif plot_type == 'polar':
            if hasattr(self, '_process_polar_plot'):
                self._process_polar_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_xy_plot not implemented for {self.__class__.__name__}")
        elif plot_type == 'xt':
            if hasattr(self, '_process_xt_plot'):
                self._process_xt_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_xt_plot not implemented for {self.__class__.__name__}")
        elif plot_type == 'tx':
            if hasattr(self, '_process_tx_plot'):
                self._process_tx_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_tx_plot not implemented for {self.__class__.__name__}")
        elif plot_type == 'sc':
            if hasattr(self, '_process_scatter_plot'):
                self._process_scatter_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_scatter_plot not implemented for {self.__class__.__name__}")
        else:
            if hasattr(self, '_process_other_plot'):
                self._process_other_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_other_plot not implemented for {self.__class__.__name__}")

    def _is_observational_data(self, data_array):
        """
        Determine if the data array should be treated as observational data.
        
        This method checks various characteristics of the data to determine
        if it should be processed as observational data (e.g., swath format)
        or as standard gridded data.
        
        Args:
            data_array: The xarray DataArray to check
            
        Returns:
            bool: True if the data should be treated as observational
        """
        if data_array is None:
            return False
            
        # Check for characteristics of observational data
        try:
            # 2D coordinate arrays (common in swath data)
            for coord_name in data_array.coords:
                if ('lon' in coord_name.lower() or 'lat' in coord_name.lower()) and len(data_array[coord_name].shape) == 2:
                    return True
            
            # Irregular grid spacing
            xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
            yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'
            
            if xc_dim in data_array.coords and yc_dim in data_array.coords:
                
                lon_vals = data_array[xc_dim].values
                if len(lon_vals) > 2:
                    lon_diffs = np.diff(lon_vals)  # Check if longitude spacing is regular
                    if not np.allclose(lon_diffs, lon_diffs[0], rtol=1e-3):
                        return True
                
                
                lat_vals = data_array[yc_dim].values
                if len(lat_vals) > 2:
                    lat_diffs = np.diff(lat_vals)  # Check if latitude spacing is regular
                    if not np.allclose(lat_diffs, lat_diffs[0], rtol=1e-3):
                        return True
            
            # Observational metadata (usually in attributes)
            for attr in ['platform', 'instrument', 'sensor', 'satellite']:
                if hasattr(data_array, attr) or attr in data_array.attrs:
                    return True
                    
            # Do we have limited geographical coverage (i.e., not global)?
            if xc_dim in data_array.coords and yc_dim in data_array.coords:
                lon_min, lon_max = np.nanmin(data_array[xc_dim]), np.nanmax(data_array[xc_dim])
                lat_min, lat_max = np.nanmin(data_array[yc_dim]), np.nanmax(data_array[yc_dim])
                
                # Hackish way to check...
                if (lon_max - lon_min < 300) or (lat_max - lat_min < 150):
                    return True
            
        except Exception as e:
            self.logger.debug(f"Error checking if data is observational: {e}")
        
        # Not gridded!
        return False

    def _get_field_to_plot(self, data_array: xr.DataArray, field_name: str,
                           file_index: int, plot_type: str, figure, time_level,
                           level=None) -> tuple:
        """Prepare the 2D data array and coordinates to be plotted."""
        dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
        data2d = None

        if 'xy' in plot_type or 'polar' in plot_type:
            data2d = self._get_xy(data_array, level=level, time_lev=time_level)
        elif 'yz' in plot_type:
            data2d = self._get_yz(data_array, time_lev=time_level)
        elif 'xt' in plot_type:
            data2d = self._get_xt(data_array, time_lev=time_level)
        elif 'tx' in plot_type:
            data2d = self._get_tx(data_array, level=level, time_lev=time_level)
        elif 'line' in plot_type:  # like xt but use in interactive backends
            data2d = self._get_line(data_array, level=level, time_lev=time_level)
        elif 'box' in plot_type:
            data2d = self._get_box(data_array, time_lev=time_level)
        else:
            self.logger.warning(
                f"Unsupported plot type for _get_field_to_plot: {plot_type}")
            return None

        if data2d is None:
            self.logger.error(
                f"Failed to prepare 2D data for field {field_name}, plot type {plot_type}")
            return None

        # For these plot types, return without coordinates
        if plot_type in ['line', 'box', 'xt', 'tx']:
            return data2d, None, None, field_name, plot_type, file_index, figure

        # Process coordinates based on domain type
        try:
            self.config_manager.is_regional = hasattr(self, 'source_name') and self.source_name in ['lis',
                                                                                'wrf']

            if self.config_manager.is_regional:
                if hasattr(self, '_process_coordinates'):
                    return self._process_coordinates(data2d, 
                                                     dim1_name, dim2_name,
                                                     field_name,
                                                     plot_type, file_index, figure)
                else:
                    xs = np.array(self._get_field(dim1_name, data2d)[0, :])
                    ys = np.array(self._get_field(dim2_name, data2d)[:, 0])
                    latN = max(ys[:])
                    latS = min(ys[:])
                    lonW = min(xs[:])
                    lonE = max(xs[:])
                    self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
                    self.config_manager.ax_opts['central_lon'] = np.mean([lonW, lonE])
                    self.config_manager.ax_opts['central_lat'] = np.mean([latS, latN])

                    return data2d, xs, ys, field_name, plot_type, file_index, figure
            else:
                x = data2d[dim1_name].values if dim1_name in data2d.coords else None
                y = data2d[dim2_name].values if dim2_name in data2d.coords else None

                if x is None or y is None:
                    dims = list(data2d.dims)
                    if len(dims) >= 2:
                        x = data2d[dims[0]].values
                        y = data2d[dims[1]].values
                    else:
                        self.logger.error(
                            "Dataset has fewer than 2 dimensions, cannot plot")
                        return None

                if np.isnan(data2d.values).all():
                    self.logger.error(
                        f"All values are NaN for {field_name}. Using original data.")
                    data2d = data_array.squeeze()
                elif np.isnan(data2d.values).any():
                    self.logger.debug(
                        f"Note: Some NaN values present ({np.sum(np.isnan(data2d.values))} NaNs).")
                    # data2d = data2d.fillna(0)

                return data2d, x, y, field_name, plot_type, file_index, figure

        except Exception as e:
            self.logger.error(f"Error processing coordinates for {field_name}: {e}")
            return None

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

    def _simple_plots(self, plotter):
        """Generate simple plots."""
        self.logger.info("Generating simple plots")

    def _single_plots(self):
        """Generate single plots."""
        self.logger.info("Generating single plots")

        if not self.config_manager.map_params:
            self.logger.error(
                "No map_params available for plotting. Check your YAML configuration.")
            return

        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error(
                "No data sources available. Check your YAML configuration and ensure data files exist.")
            self.logger.info(
                "Map parameters found but no data sources loaded. Here are the expected files:")

            for i, entry in enumerate(self.config_manager.app_data.inputs):
                file_path = os.path.join(entry.get('location', ''), entry.get('name', ''))
                print(f"  {i + 1}. {file_path}")
            return

        for idx, params in self.config_manager.map_params.items():
            field_name = params.get('field')
            if not field_name:
                continue
            self.config_manager.current_field_name = field_name

            filename = params.get('filename')
            self.config_manager.findex = self.config_manager.get_file_index_by_filename(filename)

            data_source = self.config_manager.pipeline.get_data_source(filename)
            # print(data_source)
            if not data_source:
                self.logger.warning(f"No data source found in pipeline for {filename}")
                continue

            # NUWRF-specific initializations
            if hasattr(self, 'source_name') and self.source_name in ['wrf']:
                self._init_wrf_domain(data_source)
            if hasattr(self, 'source_name') and self.source_name in ['lis']:
                self._init_lis_domain(data_source)

            if hasattr(data_source, 'dataset') and data_source.dataset is not None:
                field_data = data_source.dataset.get(field_name)
            else:
                field_data = None

            if field_data is None:
                self.logger.warning(
                    f"Field {field_name} not found in data source for {filename}")
                continue

            field_data_array = data_source.dataset[field_name]
            plot_types = params.get('to_plot', ['xy'])
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
            for plot_type in plot_types:
                self.logger.info(f"Plotting {field_name}, {plot_type} plot")
                self.process_plot(field_data_array, field_name, idx, plot_type)

        if self.config_manager.make_gif:
            pu.create_gif(self.config_manager)

    def _comparison_plots(self):
        """Generate comparison plots for paired data sources according to configuration.

        Args:
            plotter (instance of ComparisonPlotter): The plotter instance to use for generating plots.
        """
        self.logger.info("Generating comparison plots")
        current_field_index = 0

        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for comparison plotting.")
            return

        if not self.config_manager.a_list or not self.config_manager.b_list:
            self.logger.error("a_list or b_list is empty, cannot perform comparison.")
            return

        idx1 = self.config_manager.a_list[0]
        idx2 = self.config_manager.b_list[0]

        # Gather all unique field names from map_params for these files
        fields_file1 = [params['field'] for i, params in
                        self.config_manager.map_params.items() if
                        params['file_index'] == idx1]
        fields_file2 = [params['field'] for i, params in
                        self.config_manager.map_params.items() if
                        params['file_index'] == idx2]

        # Pair fields by order, not by name
        num_pairs = min(len(fields_file1), len(fields_file2))
        field_pairs = list(zip(fields_file1[:num_pairs], fields_file2[:num_pairs]))

        self.logger.debug(f"Comparing files {idx1} and {idx2}")
        self.logger.debug(f"Fields in file 1: {fields_file1}")
        self.logger.debug(f"Fields in file 2: {fields_file2}")
        self.logger.debug(f"Field pairs to compare: {field_pairs}")

        for field1, field2 in field_pairs:
            # Find map_params for this field in both files
            idx1_field = next((i for i, params in self.config_manager.map_params.items()
                               if params['file_index'] == idx1 and params[
                                   'field'] == field1), None)
            idx2_field = next((i for i, params in self.config_manager.map_params.items()
                               if params['file_index'] == idx2 and params[
                                   'field'] == field2), None)
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

            sdat1_dataset = data_source1.dataset if hasattr(data_source1,
                                                            'dataset') else None
            sdat2_dataset = data_source2.dataset if hasattr(data_source2,
                                                            'dataset') else None

            if sdat1_dataset is None or sdat2_dataset is None:
                continue

            # NUWRF-specific initializations
            if hasattr(self, 'source_name') and self.source_name in ['wrf']:
                self._init_wrf_domain(sdat1_dataset)
                self._init_wrf_domain(sdat2_dataset)
            if hasattr(self, 'source_name') and self.source_name in ['lis']:
                self._init_lis_domain(sdat1_dataset)
                self._init_lis_domain(sdat2_dataset)

            file_indices = (idx1_field, idx2_field)

            self.field_names = (field1, field2)

            # Assuming plot types are the same for comparison
            plot_types = map1_params.get('to_plot', ['xy'])
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
            for plot_type in plot_types:
                self.logger.info(f"Plotting {field1} vs {field2}, {plot_type} plot")
                self.data2d_list = []  # Reset for each plot type

                if 'xy' in plot_type or 'po' in plot_type or 'polar' in plot_type:
                    self._process_xy_comparison_plots(file_indices,
                                                      current_field_index,
                                                      field1, field2, plot_type,
                                                      sdat1_dataset, sdat2_dataset)
                else:
                    self._process_other_comparison_plots(file_indices,
                                                         current_field_index,
                                                         field1, field2,
                                                         plot_type, sdat1_dataset,
                                                         sdat2_dataset)

            current_field_index += 1

    def _side_by_side_plots(self):
        """
        Generate side-by-side comparison plots for the given plotter.

        Args:
            plotter (instance of ComparisonPlotter): The plotter instance to use for generating plots.

        """
        self.logger.info("Generating side-by-side comparison plots")
        current_field_index = 0
        self.data2d_list = []

        # Get the file indices for the two files being compared
        if not self.config_manager.a_list or not self.config_manager.b_list:
            self.logger.error(
                "a_list or b_list is empty, cannot perform side-by-side comparison.")
            return

        idx1 = self.config_manager.a_list[0]
        idx2 = self.config_manager.b_list[0]

        # Gather all unique field names from map_params for these files
        fields_file1 = [params['field'] for i, params in
                        self.config_manager.map_params.items() if
                        params['file_index'] == idx1]
        fields_file2 = [params['field'] for i, params in
                        self.config_manager.map_params.items() if
                        params['file_index'] == idx2]

        # Pair fields by order, not by name
        num_pairs = min(len(fields_file1), len(fields_file2))
        field_pairs = list(zip(fields_file1[:num_pairs], fields_file2[:num_pairs]))

        for field1, field2 in field_pairs:
            # Find map_params for this field in both files
            idx1_field = next((i for i, params in self.config_manager.map_params.items()
                               if params['file_index'] == idx1 and params[
                                   'field'] == field1), None)
            idx2_field = next((i for i, params in self.config_manager.map_params.items()
                               if params['file_index'] == idx2 and params[
                                   'field'] == field2), None)
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

            sdat1_dataset = data_source1.dataset if hasattr(data_source1,
                                                            'dataset') else None
            sdat2_dataset = data_source2.dataset if hasattr(data_source2,
                                                            'dataset') else None

            if sdat1_dataset is None or sdat2_dataset is None:
                continue

            # NUWRF-specific initializations
            if hasattr(self, 'source_name') and self.source_name in ['wrf']:
                self._init_wrf_domain(sdat1_dataset)
                self._init_wrf_domain(sdat2_dataset)
            if hasattr(self, 'source_name') and self.source_name in ['lis']:
                self._init_lis_domain(sdat1_dataset)
                self._init_lis_domain(sdat2_dataset)

            self.file_indices = (idx1_field, idx2_field)

            self.field_names = (field1, field2)

            plot_types = map1_params.get('to_plot', ['xy'])
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
            for plot_type in plot_types:
                self.data2d_list = []
                self.logger.info(f"Plotting {field1} vs {field2} , {plot_type} plot")

                if 'xy' in plot_type or 'polar' in plot_type:
                    self._process_xy_side_by_side_plots(current_field_index,
                                                        field1, field2,
                                                        plot_type,
                                                        sdat1_dataset, sdat2_dataset)
                else:
                    self._process_other_side_by_side_plots(current_field_index,
                                                           field1, field2,
                                                           plot_type, sdat1_dataset,
                                                           sdat2_dataset)
            current_field_index += 1

    # DATA SLICE PROCESSING METHODS
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
        zonal_mean.attrs = data_array.attrs.copy()

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
            # Handle negative indices (e.g., -1 for the last time level)
            if isinstance(time_lev, int):
                # Convert negative index to positive if needed
                actual_time_lev = time_lev if time_lev >= 0 else num_tc + time_lev
                
                # Check if the index is valid
                if 0 <= actual_time_lev < num_tc:
                    d_temp = d_temp.isel({tc_dim: actual_time_lev})
                    self.logger.debug(f"Selected time level {actual_time_lev} (specified as {time_lev})")
                else:
                    self.logger.warning(f"Time level {time_lev} out of range (0-{num_tc-1}), using first time level")
                    d_temp = d_temp.isel({tc_dim: 0})
            else:
                self.logger.warning(f"No time dimension found matching {tc_dim}")



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
                data2d.attrs = data_array.attrs.copy()
                return apply_conversion(self.config_manager, data2d, data_array.name)

        if self.config_manager.ax_opts.get('zave', False):
            self.logger.debug("Averaging over vertical levels.")
            data2d = apply_mean(self.config_manager, data2d, level='all')
            data2d.attrs = data_array.attrs.copy()
            return apply_conversion(self.config_manager, data2d, data_array.name)

        if self.config_manager.ax_opts.get('zsum', False):
            self.logger.debug("Summing over vertical levels.")
            data2d_zsum = apply_zsum(data2d)
            self.logger.debug(
                "Min: {data2d_zsum.min().values}, Max: {data2d_zsum.max().values}")
            data2d.attrs = data_array.attrs.copy()
            return apply_conversion(self.config_manager, data2d_zsum, data_array.name)

        if np.isnan(data2d.values).any():
            self.logger.debug(
                f"Output contains NaN values: {np.sum(np.isnan(data2d.values))} NaNs")
        data2d.attrs = data_array.attrs.copy()

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
        num_times = data_array[tc_dim].size
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
                self.logger.info(f"Averaging method: {mean_type}")

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
                            data2d = data2d.groupby(time_attr).mean(dim=tc_dim)
                        else:
                            if 'time' in data2d.dims:
                                time_attr = f"time.{mean_type}"
                                data2d = data2d.groupby(time_attr).mean(dim='time')
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
                    self.logger.debug(
                        f"Averaging over non-time dimensions: {non_time_dims}")
                    data2d = data2d.mean(dim=non_time_dims)
            else:
                # Could not identify time dimension. Using first dimension
                other_dims = dims[1:]
                if other_dims:
                    data2d = data2d.mean(dim=other_dims)

        data2d = data2d.squeeze()
        data2d.attrs = data_array.attrs.copy()

        if np.isnan(data2d.values).any():
            self.logger.debug(
                f"Output contains NaN values: {np.sum(np.isnan(data2d.values))} NaNs")
            
        return apply_conversion(self.config_manager, data2d, data_array.name)
    
    def _get_box(self, data_array, time_lev=None):
        """Extract data for a box plot.
        
        This method prepares data for box plots by extracting values across a dimension
        (typically spatial) for statistical analysis.
        
        Args:
            data_array: xarray.DataArray to extract data from
            time_lev: Time level to extract (optional)
            
        Returns:
            tuple: (data_df, field_name, plot_type, file_index)
                data_df: pandas DataFrame with columns for categories and values
                field_name: Name of the field being plotted
                plot_type: Type of plot ('box')
                file_index: Index of the file being plotted
        """
        self.logger.debug(f"Extracting box plot data from {data_array.name if hasattr(data_array, 'name') else 'unnamed array'}")
        
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'
        xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
        yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'
        num_times = data_array[tc_dim].size
        self.logger.debug(f"'{data_array.name}' field has {num_times} time levels")
        
        # Check if all data is NaN
        if np.isnan(data_array).all():
            self.logger.warning(f"All values are NaN for {data_array.name if hasattr(data_array, 'name') else 'unnamed field'}")
            return None
        
        if np.isnan(data_array.values).any():
            self.logger.info(
                f"Output contains NaN values: {np.sum(np.isnan(data_array.values))} NaNs")
        
        # If time_lev is None and we have a time dimension, use all time levels
        if time_lev is None and tc_dim in data_array.dims:
            self.logger.debug("Using all time levels for box plot")
            # No need to select a specific time level
        elif time_lev is not None and tc_dim in data_array.dims:
            self.logger.debug(f"Selecting time level: {time_lev}")
            data_array = data_array.isel({tc_dim: time_lev})
        
        # Get field name
        field_name = data_array.name if hasattr(data_array, 'name') else 'unnamed'

        # Convert to pandas DataFrame
        try:
            # df = data_array.stack(points=(yc_dim, xc_dim)).to_dataframe(name=field_name).reset_index()
            # For spatial data, we want to create a box plot of values across the spatial domain
            # First, flatten the spatial dimensions
            if len(data_array.dims) > 1:
                # Stack all non-time dimensions to create a single dimension for spatial points
                dims_to_stack = [dim for dim in data_array.dims if dim != tc_dim]
                if dims_to_stack:
                    stacked = data_array.stack(point=dims_to_stack)
                    df = stacked.to_dataframe()
                else:
                    df = data_array.to_dataframe()
            else:
                # If there's only one dimension, use it directly
                df = data_array.to_dataframe()
            
            # If we have a time dimension and didn't select a specific level,
            # we can create a box plot for each time step
            if tc_dim in data_array.dims and time_lev is None:
                # Ensure the time column is properly formatted
                if tc_dim in df.index.names:
                    # Reset index to make time a column
                    df = df.reset_index()
                    
                    # Convert time to string format for better display
                    if pd.api.types.is_datetime64_any_dtype(df[tc_dim]):
                        df[tc_dim] = df[tc_dim].dt.strftime('%Y-%m-%d %H:%M')
                
                # The DataFrame should have a column for time and a column for values
                data_df = df.rename(columns={field_name: 'value'})
                category_col = tc_dim
            else:
                # Without time or with a specific time selected, we might want to use another
                # categorical variable if available, otherwise just use the values
                data_df = df.reset_index()
                
                # Try to find a suitable categorical column
                categorical_cols = [col for col in data_df.columns 
                                if col != field_name and data_df[col].nunique() < 30]
                
                if categorical_cols:
                    category_col = categorical_cols[0]
                else:
                    # If no good categorical column, we'll just have a single box
                    data_df['category'] = 'All Data'
                    category_col = 'category'
                
                # Rename the value column
                data_df = data_df.rename(columns={field_name: 'value'})

            # Handle fill values if specified
            if hasattr(self.config_manager, 'spec_data') and field_name in self.config_manager.spec_data:
                if 'fill_value' in self.config_manager.spec_data[field_name].get('boxplot', {}):
                    fill_value = self.config_manager.spec_data[field_name]['boxplot']['fill_value']
                    data_df = data_df[data_df['value'] != fill_value]
            
            # Remove NaN values
            data_df = data_df.dropna(subset=['value'])
            
            # Check if we have data
            if len(data_df) == 0:
                self.logger.warning(f"No valid data for box plot of {field_name}")
                return None
            
            self.logger.debug(f"Created DataFrame with {len(data_df)} rows for box plot")
            return data_df
        
        except Exception as e:
            self.logger.error(f"Error creating box plot data: {e}")
            return None

    def _get_line(self, data_array, time_lev=None, level=None):
        """Extract data for a line plot.
        
        This method prepares data for line plots, typically extracting a time series
        or a spatial transect.
        
        Args:
            data_array: xarray.DataArray to extract data from
            time_lev: Time level to extract (optional)
            level: Vertical level to extract (optional)
            
        Returns:
            tuple: (data_df, x_col, y_col, field_name, plot_type, file_index)
                data_df: pandas DataFrame with x and y columns
                x_col: Name of the x-axis column
                y_col: Name of the y-axis column
                field_name: Name of the field being plotted
                plot_type: Type of plot ('line')
                file_index: Index of the file being plotted
        """
        self.logger.debug(f"Extracting line plot data from {data_array.name if hasattr(data_array, 'name') else 'unnamed array'}")
        
        # Get dimension names
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'
        xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
        yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'
        
        # Get field name
        field_name = data_array.name if hasattr(data_array, 'name') else 'unnamed'
        
        try:
            # Determine the type of line plot based on available dimensions
            
            # Case 1: Time series (most common)
            if tc_dim in data_array.dims:
                # If we have spatial dimensions, we need to select a point or average
                spatial_dims = [dim for dim in [xc_dim, yc_dim, zc_dim] if dim in data_array.dims]
                
                if spatial_dims:
                    # If level is specified and vertical dimension exists, select that level
                    if level is not None and zc_dim in data_array.dims:
                        data_array = data_array.sel({zc_dim: level}, method='nearest')
                    
                    # For spatial dimensions, we have options:
                    # 1. Select a specific point (if coordinates are provided in config)
                    # 2. Average over the spatial domain
                    
                    # Check if specific coordinates are provided
                    x_point = self.config_manager.ax_opts.get('x_point', None)
                    y_point = self.config_manager.ax_opts.get('y_point', None)
                    
                    if x_point is not None and y_point is not None and xc_dim in data_array.dims and yc_dim in data_array.dims:
                        # Select the nearest point to the specified coordinates
                        data_array = data_array.sel({xc_dim: x_point, yc_dim: y_point}, method='nearest')
                        point_label = f"({x_point}, {y_point})"
                    else:
                        # Average over remaining spatial dimensions
                        for dim in spatial_dims:
                            if dim in data_array.dims:
                                data_array = data_array.mean(dim=dim)
                        point_label = "Spatial Average"
                
                # Convert to DataFrame
                df = data_array.to_dataframe()
                
                # Reset index to make time a column
                df = df.reset_index()
                
                # Rename columns for clarity
                x_col = tc_dim
                y_col = field_name
                
                # Format time column if it's datetime
                if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    df[x_col] = df[x_col].dt.strftime('%Y-%m-%d %H:%M')
                
                # Add a label column if we have point information
                if 'point_label' in locals():
                    df['label'] = point_label
            
            # Case 2: Spatial transect (e.g., along longitude)
            elif xc_dim in data_array.dims:
                # If we have a vertical dimension, select the specified level
                if level is not None and zc_dim in data_array.dims:
                    data_array = data_array.sel({zc_dim: level}, method='nearest')
                
                # If we have a latitude dimension, we need to select a specific latitude
                if yc_dim in data_array.dims:
                    # Check if a specific latitude is provided
                    y_point = self.config_manager.ax_opts.get('y_point', None)
                    
                    if y_point is not None:
                        # Select the nearest latitude
                        data_array = data_array.sel({yc_dim: y_point}, method='nearest')
                        lat_label = f"Latitude: {y_point}"
                    else:
                        # Average over latitude
                        data_array = data_array.mean(dim=yc_dim)
                        lat_label = "Latitude Average"
                
                # Convert to DataFrame
                df = data_array.to_dataframe()
                
                # Reset index to make longitude a column
                df = df.reset_index()
                
                # Rename columns for clarity
                x_col = xc_dim
                y_col = field_name
                
                # Add a label column if we have latitude information
                if 'lat_label' in locals():
                    df['label'] = lat_label
            
            # Case 3: Vertical profile
            elif zc_dim in data_array.dims:
                # If we have horizontal dimensions, we need to select a point or average
                if xc_dim in data_array.dims or yc_dim in data_array.dims:
                    # Check if specific coordinates are provided
                    x_point = self.config_manager.ax_opts.get('x_point', None)
                    y_point = self.config_manager.ax_opts.get('y_point', None)
                    
                    if x_point is not None and xc_dim in data_array.dims:
                        data_array = data_array.sel({xc_dim: x_point}, method='nearest')
                    elif xc_dim in data_array.dims:
                        data_array = data_array.mean(dim=xc_dim)
                    
                    if y_point is not None and yc_dim in data_array.dims:
                        data_array = data_array.sel({yc_dim: y_point}, method='nearest')
                    elif yc_dim in data_array.dims:
                        data_array = data_array.mean(dim=yc_dim)
                    
                    if x_point is not None and y_point is not None:
                        point_label = f"({x_point}, {y_point})"
                    else:
                        point_label = "Horizontal Average"
                
                # Convert to DataFrame
                df = data_array.to_dataframe()
                
                # Reset index to make level a column
                df = df.reset_index()
                
                # Rename columns for clarity
                x_col = field_name
                y_col = zc_dim  # For vertical profiles, we typically put height/pressure on y-axis
                
                # Add a label column if we have point information
                if 'point_label' in locals():
                    df['label'] = point_label
            
            # Case 4: Simple 1D array (just plot as is)
            else:
                # Convert to DataFrame
                df = data_array.to_dataframe()
                
                # Reset index
                df = df.reset_index()
                
                # If there's only one column besides the index, create an index column
                if len(df.columns) == 1:
                    df['index'] = np.arange(len(df))
                    x_col = 'index'
                    y_col = df.columns[0]
                else:
                    # Try to find suitable x and y columns
                    numeric_cols = [col for col in df.columns 
                                if np.issubdtype(df[col].dtype, np.number)]
                    
                    if len(numeric_cols) >= 2:
                        x_col = numeric_cols[0]
                        y_col = numeric_cols[1]
                    elif len(numeric_cols) == 1:
                        df['index'] = np.arange(len(df))
                        x_col = 'index'
                        y_col = numeric_cols[0]
                    else:
                        # Fallback
                        df['index'] = np.arange(len(df))
                        df['value'] = 0
                        x_col = 'index'
                        y_col = 'value'
            
            # Handle fill values if specified
            if hasattr(self.config_manager, 'spec_data') and field_name in self.config_manager.spec_data:
                if 'fill_value' in self.config_manager.spec_data[field_name].get('lineplot', {}):
                    fill_value = self.config_manager.spec_data[field_name]['lineplot']['fill_value']
                    df = df[df[y_col] != fill_value]
            
            # Remove NaN values
            df = df.dropna(subset=[x_col, y_col])
            
            # Check if we have data
            if len(df) == 0:
                self.logger.warning(f"No valid data for line plot of {field_name}")
                return None
            
            # Sort by x column for proper line plotting
            df = df.sort_values(by=x_col)
            
            self.logger.debug(f"Created DataFrame with {len(df)} rows for line plot")
            return (df, x_col, y_col, field_name, 'line', self.config_manager.findex)
        
        except Exception as e:
            self.logger.error(f"Error creating line plot data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

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
            
        data2d.attrs = data_array.attrs.copy()

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

