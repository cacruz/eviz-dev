import os
from dataclasses import dataclass
import logging
import matplotlib

from eviz.lib.autoviz.plotter import SimplePlotter
from eviz.lib.autoviz.plotting.factory import PlotterFactory
from eviz.lib.autoviz.figure import Figure
import eviz.lib.utils as u
import eviz.lib.autoviz.utils as pu
from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.data import DataSource
from eviz.models.base import BaseSource


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
            pu.create_gif(self.config_manager.config)

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

