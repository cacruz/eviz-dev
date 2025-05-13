import os
from dataclasses import dataclass, field
import logging

import matplotlib
import matplotlib.pyplot as plt

from eviz.lib.autoviz.plotter import SimplePlotter, ComparisonPlotter, SinglePlotter
import eviz.lib.utils as u
from eviz.lib import const as constants

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.models.base import AbstractRoot
from eviz.lib.data.factory.source_factory import DataSourceFactory


@dataclass
class Root(AbstractRoot):
    """This class defines generic interfaces and plotting for all supported sources.

    Parameters
        config_manager :
            The ConfigManager instance that provides access to all configuration data.
    """
    config_manager: ConfigManager
    data_sources: dict = field(default_factory=dict)  # Map of data source names to DataSource instances

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")

        # Access app_data and spec_data via ConfigManager
        self.app = self.config_manager.app_data
        self.specs = self.config_manager.spec_data

        # Access system options via app_data
        self.use_mp_pool = self.app.system_opts.get('use_mp_pool', False)

        # Initialize other attributes
        self.dims_name = None
        self.source_data = None
        self.comparison_plot = False
        self.map_data = None
        self.output_fname = None
        self.ax = None
        self.fig = None
        self.ax_opts = None
        self.findex = {}

        # Handle multiprocessing pool setup
        if self.use_mp_pool:
            # Set to avoid establishing a GUI in each sub-process:
            matplotlib.use('agg')
            self.procs = list()

    def add_data_source(self, name: str, data_source):
        """Add a data source to the model."""
        self.data_sources[name] = data_source
        self.logger.info(f"Added data source: {name}")

    def get_data_source(self, name: str):
        """Retrieve a data source by name."""
        return self.data_sources.get(name)
        
    def set_map_params(self, map_params):
        """Set the map parameters for plotting.
        
        Args:
            map_params: Dictionary of map parameters from YAML parser
        """
        self.map_data = map_params
        self.logger.info(f"Set map_params with {len(map_params)} entries")

    def load_data_sources(self, file_list: list):
        """Load multiple data sources from a list of file paths."""
        for file_path in file_list:
            file_extension = file_path.split('.')[-1]
            data_source = DataSourceFactory.get_data_class(file_extension)
            data = data_source.load_data(file_path)
            self.add_data_source(file_path, data)

    def __call__(self):
        self.plot()

    def plot(self):
        """Top-level interface for generic (NetCDF) maps."""
        self.logger.info("Generate plots.")

        # Create the appropriate plotter based on configuration
        if not self.config_manager.spec_data:
            plotter = SimplePlotter()
            self._simple_plots(plotter)
        else:
            if self.config_manager.compare:
                plotter = ComparisonPlotter(self.config_manager.compare_exp_ids)
                self._side_by_side_plots(plotter)
            elif self.config_manager.compare_diff:
                plotter = ComparisonPlotter(self.config_manager.compare_exp_ids)
                self._comparison_plots(plotter)
            else:
                plotter = SinglePlotter()
                self._single_plots(plotter)

        # Handle output file generation
        if self.config_manager.output_config.print_to_file:
            output_dirs = []
            for i in range(len(self.config_manager.config.map_params)):
                if self.config_manager.compare or self.config_manager.compare_diff:
                    entry = u.get_nested_key_value(self.config_manager.config.map_params[i], ['outputs', 'output_dir'])
                    if entry:
                        output_dirs.append(entry)
                    break
                else:
                    entry = u.get_nested_key_value(self.config_manager.config.map_params[i], ['outputs', 'output_dir'])
                    if entry:
                        output_dirs.append(entry)
            if not output_dirs:
                output_dirs = [constants.output_path]

            unique_dirs = set(output_dirs)
            for dir_path in unique_dirs:
                self.logger.info(f"Output files are in {dir_path}")

        self.logger.info("Done.")

    # Add a new method for side-by-side comparison plots
    def _side_by_side_plots(self, plotter):
        """Generate side-by-side comparison plots."""
        self.logger.info("Generating side-by-side plots")
        # Add logic to generate side-by-side plots using the plotter

    def _simple_plots(self, plotter):
        """Generate simple plots."""
        self.logger.info("Generating simple plots")
        # Add logic to generate simple plots using the plotter

    def _single_plots(self, plotter):
        """Generate single plots."""
        self.logger.info("Generating single plots")
        
        # Check if map_data is available
        if not self.map_data:
            self.logger.error("No map_params available for plotting. Make sure set_map_params is called.")
            return
            
        # Check if we have any data sources
        if not self.data_sources:
            self.logger.error("No data sources available. Check your YAML configuration and ensure data files exist.")
            self.logger.info("Map parameters found but no data sources loaded. Here are the expected files:")
            for idx, params in self.map_data.items():
                filename = params.get('filename')
                if filename:
                    self.logger.info(f"  - {filename}")
            return
            
        # Iterate through map_params to generate plots
        for idx, params in self.map_data.items():
            field_name = params.get('field')
            if not field_name:
                continue
                
            self.logger.info(f"Processing field: {field_name}")
            
            # Get the data source based on the filename in map_params
            filename = params.get('filename')
            data_source = self.data_sources.get(filename)
            
            if not data_source:
                self.logger.warning(f"No data source found for {filename}")
                continue
                
            # Get the field data from the data source
            field_data = data_source.get_variable(field_name)
            
            if field_data is None:
                self.logger.warning(f"Field {field_name} not found in data source")
                continue
                
            # Get plot type from map_params
            plot_type = params.get('to_plot', ['xy'])[0]  # Default to 'xy' if not specified
            
            # Generate the plot using the plotter
            self.logger.info(f"Plotting {field_name} as {plot_type} plot")
            plotter.plot(self.config_manager, (field_data, None, None, field_name, plot_type, 0, None, None), level=0)

    def _comparison_plots(self, plotter):
        """Generate comparison plots."""
        self.logger.info("Generating comparison plots")
        # Add logic to generate comparison plots using the plotter

    def _get_field_for_simple_plot(self, *args):
        pass

    def _get_field_to_plot(self, *args, **kwargs):
        pass

    def _get_field_to_plot_compare(self, *args, **kwargs):
        pass

    def _plot_dest(self, name):
        if self.config_manager.print_to_file:
            output_fname = name + "." + self.config_manager.print_format
            filename = os.path.join(self.config_manager.output_dir, output_fname)
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()
