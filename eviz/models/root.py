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
from eviz.lib.data.data_source import DataSourceFactory


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

        # Iterate over each data source and apply plotting logic
        for name, data_source in self.data_sources.items():
            self.logger.info(f"Plotting data source: {name}")

            # If there is no SPECS file, produce "simple" visualizations
            if not self.config_manager.spec_data:
                plotter = SimplePlotter()
                self._simple_plots(plotter, data_source)
            else:
                if self.config_manager.compare:
                    plotter = ComparisonPlotter(self.config_manager.compare_exp_ids)
                    self._side_by_side_plots(plotter, data_source)
                elif self.config_manager.compare_diff:
                    plotter = ComparisonPlotter(self.config_manager.compare_exp_ids)
                    self._comparison_plots(plotter, data_source)
                else:
                    plotter = SinglePlotter()
                    self._single_plots(plotter, data_source)

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
    def _side_by_side_plots(self, plotter, data_source):
        """Generate side-by-side comparison plots for the given data source."""
        self.logger.info(f"Generating side-by-side plots for data source: {data_source}")
        # Add logic to generate side-by-side plots using the plotter and data_source

    def _simple_plots(self, plotter, data_source):
        """Generate simple plots for the given data source."""
        self.logger.info(f"Generating simple plots for data source: {data_source}")
        # Add logic to generate simple plots using the plotter and data_source

    def _single_plots(self, plotter, data_source):
        """Generate single plots for the given data source."""
        self.logger.info(f"Generating single plots for data source: {data_source}")
        # Add logic to generate single plots using the plotter and data_source

    def _comparison_plots(self, plotter, data_source):
        """Generate comparison plots for the given data source."""
        self.logger.info(f"Generating comparison plots for data source: {data_source}")
        # Add logic to generate comparison plots using the plotter and data_source

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
