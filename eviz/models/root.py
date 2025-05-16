# File: eviz/models/root.py
import os
from dataclasses import dataclass
import logging

import matplotlib
import matplotlib.pyplot as plt

from eviz.lib.autoviz.plotter import SimplePlotter, ComparisonPlotter, SinglePlotter
import eviz.lib.utils as u
from eviz.lib import const as constants

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.models.base import AbstractRoot
# Removed import of DataSourceFactory as data loading is handled by the pipeline


@dataclass
class Root(AbstractRoot):
    """This class defines generic interfaces and plotting for all supported sources.

    Parameters
        config_manager :
            The ConfigManager instance that provides access to all configuration data.
    """
    config_manager: ConfigManager
    # Removed data_sources: dict = field(default_factory=dict) - DataPipeline will store them

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
        # self.source_data = None # This seems to be a temporary holder, might need refactoring
        self.comparison_plot = False
        # self.map_data = None # Access via config_manager.map_params
        self.output_fname = None
        self.ax = None
        self.fig = None
        # self.ax_opts = None # Access via config_manager.ax_opts
        # self.findex = {} # Access via config_manager.findex

        # Handle multiprocessing pool setup
        if self.use_mp_pool:
            # Set to avoid establishing a GUI in each sub-process:
            matplotlib.use('agg')
            self.procs = list()

    # Removed add_data_source and get_data_source methods - access via config_manager.pipeline

    def set_map_params(self, map_params):
        """Set the map parameters for plotting.

        Args:
            map_params: Dictionary of map parameters from YAML parser
        """
        # Assuming map_params is already loaded into config_manager.map_params
        # This method might be redundant if map_params is always loaded during config initialization.
        # If it's needed to *override* map_params, the logic should update config_manager.map_params
        self.logger.warning("set_map_params called. Assuming map_params is already in ConfigManager.")
        # If overriding is needed:
        # self.config_manager.map_params = map_params
        # self.logger.info(f"Overrode map_params with {len(map_params)} entries")


    # Removed load_data_sources method - data loading is handled by ConfigurationAdapter and DataPipeline

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
        if self.config_manager.print_to_file:
            output_dirs = []
            # Access map_params via config_manager
            for i in range(len(self.config_manager.map_params)):
                if self.config_manager.compare or self.config_manager.compare_diff:
                    entry = u.get_nested_key_value(self.config_manager.map_params[i], ['outputs', 'output_dir'])
                    if entry:
                        output_dirs.append(entry)
                    break
                else:
                    entry = u.get_nested_key_value(self.config_manager.map_params[i], ['outputs', 'output_dir'])
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

        # Check if map_params is available via config_manager
        if not self.config_manager.map_params:
            self.logger.error("No map_params available for plotting. Check your YAML configuration.")
            return

        # Check if any data sources were loaded via the pipeline
        # Access data sources via the pipeline
        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available. Check your YAML configuration and ensure data files exist.")
            self.logger.info("Map parameters found but no data sources loaded. Here are the expected files:")
            # Access app_data via config_manager
            for i, entry in enumerate(self.config_manager.app_data.inputs):
                file_path = os.path.join(entry.get('location', ''), entry.get('name', ''))
                print(f"  {i+1}. {file_path}")
            return

        # Iterate through map_params to generate plots
        # Access map_params via config_manager
        for idx, params in self.config_manager.map_params.items():
            field_name = params.get('field')
            if not field_name:
                continue

            self.logger.info(f"Processing field: {field_name}")

            # Get the data source based on the filename in map_params from the pipeline
            filename = params.get('filename')
            data_source = self.config_manager.pipeline.get_data_source(filename)

            if not data_source:
                self.logger.warning(f"No data source found in pipeline for {filename}")
                continue

            # Get the field data from the data source
            # Access dataset via the DataSource object
            if hasattr(data_source, 'dataset') and data_source.dataset is not None:
                 field_data = data_source.dataset.get(field_name)
            else:
                 field_data = None


            if field_data is None:
                self.logger.warning(f"Field {field_name} not found in data source for {filename}")
                continue

            # Get plot type from map_params
            plot_type = params.get('to_plot', ['xy'])[0]  # Default to 'xy' if not specified

            # Update config_manager state variables before plotting
            self.config_manager.findex = idx # Assuming idx corresponds to file index in map_params
            self.config_manager.pindex = idx # Assuming idx corresponds to plot index
            self.config_manager.axindex = 0 # Reset axis index for each plot

            # Generate the plot using the plotter
            self.logger.info(f"Plotting {field_name} as {plot_type} plot")
            # Pass the data array directly, not the whole source_data dict
            # The plotter will need to be updated to handle this
            # For now, let's pass a tuple similar to the old structure but with the DataArray
            # (data2d, x_values, y_values, field_name, plot_type, file_index, figure, ax)
            # We need to determine x and y values here or in the plotter
            # Let's pass the DataArray and let the plotter handle coordinates
            # The plotter will need access to config_manager to get dim names and coordinates
            plotter.single_plots(self.config_manager, (field_data, None, None, field_name, plot_type, idx, None, None), level=0)


    def _comparison_plots(self, plotter):
        """Generate comparison plots."""
        self.logger.info("Generating comparison plots")
        # Add logic to generate comparison plots using the plotter

    # Removed _get_field_for_simple_plot, _get_field_to_plot, _get_field_to_plot_compare
    # Data slicing and preparation should ideally happen before calling the plotter,
    # or the plotter methods should be updated to handle DataArray directly.

    def _plot_dest(self, name):
        # Access output_dir and print_format via config_manager
        if self.config_manager.print_to_file:
            output_fname = name + "." + self.config_manager.print_format
            filename = os.path.join(self.config_manager.output_dir, output_fname)
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()
