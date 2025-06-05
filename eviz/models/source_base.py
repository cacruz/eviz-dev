import os
from dataclasses import dataclass
import logging
import matplotlib

from eviz.lib.autoviz.plotter import SimplePlotter, ComparisonPlotter, SinglePlotter
import eviz.lib.utils as u
from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.data import DataSource
from eviz.models.base import BaseSource


logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


@dataclass
class GenericSource(BaseSource):
    """This class defines gridded interfaces and plotting for all supported sources.

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
        """Top-level interface for gridded (NetCDF) maps."""
        self.logger.info("Generate plots.")

        if not self.config_manager.spec_data:
            plotter = SimplePlotter()
            self._simple_plots(plotter)
        else:
            if self.config_manager.compare and not self.config_manager.compare_diff:
                plotter = ComparisonPlotter(self.config_manager.compare_exp_ids)
                self._side_by_side_plots(plotter)
            elif self.config_manager.compare_diff:
                plotter = ComparisonPlotter(self.config_manager.compare_exp_ids)
                self._comparison_plots(plotter)
            elif self.config_manager.overlay:
                plotter = ComparisonPlotter(self.config_manager.overlay_exp_ids)
                self._side_by_side_plots(plotter)
            else:
                plotter = SinglePlotter()
                self._single_plots(plotter)

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

    def _side_by_side_plots(self, plotter):
        """Generate side-by-side comparison plots."""
        self.logger.info("Generating side-by-side plots")

    def _simple_plots(self, plotter):
        """Generate simple plots."""
        self.logger.info("Generating simple plots")

    def _single_plots(self, plotter):
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

            filename = params.get('filename')
            data_source = self.config_manager.pipeline.get_data_source(filename)

            if not data_source:
                self.logger.warning(f"No data source found in pipeline for {filename}")
                continue

            if hasattr(data_source, 'dataset') and data_source.dataset is not None:
                field_data = data_source.dataset.get(field_name)
            else:
                field_data = None

            if field_data is None:
                self.logger.warning(
                    f"Field {field_name} not found in data source for {filename}")
                continue

            plot_type = params.get('to_plot', ['xy'])[
                0]  # Default to 'xy' if not specified

            self.config_manager.findex = idx
            self.config_manager.pindex = idx
            self.config_manager.axindex = 0

            # Generate the plot using the plotter
            self.logger.info(f"Plotting {field_name} as {plot_type} plot")
            plotter.single_plots(self.config_manager, (
                field_data, None, None, field_name, plot_type, idx, None, None), level=0)

    def _comparison_plots(self, plotter):
        """Generate comparison plots."""
        self.logger.info("Generating comparison plots")

