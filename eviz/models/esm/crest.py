import warnings
import logging
import numpy as np
from dataclasses import dataclass
from eviz.lib.autoviz.figure import Figure
from eviz.models.source_base import GenericSource
from eviz.models.gridded_source import GriddedSource
from eviz.models.obs_source import ObsSource


warnings.filterwarnings("ignore")


@dataclass
class Crest(GenericSource):
    """ The Crest class contains definitions for handling CREST data. This is data
        produced by the Coupled Reusable Earth System Tensor-framework.
    """
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.comparison_plot = False
        self.source_name = 'crest'
        
        # Create instances of both source types
        self.gridded_handler = GriddedSource(self.config_manager)
        self.obs_handler = ObsSource(self.config_manager)
        
    def process_plot(self, data_array, field_name, file_index, plot_type):
        """Process a plot, delegating to the appropriate handler based on data type."""
        self.register_plot_type(field_name, plot_type)
        
        figure = Figure.create_eviz_figure(self.config_manager, plot_type)
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        
        is_obs = self._is_observational_data(data_array)
        
        # For observational data, extract extent information early
        if is_obs:
            self.logger.info(f"Processing {field_name} as observational data")
            handler = self.obs_handler
            
            # Extract and apply extent information
            extent = handler.get_data_extent(data_array)
            self.config_manager.ax_opts['extent'] = extent
            self.config_manager.ax_opts['central_lon'] = (extent[0] + extent[1]) / 2
            self.config_manager.ax_opts['central_lat'] = (extent[2] + extent[3]) / 2
        else:
            self.logger.info(f"Processing {field_name} as gridded data")
            handler = self.gridded_handler
        
        # Call the appropriate method on the selected handler
        if plot_type == 'xy':
            if hasattr(handler, '_process_xy_plot'):
                handler._process_xy_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_xy_plot not implemented for {handler.__class__.__name__}")
        elif plot_type == 'polar':
            if hasattr(handler, '_process_polar_plot'):
                handler._process_polar_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_polar_plot not implemented for {handler.__class__.__name__}")
        elif plot_type == 'xt':
            if hasattr(handler, '_process_xt_plot'):
                handler._process_xt_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_xt_plot not implemented for {handler.__class__.__name__}")
        elif plot_type == 'box':
            if hasattr(handler, '_process_box_plot'):
                handler._process_box_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_box_plot not implemented for {handler.__class__.__name__}")
        elif plot_type == 'line':
            if hasattr(handler, '_process_line_plot'):
                handler._process_line_plot(data_array, field_name, file_index, plot_type, figure)
            else:
                self.logger.warning(f"_process_line_plot not implemented for {handler.__class__.__name__}")


        
