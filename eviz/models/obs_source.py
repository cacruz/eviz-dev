from dataclasses import dataclass
import logging
import warnings
from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.models.source_base import GenericSource
import eviz.lib.autoviz.utils as pu
from eviz.lib.autoviz.figure import Figure

warnings.filterwarnings("ignore")


@dataclass
class ObsSource(GenericSource):
    """
    The ObsSource class provides specialized functionality for handling unstructured observation data.

    This class extends the GenericSource implementation to work specifically with unstructured data formats
    commonly used in observational datasets, including point measurements, irregular networks,
    and sparse spatial coverage. It implements methods for extracting, processing, and visualizing
    various representations of unstructured data, such as:

    - Scatter plots of point observations
    - Time series at specific locations
    - Spatial interpolation to regular grids when needed
    - Statistical aggregations by region, time period, or other dimensions

    Unlike the ESM modules which primarily handle gridded data formats with consistent coordinate
    systems, this class is optimized for irregular data structures. It provides specialized
    operations including:

    - Spatial binning and aggregation
    - Temporal resampling and aggregation
    - Quality control and filtering
    - Comparison with gridded model outputs
    - Visualization techniques appropriate for sparse data

    This class serves as a base for more specific observation implementations while providing
    comprehensive functionality for the most common unstructured data operations.
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
        
        This method is required by BaseSource but is now a no-op since data sources
        are managed by the pipeline. It's kept for backward compatibility.
        """
        pass

    def get_data_source(self, file_path):
        """
        Get a data source from the model.
        This method is required by BaseSource but now delegates to the pipeline.
        """
        return self.config_manager.pipeline.get_data_source(file_path)

    def load_data_sources(self):
        """
        Load data sources for the model.
        
        This method is required by BaseSource but is now a no-op since data sources
        are loaded by the ConfigurationAdapter. It's kept for backward compatibility.
        """
        pass

    def _simple_plots(self, plotter):
        """
        Generate simple plots for all fields in the dataset when no SPECS file is provided

        Args:
            plotter: The plotter object to use for generating plots
        """
        map_params = self.config_manager.map_params
        field_num = 0
        self.config_manager.findex = 0
        for i in map_params.keys():
            field_name = map_params[i]['field']
            filename = map_params[i]['filename']
            file_index = self.config_manager.get_file_index(filename)
            data_source = self.config_manager.pipeline.get_data_source(filename)
            if field_name not in data_source.dataset:
                continue
            self.config_manager.findex = file_index
            self.config_manager.pindex = field_num
            self.config_manager.axindex = 0
            field_data_array = data_source.dataset[field_name]

            for plot_type in map_params[i]['to_plot']:
                self.logger.info(f"Plotting {field_name}, {plot_type} plot")
                field_to_plot = self._get_field_for_simple_plot(field_data_array,
                                                              field_name, plot_type)
                plotter.simple_plot(self.config_manager, field_to_plot)
            field_num += 1

    def _get_field_for_simple_plot(self, data_array, field_name, plot_type):
        """
        Prepare data for simple plots. This method is used when no SPECS file is provided.
        It extracts the appropriate representation of data for the given plot type.
        
        Args:
            data_array: The data array to process
            field_name: The name of the field
            plot_type: The type of plot to generate
            
        Returns:
            tuple: A tuple containing the data, coordinates, field name, and plot type
        """
        if data_array is None:
            return None
            
        # For scatter plots (the most common for unstructured data)
        if 'sc' in plot_type:
            # Extract lat/lon coordinates
            if hasattr(data_array, 'lon') and hasattr(data_array, 'lat'):
                lon = data_array.lon
                lat = data_array.lat
                return data_array, lon, lat, field_name, plot_type
            else:
                # Try to find coordinate variables
                lon = None
                lat = None
                for coord_name in data_array.coords:
                    if coord_name.lower() in ['lon', 'longitude', 'long', 'x']:
                        lon = data_array[coord_name]
                    elif coord_name.lower() in ['lat', 'latitude', 'y']:
                        lat = data_array[coord_name]
                
                if lon is not None and lat is not None:
                    return data_array, lon, lat, field_name, plot_type
                else:
                    self.logger.error(f"Could not find lat/lon coordinates for {field_name}")
                    return None
        
        # For time series plots
        elif 'xt' in plot_type:
            # For time series, we typically want to return the data as is
            # The plotter will handle the time dimension
            return data_array, None, None, field_name, plot_type
            
        # For other plot types (less common for unstructured data)
        else:
            self.logger.warning(f"Plot type {plot_type} may not be suitable for unstructured data")
            return data_array, None, None, field_name, plot_type

    def _single_plots(self, plotter):
        """
        Generate single plots for each source and field according to configuration.
        
        Args:
            plotter: The plotter object to use for generating plots
        """
        self.logger.info("Generating single plots")

        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for single plotting.")
            return

        for idx, params in self.config_manager.map_params.items():
            field_name = params.get('field')
            if not field_name:
                continue

            filename = params.get('filename')
            data_source = self.config_manager.pipeline.get_data_source(filename)

            if not data_source or not hasattr(data_source, 'dataset') or data_source.dataset is None:
                continue

            if field_name not in data_source.dataset:
                continue

            self.config_manager.findex = idx
            self.config_manager.pindex = idx
            self.config_manager.axindex = 0

            field_data_array = data_source.dataset[field_name]
            plot_types = params.get('to_plot', ['sc'])  # Default to scatter for unstructured data
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
            for plot_type in plot_types:
                self._process_plot(field_data_array, field_name, idx, plot_type, plotter)

        if self.config_manager.make_gif:
            pu.create_gif(self.config_manager.config)

    def _process_plot(self, data_array, field_name, file_index, plot_type, plotter):
        """
        Process a single plot type for a given field.
        
        Args:
            data_array: The data array to process
            field_name: The name of the field
            file_index: The index of the file
            plot_type: The type of plot to generate
            plotter: The plotter object to use for generating plots
        """
        self.logger.info(f"Plotting {field_name}, {plot_type} plot")
        figure = Figure.create_eviz_figure(self.config_manager, plot_type)
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        # For unstructured data, we typically don't need to handle levels
        # But we do need to handle time if present
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        
        field_to_plot = self._get_field_to_plot(data_array, field_name, file_index,
                                              plot_type, figure, time_level=time_level_config)
        if field_to_plot:
            plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def _get_field_to_plot(self, data_array, field_name, file_index, plot_type, figure, time_level=None):
        """
        Prepare the data array and coordinates for plotting.
        
        Args:
            data_array: The data array to process
            field_name: The name of the field
            file_index: The index of the file
            plot_type: The type of plot to generate
            figure: The figure object
            time_level: The time level to use
            
        Returns:
            tuple: A tuple containing the data, coordinates, field name, plot type, file index, and figure
        """
        if data_array is None:
            self.logger.error(f"No data array provided for field {field_name}")
            return None

        # Handle time selection if needed
        if 'time' in data_array.dims:
            if isinstance(time_level, int) and time_level < data_array.time.size:
                data_array = data_array.isel(time=time_level)
            elif time_level == 'all':
                # For 'all' time levels, we might want to create an animation
                # But for now, just use the first time level
                data_array = data_array.isel(time=0)
            else:
                data_array = data_array.isel(time=0)

        # For scatter plots (most common for unstructured data)
        if 'sc' in plot_type:
            # Set map extent if not already set
            if 'extent' not in self.config_manager.ax_opts:
                if hasattr(data_array, 'lon') and hasattr(data_array, 'lat'):
                    lon = data_array.lon
                    lat = data_array.lat
                    self.config_manager.ax_opts['extent'] = [
                        lon.min().item() - 1, lon.max().item() + 1,
                        lat.min().item() - 1, lat.max().item() + 1
                    ]
                    self.config_manager.ax_opts['central_lon'] = lon.mean().item()
                    self.config_manager.ax_opts['central_lat'] = lat.mean().item()
            
            # Return the data and coordinates
            if hasattr(data_array, 'lon') and hasattr(data_array, 'lat'):
                return data_array, data_array.lon, data_array.lat, field_name, plot_type, file_index, figure
            else:
                # Try to find coordinate variables
                lon = None
                lat = None
                for coord_name in data_array.coords:
                    if coord_name.lower() in ['lon', 'longitude', 'long', 'x']:
                        lon = data_array[coord_name]
                    elif coord_name.lower() in ['lat', 'latitude', 'y']:
                        lat = data_array[coord_name]
                
                if lon is not None and lat is not None:
                    return data_array, lon, lat, field_name, plot_type, file_index, figure
                else:
                    self.logger.error(f"Could not find lat/lon coordinates for {field_name}")
                    return None
        
        # For time series plots
        elif 'xt' in plot_type:
            # For time series, we typically want to return the data as is
            # The plotter will handle the time dimension
            return data_array, None, None, field_name, plot_type, file_index, figure
            
        # For other plot types (less common for unstructured data)
        else:
            self.logger.warning(f"Plot type {plot_type} may not be suitable for unstructured data")
            return data_array, None, None, field_name, plot_type, file_index, figure

    def _comparison_plots(self, plotter):
        """
        Generate comparison plots for paired data sources according to configuration.
        
        Args:
            plotter: The plotter object to use for generating plots
        """
        self.logger.info("Generating comparison plots for unstructured data")
        # Implementation would be similar to GriddedSource but adapted for unstructured data
        # This would typically involve comparing point observations from different sources
        # or comparing observations to gridded model output
        
        # For now, we'll provide a basic implementation that can be expanded later
        if not self.config_manager.a_list or not self.config_manager.b_list:
            self.logger.error("a_list or b_list is empty, cannot perform comparison.")
            return
            
        # Basic implementation that can be expanded as needed
        self.logger.warning("Comparison plots for unstructured data are not fully implemented")
        
    def _side_by_side_plots(self, plotter):
        """
        Generate side-by-side comparison plots for paired data sources.
        
        Args:
            plotter: The plotter object to use for generating plots
        """
        self.logger.info("Generating side-by-side plots for unstructured data")
        # Implementation would be similar to GriddedSource but adapted for unstructured data
        
        # For now, we'll provide a basic implementation that can be expanded later
        if not self.config_manager.a_list or not self.config_manager.b_list:
            self.logger.error("a_list or b_list is empty, cannot perform side-by-side comparison.")
            return
            
        # Basic implementation that can be expanded as needed
        self.logger.warning("Side-by-side plots for unstructured data are not fully implemented")

    def process_data(self, filename, field_name):
        """
        Process data for a specific field from a file.
        
        Args:
            filename: The name of the file
            field_name: The name of the field
            
        Returns:
            The processed data
        """
        data_source = self.config_manager.pipeline.get_data_source(filename)
        if not data_source or not hasattr(data_source, 'dataset') or data_source.dataset is None:
            self.logger.error(f"No data source available for {filename}")
            return None
            
        if field_name not in data_source.dataset:
            self.logger.error(f"Field {field_name} not found in {filename}")
            return None
            
        return data_source.dataset
