from dataclasses import dataclass
import logging
import warnings
import numpy as np
import xarray as xr
from eviz.lib.data.pipeline.processor import DataProcessor
from eviz.models.source_base import GenericSource
import eviz.lib.autoviz.utils as pu
from eviz.lib.autoviz.figure import Figure

warnings.filterwarnings("ignore")


@dataclass
class GriddedSource(GenericSource):
    """
    The GriddedSource class provides specialized functionality for handling 
    gridded Earth System Model (ESM) data.

    This class extends the GenericSource implementation to work specifically with 
    structured grid data formats commonly used in ESMs, including 2D (lat-lon), 
    3D (lat-lon-time or lat-lon-level), and 4D  (lat-lon-level-time) datasets. 
    It implements methods for extracting, processing, and visualizing various 
    slices and projections of gridded data, such as:

    - Horizontal (XY) slices at specific vertical levels or times
    - Vertical (YZ) slices (zonal means)
    - Time series (XT) at points or averaged over regions
    - Hovmöller diagrams (TX) showing time-longitude evolution

    Unlike the observation modules which may handle both gridded and unstructured 
    data formats, this class is optimized specifically for regular grid structures 
    with consistent coordinate systems. It provides specialized grid-aware operations 
    including:

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

    def process_simple_plots(self, plotter):
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

    # Simple plots do not use configurations in SPECS file
    def _get_field_for_simple_plot(self, 
                                   data_array: xr.DataArray, 
                                   field_name: str,
                                   plot_type: str) -> tuple:
        """Prepare data for simple plots. This method is used when no SPECS file is provided.
        It extracts the appropriate slice of data for the given plot type.
        Args:
            data_array (xr.DataArray): The data array to process.
            field_name (str): The name of the field.
            plot_type (str): The type of plot to generate.
        
        Returns:
            tuple: A tuple containing the 2D data array, dimension names, and plot type.
            Returns None if the plot type is not supported.
        """
        if data_array is None:
            return None
        data2d = None
        dim1_name, dim2_name = None, None

        if 'xy' in plot_type:
            dim1_name = self.config_manager.get_model_dim_name('xc')
            dim2_name = self.config_manager.get_model_dim_name('yc')
            data2d = self._extract_xy_simple(data_array)
        elif 'yz' in plot_type:
            dim1_name = self.config_manager.get_model_dim_name('yc')
            dim2_name = self.config_manager.get_model_dim_name('zc')
            data2d = self._extract_yz_simple(data_array)
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

    def _extract_xy_simple(self, data_array: xr.DataArray) -> xr.DataArray:
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

    def _extract_yz_simple(self, data_array: xr.DataArray) -> xr.DataArray:
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

    def _process_xy_plot(self, data_array, field_name, file_index, plot_type, figure):
        """Process an XY plot."""
        levels = self.config_manager.get_levels(field_name, plot_type + 'plot')
        do_zsum = self.config_manager.ax_opts.get('zsum', False)

        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        num_times = data_array[tc_dim].size if tc_dim in data_array.dims else 1
        time_levels = range(num_times) if time_level_config == 'all' else [time_level_config]

        if not levels and not do_zsum:
            return

        if levels:
            self._process_level_plots(data_array, 
                                      field_name, 
                                      file_index, 
                                      plot_type, 
                                      figure, 
                                      time_levels, 
                                      levels)
        else:
            self._process_zsum_plots(data_array, 
                                     field_name, 
                                     file_index, 
                                     plot_type, 
                                     figure, 
                                     time_levels)

    def _process_level_plots(self, 
                             data_array, 
                             field_name, 
                             file_index, 
                             plot_type, 
                             figure, 
                             time_levels, 
                             levels):
        """Process plots for specific vertical levels."""
        self.logger.debug("Processing XY level plots")
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        has_vertical_dim = zc_dim and zc_dim in data_array.dims

        for level_val in levels.keys():
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

                # If the data doesn't have a vertical dimension, we can't select a level
                # In this case, we'll just use the data as is
                if not has_vertical_dim:
                    field_to_plot = self._prepare_field_to_plot(data_at_time, 
                                                                field_name, 
                                                                file_index, 
                                                                plot_type, 
                                                                figure, 
                                                                t)
                else:
                    field_to_plot = self._prepare_field_to_plot(data_at_time, 
                                                                field_name, 
                                                                file_index, 
                                                                plot_type, 
                                                                figure, 
                                                                t, 
                                                                level=level_val)

                if field_to_plot:
                    plot_result = self.create_plot(field_name, field_to_plot)                    
                    pu.print_map(self.config_manager, 
                                 plot_type, 
                                 self.config_manager.findex, 
                                 plot_result, 
                                 level=level_val)

    def _process_polar_plot(self, data_array, field_name, file_index, plot_type, figure):
        """Process plots for specific vertical levels."""
        levels = self.config_manager.get_levels(field_name, plot_type + 'plot')
        do_zsum = self.config_manager.ax_opts.get('zsum', False)

        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        num_times = data_array[tc_dim].size if tc_dim in data_array.dims else 1
        time_levels = range(num_times) if time_level_config == 'all' else [time_level_config]

        if not levels and not do_zsum:
            return
        
        self.logger.debug(f' -> Processing {len(time_levels)} time levels')
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        has_vertical_dim = zc_dim and zc_dim in data_array.dims

        for level_val in levels.keys():
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

                # If the data doesn't have a vertical dimension, we can't select a level
                # In this case, we'll just use the data as is
                if not has_vertical_dim:
                    field_to_plot = self._prepare_field_to_plot(data_at_time, 
                                                                field_name, 
                                                                file_index, 
                                                                plot_type, 
                                                                figure, 
                                                                t)
                else:
                    field_to_plot = self._prepare_field_to_plot(data_at_time, 
                                                                field_name, 
                                                                file_index, 
                                                                plot_type, 
                                                                figure, 
                                                                t, 
                                                                level=level_val)
                
                if field_to_plot:
                    plot_result = self.create_plot(field_name, field_to_plot)                    
                    pu.print_map(self.config_manager, 
                                 plot_type, 
                                 self.config_manager.findex, 
                                 plot_result, level=level_val)

    def _process_xt_plot(self, data_array, field_name, file_index, plot_type, figure):
        """Process an XT plot."""
        self.config_manager.level = None
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        if tc_dim in data_array.dims:
            num_times = data_array[tc_dim].size
            time_levels = range(num_times) if time_level_config == 'all' else [time_level_config]
        else:
            time_levels = [0]

        field_to_plot = self._prepare_field_to_plot(data_array, 
                                                    field_name, 
                                                    file_index, 
                                                    plot_type, 
                                                    figure, 
                                                    time_level=time_level_config)
        
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, 
                         plot_type, 
                         self.config_manager.findex, 
                         plot_result)
    
    def _process_tx_plot(self, data_array, field_name, file_index, plot_type, figure):
        """Process a TX (Hovmoller) plot."""
        self.config_manager.level = None
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        if tc_dim in data_array.dims:
            num_times = data_array[tc_dim].size
            time_levels = range(num_times) if time_level_config == 'all' else [
                time_level_config]
        else:
            time_levels = [0]

        field_to_plot = self._prepare_field_to_plot(data_array, 
                                                    field_name, 
                                                    file_index,
                                                    plot_type, 
                                                    figure,
                                                    time_level=time_level_config)
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, 
                         plot_type, 
                         self.config_manager.findex, 
                         plot_result)
         
    def _process_scatter_plot(self, data_array, field_name, file_index, plot_type, figure):
        """Process a scatter plot."""
        # Get x and y data for scatter plot
        x_data, y_data, z_data = self._get_scatter_data(data_array, field_name)
        
        if x_data is not None and y_data is not None:
            # Create field_to_plot tuple
            field_to_plot = (x_data, y_data, z_data, field_name, 'sc', file_index, figure)
            
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, 
                         plot_type, 
                         self.config_manager.findex, 
                         plot_result)
    
    def _get_scatter_data(self, data_array, field_name):
        """Get data for scatter plot.
        
        This is a placeholder implementation. You'll need to implement this
        based on your specific requirements.
        """
        # This is where you would extract x, y, and optionally z data
        # from data_array based on the field_name and configuration
        
        # For example:
        if field_name in self.config_manager.spec_data and 'scplot' in self.config_manager.spec_data[field_name]:
            sc_config = self.config_manager.spec_data[field_name]['scplot']
            
            # Get x field
            x_field = sc_config.get('x_field')
            if x_field and x_field in self.config_manager.pipeline.get_all_variables():
                x_data = self.config_manager.pipeline.get_variable(x_field)
            else:
                # Default x data
                x_data = np.arange(len(data_array))
            
            # Get y field
            y_field = sc_config.get('y_field')
            if y_field and y_field in self.config_manager.pipeline.get_all_variables():
                y_data = self.config_manager.pipeline.get_variable(y_field)
            else:
                # Use the input data_array as y data
                y_data = data_array
            
            # Get z field (optional)
            z_field = sc_config.get('z_field')
            if z_field and z_field in self.config_manager.pipeline.get_all_variables():
                z_data = self.config_manager.pipeline.get_variable(z_field)
            else:
                z_data = None
            
            return x_data, y_data, z_data
        
        # Default implementation: use data_array as y data, create x data
        x_data = np.arange(len(data_array))
        y_data = data_array
        z_data = None
        
        return x_data, y_data, z_data
    
    def _process_other_plot(self, 
                            data_array: xr.DataArray, 
                            field_name: str,
                            file_index: int, 
                            plot_type: str, 
                            figure):
        """Process non-xy and non-polar plot types."""
        self.config_manager.level = None
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        if tc_dim in data_array.dims:
            num_times = data_array[tc_dim].size
            # TODO: Handle yx_plot Gifs
            time_levels = range(num_times) if time_level_config == 'all' else [
                time_level_config]
        else:
            time_levels = [0]

        field_to_plot = self._prepare_field_to_plot(data_array, 
                                                    field_name, 
                                                    file_index,
                                                    plot_type, 
                                                    figure,
                                                    time_level=time_level_config)
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            pu.print_map(self.config_manager, 
                         plot_type, 
                         self.config_manager.findex, 
                         plot_result)
 
    def _process_zsum_plots(self, 
                            data_array: xr.DataArray, 
                            field_name: str,
                            file_index: int, 
                            plot_type: str, 
                            figure,
                            time_levels: list):
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
            field_to_plot = self._prepare_field_to_plot(data_at_time, 
                                                        field_name,
                                                        file_index, 
                                                        plot_type, 
                                                        figure, 
                                                        t)
            if field_to_plot:
                plot_result = self.create_plot(field_name, field_to_plot)
                pu.print_map(self.config_manager, 
                             plot_type, 
                             self.config_manager.findex, 
                             plot_result)

    def _process_xy_comparison_plots(self, 
                                     file_indices: tuple,
                                     current_field_index: int,
                                     field_name1: str, 
                                     field_name2: str, 
                                     plot_type: str,
                                     sdat1_dataset: xr.Dataset,
                                     sdat2_dataset: xr.Dataset):
        """Process comparison plots for xy or polar plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels

        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            return

        for level_val in levels:
            figure = Figure.create_eviz_figure(self.config_manager, 
                                               plot_type,
                                               nrows=nrows, 
                                               ncols=ncols)
            figure.set_axes()
            self.config_manager.level = level_val

            if figure.subplots == (3, 1):
                self._create_3x1_comparison_plot(file_indices,
                                                 current_field_index,
                                                 field_name1, 
                                                 field_name2, 
                                                 figure,
                                                 plot_type, 
                                                 sdat1_dataset, 
                                                 sdat2_dataset,
                                                 level_val)
            elif figure.subplots == (2, 2):
                self._create_2x2_comparison_plot(file_indices,
                                                 current_field_index,
                                                 field_name1, 
                                                 field_name2, 
                                                 figure,
                                                 plot_type, 
                                                 sdat1_dataset, 
                                                 sdat2_dataset,
                                                 level_val)

            self.config_manager.findex = file_index1
            pu.print_map(self.config_manager, 
                         plot_type, 
                         self.config_manager.findex, 
                         self.plot_result,
                        level=level_val)
            self.comparison_plot = False  # Reset comparison flag

    def _process_other_comparison_plots(self, file_indices: tuple,
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
        figure.set_axes()
        self.config_manager.level = None

        if figure.subplots == (3, 1):
            self._create_3x1_comparison_plot(file_indices, 
                                             current_field_index,
                                             field_name1, 
                                             field_name2, 
                                             figure,
                                             plot_type, 
                                             sdat1_dataset, 
                                             sdat2_dataset)
        elif figure.subplots == (2, 2):
            self._create_2x2_comparison_plot(file_indices, 
                                             current_field_index,
                                             field_name1, 
                                             field_name2, 
                                             figure,
                                             plot_type, 
                                             sdat1_dataset, 
                                             sdat2_dataset)

        self.config_manager.findex = file_index1
        pu.print_map(self.config_manager, 
                     plot_type, 
                     self.config_manager.findex, 
                     self.plot_result)
        self.comparison_plot = False  # Reset comparison flag

    def _create_3x1_comparison_plot(self, 
                                    file_indices, 
                                    current_field_index,
                                    field_name1, 
                                    field_name2, 
                                    figure,
                                    plot_type, 
                                    sdat1_dataset, 
                                    sdat2_dataset, 
                                    level=None):
        """Create a 3x1 comparison plot."""
        file_index1, file_index2 = file_indices

        # Plot the first dataset
        self._process_3x1_comparison_plot(file_index1, 
                                          current_field_index,
                                          field_name1, 
                                          figure, 
                                          0,
                                          sdat1_dataset[field_name1], 
                                          plot_type,
                                          level=level)

        # Plot the second dataset
        self._process_3x1_comparison_plot(file_index2, 
                                          current_field_index,
                                          field_name2, 
                                          figure, 
                                          1,
                                          sdat2_dataset[field_name2], 
                                          plot_type,
                                          level=level)

        # Plot the comparison (difference)
        self.comparison_plot = True
        self.config_manager.comparison_plot = True
        # For the comparison, we need to pass both datasets
        # The _process_comparison_plot method will need to handle this special case
        self._process_3x1_comparison_plot(file_index1, 
                                          current_field_index,
                                          field_name1, 
                                          figure, 
                                          2,
                                          (sdat1_dataset[field_name1], sdat2_dataset[field_name2]),
                                          plot_type, 
                                          level=level)

    def _create_2x2_comparison_plot(self, 
                                    file_indices, 
                                    current_field_index,
                                    field_name1, 
                                    field_name2, 
                                    figure,
                                    plot_type, 
                                    sdat1_dataset, 
                                    sdat2_dataset, 
                                    level=None):
        """Create a 2x2 comparison plot."""
        file_index1, file_index2 = file_indices

        # Plot the first dataset in the top-left
        self._process_2x2_comparison_plot(file_index1, 
                                          current_field_index,
                                          field_name1, 
                                          figure, 
                                          [0, 0], 
                                          0,
                                          sdat1_dataset[field_name1], 
                                          plot_type,
                                          level=level)

        # Plot the second dataset in the top-right
        self._process_2x2_comparison_plot(file_index2, 
                                          current_field_index,
                                          field_name2, 
                                          figure, 
                                          [0, 1], 
                                          1,
                                          sdat2_dataset[field_name2], 
                                          plot_type,
                                          level=level)

        # Plot comparison in the bottom row
        self.comparison_plot = True
        self.config_manager.comparison_plot = True
        # For the comparison, we need to pass both datasets
        self._process_2x2_comparison_plot(file_index1, 
                                          current_field_index,
                                          field_name1, 
                                          figure, 
                                          [1, 0], 
                                          2,
                                          (sdat1_dataset[field_name1], sdat2_dataset[field_name2]),
                                          plot_type, 
                                          level=level)

        # If extra field type is enabled, plot another comparison view
        # if self.config_manager.ax_opts.get('add_extra_field_type', False):
        self._process_2x2_comparison_plot(file_index1, 
                                          current_field_index,
                                          field_name1, 
                                          figure, 
                                          [1, 1], 
                                          2,
                                          (sdat1_dataset[field_name1], sdat2_dataset[field_name2]),
                                          plot_type, 
                                          level=level)

    def _process_3x1_comparison_plot(self, 
                                     file_index, 
                                     current_field_index,
                                     field_name, 
                                     figure, 
                                     ax_index, 
                                     data_array, 
                                     plot_type,
                                     level=None):
        """Process a comparison plot."""
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)

        # TODO: This is a hack to get the plot type to work with the
        #       side-by-side comparison plots.
        #       Need to fix this in the future.
        self.register_plot_type(field_name, plot_type)

        if ax_index == 2:  # Third panel in 3x1 layout is the difference
            self.config_manager.ax_opts['is_diff_field'] = True
        
        if ax_index == 2:
            # Compute and plot the difference field
            if len(self.data2d_list) == 2:
                data2d1, data2d2 = self.data2d_list
                proc = self.processor
                dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
                
                self.logger.debug(
                    f"Regridding {field_name} over {dim1_name} and {dim2_name} for difference plot")
                self.logger.debug(f"data2d1 shape: {data2d1.shape}, dims: {data2d1.dims}")
                self.logger.debug(f"data2d2 shape: {data2d2.shape}, dims: {data2d2.dims}")
                
                # Regrid data2d2 to match data2d1's grid
                try:
                    # Regrid data2d2 to match data2d1's grid
                    d2_on_d1 = proc.regrid(data2d1, data2d2, dims=(dim1_name, dim2_name))                    
                    diff_result = proc.compute_difference(data2d1, d2_on_d1)
                    
                    field_to_plot = (diff_result, 
                                    diff_result[dim1_name], 
                                    diff_result[dim2_name], 
                                    field_name, plot_type,
                                    file_index, figure)
                except Exception as e:
                    self.logger.error(f"Error computing difference: {e}")
                    field_to_plot = (xr.zeros_like(data2d1), 
                                    data2d1[dim1_name], 
                                    data2d1[dim2_name], 
                                    field_name, plot_type,
                                    file_index, figure)
            else:
                self.logger.error("Not enough data for difference plot")
                field_to_plot = None
            self.data2d_list = []
        else:
            # For the first two panels, plot as usual and store data for diff
            field_to_plot = self._prepare_field_to_plot(data_array, 
                                                        field_name,
                                                        file_index,
                                                        plot_type, 
                                                        figure,
                                                        time_level=time_level_config,
                                                        level=level)
            if field_to_plot:
                self.data2d_list.append(field_to_plot[0])
        if field_to_plot:
            self.plot_result = self.create_plot(field_name, field_to_plot)

    def _process_2x2_comparison_plot(self, 
                                     file_index, 
                                     current_field_index,
                                     field_name, 
                                     figure, 
                                     gsi, 
                                     ax_index, 
                                     data_array, 
                                     plot_type,
                                     level=None):
        """Process a 2x2 comparison plot."""
        ax = figure.get_axes()
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)

        # TODO: This is a hack to get the plot type to work with the
        #       side-by-side comparison plots.
        #       Need to fix this in the future.
        self.register_plot_type(field_name, plot_type)

        # Initialize ax_opts BEFORE setting flags
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        # Set difference field flag if this is a comparison panel (bottom row)
        if gsi[0] == 1:  # Bottom row in 2x2 layout is for differences
            self.config_manager.ax_opts['is_diff_field'] = True
            # Set extra field type flag for the bottom-right panel
            if gsi[1] == 1:
                self.config_manager.ax_opts['add_extra_field_type'] = True

        figure.set_ax_opts_diff_field(ax[ax_index])
        
        # Handle difference calculation for bottom row panels
        if isinstance(data_array, tuple):
            if len(self.data2d_list) == 2:
                data2d1, data2d2 = self.data2d_list
                proc = self.processor
                dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
                                
                try:
                    # Regrid data2d2 to match data2d1's grid
                    d2_on_d1 = proc.regrid(data2d1, data2d2, dims=(dim1_name, dim2_name))
                    if self.config_manager.ax_opts['add_extra_field_type'] :
                        diff_result = proc.compute_difference(data2d1, 
                                                              d2_on_d1, 
                                                              method=self.config_manager.extra_diff_plot)
                    else:
                        diff_result = proc.compute_difference(data2d1, 
                                                              d2_on_d1)
                    
                    self.logger.debug(
                        f"Diff data min/max: {diff_result.min().values}/{diff_result.max().values}")
                    
                    # Create field_to_plot tuple with the difference result
                    field_to_plot = (diff_result, 
                                    diff_result[dim1_name], 
                                    diff_result[dim2_name], 
                                    field_name, plot_type,
                                    file_index, figure)
                                    
                except Exception as e:
                    self.logger.error(f"Error computing difference: {e}")
                    # Create a dummy field with zeros if calculation fails
                    field_to_plot = (xr.zeros_like(data2d1), 
                                    data2d1[dim1_name], 
                                    data2d1[dim2_name], 
                                    field_name, plot_type,
                                    file_index, figure)
            else:
                self.logger.error("Not enough data for difference plot")
                field_to_plot = None
        else:
            # For the top row panels, plot as usual and store data for diff
            field_to_plot = self._prepare_field_to_plot(data_array, 
                                                        field_name,
                                                        file_index,
                                                        plot_type, 
                                                        figure,
                                                        time_level=time_level_config,
                                                        level=level)
            if field_to_plot and field_to_plot[0] is not None:
                self.data2d_list.append(field_to_plot[0])

        if field_to_plot:
            self.plot_result = self.create_plot(field_name, field_to_plot)

    def _process_xy_side_by_side_plots(self, 
                                       current_field_index,
                                       field_name1, 
                                       field_name2, 
                                       plot_type, 
                                       sdat1_dataset,
                                       sdat2_dataset):
        """Process side-by-side comparison plots for xy or polar plot types."""
        num_plots = len(self.config_manager.compare_exp_ids)
        nrows = 1
        ncols = num_plots

        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            return

        for level_val in levels:
            figure = Figure.create_eviz_figure(self.config_manager, plot_type,
                                               nrows=nrows, ncols=ncols)
            figure.set_axes()
            self.config_manager.level = level_val

            # Store domain information for regional plots if available
            self.config_manager.is_regional = hasattr(self, 'source_name') and self.source_name in ['lis',
                                                                                'wrf']
            if self.config_manager.is_regional:
                if hasattr(sdat1_dataset, 'lon') and hasattr(sdat1_dataset, 'lat'):
                    self.lon = sdat1_dataset.lon
                    self.lat = sdat1_dataset.lat

            self._create_xy_side_by_side_plot(current_field_index,
                                              field_name1, 
                                              field_name2, 
                                              figure,
                                              plot_type, 
                                              sdat1_dataset, 
                                              sdat2_dataset,
                                              level_val)

            pu.print_map(self.config_manager, 
                         plot_type, 
                         self.config_manager.findex, 
                         self.plot_result,
                         level=level_val)
            
    def _create_xy_side_by_side_plot(self, 
                                     current_field_index,
                                     field_name1, 
                                     field_name2, 
                                     figure,
                                     plot_type, 
                                     sdat1_dataset, 
                                     sdat2_dataset, 
                                     level=None):
        """
        Create a side-by-side comparison plot for the given data.
        
        The layout is:
        - Left subplot: First dataset
        - Middle subplot: Second dataset
        - Right subplot: Third dataset (if present)
        """
        num_plots = len(self.config_manager.compare_exp_ids)
        self.comparison_plot = False

        # Plot first dataset (from a_list)
        if self.config_manager.a_list:
            self._process_single_side_by_side_plot(self.config_manager.a_list[0],
                                            current_field_index,
                                            field_name1, 
                                            figure, 
                                            0,
                                            sdat1_dataset[field_name1], 
                                            plot_type,
                                            level=level)

        # Plot remaining datasets (from b_list)
        for i, file_idx in enumerate(self.config_manager.b_list, start=1):
            if i < num_plots:  # Only plot if we have a corresponding axis
                map_params = self.config_manager.map_params.get(file_idx)
                filename = map_params.get('filename')
                data_source = self.config_manager.pipeline.get_data_source(filename)
                dataset = data_source.dataset

                self._process_single_side_by_side_plot(file_idx,
                                                current_field_index,
                                                field_name2, 
                                                figure, 
                                                i,
                                                dataset[field_name2], 
                                                plot_type,
                                                level=level)


    def _process_other_side_by_side_plots(self, 
                                          current_field_index,
                                          field_name1, 
                                          field_name2, 
                                          plot_type, 
                                          sdat1_dataset,
                                          sdat2_dataset):
        """Process side-by-side comparison plots for xy or polar plot types."""
        num_plots = len(self.config_manager.compare_exp_ids)
        nrows = 1
        ncols = num_plots

        use_overlay = self.config_manager.should_overlay_plots(field_name1, plot_type[:2])
        if use_overlay:
            ncols = 1  # Use a single plot for overlay


        figure = Figure.create_eviz_figure(self.config_manager, plot_type,
                                            nrows=nrows, ncols=ncols)
        figure.set_axes()
        self.config_manager.level = None

        # Plot first dataset (from a_list)
        if self.config_manager.a_list:
            self._process_single_side_by_side_plot(self.config_manager.a_list[0],
                                                   current_field_index,
                                                   field_name1, 
                                                   figure, 
                                                   0,
                                                   sdat1_dataset[field_name1], 
                                                   plot_type)

        # Plot remaining datasets (from b_list)
        for i, file_idx in enumerate(self.config_manager.b_list, start=1):
            if i < num_plots:  # Only plot if we have a corresponding axis
                map_params = self.config_manager.map_params.get(file_idx)
                if not map_params:
                    continue
                    
                filename = map_params.get('filename')
                if not filename:
                    continue
                data_source = self.config_manager.pipeline.get_data_source(filename)
                if not data_source or not hasattr(data_source, 'dataset') or data_source.dataset is None:
                    continue
                    
                dataset = data_source.dataset
                
                # Use the correct axis index based on overlay mode
                axis_index = 0 if use_overlay else i
                
                self._process_single_side_by_side_plot(file_idx,
                                                current_field_index,
                                                field_name2, 
                                                figure, 
                                                axis_index,
                                                dataset[field_name2], 
                                                plot_type)

        pu.print_map(self.config_manager, 
                     plot_type, 
                     self.config_manager.findex, 
                     self.plot_result)

    def _process_single_side_by_side_plot(self, file_index, 
                                          current_field_index,
                                          field_name, 
                                          figure, 
                                          ax_index, 
                                          data_array, 
                                          plot_type,
                                          level=None):
        """Process a single plot for side-by-side comparison."""
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        time_level_config = self.config_manager.ax_opts.get('time_lev', -1)

        # TODO: This is a hack to get the plot type to work with the
        #       side-by-side comparison plots.
        #       Need to fix this in the future.
        self.register_plot_type(field_name, plot_type)

        # Track which dataset we're currently plotting and how many total
        if self.config_manager.should_overlay_plots(field_name, plot_type[:2]):
            if file_index in self.config_manager.a_list:
                dataset_index = self.config_manager.a_list.index(file_index)
            elif file_index in self.config_manager.b_list:
                dataset_index = len(self.config_manager.a_list) + self.config_manager.b_list.index(file_index)
            else:
                dataset_index = 0
                
            self.config_manager.current_dataset_index = dataset_index
            self.config_manager.total_datasets = len(self.config_manager.a_list) + len(self.config_manager.b_list)
            
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        
        field_to_plot = self._prepare_field_to_plot(data_array, 
                                                    field_name,
                                                    file_index,
                                                    plot_type, 
                                                    figure,
                                                    time_level=time_level_config,
                                                    level=level)

        if field_to_plot and field_to_plot[0] is not None:
            self.data2d_list.append(field_to_plot[0])
        if field_to_plot:
            self.plot_result = self.create_plot(field_name, field_to_plot)
