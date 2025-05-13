from dataclasses import dataclass
import logging
import warnings
import sys
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from collections.abc import Iterable

from eviz.lib.data.processor import Interp
from eviz.models.root import Root
from eviz.lib.data.data_utils import apply_conversion, apply_mean, apply_zsum
import eviz.lib.autoviz.plot_utils as pu
from eviz.lib.autoviz.figure import Figure
import multiprocessing

warnings.filterwarnings("ignore")


@dataclass
class Generic(Root):
    """ The generic class contains definitions for handling generic ESM data, that is 2D, 3D, and 4D
     field data. This is typically not the case for observational data which may be unstructured and very
     non-standard in its internal arrangement.
     Specific model functionality should be overridden in subclasses.
    """
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

    def _load_source_data(self, source_name, filename):
        """Load data from the specified source and filename."""
        # First, try to load from the specific file
        reader = self.config_manager.get_reader_for_file(source_name, filename)
        if reader:
            try:
                # Try the new data source interface first (load_data)
                if hasattr(reader, 'load_data'):
                    self.logger.debug(f"Using load_data method for {filename}")
                    reader.load_data(filename)
                    if hasattr(reader, 'dataset'):
                        return {'vars': reader.dataset}
                
                # Fall back to the old reader interface (read_data)
                if hasattr(reader, 'read_data'):
                    self.logger.debug(f"Using read_data method for {filename}")
                    source_data = reader.read_data(filename)
                    if source_data:
                        return source_data
            except Exception as e:
                self.logger.error(f"Error loading data from {filename}: {e}")
        
        # If that fails or if we need to combine data from multiple files
        self.logger.info(f"Attempting to integrate data from multiple files for {source_name}")
        integrated_data = self._integrate_datasets(source_name)
        
        if integrated_data:
            # Convert xarray Dataset to the expected format
            return {'vars': integrated_data}
        
        self.logger.error(f"Failed to load data for {source_name} from {filename}")
        return None    

    def _integrate_datasets(self, source_name, field_name=None):
        """
        Integrate datasets from multiple files for a single source.
        
        Args:
            source_name (str): The source name
            field_name (str, optional): If provided, ensures this field exists in the result
            
        Returns:
            xr.Dataset: The integrated dataset or None if integration failed
        """
        # Get all files for this source
        file_paths = []
        for file_idx, file_entry in self.config_manager.file_list.items():
            if file_entry.get('source_name', None) == source_name:
                file_paths.append(file_entry['filename'])
        
        # No files found
        if not file_paths:
            self.logger.warning(f"No files found for source {source_name}")
            return None
        
        # Use the integrator to combine datasets
        integrated_data = self.config_manager.integrator.integrate_datasets(source_name, file_paths)
        
        # Check if the requested field exists in the integrated dataset
        if field_name and integrated_data is not None:
            if field_name not in integrated_data:
                self.logger.warning(f"Field {field_name} not found in integrated dataset")
                return None
        
        return integrated_data


    # SIMPLE PLOTS METHODS (no SPECS file)
    #--------------------------------------------------------------------------
    def _simple_plots(self, plotter):
        map_params = self.config_manager.map_params
        field_num = 0
        self.config_manager.findex = 0
        for i in map_params.keys():
            field_name = map_params[i]['field']
            source_name = map_params[i]['source_name']
            filename = map_params[i]['filename']
            file_index = self.config_manager.get_file_index(filename)
            source_data = self._load_source_data(source_name, filename)
            if field_name not in source_data['vars']:
                continue
            self.config_manager.findex = file_index
            self.config_manager.pindex = field_num
            self.config_manager.axindex = 0
            for pt in map_params[i]['to_plot']:
                self.logger.info(f"Plotting {field_name}, {pt} plot")
                field_to_plot = self._get_field_for_simple_plot(source_data, field_name, pt)
                plotter.simple_plot(self.config_manager, field_to_plot)
            field_num += 1

    # Simple plots do not use configurations in SPECS file
    def _get_field_for_simple_plot(self, model_data, field_name, plot_type):
        d = model_data['vars']
        if 'xy' in plot_type:
            dim1 = self.config_manager.get_model_dim_name('xc')
            dim2 = self.config_manager.get_model_dim_name('yc')
            data2d = self._get_xy_simple(d, field_name, 0)
        elif 'yz' in plot_type:
            dim1 = self.config_manager.get_model_dim_name('yc')
            dim2 = self.config_manager.get_model_dim_name('zc')
            data2d = self._get_yz_simple(d, field_name)
        else:
            self.logger.error(f'Plot type [{plot_type}] error: Either specify in SPECS file or create plot type.')
            sys.exit()
        coords = data2d.coords
        return data2d, coords[dim1], coords[dim2], field_name, plot_type

    def _get_xy_simple(self, d, name, level):
        """ Extract XY slice from N-dim data field"""
        if d is None:
            return
        data2d = d[name].squeeze()
        # Hackish
        if len(data2d.shape) == 4:
            data2d = data2d.isel(time=0)
        if len(data2d.shape) == 3:
            if self.config_manager.get_model_dim_name('tc') in data2d.dims:
                data2d = data2d.isel(time=0)
            else:
                data2d = data2d.isel(lev=0)
        return data2d

    def _get_yz_simple(self, d, name):
        """ Create YZ slice from N-dim data field"""
        if d is None:
            return
        data2d = d[name].squeeze()
        if len(data2d.shape) == 4:
            data2d = data2d.isel(time=0)
        data2d = data2d.mean(dim=self.config_manager.get_model_dim_name('xc'))
        return data2d

    def _get_model_dim_name(self, source_name: str, dim_name: str):
        try:
            dim = self.config_manager.meta_coords[dim_name][source_name]
            return dim
        except KeyError:
            return None

    # SINGLE PLOTS METHODS (using SPECS file)
    #--------------------------------------------------------------------------
    def _single_plots(self, plotter):
        """Generate single plots for each source and field according to configuration."""
        for source_idx, source_name in enumerate(self.config_manager.source_names):
            # self.config_manager.config.ds_index = source_idx
            self._process_source_fields(source_name, plotter)

        if self.config_manager.make_gif:
            pu.create_gif(self.config_manager.config)

    def _process_source_fields(self, source_name, plotter):
        """Process all fields for a given source."""
        field_num = 0
        for param_key, param_config in self.config_manager.map_params.items():
            if param_config['source_name'] != source_name:
                continue

            field_name = param_config['field']
            filename = param_config['filename']

            # Load source data
            source_data = self._load_source_data(source_name, filename)
            # if not source_data:
            #     return

            # Update configuration
            self.config_manager.findex = field_num
            self.config_manager.pindex = field_num
            self.config_manager.axindex = 0

            for plot_type in param_config['to_plot']:
                self._process_plot(source_data, field_name, field_num, plot_type, plotter)

            field_num += 1

    def _process_plot(self, source_data, field_name, field_num, plot_type, plotter):
        """Process a single plot type for a given field."""
        self.logger.info(f"Plotting {field_name}, {plot_type} plot")
        figure = Figure.create_eviz_figure(self.config_manager, plot_type)  
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        if 'xy' in plot_type or 'po' in plot_type:
            self._process_xy_plot(source_data, field_name, field_num, plot_type,
                                  figure, plotter)
        else:
            self._process_other_plot(source_data, field_name, field_num, plot_type,
                                     figure, plotter)

    def _process_xy_plot(self, source_data, field_name, field_num, plot_type, figure, plotter):
        """Process xy or polar plot types."""
        # Get vertical levels to process
        levels = self.config_manager.get_levels(field_name, plot_type + 'plot')
        do_zsum = self.config_manager.ax_opts['zsum']
        # Get time levels to process
        time_level = self.config_manager.ax_opts['time_lev']
        num_times = 1 if time_level != 'all' else np.size(source_data['vars'][field_name].time)
        time_levels = range(num_times)

        if not levels and not do_zsum:
            self.logger.warning(f' -> No levels specified for {field_name}')
            return

        if levels:
            self._process_level_plots(source_data, field_name, field_num, plot_type,
                                      figure, time_levels, levels, plotter)
        else:
            self._process_zsum_plots(source_data, field_name, field_num, plot_type,
                                     figure, time_levels, plotter)

    def _process_level_plots(self, source_data, field_name, field_num, plot_type, figure,
                             time_levels, levels, plotter):
        """Process plots for specific vertical levels."""
        self.logger.info(f' -> Processing {len(time_levels)} time levels')
        for level in levels.keys():
            self.config_manager.level = level
            for t in time_levels:
                self._set_time_config(t, source_data['vars'][field_name])
 
                # Create a new figure for each level to avoid reusing axes
                figure = Figure.create_eviz_figure(self.config_manager, plot_type)
                self.config_manager.ax_opts = figure.init_ax_opts(field_name)

                ax = figure.get_axes()
                field_to_plot = self._get_field_to_plot(ax, source_data, field_name,
                                                        field_num, plot_type, figure, t,
                                                        level=level)

                plotter.single_plots(self.config_manager, field_to_plot=field_to_plot,
                                     level=level)

                pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure,
                             level=level)

    def _process_zsum_plots(self, source_data, field_name, field_num, plot_type, figure,
                            time_levels, plotter):
        """Process plots with vertical summation."""
        self.config_manager.level = None
        for t in time_levels:
            self._set_time_config(t, source_data['vars'][field_name])
            field_to_plot = self._get_field_to_plot(source_data, field_name, field_num,
                                                    plot_type, figure, t)
            plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def _process_other_plot(self, source_data, field_name, field_num, plot_type, figure,
                            plotter):
        """Process non-xy and non-po plot types."""
        self.config_manager.level = None
        # Get time levels to process
        time_level = self.config_manager.ax_opts['time_lev']
 
        # TODO: Handle yx_plot Gifs
        num_times = 1 if time_level != 'all' else np.size(source_data['vars'][field_name].time)
        time_levels = range(num_times)

        ax = figure.get_axes()
        field_to_plot = self._get_field_to_plot(ax, source_data, field_name, field_num,
                                                plot_type, figure, time_level=time_level)
        plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
        pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def _get_field_to_plot(self, ax, source_data, field_name, file_index, 
                        plot_type, figure, time_level, level=None) -> tuple:
        # Try to get the field from source_data
        if source_data and 'vars' in source_data and field_name in source_data['vars']:
            # Use existing data
            dim1, dim2 = self.config_manager.get_dim_names(plot_type)
            data2d = None
            if 'yz' in plot_type:
                data2d = self._get_yz(source_data, field_name, time_lev=time_level)
            elif 'xt' in plot_type:
                data2d = self._get_xt(source_data, field_name, time_lev=time_level)
            elif 'tx' in plot_type:
                data2d = self._get_tx(source_data, field_name, level=None, time_lev=time_level)
            elif 'xy' in plot_type or 'polar' in plot_type:
                data2d = self._get_xy(source_data, field_name, level=level, time_lev=time_level)
            else:
                pass
        else:
            # Try to get the field from any available source using the integrator
            self.logger.info(f"Field {field_name} not found in primary data source. Trying other sources...")
            source_name = self.config_manager.source_names[self.config_manager.ds_index]
            data_array = self.config_manager.integrator.get_variable_from_any_source(source_name, field_name)
            
            if data_array is not None:
                # Process the data array based on plot type
                dim1, dim2 = self.config_manager.get_dim_names(plot_type)
                data2d = data_array
                
                # Further processing based on plot type...
                if 'yz' in plot_type:
                    data2d = data_array.mean(dim=self.config_manager.get_model_dim_name('xc'))
                elif 'xt' in plot_type:
                    # Process for xt plot
                    pass
                elif 'tx' in plot_type:
                    # Process for tx plot
                    pass
                elif 'xy' in plot_type or 'polar' in plot_type:
                    # Process for xy or polar plot
                    if level is not None:
                        data2d = data_array.sel(lev=level)
                
                # Apply any necessary conversions
                data2d = self._apply_conversions(data2d, field_name)
            else:
                self.logger.error(f"Field {field_name} not found in any data source")
                return None
        
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        return data2d, data2d[dim1].values, data2d[dim2].values, field_name, plot_type, file_index, figure, ax

    # COMPARE_DIFF METHODS (always need SPECS file)
    #--------------------------------------------------------------------------
    def _comparison_plots(self, plotter):
        """Generate comparison plots for paired data sources according to configuration."""
        current_field_index = 0

        # map1 and map2 are indices, not maps
        for idx1, idx2 in zip(self.config_manager.a_list, self.config_manager.b_list):
            # Get map parameters for these indices
            map1 = self.config_manager.config.map_params[idx1]
            map2 = self.config_manager.config.map_params[idx2]

            # Load data from both sources
            source_data_pair = self._load_comparison_data(map1, map2)
            if not source_data_pair:
                continue

            sdat1, sdat2 = source_data_pair
            sdat = (sdat1, sdat2)

            # Determine file indices
            source_name1, source_name2 = map1['source_name'], map2['source_name']
            filename1, filename2 = map1['filename'], map2['filename']
            file_indices = self._get_file_indices_compare(source_name1, source_name2, filename1,
                                                filename2)
            # Process each plot type
            field_name1, field_name2 = map1['field'], map2['field']
            self.field_names = (field_name1, field_name2)

            for pt1, pt2 in zip(map1['to_plot'], map2['to_plot']):
                plot_type = pt1  # Using the first plot type
                self.logger.info(
                    f"Plotting {field_name1} vs {field_name2}, {plot_type} plot")
                self.data2d_list = []

                if 'xy' in plot_type or 'po' in plot_type:
                    self._process_xy_comparison_plots(plotter, file_indices,
                                                    current_field_index,
                                                    field_name1, field_name2, plot_type,
                                                    sdat1, sdat2, sdat)
                else:
                    self._process_other_comparison_plots(plotter, file_indices,
                                                    current_field_index,
                                                    field_name1, field_name2,
                                                    plot_type, sdat1, sdat2, sdat)

            current_field_index += 1

    def _load_comparison_data(self, map1, map2):
        """Load data from both sources for comparison."""
        source_name1, source_name2 = map1['source_name'], map2['source_name']
        filename1, filename2 = map1['filename'], map2['filename']

        # Read data from files to be compared using appropriate readers
        reader1 = self.config_manager.get_reader_for_file(source_name1, filename1)
        reader2 = self.config_manager.get_reader_for_file(source_name2, filename2)
        
        if not reader1 or not reader2:
            self.logger.error("No suitable readers found")
            sys.exit()
        
        # Load data from the first reader
        sdat1 = None
        try:
            # Try the new data source interface first (load_data)
            if hasattr(reader1, 'load_data'):
                self.logger.debug(f"Using load_data method for {filename1}")
                reader1.load_data(filename1)
                if hasattr(reader1, 'dataset'):
                    sdat1 = {'vars': reader1.dataset}
            
            # Fall back to the old reader interface (read_data)
            if not sdat1 and hasattr(reader1, 'read_data'):
                self.logger.debug(f"Using read_data method for {filename1}")
                sdat1 = reader1.read_data(filename1)
        except Exception as e:
            self.logger.error(f"Error loading data from {filename1}: {e}")
        
        # Load data from the second reader
        sdat2 = None
        try:
            # Try the new data source interface first (load_data)
            if hasattr(reader2, 'load_data'):
                self.logger.debug(f"Using load_data method for {filename2}")
                reader2.load_data(filename2)
                if hasattr(reader2, 'dataset'):
                    sdat2 = {'vars': reader2.dataset}
            
            # Fall back to the old reader interface (read_data)
            if not sdat2 and hasattr(reader2, 'read_data'):
                self.logger.debug(f"Using read_data method for {filename2}")
                sdat2 = reader2.read_data(filename2)
        except Exception as e:
            self.logger.error(f"Error loading data from {filename2}: {e}")

        if not sdat1 or not sdat2:
            self.logger.error("Cannot continue - failed to load data from one or both sources")
            sys.exit()

        return sdat1, sdat2
    
    def _get_file_indices_compare(self, source_name1, source_name2, filename1, filename2):
        """Determine file indices for comparison."""
        if source_name1 == source_name2:
            return 0, 1  # Same data source
        else:
            # Change from self.config.get_comp_file_index to self.config_manager.get_file_index or similar
            file_index1 = self.config_manager.get_file_index(filename1)
            file_index2 = self.config_manager.get_file_index(filename2)
            return file_index1, file_index2
        
    def _process_xy_comparison_plots(self, plotter, file_indices, current_field_index,
                                     field_name1, field_name2, plot_type, sdat1, sdat2,
                                     sdat):
        """Process comparison plots for xy or polar plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels

        # Get levels for the plots
        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            self.logger.warning(f' -> No levels specified for {field_name1}')
            return

        for level in levels:
            figure = Figure.create_eviz_figure(self.config_manager, plot_type, nrows=nrows, ncols=ncols)  
            # ax = figure.get_axes()
            ax = figure.get_fig_ax()
            axes_shape = figure.get_gs_geometry()
            self.config_manager.level = level

            if axes_shape == (3, 1):
                self._create_3x1_comparison_plot(plotter, file_indices,
                                                 current_field_index,
                                                 field_name1, field_name2, figure, ax,
                                                 plot_type, sdat1, sdat2, sdat, level)
            elif axes_shape == (2, 2):
                self._create_2x2_comparison_plot(plotter, file_indices,
                                                 current_field_index,
                                                 field_name1, field_name2, figure,
                                                 plot_type, sdat1, sdat2, sdat, level)

            # # Set findex to file_index1 before calling print_map
            # self.config_manager.config._findex = file_index1
            # pu.print_map(self.config_manager, plot_type, self.config_manager.config._findex, figure, level=level)
            self.comparison_plot = False

    def _process_other_comparison_plots(self, plotter, file_indices, current_field_index,
                                        field_name1, field_name2, plot_type, sdat1, sdat2,
                                        sdat):
        """Process comparison plots for other plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels

        figure = Figure.create_eviz_figure(self.config_manager, plot_type, nrows=nrows, ncols=ncols)  
        axes_shape = figure.subplots
        ax = figure.get_axes()
        self.config_manager.level = None

        if axes_shape == (3, 1):
            self._create_3x1_comparison_plot(plotter, file_indices, current_field_index,
                                             field_name1, field_name2, figure, ax,
                                             plot_type, sdat1, sdat2, sdat)
        elif axes_shape == (2, 2):
            self._create_2x2_comparison_plot(plotter, file_indices, current_field_index,
                                             field_name1, field_name2, figure,
                                             plot_type, sdat1, sdat2, sdat)


    def _create_3x1_comparison_plot(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, figure, ax,
                                    plot_type, sdat1, sdat2, sdat, level=None):
        """Create a 3x1 comparison plot."""
        file_index1, file_index2 = file_indices

        # Plot the first dataset
        self._process_comparison_plot(plotter, file_index1, current_field_index,
                                      field_name1,
                                      figure, ax, 0, sdat1, plot_type, level=level)

        # Plot the second dataset
        self._process_comparison_plot(plotter, file_index2, current_field_index,
                                      field_name2,
                                      figure, ax, 1, sdat2, plot_type, level=level)

        # Plot the comparison
        self.comparison_plot = True
        self._process_comparison_plot(plotter, file_index1, current_field_index,
                                      field_name1,
                                      figure, ax, 2, sdat, plot_type, level=level)

    def _create_2x2_comparison_plot(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, figure,
                                    plot_type, sdat1, sdat2, sdat, level=None):
        """Create a 2x2 comparison plot."""
        file_index1, file_index2 = file_indices

        # Plot the first dataset in the top-left
        self._process_comparison_plot_2x2(plotter, file_index1, current_field_index,
                                          field_name1,
                                          figure, [0, 0], 0, sdat1, plot_type,
                                          level=level)

        # Plot the second dataset in the top-right
        self._process_comparison_plot_2x2(plotter, file_index2, current_field_index,
                                          field_name2,
                                          figure, [0, 1], 1, sdat2, plot_type,
                                          level=level)

        # Plot comparison in the bottom row
        self.comparison_plot = True
        self._process_comparison_plot_2x2(plotter, file_index1, current_field_index,
                                          field_name1,
                                          figure, [1, 0], 2, sdat, plot_type, level=level)
        self._process_comparison_plot_2x2(plotter, file_index1, current_field_index,
                                          field_name1,
                                          figure, [1, 1], 2, sdat, plot_type, level=level)

    def _process_comparison_plot(self, plotter, file_index, current_field_index, field_name, figure, ax, ax_index,
                           source_data, pt, level=None):
        """Process a comparison plot."""
        # Set findex on the config object instead of directly on config_manager
        self.config_manager.config.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        figure.set_ax_opts_diff_field(ax[ax_index])
        
        # Handle source_data differently based on its type
        if isinstance(source_data, tuple):
            # If source_data is a tuple (from comparison plots), use it directly
            field_to_plot = self._get_field_to_plot_compare(source_data, field_name, file_index,
                                                        pt, figure, level=level)
        else:
            # Check if we have the field in the source data
            if source_data and 'vars' in source_data and field_name in source_data['vars']:
                field_to_plot = self._get_field_to_plot_compare(source_data, field_name, file_index,
                                                            pt, figure, level=level)
            else:
                # Try to integrate data from multiple sources
                source_name = self.config_manager.file_list[file_index].get('source_name')
                if source_name:
                    integrated_data = self._integrate_datasets(source_name, field_name)
                    if integrated_data:
                        source_data = {'vars': integrated_data}
                        field_to_plot = self._get_field_to_plot_compare(source_data, field_name, file_index,
                                                                    pt, figure, level=level)
                    else:
                        self.logger.error(f"Failed to load integrated data for {field_name}")
                        return
                else:
                    self.logger.error(f"No source name found for file index {file_index}")
                    return
        
        plotter.comparison_plots(self.config_manager, field_to_plot, level=level)
        pu.print_map(self.config_manager, pt, file_index, figure, level=level)

    def _process_comparison_plot_2x2(self, plotter, file_index, current_field_index, field_name, figure, gsi, ax_index,
                                     source_data, pt, level=None):
        # Ensure figure and axes are properly initialized
        fig, axes = figure.get_fig_ax()
        ax1 = axes[gsi[0], gsi[1]] if isinstance(axes, list) else plt.subplot(figure.gs[gsi[0], gsi[1]])
        figure.set_ax_opts_diff_field(ax1)
        # Set findex on the config object instead of directly on config_manager
        self.config_manager.config._findex = file_index
        field_to_plot = self._get_field_to_plot_compare(source_data, field_name, file_index,
                                                        pt, figure, level=level)
        plotter.comparison_plots(self.config_manager, field_to_plot, level=level)

    def _get_field_to_plot_compare(self, source_data,
                                   field_name, file_index, plot_type, figure, ax=None, level=None) -> tuple:
        if ax is None:
            # ax = figure.get_axes()
            ax = figure.get_fig_ax()
        dim1, dim2 = self.config_manager.get_dim_names(plot_type)
        data2d = None
        if self.config_manager.ax_opts['is_diff_field']:
            proc = Interp(self.config_manager, self.data2d_list)
            data2d, xx, yy = proc.regrid(plot_type)
            return data2d, xx, yy, self.field_names[0], plot_type, file_index, figure, ax
        else:
            if 'yz' in plot_type:
                data2d = self._get_yz(source_data, field_name, time_lev=self.config_manager.ax_opts['time_lev'])
            elif 'xt' in plot_type:
                data2d = self._get_xt(source_data, field_name, time_lev=self.config_manager.ax_opts['time_lev'])
            elif 'tx' in plot_type:
                data2d = self._get_tx(source_data, field_name, level=None, time_lev=self.config_manager.ax_opts['time_lev'])
            elif 'xy' in plot_type or 'polar' in plot_type:
                data2d = self._get_xy(source_data, field_name, level=level, time_lev=self.config_manager.ax_opts['time_lev'])
            else:
                pass

        self.data2d_list.append(data2d)
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        return data2d, data2d[dim1].values, data2d[dim2].values, field_name, plot_type, file_index, figure, ax

    # DATA SLICE PROCESSING METHODS
    #--------------------------------------------------------------------------
    def _get_yz(self, source_data, field_name, time_lev):
        """ Extract YZ slice (zonal mean) from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 2D (lat, lev) slice
        """
        d_temp = source_data['vars'][field_name]
        if d_temp is None:
            return

        # Compute zonal mean
        zonal_mean = d_temp.mean(dim=self.config_manager.get_model_dim_name('xc'))
        zonal_mean.attrs = d_temp.attrs.copy()

        # Do we have multiple time levels and if so do we want to average?
        if self.config_manager.get_model_dim_name('tc') in zonal_mean.dims:
            num_times = np.size(zonal_mean.time)
            if self.config_manager.ax_opts['tave'] and num_times > 1:
                self.logger.debug(f"Averaging over {num_times} time levels.")
                zonal_mean = apply_mean(self.config_manager.config, zonal_mean)
            else:
                zonal_mean = zonal_mean.isel(time=time_lev)
        else:
            # single time level
            zonal_mean = zonal_mean.squeeze()

        zonal_mean = self._select_yrange(zonal_mean, field_name)
        return apply_conversion(self.config_manager.config, zonal_mean, field_name)

    def _get_xy(self, source_data, field_name, level, time_lev):
        """ Extract XY slice (latlon) from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 2D (lon, lat) slice
        """
        d_temp = source_data['vars'][field_name]
        if d_temp is None:
            return

        if level:
            level = int(level)
        d_temp = d_temp.isel(time=time_lev)
        data2d = d_temp.squeeze()

        # Do we have multiple time levels and if so do we want to average?
        if self.config_manager.get_model_dim_name('tc') in d_temp.dims:
            num_tc = np.size(d_temp.time)
            if self.config_manager.ax_opts['tave'] and num_tc > 1:
                self.logger.debug(f"Averaging over {num_tc} time levels.")
                data2d = apply_mean(self.config_manager.config, data2d, level)
                return apply_conversion(self.config_manager.config, data2d, field_name)
            else:  # just select the specified time level
                if num_tc > 1:
                    data2d = d_temp.isel(time=time_lev)

        # Do we have multiple vertical levels and if so do we want to average?
        if self.config_manager.ax_opts['zave']:
            self.logger.debug(f"Averaging over vertical levels.")
            data2d = apply_mean(self.config_manager.config, data2d, level='all')
            return apply_conversion(self.config_manager.config, data2d, field_name)

        # Add total column
        if self.config_manager.ax_opts['zsum']:
            self.logger.debug(f"Summing over vertical levels.")
            data2d_zsum = apply_zsum(self.config_manager.config, data2d)
            self.logger.debug(f"Min: {data2d_zsum.min()}, Max: {data2d_zsum.max()}")
            return apply_conversion(self.config_manager.config, data2d_zsum, field_name)

        num_zc = 0
        if self.config_manager.get_model_dim_name('zc') in d_temp.dims:
            num_zc = np.size(data2d.lev)
        if level and num_zc > 1:
            lev_to_plot = int(np.where(data2d.coords[self.config_manager.get_model_dim_name('zc')].values == level)[0])
            data2d = data2d.isel(lev=lev_to_plot)
        return apply_conversion(self.config_manager.config, data2d, field_name)

    def _get_xt(self, d, name, time_lev):
        """ Extract time-series from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 1D (time) series
        """
        d_temp = d['vars'][name]
        if d_temp is None:
            return
            
        # Get time dimension safely
        time_dim = 'time'
        tc_dim = self.config_manager.get_model_dim_name('tc')
        if tc_dim:
            time_dim = tc_dim
        
        # Try to get the number of time steps safely
        try:
            if time_dim in d_temp.dims:
                num_times = d_temp[time_dim].size
            else:
                # Fall back to 'time' if tc_dim not found in dimensions
                num_times = d_temp.time.size
        except (AttributeError, KeyError):
            # If all else fails, try to infer
            if hasattr(d_temp, 'shape') and len(d_temp.shape) > 0:
                num_times = d_temp.shape[0]  # Assume time is the first dimension
            else:
                self.logger.error(f"Cannot determine time dimension for {name}")
                return None
        
        self.logger.info(f"'{name}' field has {num_times} time levels")

        if isinstance(time_lev, list):
            self.logger.info(f"Computing time series on {time_lev} time range")
            try:
                if time_dim in d_temp.dims:
                    data2d = d_temp.isel({time_dim: slice(time_lev)})
                else:
                    data2d = d_temp.isel(time=slice(time_lev))
            except (AttributeError, KeyError):
                self.logger.error(f"Error slicing time dimension for {name}")
                return None
        else:
            data2d = d_temp.squeeze()

        if 'mean_type' in self.config_manager.spec_data[name]['xtplot']:
            mean_type = self.config_manager.spec_data[name]['xtplot']['mean_type']
            self.logger.info(f"Averaging method: {mean_type}")
            # annual:
            if mean_type == 'point_sel':
                xc = self.config_manager.spec_data[name]['xtplot']['point_sel'][0]
                yc = self.config_manager.spec_data[name]['xtplot']['point_sel'][1]
                data2d = data2d.sel(lon=xc, lat=yc, method='nearest')
            elif mean_type == 'area_sel':
                x1 = self.config_manager.spec_data[name]['xtplot']['area_sel'][0]
                x2 = self.config_manager.spec_data[name]['xtplot']['area_sel'][1]
                y1 = self.config_manager.spec_data[name]['xtplot']['area_sel'][2]
                y2 = self.config_manager.spec_data[name]['xtplot']['area_sel'][3]
                data2d = data2d.sel(lon=np.arange(x1, x2, 0.5), lat=np.arange(y1, y2, 0.5), method='nearest')
                
                # Get dimension names safely
                xc_dim = self.config_manager.get_model_dim_name('xc') or 'lon'
                yc_dim = self.config_manager.get_model_dim_name('yc') or 'lat'
                
                try:
                    data2d = data2d.mean(dim=(xc_dim, yc_dim))
                except (ValueError, KeyError):
                    # Fall back to literal 'lon' and 'lat' if the above fails
                    try:
                        data2d = data2d.mean(dim=('lon', 'lat'))
                    except (ValueError, KeyError):
                        self.logger.error(f"Cannot compute area mean for {name}")
                        return None
                        
            elif mean_type in ['year', 'season', 'month']:
                # Safe handling of time grouping
                try:
                    time_attr = f"{time_dim}.{mean_type}"
                    data2d = data2d.groupby(time_attr).mean(dim=time_dim, keep_attrs=True)
                except (AttributeError, KeyError):
                    try:
                        time_attr = f"time.{mean_type}"
                        data2d = data2d.groupby(time_attr).mean(dim='time', keep_attrs=True)
                    except (AttributeError, KeyError):
                        self.logger.error(f"Cannot group by {mean_type} for {name}")
                        return None
            else:
                # Safe handling of general mean
                try:
                    data2d = data2d.groupby(time_dim).mean(dim=xr.ALL_DIMS, keep_attrs=True)
                except (AttributeError, KeyError):
                    try:
                        data2d = data2d.groupby('time').mean(dim=xr.ALL_DIMS, keep_attrs=True)
                    except (AttributeError, KeyError):
                        self.logger.error(f"Cannot compute general mean for {name}")
                        return None
                        
                if 'mean_type' in self.config_manager.spec_data[name]['xtplot']:
                    if self.config_manager.spec_data[name]['xtplot']['mean_type'] == 'rolling':
                        window_size = 5
                        if 'window_size' in self.config_manager.spec_data[name]['xtplot']:
                            window_size = self.config_manager.spec_data[name]['xtplot']['window_size']
                        self.logger.info(f" -- smoothing window size: {window_size}")
                        kernel = np.ones(window_size) / window_size
                        convolved_data = np.convolve(data2d, kernel, mode="same")
                        
                        # Create a new DataArray with the convolved data
                        try:
                            if time_dim in data2d.coords:
                                data2d = xr.DataArray(convolved_data, dims=time_dim, coords=data2d.coords)
                            else:
                                data2d = xr.DataArray(convolved_data, dims='time', coords=data2d.coords)
                        except (AttributeError, KeyError):
                            self.logger.error(f"Error creating DataArray for convolved data for {name}")
                            return None

        else:
            # Safe handling of default time mean
            try:
                data2d = data2d.groupby(time_dim).mean(dim=xr.ALL_DIMS, keep_attrs=True)
            except (AttributeError, KeyError):
                try:
                    data2d = data2d.groupby('time').mean(dim=xr.ALL_DIMS, keep_attrs=True)
                except (AttributeError, KeyError):
                    self.logger.error(f"Cannot compute default time mean for {name}")
                    return None

        if 'level' in self.config_manager.spec_data[name]['xtplot']:
            level = int(self.config_manager.spec_data[name]['xtplot']['level'])
            
            # Get vertical dimension safely
            zc_dim = self.config_manager.get_model_dim_name('zc')
            if zc_dim:
                try:
                    if zc_dim in data2d.coords:
                        lev_values = data2d.coords[zc_dim].values
                        lev_to_plot = int(np.where(lev_values == level)[0])
                        data2d = data2d.isel({zc_dim: lev_to_plot}).squeeze()
                    else:
                        # Fall back to 'lev' if zc_dim not found in coordinates
                        lev_values = data2d.coords['lev'].values
                        lev_to_plot = int(np.where(lev_values == level)[0])
                        data2d = data2d.isel(lev=lev_to_plot).squeeze()
                except (AttributeError, KeyError, IndexError):
                    self.logger.error(f"Cannot select level {level} for {name}")
                    pass  # Continue with full data if level selection fails

        return data2d  # Already converted to appropriate units through DataReader

    def _get_tx(self, source_data, field_name, level=None, time_lev=0):
        """ Extract a time-series map from a DataArray

        Note:
            Assume input DataArray is at most 4-dimensional (time, lev, lon, lat)
            and return a 2D Hovmoller plot field where time is plotted on one axis (default y-axis)
            and the spatial dimension (either lon or lat)) is plotted on the other axis  (default x-axis)

        """
        d_temp = source_data['vars'][field_name]
        if d_temp is None:
            return

        data2d = d_temp.squeeze()
        
        # Get the vertical coordinate dimension name
        zc = self.config_manager.get_model_dim_name('zc')
        
        # Check if the vertical dimension exists in the data
        if zc and zc in d_temp.dims:
            if 'level' in self.config_manager.spec_data[field_name]['txplot']:
                lev_to_plot = self.config_manager.spec_data[field_name]['txplot']['level']
                lev_index = int(np.where(data2d.coords['lev'].values == lev_to_plot)[0])
                data2d = data2d.isel(lev=lev_index)
            else:
                data2d = d_temp.isel(lev=0)

        if 'trange' in self.config_manager.spec_data[field_name]['txplot']:
            start_time = self.config_manager.spec_data[field_name]['txplot']['trange'][0]
            end_time = self.config_manager.spec_data[field_name]['txplot']['trange'][1]
            data2d = data2d.sel(time=slice(start_time, end_time))
        if 'yrange' in self.config_manager.spec_data[field_name]['txplot']:
            lats0 = self.config_manager.spec_data[field_name]['txplot']['yrange'][0]
            lats1 = self.config_manager.spec_data[field_name]['txplot']['yrange'][1]
            data2d = data2d.sel(lat=slice(lats0, lats1))
        if 'xrange' in self.config_manager.spec_data[field_name]['txplot']:
            lons0 = self.config_manager.spec_data[field_name]['txplot']['xrange'][0]
            lons1 = self.config_manager.spec_data[field_name]['txplot']['xrange'][1]
            data2d = data2d.sel(lon=slice(lons0, lons1))
        weights = np.cos(np.deg2rad(data2d.lat.values))

        d1 = data2d * weights[None, :, None]
        d2 = d1.sum(dim='lat')
        d3 = d2 / np.sum(weights)
        return apply_conversion(self.config_manager, d3, field_name)

    def _select_yrange(self, data2d, name):
        """ Select a range of vertical levels"""
        if 'zrange' in self.config_manager.spec_data[name]['yzplot']:
            if not self.config_manager.spec_data[name]['yzplot']['zrange']:
                return data2d
            lo_z = self.config_manager.spec_data[name]['yzplot']['zrange'][0]
            hi_z = self.config_manager.spec_data[name]['yzplot']['zrange'][1]
            if hi_z >= lo_z:
                self.logger.error(f"Upper level value ({hi_z}) must be less than low level value ({lo_z})")
                return
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
        real_time = data_var.time.isel(time=time_index).values
        real_time_readable = pd.to_datetime(real_time).strftime('%Y-%m-%d %H')
        self.config_manager.real_time = real_time_readable

    def _side_by_side_plots(self, plotter):
        """
        Generate side-by-side comparison plots (2x1 subplots) without difference.
        """
        current_field_index = 0
        self.data2d_list = []  # Initialize list to store data for comparison

        # Process each pair of indices from the comparison configuration
        for idx1, idx2 in zip(self.config_manager.a_list, self.config_manager.b_list):
            # Get map parameters for these indices
            map1 = self.config_manager.config.map_params[idx1]
            map2 = self.config_manager.config.map_params[idx2]

            # Load data from both sources
            source_data_pair = self._load_comparison_data(map1, map2)
            if not source_data_pair:
                continue

            sdat1, sdat2 = source_data_pair
            
            # Determine file indices
            source_name1, source_name2 = map1['source_name'], map2['source_name']
            filename1, filename2 = map1['filename'], map2['filename']
            file_indices = self._get_file_indices_compare(source_name1, source_name2, filename1, filename2)

            # Process each plot type
            field_name1, field_name2 = map1['field'], map2['field']
            self.field_names = (field_name1, field_name2)

            for pt1, pt2 in zip(map1['to_plot'], map2['to_plot']):
                plot_type = pt1  # Using the first plot type
                self.logger.info(f"Plotting {field_name1} vs {field_name2} side by side, {plot_type} plot")
                self.data2d_list = []  # Reset for each plot type

                if 'xy' in plot_type or 'polar' in plot_type:
                    self._process_xy_side_by_side_plots(plotter, file_indices,
                                                    current_field_index,
                                                    field_name1, field_name2, plot_type,
                                                    sdat1, sdat2)
                else:
                    self._process_other_side_by_side_plots(plotter, file_indices,
                                                        current_field_index,
                                                        field_name1, field_name2,
                                                        plot_type, sdat1, sdat2)

            current_field_index += 1

    def _process_xy_side_by_side_plots(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, plot_type, sdat1, sdat2):
        """Process side-by-side comparison plots for xy or polar plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels
       
        # Get levels for the plots
        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            self.logger.warning(f' -> No levels specified for {field_name1}')
            return
        
        for level in levels:
            # Create a figure with 2x1 subplots (side by side)
            figure = Figure(self.config_manager, plot_type, nrows=nrows, ncols=ncols)
            ax = figure.get_axes()
            self.config_manager.level = level
            
            # Create the 2x1 side-by-side comparison plot
            self._create_2x1_side_by_side_plot(plotter, file_indices,
                                            current_field_index,
                                            field_name1, field_name2, figure, ax,
                                            plot_type, sdat1, sdat2, level)
            
            # Save the plot
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure, level=level)

    def _process_other_side_by_side_plots(self, plotter, file_indices, current_field_index,
                                        field_name1, field_name2, plot_type, sdat1, sdat2):
        """Process side-by-side comparison plots for other plot types."""
        file_index1, file_index2 = file_indices
        nrows, ncols = self.config_manager.input_config._comp_panels
        
        # Create a figure with 2x1 subplots (side by side)
        figure = Figure.create_eviz_figure(self.config_manager, plot_type, nrows=nrows, ncols=ncols)  
        axes_shape = figure.subplots
        ax = figure.get_axes()
        self.config_manager.level = None
        
        # Create the nx1 side-by-side comparison plot
        self._create_nx1_side_by_side_plot(plotter, file_indices, current_field_index,
                                        field_name1, field_name2, figure, ax,
                                        plot_type, sdat1, sdat2)
        
        # Save the plot
        pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def _create_nx1_side_by_side_plot(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, figure, ax,
                                    plot_type, sdat1, sdat2, level=None):
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
        # TODO: add loop for n in range(3)...
        self._process_side_by_side_plot(plotter, file_index1, current_field_index,
                                    field_name1,
                                    figure, ax, 0, sdat1, plot_type, level=level)
        
        # Plot the second dataset in the right subplot
        self._process_side_by_side_plot(plotter, file_index2, current_field_index,
                                    field_name2,
                                    figure, ax, 1, sdat2, plot_type, level=level)

    def _process_side_by_side_plot(self, plotter, file_index, current_field_index, field_name, 
                                figure, ax, ax_index, source_data, plot_type, level=None):
        self.config_manager.findex = file_index
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        
        # Set up the axis
        if isinstance(ax, list):
            current_ax = ax[ax_index]
        else:
            current_ax = ax
        
        # Get field to plot
        field_to_plot = self._get_field_to_plot_compare(source_data, field_name, file_index,
                                                        plot_type, figure, ax=current_ax, level=level)
        
        # Check which type of plotter we're using and call the appropriate method
        if hasattr(plotter, 'single_plots'):
            # SinglePlotter
            plotter.single_plots(self.config_manager, field_to_plot, level=level)
        elif hasattr(plotter, 'comparison_plots'):
            # ComparisonPlotter
            plotter.comparison_plots(self.config_manager, field_to_plot, level=level)
        else:
            # Fallback - try to call plot directly
            self.logger.warning(f"Unknown plotter type: {type(plotter).__name__}. Trying to call plot method.")
            if hasattr(plotter, 'plot'):
                plotter.plot(self.config_manager, field_to_plot, level=level)
            else:
                self.logger.error(f"Plotter {type(plotter).__name__} has no plot method.")


    def _get_field_to_plot_side_by_side(self, source_data, field_name, file_index, 
                                    plot_type, figure, ax=None, level=None):
        """Get field data for side-by-side plots with WRF-specific handling."""
        if ax is None:
            ax = figure.get_axes()
        
        # Get source name safely
        source_name = self._get_source_name_for_file_index(file_index)
        
        # Set source_data for dimension handling
        self.source_data = source_data
        
        # Get dimension names
        dim1, dim2 = self.coord_names(source_name, source_data, field_name, plot_type)
        
        # Get time level
        time_level = self.config_manager.ax_opts['time_lev']
        if isinstance(time_level, str) and time_level == 'all':
            time_level = 0  # Default to first time level for comparison
        
        # Extract data based on plot type
        d = source_data['vars'][field_name]
        
        # Process data based on plot type
        if 'yz' in plot_type:
            data2d = self._get_yz(d, field_name, time_lev=time_level)
        elif 'xy' in plot_type or 'polar' in plot_type:
            data2d = self._get_xy(d, field_name, level=level, time_lev=time_level)
        else:
            # Other plot types (xt, tx)
            self.logger.warning(f"Plot type {plot_type} not fully supported for side-by-side comparison")
            return None
        
        # Process coordinates
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        elif 'yz' in plot_type:
            # For YZ plots, use latitude and pressure levels
            xs = np.array(self._get_field(dim1[0], d)[0, :][:, 0])
            ys = self.levs  # Pressure levels
            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax
        else:
            # For XY plots, use longitude and latitude
            xs = np.array(self._get_field(dim1[0], data2d)[0, :])
            ys = np.array(self._get_field(dim2[0], data2d)[:, 0])
            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax

    def get_field_dim_name(self, source_data: dict, dim_name: str, field_name: str):
        d = source_data['vars'][field_name]
        field_dims = list(d.dims)
        names = self.get_model_dim_name(self.source_name, dim_name).split(',')
        common = list(set(names).intersection(field_dims))
        dim = list(common)[0] if common else None
        return dim

    def _get_source_name_for_file_index(self, file_index):
        """
        Get the source name for a given file index.
        This handles the case where multiple files come from the same source.
        """
        # Try to get source name from file_list
        if hasattr(self.config_manager, 'file_list') and file_index in self.config_manager.file_list:
            return self.config_manager.file_list[file_index].get('source_name', 'generic')
        
        # Try to get source name from map_params
        if hasattr(self.config_manager, 'map_params'):
            for param_key, param_config in self.config_manager.map_params.items():
                if param_key == file_index or param_config.get('file_index') == file_index:
                    return param_config.get('source_name', 'generic')
        
        return 'generic'
