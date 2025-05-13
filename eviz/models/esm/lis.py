import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eviz.lib.data.data_utils import apply_mean
from eviz.lib.data.data_utils import apply_conversion
from eviz.lib.autoviz.figure import Figure
from eviz.models.esm.nuwrf import NuWrf
from eviz.lib.autoviz.plot_utils import print_map
from eviz.lib.data.processor import Interp

warnings.filterwarnings("ignore")


@dataclass
class Lis(NuWrf):
    """ Define LIS specific model data and functions.
    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.comparison_plot = False
        self.source_name = 'lis'
        
    @property
    def global_attrs(self):
        return self._global_attrs

    def _simple_plots(self, plotter):
        map_params = self.config_manager.map_params
        field_num = 0
        self.config_manager.findex = 0
        for i in map_params.keys():
            field_name = map_params[i]['field']
            source_name = map_params[i]['source_name']
            self.source_name = source_name
            filename = map_params[i]['filename']
            file_index = self.config_manager.get_file_index(filename)
            
            # Get the appropriate reader
            reader = self._get_reader(source_name)
            if not reader:
                self.logger.error(f"No reader found for source {source_name}")
                continue
                
            self.source_data = reader.read_data(filename)
            if not self.source_data:
                self.logger.error(f"Failed to read data from {filename}")
                continue
                
            self._global_attrs = self.set_global_attrs(source_name, self.source_data['attrs'])
            self.config_manager.findex = file_index
            self.config_manager.pindex = field_num
            self.config_manager.axindex = 0
            for pt in map_params[i]['to_plot']:
                self.logger.info(f"Plotting {field_name}, {pt} plot")
                field_to_plot = self._get_field_for_simple_plot(field_name, pt)
                plotter.simple_plot(self.config_manager, field_to_plot)
            field_num += 1


    def _single_plots(self, plotter):
        for s in range(len(self.config_manager.source_names)):
            map_params = self.config_manager.map_params
            field_num = 0
            for i in map_params.keys():
                source_name = map_params[i]['source_name']
                if source_name == self.config_manager.source_names[s]:
                    field_name = map_params[i]['field']
                    self.source_name = source_name
                    filename = map_params[i]['filename']
                    file_index = field_num  # self.config_manager.get_file_index(filename)

                    # Get the appropriate reader
                    reader = self._get_reader(source_name)
                    if not reader:
                        self.logger.error(f"No reader found for source {source_name}")
                        continue
                        
                    self.source_data = reader.read_data(filename)
                    if not self.source_data:
                        self.logger.error(f"Failed to read data from {filename}")
                        continue

                    self._global_attrs = self.source_data['attrs']
                    self.config_manager.findex = file_index
                    self.config_manager.pindex = field_num
                    self.config_manager.axindex = 0
                    for pt in map_params[i]['to_plot']:
                        self.logger.info(f"Plotting {field_name}, {pt} plot")
                        figure = Figure(self.config_manager, pt)
                        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
                        time_level = self.config_manager.ax_opts['time_lev']
                        d = self.source_data['vars'][field_name]
                        
                        # Get the time dimension name - try several common variants
                        time_dim = None
                        for dim_name in ['time', 'Time', 't', 'T']:
                            if dim_name in d.dims:
                                time_dim = dim_name
                                break
                        
                        # If no time dimension found, create a dummy
                        if time_dim is None:
                            self.logger.warning(f"No time dimension found in {field_name}")
                            num_times = 1
                        else:
                            num_times = d.dims[time_dim] if time_level == 'all' else 1
                        
                        time_levels = range(num_times)

                        # vertical levels (soil moisture or soil temperature)
                        levels = self.config_manager.get_levels(field_name, pt + 'plot')

                        if not levels:
                            self.logger.warning(f' -> No levels specified for {field_name}')
                            for t in time_levels:
                                self.config_manager.time_level = t
                                
                                # Get time value safely
                                real_time = self._get_time_value(d, t, time_dim)
                                
                                # Convert time to a readable format
                                try:
                                    real_time_readable = pd.to_datetime(real_time).strftime('%Y-%m-%d %H')
                                except:
                                    real_time_readable = f"Time step {t}"
                                    
                                self.config_manager.real_time = real_time_readable
                                
                                field_to_plot = self._get_field_to_plot(field_name, file_index, pt, figure)
                                plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
                                print_map(self.config_manager, pt, self.config_manager.findex, figure)
                        else:
                            for level in levels:
                                self.logger.info(f' -> Processing {num_times} time levels')
                                for t in time_levels:
                                    self.config_manager.time_level = t
                                    
                                    # Get time value safely
                                    real_time = self._get_time_value(d, t, time_dim)
                                    
                                    # Convert time to a readable format
                                    try:
                                        real_time_readable = pd.to_datetime(real_time).strftime('%Y-%m-%d %H')
                                    except:
                                        real_time_readable = f"Time step {t}"
                                        
                                    self.config_manager.real_time = real_time_readable
                                    
                                    field_to_plot = self._get_field_to_plot(field_name, file_index, pt, figure, level=level)
                                    plotter.single_plots(self.config_manager, field_to_plot=field_to_plot, level=level)
                                    print_map(self.config_manager, pt, self.config_manager.findex, figure, level=level)

                    field_num += 1

    def _get_time_value(self, data_array, time_index, time_dim=None):
        """
        Get the time value for the given index.
        
        Args:
            data_array: The data array containing time values
            time_index: The index to extract
            time_dim: The name of the time dimension
            
        Returns:
            The time value or a default timestamp if unavailable
        """
        # If no time dimension, return a default timestamp
        if time_dim is None:
            return pd.Timestamp('2000-01-01')
        
        # Try different approaches to get the time value
        try:
            # First try: Check if there's a time coordinate with the time_dim
            if time_dim in data_array.coords:
                return data_array.coords[time_dim].values[time_index]
            
            # Second try: Check for a time variable attribute
            if hasattr(data_array, 'time') and hasattr(data_array.time, 'isel'):
                return data_array.time.isel({time_dim: time_index}).values
            
            # Third try: Check for time-related variables in source_data
            for var_name in ['time', 'Time', 'times', 'Times']:
                if var_name in self.source_data['vars']:
                    time_var = self.source_data['vars'][var_name]
                    if time_dim in time_var.dims:
                        return time_var.isel({time_dim: time_index}).values
            
            # Last resort: create a dummy timestamp
            return pd.Timestamp('2000-01-01') + pd.Timedelta(days=time_index)
        
        except Exception as e:
            self.logger.warning(f"Error getting time value: {e}")
            return pd.Timestamp('2000-01-01') + pd.Timedelta(days=time_index)
        
    def _get_field_to_plot(self, field_name, file_index, plot_type, figure, level=None):
        ax = figure.get_axes()
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        data2d = None
        d = self.source_data['vars'][field_name]
        
        # Get the time level from config
        time_lev = self.config_manager.time_level if hasattr(self.config_manager, 'time_level') else 0
        
        if 'xt' in plot_type:
            data2d = self._get_xt(d, field_name, time_lev=self.ax_opts['time_lev'])
        elif 'tx' in plot_type:
            data2d = self._get_tx(d, field_name, level=None, time_lev=self.ax_opts['time_lev'])
        elif 'xy' in plot_type:
            # Pass all required arguments to _get_xy
            data2d = self._get_xy(d, field_name, level, time_lev)
        else:
            pass

        xs, ys, extent, central_lon, central_lat = None, None, [], 0.0, 0.0
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        else:
            lon = self._get_field('east_west', data2d)
            lat = self._get_field('north_south', data2d)
            xs = np.array(lon)
            ys = np.array(lat)
            # Some LIS coordinates are NaN. The following workaround fills out those elements
            # with reasonable values:
            idx = np.argwhere(np.isnan(xs))
            for i in idx:
                xs[i] = xs[i - 1] + self._global_attrs["DX"] / 1000.0 / 100.0
            idx = np.argwhere(np.isnan(ys))
            for i in idx:
                ys[i] = ys[i - 1] + self._global_attrs["DY"] / 1000.0 / 100.0

            latN = max(ys[:])
            latS = min(ys[:])
            lonW = min(xs[:])
            lonE = max(xs[:])
            self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
            self.config_manager.ax_opts['central_lon'] = np.mean(self.config_manager.ax_opts['extent'][:2])
            self.config_manager.ax_opts['central_lat'] = np.mean(self.config_manager.ax_opts['extent'][2:])

            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax

    def _get_field_for_simple_plot(self, field_name, plot_type):
        data2d = None
        d = self.source_data['vars'][field_name]
        
        # Default values for level and time_lev
        level = 0
        time_lev = 0
        
        if 'xt' in plot_type:
            data2d = self._get_xt(d, field_name, time_lev=self.ax_opts['time_lev'])
        elif 'tx' in plot_type:
            data2d = self._get_tx(d, field_name, level=None, time_lev=self.ax_opts['time_lev'])
        elif 'xy' in plot_type:
            # Pass all required arguments to _get_xy
            data2d = self._get_xy(d, field_name, level, time_lev)
        else:
            pass

        xs, ys = None, None
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, xs, ys, field_name, plot_type
        else:
            lon = self._get_field('east_west', data2d)
            lat = self._get_field('north_south', data2d)
            xs = np.array(lon)
            ys = np.array(lat)
            # Some LIS coordinates are NaN. The following workaround fills out those elements
            # with reasonable values:
            idx = np.argwhere(np.isnan(xs))
            for i in idx:
                xs[i] = xs[i - 1] + self._global_attrs["DX"] / 1000.0 / 100.0
            idx = np.argwhere(np.isnan(ys))
            for i in idx:
                ys[i] = ys[i - 1] + self._global_attrs["DY"] / 1000.0 / 100.0

            return data2d, xs, ys, field_name, plot_type


    def _calculate_diff(self, name1, name2, ax_opts):
        """ Helper method for get_diff_data """
        d1 = self._get_data(name1, ax_opts, 0)
        d2 = self._get_data(name2, ax_opts, 1)
        d1 = apply_conversion(self.config_manager, d1, name1).squeeze()
        d2 = apply_conversion(self.config_manager, d2, name2).squeeze()
        return d1 - d2

    def _get_data(self, field_name, ax_opts, pid):
        d = self.config_manager.readers[0].get_field(field_name, self.config_manager.findex)
        # Only XY plots and top layer
        return self._get_xy(d, field_name, level=0, time_lev=ax_opts['time_lev'])

    @staticmethod
    def __get_xy(d, name):
        """ Extract XY slice from N-dim data field"""
        if d is None:
            return
        dlength = len(d.shape)
        if dlength == 2:
            return d[:, :]
        if dlength == 3:
            return d[0, :, :]
        if dlength == 4:
            return d[0, 0, :, :]

    # TODO: put in nuwrf_utils.py
    def _get_field(self, name, data):
        try:
            return data[name]
        except Exception as e:
            self.logger.error('key error: %s, not found' % str(e))
            return None
    
    def _process_coordinates(self, data2d, dim1, dim2, field_name, plot_type, file_index, figure, ax):
        """
        Process coordinates for LIS plots, handling NaN values in coordinates.
        """
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        else:
            # Get longitude and latitude coordinates
            lon = self._get_field('east_west', data2d)
            lat = self._get_field('north_south', data2d)
            xs = np.array(lon)
            ys = np.array(lat)
            
            # Handle NaN values in LIS coordinates
            idx = np.argwhere(np.isnan(xs))
            for i in idx:
                xs[i] = xs[i - 1] + self._global_attrs["DX"] / 1000.0 / 100.0
            idx = np.argwhere(np.isnan(ys))
            for i in idx:
                ys[i] = ys[i - 1] + self._global_attrs["DY"] / 1000.0 / 100.0

            # Set extent for the plot
            latN = max(ys[:])
            latS = min(ys[:])
            lonW = min(xs[:])
            lonE = max(xs[:])
            self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
            self.config_manager.ax_opts['central_lon'] = np.mean(self.config_manager.ax_opts['extent'][:2])
            self.config_manager.ax_opts['central_lat'] = np.mean(self.config_manager.ax_opts['extent'][2:])

            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax

    def _apply_vertical_level_selection(self, data2d, field_name, level):
        """
        Apply vertical level selection for LIS data.
        """
        # Get the vertical dimension name
        dim = self.get_dd(self.source_name, self.source_data, 'zc', field_name)
        
        # If vertical dimension exists and level is specified, select it
        if dim and level is not None:
            try:
                data2d = eval(f"data2d.isel({dim}=level)")
            except Exception as e:
                self.logger.warning(f"Error selecting vertical level: {e}")
        
        return data2d

    def _apply_time_selection(self, original_data, data2d, time_dim, time_lev, field_name, level):
        """
        LIS has more complex time handling with optional time averaging.
        """
        if time_dim and time_dim in original_data.dims:
            num_times = original_data.dims[time_dim]
            
            # Check if time averaging is enabled
            if hasattr(self, 'ax_opts') and self.ax_opts.get('tave', False) and num_times > 1:
                self.logger.debug(f"Averaging over {num_times} time levels.")
                return apply_mean(self.config_manager, data2d, level)
            else:
                try:
                    return original_data.isel({time_dim: time_lev}).squeeze()
                except Exception as e:
                    self.logger.warning(f"Error selecting time level: {e}")
                    # If time selection fails, use the data as is
        
        return data2d

    def _init_model_specific_data_for_comparison(self, sdat1, sdat2):
        """
        Initialize any LIS-specific data needed for comparison.
        For LIS, we don't need to initialize domain information like in WRF,
        but we do need to set global attributes.
        """
        # Save current source_data and global_attrs
        original_source_data = self.source_data
        original_global_attrs = self._global_attrs if hasattr(self, '_global_attrs') else None
        
        # Set global attributes for first dataset
        self.source_data = sdat1
        self._global_attrs = self.set_global_attrs(self.source_name, sdat1['attrs'])
        
        # Set global attributes for second dataset
        self.source_data = sdat2
        second_global_attrs = self.set_global_attrs(self.source_name, sdat2['attrs'])
        
        # Restore original source_data and global_attrs
        self.source_data = original_source_data
        if original_global_attrs:
            self._global_attrs = original_global_attrs

    def _comparison_plots(self, plotter):
        """Generate comparison plots for paired LIS data sources according to configuration."""
        current_field_index = 0
        self.data2d_list = []  # Initialize list to store data for comparison

        # Process each pair of indices from the comparison configuration
        for idx1, idx2 in zip(self.config_manager.a_list, self.config_manager.b_list):
            # Get map parameters for these indices
            map1 = self.config_manager.map_params[idx1]
            map2 = self.config_manager.map_params[idx2]

            # Load data from both sources
            source_data_pair = self._load_comparison_data(map1, map2)
            if not source_data_pair:
                continue

            sdat1, sdat2 = source_data_pair
            
            # Initialize model-specific data for comparison
            self._init_model_specific_data_for_comparison(sdat1, sdat2)
            
            # Create a tuple of both datasets for difference calculations
            sdat = (sdat1, sdat2)

            # Determine file indices
            source_name1, source_name2 = map1['source_name'], map2['source_name']
            filename1, filename2 = map1['filename'], map2['filename']
            file_indices = self._get_file_indices(source_name1, source_name2, filename1, filename2)

            # Process each plot type
            field_name1, field_name2 = map1['field'], map2['field']
            self.field_names = (field_name1, field_name2)

            for pt1, pt2 in zip(map1['to_plot'], map2['to_plot']):
                plot_type = pt1  # Using the first plot type
                self.logger.info(f"Plotting {field_name1} vs {field_name2}, {plot_type} plot")
                self.data2d_list = []  # Reset for each plot type

                if 'xy' in plot_type or 'polar' in plot_type:
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
        """Load data from both LIS sources for comparison."""
        source_name1, source_name2 = map1['source_name'], map2['source_name']
        filename1, filename2 = map1['filename'], map2['filename']

        # Get readers for both sources
        reader1 = self._get_reader(source_name1)
        reader2 = self._get_reader(source_name2)
        
        if not reader1 or not reader2:
            self.logger.error("No suitable readers found for comparison")
            return None
        
        # Read data from both files
        sdat1 = reader1.read_data(filename1)
        sdat2 = reader2.read_data(filename2)

        if not sdat1 or not sdat2:
            self.logger.error("Failed to read data for comparison")
            return None

        return sdat1, sdat2

    def _get_file_indices(self, source_name1, source_name2, filename1, filename2):
        """Determine file indices for comparison."""
        if source_name1 == source_name2:
            return 0, 1  # Same data source
        else:
            file_index1 = self.config_manager.get_file_index(filename1)
            file_index2 = self.config_manager.get_file_index(filename2)
            return file_index1, file_index2

    def _process_xy_comparison_plots(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, plot_type, sdat1, sdat2, sdat):
        """Process comparison plots for xy or polar plot types."""
        file_index1, file_index2 = file_indices

        # Get levels for the plots
        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            self.logger.warning(f' -> No levels specified for {field_name1}')
            return

        for level in levels:
            figure = Figure(self.config_manager, plot_type)
            ax = figure.get_axes()
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

            print_map(self.config_manager, plot_type, self.config_manager.findex, figure, level=level)
            self.comparison_plot = False

    def _process_other_comparison_plots(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, plot_type, sdat1, sdat2, sdat):
        """Process comparison plots for other plot types."""
        file_index1, file_index2 = file_indices

        figure = Figure(self.config_manager, plot_type)
        ax = figure.get_axes()
        axes_shape = figure.get_gs_geometry()
        self.config_manager.level = None

        if axes_shape == (3, 1):
            self._create_3x1_comparison_plot(plotter, file_indices, current_field_index,
                                            field_name1, field_name2, figure, ax,
                                            plot_type, sdat1, sdat2, sdat)
        elif axes_shape == (2, 2):
            self._create_2x2_comparison_plot(plotter, file_indices, current_field_index,
                                            field_name1, field_name2, figure,
                                            plot_type, sdat1, sdat2, sdat)

        print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def _create_3x1_comparison_plot(self, plotter, file_indices, current_field_index,
                                field_name1, field_name2, figure, ax,
                                plot_type, sdat1, sdat2, sdat, level=None):
        """Create a 3x1 comparison plot for LIS data."""
        file_index1, file_index2 = file_indices

        # Plot the first dataset
        self._process_comparison_plot(plotter, file_index1, current_field_index,
                                    field_name1,
                                    figure, ax, 0, sdat1, plot_type, level=level)

        # Plot the second dataset
        self._process_comparison_plot(plotter, file_index2, current_field_index,
                                    field_name2,
                                    figure, ax, 1, sdat2, plot_type, level=level)

        # Plot the comparison (difference)
        self.comparison_plot = True
        self._process_comparison_plot(plotter, file_index1, current_field_index,
                                    field_name1,
                                    figure, ax, 2, sdat, plot_type, level=level)

    def _create_2x2_comparison_plot(self, plotter, file_indices, current_field_index,
                                field_name1, field_name2, figure,
                                plot_type, sdat1, sdat2, sdat, level=None):
        """Create a 2x2 comparison plot for LIS data."""
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

    def _process_comparison_plot(self, plotter, file_index, current_field_index, field_name, 
                                figure, ax, ax_index, source_data, plot_type, level=None):
        """Process a comparison plot."""
        self.config_manager.config._findex = file_index
        
        self.config_manager.pindex = current_field_index
        self.config_manager.axindex = ax_index

        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        figure.set_ax_opts_diff_field(ax[ax_index])
        
        # Get field to plot
        field_to_plot = self._get_field_to_plot_compare(source_data, field_name, file_index,
                                                    plot_type, figure, ax=ax[ax_index], level=level)
        
        # Plot the field
        plotter.comparison_plots(self.config_manager, field_to_plot, level=level)

    def _process_comparison_plot_2x2(self, plotter, file_index, current_field_index, field_name, 
                                    figure, gsi, ax_index, source_data, plot_type, level=None):
        """Process a 2x2 comparison plot for LIS data."""
        fig, axes = figure.get_fig_ax()
        ax1 = axes[gsi[0], gsi[1]] if isinstance(axes, np.ndarray) else plt.subplot(figure.gs[gsi[0], gsi[1]])
        figure.set_ax_opts_diff_field(ax1)
        
        # Get field to plot
        field_to_plot = self._get_field_to_plot_compare(source_data, field_name, file_index,
                                                    plot_type, figure, ax=ax1, level=level)
        
        # Plot the field
        plotter.comparison_plots(self.config_manager, field_to_plot, level=level)

    def _get_field_to_plot_compare(self, source_data, field_name, file_index, 
                                plot_type, figure, ax=None, level=None):
        """Get field data for comparison plots with LIS-specific handling."""
        if ax is None:
            ax = figure.get_axes()
        
        # For difference field
        if self.comparison_plot and self.config_manager.ax_opts['is_diff_field']:
            from eviz.lib.data.processor import Interp
            proc = Interp(self.config_manager, self.data2d_list)
            data2d, xx, yy = proc.regrid(plot_type)
            return data2d, xx, yy, self.field_names[0], plot_type, file_index, figure, ax
        
        # Get source name safely - don't use source_names[file_index] directly
        source_name = self._get_source_name_for_file_index(file_index)
        
        # For regular fields
        if isinstance(source_data, tuple):
            # This is a tuple of two datasets for comparison
            sdat1, sdat2 = source_data
            self.source_data = sdat1  # Temporarily set source_data for dimension handling
            
            source_name = self._get_source_name_for_file_index(file_index)
            dim1, dim2 = self.coord_names(source_name, source_data, field_name, plot_type)
  
            # Get time level
            time_level = self.config_manager.ax_opts['time_lev']
            if isinstance(time_level, str) and time_level == 'all':
                time_level = 0  # Default to first time level for comparison
            
            # Extract data based on plot type
            d1 = sdat1['vars'][field_name]
            d2 = sdat2['vars'][field_name]
            lon = self._get_field('east_west', d1)
            lat = self._get_field('north_south', d1)
            xs = np.array(lon)
            ys = np.array(lat)
            # Process data based on plot type
            if 'xy' in plot_type or 'polar' in plot_type:
                data2d1 = self._get_xy(d1, field_name, level=level, time_lev=time_level)
                
                # Temporarily switch source_data for second dataset
                temp_source_data = self.source_data
                self.source_data = sdat2
                data2d2 = self._get_xy(d2, field_name, level=level, time_lev=time_level)
                self.source_data = temp_source_data
                
                # Calculate difference
                data2d = data2d1 - data2d2
                
            else:
                # Other plot types (xt, tx)
                self.logger.warning(f"Plot type {plot_type} not fully supported for comparison")
                return None
            
            # Process coordinates
            if 'xt' in plot_type or 'tx' in plot_type:
                return data2d, None, None, field_name, plot_type, file_index, figure, ax
            else:
                # For XY plots, use longitude and latitude
               
                # Store data for difference calculation
                self.data2d_list.append(data2d)                
                return data2d, xs, ys, field_name, plot_type, file_index, figure, ax
        else:
            # Single dataset
            self.source_data = source_data  # Set source_data for dimension handling
            
            # Get dimension names using source_name instead of indexing into source_names
            dim1, dim2 = self.coord_names(source_name, source_data, field_name, plot_type)
            
            # Get time level
            time_level = self.config_manager.ax_opts['time_lev']
            if isinstance(time_level, str) and time_level == 'all':
                time_level = 0  # Default to first time level for comparison
            
            # Extract data based on plot type
            d = source_data['vars'][field_name]
            
            # Process data based on plot type
            if 'xy' in plot_type or 'polar' in plot_type:
                data2d = self._get_xy(d, field_name, level=level, time_lev=time_level)
            else:
                # Other plot types (xt, tx)
                self.logger.warning(f"Plot type {plot_type} not fully supported for comparison")
                return None
            
            # Store data for difference calculation
            self.data2d_list.append(data2d)
            
            # Process coordinates
            if 'xt' in plot_type or 'tx' in plot_type:
                return data2d, None, None, field_name, plot_type, file_index, figure, ax
            else:
                # For XY plots, use longitude and latitude
                return data2d, xs, ys, field_name, plot_type, file_index, figure, ax

    def _get_source_name_for_file_index(self, file_index):
        """
        Get the source name for a given file index.
        This handles the case where multiple files come from the same source.
        """
        # Try to get source name from file_list
        if hasattr(self.config_manager, 'file_list') and file_index in self.config_manager.file_list:
            return self.config_manager.file_list[file_index].get('source_name', 'lis')
        
        # Try to get source name from map_params
        if hasattr(self.config_manager, 'map_params'):
            for param_key, param_config in self.config_manager.map_params.items():
                if param_key == file_index or param_config.get('file_index') == file_index:
                    return param_config.get('source_name', 'lis')
        
        # If we can't find the source name, default to 'lis' since we're in the Lis class
        return 'lis'
    

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
            
            # Initialize p_top for both datasets if needed
            self._init_domain_for_comparison(sdat1, sdat2)

            # Determine file indices
            source_name1, source_name2 = map1['source_name'], map2['source_name']
            filename1, filename2 = map1['filename'], map2['filename']
            file_indices = self._get_file_indices(source_name1, source_name2, filename1, filename2)

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
        
        # Get levels for the plots
        levels = self.config_manager.get_levels(field_name1, plot_type + 'plot')
        if not levels:
            self.logger.warning(f' -> No levels specified for {field_name1}')
            return
        
        for level in levels:
            # Create a figure with 2x1 subplots (side by side)
            figure = Figure(self.config_manager, plot_type, nrows=1, ncols=2)
            ax = figure.get_axes()
            self.config_manager.level = level
            
            # Create the 2x1 side-by-side comparison plot
            self._create_2x1_side_by_side_plot(plotter, file_indices,
                                            current_field_index,
                                            field_name1, field_name2, figure, ax,
                                            plot_type, sdat1, sdat2, level)
            
            # Save the plot
            print_map(self.config_manager, plot_type, self.config_manager.findex, figure, level=level)

    def _process_other_side_by_side_plots(self, plotter, file_indices, current_field_index,
                                        field_name1, field_name2, plot_type, sdat1, sdat2):
        """Process side-by-side comparison plots for other plot types."""
        file_index1, file_index2 = file_indices
        
        # Create a figure with side by side plots
        figure = Figure(self.config_manager, plot_type, nrows=1, ncols=2)
        ax = figure.get_axes()
        self.config_manager.level = None
        
        # Create the 2x1 side-by-side comparison plot
        self._create_2x1_side_by_side_plot(plotter, file_indices, current_field_index,
                                        field_name1, field_name2, figure, ax,
                                        plot_type, sdat1, sdat2)
        
        # Save the plot
        print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def _create_2x1_side_by_side_plot(self, plotter, file_indices, current_field_index,
                                    field_name1, field_name2, figure, ax,
                                    plot_type, sdat1, sdat2, level=None):
        """
        Create a 2x1 side-by-side comparison plot for LIS data.
        
        The layout is:
        - Left subplot: First dataset
        - Right subplot: Second dataset
        """
        file_index1, file_index2 = file_indices
        
        # Plot the first dataset in the left subplot
        self.comparison_plot = False
        self._process_side_by_side_plot(plotter, file_index1, current_field_index,
                                    field_name1,
                                    figure, ax, 0, sdat1, plot_type, level=level)
        
        # Plot the second dataset in the right subplot
        self._process_side_by_side_plot(plotter, file_index2, current_field_index,
                                    field_name2,
                                    figure, ax, 1, sdat2, plot_type, level=level)

    def _process_side_by_side_plot(self, plotter, file_index, current_field_index, field_name, 
                                figure, ax, ax_index, source_data, plot_type, level=None):
        """Process a side-by-side plot for LIS data."""
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
        field_to_plot = self._get_field_to_plot_side_by_side(source_data, field_name, file_index,
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
        """Get field data for side-by-side plots with LIS-specific handling."""
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
        if 'xy' in plot_type or 'polar' in plot_type:
            data2d = self._get_xy(d, field_name, level=level, time_lev=time_level)
        else:
            # Other plot types (xt, tx)
            self.logger.warning(f"Plot type {plot_type} not fully supported for side-by-side comparison")
            return None
        
        lon = self._get_field('east_west', data2d)
        lat = self._get_field('north_south', data2d)
        xs = np.array(lon)
        ys = np.array(lat)

        # Process coordinates
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        else:
            return data2d, xs, ys, field_name, plot_type, file_index, figure, ax

    def _init_domain_for_comparison(self, sdat1, sdat2):
        """Initialize domain information for both datasets in a comparison."""
        # Save current source_data
        original_source_data = self.source_data
        
        # Initialize domain for first dataset
        self.source_data = sdat1        
        # Initialize domain for second dataset
        self.source_data = sdat2
        
        # Restore original source_data
        self.source_data = original_source_data

    def get_field_dim_name(self, source_data, dim_name, field_name):
        d = source_data['vars'][field_name]
        field_dims = list(d.dims) 
        names = self.get_model_dim_name(self.source_name, dim_name).split(',')
        common = list(set(names).intersection(field_dims))
        dim = list(common)[0] if common else None
        return dim