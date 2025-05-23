import logging
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from eviz.models.esm.gridded import Gridded
from eviz.lib.autoviz.figure import Figure
from eviz.lib.autoviz.utils import print_map, create_gif

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class NuWrf(Gridded):
    """ Define NUWRF specific model data and functions."""

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.p_top = None

    def _get_reader_old(self, source_name):
        """Get the appropriate reader for the source."""
        if source_name in self.config_manager.readers:
            if isinstance(self.config_manager.readers[source_name], dict):
                # New structure - get the primary reader
                readers_dict = self.config_manager.readers[source_name]
                if 'NetCDF' in readers_dict:
                    return readers_dict['NetCDF']
                elif readers_dict:
                    return next(iter(readers_dict.values()))
            else:
                # Old structure - direct access
                return self.config_manager.readers[source_name]
        
        self.logger.error(f"Source {source_name} not found in readers")
        return None

    def _get_reader(self, source_name):
        """Get the appropriate reader for the source.
        
        This method now returns a wrapper that provides the expected interface
        while using the pipeline's data sources.
        """
        if not hasattr(self.config_manager, '_pipeline'):
            self.logger.error("No pipeline available in config_manager")
            return None
            
        class ReaderWrapper:
            def __init__(self, pipeline):
                self.pipeline = pipeline
                
            def read_data(self, file_path):
                data_source = self.pipeline.get_data_source(file_path)
                if data_source is None:
                    return None
                    
                if not hasattr(data_source, 'dataset') or data_source.dataset is None:
                    return None
                    
                return {
                    'vars': data_source.dataset.data_vars,
                    'attrs': data_source.dataset.attrs
                }
                
        return ReaderWrapper(self.config_manager._pipeline)

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
            
            # Model-specific initialization (hook for subclasses)
            self._init_model_specific_data()
            
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
                    file_index = field_num
                    
                    reader = self._get_reader(source_name)
                    if not reader:
                        self.logger.error(f"No reader found for source {source_name}")
                        continue
                        
                    self.source_data = reader.read_data(filename)
                    if not self.source_data:
                        self.logger.error(f"Failed to read data from {filename}")
                        continue
                        
                    self._global_attrs = self.source_data['attrs']
                    if source_name == 'wrf' and not self.p_top:
                        self._init_domain()

                    self.config_manager.findex = file_index
                    self.config_manager.pindex = field_num
                    self.config_manager.axindex = 0
                    
                    self._process_field_plots(field_name, map_params[i], plotter)
                    field_num += 1
            
            if self.config_manager.make_gif:
                create_gif(self.config_manager)

    def _process_field_plots(self, field_name, field_config, plotter):
        """Process all plot types for a given field."""
        for plot_type in field_config['to_plot']:
            self.logger.info(f"Plotting {field_name}, {plot_type} plot")
            
            # Initialize plot configuration
            temp_figure = Figure(self.config_manager, plot_type)
            self.config_manager.ax_opts = temp_figure.init_ax_opts(field_name)
            
            d = self.source_data['vars'][field_name]
            time_levels = self._get_time_levels(d)
            
            if 'xy' in plot_type:
                self._process_xy_plots(field_name, plot_type, d, time_levels, plotter)
            else:
                self._process_non_xy_plots(field_name, plot_type, d, time_levels, plotter)

    def _get_time_levels(self, data_array):
        """Get the list of time levels to process based on configuration."""
        time_dim = self._get_time_dimension_name(data_array)
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        
        if time_level_config == 'all':
            if time_dim and time_dim in data_array.dims:
                num_times = data_array.sizes[time_dim]
                return list(range(num_times))
            else:
                return [0]
        else:
            return [time_level_config]

    def _process_xy_plots(self, field_name, plot_type, data_array, time_levels, plotter):
        """Process XY plots with multiple levels and time steps."""
        levels = self.config_manager.get_levels(field_name, plot_type + 'plot')
        if not levels:
            self.logger.warning(f' -> No levels specified for {field_name}')
            return
            
        for level in levels:
            self.logger.info(f' -> Processing {len(time_levels)} time levels for level {level}')
            
            for time_step in time_levels:
                self._create_single_plot(field_name, plot_type, data_array, time_step, plotter, level)

    def _process_non_xy_plots(self, field_name, plot_type, data_array, time_levels, plotter):
        """Process non-XY plots (yz, xt, tx, etc.)."""
        for time_step in time_levels:
            self._create_single_plot(field_name, plot_type, data_array, time_step, plotter, level=None)

    def _create_single_plot(self, field_name, plot_type, data_array, time_step, plotter, level=None):
        """Create a single plot for the given parameters."""
        # Create new figure for each time level
        figure = Figure(self.config_manager, plot_type)
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)
        
        # Set time-related configuration
        self.config_manager.time_level = time_step
        self.config_manager.real_time = self._get_time_string(data_array, time_step)
        
        field_to_plot = self._get_field_to_plot(
            field_name, self.config_manager.findex, plot_type, figure, time_step, level=level
        )        
        if level is not None:
            plotter.single_plots(self.config_manager, field_to_plot=field_to_plot, level=level)
            print_map(self.config_manager, plot_type, self.config_manager.findex, figure, level=level)
        else:
            plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
            print_map(self.config_manager, plot_type, self.config_manager.findex, figure)
        
        plt.close(figure)

    def _get_time_string(self, data_array, time_index):
        """
        Get a readable time string for the given time index.
        
        Args:
            data_array: The data array containing time information
            time_index: The time index to extract
            
        Returns:
            A formatted time string or a fallback representation
        """
        try:
            time_dim = self._get_time_dimension_name(data_array)
            
            # Try different approaches to get the time value
            time_value = None
            
            # Method 1: Check for XTIME (WRF-specific)
            if hasattr(data_array, 'XTIME') and time_dim in data_array.XTIME.dims:
                time_value = data_array.XTIME.isel({time_dim: time_index}).values
            
            # Method 2: Check for time coordinates in the data array
            elif time_dim and time_dim in data_array.coords:
                time_value = data_array.coords[time_dim].values[time_index]
            
            # Method 3: Check for time variable in source_data
            elif hasattr(self, 'source_data') and self.source_data:
                for var_name in ['Times', 'time', 'Time', 'XTIME']:
                    if var_name in self.source_data['vars']:
                        time_var = self.source_data['vars'][var_name]
                        if time_dim and time_dim in time_var.dims:
                            time_value = time_var.isel({time_dim: time_index}).values
                            break
            
            # Method 4: Check for time attribute in data array
            elif hasattr(data_array, 'time') and hasattr(data_array.time, 'isel') and time_dim:
                time_value = data_array.time.isel({time_dim: time_index}).values
            
            # Convert to readable format if we found a time value
            if time_value is not None:
                # Handle different time formats
                if isinstance(time_value, (pd.Timestamp, np.datetime64)):
                    return pd.to_datetime(time_value).strftime('%Y-%m-%d %H:%M')
                elif isinstance(time_value, str):
                    try:
                        return pd.to_datetime(time_value).strftime('%Y-%m-%d %H:%M')
                    except:
                        return time_value
                else:
                    return pd.to_datetime(time_value).strftime('%Y-%m-%d %H:%M')
            
            # Fallback: create a synthetic time
            base_time = pd.Timestamp('2000-01-01')
            synthetic_time = base_time + pd.Timedelta(days=time_index)
            return synthetic_time.strftime('%Y-%m-%d %H:%M')
            
        except Exception as e:
            self.logger.warning(f"Error getting time string: {e}")
            return f"Time step {time_index}"

    def _get_time_dimension_name(self, data_array):
        """
        Get the name of the time dimension from the data array.
        
        Args:
            data_array: The data array to examine
            
        Returns:
            The name of the time dimension or None if not found
        """
        # Common time dimension names, ordered by preference
        time_dim_names = ['Time', 'time', 't', 'T', 'times', 'Times']
        
        for dim_name in time_dim_names:
            if dim_name in data_array.dims:
                return dim_name
        
        return None

    def _init_model_specific_data(self):
        """Hook for model-specific initialization. Override in subclasses."""
        pass

    # def _get_field_to_plot(self, field_name, file_index, plot_type, figure, time_level, level=None):
    #     """Template method for getting field to plot."""
    #     ax = figure.get_axes()
        
    #     # Get dimension names (model-specific)
    #     dim1, dim2 = self.coord_names(self.source_name, self.source_data, field_name, plot_type)
        
    #     # Get data based on plot type
    #     data2d = self._get_data_for_plot_type(field_name, plot_type, time_level, level)
        
    #     # Process coordinates (model-specific)
    #     return self._process_coordinates(data2d, dim1, dim2, field_name, plot_type, file_index, figure, ax)

    def _process_coordinates(self, data2d, dim1, dim2, field_name, plot_type, file_index, figure, ax):
        """
        Process coordinates for the plot.
        """
        pass  # Placeholder for coordinate processing logic
    
    def _apply_vertical_level_selection(self, data2d, field_name, level):
        """
        Apply vertical level selection to the data.
        
        Args:
            data2d: The data array after time selection
            field_name: Name of the field being processed
            level: The vertical level to select
            
        Returns:
            Data array with vertical level selection applied
        """
        # Get the vertical dimension name
        zc_dim = self.get_field_dim_name(self.source_name, self.source_data, 'zc', field_name)
        
        # If no vertical dimension or no level specified, return as is
        if not zc_dim or level is None:
            return data2d
        
        # If the vertical dimension exists in the data, select the level
        if zc_dim in data2d.dims:
            try:
                # Basic implementation - subclasses will override with model-specific logic
                return data2d.isel({zc_dim: level})
            except Exception as e:
                self.logger.warning(f"Error selecting vertical level: {e}")
        
        return data2d

    # Hook methods that can be overridden by subclasses
    @staticmethod
    def _preprocess_data(d):
        """Pre-process the data array. Default implementation just squeezes dimensions."""
        return d.squeeze()

    def _apply_time_selection(self, original_data, data2d, time_dim, time_lev, field_name, level):
        """
        Apply time selection or averaging to the data.
        Default implementation just selects the time level if a time dimension exists.
        """
        if time_dim and time_dim in original_data.dims:
            try:
                return original_data.isel({time_dim: time_lev}).squeeze()
            except Exception as e:
                self.logger.warning(f"Error selecting time level: {e}")
        return data2d


    @staticmethod
    def set_global_attrs(source_name, ds_attrs):
        """Return a tuple of global attributes from WRF or LIS dataset """
        tmp = dict()
        for attr in ds_attrs.keys():
            try:
                tmp[attr] = ds_attrs[attr]
                if source_name == "lis":
                    if attr == "DX" or attr == "DY":
                        # Convert LIS units to MKS
                        tmp[attr] = ds_attrs[attr] * 1000.0
            except KeyError:
                tmp[attr] = None
        return tmp

    def coord_names(self, source_name, source_data, field_name, pid):
        """ Get WRF or LIS coord names based on field and plot type

        Parameters:
            source_name (str) : source name
            source_data (dict) : source data
            field_name(str) : Field name associated with this plot
            pid (str) : plot type
        """
        coords = []
        field = source_data['vars'][field_name]

        if source_name == 'wrf':
            stag = source_data['vars'][field_name].stagger
            xsuf, ysuf, zsuf = "", "", ""
            if stag == "X":
                xsuf = "_stag"
            elif stag == "Y":
                ysuf = "_stag"
            elif stag == "Z":
                zsuf = "_stag"

            for name in self.get_model_coord_name(source_name, 'xc').split(","):
                if name in field.coords.keys():
                    coords.append((name, self.get_model_dim_name(source_name, 'xc')+xsuf))
                    break

            for name in self.get_model_coord_name(source_name, 'yc').split(","):
                if name in field.coords.keys():
                    coords.append((name, self.get_model_dim_name(source_name, 'yc')+ysuf))
                    break
        else:  # 'lis'
            xc = self.get_model_dim_name(source_name, 'xc')
            if xc:
                coords.append(xc)
            yc = self.get_model_dim_name(source_name, 'yc')
            if yc:
                coords.append(yc)

        if source_name == 'wrf':
            zc = self.get_field_dim_name(source_name, field, 'zc')
            if zc:
                coords.append(zc)
        else:
            for name in self.get_model_dim_name(source_name, 'zc').split(","):
                if hasattr(field, "coords") and name in field.coords.keys():
                    coords.append(name)
                    break

        if source_name == 'wrf':
            tc = self.get_field_dim_name(source_name, field, 'tc')
        else:
            tc = self.get_field_dim_name(source_name, field, 'tc')

        if tc:
            coords.append(tc)

        dim1, dim2 = None, None
        if source_name == 'wrf':
            if 'yz' in pid:
                dim1 = coords[1]
                dim2 = coords[2]
            elif 'xt' in pid:
                dim1 = coords[3] if len(coords) > 3 else None
            elif 'tx' in pid:
                dim1 = coords[0]
                dim2 = 'Time'
            else:
                dim1 = coords[0]
                dim2 = coords[1]
        else:
            if 'xt' in pid:
                dim1 = coords[3]
            elif 'tx' in pid:
                dim1 = coords[0]
                dim2 = coords[3]
            else:
                dim1 = coords[0]
                dim2 = coords[1]
        return dim1, dim2

    def get_field_dim_name(self, source_name: str, source_data: dict, dim_name: str):
        field_dims = list(source_data.dims)
        model_dim = self.get_model_dim_name(source_name, dim_name)
        if not model_dim:
            return None
        names = model_dim.split(',')
        common = list(set(names).intersection(field_dims))
        dim = list(common)[0] if common else None
        return dim

    def get_model_dim_name(self, source_name: str, dim_name: str):
        try:
            dim = self.config_manager.meta_coords[dim_name][source_name]['dim']
            return dim
        except KeyError:
            return None

    def get_model_coord_name(self, source_name: str, dim_name: str):
        try:
            coord = self.config_manager.meta_coords[dim_name][source_name]['coords']
            return coord
        except KeyError:
            return None

    def get_dd(self, source_name, source_data, dim_name, field_name):
        d = source_data['vars'][field_name]
        field_dims = d.dims
        names = self.get_model_dim_name(source_name, dim_name)
        for d in field_dims:
            if d in names:
                return d

    def _get_field(self, name, data):
        try:
            return data[name]
        except Exception as e:
            self.logger.error('key error: %s, not found' % str(e))
            return None
    