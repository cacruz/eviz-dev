import logging
import warnings
from dataclasses import dataclass

import pandas as pd

from eviz.models.esm.grid_data import GridData
from eviz.lib.data.utils import apply_conversion

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class NuWrf(GridData):
    """ Define NUWRF specific model data and functions."""

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.p_top = None

    def _get_reader(self, source_name):
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
        # If no time dimension provided, try to find one
        if time_dim is None:
            for dim_name in ['time', 'Time', 't', 'T']:
                if dim_name in data_array.dims:
                    time_dim = dim_name
                    break
        
        # If still no time dimension, return a default timestamp
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
            
            # Third try: Check for WRF-specific time variables
            if hasattr(data_array, 'XTIME'):
                if time_dim in data_array.XTIME.dims:
                    return data_array.XTIME.isel({time_dim: time_index}).values
            
            # Fourth try: Check for time-related variables in source_data
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


    def _init_model_specific_data(self):
        """Hook for model-specific initialization. Override in subclasses."""
        pass

    def _get_field_to_plot(self, field_name, file_index, plot_type, figure, time_level, level=None):
        """Template method for getting field to plot."""
        ax = figure.get_axes()
        
        # Get dimension names (model-specific)
        dim1, dim2 = self.coord_names(self.source_name, self.source_data, field_name, plot_type)
        
        # Get data based on plot type
        data2d = self._get_data_for_plot_type(field_name, plot_type, time_level, level)
        
        # Process coordinates (model-specific)
        return self._process_coordinates(data2d, dim1, dim2, field_name, plot_type, file_index, figure, ax)

    def _process_coordinates(self, data2d, dim1, dim2, field_name, plot_type, file_index, figure, ax):
        """
        Process coordinates for the plot.
        
        Args:
            data2d: The 2D data array to plot
            dim1, dim2: Dimension names for the plot
            field_name: Name of the field being plotted
            plot_type: Type of plot (xy, yz, etc.)
            file_index: Index of the file being processed
            figure: The figure object
            ax: The axes object
            
        Returns:
            Tuple containing processed data and metadata for plotting
        """
        # For time series plots, no coordinate processing needed
        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        
        # For other plot types, extract coordinates but don't process them
        # (Subclasses will override this with specific processing)
        try:
            # Try to get coordinates from data2d
            if hasattr(data2d, 'coords') and dim1 in data2d.coords and dim2 in data2d.coords:
                xs = data2d.coords[dim1].values
                ys = data2d.coords[dim2].values
                return data2d, xs, ys, field_name, plot_type, file_index, figure, ax
        except Exception as e:
            self.logger.error(f"Error processing coordinates: {e}")
        
        # Default fallback
        return data2d, None, None, field_name, plot_type, file_index, figure, ax


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

    def _get_xy(self, d, field_name, level, time_lev):
        """
        Extract XY slice from N-dim data field with common logic.
        
        This is a template method that defines the skeleton of the algorithm,
        delegating model-specific steps to hook methods that subclasses can override.
        """
        if d is None:
            return None
            
        # Convert level to int if provided
        if level:
            level = int(level)
        
        # 1. Pre-process data (squeeze dimensions, etc.)
        data2d = self._preprocess_data(d)
        
        # 2. Get time dimension name (model-specific)
        time_dim = self._get_time_dimension_name(d)
        
        # 3. Apply time selection or averaging (model-specific)
        data2d = self._apply_time_selection(d, data2d, time_dim, time_lev, field_name, level)
        
        # 4. Apply vertical level selection (model-specific)
        data2d = self._apply_vertical_level_selection(data2d, field_name, level)
        
        # 5. Apply conversion and return
        return apply_conversion(self.config_manager, data2d, field_name)

    # Hook methods that can be overridden by subclasses
    @staticmethod
    def _preprocess_data(d):
        """Pre-process the data array. Default implementation just squeezes dimensions."""
        return d.squeeze()

    def _get_time_dimension_name(self, d):
        """
        Get the name of the time dimension.
        Default implementation tries common time dimension names.
        """
        for dim_name in ['Time', 'time', 't', 'T']:
            if dim_name in d.dims:
                return dim_name
        return None

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
        d = source_data['vars'][field_name]
        if self.source_name == 'wrf':
            stag = d.stagger
            xsuf, ysuf, zsuf = "", "", ""
            if stag == "X":
                xsuf = "_stag"
            elif stag == "Y":
                ysuf = "_stag"
            elif stag == "Z":
                zsuf = "_stag"

            for name in self.get_model_coord_name(source_name, 'xc').split(","):
                if name in d.coords.keys():
                    coords.append((name, self.get_model_dim_name(source_name, 'xc')+xsuf))
                    break

            for name in self.get_model_coord_name(source_name, 'yc').split(","):
                if name in d.coords.keys():
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
            zc = self.get_field_dim_name(source_name, source_data, 'zc', field_name)
            if zc:
                coords.append(zc)
        else:
            for name in self.get_model_dim_name(source_name, 'zc').split(","):
                if name in d.coords.keys():
                    coords.append(name)
                    break

        if source_name == 'wrf':
            tc = self.get_field_dim_name(source_name, source_data, 'tc', field_name)
        else:
            tc = self.get_field_dim_name(source_data, 'tc', field_name)

        if tc:
            coords.append(tc)

        dim1, dim2 = None, None
        if source_name == 'wrf':
            if 'yz' in pid:
                dim1 = coords[1]
                dim2 = coords[2]
            elif 'xt' in pid:
                dim1 = 'Time'
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

    def get_field_dim_name(self):
        """Hook for model-specific field-dim-name. Override in subclasses."""
        pass

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
