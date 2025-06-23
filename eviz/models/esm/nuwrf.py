import logging
import os
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from eviz.models.gridded_source import GriddedSource
from eviz.lib.autoviz.figure import Figure
from eviz.lib.autoviz.utils import create_gif
from eviz.lib.data.utils import apply_conversion

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


_COORD_PAIR_MAP = {"XLAT": ("XLAT", "XLONG"),
                   "XLONG": ("XLAT", "XLONG"),
                   "XLAT_M": ("XLAT_M", "XLONG_M"),
                   "XLONG_M": ("XLAT_M", "XLONG_M"),
                   "XLAT_U": ("XLAT_U", "XLONG_U"),
                   "XLONG_U": ("XLAT_U", "XLONG_U"),
                   "XLAT_V": ("XLAT_V", "XLONG_V"),
                   "XLONG_V": ("XLAT_V", "XLONG_V"),
                   "CLAT": ("CLAT", "CLONG"),
                   "CLONG": ("CLAT", "CLONG")}


_COORD_VARS = ("XLAT", "XLONG", "XLAT_M", "XLONG_M", "XLAT_U", "XLONG_U",
               "XLAT_V", "XLONG_V", "CLAT", "CLONG")

_LAT_COORDS = ("XLAT", "XLAT_M", "XLAT_U", "XLAT_V", "CLAT")

_LON_COORDS = ("XLONG", "XLONG_M", "XLONG_U", "XLONG_V", "CLONG")

_TIME_COORD_VARS = ("XTIME",)


@dataclass
class NuWrf(GriddedSource):
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

    def _load_source_data(self, source_name, filename):
        """Common method to load source data."""
        reader = self._get_reader(source_name)
        if not reader:
            self.logger.error(f"No reader found for source {source_name}")
            return None
            
        source_data = reader.read_data(filename)
        if not source_data:
            self.logger.error(f"Failed to read data from {filename}")
            return None
            
        return source_data

    def process_simple_plots(self, plotter):
        """Common implementation for simple plots."""
        map_params = self.config_manager.map_params
        field_num = 0
        self.config_manager.findex = 0
        
        for i in map_params.keys():
            field_name = map_params[i]['field']
            source_name = map_params[i]['source_name']
            self.source_name = source_name
            filename = map_params[i]['filename']
            file_index = self.config_manager.get_file_index(filename)
            
            self.source_data = self._load_source_data(source_name, filename)
            if not self.source_data:
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

    def _init_model_specific_data(self):
        """Hook for model-specific initialization. Override in subclasses."""
        pass

    def _get_field_for_simple_plot(self, field_name, plot_type):
        """Hook for model-specific simple plot field processing. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_field_for_simple_plot")

    # Common utility methods
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

    def _get_time_string(self, data_array, time_index):
        """Get a readable time string for the given time index."""
        try:
            time_dim = self._get_time_dimension_name(data_array)
            time_value = None
            
            if hasattr(data_array, 'XTIME') and time_dim in data_array.XTIME.dims:
                time_value = data_array.XTIME.isel({time_dim: time_index}).values
            elif time_dim and time_dim in data_array.coords:
                time_value = data_array.coords[time_dim].values[time_index]
            elif hasattr(self, 'source_data') and self.source_data:
                for var_name in ['Times', 'time', 'Time', 'XTIME']:
                    if var_name in self.source_data['vars']:
                        time_var = self.source_data['vars'][var_name]
                        if time_dim and time_dim in time_var.dims:
                            time_value = time_var.isel({time_dim: time_index}).values
                            break
            
            # Convert to readable format
            if time_value is not None:
                if isinstance(time_value, (pd.Timestamp, np.datetime64)):
                    return pd.to_datetime(time_value).strftime('%Y-%m-%d %H:%M')
                elif isinstance(time_value, str):
                    try:
                        return pd.to_datetime(time_value).strftime('%Y-%m-%d %H:%M')
                    except:
                        return time_value
                else:
                    return pd.to_datetime(time_value).strftime('%Y-%m-%d %H:%M')
            
            base_time = pd.Timestamp('2000-01-01')
            synthetic_time = base_time + pd.Timedelta(days=time_index)
            return synthetic_time.strftime('%Y-%m-%d %H:%M')
            
        except Exception as e:
            self.logger.warning(f"Error getting time string: {e}")
            return f"Time step {time_index}"

    def _get_time_dimension_name(self, data_array):
        """Get the name of the time dimension from the data array."""
        time_dim_names = ['Time', 'time', 't', 'T', 'times', 'Times']
        
        for dim_name in time_dim_names:
            if dim_name in data_array.dims:
                return dim_name
        
        return None

    def _extract_xy_data(self, d, level, time_lev):
        """ Extract XY slice from N-dim data field"""
        if d is None:
            return
        if level:
            level = int(level)

        self.logger.debug(f"Selecting time level: {time_lev}")
        tc_dim = self.get_model_dim_name('tc') or 'Time'
        zc_dim = self.get_model_dim_name('zc') 

        if tc_dim in d.dims:
            data2d = d.isel({tc_dim: time_lev})
        else:
            data2d = d
        data2d = data2d.squeeze()

        zname = self.find_matching_dimension(d.dims, 'zc')
        if zname in data2d.dims:
            if 'soil' in zname:
                data2d = data2d.isel({zname: 0})
            else:
                if self.source_name == 'lis':
                    lev_to_plot = level
                    data2d = data2d.isel({zname: 0})
                else:
                    difference_array = np.absolute(self.levs - level)
                    index = difference_array.argmin()
                    lev_to_plot = self.levs[index]  # should I use this instead of index?
                    data2d = data2d.isel({zname: index})

        return apply_conversion(self.config_manager, data2d, d.name)

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

    def coord_names(self, source_name, source_data, pid):
        """ Get WRF or LIS coord names based on field and plot type

        Parameters:
            source_name (str) : source name
            source_data (dict) : source data
            pid (str) : plot type
        """
        coords = []
        if source_name == 'wrf':
            stag = source_data.attrs.get('stagger', None)
            xsuf, ysuf, zsuf = "", "", ""
            if stag == "X":
                xsuf = "_stag"
            elif stag == "Y":
                ysuf = "_stag"
            elif stag == "Z":
                zsuf = "_stag"

            for name in self.get_model_coord_name(source_name, 'xc').split(","):
                if name in source_data.coords.keys():
                    if xsuf:
                        coords.append((name, self.get_model_dim_name('xc')+xsuf))
                    else:
                        coords.append((name, self.get_model_dim_name('xc')))
                    break

            for name in self.get_model_coord_name(source_name, 'yc').split(","):
                if name in source_data.coords.keys():
                    if ysuf:
                        coords.append((name, self.get_model_dim_name('yc')+xsuf))
                    else:
                        coords.append((name, self.get_model_dim_name('yc')))
                    break
        else:
            xc = self.get_model_dim_name('xc')
            if xc:
                coords.append(xc)
            yc = self.get_model_dim_name('yc')
            if yc:
                coords.append(yc)

        if source_name == 'wrf':
            zc = self.get_model_dim_name('zc')   # field_dim_name(source_data, 'zc')?
            if zc:
                coords.append(zc)
        else:
            for name in self.get_model_dim_name('zc').split(","):
                if hasattr(source_data, "coords") and name in source_data.coords.keys():
                    coords.append(name)
                    break

        if source_name == 'wrf':
            tc = self.get_model_dim_name('tc')   # field_dim_name(source_data, 'tc')?
        else:
            tc = self.get_model_dim_name('tc')

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

    def get_field_dim_name(self, source_data: dict, dim_name: str):
        field_dims = list(source_data.dims)
        model_dim = self.get_model_dim_name(dim_name)
        if not model_dim:
            return None
        names = model_dim.split(',')
        common = list(set(names).intersection(field_dims))
        dim = list(common)[0] if common else None
        return dim

    def get_model_dim_name(self, dim_name: str):
        return self.config_manager.get_model_dim_name(dim_name=dim_name)

    def get_model_coord_name(self, source_name: str, dim_name: str):
        try:
            coord = self.config_manager.meta_coords[dim_name][source_name]['coords']
            return coord
        except KeyError:
            return None

    def get_dd(self, source_name, source_data, dim_name, field_name):
        d = source_data['vars'][field_name]
        field_dims = d.dims
        names = self.get_model_dim_name(dim_name)
        for d in field_dims:
            if d in names:
                return d

    def _get_field(self, name, data):
        try:
            return data[name]
        except Exception as e:
            self.logger.error('key error: %s, not found' % str(e))
            return None

    def find_matching_dimension(self, field_dims, dim_name):
        """
        Returns the first matching dimension name found in `field_dims` 
        that is also in the meta_coords dictionary for the given `dim_name`.
        
        Parameters:
        - field_dims: tuple of dimension names (e.g., from xarray.DataArray.dims)
        - dim_name: str
        
        Returns:
        - matched dimension name (str) or None
        """
        for dim in field_dims:
            if dim in self.config_manager.meta_coords[dim_name][self.source_name]['dim']:
                return dim
        return None
    

