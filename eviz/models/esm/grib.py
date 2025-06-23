import warnings
from dataclasses import dataclass
import numpy as np
import xarray as xr
from eviz.lib.autoviz.figure import Figure
from eviz.lib.autoviz.utils import create_gif, print_map
from eviz.lib.data.utils import apply_mean
from eviz.lib.data.utils import apply_conversion
from eviz.models.gridded_source import GriddedSource

warnings.filterwarnings("ignore")


@dataclass
class Grib(GriddedSource):
    """ Define Grib specific model data and functions."""

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()
        self.comparison_plot = False
        self.source_name = 'grib'

    def _init_model_specific_data(self):
        """Grib-specific initialization."""
        if not self.p_top:
            self._init_domain()

    def _init_domain(self):
        """ Approximate unknown fields """
        pass    # Implement domain initialization logic here

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

    def _process_xy_plot(self, data_array: xr.DataArray, field_name: str, file_index: int,
                         plot_type: str, figure):
        """Process xy or polar plot types."""
        levels = self.config_manager.get_levels(field_name, plot_type + 'plot')
        do_zsum = self.config_manager.ax_opts.get('zsum', False)  # Use .get with default

        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'
        num_times = data_array[tc_dim].size if tc_dim in data_array.dims else 1
        time_levels = range(num_times) if time_level_config == 'all' else [
            time_level_config]

        if not levels and not do_zsum:
            return

        if levels:
            self._process_level_plots(data_array, field_name, file_index, plot_type,
                                      figure, time_levels, levels)
        else:
            self._process_zsum_plots(data_array, field_name, file_index, plot_type,
                                     figure, time_levels)

    def _process_level_plots(self, data_array: xr.DataArray, field_name: str,
                             file_index: int, plot_type: str, figure,
                             time_levels: list, levels: dict):
        """Process plots for specific vertical levels."""
        self.logger.debug(f' -> Processing {len(time_levels)} time levels')
        zc_dim = self.config_manager.get_model_dim_name('zc') or 'lev'
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        has_vertical_dim = zc_dim and zc_dim in data_array.dims

        for level_val in levels.keys():  # Iterate through level values
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
                    field_to_plot = self._get_field_to_plot(data_at_time, field_name,
                                                            file_index, plot_type, figure,
                                                            t)
                else:
                    field_to_plot = self._get_field_to_plot(data_at_time, field_name,
                                                            file_index, plot_type, figure,
                                                            t,
                                                            level=level_val)

                if field_to_plot:
                    plot_result = self.create_plot(field_name, field_to_plot)                    
                    print_map(self.config_manager, plot_type, self.config_manager.findex, plot_result, 
                              level=level_val)

    def _process_other_plot(self, data_array: xr.DataArray, field_name: str,
                            file_index: int, plot_type: str, figure):
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

        # Assuming these plot types (xt, tx) might not need time slicing here,
        # or slicing is handled within _get_field_to_plot
        # Pass the full data_array and let _get_field_to_plot handle slicing if needed
        field_to_plot = self._get_field_to_plot(data_array, field_name, file_index,
                                                plot_type, figure,
                                                time_level=time_level_config)
        if field_to_plot:
            plot_result = self.create_plot(field_name, field_to_plot)
            print_map(self.config_manager, plot_type, self.config_manager.findex, plot_result)

    def _process_zsum_plots(self, data_array: xr.DataArray, field_name: str,
                            file_index: int, plot_type: str, figure,
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
            field_to_plot = self._get_field_to_plot(data_at_time, field_name,
                                                    # Pass None for ax initially
                                                    file_index, plot_type, figure, t)
            if field_to_plot:
                plot_result = self.create_plot(field_name, field_to_plot)
                print_map(self.config_manager, plot_type, self.config_manager.findex,
                             plot_result)

    def _set_grib_extents(self, xs, ys):
        """Set GRIB-specific map extents."""

        latN = max(ys)
        latS = min(ys)
        lonW = min(xs)
        lonE = max(xs)
        self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
        self.config_manager.ax_opts['central_lon'] = np.mean([lonW, lonE])
        self.config_manager.ax_opts['central_lat'] = np.mean([latS, latN])

    def _get_field_to_plot(self, data_array: xr.DataArray, field_name: str,
                           file_index: int, plot_type: str, figure, time_level,
                           level=None) -> tuple:
        """Prepare the data array and coordinates for plotting."""
        if data_array is None:
            self.logger.error(f"No data array provided for field {field_name}")
            return None

        dim1_name, dim2_name = 'longitude', 'latitude'
        # dim1_name, dim2_name = self.config_manager.get_dim_names(plot_type)
        data2d = None

        # Apply slicing and processing based on plot type
        if 'yz' in plot_type:
            data2d = self._extract_yz_data(data_array, time_lev=time_level)
        elif 'xt' in plot_type:
            data2d = self._extract_xt_data(data_array, time_lev=time_level)
        elif 'tx' in plot_type:
            data2d = self._extract_tx_data(data_array, level=level, time_lev=time_level)
        elif 'xy' in plot_type or 'polar' in plot_type:
            data2d = self._extract_xy_data(data_array, level=level, time_lev=time_level)
        else:
            return None

        if data2d is None:
            self.logger.error(
                f"Failed to prepare 2D data for field {field_name}, plot type {plot_type}")
            return None

        x_values = None
        y_values = None
        if 'xt' in plot_type or 'tx' in plot_type:
            # For time-series or Hovmoller plots, coordinates are handled differently
            # The plotter functions for these types will need to extract them from data2d
            pass
        else:
            x_values = data_array.coords['lon'].values
            y_values = data_array.coords['lat'].values    
            self._set_grib_extents(x_values, y_values)
        # Return the prepared data and coordinates in the expected tuple format
        return data2d, x_values, y_values, field_name, plot_type, file_index, figure
    
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


    def _calculate_diff(self, name1, name2, ax_opts):
        """ Helper method for get_diff_data """
        d1 = self._get_data(name1, ax_opts, 0)
        d2 = self._get_data(name2, ax_opts, 1)
        d1 = apply_conversion(self.config_manager, d1, name1).squeeze()
        d2 = apply_conversion(self.config_manager, d2, name2).squeeze()
        return d1 - d2

    def _get_data(self, field_name, ax_opts, pid):
        d = self.config_manager.readers[0].get_field(field_name, self.config_manager.findex)
        return self._extract_xy_data(d, field_name, time_lev=ax_opts['time_lev'])

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

    def _process_coordinates(self, data2d, dim1, dim2, field_name, plot_type, file_index, figure, ax):
        """
        Process coordinates for GRIB plots, handling NaN values in coordinates.
        """
        xr = np.array(self.lon[0, :])
        yr = np.array(self.lat[:, 0])        
        latN = max(yr[:])
        latS = min(yr[:])
        lonW = min(xr[:])
        lonE = max(xr[:])
        self.config_manager.ax_opts['extent'] = [lonW, lonE, latS, latN]
        self.config_manager.ax_opts['central_lon'] = np.mean([lonW, lonE])
        self.config_manager.ax_opts['central_lat'] = np.mean([latS, latN])
        return data2d, xr, yr, field_name, plot_type, file_index, figure, ax

    def _apply_vertical_level_selection(self, data2d, field_name, level):
        """
        Apply vertical level selection for GRIB data.
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
        GRIB has more complex time handling with optional time averaging.
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

    def _extract_xy_data(self, data_array, level, time_lev):
        """ Extract XY slice from N-dim data field"""
        if data_array is None:
            return None

        data2d = data_array.squeeze()
        tc_dim = self.config_manager.get_model_dim_name('tc')
        zc_dim = self.config_manager.get_model_dim_name('zc')
        
        if zc_dim in data2d.dims:
            data2d = data2d.isel({zc_dim: level})
        if tc_dim in data2d.dims:
            num_times = data_array[tc_dim].size
            if self.ax_opts['tave'] and num_times > 1:
                self.logger.debug(f"Averaging over {num_times} time levels.")
                data2d = apply_mean(self.config_manager, data2d, level)
                return apply_conversion(self.config_manager, data2d, data_array.name)
            else:
                data2d = data2d.isel({tc_dim: time_lev})

        return apply_conversion(self.config_manager, data2d, data_array.name)

