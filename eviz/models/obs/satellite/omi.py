import logging
import sys
import warnings
from typing import Any
import numpy as np
import xarray as xr
from dataclasses import dataclass, field
import eviz.lib.autoviz.utils as pu
from eviz.lib.autoviz.figure import Figure
from eviz.models.root import Root

warnings.filterwarnings("ignore")


@dataclass
class Omi(Root):
    """ Define OMI satellite data and functions.
    """
    source_data: Any = None
    _ds_attrs: dict = field(default_factory=dict)
    maps_params: dict = field(default_factory=dict)
    frame_params: Any = None

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
    
    def add_data_source(self, *args, **kwargs):
        # Implement as needed, or just pass if not used
        pass

    def get_data_source(self, *args, **kwargs):
        # Implement as needed, or just return None if not used
        return None

    def load_data_sources(self, *args, **kwargs):
        # Implement as needed, or just pass if not used
        pass

    def _single_plots(self, plotter):
        """Generate single plots for each source and field according to configuration."""
        self.logger.info("Generating single plots")

        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for single plotting.")
            return

        # Iterate through map_params to generate plots
        for idx, params in self.config_manager.map_params.items():
            field_name = params.get('field')
            if not field_name:
                continue

            filename = params.get('filename')
            data_source = self.config_manager.pipeline.get_data_source(filename)

            if not data_source or not hasattr(data_source,
                                              'dataset') or data_source.dataset is None:
                continue

            # dataset without the long dataset structures, e.g. reduce
            #    HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/ColumnAmountO3
            # into  
            #    ColumnAmountO3
            new_names = {name: name.split('/')[-1] for name in data_source.dataset.data_vars}
            ds_short = data_source.dataset.rename(new_names)

            if field_name not in ds_short:
                continue


            self.config_manager.findex = idx  
            self.config_manager.pindex = idx
            self.config_manager.axindex = 0


            plot_types = params.get('to_plot', ['xy'])
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
            for plot_type in plot_types:
                self._process_plot(ds_short, field_name, idx, plot_type, plotter)

        if self.config_manager.make_gif:
            pu.create_gif(self.config_manager.config)

    def _process_plot(self, ds_short: xr.Dataset, field_name: str, file_index: int,
                      plot_type: str, plotter):
        """Process a single plot type for a given field."""
        self.logger.info(f"Plotting {field_name}, {plot_type} plot")
        figure = Figure.create_eviz_figure(self.config_manager, plot_type)
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        self._process_obs_plot(ds_short, field_name, file_index, plot_type, figure, plotter)

    def _process_obs_plot(self, ds_short: xr.Dataset, field_name: str,
                            file_index: int, plot_type: str, figure,
                            plotter):
        """Process non-xy and non-polar plot types."""
        self.config_manager.level = None
        time_level_config = self.config_manager.ax_opts.get('time_lev', 0)
        tc_dim = self.config_manager.get_model_dim_name('tc') or 'time'

        if tc_dim in ds_short.dims:
            num_times = ds_short[tc_dim].size
            # TODO: Handle yx_plot Gifs
            time_levels = range(num_times) if time_level_config == 'all' else [
                time_level_config]
        else:
            time_levels = [0]

        ax = figure.get_axes()
        field_to_plot = self._get_field_to_plot(ax, ds_short, field_name, file_index,
                                                plot_type, figure,
                                                time_level=time_level_config)
        if field_to_plot:
            plotter.single_plots(self.config_manager, field_to_plot=field_to_plot)
            pu.print_map(self.config_manager, plot_type, self.config_manager.findex,
                         figure)

    def _get_field_to_plot(self, ax, ds_short: xr.Dataset, field_name: str,
                           file_index: int, plot_type: str, figure, time_level=None,
                           level=None) -> tuple:
        ax = figure.get_axes()
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        data2d, lats, lons = extract_field_with_coords(ds_short, field_name)

        self.config_manager.ax_opts['extent'] = [-180, 180, -90, 90]
        self.config_manager.ax_opts['central_lon'] = np.mean(self.config_manager.ax_opts['extent'][:2])
        self.config_manager.ax_opts['central_lat'] = np.mean(self.config_manager.ax_opts['extent'][2:])

        if 'xt' in plot_type or 'tx' in plot_type:
            return data2d, None, None, field_name, plot_type, file_index, figure, ax
        return data2d, lons, lats, field_name, plot_type, file_index, figure, ax
        

def extract_field_with_coords(ds, field_name, 
                              lat_bounds=(-90, 90), lon_bounds=(-180, 180)) -> tuple:
    """
    Extracts a field from an xarray.Dataset, reconstructs lat/lon coordinates assuming regular global grid,
    and masks invalid values using the _FillValue attribute.

    Parameters:
        ds (xarray.Dataset): The dataset.
        field_name (str): The name of the variable to extract (e.g., 'ColumnAmountO3').
        lat_bounds (tuple): Latitude bounds (min, max), default (-90, 90).
        lon_bounds (tuple): Longitude bounds (min, max), default (-180, 180).

    Returns:
        xarray.DataArray: Cleaned data array with lat/lon coordinates and invalid values masked.
        lats (numpy.ndarray): Latitude coordinates.
        lons (numpy.ndarray): Longitude coordinates.
    """
    if field_name not in ds:
        raise ValueError(f"{field_name} not found in dataset.")

    da = ds[field_name]

    n_lat, n_lon = da.shape
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    # Construct 1D coordinate arrays centered in each grid cell
    lat_step = (lat_max - lat_min) / n_lat
    lon_step = (lon_max - lon_min) / n_lon
    lats = np.linspace(lat_max - lat_step / 2, lat_min + lat_step / 2, n_lat)
    lons = np.linspace(lon_min + lon_step / 2, lon_max - lon_step / 2, n_lon)

    da_new = xr.DataArray(
        data=da.values,
        dims=('lat', 'lon'),
        coords={'lat': lats, 'lon': lons},
        attrs=da.attrs
    )

    fill_value = da.attrs.get('_FillValue', None)
    if fill_value is not None:
        da_new = da_new.where(da_new != fill_value)

    return da_new, lats, lons
