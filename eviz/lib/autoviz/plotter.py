from dataclasses import dataclass
import cftime
import logging
import warnings
import matplotlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter, FixedLocator, FuncFormatter
from matplotlib.ticker import NullFormatter
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import networkx as nx
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from sklearn.metrics import mean_squared_error
import eviz.lib.autoviz.utils as pu
import eviz.lib.utils as u
from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.data.pipeline.reader import get_data_coords

warnings.filterwarnings("ignore")

logger = logging.getLogger('plotter')


def _simple_graph_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    graph, dim1, dim2, field_name, plot_type = data_to_plot
    if not graph:
        return
    node_attrs = config.reader.get_node_attributes(graph)
    print("\nNode attributes:")
    for attr, values in node_attrs.items():
        print(f"  - {attr}: {len(values)} values")

    # Get edge attributes
    edge_attrs = config.reader.get_edge_attributes(graph)
    print("\nEdge attributes:")
    for attr, values in edge_attrs.items():
        print(f"  - {attr}: {len(values)} values")

    print("\nNodes:")
    for node, attrs in graph.nodes(data=True):
        print(f"  - Node {node}: {attrs}")

    print("\nEdges:")
    for source, target, attrs in graph.edges(data=True):
        print(f"  - Edge {source}-{target}: {attrs}")

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_labels(graph, pos, labels=nx.get_node_attributes(graph, 'name'))

    edge_weights = [graph[p][q]['weight'] * 2 for p, q in graph.edges()]
    nx.draw_networkx_edges(graph, pos, width=edge_weights)

    edge_labels = nx.get_edge_attributes(graph, 'relationship')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Company Collaboration Network")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("graph_visualization.png")
    plt.show()


def _simple_xy_plot_regional(config: ConfigManager, data_to_plot: tuple, level=0):
    """Helper function for basic_plot() """
    data2d, dim1, dim2, field_name, plot_type = data_to_plot
    pid = config.app_data['inputs'][0]['to_plot'][field_name]
    if data2d is None:
        return
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if isinstance(dim1, np.ndarray) and isinstance(dim2, np.ndarray):
        cf = ax.contourf(dim1, dim2, data2d, cmap=config.cmap)
    elif isinstance(dim1, xr.DataArray) and isinstance(dim2, xr.DataArray):
        cf = ax.contourf(dim1.values, dim2.values, data2d, cmap=config.cmap)
    else:
        raise TypeError(
            'dim1 and dim2 must be either numpy.ndarrays or xarray.DataArrays.')

    cbar = fig.colorbar(cf, ax=ax,
                        orientation='vertical',
                        pad=0.05,
                        fraction=0.05)
    source_name = config.source_names[config.ds_index]
    dvars = config.readers[source_name].datasets[config.findex]['vars'][field_name]
    dim1_name, dim2_name = config.get_dim_names(plot_type)

    if pid == 'xy':
        ax.set_title(config.meta_attrs['field_name'][source_name])
        ax.set_xlabel(dim1_name)
        ax.set_ylabel(dim2_name)
        if 'units' in dvars.attrs:
            cbar.set_label(dvars.attrs['units'])
    u.squeeze_fig_aspect(fig)
    plt.show()


def _simple_yz_plot_regional(config: ConfigManager, data_to_plot: tuple):
    """Helper function for basic_plot() """

    def _prof_plot(ax, data2d, ax_dims):
        if ax_dims[0] == 'zc':
            y0 = data2d.coords[config.meta_coords['yc'][config.model_name](ax_dims[1])][
                0].values
            y1 = data2d.coords[config.meta_coords['zc'][config.model_name](ax_dims[1])][
                -1].values
            ax.plot(data2d, data2d.coords[config.meta_coords['yc'][config.model_name]])
            ax.set_ylim(y0, y1)
        elif ax_dims[0] == 'yc':
            dim_names = config.meta_coords['yc'][config.model_name](ax_dims[1]).split(',')
            for i in dim_names:
                if i in data2d.dims:
                    gooddim = i
            ax.plot(data2d, data2d.coords[gooddim].values)


def _simple_sc_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a simple scatter plot """
    data2d, dim1, dim2, field_name, plot_type = data_to_plot
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    scat = ax.scatter(dim1, dim2, c=data2d,
                      cmap='coolwarm', s=5,
                      transform=ccrs.PlateCarree())
    cbar = fig.colorbar(scat, ax=ax, shrink=0.5)
    cbar.set_label("ppb")
    ax.stock_img()
    ax.coastlines()
    ax.gridlines()
    ax.set_title(f'{field_name}')


def _simple_xy_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a simple xy (lat-lon) plot """
    source_name = config.source_names[config.ds_index]
    if source_name in ['lis', 'wrf']:
        _simple_xy_plot_regional(config, data_to_plot)
        return

    def shift_columns(arr):
        m, n = arr.shape
        mid = math.ceil(n / 2)
        shifted_arr = np.zeros((m, n), dtype=arr.dtype)
        shifted_arr[:, :mid] = arr[:, mid:]
        shifted_arr[:, mid:] = arr[:, :mid]
        return shifted_arr

    data2d, dim1, dim2, field_name, plot_type = data_to_plot
    dmin = data2d.min(skipna=True).values
    dmax = data2d.max(skipna=True).values
    if dmin < 1:  # TODO: This is hackish, please fix
        levels = np.linspace(dmin, dmax, 10)
    else:
        levels = np.around(np.linspace(dmin, dmax, 10), decimals=1)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if data2d is None:
        return
    if dim1 is None or dim2 is None:
        print(
            f"ERROR: dim1 or dim2 is None for field {field_name}, plot type {plot_type}")
        return
    cf = ax.contourf(dim1.values, dim2.values, shift_columns(data2d), cmap=config.cmap)
    co = ax.contour(dim1.values, dim2.values, shift_columns(data2d), levels,
                    linewidths=(1,), origin='lower')
    ax.clabel(co, fmt='%2.1f', colors='w', fontsize=8)
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.05, fraction=0.05)

    if 'field_name' in config.meta_attrs['field_name']:
        t_label = config.meta_attrs['field_name'][config.source_names[config.findex]]
    else:
        t_label = 'name'

    if config.source_names[config.ds_index] in ['lis', 'wrf']:
        dim1_name = config.meta_coords['xc'][config.source_names[config.ds_index]]
        dim2_name = config.meta_coords['yc'][config.source_names[config.ds_index]]
    else:
        try:
            dim1_name = dim1.attrs[t_label]
            dim2_name = dim2.attrs[t_label]
            ax.set_title(data2d.attrs[t_label])
        except KeyError:
            dim1_name = dim1.name
            dim2_name = dim2.name
            ax.set_title(field_name)

    ax.set_xlabel(dim1_name)
    ax.set_ylabel(dim2_name)
    if 'units' in data2d.attrs:
        cbar.set_label(data2d.attrs['units'])


def _simple_yz_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a simple yz (zonal-mean) plot """
    source_name = config.source_names[0]
    if source_name in ['lis', 'wrf']:
        _simple_yz_plot_regional(config, data_to_plot)
        return

    data2d, dim1, dim2, field_name, plot_type = data_to_plot
    dmin = data2d.min(skipna=True).values
    dmax = data2d.max(skipna=True).values
    if dmin < 1:  # TODO: This is hackish, please fix
        levels = np.linspace(dmin, dmax, 10)
    else:
        levels = np.around(np.linspace(dmin, dmax, 10), decimals=1)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if data2d is None:
        return
    if dim1 is None or dim2 is None:
        print(
            f"ERROR: dim1 or dim2 is None for field {field_name}, plot type {plot_type}")
        return
    cf = ax.contourf(dim1.values, dim2.values, data2d, cmap=config.cmap)
    co = ax.contour(dim1.values, dim2.values, data2d, levels, linewidths=(1,),
                    origin='lower')
    ax.clabel(co, fmt='%2.1f', colors='w', fontsize=8)
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.05, fraction=0.05)
    if 'field_name' in config.meta_attrs['field_name']:
        t_label = config.meta_attrs['field_name'][config.source_names[config.ds_index]]
    else:
        t_label = 'name'
    if config.source_names[config.ds_index] in ['lis', 'wrf']:
        dim1_name = config.meta_coords['xc'][config.source_names[config.ds_index]]
        dim2_name = config.meta_coords['yc'][config.source_names[config.ds_index]]
    else:
        try:
            dim1_name = dim1.attrs[t_label]
            dim2_name = dim2.attrs[t_label]
            ax.set_title(data2d.attrs[t_label])
        except KeyError:
            dim1_name = dim1.name
            dim2_name = dim2.name
            ax.set_title(field_name)

    ax.set_xlabel(dim1_name)
    ax.set_ylabel(dim2_name)
    if 'units' in data2d.attrs:
        cbar.set_label(data2d.attrs['units'])


def _single_scat_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a single scatter using SPECS data
        config (Config) : configuration used to specify data sources
        data_to_plot (dict) : dict with plotted data and specs

    Parameters:
        config (Config) : configuration used to specify data sources
        data_to_plot (tuple) : dict with plotted data and specs
    """
    data2d, x, y, field_name, plot_type, findex, fig = data_to_plot
    ax_opts = config.ax_opts

    fig.set_axes()
    ax = fig.get_axes()
    if isinstance(ax, (list, tuple, np.ndarray)):
        ax = ax[0]

    with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
        is_cartopy_axis = False
        try:
            is_cartopy_axis = isinstance(ax, GeoAxes)
        except ImportError:
            pass
        
        if fig.use_cartopy and is_cartopy_axis:
            # ax.stock_img()
            scat = ax.scatter(x, y, c=data2d, cmap=ax_opts['use_cmap'], s=5,
                              transform=ccrs.PlateCarree())
        else:
            scat = ax.scatter(x, y, c=data2d, cmap=ax_opts['use_cmap'], s=2)

        if scat is None:
            logger.error("Scatter plot failed")
            return 
        else:
            _set_cartopy_ticks_alt(ax, ax_opts['extent'])
            _set_colorbar(config, scat, fig, ax, ax_opts, findex, field_name, data2d)

        ax.set_title(f'{field_name}')


def _single_xy_plot(config: ConfigManager, data_to_plot: tuple, level: int) -> None:
    """ Create a single xy (lat-lon) plot using SPECS data

    Parameters:
        config (ConfigManager) : configuration used to specify data sources
        data_to_plot (tuple) : dict with plotted data and specs
        level (int) : vertical level
    """
    data2d, x, y, field_name, plot_type, findex, fig = data_to_plot
    
    ax_opts = config.ax_opts
    if not config.compare and not config.compare_diff:
        fig.set_axes()
    
    ax_temp = fig.get_axes()
    axes_shape = fig.subplots

    if axes_shape == (3, 1):
        if ax_opts['is_diff_field']:
            ax = ax_temp[2]
        else:
            ax = ax_temp[config.axindex]
    elif axes_shape == (2, 2):
        if ax_opts['is_diff_field']:
            ax = ax_temp[2]
            if config.ax_opts['add_extra_field_type']:
                ax = ax_temp[3]
        else:
            ax = ax_temp[config.axindex]
    elif axes_shape == (1, 2) or axes_shape == (1, 3):
        if isinstance(ax_temp, list):
            ax = ax_temp[config.axindex]  # Use the correct axis based on axindex
        else:
            ax = ax_temp
    else:
        ax = ax_temp[0]
    

    if data2d is None:
        return

    ax_opts = fig.update_ax_opts(field_name, ax, 'xy', level=level)
    fig.plot_text(field_name, ax, 'xy', level=level, data=data2d)

    # Handle single axes or list of axes
    if isinstance(ax, list):
        for single_ax in ax:
            _plot_xy_data(config, single_ax, data2d, x, y, field_name, fig, ax_opts,
                          level,
                          plot_type, findex)
    else:
        _plot_xy_data(config, ax, data2d, x, y, field_name, fig, ax_opts, level,
                      plot_type, findex)


def _plot_xy_data(config, ax, data2d, x, y, field_name, fig, ax_opts, level,
                  plot_type, findex):
    """Helper function to plot XY data on a single axes."""
    if 'fill_value' in config.spec_data[field_name]['xyplot']:
        fill_value = config.spec_data[field_name]['xyplot']['fill_value']
        data2d = data2d.where(data2d != fill_value, np.nan)

    # Check if we're using Cartopy and if the axis is a GeoAxes
    is_cartopy_axis = False
    try:
        is_cartopy_axis = isinstance(ax, GeoAxes)
    except ImportError:
        pass

    data_transform = ccrs.PlateCarree()
    if fig.use_cartopy and is_cartopy_axis:
        cfilled = _filled_contours(config, field_name, ax, x, y, data2d, transform=data_transform)
        if 'extent' in ax_opts:
            _set_cartopy_ticks(ax, ax_opts['extent'])
        else:
            _set_cartopy_ticks(ax, [-180, 180, -90, 90])
    else:
        cfilled = _filled_contours(config, field_name, ax, x, y, data2d)

    if cfilled is None:
        _set_const_colorbar(cfilled, fig, ax)
    else:
        _set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d)
        if 	ax_opts.get('line_contours', False):
            if fig.use_cartopy and is_cartopy_axis:
                _line_contours(fig, ax, ax_opts, x, y, data2d, transform=data_transform)
            else:
                _line_contours(fig, ax, ax_opts, x, y, data2d)  

    if config.compare_diff:
        name = field_name
        if 'name' in config.spec_data[field_name]:
            name = config.spec_data[field_name]['name']

        level_text = None
        if config.ax_opts.get('zave', False):
            level_text = ' (Column Mean)'
        elif config.ax_opts.get('zsum', False):
            level_text = ' (Total Column)'
        else:
            if str(level) == '0':
                level_text = ''
            else:
                if level is not None:
                    if level > 10000:
                        level_text = '@ ' + str(level) + ' Pa'
                    else:
                        level_text = '@ ' + str(level) + ' mb'

        if level_text:
            name = name + level_text

        fig.suptitle_eviz(name, 
                          fontweight='bold',
                          fontstyle='italic',
                          fontsize=pu.image_font_size(fig.subplots))        
        
    elif config.compare:

        fig.suptitle_eviz(text=config.map_params[findex].get('field', 'No name'), 
                          fontweight='bold',
                          fontstyle='italic',
                          fontsize=pu.image_font_size(fig.subplots))        

        if config.add_logo:
            pu.add_logo_ax(fig, desired_width_ratio=0.05)

    # This works well in xy (i.e. lat-lon) plots
    # fig.squeeze_fig_aspect(fig.fig)


def _single_yz_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a single yz (zonal-mean) plot using SPECS data
        Note:
            YZ fields are treated like XY fields where Y->X and Z->Y

    Parameters:
        config (ConfigManager) : configuration used to specify data sources
        data_to_plot (tuple) : dict with plotted data and specs
    """
    data2d, x, y, field_name, plot_type, findex, fig = data_to_plot
    
    ax_opts = config.ax_opts
    # Test applying rcparams to the figure via specification in specs
    # fig.apply_rc_params()

    if not config.compare and not config.compare_diff and not config.overlay:
        fig.set_axes()
    
    ax_temp = fig.get_axes()
    axes_shape = fig.subplots

    if axes_shape == (3, 1):
        if ax_opts['is_diff_field']:
            ax = ax_temp[2]
        else:
            ax = ax_temp[config.axindex]
    elif axes_shape == (2, 2):
        if ax_opts['is_diff_field']:
            ax = ax_temp[2]
            if config.ax_opts['add_extra_field_type']:
                ax = ax_temp[3]
        else:
            ax = ax_temp[config.axindex]
    elif axes_shape == (1, 2) or axes_shape == (1, 3):
        if isinstance(ax_temp, list):
            ax = ax_temp[config.axindex]  # Use the correct axis based on axindex
        else:
            ax = ax_temp
    else:
        ax = ax_temp[0]

    if data2d is None:
        return

    ax_opts = fig.update_ax_opts(field_name, ax, 'yz')
    fig.plot_text(field_name, ax, 'yz', level=None, data=data2d)

    # Determine the vertical coordinate and its units
    zc = config.get_model_dim_name('zc')
    vertical_coord = None
    vertical_units = 'n.a.'

    if isinstance(zc, dict):
        for dim_name in data2d.coords:
            if 'lev' in dim_name or 'z' in dim_name or 'height' in dim_name:
                vertical_coord = data2d.coords[dim_name]
                break
    elif isinstance(zc, list):
        for z in zc:
            if z in data2d.coords:
                vertical_coord = data2d.coords[z]
                break
    else:
        if zc and zc in data2d.coords:
            vertical_coord = data2d.coords[zc]
        else:
            # Try common vertical dimension names
            for dim_name in ['lev', 'level', 'z', 'height', 'altitude', 'pressure']:
                if dim_name in data2d.coords:
                    vertical_coord = data2d.coords[dim_name]
                    break

    if vertical_coord is not None:
        vertical_units = vertical_coord.attrs.get('units', 'n.a.')
    else:
        if len(data2d.shape) >= 2:
            # Assume first dimension is vertical in a YZ plot
            vertical_coord = np.arange(data2d.shape[0])

    if isinstance(ax, list):
        for single_ax in ax:
            _plot_yz_data(config, single_ax, data2d, x, y, field_name, fig, ax_opts,
                          vertical_units,
                          plot_type, findex)
    else:
        _plot_yz_data(config, ax, data2d, x, y, field_name, fig, ax_opts, vertical_units,
                      plot_type, findex)

    # reset rc params to default
    # matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def _plot_yz_data(config, ax, data2d, x, y, field_name, fig, ax_opts, vertical_units,
                  plot_type, findex):
    """Helper function to plot YZ data on a single axes."""
    source_name = config.source_names[config.ds_index]
    prof_dim = None
    if ax_opts['profile_dim']:
        logger.debug(f"Creating profile plot for {field_name}")
        prof_dim = ax_opts['profile_dim']
        dep_var = None
        if prof_dim == 'yc':
            dep_var = 'zc'
        if prof_dim == 'zc':
            dep_var = 'yc'
        prof_dim = ax_opts['profile_dim']
        data2d = data2d.mean(dim=config.get_model_dim_name(prof_dim))
        _single_prof_plot(config, data2d, fig, ax, ax_opts, (prof_dim, dep_var))

        # TODO: Fix units issue (missing sometimes)
        if 'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        else:
            try:
                units = data2d.attrs['units']
                if units == '':
                    units = "n.a."
            except Exception as e:
                logger.error(f"{e}: Please specify {field_name} units in specs file")
                units = "n.a."
        if dep_var == 'zc':
            ax.set_xlabel(units)
            ax.set_ylabel("Pressure (" + vertical_units + ")",
                          size=pu.axes_label_font_size(fig.subplots))

    else:
        cfilled = _filled_contours(config, field_name, ax, x, y, data2d)

        ylabels = ax.get_yticklabels()
        for label in ylabels:
            label.set_fontsize(pu.axis_tick_font_size(fig.subplots))

        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_fontsize(pu.axis_tick_font_size(fig.subplots))

        _set_ax_ranges(config, field_name, fig, ax, ax_opts, y, vertical_units)

        _line_contours(fig, ax, ax_opts, x, y, data2d)

        _set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d)

    # The following is only supported for GEOS datasets:
    # TODO: move to 'model' layer!?
    # print(config.use_trop_height)
    # if config.use_trop_height and not prof_dim:
    #     proc = DataProcessor(config)
    #     tropp = proc.process_data_source(data2d)
    #     # This should be the only call needed here:
    #     if proc.trop_ok:
    #         ax.plot(x, tropp, linewidth=2, color="k", linestyle="--")
    #     # The following is temporary, while the TODO above is not done.
    #     config.use_trop_height = None

    if config.compare_diff and config.ax_opts['is_diff_field']:
        try:
            if 'name' in config.spec_data[field_name]:
                name = config.spec_data[field_name]['name']
            else:
                reader = None
                if source_name in config.readers:
                    if isinstance(config.readers[source_name], dict):
                        readers_dict = config.readers[source_name]
                        if 'NetCDF' in readers_dict:
                            reader = readers_dict['NetCDF']
                        elif readers_dict:
                            reader = next(iter(readers_dict.values()))
                    else:
                        # direct access
                        reader = config.readers[source_name]

                if reader and hasattr(reader, 'datasets'):
                    if findex in reader.datasets and 'vars' in reader.datasets[findex]:
                        var_attrs = reader.datasets[findex]['vars'][field_name].attrs
                        if 'long_name' in var_attrs:
                            name = var_attrs['long_name']
                        else:
                            name = field_name
                    else:
                        name = field_name
                else:
                    if hasattr(data2d, 'attrs') and 'long_name' in data2d.attrs:
                        name = data2d.attrs['long_name']
                    else:
                        name = field_name
        except Exception as e:
            logger.warning(f"Error getting field name: {e}")
            name = field_name

        fig.suptitle_eviz(name, 
                          fontweight='bold',
                          fontstyle='italic',
                          fontsize=pu.image_font_size(fig.subplots))        
        
    elif config.compare:

        fig.suptitle_eviz(text=config.map_params[findex].get('field', 'No name'), 
                          fontweight='bold',
                          fontstyle='italic',
                          fontsize=pu.image_font_size(fig.subplots))        

        # fig.text(0.5, 0.98, name,
        #         fontweight='bold',
        #         fontstyle='italic',
        #         fontsize=pu.image_font_size(fig.subplots),
        #         ha='center',
        #         va='top',
        #         transform=fig.transFigure)

        if config.add_logo:
            pu.add_logo_ax(fig, desired_width_ratio=0.05)


def _set_ax_ranges(config, field_name, fig, ax, ax_opts, y, units):
    """ Create a sensible number of vertical levels """
    y_ranges = np.array([1000, 700, 500, 300, 200, 100])
    if units == "Pa":
        y_ranges = y_ranges * 100
        if y.min() <= 1000.0:
            y_ranges = np.append(y_ranges, np.array([70, 50, 30, 20, 10]) * 100)
        if y.min() <= 20.:
            y_ranges = np.append(y_ranges,
                                 np.array([7, 5, 3, 2, 1, .7, .5, .3, .2, .1]) * 100)
        if y_ranges[-1] != y.min():
            y_ranges = np.append(y_ranges, y.min())
    else:  # TODO hPa (mb) only?, do we need meters?
        if y.min() <= 10.0:
            y_ranges = np.append(y_ranges, np.array([70, 50, 30, 20, 10]))
        if y.min() <= 0.2:
            y_ranges = np.append(y_ranges, np.array([7, 5, 3, 2, 1, .7, .5, .3, .2, .1]))
        if y_ranges[-1] != y.min():
            y_ranges = np.append(y_ranges, y.min())

    lo_z, hi_z = y_ranges.max(), y_ranges.min()
    if 'zrange' in config.spec_data[field_name]['yzplot']:
        if not config.spec_data[field_name]['yzplot']['zrange']:
            pass  # if nothing specified (it happens)
        else:
            lo_z = config.spec_data[field_name]['yzplot']['zrange'][0]
            hi_z = config.spec_data[field_name]['yzplot']['zrange'][1]
            if hi_z >= lo_z:
                logger.error(
                    f"Upper level value ({hi_z}) must be less than low level value ({lo_z})")
                return None

    # These can be defined for global or regional models. We let the respective model
    # override the extents. For gridded, we assume they are global extents.
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
    ax.tick_params(width=3, length=6)
    # The vertical coordinate can have different units. For gridded, we assume pressure
    # and, again, let specialized models override the definition.
    # Assume surface is the first level
    ax.set_ylim(lo_z, hi_z)
    # scale is by default "log"
    ax.set_yscale(ax_opts['zscale'])
    ax.yaxis.set_minor_formatter(NullFormatter())
    if 'linear' in ax_opts['zscale']:
        # y_ranges = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0]
        if units == 'Pa':
            y_ranges = y_ranges * 100
    # TODO: This may not be needed: please test!
    # ax.set_yticks(y_ranges)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%3.1f'))
    ax.set_ylabel("Pressure (" + units + ")", size=pu.axes_label_font_size(fig.subplots))
    if ax_opts['add_grid']:
        ax.grid()


def _single_polar_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a single polar plot using SPECS data

    Parameters:
        config (Config) : configuration used to specify data sources
        data_to_plot (tuple) : dict with plotted data and specs
    """
    source_name = config.source_names[config.ds_index]
    data2d, x, y, field_name, plot_type, findex, fig = data_to_plot
    if data2d is None:
        return
    ax_opts = config.ax_opts

    fig.set_axes()
    ax_temp = fig.get_axes()

    axes_shape = fig.subplots
    if axes_shape == (3, 1):
        if ax_opts['is_diff_field']:
            ax = ax_temp[2]
        else:
            ax = ax_temp[config.axindex]
    elif axes_shape == (1, 2):
        ax = ax_temp
    else:
        ax = ax_temp[0]

    ax_opts = fig.update_ax_opts(field_name, ax, 'polar', level=0)

    if ax_opts['use_pole'] == 'south':
        projection = ccrs.SouthPolarStereo()
        extent_lat = -60  # Southern limit for South Polar plot
    else:
        projection = ccrs.NorthPolarStereo()
        extent_lat = 60  # Northern limit for North Polar plot

    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.set_extent([-180, 180, extent_lat, 90 if ax_opts['use_pole'] == 'north' else -90],
                  ccrs.PlateCarree())
    if ax_opts['boundary']:
        theta = np.linspace(0, 2 * np.pi, 100)
        center = [0.5, 0.5]
        radius = 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    _create_clevs(field_name, ax_opts, data2d)
    clevs = pu.formatted_contours(ax_opts['clevs'])

    extend_value = "both"
    if ax_opts['clevs'][0] == 0:
        extend_value = "max"
    norm = colors.BoundaryNorm(ax_opts['clevs'], ncolors=256, clip=False)

    trans = ccrs.PlateCarree()

    try:
        # contourf
        pcm = ax.contourf(x, y, data2d,
                          cmap=ax_opts['use_cmap'],
                          levels=clevs,
                          transform=trans,
                          extend=extend_value,
                          norm=norm)

        plot_success = True
    except Exception as e:
        logger.warning(f"contourf failed: {e}")
        try:
            pcm = ax.imshow(
                data2d,
                extent=[-180, 180, -90 if ax_opts['use_pole'] == 'south' else 0,
                        0 if ax_opts['use_pole'] == 'south' else 90],
                transform=ccrs.PlateCarree(),
                cmap=ax_opts['use_cmap'],
                aspect='auto'
            )
            plot_success = True
        except Exception as e2:
            logger.error(f"All plotting methods failed: {e2}")
            plot_success = False

    if not plot_success:
        logger.error("Failed to create polar plot")
        return

    if ax_opts['line_contours']:
        clines = ax.contour(x, y, data2d,
                            levels=ax_opts['clevs'], colors="black",
                            linewidths=0.5, alpha=0.5, linestyles='solid',
                            transform=trans)
        ax.clabel(clines, inline=1, fontsize=8,
                  inline_spacing=10, colors="black",
                  rightside_up=True,  # fmt=contour_format,
                  use_clabeltext=True)
    else:
        _ = ax.contour(x, y, data2d, linewidths=0.0)

    ax.add_feature(cfeature.BORDERS, zorder=10, linewidth=0.5, edgecolor='grey')
    ax.add_feature(cfeature.LAKES, alpha=0.9)
    ax.add_feature(cfeature.LAND, color='silver', zorder=1, facecolor=0.9)
    ax.add_feature(cfeature.COASTLINE, zorder=10, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=0)

    cbar = plt.colorbar(pcm, ax=ax, shrink=0.5, pad=0.05)
    if 'units' in config.spec_data[field_name]:
        units = config.spec_data[field_name]['units']
    else:
        try:
            units = data2d.units
        except Exception as e:
            logger.error(f"{e}: Please specify {field_name} units in specs file")
            units = "n.a."
    if units == '1':
        units = '%'
    if ax_opts['clabel'] is None:
        cbar_label = units
    else:
        cbar_label = ax_opts['clabel']
    cbar.set_label(label=cbar_label, size=12, weight='bold')

    if ax_opts['add_grid']:
        _ = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.75,
                         linestyle='--')

    if 'name' in config.spec_data[field_name]:
        ax.set_title(config.spec_data[field_name]['name'], y=1.03, fontsize=14,
                     weight='bold')
    else:
        ax.set_title(source_name, y=1.03, fontsize=14, weight='bold')


def _single_xt_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a single xt (time-series) plot using SPECS data

    Note:
        XT plots are 2D plots where the variable, X, is averaged over all space,
        is plotted against time, T.

    Parameters:
        config (Config) : configuration used to specify data sources
        data_to_plot (tuple) : dict with plotted data and specs
    """
    data2d, _, _, field_name, plot_type, findex, fig = data_to_plot
    
    ax_opts = config.ax_opts
    if not config.compare and not config.compare_diff:
        fig.set_axes()
    
    ax_temp = fig.get_axes()
    axes_shape = fig.subplots

    if axes_shape == (3, 1):
        if ax_opts['is_diff_field']:
            ax = ax_temp[2]
        else:
            ax = ax_temp[config.axindex]
    elif axes_shape == (2, 2):
        if ax_opts['is_diff_field']:
            ax = ax_temp[2]
            if config.ax_opts['add_extra_field_type']:
                ax = ax_temp[3]
        else:
            ax = ax_temp[config.axindex]
    elif axes_shape == (1, 2) or axes_shape == (1, 3):
        if isinstance(ax_temp, list):
            ax = ax_temp[config.axindex]  # Use the correct axis based on axindex
        else:
            ax = ax_temp
    else:
        ax = ax_temp[0]
    

    if data2d is None:
        return

    ax_opts = fig.update_ax_opts(field_name, ax, 'xt', level=0)
    fig.plot_text(field_name, ax, 'xt', data=data2d)

    _time_series_plot(config, ax, ax_opts, fig, data2d, field_name, findex)

    if config.compare_diff:
        name = field_name
        if 'name' in config.spec_data[field_name]:
            name = config.spec_data[field_name]['name']

        fig.suptitle_eviz(name, 
                          fontweight='bold',
                          fontstyle='italic',
                          fontsize=pu.image_font_size(fig.subplots))        
        
    elif config.compare:

        fig.suptitle_eviz(text=config.map_params[findex].get('field', 'No name'), 
                          fontweight='bold',
                          fontstyle='italic',
                          fontsize=pu.image_font_size(fig.subplots))        

        if config.add_logo:
            pu.add_logo_ax(fig, desired_width_ratio=0.05)


def _time_series_plot(config, ax, ax_opts, fig, data2d, field_name, findex):
    with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
        dmin = data2d.min(skipna=True).values
        dmax = data2d.max(skipna=True).values
        logger.debug(f"dmin: {dmin}, dmax: {dmax}")
        tc_dim = config.get_model_dim_name('tc')
        try:
            if tc_dim and tc_dim in data2d.coords:
                time_coords = data2d.coords[tc_dim].values
            else:
                if len(data2d.dims) > 0:
                    time_coords = data2d[data2d.dims[0]].values
                else:
                    time_coords = np.arange(len(data2d))

            # Handle cftime objects
            if isinstance(time_coords[0], cftime._cftime.DatetimeNoLeap):
                try:
                    # convert to pandas datetime
                    time_coords = pd.to_datetime([str(t) for t in time_coords])
                except Exception as e:
                    logger.warning(f"Error converting cftime to pandas datetime: {e}")
                    time_coords = np.arange(len(data2d))

        except Exception as e:
            logger.warning(f"Error getting time coordinates: {e}")
            time_coords = np.arange(len(data2d))

        t0 = time_coords[0]
        t1 = time_coords[-1]

        window_size = 0
        if 'mean_type' in config.spec_data[field_name]['xtplot']:
            if config.spec_data[field_name]['xtplot']['mean_type'] == 'rolling':
                if 'window_size' in config.spec_data[field_name]['xtplot']:
                    window_size = config.spec_data[field_name]['xtplot']['window_size']

        if window_size > 0 and len(data2d) > 2 * window_size:
            end_idx = max(0, len(time_coords) - window_size - 1)
            ax.plot(time_coords[window_size:end_idx], data2d[window_size:end_idx])
        else:
            ax.plot(time_coords, data2d)
    
        # TODO: Need to fix trend line for windowed series (not showing!)
        if 'add_trend' in config.spec_data[field_name]['xtplot']:
            logger.debug('Adding trend')
            if config.spec_data[field_name]['xtplot']['add_trend']:
                try:
                    if isinstance(t0, (pd.Timestamp, np.datetime64)):
                        time_numeric = (time_coords - t0).astype('timedelta64[D]').astype(
                            float)
                    else:
                        time_numeric = np.arange(len(time_coords))

                    errors = []
                    if 'trend_polyfit' in config.spec_data[field_name]['xtplot']:
                        degree = config.spec_data[field_name]['xtplot']['trend_polyfit']
                    else:
                        for degree in range(1, 6):
                            coeffs = np.polyfit(time_numeric, data2d, degree)
                            y_fit = np.polyval(coeffs, time_numeric)
                            mse = mean_squared_error(data2d, y_fit)
                            errors.append(mse)
                            best_degree = np.argmin(errors) + 1
                        degree = best_degree
                    logger.debug(f' -- polynomial degree: {degree}')
                    coeffs = np.polyfit(time_numeric, data2d, degree)
                    trend_poly = np.polyval(coeffs, time_numeric)
                    ax.plot(time_coords, trend_poly, color="red", linewidth=1)
                except Exception as e:
                    logger.warning(f"Error calculating trend: {e}")

        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
        fig.autofmt_xdate()
        ax.set_xlim(t0, t1)

        try:
            davg = 0.5 * (abs(dmin - dmax))
            ax.set_ylim([dmin - davg, dmax + davg])
        except Exception as e:
            logger.warning(f"Error setting y limits: {e}")

        try:
            source_name = config.source_names[config.ds_index]

            if 'units' in config.spec_data[field_name]:
                units = config.spec_data[field_name]['units']
            else:
                units = getattr(data2d, 'units', None)

                if not units:
                    reader = config.get_primary_reader(source_name)
                    if reader and hasattr(reader, 'datasets'):
                        if findex in reader.datasets and 'vars' in reader.datasets[
                            findex]:
                            field_var = reader.datasets[findex]['vars'].get(field_name)
                            if field_var:
                                units = getattr(field_var, 'units', 'n.a.')
                            else:
                                units = 'n.a.'
                        else:
                            units = 'n.a.'
                    else:
                        units = 'n.a.'
        except Exception as e:
            logger.warning(f"Error getting units for {field_name}: {e}")
            units = 'n.a.'

        ax.set_ylabel(units)

        if ax_opts['add_grid']:
            ax.grid()


def _single_prof_plot(config, data2d, fig, ax, ax_opts, ax_dims) -> None:
    """ Create a single prof (vertical profile) plot using SPECS data"""
    if ax_dims[0] == 'zc':
        y0 = data2d.coords[config.get_model_dim_name(ax_dims[1])][0].values
        y1 = data2d.coords[config.get_model_dim_name(ax_dims[1])][-1].values
        ax.plot(data2d, data2d.coords[config.get_model_dim_name('yc')])
        ax.set_ylim(y0, y1)
    elif ax_dims[0] == 'yc':
        y0 = data2d.coords[config.get_model_dim_name(ax_dims[1])][0].values
        y1 = data2d.coords[config.get_model_dim_name(ax_dims[1])][-1].values
        ax.plot(data2d, data2d.coords[config.get_model_dim_name('zc')])
        ax.set_ylim(y0, y1)

    ax.set_yscale(ax_opts['zscale'])
    ax.yaxis.set_minor_formatter(NullFormatter())
    if 'linear' in ax_opts['zscale']:
        y_ranges = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0]
        ax.set_yticks(y_ranges)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%3.1f'))
    if ax_opts['add_grid']:
        ax.grid()

    ylabels = ax.get_yticklabels()
    for label in ylabels:
        label.set_fontsize(pu.axis_tick_font_size(fig.subplots))

    xlabels = ax.get_xticklabels()
    for label in xlabels:
        label.set_fontsize(pu.axis_tick_font_size(fig.subplots))

    if ax_opts['add_grid']:
        ax.grid()


def _single_tx_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a single tx (Hovmoller) plot using SPECS data

    Note:
        A Hovmoller plot shows zonal and meridional shifts in a given field
        Can also be used to plot time evolution of atmospheric profiles
        that vary spatially.

    Parameters:
        config (Config) : configuration used to specify data sources
        data_to_plot (tuple) : dict with plotted data and specs

    Reference:
        https://unidata.github.io/python-gallery/examples/Hovmoller_Diagram.html
    """
    data2d, _, _, field_name, plot_type, findex, fig = data_to_plot
    if data2d is None:
        return
    ax_opts = config.ax_opts

    with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
        # gs = mgridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 5], hspace=0.1)
        gs = mgridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 6], hspace=0.05)
        ax = list()
        ax.append(
            fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=180)))
        ax.append((fig.add_subplot(gs[1, 0])))
        fig.set_size_inches(12, 10, forward=True)

        ax_opts = fig.update_ax_opts(field_name, ax, 'tx')

        dmin = data2d.min(skipna=True).values
        dmax = data2d.max(skipna=True).values
        logger.debug(f"Field: {field_name}; Min:{dmin}; Max:{dmax}")

        _create_clevs(field_name, ax_opts, data2d)
        extend_value = "both"
        if ax_opts['clevs'][0] == 0:
            extend_value = "max"

        norm = colors.BoundaryNorm(ax_opts['clevs'], ncolors=256, clip=False)

        vtimes = data2d.time.values.astype('datetime64[ms]').astype('O')
        lon_dim = config.get_model_dim_name('xc')

        try:
            if lon_dim:
                lons = get_data_coords(data2d, lon_dim)
            else:
                if 'lon' in data2d.dims:
                    lons = data2d.lon.values
                elif 'longitude' in data2d.dims:
                    lons = data2d.longitude.values
                elif 'x' in data2d.dims:
                    lons = data2d.x.values
                else:
                    lons = np.arange(data2d.shape[1])
        except Exception as e:
            logger.error(f"Error getting longitude coordinates: {e}")
            lons = np.arange(data2d.shape[1])

        try:
            if len(data2d.shape) > 2:
                if data2d.shape[0] == len(vtimes):
                    # data shape is (time, level, lon), we need to average over level
                    if len(data2d.shape) == 3 and data2d.shape[2] == len(lons):
                        # average over the middle dimension (level)
                        data2d_reduced = data2d.mean(axis=1)

                    # data shape is (time, lat, lon), we need to average over lat
                    elif len(data2d.shape) == 3 and data2d.shape[2] == len(lons):
                        # average over the middle dimension (lat)
                        data2d_reduced = data2d.mean(axis=1)

                    else:
                        dim_names = list(data2d.dims)
                        time_dim_idx = None
                        lon_dim_idx = None

                        for i, dim in enumerate(dim_names):
                            if dim in ['time', 't', 'TIME']:
                                time_dim_idx = i
                                break

                        for i, dim in enumerate(dim_names):
                            if dim in ['lon', 'longitude', 'x']:
                                lon_dim_idx = i
                                break

                        if time_dim_idx is not None and lon_dim_idx is not None:
                            dims_to_avg = [i for i in range(len(dim_names))
                                        if i != time_dim_idx and i != lon_dim_idx]

                            data2d_reduced = data2d.copy()
                            for dim_idx in sorted(dims_to_avg, reverse=True):
                                data2d_reduced = data2d_reduced.mean(axis=dim_idx)

                            # transpose if needed to get (time, lon) order
                            if time_dim_idx > lon_dim_idx:
                                data2d_reduced = data2d_reduced.T
                        else:
                            # flatten all dimensions except time
                            data2d_reduced = data2d.reshape(data2d.shape[0], -1).mean(axis=1)
                            lons = np.arange(data2d_reduced.shape[1])
                else:
                    data2d_reduced = data2d
                    if len(vtimes) != data2d.shape[0]:
                        vtimes = np.arange(data2d.shape[0])
                    if len(lons) != data2d.shape[1]:
                        lons = np.arange(data2d.shape[1])
            else:
                data2d_reduced = data2d

                if data2d.shape != (len(vtimes), len(lons)):
                    if data2d.shape == (len(lons), len(vtimes)):
                        data2d_reduced = data2d.T
                    else:
                        vtimes = np.arange(data2d.shape[0])
                        lons = np.arange(data2d.shape[1])
        except Exception as e:
            logger.error(f"Error processing data for Hovmoller plot: {e}")
            data2d_reduced = data2d
            if len(data2d.shape) >= 2:
                vtimes = np.arange(data2d.shape[0])
                lons = np.arange(data2d.shape[1])

        if hasattr(data2d_reduced, 'shape') and len(data2d_reduced.shape) >= 2:
            if data2d_reduced.shape[0] != len(vtimes) or data2d_reduced.shape[1] != len(lons):
                vtimes = np.arange(data2d_reduced.shape[0])
                lons = np.arange(data2d_reduced.shape[1])

        x_tick_labels = [u'0\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}E',
                        u'180\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}W',
                        u'0\N{DEGREE SIGN}E']

        ax[0].set_extent([0, 357.5, 35, 65], ccrs.PlateCarree(central_longitude=180))
        ax[0].set_yticks([40, 60])
        ax[0].set_yticklabels([u'40\N{DEGREE SIGN}N', u'60\N{DEGREE SIGN}N'])
        ax[0].set_xticks([-180, -90, 0, 90, 180])
        ax[0].set_xticklabels(x_tick_labels)
        ax[0].grid(linestyle='dotted', linewidth=2)

        ax[0].add_feature(cfeature.COASTLINE.with_scale('50m'))
        ax[0].add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)
        fig.plot_text(field_name=field_name, ax=ax[0], pid='tx', data=data2d, fontsize=8,
                    loc='left')

        if ax_opts['torder']:
            ax[1].invert_yaxis()  # Reverse the time order

        try:
            cfilled = ax[1].contourf(lons, vtimes, data2d_reduced, ax_opts['clevs'],
                                    norm=norm,
                                    cmap=ax_opts['use_cmap'], extend=extend_value)
        except Exception as e:
            logger.error(f"Error creating contour plot: {e}")
            try:
                logger.info("Falling back to pcolormesh")
                lon_mesh, time_mesh = np.meshgrid(lons, vtimes)
                cfilled = ax[1].pcolormesh(lon_mesh, time_mesh, data2d_reduced,
                                        norm=norm, cmap=ax_opts['use_cmap'])
            except Exception as e2:
                logger.error(f"Error creating pcolormesh plot: {e2}")
                # just show something
                cfilled = ax[1].imshow(data2d_reduced, aspect='auto', origin='lower',
                                    norm=norm, cmap=ax_opts['use_cmap'])

        ax[1].set_xlabel("Longitude")
        ax[1].set_ylabel("Time")
        ax[1].grid(linestyle='dotted', linewidth=0.5)
    
        try:
            if ax_opts['line_contours']:
                _line_contours(fig, ax[1], ax_opts, lons, vtimes, data2d_reduced)
        except Exception as e:
            logger.error(f"Error adding contour lines: {e}")

        cbar = fig.colorbar(cfilled, orientation='horizontal', pad=0.1, aspect=70,
                            extendrect=True)
        
        if hasattr(data2d, 'attrs') and 'units' in data2d.attrs:
            units = data2d.attrs['units']
        elif hasattr(data2d, 'units'):
            units = data2d.units
        cbar.set_label(units)

        if lons[0] <= -179:
            ax[1].set_xticks([-180, -90, 0, 90, 180])
        else:
            ax[1].set_xticks([0, 90, 180, 270, 360])
        ax[1].set_xticklabels(x_tick_labels, fontsize=10)

        y_labels = ax[1].get_yticklabels()
        y_labels[0].set_visible(False)  # hide first label
        for i, label in enumerate(y_labels):
            label.set_rotation(45)
            label.set_ha('right')

        if ax_opts['add_grid']:
            kwargs = {'linestyle': '-', 'linewidth': 2}
            ax[1].grid(**kwargs)

        if fig.subplots != (1, 1):
            fig.squeeze_fig_aspect(fig)


def _single_box_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a single box plot using SPECS data"""
    pass


def _set_cartopy_ticks_alt(ax, extent, labelsize=10):
    """
    Adds gridlines and tick labels (in degrees) outside the map for Lambert and PlateCarree.
    Places longitude labels below the map, latitude on the left.
    """
    import numpy as np

    if not extent or len(extent) != 4:
        logger.warning(f"Invalid extent {extent}, using default")
        extent = [-180, 180, -90, 90]

    try:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    except Exception as e:
        logger.warning(f"Could not set extent: {e}")

    try:
        xticks_deg = np.arange(extent[0], extent[1] + 1, 10)
        yticks_deg = np.arange(extent[2], extent[3] + 1, 10)

        # Use Cartopy's gridlines just for visual grid, no labels
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,
            linewidth=0.8,
            color='gray',
            alpha=0.6,
            linestyle='--'
        )
        gl.xlocator = FixedLocator(xticks_deg)
        gl.ylocator = FixedLocator(yticks_deg)

        # Compute projected x/y positions of tick lines
        x_tick_positions = []
        for lon in xticks_deg:
            try:
                x, _ = ax.projection.transform_point(lon, extent[2], ccrs.PlateCarree())
                x_tick_positions.append(x)
            except:
                continue

        y_tick_positions = []
        for lat in yticks_deg:
            try:
                _, y = ax.projection.transform_point(extent[0], lat, ccrs.PlateCarree())
                y_tick_positions.append(y)
            except:
                continue

        # Now map projected positions to geographic labels using the original values
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels([f"{lon}°" for lon in xticks_deg], fontsize=labelsize)
        ax.tick_params(axis='x', direction='out', pad=5)

        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels([f"{lat}°" for lat in yticks_deg], fontsize=labelsize)
        ax.tick_params(axis='y', direction='out', pad=5)

        return True

    except Exception as e:
        logger.error(f"Could not set ticks and labels: {e}")
        return False


def _set_cartopy_ticks(ax, extent, labelsize=10):
    """
    Adds gridlines and tick labels to a Cartopy GeoAxes with a non-rectangular projection.
    """
    if not extent or len(extent) != 4:
        logger.warning(f"Invalid extent {extent}, using default")
        extent = [-180, 180, -90, 90]
    
    try:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    except Exception as e:
        logger.warning(f"Could not set extent: {e}")
    
    try:
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.8, 
            color='gray', 
            alpha=0.6, 
            linestyle='--'
        )
        
        gl.top_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        
        gl.xlabel_style = {'size': labelsize, 'rotation': 0}
        gl.ylabel_style = {'size': labelsize, 'rotation': 0}
        
        return True
    except Exception as e:
        logger.error(f"Could not set ticks and labels: {e}")
        return False


def _line_contours(fig, ax, ax_opts, x, y, data2d, transform=None):
    with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
        contour_format = pu.contour_format_from_levels(
            pu.formatted_contours(ax_opts['clevs']),
            scale=ax_opts['cscale'])
        clines = ax.contour(x, y, data2d, levels=ax_opts['clevs'], colors="black",
                            alpha=0.5, transform=transform)
        if len(clines.allsegs) == 0 or all(len(seg) == 0 for seg in clines.allsegs):
            logger.warning("No contours were generated. Skipping contour labeling.")
            return
        ax.clabel(clines, inline=1, fontsize=pu.contour_label_size(fig.subplots),
                  colors="black", fmt=contour_format)


def _create_clevs(field_name, ax_opts, data2d):
    if ax_opts['clevs']:
        return
    dmin = data2d.min(skipna=True).values
    dmax = data2d.max(skipna=True).values
    logger.debug(f"dmin: {dmin}, dmax: {dmax}")

    range_val = abs(dmax - dmin)
    precision = max(0, int(np.ceil(-np.log10(range_val)))) if range_val != 0 else 6
    if range_val <= 9.0:
        precision = 1
    ax_opts['clevs_prec'] = precision
    logger.debug(f"range_val: {range_val}, precision: {precision}")

    # Generate levels
    if not ax_opts.get('create_clevs', True):
        clevs = np.around(np.linspace(dmin, dmax, 10), decimals=precision)
    else:
        clevs = np.around(np.linspace(dmin, dmax, ax_opts.get('num_clevs', 10)),
                          decimals=precision)
        clevs = np.unique(clevs)  # Remove duplicates

    # Check if levels are strictly increasing
    # If not enough unique levels, regenerate with more precision or fallback
    if len(set(clevs)) <= 2:
        logger.debug(f"Not enough unique contour levels for {field_name}.")
        # Try with more levels and higher precision
        clevs = np.linspace(dmin, dmax, 10)
        clevs = np.unique(np.around(clevs, decimals=6))
        if len(clevs) <= 2:
            # As a last resort, just use [dmin, dmax]
            clevs = np.array([dmin, dmax])

    # Ensure strictly increasing
    clevs = np.unique(clevs)  # Remove duplicates, again
    ax_opts['clevs'] = clevs

    logger.debug(f'Created contour levels for {field_name}: {ax_opts["clevs"]}')
    if ax_opts['clevs'][0] == 0.0:
        ax_opts['extend_value'] = "max"


def _filled_contours(config, field_name, ax, x, y, data2d, transform=None):
    """ Plot filled contours"""
    _create_clevs(field_name, config.ax_opts, data2d)
    norm = colors.BoundaryNorm(config.ax_opts['clevs'], ncolors=256, clip=False)
    if config.compare:  # and config.comparison_plot:
        cmap_str = config.ax_opts['use_diff_cmap']
    else:
        cmap_str = config.ax_opts['use_cmap']

    # Check for constant field
    vmin, vmax = np.nanmin(data2d), np.nanmax(data2d)
    if np.isclose(vmin, vmax):
        logger.debug("Fill with a neutral color and print text")
        ax.set_facecolor('whitesmoke')
        ax.text(0.5, 0.5, 'zero field', transform=ax.transAxes,
                ha='center', va='center', fontsize=16, color='gray', fontweight='bold')
        return None

    try:
        if np.all(np.diff(config.ax_opts['clevs']) > 0):
            cfilled = ax.contourf(x, y, data2d,
                                  levels=config.ax_opts['clevs'],
                                  cmap=cmap_str,
                                  extend=config.ax_opts['extend_value'],
                                  norm=norm,
                                  transform=transform)
            if config.ax_opts['cmap_set_under']:
                cfilled.cmap.set_under(config.ax_opts['cmap_set_under'])
            if config.ax_opts['cmap_set_over']:
                cfilled.cmap.set_over(config.ax_opts['cmap_set_over'])
            ax.set_aspect('auto')
            return cfilled
        else:
            raise ValueError("Contour levels must be increasing")
    except ValueError as e:
        logger.error(f"Error: {e}")
        try:
            cfilled = ax.contourf(x, y, data2d, extend='both',
                                  transform=transform)
        except Exception:
            cfilled = ax.contourf(x, y, data2d, extend='both')

        return cfilled


def _set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d):
    try:
        source_name = config.source_names[config.ds_index]
        if ax_opts['cbar_sci_notation']:
            fmt = pu.FlexibleOOMFormatter(min_val=data2d.min().compute().item(),
                                          max_val=data2d.max().compute().item(),
                                          math_text=True)
        else:
            fmt = pu.OOMFormatter(prec=ax_opts['clevs_prec'], math_text=True)

        if not fig.use_cartopy:
            cbar = fig.colorbar(cfilled)
        else:
            cbar = fig.colorbar(cfilled, ax=ax,
                                orientation='vertical' if config.compare or config.compare_diff else 'horizontal',
                                # extendfrac=True if config.compare else 'auto',
                                pad=pu.cbar_pad(fig.subplots),
                                fraction=pu.cbar_fraction(fig.subplots),
                                ticks=ax_opts.get('clevs', None),
                                format=fmt,
                                shrink=pu.cbar_shrink(fig.subplots))

        # Use the following ONLY with the FlexibleOOMFormatter()
        if ax_opts['cbar_sci_notation']:
            cbar.ax.text(1.05, -0.5, r'$\times 10^{%d}$' % fmt.oom,
                         transform=cbar.ax.transAxes, va='center', ha='left', fontsize=12)

        try:
            if field_name in config.spec_data and 'units' in config.spec_data[field_name]:
                units = config.spec_data[field_name]['units']
            else:
                if hasattr(data2d, 'attrs') and 'units' in data2d.attrs:
                    units = data2d.attrs['units']
                elif hasattr(data2d, 'units'):
                    units = data2d.units
                else:
                    reader = None
                    if source_name in config.readers:
                        if isinstance(config.readers[source_name], dict):
                            readers_dict = config.readers[source_name]
                            if 'NetCDF' in readers_dict:
                                reader = readers_dict['NetCDF']
                            elif readers_dict:
                                reader = next(iter(readers_dict.values()))
                        else:
                            reader = config.readers[source_name]

                    if reader and hasattr(reader, 'datasets'):
                        if findex in reader.datasets and 'vars' in reader.datasets[
                            findex]:
                            field_var = reader.datasets[findex]['vars'].get(field_name)
                            if field_var and hasattr(field_var,
                                                     'attrs') and 'units' in field_var.attrs:
                                units = field_var.attrs['units']
                            elif field_var and hasattr(field_var, 'units'):
                                units = field_var.units
                            else:
                                units = "n.a."
                        else:
                            units = "n.a."
                    else:
                        units = "n.a."
        except Exception as e:
            logger.warning(f"Error getting units: {e}")
            units = "n.a."

        if ax_opts['clabel'] is None:
            cbar_label = units
        else:
            cbar_label = ax_opts['clabel']
        cbar.set_label(cbar_label, size=pu.bar_font_size(fig.subplots))

        # Set font size for colorbar ticks
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(pu.contour_tick_font_size(fig.subplots))
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(pu.contour_tick_font_size(fig.subplots))

    except Exception as e:
        logger.error(f"Failed to add colorbar: {e}")


def _set_const_colorbar(cfilled, fig, ax):
    _ = fig.colorbar(cfilled, ax=ax, shrink=0.5)


def colorbar(mappable):
    """
    Create a colorbar that works with both standard Matplotlib Axes and Cartopy GeoAxes.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from cartopy.mpl.geoaxes import GeoAxes

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure

    # Check if the axes is a GeoAxes
    if isinstance(ax, GeoAxes):
        # Create a new axes for the colorbar with the same projection
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=type(ax),
                                  projection=ax.projection)
    else:
        # Standard Matplotlib Axes
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create the colorbar using the updated Figure.colorbar()
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


@dataclass()
class Plotter:
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @staticmethod
    def simple_plot(config, data_to_plot):
        """
        Create a basic plot, i.e. one without specifications.
        """
        no_specs_plotter = SimplePlotter()
        no_specs_plotter.plot(config, data_to_plot)
        pu.output_basic(config, data_to_plot[3])


@dataclass()
class SimplePlotter:

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")

    def simple_plot(self, config, field_to_plot):
        self.plot(config, field_to_plot)
        pu.output_basic(config, field_to_plot[3])

    @staticmethod
    def plot(config, field_to_plot):
        """ Create a basic plot (ala ncview)
        Parameters:
            config: ConfigManager
            field_to_plot: tuple (data2d, dim1, dim2, field_name, plot_type, findex, map_params)
        """
        plot_type = field_to_plot[4]
        if plot_type == 'xy':
            _simple_xy_plot(config, field_to_plot)
        elif plot_type == 'yz':
            _simple_yz_plot(config, field_to_plot)
        elif plot_type == 'sc':
            _simple_sc_plot(config, field_to_plot)
        elif plot_type == 'graph':
            _simple_graph_plot(config, field_to_plot)


@dataclass()
class SinglePlotter(Plotter):

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")

    def single_plots(self, config: ConfigManager, field_to_plot: tuple,
                     level: int = None):
        self.plot(config, field_to_plot, level)

    @staticmethod
    def plot(config, field_to_plot, level):
        """ Create a single plot using specs data
        Parameters:
            config: ConfigManager
            field_to_plot: tuple (data2d, dim1, dim2, field_name, plot_type, findex, map_params)
            level: int (optional)
        """
        plot_type = field_to_plot[4] + 'plot'
        if plot_type == 'yzplot':
            _single_yz_plot(config, field_to_plot)
        if plot_type == 'xtplot':
            _single_xt_plot(config, field_to_plot)
        if plot_type == 'txplot':
            _single_tx_plot(config, field_to_plot)
        if plot_type == 'xyplot':
            _single_xy_plot(config, field_to_plot, level)
        if plot_type == 'polarplot':
            _single_polar_plot(config, field_to_plot)
        if plot_type == 'scplot':
            _single_scat_plot(config, field_to_plot)
        # TODO: for user defined functions you need to do the following:
        # elif plot_type == constants.myplot:
        #     self._myplot_subplot(config, field_to_plot)


@dataclass()
class ComparisonPlotter:
    to_compare: list

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")

    def comparison_plots(self, config: ConfigManager, field_to_plot: tuple,
                        level: int = None):
        """Create a comparison plot using specs data

        Parameters:
            config: ConfigManager
            field_to_plot: tuple
            level: int (optional)
        """
        plot_type = field_to_plot[4] + 'plot'
        if plot_type not in ['xyplot', 'yzplot', 'xtplot', 'polarplot', 'scplot']:
            plot_type = field_to_plot[2]

        if plot_type == 'yzplot':
            _single_yz_plot(config, field_to_plot)
        elif plot_type == 'xtplot':
            _single_xt_plot(config, field_to_plot)
        elif plot_type == 'txplot':
            _single_tx_plot(config, field_to_plot)
        elif plot_type == 'xyplot':
            _single_xy_plot(config, field_to_plot, level)
        elif plot_type == 'polarplot':
            _single_polar_plot(config, field_to_plot)
        elif plot_type == 'scplot':
            _single_scat_plot(config, field_to_plot)
        else:
            self.logger.error(f'{plot_type} is not implemented')
