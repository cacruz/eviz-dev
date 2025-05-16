from dataclasses import dataclass
import os
import re
import cftime
import logging
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import networkx as nx

from sklearn.metrics import mean_squared_error

import eviz.lib.const as constants
import eviz.lib.autoviz.utils as pu
import eviz.lib.utils as u
from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.lib.data.pipeline import DataProcessor
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

    edge_weights = [graph[u][v]['weight'] * 2 for u, v in graph.edges()]
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
        raise TypeError('dim1 and dim2 must be either numpy.ndarrays or xarray.DataArrays.')

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
            y0 = data2d.coords[config.meta_coords['yc'][config.model_name](ax_dims[1])][0].values
            y1 = data2d.coords[config.meta_coords['zc'][config.model_name](ax_dims[1])][-1].values
            ax.plot(data2d, data2d.coords[config.meta_coords['yc'][config.model_name]])
            ax.set_ylim(y0, y1)
        elif ax_dims[0] == 'yc':
            dim_names = config.meta_coords['yc'][config.model_name](ax_dims[1]).split(',')
            for i in dim_names:
                if i in data2d.dims:
                    gooddim = i
            ax.plot(data2d, data2d.coords[gooddim].values)


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
    cf = ax.contourf(dim1.values, dim2.values, shift_columns(data2d), cmap=config.cmap)
    co = ax.contour(dim1.values, dim2.values, shift_columns(data2d), levels, linewidths=(1,), origin='lower')
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


def _simple_sc_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a simple scatter plot """
    data2d, dim1, dim2, field_name, plot_type = data_to_plot
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    scat = ax.scatter(dim1, dim2, c=data2d,
                      cmap='coolwarm', s=5,
                      transform=ccrs.PlateCarree())
    cbar = fig.colorbar(scat, ax=ax, shrink=0.5)
    cbar.set_label("ppb")
    ax.stock_img()
    ax.coastlines()
    ax.gridlines()
    ax.set_title(f'{field_name}')


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
    cf = ax.contourf(dim1.values, dim2.values, data2d, cmap=config.cmap)
    co = ax.contour(dim1.values, dim2.values, data2d, levels, linewidths=(1,), origin='lower')
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
    source_name = config.source_names[config.ds_index]
    data2d, x, y, field_name, plot_type, findex, fig, ax_temp = data_to_plot
    ax_opts = config.ax_opts
    ax = ax_temp
    logger.debug(f'Plotting {field_name}')
    rc = {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
    with mpl.rc_context(rc=rc):
        # TODO: Need set extent function!
        if ax_opts['extent']:
            if ax_opts['extent'] == 'conus':
                extent = [-140, -40, 15, 65]  # [lonW, lonE, latS, latN]
            else:
                extent = ax_opts['extent']
        else:
            extent = [-180, 180, -90, 90]

        if fig.use_cartopy:
            ax = fig.set_cartopy_latlon_opts(ax, extent)
            ax.set_extent(extent)
            ax = fig.set_cartopy_features(ax)
            ax.stock_img()
            scat = ax.scatter(x, y, c=data2d, cmap=ax_opts['use_cmap'], s=5, transform=ccrs.PlateCarree())
        else:
            scat = ax.scatter(x, y, c=data2d, cmap=ax_opts['use_cmap'], s=2)

        cbar = fig.fig.colorbar(scat, ax=ax,
                                orientation='horizontal',
                                pad=pu.cbar_pad(fig.subplots),
                                fraction=pu.cbar_fraction(fig.subplots),
                                extendfrac='auto',
                                shrink=0.5)

        if ax_opts['clabel'] is None:
            if 'units' in config.spec_data[field_name]:
                cbar_label = config.spec_data[field_name]['units']
            else:
                try:
                    cbar_label = data_to_plot['vars'][field_name].units
                except:
                    logger.error(f"Please specify {field_name} units in specs file")
                    cbar_label = "n.a."
        else:
            cbar_label = ax_opts['clabel']
        cbar.set_label(cbar_label, size=pu.bar_font_size(fig.subplots))
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(pu.contour_tick_font_size(fig.subplots))

        ax.set_title(f'{field_name}')


def _determine_axes_shape(fig, ax_temp):
    """Determine the shape of the axes."""
    if isinstance(ax_temp, list):
        # If ax_temp is a list of axes, determine the shape from the list
        num_axes = len(ax_temp)
        if num_axes == 1:
            return 1, 1
        else:
            return fig.get_gs_geometry()
    else:
        # If ax_temp is a single GeoAxes, assume shape is (1, 1)
        return 1, 1


def _select_axes(ax_temp, axes_shape, ax_opts, axindex):
    """Select the appropriate axes based on the shape and options."""
    if axes_shape == (3, 1):
        if ax_opts['is_diff_field']:
            return ax_temp[2]
        else:
            return ax_temp[axindex]
    elif axes_shape == (2, 2):
        if ax_opts['is_diff_field']:
            if ax_opts['add_extra_field_type']:
                return ax_temp[3]
            return ax_temp[2]
        else:
            return ax_temp[axindex]
    return ax_temp


def _single_xy_plot(config: ConfigManager, data_to_plot: tuple, level: int) -> None:
    """ Create a single xy (lat-lon) plot using SPECS data

    Parameters:
        config (ConfigManager) : configuration used to specify data sources
        data_to_plot (tuple) : dict with plotted data and specs
        level (int) : vertical level
    """
    data2d, x, y, field_name, plot_type, findex, fig, ax_temp = data_to_plot

    # Determine the shape of the axes
    axes_shape = _determine_axes_shape(fig, ax_temp)

    # Select the appropriate axes
    ax_opts = config.ax_opts
    ax = _select_axes(ax_temp, axes_shape, ax_opts, config.axindex)

    if data2d is None:
        return
    
    logger.info(f'Plotting {field_name}')
    ax_opts = fig.update_ax_opts(field_name, ax, 'xy', level=level)
    fig.plot_text(field_name, ax, 'xy', level=level, data=data2d)

    # Handle single axes or list of axes
    if isinstance(ax, list):
        for single_ax in ax:
            _plot_xy_data(config, single_ax, data2d, x, y, field_name, fig, ax_opts, level,
                          plot_type, findex)
    else:
        _plot_xy_data(config, ax, data2d, x, y, field_name, fig, ax_opts, level,
                      plot_type, findex)


def _plot_xy_data(config, ax, data2d, x, y, field_name, fig, ax_opts, level,
                  plot_type, findex):
    """Helper function to plot YZ data on a single axes."""
    source_name = config.source_names[config.ds_index]
    if ax_opts.get('extent'):
        if ax_opts['extent'] == 'conus':
            extent = [-140, -40, 15, 65]  # [lonW, lonE, latS, latN]
        else:
            extent = ax_opts['extent']
    else:
        extent = [-180, 180, -90, 90]

    if 'fill_value' in config.spec_data[field_name]['xyplot']:
        fill_value = config.spec_data[field_name]['xyplot']['fill_value']
        data2d = data2d.where(data2d != fill_value, np.nan)

    # Check if we're using Cartopy and if the axis is a GeoAxes
    is_cartopy_axis = False
    try:
        from cartopy.mpl.geoaxes import GeoAxes
        is_cartopy_axis = isinstance(ax, GeoAxes)
    except ImportError:
        pass

    if fig.use_cartopy and is_cartopy_axis:
        # Only pass transform if using Cartopy GeoAxes
        cfilled = _filled_contours(config, field_name, ax, x, y, data2d, transform=ccrs.PlateCarree())
    else:
        # Don't pass transform for regular Matplotlib axes
        cfilled = _filled_contours(config, field_name, ax, x, y, data2d)

    if np.all(np.diff(config.ax_opts['clevs']) > 0):
        if ax_opts['cscale'] is not None:
            contour_format = pu.contour_format_from_levels(pu.formatted_contours(ax_opts['clevs']),
                                                           scale=ax_opts['cscale'])
        else:
            contour_format = pu.contour_format_from_levels(pu.formatted_contours(ax_opts['clevs']))
        _set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d)
        _line_contours(fig, ax, ax_opts, x, y, data2d)
    else:
        _set_const_colorbar(cfilled, fig, ax)

    if config.compare and config.ax_opts['is_diff_field']:
        # Get the field name in a way that works with the new reader structure
        try:
            if 'name' in config.spec_data[field_name]:
                name = config.spec_data[field_name]['name']
            else:
                # Try to get the name from the reader
                reader = None
                if source_name in config.readers:
                    if isinstance(config.readers[source_name], dict):
                        # New structure - get the primary reader
                        readers_dict = config.readers[source_name]
                        if 'NetCDF' in readers_dict:
                            reader = readers_dict['NetCDF']
                        elif readers_dict:
                            reader = next(iter(readers_dict.values()))
                    else:
                        # Old structure - direct access
                        reader = config.readers[source_name]
                
                # If we found a reader, try to get the field name
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
                    # Try to get name from data directly
                    if hasattr(data2d, 'attrs') and 'long_name' in data2d.attrs:
                        name = data2d.attrs['long_name']
                    else:
                        name = field_name
        except Exception as e:
            logger.warning(f"Error getting field name: {e}")
            name = field_name

        level_text = None
        if config.ax_opts['zave']:
            level_text = ' (Column Mean)'
        elif config.ax_opts['zsum']:
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
        plt.suptitle(
            name, fontweight='bold',
            fontstyle='italic',
            fontsize=pu.image_font_size(fig.subplots))

    if config.add_logo:
        ax0 = fig.get_axes()[0]
        pu.add_logo_fig(fig, fig.EVIZ_LOGO)

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
    data2d, x, y, field_name, plot_type, findex, fig, ax_temp = data_to_plot
    
    # Get vertical coordinate safely
    zc = config.get_model_dim_name('zc')
    vertical_coord = None
    vertical_units = 'n.a.'
    
    # Handle zc being a dictionary, list, or string
    if isinstance(zc, dict):
        # Log the issue
        logger.warning(f"get_model_dim_name('zc') returned a dictionary: {zc}")
        # Try to extract a usable dimension
        for dim_name in data2d.coords:
            if 'lev' in dim_name or 'z' in dim_name or 'height' in dim_name:
                vertical_coord = data2d.coords[dim_name]
                break
    elif isinstance(zc, list):
        # Try each dimension in the list
        for z in zc:
            if z in data2d.coords:
                vertical_coord = data2d.coords[z]
                break
    else:
        # Handle the case where zc is a string (the expected case)
        if zc and zc in data2d.coords:
            vertical_coord = data2d.coords[zc]
        else:
            # Try common vertical dimension names
            for dim_name in ['lev', 'level', 'z', 'height', 'altitude', 'pressure']:
                if dim_name in data2d.coords:
                    vertical_coord = data2d.coords[dim_name]
                    break
    
    # If we found a vertical coordinate, get its units
    if vertical_coord is not None:
        vertical_units = vertical_coord.attrs.get('units', 'n.a.')
    else:
        # Create a dummy vertical coordinate based on the shape
        logger.warning("Could not find vertical coordinate, creating dummy values")
        if len(data2d.shape) >= 2:
            # Assume first dimension is vertical in a YZ plot
            vertical_coord = np.arange(data2d.shape[0])
    
    # Determine the shape of the axes
    axes_shape = _determine_axes_shape(fig, ax_temp)

    # Select the appropriate axes
    ax_opts = config.ax_opts
    ax = _select_axes(ax_temp, axes_shape, ax_opts, config.axindex)

    if data2d is None:
        return
    
    logger.info(f'Plotting {field_name}')
    ax_opts = fig.update_ax_opts(field_name, ax, 'yz')
    fig.plot_text(field_name, ax, 'yz', level=None, data=data2d)
    # Handle single axes or list of axes
    if isinstance(ax, list):
        for single_ax in ax:
            _plot_yz_data(config, single_ax, data2d, x, y, field_name, fig, ax_opts, vertical_units,
                          plot_type, findex)
    else:
        _plot_yz_data(config, ax, data2d, x, y, field_name, fig, ax_opts, vertical_units,
                      plot_type, findex)
        
def _plot_yz_data(config, ax, data2d, x, y, field_name, fig, ax_opts, vertical_units,
                  plot_type, findex):
    """Helper function to plot YZ data on a single axes."""
    source_name = config.source_names[config.ds_index]
    prof_dim = None
    if ax_opts['profile_dim']:
        prof_dim = ax_opts['profile_dim']
        dep_var = None
        if prof_dim == 'yc':
            dep_var = 'zc'
        if prof_dim == 'zc':
            dep_var = 'yc'
        prof_dim = ax_opts['profile_dim']
        data2d = data2d.mean(dim=config.get_model_dim_name(prof_dim))
        _single_prof_plot(config, data2d, fig, ax, ax_opts, (prof_dim, dep_var))
        # TODO: Refactor
        if 'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        else:
            try:
                units = data2d.units
            except Exception as e:
                logger.error(f"{e}: Please specify {field_name} units in specs file")
                units = "n.a."
        if dep_var == 'zc':
            ax.set_xlabel(units)
            ax.set_ylabel("Pressure (" + vertical_units + ")", size=pu.axes_label_font_size(fig.subplots))

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
    if config.use_trop_height and not prof_dim:
        # TODO: move to 'model' layer so that we can easily do regridding and reset flag
        overlay = DataProcessor(config, plot_type)
        tropp = overlay.process_data()
        # This should be the only call needed here:
        if overlay.trop_ok:
            ax.plot(x, tropp, linewidth=2, color="k", linestyle="--")
        # The following is temporary, while the TODO above is not done.
        config.use_trop_height = None

    if config.compare and config.ax_opts['is_diff_field']:
        # Get the field name in a way that works with the new reader structure
        try:
            if 'name' in config.spec_data[field_name]:
                name = config.spec_data[field_name]['name']
            else:
                # Try to get the name from the reader
                reader = None
                if source_name in config.readers:
                    if isinstance(config.readers[source_name], dict):
                        # New structure - get the primary reader
                        readers_dict = config.readers[source_name]
                        if 'NetCDF' in readers_dict:
                            reader = readers_dict['NetCDF']
                        elif readers_dict:
                            reader = next(iter(readers_dict.values()))
                    else:
                        # Old structure - direct access
                        reader = config.readers[source_name]
                
                # If we found a reader, try to get the field name
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
                    # Try to get name from data directly
                    if hasattr(data2d, 'attrs') and 'long_name' in data2d.attrs:
                        name = data2d.attrs['long_name']
                    else:
                        name = field_name
        except Exception as e:
            logger.warning(f"Error getting field name: {e}")
            name = field_name
            
        plt.suptitle(
            name, fontweight='bold',
            fontstyle='italic',
            fontsize=pu.image_font_size(fig.subplots))

    if config.add_logo:
        pu.add_logo(fig, fig.EVIZ_LOGO)


def _set_ax_ranges(config, field_name, fig, ax, ax_opts, y, units):
    """ Create a sensible number of vertical levels """
    y_ranges = np.array([1000, 700, 500, 300, 200, 100])
    if units == "Pa":
        y_ranges = y_ranges * 100
        if y.min() <= 1000.0:
            y_ranges = np.append(y_ranges, np.array([70, 50, 30, 20, 10]) * 100)
        if y.min() <= 20.:
            y_ranges = np.append(y_ranges, np.array([7, 5, 3, 2, 1, .7, .5, .3, .2, .1]) * 100)
        if y_ranges[-1] != y.min():
            y_ranges = np.append(y_ranges, y.min())
    else:   # TODO hPa (mb) only?, do we need meters?
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
                logger.error(f"Upper level value ({hi_z}) must be less than low level value ({lo_z})")
                return None

    # These can be defined for global or regional models. We let the respective model
    # override the extents. For generic, we assume they are global extents.
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
    ax.tick_params(width=3, length=6)
    # The vertical coordinate can have different units. For generic, we assume pressure
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


# Fix the wrapped lines problem in POLAR plots
#
# Function z_masked_overlap()
#
# The function z_masked_overlap was taken from
# https://github.com/SciTools/cartopy/issues/1421 from htonchia
def z_masked_overlap(axe, X, Y, Z, source_projection=None):
    """
    for data in projection axe.projection
    find and mask the overlaps (more 1/2 the axe.projection range)

    X, Y either the coordinates in axe.projection or longitudes latitudes
    Z the data
    operation one of 'pcorlor', 'pcolormesh', 'countour', 'countourf'

    if source_projection is a geodetic CRS data is in geodetic coordinates
    and should first be projected in axe.projection

    X, Y are 2D same dimension as Z for contour and contourf
    same dimension as Z or with an extra row and column for pcolor
    and pcolormesh

    return ptx, pty, Z
    """
    if not hasattr(axe, 'projection'):
        return X, Y, Z
    if not isinstance(axe.projection, ccrs.Projection):
        return X, Y, Z
    if len(X.shape) != 2 or len(Y.shape) != 2:
        return X, Y, Z
    if (source_projection is not None and
        isinstance(source_projection, ccrs.Geodetic)):
        transformed_pts = axe.projection.transform_points(
                                   source_projection, X, Y)
        ptx, pty = transformed_pts[..., 0], transformed_pts[..., 1]
    else:
        ptx, pty = X, Y

    with np.errstate(invalid='ignore'):
        # diagonals have one less row and one less columns
        diagonal0_lengths = np.hypot(
                               ptx[1:, 1:] - ptx[:-1, :-1],
                               pty[1:, 1:] - pty[:-1, :-1])
        diagonal1_lengths = np.hypot(
                               ptx[1:, :-1] - ptx[:-1, 1:],
                               pty[1:, :-1] - pty[:-1, 1:])
        to_mask = ((diagonal0_lengths > (
                       abs(axe.projection.x_limits[1]
                           - axe.projection.x_limits[0])) / 2) |
                   np.isnan(diagonal0_lengths) |
                   (diagonal1_lengths > (
                       abs(axe.projection.x_limits[1]
                           - axe.projection.x_limits[0])) / 2) |
                   np.isnan(diagonal1_lengths))
        # TODO check if we need to do something about surrounding vertices
        # add one extra colum and row for contour and contourf
        if (to_mask.shape[0] == Z.shape[0] - 1 and
            to_mask.shape[1] == Z.shape[1] - 1):
            to_mask_extended = np.zeros(Z.shape, dtype=bool)
            to_mask_extended[:-1, :-1] = to_mask
            to_mask_extended[-1, :] = to_mask_extended[-2, :]
            to_mask_extended[:, -1] = to_mask_extended[:, -2]
            to_mask = to_mask_extended
        if np.any(to_mask):
            Z_mask = getattr(Z, 'mask', None)
            to_mask = to_mask if Z_mask is None else to_mask | Z_mask
            Z = np.ma.masked_where(to_mask, Z)

        return ptx, pty, Z


def _single_polar_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a single polar plot using SPECS data

    Parameters:
        config (Config) : configuration used to specify data sources
        data_to_plot (tuple) : dict with plotted data and specs
    """
    source_name = config.source_names[config.ds_index]
    data2d, x, y, field_name, plot_type, findex, fig, ax_temp = data_to_plot
    ax_opts = config.ax_opts
    
    logger.debug(f'Plotting {field_name}')
    if data2d is None:
        logger.error(f"No data to plot for {field_name}")
        return
    
    # Debug: Log data shape and stats
    logger.info(f"Polar plot data shape: {data2d.shape}, dims: {data2d.dims}")
    logger.info(f"Polar plot data stats: min={data2d.min().values}, max={data2d.max().values}")
    logger.info(f"Polar plot x shape: {x.shape if x is not None else None}")
    logger.info(f"Polar plot y shape: {y.shape if y is not None else None}")
    
    # Determine which pole to use
    if ax_opts['use_pole'] == 'south':
        projection = ccrs.SouthPolarStereo()
        extent_lat = -60  # Southern limit for South Polar plot
    else:
        projection = ccrs.NorthPolarStereo()
        extent_lat = 60   # Northern limit for North Polar plot
    
    # Create a new figure with the polar projection
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=projection)
    
    # Set the extent to focus on the polar region
    ax.set_extent([-180, 180, extent_lat, 90 if ax_opts['use_pole'] == 'north' else -90], ccrs.PlateCarree())
    
    # Create contour levels
    if 'clevs' not in ax_opts or not ax_opts['clevs']:
        vmin = np.nanmin(data2d.values)
        vmax = np.nanmax(data2d.values)
        clevs = np.linspace(vmin, vmax, 11)
        ax_opts['clevs'] = clevs
    else:
        clevs = ax_opts['clevs']
    
    # Get coordinates
    if x is None or y is None:
        # Try to get coordinates from data2d
        dims = list(data2d.dims)
        if len(dims) >= 2:
            try:
                lon_name = dims[1] if 'lon' in dims[1] or 'x' in dims[1] else dims[0]
                lat_name = dims[0] if 'lat' in dims[0] or 'y' in dims[0] else dims[1]
                
                lons = data2d[lon_name].values
                lats = data2d[lat_name].values
                
                # Create meshgrid
                lon_mesh, lat_mesh = np.meshgrid(lons, lats)
                
                logger.info(f"Created meshgrid from dimensions: {lon_name}, {lat_name}")
                logger.info(f"Longitude range: {lons.min()} to {lons.max()}")
                logger.info(f"Latitude range: {lats.min()} to {lats.max()}")
            except Exception as e:
                logger.error(f"Error creating meshgrid from data2d: {e}")
                return
        else:
            logger.error(f"Data has fewer than 2 dimensions: {dims}")
            return
    else:
        # Use provided coordinates
        try:
            # Check if coordinates are DataArrays or numpy arrays
            x_values = x.values if hasattr(x, 'values') else x
            y_values = y.values if hasattr(y, 'values') else y
            
            # Create meshgrid if coordinates are 1D
            if len(x_values.shape) == 1 and len(y_values.shape) == 1:
                lon_mesh, lat_mesh = np.meshgrid(x_values, y_values)
                logger.info(f"Created meshgrid from provided coordinates")
                logger.info(f"Longitude range: {x_values.min()} to {x_values.max()}")
                logger.info(f"Latitude range: {y_values.min()} to {y_values.max()}")
            else:
                # Assume coordinates are already 2D
                lon_mesh, lat_mesh = x_values, y_values
        except Exception as e:
            logger.error(f"Error processing provided coordinates: {e}")
            return
    
    # Get data values
    data_values = data2d.values
    
    # Check if data shape matches coordinate shape
    if lon_mesh.shape != data_values.shape:
        logger.warning(f"Data shape {data_values.shape} doesn't match coordinate shape {lon_mesh.shape}")
        # Try transposing the data
        if data_values.shape == (lon_mesh.shape[1], lon_mesh.shape[0]):
            data_values = data_values.T
            logger.info("Transposed data to match coordinate shape")
        else:
            logger.error(f"Cannot reconcile data shape {data_values.shape} with coordinate shape {lon_mesh.shape}")
            return
    
    # Try different plotting methods until one works
    try:
        # Method 1: pcolormesh (most reliable for irregular grids)
        logger.info("Trying pcolormesh for polar plot")
        pcm = ax.pcolormesh(
            lon_mesh, lat_mesh, data_values,
            transform=ccrs.PlateCarree(),
            cmap=ax_opts['use_cmap'],
            norm=colors.BoundaryNorm(clevs, ncolors=256) if len(clevs) > 1 else None,
            shading='auto'
        )
        plot_success = True
    except Exception as e1:
        logger.warning(f"pcolormesh failed: {e1}")
        try:
            # Method 2: contourf
            logger.info("Trying contourf for polar plot")
            pcm = ax.contourf(
                lon_mesh, lat_mesh, data_values,
                levels=clevs,
                transform=ccrs.PlateCarree(),
                cmap=ax_opts['use_cmap'],
                extend='both'
            )
            plot_success = True
        except Exception as e2:
            logger.warning(f"contourf failed: {e2}")
            try:
                # Method 3: imshow as last resort
                logger.info("Trying imshow for polar plot")
                pcm = ax.imshow(
                    data_values,
                    extent=[-180, 180, -90 if ax_opts['use_pole'] == 'south' else 0, 
                            0 if ax_opts['use_pole'] == 'south' else 90],
                    transform=ccrs.PlateCarree(),
                    cmap=ax_opts['use_cmap'],
                    aspect='auto'
                )
                plot_success = True
            except Exception as e3:
                logger.error(f"All plotting methods failed: {e3}")
                plot_success = False
    
    if not plot_success:
        logger.error("Failed to create polar plot")
        return
    
    # Add colorbar
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.5, pad=0.05)
    
    # Set colorbar label
    if 'units' in config.spec_data[field_name]:
        units = config.spec_data[field_name]['units']
    else:
        try:
            units = data2d.attrs.get('units', 'n.a.')
        except Exception:
            units = 'n.a.'
    
    if units == '1':
        units = '%'
    
    if ax_opts['clabel'] is None:
        cbar_label = units
    else:
        cbar_label = ax_opts['clabel']
    
    cbar.set_label(label=cbar_label, size=12, weight='bold')
    
    # Add title
    if 'name' in config.spec_data[field_name]:
        ax.set_title(config.spec_data[field_name]['name'], y=1.03, fontsize=14, weight='bold')
    else:
        ax.set_title(field_name, y=1.03, fontsize=14, weight='bold')
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=0)
    
    # Add gridlines if requested
    if ax_opts['add_grid']:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    
    # Set circular boundary if requested
    if ax_opts['boundary']:
        theta = np.linspace(0, 2 * np.pi, 100)
        center = [0.5, 0.5]
        radius = 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
    
    # Save the figure
    if config.print_to_file:
        output_dir = config.output_dir
        output_fname = f"{field_name}_polar_{ax_opts['use_pole']}.{config.print_format}"
        filename = os.path.join(output_dir, output_fname)
        plt.savefig(filename, bbox_inches='tight')
        logger.info(f"Saved polar plot to {filename}")
    else:
        plt.tight_layout()
        plt.show()


def _single_xt_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a single xt (time-series) plot using SPECS data

    Note:
        XT plots are 2D plots where the variable, X, is averaged over all space,
        is plotted against time, T.

    Parameters:
        config (Config) : configuration used to specify data sources
        data_to_plot (tuple) : dict with plotted data and specs
    """

    data2d, _, _, field_name, plot_type, findex, fig, ax_temp = data_to_plot
    ax_opts = config.ax_opts
    ax = ax_temp
    if np.shape(ax_temp) == (3,):
        if ax_opts['is_diff_field']:
            ax = ax_temp[2]
        else:
            ax = ax_temp[findex]
    elif np.shape(ax_temp) == (2, 2):
        if ax_opts['is_diff_field']:
            ax = ax_temp[1, 0]
            if config.ax_opts['add_extra_field_type']:
                ax = ax_temp[1, 1]
        else:
            ax = ax_temp[0, findex]

    logger.debug(f'Plotting {field_name}')
    if data2d is None:
        return
    ax_opts = fig.update_ax_opts(field_name, ax, 'xt', level=0)
    fig.plot_text(field_name, ax, 'xt', data=data2d)

    _time_series_plot(config, ax, ax_opts, fig, data2d, field_name, findex)

    if fig.subplots != (1, 1):
        fig.squeeze_fig_aspect(fig)


def _time_series_plot(config, ax, ax_opts, fig, data2d, field_name, findex):
    with mpl.rc_context(ax_opts['time_series_plot_linestyle']):
        dmin = data2d.min(skipna=True).values
        dmax = data2d.max(skipna=True).values
        logger.debug(f"dmin: {dmin}, dmax: {dmax}")

        # Get time coordinates safely
        try:
            # First try to get the time dimension from config
            tc_dim = config.get_model_dim_name('tc')
            
            # If that fails, try common time dimension names
            if tc_dim is None or tc_dim not in data2d.coords:
                # Try common time dimension names
                for time_dim in ['time', 't', 'TIME', 'Time']:
                    if time_dim in data2d.coords:
                        tc_dim = time_dim
                        break
            
            # Get the time values
            if tc_dim and tc_dim in data2d.coords:
                time_coords = data2d.coords[tc_dim].values
            else:
                # If we still don't have time coordinates, try to get them from the first dimension
                if len(data2d.dims) > 0:
                    time_coords = data2d[data2d.dims[0]].values
                else:
                    # Last resort: create a dummy time array
                    logger.warning("Could not find time coordinates, creating dummy time values")
                    time_coords = np.arange(len(data2d))
                    
            # Handle cftime objects if needed
            if isinstance(time_coords[0], cftime._cftime.DatetimeNoLeap):
                try:
                    # Try to convert to pandas datetime
                    time_coords = pd.to_datetime([str(t) for t in time_coords])
                except Exception as e:
                    logger.warning(f"Error converting cftime to pandas datetime: {e}")
                    # Fall back to creating dummy time values
                    time_coords = np.arange(len(data2d))
                    
        except Exception as e:
            logger.warning(f"Error getting time coordinates: {e}")
            # Fall back to creating dummy time values
            time_coords = np.arange(len(data2d))

        t0 = time_coords[0]
        t1 = time_coords[-1]

        window_size = 0
        if 'mean_type' in config.spec_data[field_name]['xtplot']:
            if config.spec_data[field_name]['xtplot']['mean_type'] == 'rolling':
                if 'window_size' in config.spec_data[field_name]['xtplot']:
                    window_size = config.spec_data[field_name]['xtplot']['window_size']
                    
        # Plot the data, safely handling indices
        if window_size > 0 and len(data2d) > 2*window_size:
            # Only apply window if there's enough data
            end_idx = max(0, len(time_coords) - window_size - 1)
            ax.plot(time_coords[window_size:end_idx], data2d[window_size:end_idx])
        else:
            # Otherwise plot all data
            ax.plot(time_coords, data2d)

        if 'add_trend' in config.spec_data[field_name]['xtplot']:
            logger.info('Adding trend')
            if config.spec_data[field_name]['xtplot']['add_trend']:
                try:
                    # Convert time to numeric safely
                    if isinstance(t0, (pd.Timestamp, np.datetime64)):
                        time_numeric = (time_coords - t0).astype('timedelta64[D]').astype(float)
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
                    logger.info(f' -- polynomial degree: {degree}')
                    coeffs = np.polyfit(time_numeric, data2d, degree)
                    trend_poly = np.polyval(coeffs, time_numeric)
                    ax.plot(time_coords, trend_poly, color="red", linewidth=1)
                except Exception as e:
                    logger.warning(f"Error calculating trend: {e}")

        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
        fig.autofmt_xdate()
        ax.set_xlim(t0, t1)
        
        # Set y limits safely
        try:
            davg = 0.5 * (abs(dmin - dmax))
            ax.set_ylim([dmin - davg, dmax + davg])
        except Exception as e:
            logger.warning(f"Error setting y limits: {e}")

        # Get units - handling the new reader structure
        try:
            source_name = config.source_names[config.ds_index]
            
            # Try to get units from spec_data first
            if 'units' in config.spec_data[field_name]:
                units = config.spec_data[field_name]['units']
            else:
                # Try to get units from the data array
                units = getattr(data2d, 'units', None)
                
                # If not found in data array, try to get from reader
                if not units:
                    reader = config.get_primary_reader(source_name)
                    if reader and hasattr(reader, 'datasets'):
                        if findex in reader.datasets and 'vars' in reader.datasets[findex]:
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

        ylabels = ax.get_yticklabels()
        for label in ylabels:
            label.set_fontsize(pu.axis_tick_font_size(fig.subplots))

        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_fontsize(pu.axis_tick_font_size(fig.subplots))

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
    data2d, _, _, field_name, plot_type, findex, fig, ax_temp = data_to_plot
    if data2d is None:
        return

    ax_opts = config.ax_opts

    # gs = mgridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 5], hspace=0.1)
    gs = mgridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 6], hspace=0.1)
    ax = list()
    ax.append(fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=180)))
    ax.append((fig.add_subplot(gs[1, 0])))

    logger.info(f'Plotting {field_name}')

    ax_opts = fig.update_ax_opts(field_name, ax, 'tx')

    dmin = data2d.min(skipna=True).values
    dmax = data2d.max(skipna=True).values

    logger.debug(f"Field: {field_name}; Min:{dmin}; Max:{dmax}")

    _create_clevs(field_name, ax_opts, data2d)
    extend_value = "both"
    if config.ax_opts['clevs'][0] == 0:
        extend_value = "max"

    norm = colors.BoundaryNorm(ax_opts['clevs'], ncolors=256, clip=False)

    # Check for data alignment
    vtimes = data2d.time.values.astype('datetime64[ms]').astype('O')
    
    # Try to get the longitude dimension using get_model_dim_name
    lon_dim = config.get_model_dim_name('xc')
    
    try:
        # If a valid dimension was found, use it
        if lon_dim:
            lons = get_data_coords(data2d, lon_dim)
        else:
            # Otherwise, try to infer it from the data
            logger.warning("Could not determine longitude dimension name. Attempting to infer from data.")
            if 'lon' in data2d.dims:
                lons = data2d.lon.values
            elif 'longitude' in data2d.dims:
                lons = data2d.longitude.values
            elif 'x' in data2d.dims:
                lons = data2d.x.values
            else:
                # Last resort: use the second dimension of the array
                # (assuming time is first, lon is second)
                logger.warning("Could not find longitude dimension. Using second dimension of data array.")
                lons = np.arange(data2d.shape[1])
    except Exception as e:
        logger.error(f"Error getting longitude coordinates: {e}")
        lons = np.arange(data2d.shape[1])
    
    # Check if data shape matches expected coordinates
    logger.warning(f"Data shape {data2d.shape} vs coordinates ({len(vtimes)}, {len(lons)})")
    
    try:
        # Handle case where data has more dimensions than expected
        if len(data2d.shape) > 2:
            logger.info(f"Data has {len(data2d.shape)} dimensions, reducing to 2D for Hovmoller plot")
            
            # Check if first dimension is time
            if data2d.shape[0] == len(vtimes):
                # If data shape is (time, level, lon), we need to average over level
                if len(data2d.shape) == 3 and data2d.shape[2] == len(lons):
                    logger.info("Averaging over vertical levels")
                    # Average over the middle dimension (level)
                    data2d_reduced = data2d.mean(axis=1)
                    
                # If data shape is (time, lat, lon), we need to average over lat
                elif len(data2d.shape) == 3 and data2d.shape[2] == len(lons):
                    logger.info("Averaging over latitude")
                    # Average over the middle dimension (lat)
                    data2d_reduced = data2d.mean(axis=1)
                    
                # Other cases - try to identify dimensions by name
                else:
                    # Try to find dimension names
                    dim_names = list(data2d.dims)
                    time_dim_idx = None
                    lon_dim_idx = None
                    
                    # Find time dimension
                    for i, dim in enumerate(dim_names):
                        if dim in ['time', 't', 'TIME']:
                            time_dim_idx = i
                            break
                    
                    # Find lon dimension
                    for i, dim in enumerate(dim_names):
                        if dim in ['lon', 'longitude', 'x']:
                            lon_dim_idx = i
                            break
                    
                    # If we found both dimensions, reduce along other axes
                    if time_dim_idx is not None and lon_dim_idx is not None:
                        # List of dimensions to average over
                        dims_to_avg = [i for i in range(len(dim_names)) 
                                     if i != time_dim_idx and i != lon_dim_idx]
                        
                        # Average over these dimensions
                        data2d_reduced = data2d.copy()
                        for dim_idx in sorted(dims_to_avg, reverse=True):
                            data2d_reduced = data2d_reduced.mean(axis=dim_idx)
                            
                        # Transpose if needed to get (time, lon) order
                        if time_dim_idx > lon_dim_idx:
                            data2d_reduced = data2d_reduced.T
                    else:
                        # Last resort - flatten all dimensions except time
                        logger.warning("Could not identify dimensions, using first dimension as time")
                        data2d_reduced = data2d.reshape(data2d.shape[0], -1).mean(axis=1)
                        lons = np.arange(data2d_reduced.shape[1])
            else:
                # If first dimension is not time, try to reshape
                logger.warning("First dimension is not time, attempting to reshape")
                # Just use the data as is and hope for the best
                data2d_reduced = data2d
                # Fix time dimension if needed
                if len(vtimes) != data2d.shape[0]:
                    vtimes = np.arange(data2d.shape[0])
                # Fix lon dimension if needed
                if len(lons) != data2d.shape[1]:
                    lons = np.arange(data2d.shape[1])
        else:
            # Data is already 2D
            data2d_reduced = data2d
            
            # Fix dimensions if needed
            if data2d.shape != (len(vtimes), len(lons)):
                logger.warning(f"Data shape {data2d.shape} doesn't match expected shape ({len(vtimes)}, {len(lons)})")
                # If data is transposed, fix it
                if data2d.shape == (len(lons), len(vtimes)):
                    logger.info("Transposing data to match coordinates")
                    data2d_reduced = data2d.T
                else:
                    # Otherwise, just use the data as is and adjust coordinates
                    vtimes = np.arange(data2d.shape[0])
                    lons = np.arange(data2d.shape[1])
    except Exception as e:
        logger.error(f"Error processing data for Hovmoller plot: {e}")
        # Fall back to using the data as is
        data2d_reduced = data2d
        # Ensure coordinates match data shape
        if len(data2d.shape) >= 2:
            vtimes = np.arange(data2d.shape[0])
            lons = np.arange(data2d.shape[1])
    
    # Make sure data shape matches coordinates for plotting
    if hasattr(data2d_reduced, 'shape') and len(data2d_reduced.shape) >= 2:
        if data2d_reduced.shape[0] != len(vtimes) or data2d_reduced.shape[1] != len(lons):
            logger.warning(f"Final data shape {data2d_reduced.shape} doesn't match coordinates ({len(vtimes)}, {len(lons)})")
            # Adjust coordinates to match data
            vtimes = np.arange(data2d_reduced.shape[0])
            lons = np.arange(data2d_reduced.shape[1])

    # Now continue with the plotting using data2d_reduced instead of data2d
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
    fig.plot_text(field_name=field_name, ax=ax[0], pid='tx', data=data2d, fontsize=10, loc='left')

    if ax_opts['torder']:
        ax[1].invert_yaxis()  # Reverse the time order

    # Plot the data
    try:
        cfilled = ax[1].contourf(lons, vtimes, data2d_reduced, ax_opts['clevs'], norm=norm,
                                cmap=ax_opts['use_cmap'], extend=extend_value)
    except Exception as e:
        logger.error(f"Error creating contour plot: {e}")
        # Fall back to pcolormesh which is more forgiving
        try:
            logger.info("Falling back to pcolormesh")
            # Create meshgrid for pcolormesh
            lon_mesh, time_mesh = np.meshgrid(lons, vtimes)
            cfilled = ax[1].pcolormesh(lon_mesh, time_mesh, data2d_reduced, 
                                    norm=norm, cmap=ax_opts['use_cmap'])
        except Exception as e2:
            logger.error(f"Error creating pcolormesh plot: {e2}")
            # Last resort - just show something
            cfilled = ax[1].imshow(data2d_reduced, aspect='auto', origin='lower',
                                norm=norm, cmap=ax_opts['use_cmap'])

    # Add gridlines and labels
    ax[1].set_xlabel("Longitude")
    ax[1].set_ylabel("Time")
    ax[1].grid(linestyle='dotted', linewidth=0.5)

    try:
        _line_contours(fig, ax[1], ax_opts, lons, vtimes, data2d_reduced)
    except Exception as e:
        logger.error(f"Error adding contour lines: {e}")

    cbar = fig.colorbar(cfilled, orientation='horizontal', pad=0.1, aspect=70, extendrect=True)    
    cbar.set_label('m $s^{-1}$')

    if lons[0] <= -179:
        ax[1].set_xticks([-180, -90, 0, 90, 180])
    else:
        ax[1].set_xticks([0, 90, 180, 270, 360])
    ax[1].set_xticklabels(x_tick_labels)

    try:
        # Set time ticks safely
        if len(vtimes) > 8:
            step = len(vtimes) // 8
            ax[1].set_yticks(vtimes[::step])
            ax[1].set_yticklabels(vtimes[::step])
        else:
            ax[1].set_yticks(vtimes)
            ax[1].set_yticklabels(vtimes)
        
        y_labels = ax[1].get_yticklabels()
        if len(y_labels) > 1:
            y_labels[1].set_visible(False)  # hide first label
            
        for i, label in enumerate(y_labels):
            label.set_rotation(45)
            label.set_ha('right')
    except Exception as e:
        logger.error(f"Error setting time ticks: {e}")

    if ax_opts['add_grid']:
        kwargs = {'linestyle': '-', 'linewidth': 2}
        ax[1].grid(**kwargs)

    if fig.subplots != (1, 1):
        fig.squeeze_fig_aspect(fig)


def _single_box_plot(config: ConfigManager, data_to_plot: tuple) -> None:
    """ Create a single box plot using SPECS data"""


def _line_contours(fig, ax, ax_opts, x, y, data2d):
    with mpl.rc_context(ax_opts['contour_linestyle']):
        contour_format = pu.contour_format_from_levels(pu.formatted_contours(ax_opts['clevs']),
                                                       scale=ax_opts['cscale'])
        # Generate line contours
        clines = ax.contour(x, y, data2d, levels=ax_opts['clevs'], colors="black", alpha=0.5)
        # Check if any contours were generated
        if len(clines.allsegs) == 0 or all(len(seg) == 0 for seg in clines.allsegs):
            logger.warning("No contours were generated. Skipping contour labeling.")
            return

        # Add labels to the contours
        ax.clabel(clines, inline=1, fontsize=pu.contour_label_size(fig.subplots),
                  colors="black", fmt=contour_format)


def _create_clevs(field_name, ax_opts, data2d):
    """ Create contour levels if none are specified in the fields' specs
    """
    dmin = data2d.min(skipna=True).values
    dmax = data2d.max(skipna=True).values
    logger.debug(f"dmin: {dmin}, dmax: {dmax}")

    # Automatically compute the precision based on range of dmin and dmax
    range_val = abs(dmax - dmin)
    precision = max(0, int(np.ceil(-np.log10(range_val)))) if range_val != 0 else 6
    # Extra hack for "small" ranges
    if range_val <= 9.0:
        precision = 1
    ax_opts['clevs_prec'] = precision
    logger.debug(f"range_val: {range_val}, precision: {precision}")

    # Initialize clevs with default values if not creating new ones
    if not ax_opts.get('create_clevs', True):
        # Use a reasonable default number of levels
        clevs = np.around(np.linspace(dmin, dmax, 10), decimals=precision)
    else:
        clevs = np.around(np.linspace(dmin, dmax, ax_opts.get('num_clevs', 10)), decimals=precision)
        # Ensure contour levels are strictly increasing
        clevs = np.unique(clevs)
        # If there are fewer than 2 levels, expand them to cover the range
        if len(clevs) < 2:
            clevs = np.linspace(dmin, dmax, ax_opts.get('num_clevs', 10))
    
    # Set the clevs in ax_opts
    ax_opts['clevs'] = clevs
    logger.debug(f'Created contour levels for {field_name}: {ax_opts["clevs"]}')

    # Check if the first contour level is zero
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

    # Check if transform is valid for this axis
    if transform is not None:
        try:
            from cartopy.mpl.geoaxes import GeoAxes
            if not isinstance(ax, GeoAxes):
                # If not a GeoAxes, don't use the transform
                transform = None
                logger.warning("Transform provided but axis is not a GeoAxes. Ignoring transform.")
        except ImportError:
            # If Cartopy is not available, don't use the transform
            transform = None

    try:
        if np.all(np.diff(config.ax_opts['clevs']) > 0):
            # If transform is None, it will be ignored
            cfilled = ax.contourf(x, y, data2d,
                                  robust=True,
                                  levels=config.ax_opts['clevs'],
                                  cmap=cmap_str,
                                  extend=config.ax_opts['extend_value'],
                                  norm=norm,
                                  transform=transform)
            if config.ax_opts['cmap_set_under']:
                cfilled.cmap.set_under(config.ax_opts['cmap_set_under'])
            if config.ax_opts['cmap_set_over']:
                cfilled.cmap.set_over(config.ax_opts['cmap_set_over'])
            return cfilled
        else:
            raise ValueError("Contour levels must be increasing")
    except ValueError as e:
        logger.error(f"Error: {e}")
        # Handle the case of a constant field
        # Only pass transform if it's valid for this axis
        if transform is not None:
            try:
                cfilled = ax.contourf(x, y, data2d, extend='both', transform=transform)
            except Exception as e2:
                logger.error(f"Error with transform: {e2}. Trying without transform.")
                cfilled = ax.contourf(x, y, data2d, extend='both')
        else:
            cfilled = ax.contourf(x, y, data2d, extend='both')
        return cfilled


def _set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d):
    logger.debug(f"Adding colorbar for field: {field_name}")
    logger.debug(f"cfilled: {cfilled}")
    logger.debug(f"ax: {ax}")
    logger.debug(f"ax_opts: {ax_opts}")
    logger.debug(f"clevs: {ax_opts.get('clevs', None)}")
    logger.debug(f"orientation: {'vertical' if config.compare else 'horizontal'}")

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
                                 orientation='vertical' if config.compare else 'horizontal',
                                 extendfrac=True if config.compare else 'auto',
                                 pad=pu.cbar_pad(fig.subplots),
                                 fraction=pu.cbar_fraction(fig.subplots),
                                 ticks=ax_opts.get('clevs', None),
                                 format=fmt,
                                 aspect=50,
                                 shrink=pu.cbar_shrink(fig.subplots))
            
        # Use the following ONLY with the FlexibleOOMFormatter()
        if ax_opts['cbar_sci_notation']:
            cbar.ax.text(1.05, -0.5, r'$\times 10^{%d}$' % fmt.oom,
                           transform=cbar.ax.transAxes, va='center', ha='left', fontsize=12)

        # Get units - handling the new reader structure
        try:
            # Try to get units from spec_data first
            if field_name in config.spec_data and 'units' in config.spec_data[field_name]:
                units = config.spec_data[field_name]['units']
            else:
                # Try to get units from the data array
                if hasattr(data2d, 'attrs') and 'units' in data2d.attrs:
                    units = data2d.attrs['units']
                elif hasattr(data2d, 'units'):
                    units = data2d.units
                else:
                    # Try to get from reader
                    reader = None
                    if source_name in config.readers:
                        if isinstance(config.readers[source_name], dict):
                            # New structure - get the primary reader
                            readers_dict = config.readers[source_name]
                            if 'NetCDF' in readers_dict:
                                reader = readers_dict['NetCDF']
                            elif readers_dict:
                                reader = next(iter(readers_dict.values()))
                        else:
                            # Old structure - direct access
                            reader = config.readers[source_name]
                    
                    if reader and hasattr(reader, 'datasets'):
                        if findex in reader.datasets and 'vars' in reader.datasets[findex]:
                            field_var = reader.datasets[findex]['vars'].get(field_name)
                            if field_var and hasattr(field_var, 'attrs') and 'units' in field_var.attrs:
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
    _ = fig.fig.colorbar(cfilled, ax=ax, shrink=0.5)



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
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=type(ax), projection=ax.projection)
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
        
    def plot_time_series_combined(self, config: ConfigManager, field_name: str, 
                                 start_date=None, end_date=None, output_name=None):
        """
        Plot a time series by combining data from multiple files spanning a date range.
        
        Args:
            config (ConfigManager): The configuration manager
            field_name (str): The field name to plot
            start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to None.
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to None.
            output_name (str, optional): Output file name. Defaults to None.
        """
        source_name = config.source_names[0]  # Use the first source
        
        # Get all relevant files
        file_paths = []
        for file_idx, file_entry in config.file_list.items():
            if file_entry.get('source_name') == source_name:
                # Include all files if dates not specified
                if start_date is None and end_date is None:
                    file_paths.append(file_entry['filename'])
                else:
                    # Try to extract date from filename or metadata
                    try:
                        file_date = self._extract_date_from_filename(file_entry['filename'])
                        if start_date and end_date:
                            if start_date <= file_date <= end_date:
                                file_paths.append(file_entry['filename'])
                        elif start_date:
                            if start_date <= file_date:
                                file_paths.append(file_entry['filename'])
                        elif end_date:
                            if file_date <= end_date:
                                file_paths.append(file_entry['filename'])
                    except:
                        # If date extraction fails, include the file anyway
                        self.logger.warning(f"Could not extract date from {file_entry['filename']}, including it anyway")
                        file_paths.append(file_entry['filename'])
        
        if not file_paths:
            self.logger.error(f"No files found for time series of {field_name}")
            return
            
        # Integrate the datasets
        integrated_data = config.integrator.integrate_datasets(source_name, file_paths)
        
        if integrated_data and field_name in integrated_data:
            # Create the figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot the time series
            time_var = integrated_data['time']
            data_var = integrated_data[field_name]
            
            # Plot the data
            ax.plot(time_var, data_var, linewidth=1.5)
            
            # Add title and labels
            title = f"Time Series of {field_name}"
            if start_date and end_date:
                title += f" ({start_date} to {end_date})"
            ax.set_title(title, fontsize=14)
            
            # Try to get units from metadata
            try:
                units = data_var.attrs.get('units', '')
                if units:
                    ax.set_ylabel(units, fontsize=12)
            except:
                pass
                
            ax.set_xlabel('Time', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis for dates
            fig.autofmt_xdate()
            
            # Save or display the plot
            if config.print_to_file:
                output_dir = config.output_dir
                if not output_name:
                    output_name = f"{field_name}_timeseries"
                filename = os.path.join(output_dir, f"{output_name}.{config.print_format}")
                plt.savefig(filename, bbox_inches='tight')
                self.logger.info(f"Saved time series plot to {filename}")
            else:
                plt.tight_layout()
                plt.show()
        else:
            self.logger.error(f"Failed to create time series for {field_name}")

    def plot_composite_field(self, config: ConfigManager, primary_field: str, 
                            secondary_field: str, operation: str = 'add',
                            plot_type: str = 'xy', level: int = None, output_name: str = None):
        """
        Create and plot a composite field by combining related variables from different file types.
        
        Args:
            config (ConfigManager): The configuration manager
            primary_field (str): Name of the primary field
            secondary_field (str): Name of the secondary field
            operation (str): Mathematical operation to combine fields ('add', 'subtract', 'multiply', 'divide')
            plot_type (str): Type of plot ('xy', 'yz')
            level (int, optional): Vertical level for xy plots. Defaults to None.
            output_name (str, optional): Output file name. Defaults to None.
        """
        source_name = config.source_names[0]  # Use the first source
        
        # Get the fields from any available source
        primary_data = config.integrator.get_variable_from_any_source(source_name, primary_field)
        secondary_data = config.integrator.get_variable_from_any_source(source_name, secondary_field)
        
        if primary_data is None or secondary_data is None:
            self.logger.error(f"Could not find one or both of the fields: {primary_field}, {secondary_field}")
            return
        
        # Align the data (assuming they might have different dimensions)
        try:
            # This will broadcast the arrays to compatible shapes if possible
            primary_data, secondary_data = xr.align(primary_data, secondary_data)
            
            # Perform the requested operation
            if operation == 'add':
                composite_data = primary_data + secondary_data
                op_symbol = '+'
            elif operation == 'subtract':
                composite_data = primary_data - secondary_data
                op_symbol = '-'
            elif operation == 'multiply':
                composite_data = primary_data * secondary_data
                op_symbol = '×'
            elif operation == 'divide':
                composite_data = primary_data / secondary_data
                op_symbol = '/'
            else:
                self.logger.error(f"Unknown operation: {operation}")
                return
                
            # Try to combine units
            try:
                primary_units = primary_data.attrs.get('units', '')
                secondary_units = secondary_data.attrs.get('units', '')
                
                if operation in ['add', 'subtract'] and primary_units == secondary_units:
                    composite_data.attrs['units'] = primary_units
                elif operation == 'multiply':
                    if primary_units and secondary_units:
                        composite_data.attrs['units'] = f"{primary_units}·{secondary_units}"
                    else:
                        composite_data.attrs['units'] = primary_units or secondary_units
                elif operation == 'divide':
                    if primary_units and secondary_units:
                        composite_data.attrs['units'] = f"{primary_units}/{secondary_units}"
                    else:
                        composite_data.attrs['units'] = primary_units or secondary_units
            except:
                pass
            
            # Set a name for the composite field
            composite_data.name = f"{primary_field}_{operation}_{secondary_field}"
            
            # Create figure and plot based on plot_type
            if plot_type == 'xy':
                self._plot_composite_xy(config, composite_data, primary_field, secondary_field, 
                                      operation, op_symbol, level, output_name)
            elif plot_type == 'yz':
                self._plot_composite_yz(config, composite_data, primary_field, secondary_field,
                                      operation, op_symbol, output_name)
            else:
                self.logger.error(f"Unsupported plot type: {plot_type}")
                
        except Exception as e:
            self.logger.error(f"Error creating composite field: {str(e)}")
            
    def _plot_composite_xy(self, config, data, primary_field, secondary_field, 
                          operation, op_symbol, level=None, output_name=None):
        """Plot a composite field as an xy (lat-lon) plot."""
        # Select the appropriate level if 3D data
        if level is not None and 'lev' in data.dims:
            data = data.sel(lev=level)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get dimensions for plotting
        dim1, dim2 = config.get_dim_names('xy')
        
        # Get coordinates
        x = data[dim1].values
        y = data[dim2].values
        
        # Create contour levels
        vmin, vmax = data.min().values, data.max().values
        levels = np.linspace(vmin, vmax, 20)
        
        # Plot filled contours
        cs = ax.contourf(x, y, data.values, levels=levels, cmap='RdBu_r', extend='both')
        
        # Add contour lines
        cont = ax.contour(x, y, data.values, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
        ax.clabel(cont, inline=1, fontsize=8, fmt='%.1f')
        
        # Add colorbar
        cbar = fig.colorbar(cs, ax=ax, orientation='vertical', pad=0.02)
        
        # Set units on colorbar
        if 'units' in data.attrs:
            cbar.set_label(data.attrs['units'])
        
        # Add title
        title = f"Composite: {primary_field} {op_symbol} {secondary_field}"
        if level is not None:
            title += f" (Level: {level})"
        ax.set_title(title)
        
        # Set labels
        ax.set_xlabel(dim1)
        ax.set_ylabel(dim2)
        
        # Add grid
        ax.grid(linestyle='--', alpha=0.6)
        
        # Save or display the plot
        if config.print_to_file:
            output_dir = config.output_dir
            if not output_name:
                output_name = f"composite_{primary_field}_{operation}_{secondary_field}"
                if level is not None:
                    output_name += f"_level{level}"
            filename = os.path.join(output_dir, f"{output_name}.{config.print_format}")
            plt.savefig(filename, bbox_inches='tight')
            self.logger.info(f"Saved composite plot to {filename}")
        else:
            plt.tight_layout()
            plt.show()
            
    def _plot_composite_yz(self, config, data, primary_field, secondary_field, 
                          operation, op_symbol, output_name=None):
        """Plot a composite field as a yz (zonal mean) plot."""
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get dimensions for plotting
        dim1, dim2 = config.get_dim_names('yz')
        
        # Get coordinates
        x = data[dim1].values
        y = data[dim2].values
        
        # Create contour levels
        vmin, vmax = data.min().values, data.max().values
        levels = np.linspace(vmin, vmax, 20)
        
        # Plot filled contours
        cs = ax.contourf(x, y, data.values, levels=levels, cmap='RdBu_r', extend='both')
        
        # Add contour lines
        cont = ax.contour(x, y, data.values, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
        ax.clabel(cont, inline=1, fontsize=8, fmt='%.1f')
        
        # Add colorbar
        cbar = fig.colorbar(cs, ax=ax, orientation='vertical', pad=0.02)
        
        # Set units on colorbar
        if 'units' in data.attrs:
            cbar.set_label(data.attrs['units'])
        
        # Add title
        title = f"Composite: {primary_field} {op_symbol} {secondary_field} (Zonal Mean)"
        ax.set_title(title)
        
        # Set labels
        ax.set_xlabel(dim1)
        ax.set_ylabel(dim2)
        
        # Special handling for pressure axis if applicable
        if 'lev' in data.dims or 'level' in data.dims:
            ax.set_yscale('log')
            ax.invert_yaxis()  # Pressure decreases with height
        
        # Add grid
        ax.grid(linestyle='--', alpha=0.6)
        
        # Save or display the plot
        if config.print_to_file:
            output_dir = config.output_dir
            if not output_name:
                output_name = f"composite_{primary_field}_{operation}_{secondary_field}_zonal"
            filename = os.path.join(output_dir, f"{output_name}.{config.print_format}")
            plt.savefig(filename, bbox_inches='tight')
            self.logger.info(f"Saved composite plot to {filename}")
        else:
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def _extract_date_from_filename(filename):
        """
        Extract date from filename using various common patterns.
        
        Args:
            filename (str): The filename to parse
            
        Returns:
            str: Date string in YYYY-MM-DD format
            
        Raises:
            ValueError: If no date pattern could be extracted
        """
        # Extract just the filename without path
        basename = os.path.basename(filename)
        
        # Try various regex patterns
        # Pattern 1: YYYYMMDD
        pattern1 = r'(\d{4})(\d{2})(\d{2})'
        match = re.search(pattern1, basename)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"
            
        # Pattern 2: YYYY_MM_DD or YYYY-MM-DD
        pattern2 = r'(\d{4})[_-](\d{2})[_-](\d{2})'
        match = re.search(pattern2, basename)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"
            
        # Pattern 3: YYYY.MM.DD
        pattern3 = r'(\d{4})\.(\d{2})\.(\d{2})'
        match = re.search(pattern3, basename)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"
            
        # Pattern 4: Try to extract just year and month: YYYYMM
        pattern4 = r'(\d{4})(\d{2})'
        match = re.search(pattern4, basename)
        if match:
            year, month = match.groups()
            return f"{year}-{month}-01"  # Default to first day of month
            
        # Pattern 5: Just year: YYYY
        pattern5 = r'(\d{4})'
        match = re.search(pattern5, basename)
        if match:
            year = match.group(1)
            return f"{year}-01-01"  # Default to first day of year
            
        # If no pattern matches, raise an error
        raise ValueError(f"Could not extract date from {basename}")


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

    # Add access to the new methods
    def time_series_combined(self, config: ConfigManager, field_name: str, 
                           start_date=None, end_date=None, output_name=None):
        """Simple wrapper for plot_time_series_combined."""
        self.plot_time_series_combined(config, field_name, start_date, end_date, output_name)
        
    def composite_field(self, config: ConfigManager, primary_field: str, 
                      secondary_field: str, operation: str = 'add',
                      plot_type: str = 'xy', level: int = None, output_name: str = None):
        """Simple wrapper for plot_composite_field."""
        self.plot_composite_field(config, primary_field, secondary_field, 
                                operation, plot_type, level, output_name)


@dataclass()
class SinglePlotter(Plotter):

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")

    def single_plots(self, config: ConfigManager, field_to_plot: tuple, level: int = None):
        self.plot(config, field_to_plot, level)

    @staticmethod
    def plot(config, field_to_plot, level):
        """ Create a single plot using specs data
        Parameters:
            config: ConfigManager
            field_to_plot: tuple (data2d, dim1, dim2, field_name, plot_type, findex, map_params)
            level: int (optional)
        """
        # data2d, dim1, dim2, field_name, plot_type, findex, map_params = field_to_plot
        plot_type = field_to_plot[4] + 'plot'

        if plot_type == constants.yzplot:
            _single_yz_plot(config, field_to_plot)
        if plot_type == constants.xtplot:
            _single_xt_plot(config, field_to_plot)
        if plot_type == constants.txplot:
            _single_tx_plot(config, field_to_plot)
        if plot_type == constants.xyplot:
            _single_xy_plot(config, field_to_plot, level)
        if plot_type == constants.polarplot:
            _single_polar_plot(config, field_to_plot)
        if plot_type == constants.scplot:
            _single_scat_plot(config, field_to_plot)
        # TODO: for user defined functions you need to do the following:
        # elif plot_type == constants.myplot:
        #     self._myplot_subplot(config, field_to_plot)

    # Add access to the new methods
    def time_series_combined(self, config: ConfigManager, field_name: str, 
                           start_date=None, end_date=None, output_name=None):
        """Single plotter wrapper for plot_time_series_combined."""
        self.plot_time_series_combined(config, field_name, start_date, end_date, output_name)
        
    def composite_field(self, config: ConfigManager, primary_field: str, 
                      secondary_field: str, operation: str = 'add',
                      plot_type: str = 'xy', level: int = None, output_name: str = None):
        """Single plotter wrapper for plot_composite_field."""
        self.plot_composite_field(config, primary_field, secondary_field, 
                                operation, plot_type, level, output_name)


@dataclass()
class ComparisonPlotter:
    to_compare: list

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info("Start init")

    def comparison_plots(self, config: ConfigManager, field_to_plot: tuple, level: int = None):
        self.plot(config, field_to_plot, level)

    @staticmethod
    def plot(config, field_to_plot, level):
        """ Create a single plot using specs data
        Parameters:
            config: ConfigManager
            field_to_plot: tuple (data2d, dim1, dim2, field_name, plot_type, findex, map_params)
            level: int (optional)
        """
        # data2d, dim1, dim2, field_name, plot_type, findex, map_params = field_to_plot
        plot_type = field_to_plot[4] + 'plot'
        if plot_type not in ['xyplot', 'yzplot', 'polarplot', 'scplot']:
            plot_type = field_to_plot[2]

        if plot_type == constants.yzplot:
            _single_yz_plot(config, field_to_plot)
        elif plot_type == constants.xtplot:
            _single_xt_plot(config, field_to_plot)
        elif plot_type == constants.txplot:
            _single_tx_plot(config, field_to_plot)
        elif plot_type == constants.xyplot:
            _single_xy_plot(config, field_to_plot, level)
        elif plot_type == constants.polarplot:
            _single_polar_plot(config, field_to_plot)
        elif plot_type == constants.scplot:
            _single_scat_plot(config, field_to_plot)
        else:
            logger.error(f'{plot_type} is not implemented')
        # TODO: for user defined functions you need to do the following:
        # elif plot_type == constants.myplot:
        #     self._myplot_subplot(config, field_to_plot)

    # Add access to the new methods
    def time_series_combined(self, config: ConfigManager, field_name: str, 
                           start_date=None, end_date=None, output_name=None):
        """Comparison plotter wrapper for plot_time_series_combined."""
        self.plot_time_series_combined(config, field_name, start_date, end_date, output_name)
        
    def composite_field(self, config: ConfigManager, primary_field: str, 
                      secondary_field: str, operation: str = 'add',
                      plot_type: str = 'xy', level: int = None, output_name: str = None):
        """Comparison plotter wrapper for plot_composite_field."""
        self.plot_composite_field(config, primary_field, secondary_field, 
                                operation, plot_type, level, output_name)
