from dataclasses import dataclass
import logging
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
import cartopy.crs as ccrs
import networkx as nx

import eviz.lib.autoviz.utils as pu
import eviz.lib.utils as u
from eviz.lib.config.config_manager import ConfigManager


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
