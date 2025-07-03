import numpy as np
import matplotlib as mpl
import logging
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import eviz.lib.autoviz.utils as pu
from .base import MatplotlibBasePlotter


class MatplotlibYZPlotter(MatplotlibBasePlotter):
    """Matplotlib implementation of YZ (zonal-mean) plotting."""
    def __init__(self):
        super().__init__()
        self.fig = None
        self.ax = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def plot(self, config, data_to_plot):
        """Create a YZ plot using Matplotlib.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created figure
        """
        data2d, x, y, field_name, plot_type, findex, fig = data_to_plot

        if data2d is None:
            return fig

        self.source_name = config.source_names[config.ds_index]
        self.units = self.get_units(config, 
                                    field_name, 
                                    data2d, 
                                    findex)
        self.fig = fig

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

        ax_opts = fig.update_ax_opts(field_name, ax, 'yz')
        fig.plot_text(field_name, ax, 'yz', level=None, data=data2d)

        # Determine the vertical coordinate and its units
        zc = config.get_model_dim_name('zc')
        try:
            vertical_units = data2d.coords[zc].attrs['units']
        except KeyError:
            vertical_units = 'n.a.'

        if isinstance(ax, list):
            for single_ax in ax:
                self._plot_yz_data(config, single_ax, data2d,
                                   x, y, field_name, fig, ax_opts, vertical_units,
                                   plot_type, findex)
        else:
            self._plot_yz_data(config, ax, data2d,
                               x, y, field_name, fig, ax_opts, vertical_units,
                               plot_type, findex)
        # reset rc params to default
        # matplotlib.rcParams.update(matplotlib.rcParamsDef   
        return fig

    def _plot_yz_data(self, config, ax, data2d,
                      x, y, field_name, fig, ax_opts, vertical_units,
                      plot_type, findex):
        """Helper function to plot YZ data on a single axes."""
        with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
            source_name = config.source_names[config.ds_index]

            # Check if profile plot is requested
            prof_dim = None
            if ax_opts['profile_dim']:
                self.logger.debug(f"Creating profile plot for {field_name}")
                prof_dim = ax_opts['profile_dim']
                dep_var = None
                if prof_dim == 'yc':
                    dep_var = 'zc'
                if prof_dim == 'zc':
                    dep_var = 'yc'
                prof_dim = ax_opts['profile_dim']
                data2d = data2d.mean(dim=config.get_model_dim_name(prof_dim))
                
                # TODO: Put in rcParams
                line_color = None
                line_style = '-'
                line_width = 1.5
                marker = None
                if config.overlay:
                    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'red',]
                    styles = ['-', '--', '-.', ':']
                    
                    dataset_index = 0
                    if hasattr(config, 'current_dataset_index'):
                        dataset_index = config.current_dataset_index
                    
                    line_color = colors[dataset_index]
                    # line_style = styles[(dataset_index // len(colors)) % len(styles)]
                    line_style = styles[dataset_index]

                    label = None
                    if hasattr(config, 'a_list') and hasattr(config, 'b_list'):
                        if findex in config.a_list:
                            list_idx = config.a_list.index(findex)
                            list_name = 'a_list'
                        elif findex in config.b_list:
                            list_idx = config.b_list.index(findex)
                            list_name = 'b_list'
                        else:
                            list_idx = 0
                            list_name = 'unknown'
                        
                        # Try to get a meaningful label
                        label = config.get_file_exp_name(findex) or config.get_file_exp_id(findex)
                        if not label:
                            label = f"Dataset {list_name}[{list_idx}]"
                    else:
                        label = f"Dataset {dataset_index}"
                
                if config.overlay and label:
                    self._plot_profile(config, data2d, fig, ax, ax_opts, (prof_dim, dep_var), 
                                    color=line_color, linestyle=line_style, linewidth=line_width, 
                                    marker=marker, label=label)
                else:
                    self._plot_profile(config, data2d, fig, ax, ax_opts, (prof_dim, dep_var))

                if config.overlay:
                    all_plotted = False
                    if hasattr(config, 'current_dataset_index') and hasattr(config, 'total_datasets'):
                        all_plotted = config.current_dataset_index == config.total_datasets - 1            
                    
                    if all_plotted or ax_opts.get('force_legend', False):
                        legend = ax.legend(loc='best', fontsize=pu.legend_font_size(fig.subplots))                
                        frame = legend.get_frame()
                        frame.set_alpha(0.7)
                        frame.set_edgecolor('gray')

                if dep_var == 'zc':
                    ax.set_xlabel(self.units)
                    ax.set_ylabel("Pressure (" + vertical_units + ")",
                                size=pu.axes_label_font_size(fig.subplots))

            else:
                cfilled = self.filled_contours(config, field_name, ax, x, y, data2d)

                ylabels = ax.get_yticklabels()
                for label in ylabels:
                    label.set_fontsize(pu.axis_tick_font_size(fig.subplots))

                xlabels = ax.get_xticklabels()
                for label in xlabels:
                    label.set_fontsize(pu.axis_tick_font_size(fig.subplots))

                self._set_ax_ranges(config, field_name, fig, ax, ax_opts, y, vertical_units)

                if ax_opts.get('line_contours', False):
                    self.line_contours(fig, ax, ax_opts, x, y, data2d)

                self.set_colorbar(config, cfilled, fig, ax, ax_opts, findex, field_name, data2d)

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
                    self.logger.warning(f"Error getting field name: {e}")
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

    @staticmethod
    def _plot_profile(config, data2d, fig, ax, ax_opts, ax_dims,
                      color=None, linestyle='-', linewidth=1.5, marker=None, label=None):
        """Plot a vertical profile."""
        if ax_dims[0] in ('zc', 'yc'):
            other_dim = config.get_model_dim_name(ax_dims[1])
            y_coord = config.get_model_dim_name('yc' if ax_dims[0] == 'zc' else 'zc')
            
            y0 = data2d.coords[other_dim][0].values
            y1 = data2d.coords[other_dim][-1].values
            
            if color or label:
                ax.plot(data2d, data2d.coords[y_coord],
                       color=color, linestyle=linestyle, linewidth=linewidth, 
                       marker=marker, label=label)
            else:
                ax.plot(data2d, data2d.coords[y_coord])
            
            ax.set_ylim(y0, y1)
        
        ax.set_yscale(ax_opts['zscale'])
        ax.yaxis.set_minor_formatter(NullFormatter())
        
        # Set y-axis ticks for linear scale
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

    def _set_ax_ranges(self, config, field_name, fig, ax, ax_opts, y, units):
        """Set axis ranges and scales for YZ plots."""
        # Define standard pressure levels
        y_ranges = np.array([1000, 700, 500, 300, 200, 100])
        
        # Convert to Pa if units are Pa
        if units == "Pa":
            y_ranges = y_ranges * 100
            if y.min() <= 1000.0:
                y_ranges = np.append(y_ranges, np.array([70, 50, 30, 20, 10]) * 100)
            if y.min() <= 20.:
                y_ranges = np.append(y_ranges,
                                    np.array([7, 5, 3, 2, 1, .7, .5, .3, .2, .1]) * 100)
            if y_ranges[-1] != y.min():
                y_ranges = np.append(y_ranges, y.min())
        else:  # Assume hPa (mb)
            if y.min() <= 10.0:
                y_ranges = np.append(y_ranges, np.array([70, 50, 30, 20, 10]))
            if y.min() <= 0.2:
                y_ranges = np.append(y_ranges, np.array([7, 5, 3, 2, 1, .7, .5, .3, .2, .1]))
            if y_ranges[-1] != y.min():
                y_ranges = np.append(y_ranges, y.min())
        
        # Get vertical range from config or use default
        lo_z, hi_z = np.max(y_ranges), np.min(y_ranges)
        if 'zrange' in config.spec_data[field_name]['yzplot']:
            if config.spec_data[field_name]['yzplot']['zrange']:
                lo_z = config.spec_data[field_name]['yzplot']['zrange'][0]
                hi_z = config.spec_data[field_name]['yzplot']['zrange'][1]
                if hi_z >= lo_z:
                    self.logger.error(
                        f"Upper level value ({hi_z}) must be less than low level value ({lo_z})")
        
        # Set x-axis ticks for latitude
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
        ax.tick_params(width=3, length=6)
        
        # Set y-axis limits and scale
        ax.set_ylim(lo_z, hi_z)
        ax.set_yscale(ax_opts['zscale'])
        ax.yaxis.set_minor_formatter(NullFormatter())
        
        # Set y-axis formatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%3.1f'))
        
        # Set y-axis label
        ax.set_ylabel(f"Pressure ({units})", size=pu.axes_label_font_size(fig.subplots))
        
        # Add grid if requested
        if ax_opts['add_grid']:
            ax.grid()
    