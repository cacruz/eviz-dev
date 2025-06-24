import numpy as np
import matplotlib as mpl
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error
import eviz.lib.autoviz.utils as pu
from eviz.lib.autoviz.base import MatplotlibBasePlotter


class MatplotlibXTPlotter(MatplotlibBasePlotter):
    """Matplotlib implementation of XT (time-series) plotting."""
    def __init__(self):
        super().__init__()
        self.fig = None
        self.ax = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def plot(self, config, data_to_plot):
        """Create an XT plot using Matplotlib.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created figure
        """
        data2d, _, _, field_name, plot_type, findex, fig = data_to_plot
        
        self.fig = fig
        
        ax_opts = config.ax_opts
        
        if not config.compare and not config.compare_diff and not config.overlay:
            fig.set_axes()
        
        ax_temp = fig.get_axes()
        axes_shape = fig.subplots
        
        if axes_shape == (3, 1):
            if ax_opts['is_diff_field']:
                self.ax = ax_temp[2]
            else:
                self.ax = ax_temp[config.axindex]
        elif axes_shape == (2, 2):
            if ax_opts['is_diff_field']:
                self.ax = ax_temp[2]
                if config.ax_opts['add_extra_field_type']:
                    self.ax = ax_temp[3]
            else:
                self.ax = ax_temp[config.axindex]
        elif axes_shape == (1, 2) or axes_shape == (1, 3):
            if isinstance(ax_temp, list):
                self.ax = ax_temp[config.axindex]
            else:
                self.ax = ax_temp
        else:
            self.ax = ax_temp[0]
        
        if data2d is None:
            return fig
        
        ax_opts = fig.update_ax_opts(field_name, self.ax, 'xt', level=0)
        fig.plot_text(field_name, self.ax, 'xt', data=data2d)
        
        self._plot_xt_data(config, self.ax, ax_opts, fig, data2d, field_name, findex)
        
        # Handle overlay mode
        if config.overlay:
            # Add legend if this is the last dataset or if forced
            all_plotted = getattr(config, 'current_dataset_index', 0) == getattr(config, 'total_datasets', 1) - 1
            if all_plotted or ax_opts.get('force_legend', False):
                legend = self.ax.legend(loc='best', fontsize=self._legend_font_size(fig.subplots))
                frame = legend.get_frame()
                frame.set_alpha(0.7)
                frame.set_edgecolor('gray')
        
        # Handle title for comparison plots
        if config.compare_diff:
            title_text = config.spec_data[field_name].get('name', field_name)
            fig.suptitle_eviz(title_text, 
                            fontweight='bold', fontstyle='italic',
                            fontsize=self._image_font_size(fig.subplots))
        elif config.compare:
            title_text = config.map_params[findex].get('field', 'No name')
            fig.suptitle_eviz(title_text, 
                            fontweight='bold', fontstyle='italic',
                            fontsize=self._image_font_size(fig.subplots))
            
            if config.add_logo:
                self._add_logo_ax(fig, desired_width_ratio=0.05)
        
        self.plot_object = fig
        
        return fig
    
    def _plot_xt_data(self, config, ax, ax_opts, fig, data2d, field_name, findex):
        """Helper method that plots the time series (xt) data."""        
        with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
            dmin = data2d.min(skipna=True).values
            dmax = data2d.max(skipna=True).values
            self.logger.debug(f"dmin: {dmin}, dmax: {dmax}")
            
            # Get time coordinates
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
                if hasattr(time_coords[0], '_cftime') or str(type(time_coords[0])).find('cftime') >= 0:
                    try:
                        # Convert to pandas datetime
                        time_coords = pd.to_datetime([str(t) for t in time_coords])
                    except Exception as e:
                        self.logger.warning(f"Error converting cftime to pandas datetime: {e}")
                        time_coords = np.arange(len(data2d))
                
            except Exception as e:
                self.logger.warning(f"Error getting time coordinates: {e}")
                time_coords = np.arange(len(data2d))
            
            t0 = time_coords[0]
            t1 = time_coords[-1]
            
            # Handle rolling mean if specified
            window_size = 0
            # Check if xtplot configuration exists
            if field_name in config.spec_data and 'xtplot' in config.spec_data[field_name]:
                if 'mean_type' in config.spec_data[field_name]['xtplot']:
                    if config.spec_data[field_name]['xtplot']['mean_type'] == 'rolling':
                        if 'window_size' in config.spec_data[field_name]['xtplot']:
                            window_size = config.spec_data[field_name]['xtplot']['window_size']
            
            # Determine plot style for overlay mode
            line_color = None
            line_style = '-'
            line_width = 1.5
            marker = None
            label = None
            
            if config.overlay:
                colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'red']
                styles = ['-', '--', '-.', ':']
                
                dataset_index = 0
                if hasattr(config, 'current_dataset_index'):
                    dataset_index = config.current_dataset_index
                
                line_color = colors[dataset_index % len(colors)]
                line_style = styles[dataset_index % len(styles)]
                
                # Create label for legend
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
                    
                    label = config.get_file_exp_name(findex) or config.get_file_exp_id(findex)
                    if not label:
                        label = f"Dataset {list_name}[{list_idx}]"
                else:
                    label = f"Dataset {dataset_index}"
            
            # Plot the time series
            if window_size > 0 and len(data2d) > 2 * window_size:
                end_idx = max(0, len(time_coords) - window_size - 1)
                if line_color or label:
                    ax.plot(time_coords[window_size:end_idx], data2d[window_size:end_idx],
                        color=line_color, linestyle=line_style, linewidth=line_width, 
                        marker=marker, label=label)
                else:
                    ax.plot(time_coords[window_size:end_idx], data2d[window_size:end_idx])
            else:
                if line_color or label:
                    ax.plot(time_coords, data2d,
                        color=line_color, linestyle=line_style, linewidth=line_width, 
                        marker=marker, label=label)
                else:
                    ax.plot(time_coords, data2d)
            
            # Add trend line if specified
            # Check if xtplot configuration exists
            if field_name in config.spec_data and 'xtplot' in config.spec_data[field_name]:
                if 'add_trend' in config.spec_data[field_name]['xtplot']:
                    self.logger.debug('Adding trend')
                    if config.spec_data[field_name]['xtplot']['add_trend']:
                        try:
                            if isinstance(t0, (pd.Timestamp, np.datetime64)):
                                time_numeric = (time_coords - t0).astype('timedelta64[D]').astype(float)
                            else:
                                time_numeric = np.arange(len(time_coords))
                            
                            # Determine polynomial degree
                            if 'trend_polyfit' in config.spec_data[field_name]['xtplot']:
                                degree = config.spec_data[field_name]['xtplot']['trend_polyfit']
                            else:
                                errors = []
                                for degree in range(1, 6):
                                    coeffs = np.polyfit(time_numeric, data2d, degree)
                                    y_fit = np.polyval(coeffs, time_numeric)
                                    mse = mean_squared_error(data2d, y_fit)
                                    errors.append(mse)
                                best_degree = np.argmin(errors) + 1
                                degree = best_degree
                            
                            self.logger.debug(f' -- polynomial degree: {degree}')
                            coeffs = np.polyfit(time_numeric, data2d, degree)
                            trend_poly = np.polyval(coeffs, time_numeric)
                            ax.plot(time_coords, trend_poly, color="red", linewidth=1)
                        except Exception as e:
                            self.logger.warning(f"Error calculating trend: {e}")
            
            # Format the axes
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
            fig.autofmt_xdate()
            ax.set_xlim(t0, t1)
            
            # Set y-axis limits
            try:
                davg = 0.5 * (abs(dmin - dmax))
                ax.set_ylim([dmin - davg, dmax + davg])
            except Exception as e:
                self.logger.warning(f"Error setting y limits: {e}")
            
            # Set y-axis label (units)
            try:
                source_name = config.source_names[config.ds_index]
                
                if 'units' in config.spec_data[field_name]:
                    units = config.spec_data[field_name]['units']
                else:
                    units = getattr(data2d, 'units', None)
                    
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
                self.logger.warning(f"Error getting units for {field_name}: {e}")
                units = 'n.a.'
            
            ax.set_ylabel(units)
            
            # Add grid if specified
            if ax_opts['add_grid']:
                ax.grid()
    