import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FixedLocator
from matplotlib import colors
from .base import MatplotlibBasePlotter
from eviz.lib.data.pipeline.reader import get_data_coords


class MatplotlibTXPlotter(MatplotlibBasePlotter):
    """Matplotlib implementation of TX (Hovmoller) plotting."""
    
    def __init__(self):
        super().__init__()
        self.fig = None
        self.ax = None

    def plot2(self, config, data_to_plot):
        """Create a TX (Hovmoller) plot using Matplotlib.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created figure
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
            self.logger.debug(f"Field: {field_name}; Min:{dmin}; Max:{dmax}")

            self._create_clevs(field_name, ax_opts, data2d)
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
                self.logger.error(f"Error getting longitude coordinates: {e}")
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
                self.logger.error(f"Error processing data for Hovmoller plot: {e}")
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
                self.logger.error(f"Error creating contour plot: {e}")
                try:
                    self.logger.info("Falling back to pcolormesh")
                    lon_mesh, time_mesh = np.meshgrid(lons, vtimes)
                    cfilled = ax[1].pcolormesh(lon_mesh, time_mesh, data2d_reduced,
                                            norm=norm, cmap=ax_opts['use_cmap'])
                except Exception as e2:
                    self.logger.error(f"Error creating pcolormesh plot: {e2}")
                    # just show something
                    cfilled = ax[1].imshow(data2d_reduced, aspect='auto', origin='lower',
                                        norm=norm, cmap=ax_opts['use_cmap'])

            ax[1].set_xlabel("Longitude")
            ax[1].set_ylabel("Time")
            ax[1].grid(linestyle='dotted', linewidth=0.5)
        
            try:
                if ax_opts['line_contours']:
                    self._line_contours(fig, ax[1], ax_opts, lons, vtimes, data2d_reduced)
            except Exception as e:
                self.logger.error(f"Error adding contour lines: {e}")

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

        return fig

    def plot(self, config, data_to_plot):
        """Create a TX (Hovmoller) plot using Matplotlib.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created figure
        """
        data2d, _, _, field_name, plot_type, findex, fig = data_to_plot
        if data2d is None:
            return fig
        
        self.fig = fig
        
        ax_opts = config.ax_opts
        
        # Create a special gridspec layout for Hovmoller plots
        with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
            # IMPORTANT: Create a new figure instead of using the provided one
            # This is critical for Hovmoller plots which need a specific layout
            
            # Create a gridspec with two rows - small map on top, hovmoller plot below
            gs = mgridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 6], hspace=0.05)
            
            # Create two axes - one for the map, one for the hovmoller
            ax = []
            ax.append(fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=180)))
            ax.append(fig.add_subplot(gs[1, 0]))
            fig.set_size_inches(12, 10, forward=True)

            ax_opts = fig.update_ax_opts(field_name, ax, 'tx')
            
            dmin = data2d.min(skipna=True).values
            dmax = data2d.max(skipna=True).values
            self.logger.debug(f"Field: {field_name}; Min:{dmin}; Max:{dmax}")
            
            self._create_clevs(field_name, ax_opts, data2d)
            
            extend_value = "both"
            if ax_opts['clevs'][0] == 0:
                extend_value = "max"
            
            norm = colors.BoundaryNorm(ax_opts['clevs'], ncolors=256, clip=False)
            
            vtimes = self._get_time_coordinates(data2d)
            
            lon_dim = config.get_model_dim_name('xc')
            lons = self._get_longitude_coordinates(data2d, lon_dim)
            
            data2d_reduced = self._process_data_for_hovmoller(data2d, vtimes, lons)
            
            # Set up the map in the top panel
            self._setup_map_panel(ax[0])            
            fig.plot_text(field_name=field_name, ax=ax[0], pid='tx', data=data2d, fontsize=8, loc='left')
            
            if ax_opts.get('torder', False):
                ax[1].invert_yaxis()  # Reverse the time order
            
            try:
                cfilled = ax[1].contourf(lons, vtimes, data2d_reduced, ax_opts['clevs'],
                                        norm=norm,
                                        cmap=ax_opts['use_cmap'], extend=extend_value)
            except Exception as e:
                self.logger.error(f"Error creating contour plot: {e}")
                try:
                    self.logger.info("Falling back to pcolormesh")
                    lon_mesh, time_mesh = np.meshgrid(lons, vtimes)
                    cfilled = ax[1].pcolormesh(lon_mesh, time_mesh, data2d_reduced,
                                            norm=norm, cmap=ax_opts['use_cmap'])
                except Exception as e2:
                    self.logger.error(f"Error creating pcolormesh plot: {e2}")
                    # just show something
                    cfilled = ax[1].imshow(data2d_reduced, aspect='auto', origin='lower',
                                        norm=norm, cmap=ax_opts['use_cmap'])
            
            ax[1].set_xlabel("Longitude")
            ax[1].set_ylabel("Time")
            ax[1].grid(linestyle='dotted', linewidth=0.5)
            
            try:
                if ax_opts.get('line_contours', False):
                    self.line_contours(fig, ax[1], ax_opts, lons, vtimes, data2d_reduced)
            except Exception as e:
                self.logger.error(f"Error adding contour lines: {e}")
            
            cbar = fig.colorbar(cfilled, orientation='horizontal', pad=0.1, aspect=70,
                                extendrect=True)
            
            units = self._get_units_for_colorbar(data2d)
            cbar.set_label(units)
            
            self._set_longitude_ticks(ax[1], lons)
            
            y_labels = ax[1].get_yticklabels()
            if len(y_labels) > 0:
                y_labels[0].set_visible(False)  # hide first label
            for i, label in enumerate(y_labels):
                label.set_rotation(45)
                label.set_ha('right')
            
            if ax_opts.get('add_grid', False):
                kwargs = {'linestyle': '-', 'linewidth': 2}
                ax[1].grid(**kwargs)
            
            fig.tight_layout()
                
        return fig
    
    def _get_time_coordinates(self, data2d):
        """Get time coordinates from the data."""
        try:
            # Try to get time coordinates
            if 'time' in data2d.coords:
                time_coords = data2d.time.values
            elif 'Time' in data2d.coords:
                time_coords = data2d.Time.values
            elif 't' in data2d.coords:
                time_coords = data2d.t.values
            else:
                # Look for time dimension in dims
                time_dim = None
                for dim in data2d.dims:
                    if 'time' in dim.lower():
                        time_dim = dim
                        break
                
                if time_dim:
                    time_coords = data2d[time_dim].values
                else:
                    # Create dummy time coordinates
                    time_coords = np.arange(data2d.shape[0])
            
            # Convert to datetime if needed
            if hasattr(time_coords[0], 'astype'):
                try:
                    time_coords = time_coords.astype('datetime64[ms]').astype('O')
                except:
                    pass
            
            return time_coords
            
        except Exception as e:
            self.logger.error(f"Error getting time coordinates: {e}")
            # Return dummy coordinates
            return np.arange(data2d.shape[0])
    
    def _get_longitude_coordinates(self, data2d, lon_dim):
        """Get longitude coordinates from the data."""
        try:
            if lon_dim:
                try:
                    from eviz.lib.data.pipeline.reader import get_data_coords
                    lons = get_data_coords(data2d, lon_dim)
                except ImportError:
                    self.logger.warning("Could not import get_data_coords")
                    lons = None
            else:
                lons = None
            
            if lons is None:
                if 'lon' in data2d.dims:
                    lons = data2d.lon.values
                elif 'longitude' in data2d.dims:
                    lons = data2d.longitude.values
                elif 'x' in data2d.dims:
                    lons = data2d.x.values
                else:
                    lons = np.arange(data2d.shape[-1])  # Use last dimension
        except Exception as e:
            self.logger.error(f"Error getting longitude coordinates: {e}")
            lons = np.arange(data2d.shape[-1])
        
        return lons
    
    def _process_data_for_hovmoller(self, data2d, vtimes, lons):
        """Process data for Hovmoller plot."""
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
            self.logger.error(f"Error processing data for Hovmoller plot: {e}")
            data2d_reduced = data2d
            if len(data2d.shape) >= 2:
                vtimes = np.arange(data2d.shape[0])
                lons = np.arange(data2d.shape[1])
        
        if hasattr(data2d_reduced, 'shape') and len(data2d_reduced.shape) >= 2:
            if data2d_reduced.shape[0] != len(vtimes) or data2d_reduced.shape[1] != len(lons):
                vtimes = np.arange(data2d_reduced.shape[0])
                lons = np.arange(data2d_reduced.shape[1])
        
        return data2d_reduced
    
    def _get_units_for_colorbar(self, data2d):
        """Get units for the colorbar."""
        units = "n.a."
        if hasattr(data2d, 'attrs') and 'units' in data2d.attrs:
            units = data2d.attrs['units']
        elif hasattr(data2d, 'units'):
            units = data2d.units
        return units
    
    def _setup_map_panel(self, ax):
        """Set up the map panel at the top of the Hovmoller plot."""
        # Set map extent
        ax.set_extent([0, 357.5, 35, 65], ccrs.PlateCarree(central_longitude=180))
        
        # Set latitude ticks
        ax.set_yticks([40, 60])
        ax.set_yticklabels([u'40\N{DEGREE SIGN}N', u'60\N{DEGREE SIGN}N'])
        
        # Set longitude ticks
        ax.set_xticks([-180, -90, 0, 90, 180])
        x_tick_labels = [u'0\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}E',
                        u'180\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}W',
                        u'0\N{DEGREE SIGN}E']
        ax.set_xticklabels(x_tick_labels)
        
        # Add grid
        ax.grid(linestyle='dotted', linewidth=2)
        
        # Add coastlines and lakes
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
        ax.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)
    
    def _set_longitude_ticks(self, ax, lons):
        """Set longitude ticks for the Hovmoller plot."""
        x_tick_labels = [u'0\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}E',
                        u'180\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}W',
                        u'0\N{DEGREE SIGN}E']
        
        if len(lons) > 0:
            if lons[0] <= -179:
                ax.set_xticks([-180, -90, 0, 90, 180])
            else:
                ax.set_xticks([0, 90, 180, 270, 360])
        
        ax.set_xticklabels(x_tick_labels, fontsize=10)
    
    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        if self.fig is not None:
            self.fig.savefig(filename, **kwargs)
            self.logger.info(f"Saved plot to {filename}")
        else:
            self.logger.warning("No figure to save")
    
    def show(self):
        """Display the plot."""
        if self.fig is not None:
            plt.figure(self.fig.number)
            plt.show()
        else:
            self.logger.warning("No figure to show")
