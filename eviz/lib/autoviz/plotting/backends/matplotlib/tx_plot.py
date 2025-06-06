import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib import colors
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .base import MatplotlibBasePlotter
from eviz.lib.data.pipeline.reader import get_data_coords


class MatplotlibTXPlotter(MatplotlibBasePlotter):
    """Matplotlib implementation of TX (Hovmoller) plotting."""
    
    def __init__(self):
        super().__init__()
        self.fig = None
        self.ax = None
    
    def plot(self, config, data_to_plot):
        """Create a TX (Hovmoller) plot using Matplotlib.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created figure
        """
        data2d, _, _, field_name, plot_type, findex, fig = data_to_plot
        
        self.fig = fig
        
        if data2d is None:
            self.logger.warning("No data to plot")
            return fig
        
        self.logger.debug(f"=== TX PLOT DEBUG INFO ===")
        self.logger.debug(f"Field: {field_name}")
        self.logger.debug(f"Data shape: {data2d.shape}")
        self.logger.debug(f"Data dims: {data2d.dims}")
        self.logger.debug(f"Data coords: {list(data2d.coords.keys())}")
        
        ax_opts = config.ax_opts
        
        with mpl.rc_context(rc=ax_opts.get('rc_params', {})):
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
            
            self.create_clevs(field_name, ax_opts, data2d)
            self.logger.debug(f"Contour levels: {ax_opts['clevs']}")
            
            extend_value = "both"
            if ax_opts['clevs'][0] == 0:
                extend_value = "max"
            self.logger.debug(f"Extend value: {extend_value}")
            
            norm = colors.BoundaryNorm(ax_opts['clevs'], ncolors=256, clip=False)
            
            # vtimes = self._get_time_coordinates(data2d)
            vtimes = data2d.time.values.astype('datetime64[ms]').astype('O')

            self.logger.debug(f"Time coordinates shape: {len(vtimes)}")
            self.logger.debug(f"Time range: {vtimes[0]} to {vtimes[-1]}")
            
            # Get longitude coordinates
            lon_dim = config.get_model_dim_name('xc')
            lons = self._get_longitude_coordinates(data2d, lon_dim)

            self.logger.debug(f"Longitude coordinates shape: {len(lons)}")
            self.logger.debug(f"Longitude range: {lons[0]} to {lons[-1]}")
            
            # Process data for plotting
            data2d_reduced = self._process_data_for_hovmoller(data2d, vtimes, lons)
            self.logger.debug(f"Processed data shape: {data2d_reduced.shape}")
            self.logger.debug(f"Processed data range: {np.nanmin(data2d_reduced)} to {np.nanmax(data2d_reduced)}")
            
            # Set up the map in the top panel
            self._setup_map_panel(ax[0])
            
            # Add text to the map
            fig.plot_text(field_name=field_name, ax=ax[0], pid='tx', data=data2d, fontsize=8, loc='left')
            
            # Set time axis order
            if ax_opts.get('torder', False):
                self.logger.debug("Inverting y-axis for time order")
                ax[1].invert_yaxis()  # Reverse the time order
            
            # Create the hovmoller plot
            try:
                self.logger.debug("Attempting contourf plot")
                cfilled = ax[1].contourf(lons, vtimes, data2d_reduced, ax_opts['clevs'],
                                        norm=norm,
                                        cmap=ax_opts['use_cmap'], extend=extend_value)
                self.logger.debug("Contourf plot successful")
            except Exception as e:
                self.logger.error(f"Error creating contour plot: {e}")
                try:
                    self.logger.info("Falling back to pcolormesh")
                    lon_mesh, time_mesh = np.meshgrid(lons, vtimes)
                    self.logger.debug(f"Meshgrid shapes: lon_mesh={lon_mesh.shape}, time_mesh={time_mesh.shape}")
                    cfilled = ax[1].pcolormesh(lon_mesh, time_mesh, data2d_reduced,
                                            norm=norm, cmap=ax_opts['use_cmap'])
                    self.logger.debug("Pcolormesh plot successful")
                except Exception as e2:
                    self.logger.error(f"Error creating pcolormesh plot: {e2}")
                    # just show something
                    self.logger.info("Falling back to imshow")
                    cfilled = ax[1].imshow(data2d_reduced, aspect='auto', origin='lower',
                                        norm=norm, cmap=ax_opts['use_cmap'])
            
            # Set axis labels
            ax[1].set_xlabel("Longitude")
            ax[1].set_ylabel("Time")
            ax[1].grid(linestyle='dotted', linewidth=0.5)
            
            # Add contour lines if specified
            try:
                if ax_opts.get('line_contours', False):
                    self.logger.debug("Adding contour lines")
                    self.line_contours(fig, ax[1], ax_opts, lons, vtimes, data2d_reduced)
            except Exception as e:
                self.logger.error(f"Error adding contour lines: {e}")
            
            # Add colorbar
            cbar = fig.colorbar(cfilled, orientation='horizontal', pad=0.1, aspect=70,
                                extendrect=True)
            
            # Set colorbar label
            units = self._get_units_for_colorbar(data2d)
            cbar.set_label(units)
            self.logger.debug(f"Colorbar units: {units}")
            
            # Set x-axis ticks
            self._set_longitude_ticks(ax[1], lons)
            
            # Format y-axis (time) labels
            y_labels = ax[1].get_yticklabels()
            if len(y_labels) > 0:
                y_labels[0].set_visible(False)  # hide first label
            for i, label in enumerate(y_labels):
                label.set_rotation(45)
                label.set_ha('right')
            
            # Add grid if specified
            if ax_opts.get('add_grid', False):
                kwargs = {'linestyle': '-', 'linewidth': 2}
                ax[1].grid(**kwargs)
            
            # Adjust layout if needed
            if hasattr(fig, 'subplots') and fig.subplots != (1, 1):
                if hasattr(fig, 'squeeze_fig_aspect'):
                    fig.squeeze_fig_aspect(fig)
            
            # Store axes for later use
            self.ax = ax
        
        # Store the plot object
        self.plot_object = fig
        
        self.logger.debug("=== TX PLOT DEBUG END ===")
        return fig
    
    def _get_time_coordinates(self, data2d):
        """Get time coordinates from the data."""
        try:
            # Try to get time coordinates - this is critical for Hovmoller plots
            if 'time' in data2d.coords:
                time_coords = data2d.time.values
                self.logger.debug("Found 'time' coordinate")
            elif 'Time' in data2d.coords:
                time_coords = data2d.Time.values
                self.logger.debug("Found 'Time' coordinate")
            elif 't' in data2d.coords:
                time_coords = data2d.t.values
                self.logger.debug("Found 't' coordinate")
            else:
                # Look for time dimension in dims
                time_dim = None
                for dim in data2d.dims:
                    if 'time' in dim.lower():
                        time_dim = dim
                        break
                
                if time_dim:
                    time_coords = data2d[time_dim].values
                    self.logger.debug(f"Found time dimension: {time_dim}")
                else:
                    # Create dummy time coordinates
                    time_coords = np.arange(data2d.shape[0])
                    self.logger.warning("No time coordinate found, creating dummy coordinates")
            
            # Convert to datetime if needed
            if hasattr(time_coords[0], 'astype'):
                try:
                    time_coords = time_coords.astype('datetime64[ms]').astype('O')
                    self.logger.debug("Converted time coordinates to datetime")
                except:
                    self.logger.debug("Could not convert time coordinates to datetime")
            
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
                    lons = get_data_coords(data2d, lon_dim)
                    self.logger.debug(f"Got longitude coordinates using lon_dim: {lon_dim}")
                except ImportError:
                    self.logger.warning("Could not import get_data_coords")
                    lons = None
            else:
                lons = None
            
            if lons is None:
                if 'lon' in data2d.dims:
                    lons = data2d.lon.values
                    self.logger.debug("Using 'lon' dimension")
                elif 'longitude' in data2d.dims:
                    lons = data2d.longitude.values
                    self.logger.debug("Using 'longitude' dimension")
                elif 'x' in data2d.dims:
                    lons = data2d.x.values
                    self.logger.debug("Using 'x' dimension")
                else:
                    lons = np.arange(data2d.shape[-1])  # Use last dimension
                    self.logger.warning("No longitude coordinate found, creating dummy coordinates")
        except Exception as e:
            self.logger.error(f"Error getting longitude coordinates: {e}")
            lons = np.arange(data2d.shape[-1])
        
        return lons
    
    def _process_data_for_hovmoller(self, data2d, vtimes, lons):
        """Process data for Hovmoller plot."""
        self.logger.debug(f"Processing data for Hovmoller: input shape {data2d.shape}")
        
        try:
            if len(data2d.shape) > 2:
                self.logger.debug(f"Data has {len(data2d.shape)} dimensions")
                
                # Check if first dimension matches time
                if data2d.shape[0] == len(vtimes):
                    self.logger.debug("First dimension matches time length")
                    
                    # Check if last dimension matches longitude
                    if data2d.shape[-1] == len(lons):
                        self.logger.debug("Last dimension matches longitude length")
                        
                        if len(data2d.shape) == 3:
                            # Shape is (time, middle_dim, lon) - average over middle dimension
                            self.logger.debug("Averaging over middle dimension")
                            data2d_reduced = data2d.mean(axis=1)
                        else:
                            # More than 3 dimensions - average over all except first and last
                            axes_to_avg = tuple(range(1, len(data2d.shape) - 1))
                            self.logger.debug(f"Averaging over axes: {axes_to_avg}")
                            data2d_reduced = data2d.mean(axis=axes_to_avg)
                    else:
                        self.logger.warning(f"Last dimension {data2d.shape[-1]} doesn't match longitude length {len(lons)}")
                        # Try to find time and longitude dimensions by name
                        data2d_reduced = self._process_by_dimension_names(data2d, vtimes, lons)
                else:
                    self.logger.warning(f"First dimension {data2d.shape[0]} doesn't match time length {len(vtimes)}")
                    data2d_reduced = self._process_by_dimension_names(data2d, vtimes, lons)
            else:
                self.logger.debug("Data is 2D")
                data2d_reduced = data2d
                
                # Check if dimensions match
                if data2d.shape != (len(vtimes), len(lons)):
                    if data2d.shape == (len(lons), len(vtimes)):
                        self.logger.debug("Transposing 2D data")
                        data2d_reduced = data2d.T
                    else:
                        self.logger.warning(f"2D data shape {data2d.shape} doesn't match expected ({len(vtimes)}, {len(lons)})")
                        # Adjust coordinates to match data
                        vtimes = np.arange(data2d.shape[0])
                        lons = np.arange(data2d.shape[1])
                        
        except Exception as e:
            self.logger.error(f"Error processing data for Hovmoller plot: {e}")
            data2d_reduced = data2d
            if len(data2d.shape) >= 2:
                vtimes = np.arange(data2d.shape[0])
                lons = np.arange(data2d.shape[1])
        
        # Final shape check
        if hasattr(data2d_reduced, 'shape') and len(data2d_reduced.shape) >= 2:
            expected_shape = (len(vtimes), len(lons))
            actual_shape = data2d_reduced.shape
            if actual_shape != expected_shape:
                self.logger.warning(f"Final data shape {actual_shape} doesn't match expected {expected_shape}")
        
        self.logger.debug(f"Final processed data shape: {data2d_reduced.shape}")
        return data2d_reduced
    
    def _process_by_dimension_names(self, data2d, vtimes, lons):
        """Process data by finding time and longitude dimensions by name."""
        self.logger.debug("Processing data by dimension names")
        
        dim_names = list(data2d.dims)
        time_dim_idx = None
        lon_dim_idx = None
        
        # Find time dimension
        for i, dim in enumerate(dim_names):
            if dim.lower() in ['time', 't', 'TIME']:
                time_dim_idx = i
                self.logger.debug(f"Found time dimension '{dim}' at index {i}")
                break
        
        # Find longitude dimension
        for i, dim in enumerate(dim_names):
            if dim.lower() in ['lon', 'longitude', 'x']:
                lon_dim_idx = i
                self.logger.debug(f"Found longitude dimension '{dim}' at index {i}")
                break
        
        if time_dim_idx is not None and lon_dim_idx is not None:
            # Average over all other dimensions
            dims_to_avg = [i for i in range(len(dim_names))
                          if i != time_dim_idx and i != lon_dim_idx]
            
            self.logger.debug(f"Averaging over dimension indices: {dims_to_avg}")
            
            data2d_reduced = data2d.copy()
            for dim_idx in sorted(dims_to_avg, reverse=True):
                data2d_reduced = data2d_reduced.mean(axis=dim_idx)
            
            # Transpose if needed to get (time, lon) order
            if time_dim_idx > lon_dim_idx:
                self.logger.debug("Transposing to get (time, lon) order")
                data2d_reduced = data2d_reduced.T
        else:
            self.logger.warning("Could not find time and longitude dimensions by name")
            # Fallback: assume first dimension is time, average over middle dimensions
            if len(data2d.shape) > 2:
                axes_to_avg = tuple(range(1, len(data2d.shape) - 1))
                data2d_reduced = data2d.mean(axis=axes_to_avg)
            else:
                data2d_reduced = data2d
        
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
