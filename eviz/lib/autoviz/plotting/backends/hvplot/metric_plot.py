import numpy as np
import logging
import holoviews as hv
import hvplot.xarray  # register the hvplot method with xarray objects
import hvplot.pandas  # noqa
import xarray as xr
from eviz.lib.autoviz.plotting.base import XYPlotter


class HvplotMetricPlotter(XYPlotter):
    """HvPlot implementation for metric visualization (e.g., correlation maps)."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._apply_numpy_compatibility_patch()

        try:
            hv.extension('bokeh')
        except Exception as e:
            self.logger.warning(f"Could not initialize HoloViews/hvplot extensions: {e}")   

    def plot(self, config, data_to_plot):
        """Create an interactive correlation map using HvPlot.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
                If data2d is a tuple of two DataArrays, compute correlation between them
                Otherwise, assume data2d is already a correlation map
        
        Returns:
            The created HvPlot object
        """
        # Check if we need to compute correlation or if it's already computed
        if isinstance(data_to_plot[0], tuple) and len(data_to_plot[0]) == 2:
            # We have two datasets to correlate
            data1, data2 = data_to_plot[0]
            self.logger.debug("Computing correlation between two datasets")
            data2d = self.compute_correlation_map(data1, data2)
            if data2d is None:
                self.logger.error("Failed to compute correlation map")
                return None
                
            # Reconstruct data_to_plot with the correlation map
            x, y = data_to_plot[1], data_to_plot[2]
            field_name = data_to_plot[3] + "_correlation"
            plot_type, findex, fig = data_to_plot[4], data_to_plot[5], data_to_plot[6]
            data_to_plot = (data2d, x, y, field_name, plot_type, findex, fig)
        else:
            # Assume data2d is already a correlation map
            data2d = data_to_plot[0]
        
        if data2d is None:
            return None

        ax_opts = config.ax_opts
        
        # Use a diverging colormap for correlation
        cmap = ax_opts.get('use_cmap', 'RdBu_r')
        
        field_name = data_to_plot[3]
        title = field_name
        if hasattr(config, 'spec_data') and \
                field_name in config.spec_data and \
                'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        units = "dimensionless"
        
        try:
            # Check if datashader is available
            import datashader
            self.logger.debug("Datashader is available")
            x_dim = config.get_model_dim_name('xc') or 'lon'
            y_dim = config.get_model_dim_name('yc') or 'lat'
            
            plot_opts = {
                'cmap': cmap,
                'title': title,
                'width': 800,
                'height': 500,
                'colorbar': True,
                'clabel': units,
                'clim': (-1, 1),  # Correlation ranges from -1 to 1
                'tools': ['pan', 'wheel_zoom', 'box_zoom', 'reset', 'hover'],
                'rasterize': True  # For better performance with large datasets
            }
            
            if x_dim in data2d.dims:
                plot_opts['x'] = x_dim
            if y_dim in data2d.dims:
                plot_opts['y'] = y_dim
            
            plot = data2d.hvplot(**plot_opts)
            
            self.logger.debug("Successfully created correlation map")
            self.plot_object = plot
            
            return plot

        except ImportError as e:
            self.logger.warning(f"Datashader not available: {e}. "
                                f"Using HoloViews without datashader.")

        except Exception as e:
            self.logger.error(f"Error creating correlation map: {e}")
            
        # Try using HoloViews directly as a fallback
        try:
            self.logger.info("Using HoloViews QuadMesh instead of Image for irregularly sampled data")
            
            if hasattr(data2d, 'values'):
                z_values = data2d.values
            else:
                z_values = np.array(data2d)
            
            x = data_to_plot[1]
            y = data_to_plot[2]
            x_values = x.values if hasattr(x, 'values') else np.array(x)
            y_values = y.values if hasattr(y, 'values') else np.array(y)
            
            # Use QuadMesh instead of Image for irregularly sampled data
            quadmesh = hv.QuadMesh((x_values, y_values, z_values), 
                                kdims=[x_dim, y_dim], 
                                vdims=[field_name])
            
            plot = quadmesh.opts(
                cmap=cmap,
                colorbar=True,
                title=title,
                width=800,
                height=500,
                tools=['hover'],
                xlabel=x_dim,
                ylabel=y_dim,
                clabel=units,
                clim=(-1, 1)
            )
            
            self.plot_object = plot
            return plot
            
        except Exception as e2:
            self.logger.error(f"Alternative approach also failed: {e2}")
            return None

    def _apply_numpy_compatibility_patch(self):
        """Apply compatibility patch for NumPy 1.20+ with older HoloViews/hvplot."""
        try:
            if not hasattr(np, 'bool'):
                self.logger.debug("Applying NumPy compatibility patch for bool")
                np.bool = bool
            
            # Add other deprecated NumPy aliases that might be needed
            if not hasattr(np, 'int'):
                np.int = int
            if not hasattr(np, 'float'):
                np.float = float
            if not hasattr(np, 'complex'):
                np.complex = complex
            if not hasattr(np, 'object'):
                np.object = object
            if not hasattr(np, 'str'):
                np.str = str
        except Exception as e:
            self.logger.warning(f"Failed to apply NumPy compatibility patch: {e}")

    @staticmethod
    def compute_correlation_map(data1, data2):
        """Compute correlation using Dask for parallel processing."""

        # Convert to dask arrays if they aren't already
        if not data1.chunks:
            data1 = data1.chunk({'time': -1, 'lat': 'auto', 'lon': 'auto'})
        if not data2.chunks:
            data2 = data2.chunk({'time': -1, 'lat': 'auto', 'lon': 'auto'})

        time_dim = data1.dims[0]
        corr = xr.corr(data1, data2, dim=time_dim)
        result = corr.compute()
        return result

    def show(self):
        """Display the plot."""
        pass

    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        pass
