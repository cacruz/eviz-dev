from .backends.matplotlib.xy_plot import MatplotlibXYPlotter
from .backends.matplotlib.yz_plot import MatplotlibYZPlotter
from .backends.matplotlib.xt_plot import MatplotlibXTPlotter
from .backends.matplotlib.tx_plot import MatplotlibTXPlotter
from .backends.matplotlib.polar_plot import MatplotlibPolarPlotter
from .backends.matplotlib.scatter_plot import MatplotlibScatterPlotter
from .backends.matplotlib.metric_plot import MatplotlibMetricPlotter
from .backends.matplotlib.box_plot import MatplotlibBoxPlotter
from .backends.hvplot.xy_plot import HvplotXYPlotter
from .backends.hvplot.xt_plot import HvplotXTPlotter
from .backends.hvplot.scatter_plot import HvplotScatterPlotter
from .backends.hvplot.box_plot import HvplotBoxPlotter
from .backends.hvplot.line_plot import HvplotLinePlotter
from .backends.hvplot.metric_plot import HvplotMetricPlotter
from .backends.altair.xy_plot import AltairXYPlotter
from .backends.altair.xt_plot import AltairXTPlotter
from .backends.altair.scatter_plot import AltairScatterPlotter
from .backends.altair.metric_plot import AltairMetricPlotter


class PlotterFactory:
    """Factory for creating appropriate plotters."""
    
    @staticmethod
    def create_plotter(plot_type, backend="matplotlib"):
        """Create a plotter for the given plot type and backend.
        
        Args:
            plot_type: Type of plot ('xy', 'yz', 'xt', 'sc', etc.)
            backend: Backend to use ('matplotlib', 'hvplot', 'altair')
            
        Returns:
            An instance of the appropriate plotter
            
        Raises:
            ValueError: If no plotter is available for the given plot type and backend
        """
        # Dictionary mapping (plot_type, backend) to plotter class
        plotters = {
            ("xy", "matplotlib"): MatplotlibXYPlotter,
            ("xt", "matplotlib"): MatplotlibXTPlotter,
            ("sc", "matplotlib"): MatplotlibScatterPlotter,
            ("yz", "matplotlib"): MatplotlibYZPlotter,
            ("tx", "matplotlib"): MatplotlibTXPlotter,
            ("polar", "matplotlib"): MatplotlibPolarPlotter,
            ("corr", "matplotlib"): MatplotlibMetricPlotter,
            ("box", "matplotlib"): MatplotlibBoxPlotter,

            ("xy", "hvplot"): HvplotXYPlotter,
            ("xt", "hvplot"): HvplotXTPlotter,
            ("sc", "hvplot"): HvplotScatterPlotter,
            ("corr", "hvplot"): HvplotMetricPlotter,
            ("box", "hvplot"): HvplotBoxPlotter,
            ("line", "hvplot"): HvplotLinePlotter,

            ("xy", "altair"): AltairXYPlotter,
            ("xt", "altair"): AltairXTPlotter,
            ("sc", "altair"): AltairScatterPlotter,
            ("corr", "altair"): AltairMetricPlotter,

            # Add other combinations as they are implemented
        }
        
        key = (plot_type, backend)
        if key in plotters:
            return plotters[key]()
        else:
            raise ValueError(f"No plotter available for plot_type={plot_type}, backend={backend}")
