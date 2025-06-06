from .backends.matplotlib.xy_plot import MatplotlibXYPlotter
from .backends.matplotlib.yz_plot import MatplotlibYZPlotter
from .backends.matplotlib.xt_plot import MatplotlibXTPlotter
from .backends.matplotlib.tx_plot import MatplotlibTXPlotter
from .backends.matplotlib.scatter_plot import MatplotlibScatterPlotter
from .backends.hvplot.xy_plot import HvplotXYPlotter
from .backends.hvplot.xt_plot import HvplotXTPlotter
from .backends.hvplot.scatter_plot import HvplotScatterPlotter
from .backends.altair.xy_plot import AltairXYPlotter
from .backends.altair.xt_plot import AltairXTPlotter
from .backends.altair.scatter_plot import AltairScatterPlotter

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
            ("yz", "matplotlib"): MatplotlibYZPlotter,
            ("xt", "matplotlib"): MatplotlibXTPlotter,
            ("tx", "matplotlib"): MatplotlibXTPlotter,
            ("sc", "matplotlib"): MatplotlibScatterPlotter,
            ("xy", "hvplot"): HvplotXYPlotter,
            ("xt", "hvplot"): HvplotXTPlotter,
            ("sc", "hvplot"): HvplotScatterPlotter,
            ("xy", "altair"): AltairXYPlotter,
            ("xt", "altair"): AltairXTPlotter,
            ("sc", "altair"): AltairScatterPlotter,
            # Add other combinations as they are implemented
        }
        
        key = (plot_type, backend)
        print(f"Creating plotter for {plot_type} plottype with backend {backend}")
        if key in plotters:
            return plotters[key]()
        else:
            raise ValueError(f"No plotter available for plot_type={plot_type}, backend={backend}")
