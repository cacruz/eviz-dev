from eviz.lib.autoviz.plotting.backends.altair.xt_plot import AltairXTPlotter
from eviz.lib.autoviz.plotting.backends.matplotlib.xt_plot import MatplotlibXTPlotter
from .backends.matplotlib.xy_plot import MatplotlibXYPlotter
from .backends.hvplot.xy_plot import HvplotXYPlotter
from .backends.altair.xy_plot import AltairXYPlotter

class PlotterFactory:
    """Factory for creating appropriate plotters."""
    
    @staticmethod
    def create_plotter(plot_type, backend="matplotlib"):
        """Create a plotter for the given plot type and backend.
        
        Args:
            plot_type: Type of plot ('xy', 'yz', 'xt', etc.)
            backend: Backend to use ('matplotlib', 'hvplot', 'altair')
            
        Returns:
            An instance of the appropriate plotter
            
        Raises:
            ValueError: If no plotter is available for the given plot type and backend
        """
        # Dictionary mapping (plot_type, backend) to plotter class
        plotters = {
            ("xy", "matplotlib"): MatplotlibXYPlotter,
            ("xy", "hvplot"): HvplotXYPlotter,
            ("xy", "altair"): AltairXYPlotter,
            ("xt", "matplotlib"): MatplotlibXTPlotter,
            ("xt", "altair"): AltairXTPlotter,
            # Add other combinations as they are implemented
        }
        
        key = (plot_type, backend)
        if key in plotters:
            return plotters[key]()
        else:
            raise ValueError(f"No plotter available for plot_type={plot_type}, backend={backend}")
