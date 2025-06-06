from .backends.altair.xy_plot import AltairXYPlotter
from .backends.altair.xt_plot import AltairXTPlotter
from .backends.matplotlib.xt_plot import MatplotlibXTPlotter
from .backends.matplotlib.xy_plot import MatplotlibXYPlotter
from .backends.hvplot.xy_plot import HvplotXYPlotter

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
            ("xy", "altair"): AltairXYPlotter,
            ("xt", "altair"): AltairXTPlotter,
            ("xy", "matplotlib"): MatplotlibXYPlotter,
            ("xt", "matplotlib"): MatplotlibXTPlotter,
            ("xy", "hvplot"): HvplotXYPlotter,
            # Add other combinations as they are implemented
        }
        
        key = (plot_type, backend)
        if key in plotters:
            return plotters[key]()
        else:
            raise ValueError(f"No plotter available for plot_type={plot_type}, backend={backend}")
