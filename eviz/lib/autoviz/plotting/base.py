from abc import ABC, abstractmethod
import logging


class BasePlotter(ABC):
    """Base class for all plotters."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.plot_object = None
    
    @abstractmethod
    def plot(self, config, data_to_plot):
        """Create a plot from the given data."""
        pass
        
    @abstractmethod
    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        pass
        
    @abstractmethod
    def show(self):
        """Display the plot."""
        pass
    
    def get_plot_object(self):
        """Return the underlying plot object."""
        return self.plot_object


class XYPlotter(BasePlotter):
    """Base class for XY (lat-lon) plotters."""
    
    @abstractmethod
    def plot(self, config, data_to_plot):
        """Create an XY plot from the given data."""
        pass


class YZPlotter(BasePlotter):
    """Base class for YZ (zonal-mean) plotters."""
    
    @abstractmethod
    def plot(self, config, data_to_plot):
        """Create a YZ plot from the given data."""
        pass


class XTPlotter(BasePlotter):
    """Base class for XT (time-series) plotters."""
    
    @abstractmethod
    def plot(self, config, data_to_plot):
        """Create an XT plot from the given data."""
        pass


class TXPlotter(BasePlotter):
    """Base class for TX (Hovmoller) plotters."""
    
    @abstractmethod
    def plot(self, config, data_to_plot):
        """Create a TX plot from the given data."""
        pass


class ScatterPlotter(BasePlotter):
    """Base class for scatter plotters."""
    
    @abstractmethod
    def plot(self, config, data_to_plot):
        """Create a scatter plot from the given data."""
        pass


class PolarPlotter(BasePlotter):
    """Base class for ploar plotters."""
    
    @abstractmethod
    def plot(self, config, data_to_plot):
        """Create a polar plot from the given data."""
        pass
