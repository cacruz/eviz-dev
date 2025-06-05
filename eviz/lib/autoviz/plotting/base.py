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
