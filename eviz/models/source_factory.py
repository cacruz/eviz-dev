from dataclasses import dataclass
from eviz.lib.config.config_manager import ConfigManager
from eviz.models.esm.crest import Crest
from eviz.models.gridded_source import GriddedSource
from eviz.models.esm.grib import Grib
from eviz.models.esm.geos import Geos
from eviz.models.esm.lis import Lis
from eviz.models.esm.wrf import Wrf
from eviz.models.obs_source import ObsSource
from eviz.models.obs.inventory.airnow import Airnow
from eviz.models.obs.inventory.ghg import Ghg
from eviz.models.obs.satellite.omi import Omi


class BaseSourceFactory:
    """
    Abstract factory base class for creating data model instances.
    
    Defines the interface for concrete factory classes that instantiate
    specific data model handlers for different data formats and sources.
    """
    config_manager: ConfigManager

    def create_root_instance(self, config_manager: ConfigManager):
        """
        Abstract method to create a root instance.
        
        Args:
            config_manager: Configuration manager containing settings for the model.
            
        Returns:
            A model instance configured for data processing and visualization.
        """
        raise NotImplementedError(
            "create_root_instance must be implemented in subclasses")


@dataclass
class GriddedSourceFactory(BaseSourceFactory):
    """
    Factory for creating GriddedSource model instances that process NetCDF data.
    
    Used for generic gridded data, MERRA data, and special CCM/CF streams.
    """

    def create_root_instance(self, config_manager: ConfigManager):
        return GriddedSource(config_manager)


@dataclass
class GribFactory(BaseSourceFactory):
    """
    Factory for creating GriddedSource model instances for GRIB data.
    
    Used for GRIB format weather and climate data from sources like ERA5, GFS, etc.
    """
    def create_root_instance(self, config_manager: ConfigManager):
        return Grib(config_manager)


@dataclass
class GeosFactory(BaseSourceFactory):
    """
    Factory for creating Geos model instances for GEOS data processing.
    """
    def create_root_instance(self, config_manager: ConfigManager):
        return Geos(config_manager)


@dataclass
class WrfFactory(BaseSourceFactory):
    """
    Factory for creating Wrf model instances for Weather Research and Forecasting data.
    
    Handles regional model data that requires special treatment.
    """
    def create_root_instance(self, config_manager: ConfigManager):
        return Wrf(config_manager)


@dataclass
class LisFactory(BaseSourceFactory):
    """
    Factory for creating Lis model instances for Land Information System data.
    
    Handles regional model data that requires special treatment.
    """
    def create_root_instance(self, config_manager: ConfigManager):
        return Lis(config_manager)


@dataclass
class GhgFactory(BaseSourceFactory):
    """
    Factory for creating Ghg model instances
    """
    def create_root_instance(self, config_manager: ConfigManager):
        return Ghg(config_manager)


@dataclass
class ObsSourceFactory(BaseSourceFactory):
    """
    Factory for creating ObsSource model instances
    """
    def create_root_instance(self, config_manager: ConfigManager):
        return ObsSource(config_manager)


@dataclass
class AirnowFactory(BaseSourceFactory):
    """
    Factory for creating Airnow model instances for processing AirNow CSV data.
    """
    def create_root_instance(self, config_manager: ConfigManager):
        return Airnow(config_manager)


@dataclass
class OmiFactory(BaseSourceFactory):
    """
    Factory for creating Omi model instances for processing OMI HDF5 satellite data.
    """
    def create_root_instance(self, config_manager: ConfigManager):
        return Omi(config_manager)


@dataclass
class CrestFactory(BaseSourceFactory):
    """
    Factory for creating Crest model instances for processing CREST-generated data.
    """
    def create_root_instance(self, config_manager: ConfigManager):
        return Crest(config_manager)
