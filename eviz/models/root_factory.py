from dataclasses import dataclass
from eviz.lib.config.config_manager import ConfigManager
from eviz.models.esm.crest import Crest
from eviz.models.esm.gridded import Gridded
from eviz.models.esm.geos import Geos
from eviz.models.esm.lis import Lis
from eviz.models.esm.wrf import Wrf
from eviz.models.obs.inventory.airnow import Airnow
from eviz.models.obs.inventory.fluxnet import Fluxnet
from eviz.models.obs.satellite.landsat import Landsat
from eviz.models.obs.satellite.mopitt import Mopitt
from eviz.models.obs.satellite.omi import Omi


class RootFactory:
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
class GriddedFactory(RootFactory):
    """
    Factory for creating Gridded model instances that process NetCDF data.
    
    Used for generic gridded data, MERRA data, and special CCM/CF streams.
    """

    def create_root_instance(self, config_manager: ConfigManager):
        return Gridded(config_manager)


@dataclass
class GeosFactory(RootFactory):
    """
    Factory for creating Geos model instances for GEOS data processing.
    """

    def create_root_instance(self, config_manager: ConfigManager):
        return Geos(config_manager)


@dataclass
class WrfFactory(RootFactory):
    """
    Factory for creating Wrf model instances for Weather Research and Forecasting data.
    
    Handles regional model data that requires special treatment.
    """

    def create_root_instance(self, config_manager: ConfigManager):
        return Wrf(config_manager)


@dataclass
class LisFactory(RootFactory):
    """
    Factory for creating Lis model instances for Land Information System data.
    
    Handles regional model data that requires special treatment.
    """

    def create_root_instance(self, config_manager: ConfigManager):
        return Lis(config_manager)


@dataclass
class AirnowFactory(RootFactory):
    """
    Factory for creating Airnow model instances for processing AirNow CSV data.
    """

    def create_root_instance(self, config_manager: ConfigManager):
        return Airnow(config_manager)


@dataclass
class OmiFactory(RootFactory):
    """
    Factory for creating Omi model instances for processing OMI HDF5 satellite data.
    """

    def create_root_instance(self, config_manager: ConfigManager):
        return Omi(config_manager)


@dataclass
class MopittFactory(RootFactory):
    """
    Factory for creating Mopitt model instances for processing MOPITT HDF5 satellite data.
    """

    def create_root_instance(self, config_manager: ConfigManager):
        return Mopitt(config_manager)


@dataclass
class LandsatFactory(RootFactory):
    def create_root_instance(self, config_manager: ConfigManager):
        return Landsat(config_manager)


@dataclass
class FluxnetFactory(RootFactory):
    def create_root_instance(self, config_manager: ConfigManager):
        return Fluxnet(config_manager)


@dataclass
class CrestFactory(RootFactory):
    def create_root_instance(self, config_manager: ConfigManager):
        return Crest(config_manager)
