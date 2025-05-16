from dataclasses import dataclass

from eviz.lib.config.config_manager import ConfigManager
from eviz.models.esm.crest import Crest
from eviz.models.esm.generic import Generic
from eviz.models.esm.geos import Geos
from eviz.models.esm.lis import Lis
from eviz.models.esm.wrf import Wrf
from eviz.models.obs.inventory.airnow import Airnow
from eviz.models.obs.inventory.fluxnet import Fluxnet
from eviz.models.obs.satellite.landsat import Landsat
from eviz.models.obs.satellite.mopitt import Mopitt
from eviz.models.obs.satellite.omi import Omi


class RootFactory:
    config_manager: ConfigManager

    def create_root_instance(self, config_manager: ConfigManager):
        """Abstract method to create a root instance."""
        raise NotImplementedError("create_root_instance must be implemented in subclasses")


@dataclass
class GenericFactory(RootFactory):
    def create_root_instance(self, config_manager: ConfigManager):
        return Generic(config_manager)


@dataclass
class GeosFactory(RootFactory):
    def create_root_instance(self, config_manager: ConfigManager):
        return Geos(config_manager)


@dataclass
class WrfFactory(RootFactory):
    def create_root_instance(self, config_manager: ConfigManager):
        return Wrf(config_manager)


@dataclass
class LisFactory(RootFactory):
    def create_root_instance(self, config_manager: ConfigManager):
        return Lis(config_manager)


@dataclass
class AirnowFactory(RootFactory):
    def create_root_instance(self, config_manager: ConfigManager):
        return Airnow(config_manager)


@dataclass
class OmiFactory(RootFactory):
    def create_root_instance(self, config_manager: ConfigManager):
        return Omi(config_manager)


@dataclass
class MopittFactory(RootFactory):
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