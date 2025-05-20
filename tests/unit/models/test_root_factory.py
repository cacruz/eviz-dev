import pytest
from eviz.models.root_factory import (
    GriddedFactory, GeosFactory, WrfFactory, LisFactory, AirnowFactory,
    OmiFactory, MopittFactory, LandsatFactory, FluxnetFactory, CrestFactory, RootFactory
)
from unittest.mock import MagicMock

# List of factories that are known to produce abstract classes
ABSTRACT_FACTORIES = {
    "AirnowFactory",
    "OmiFactory",
    "MopittFactory",
    "LandsatFactory",
    "FluxnetFactory",
    "CrestFactory",
}


@pytest.mark.parametrize("factory_cls,expected_cls_name", [
    (GriddedFactory, "Gridded"),
    (GeosFactory, "Geos"),
    (WrfFactory, "Wrf"),
    (LisFactory, "Lis"),
    (AirnowFactory, "Airnow"),
    (OmiFactory, "Omi"),
    (MopittFactory, "Mopitt"),
    (LandsatFactory, "Landsat"),
    (FluxnetFactory, "Fluxnet"),
    (CrestFactory, "Crest"),
])
def test_factory_creates_correct_type(factory_cls, expected_cls_name):
    cm = MagicMock()
    factory = factory_cls()
    if factory_cls.__name__ in ABSTRACT_FACTORIES:
        with pytest.raises(TypeError):
            factory.create_root_instance(cm)
    else:
        instance = factory.create_root_instance(cm)
        assert instance.__class__.__name__ == expected_cls_name


def test_root_factory_not_implemented():
    cm = MagicMock()
    rf = RootFactory()
    with pytest.raises(NotImplementedError):
        rf.create_root_instance(cm)
