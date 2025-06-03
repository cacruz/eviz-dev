from unittest.mock import MagicMock, patch
from eviz.models.source_base import GenericSource


# Minimal concrete subclass for testing
class ConcreteGenericSource(GenericSource):
    def add_data_source(self, *a, **kw): pass

    def get_data_source(self, *a, **kw): pass

    def load_data_sources(self, *a, **kw): pass


def make_config_manager():
    cm = MagicMock()
    app_data = MagicMock()
    app_data.system_opts = {}
    app_data.inputs = [{'location': '/tmp', 'name': 'file.nc'}]
    cm.app_data = app_data
    cm.spec_data = {}
    cm.map_params = {0: {'field': 'f', 'filename': 'file.nc', 'to_plot': ['xy']}}
    cm.pipeline.get_all_data_sources.return_value = [MagicMock(dataset={'f': 1})]
    cm.pipeline.get_data_source.return_value = MagicMock(dataset={'f': 1})
    cm.print_to_file = False
    cm.compare = False
    cm.compare_diff = False
    cm.print_format = "png"
    cm.output_dir = "/tmp"
    return cm


def test_root_init_and_logger():
    cm = make_config_manager()
    r = ConcreteGenericSource(config_manager=cm)
    assert hasattr(r, "logger")
    assert r.app is cm.app_data
    assert r.specs is cm.spec_data


def test_root_call_triggers_plot(monkeypatch):
    cm = make_config_manager()
    r = ConcreteGenericSource(config_manager=cm)
    called = {}
    monkeypatch.setattr(r, "plot", lambda: called.setdefault("plot", True))
    r()
    assert called["plot"]

