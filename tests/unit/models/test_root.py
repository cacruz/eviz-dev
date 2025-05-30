from unittest.mock import MagicMock, patch
from eviz.models.root import Root


# Minimal concrete subclass for testing
class ConcreteRoot(Root):
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
    r = ConcreteRoot(config_manager=cm)
    assert hasattr(r, "logger")
    assert r.app is cm.app_data
    assert r.specs is cm.spec_data


def test_root_call_triggers_plot(monkeypatch):
    cm = make_config_manager()
    r = ConcreteRoot(config_manager=cm)
    called = {}
    monkeypatch.setattr(r, "plot", lambda: called.setdefault("plot", True))
    r()
    assert called["plot"]


def test_root_plot_simple(monkeypatch):
    cm = make_config_manager()
    r = ConcreteRoot(config_manager=cm)
    called = {}
    monkeypatch.setattr(r, "_simple_plots", lambda plotter: called.setdefault("simple", True))
    r.config_manager.spec_data = {}
    r.plot()
    assert called["simple"]


def test_root_plot_single(monkeypatch):
    cm = make_config_manager()
    r = ConcreteRoot(config_manager=cm)
    called = {}
    monkeypatch.setattr(r, "_single_plots", lambda plotter: called.setdefault("single", True))
    r.config_manager.spec_data = {"foo": "bar"}
    r.config_manager.overlay = False
    r.config_manager.compare = False
    r.config_manager.compare_diff = False
    r.plot()
    assert called["single"]


def test_root_plot_comparison(monkeypatch):
    cm = make_config_manager()
    r = ConcreteRoot(config_manager=cm)
    called = {}
    monkeypatch.setattr(r, "_comparison_plots", lambda plotter: called.setdefault("comp", True))
    r.config_manager.spec_data = {"foo": "bar"}
    r.config_manager.compare = False
    r.config_manager.compare_diff = True
    r.plot()
    assert called["comp"]


def test_root_plot_side_by_side(monkeypatch):
    cm = make_config_manager()
    r = ConcreteRoot(config_manager=cm)
    called = {}
    monkeypatch.setattr(r, "_side_by_side_plots", lambda plotter: called.setdefault("side", True))
    r.config_manager.spec_data = {"foo": "bar"}
    r.config_manager.compare = True
    r.config_manager.compare_diff = False
    r.plot()
    assert called["side"]


def test_root_single_plots_handles_missing_map_params(caplog):
    cm = make_config_manager()
    r = ConcreteRoot(config_manager=cm)
    r.config_manager.map_params = {}
    r._single_plots(MagicMock())
    assert "No map_params available" in caplog.text


def test_root_single_plots_handles_missing_data_sources(caplog):
    cm = make_config_manager()
    r = ConcreteRoot(config_manager=cm)
    r.config_manager.map_params = {0: {'field': 'f', 'filename': 'file.nc', 'to_plot': ['xy']}}
    r.config_manager.pipeline.get_all_data_sources.return_value = []
    r._single_plots(MagicMock())
    assert "No data sources available" in caplog.text


def test_root_plot_dest_print_to_file(monkeypatch):
    cm = make_config_manager()
    r = ConcreteRoot(config_manager=cm)
    r.config_manager.print_to_file = True
    r.config_manager.print_format = "png"
    r.config_manager.output_dir = "/tmp"
    with patch("matplotlib.pyplot.savefig") as msave:
        r._plot_dest("testfile")
        msave.assert_called_once()
