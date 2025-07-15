import os
import pytest
from unittest.mock import MagicMock, patch
from eviz.lib.config.output_config import OutputConfig


def make_app_data(outputs=None):
    class Dummy:
        pass

    d = Dummy()
    d.outputs = outputs or {}
    return d


def test_initialize_defaults():
    config = OutputConfig(app_data=make_app_data())
    config.initialize()
    assert config.add_logo is False
    assert config.print_to_file is False
    assert config.print_format == "png"
    assert config.make_pdf is False
    assert config.print_basic_stats is False
    assert config.make_gif is False
    assert config.gif_fps == 10
    assert config.output_dir == "./output_plots"
    assert config.dpi == 300
    assert config.fig_style == "default"
    assert config.backend == "matplotlib"


def test_initialize_with_outputs():
    outputs = {
        "add_logo": True,
        "print_to_file": True,
        "print_format": "pdf",
        "make_pdf": True,
        "print_basic_stats": True,
        "mpl_style": "seaborn",
        "fig_style": "fancy",
        "make_gif": True,
        "gif_fps": 20,
        "dpi": 150,
        "output_dir": "/tmp/plots",
        "visualization": {
            "backend": "plotly",
            "colormap": "viridis",
            "fig_style": "fancy",
            "dpi": 150,
            "gif_fps": 20,
            "mpl_style": "seaborn-dark",
        },
    }
    config = OutputConfig(app_data=make_app_data(outputs))
    with patch("os.makedirs") as makedirs, patch("os.path.exists", return_value=False):
        config.initialize()
        makedirs.assert_called_once_with("./output_plots")
    assert config.add_logo is True
    assert config.print_to_file is True
    assert config.print_format == "pdf"
    assert config.make_pdf is True
    assert config.print_basic_stats is True
    assert config.mpl_style == "seaborn-dark"
    assert config.make_gif is True
    assert config.gif_fps == 20
    assert config.dpi == 150
    assert config.backend == "plotly"
    assert config.colormap == "viridis"
    assert config.fig_style == "fancy"


def test_set_output_dir_creates_dir():
    config = OutputConfig(app_data=make_app_data({"print_to_file": True}))
    config.output_dir = "/tmp/test_output_dir"
    config.print_to_file = True
    with patch("os.makedirs") as makedirs, patch("os.path.exists", return_value=False):
        config._set_output_dir()
        makedirs.assert_called_once_with("/tmp/test_output_dir")


def test_set_output_dir_exists():
    config = OutputConfig(app_data=make_app_data({"print_to_file": True}))
    config.output_dir = "/tmp/test_output_dir"
    config.print_to_file = True
    with patch("os.makedirs") as makedirs, patch("os.path.exists", return_value=True):
        config._set_output_dir()
        makedirs.assert_not_called()


def test_logger_property():
    config = OutputConfig()
    logger = config.logger
    import logging

    assert isinstance(logger, logging.Logger)


def test_to_dict_serialization():
    config = OutputConfig()
    config.backend = "matplotlib"
    config.colormap = "coolwarm"
    config.fig_style = "default"
    config.dpi = 300
    config.gif_fps = 10
    config.mpl_style = "classic"
    config.output_dir = "/tmp/plots"
    config.print_to_file = True
    config.print_format = "png"
    config.add_logo = True
    config.make_pdf = True
    config.print_basic_stats = True
    config.make_gif = True
    d = config.to_dict()
    assert d["backend"] == "matplotlib"
    assert d["colormap"] == "coolwarm"
    assert d["fig_style"] == "default"
    assert d["dpi"] == 300
    assert d["gif_fps"] == 10
    assert d["mpl_style"] == "classic"
    assert d["output_dir"] == "/tmp/plots"
    assert d["print_to_file"] is True
    assert d["print_format"] == "png"
    assert d["add_logo"] is True
    assert d["make_pdf"] is True
    assert d["print_basic_stats"] is True
    assert d["make_gif"] is True


def test_init_visualization_branch():
    outputs = {
        "visualization": {
            "backend": "matplotlib",
            "colormap": "coolwarm",
            "fig_style": "default",
            "dpi": 300,
            "gif_fps": 10,
            "mpl_style": "seaborn",
        }
    }
    config = OutputConfig(app_data=make_app_data(outputs))
    config._init_visualization(outputs)
    assert config.backend == "matplotlib"
    assert config.colormap == "coolwarm"
    assert config.fig_style == "default"
    assert config.dpi == 300
    assert config.gif_fps == 10
    assert config.mpl_style == "seaborn"
