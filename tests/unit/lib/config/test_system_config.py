import pytest
from unittest.mock import patch, MagicMock
from eviz.lib.config.system_config import SystemConfig


def make_app_data(system_opts=None):
    class Dummy:
        pass

    d = Dummy()
    d.system_opts = system_opts or {}
    return d


def test_initialize_defaults():
    config = SystemConfig(app_data=make_app_data())
    with patch("eviz.lib.config.system_config.logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config.initialize()
        assert config.use_mp_pool is False
        assert config.archive_web_results is False
        assert config.collection == ""
        assert config.event_stamp is None



def test_initialize_with_options_sets_event_stamp(monkeypatch):
    # Patch strftime to return a fixed value
    monkeypatch.setattr(
        "eviz.lib.config.system_config.strftime", lambda fmt: "20240101-123456"
    )
    opts = {"use_mp_pool": True, "archive_web_results": True, "collection": "mycoll"}
    config = SystemConfig(app_data=make_app_data(opts))
    with patch("eviz.lib.config.system_config.logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config.initialize()
        assert config.use_mp_pool is True
        assert config.archive_web_results is True
        assert config.collection == "mycoll"
        assert config.event_stamp == "20240101-123456"


def test_logger_property():
    config = SystemConfig()
    logger = config.logger
    import logging

    assert isinstance(logger, logging.Logger)


def test_to_dict_serialization():
    config = SystemConfig()
    config.use_mp_pool = True
    config.archive_web_results = True
    config.collection = "abc"
    config.event_stamp = "20240101-123456"
    d = config.to_dict()
    assert d["use_mp_pool"] is True
    assert d["archive_web_results"] is True
    assert d["collection"] == "abc"
    assert d["event_stamp"] == "20240101-123456"
