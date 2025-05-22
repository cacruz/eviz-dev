import os
import pytest
from eviz.lib.config.paths_config import PathsConfig

def test_defaults(monkeypatch):
    # Unset environment variables to test defaults
    monkeypatch.delenv("EVIZ_DATA_PATH", raising=False)
    monkeypatch.delenv("EVIZ_OUTPUT_PATH", raising=False)
    monkeypatch.delenv("EVIZ_CONFIG_PATH", raising=False)

    paths = PathsConfig()
    # Should default to 'data'
    assert paths.data_path == "data"
    # Should default to './output_plots'
    assert paths.output_path == "./output_plots"
    # Should default to <project_root>/config
    assert paths.config_path.endswith(os.path.join("config"))
    # meta/config/species paths should be under <project_root>/config
    assert paths.meta_attrs_path.endswith(os.path.join("config", "meta_attributes.yaml"))
    assert paths.meta_coords_path.endswith(os.path.join("config", "meta_coordinates.yaml"))
    assert paths.species_db_path.endswith(os.path.join("config", "species_database.yaml"))

def test_env_overrides(monkeypatch):
    monkeypatch.setenv("EVIZ_DATA_PATH", "/tmp/data")
    monkeypatch.setenv("EVIZ_OUTPUT_PATH", "/tmp/output")
    monkeypatch.setenv("EVIZ_CONFIG_PATH", "/tmp/config")

    paths = PathsConfig()
    assert paths.data_path == "/tmp/data"
    assert paths.output_path == "/tmp/output"
    assert paths.config_path == "/tmp/config"
    # meta/config/species paths should still be under <project_root>/config
    assert paths.meta_attrs_path.endswith(os.path.join("config", "meta_attributes.yaml"))
    assert paths.meta_coords_path.endswith(os.path.join("config", "meta_coordinates.yaml"))
    assert paths.species_db_path.endswith(os.path.join("config", "species_database.yaml"))

def test_model_yaml_path():
    paths = PathsConfig()
    model_name = "testmodel"
    expected = os.path.join(paths.config_path, "testmodel.yaml")
    assert paths.model_yaml_path(model_name) == expected

def test_paths_are_absolute():
    paths = PathsConfig()
    assert os.path.isabs(paths.root_filepath)
    assert os.path.isabs(paths.meta_attrs_path)
    assert os.path.isabs(paths.meta_coords_path)
    assert os.path.isabs(paths.species_db_path)
