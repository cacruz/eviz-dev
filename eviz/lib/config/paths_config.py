from dataclasses import dataclass, field
import os

@dataclass(frozen=True)
class PathsConfig:
    root_filepath: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    data_path: str = field(default_factory=lambda: os.environ.get('EVIZ_DATA_PATH', 'data'))
    output_path: str = field(default_factory=lambda: os.environ.get('EVIZ_OUTPUT_PATH', './output_plots'))
    config_path: str = field(default_factory=lambda: os.environ.get('EVIZ_CONFIG_PATH', os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')), 'config')))
    meta_attrs_path: str = field(init=False)
    meta_coords_path: str = field(init=False)
    species_db_path: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'meta_attrs_path', os.path.join(self.config_path, 'meta_attributes.yaml'))
        object.__setattr__(self, 'meta_coords_path', os.path.join(self.config_path, 'meta_coordinates.yaml'))
        object.__setattr__(self, 'species_db_path', os.path.join(self.config_path, 'species_database.yaml'))

    def model_yaml_path(self, model_name: str) -> str:
        return os.path.join(self.config_path, f"{model_name}.yaml")


