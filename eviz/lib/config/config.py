import yaml
import os
from eviz.lib.data.data_source import DataSourceFactory

class Config:
    """Class to manage runtime configuration for Eviz."""

    def __init__(self, config_file: str):
        """
        Initialize the Config class by loading the YAML configuration file.

        Parameters:
            config_file (str): Path to the YAML configuration file.
        """
        self.config_file = config_file
        self.data_sources = []  # List to store initialized DataSource instances
        self._load_config()

    def _load_config(self):
        """Load and parse the YAML configuration file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        with open(self.config_file, 'r') as file:
            self.config_data = yaml.safe_load(file)

        # Initialize data sources based on the configuration
        self._initialize_data_sources()

    def _initialize_data_sources(self):
        """Initialize DataSource instances based on the configuration."""
        file_list = self.config_data.get('files', [])
        for file_entry in file_list:
            file_path = file_entry.get('name')
            if not file_path:
                continue

            # Determine file extension
            file_extension = file_path.split('.')[-1]

            # Use the factory to get the appropriate DataSource instance
            data_source = DataSourceFactory.get_data_class(file_extension)
            self.data_sources.append(data_source)

    def get_data_sources(self):
        """Return the list of initialized DataSource instances."""
        return self.data_sources

    def get_config(self, key: str, default=None):
        """Retrieve a specific configuration value."""
        return self.config_data.get(key, default)
