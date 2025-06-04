import glob
import os
import logging
import time
from typing import Optional
from argparse import Namespace
from dataclasses import dataclass, field
from eviz.lib.config.config import Config
from eviz.lib.config.config_manager import ConfigManager
from eviz.lib.config.configuration_adapter import ConfigurationAdapter
from eviz.models.source_factory import (AirnowFactory, CrestFactory, GhgFactory, GribFactory, GriddedSourceFactory,
                                        WrfFactory,
                                        LisFactory,
                                        ObsSourceFactory,
                                        MopittFactory,
                                        LandsatFactory,
                                        OmiFactory,
                                        FluxnetFactory,
                                        )
from eviz.lib.config.paths_config import PathsConfig


def get_config_path_from_env() -> Optional[str]:
    """
    This function checks for the EVIZ_CONFIG_PATH environment variable, which should
    point to the directory containing source-specific configuration files.
    """
    env_var_name = "EVIZ_CONFIG_PATH"
    return os.environ.get(env_var_name)


def create_config(args) -> ConfigManager:
    """
    Create a ConfigManager instance from command-line arguments.
    
    Args:
        args (argparse.Namespace): Command-line arguments containing configuration options.
            Expected attributes include:
            - sources: List of source names (e.g., 'gridded', 'wrf')
            - config: Optional path to configuration directory
            - configfile: Optional path to specific configuration file
    
    Returns:
        ConfigManager: A fully initialized configuration manager with input, output,
                      system, and history configurations.
    
    If no configuration directory is specified, it attempts to use the EVIZ_CONFIG_PATH
    environment variable. If that is not set, it falls back to the default path defined
    in constants.config_path.
    """
    source_names = args.sources[0].split(',')
    config_dir = args.config
    config_file = args.configfile

    paths = PathsConfig()

    if config_file:
        config = Config(source_names=source_names, config_files=config_file)
    else:
        if config_dir:
            config_files = [
                os.path.join(config_dir[0], source_name, f"{source_name}.yaml") for
                source_name in source_names]
        else:
            config_dir = get_config_path_from_env()
            if not config_dir:
                # No configuration directory specified. Using eviz default.
                config_dir = paths.config_path
            config_files = [os.path.join(config_dir, source_name, f"{source_name}.yaml")
                            for source_name in source_names]
        config = Config(source_names=source_names, config_files=config_files)

    # Initialize sub-configurations
    input_config = config.input_config
    output_config = config.output_config
    system_config = config.system_config
    history_config = config.history_config

    return ConfigManager(input_config, output_config, system_config, history_config,
                         config=config)


def get_factory_from_user_input(inputs) -> list:
    """
    Return factory classes associated with user input sources.
    
    Args:
        inputs (list): List of source names (e.g., 'gridded', 'wrf', 'omi')
    
    Returns:
        list: List of factory instances corresponding to the specified source names.
    
    This function maps source names to their corresponding factory classes, which are
    responsible for creating appropriate model instances for data processing and visualization.
    
    Each source name is associated with a unique named configuration existing within
    the EVIZ_CONFIG_PATH directory structure.
    
    Supported sources include:
    - 'test': GriddedSourceFactory (for unit tests)
    - 'gridded': GriddedSourceFactory (for generic NetCDF data)
    - 'geos': GriddedSourceFactory (for MERRA data)
    - 'ccm', 'cf': GriddedSourceFactory (for special streams)
    - 'crest' : GriddedSourceFactory (temporary)
    - 'lis': LisFactory (for Land Information System data)
    - 'wrf': WrfFactory (for Weather Research and Forecasting model data)
    - 'grib': GribFactory (for GRIB data)
    - 'airnow': ObsSourceFactory (for AirNow CSV data)
    - 'ghg': ObsSourceFactory (for GHG inventories, e.g. NOAA GML)
    - 'fluxnet': FluxnetFactory (for FluxNet CSV data)
    - 'omi': OmiFactory (for OMI HDF5 data)
    - 'mopitt': MopittFactory (for MOPITT HDF5 data)
    - 'landsat': LandsatFactory (for Landsat HDF4 data)
    """
    mappings = {
        "test": GriddedSourceFactory(),    # for unit tests
        "gridded": GriddedSourceFactory(), # default for all gridded data such as NetCDF
        "geos": GriddedSourceFactory(),    # special alias for GEOS datasets such as MERRA
        "ccm": GriddedSourceFactory(),     # special alias for GEOS datasets CCM
        "cf": GriddedSourceFactory(),      # and CF
        "crest": GriddedSourceFactory(),   # and CREST
        "obs": ObsSourceFactory(),         # for all observation data such
        "lis": LisFactory(),               # LIS and WRF are gridded but require special
        "wrf": WrfFactory(),               # "treatment" due to the "regional" nature of the data
        "grib": GribFactory(),             #  Grib data sources like ERA5, GFS, etc.
        "airnow": AirnowFactory(),         # CSV
        "ghg": GhgFactory(),               # CSV
        "fluxnet": FluxnetFactory(),       # CSV
        "omi": OmiFactory(),               # HDF5
        "mopitt": MopittFactory(),         # HDF5
        "landsat": LandsatFactory(),       # HDF4
        # Add other mappings for other subclasses
        # Need MODIS, CEDS, EDGAR
    }
    factories = []
    for i in inputs:
        if i not in mappings:
            print(
                f"\nERROR: '{i}' is not a valid source name. Valid options are: {list(mappings.keys())}\n")
            import sys
            sys.exit(1)
        factories.append(mappings[i])
    return factories


@dataclass
class Autoviz:
    """
    Main class for automatic visualization of Earth System Model and observational data.
    
    The Autoviz class orchestrates the entire visualization process, from configuration
    loading to data processing and plot generation. It serves as the primary entry point
    for the eViz visualization system.
    
    This class takes source names as input, creates appropriate factory instances for those
    sources, initializes configuration, and manages the visualization workflow. It supports
    various data sources including gridded data, regional models, and observational datasets.
    
    Attributes:

        source_names (list): List of source model names to process (e.g., 'gridded', 'wrf')
        args (argparse.Namespace, optional): Command-line arguments for configuration.
            If not provided, default arguments are created.
        model_info (dict): Dictionary storing information about processed models.
        model_name (str, optional): Name of the current model being processed.
        _config_manager (ConfigManager): Configuration manager instance.
        factory_sources (list): List of factory instances for the specified sources.
    
    Methods:

        run(): Execute the visualization process.
        set_data(input_files): Assign model input files as specified in model config file.
        set_output(output_dir): Assign model output directory as specified in model config file.
        _check_input_files(): Verify existence of input files and provide warnings for missing files.

        # Create an Autoviz instance with gridded data source
        viz = Autoviz(['gridded'])
        
        # Run the visualization process
        viz.run()
    """
    source_names: list
    args: Namespace = None
    model_info: dict = field(default_factory=dict)
    model_name: str = None
    _config_manager: ConfigManager = None

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __post_init__(self):
        """
        Initialize the Autoviz instance after dataclass initialization.

        Raises:
            ValueError: If no factories are found for the specified sources.
        """
        self.logger.info("Start init")
        # Add this workaround to simplify working within a Jupyter notebook, that is, to avoid
        # having to pass a Namespace() object, we create args with the appropriate defaults
        if not self.args:
            self.args = Namespace(
                sources=self.source_names,
                compare=False,
                file=None,
                vars=None,
                configfile=None,
                config=None,
                data_dirs=None,
                output_dirs=None,
                verbose=1,
            )
        self.factory_sources = get_factory_from_user_input(self.source_names)
        if not self.factory_sources:
            raise ValueError(f"No factories found for sources: {self.source_names}")
        self._config_manager = create_config(
            self.args)  # Use ConfigManager instead of Config
        # TODO: enable processing of S3 buckets

    def run(self):
        """
        Execute the visualization process.
        """
        _start_time = time.time()
        self._config_manager.input_config.start_time = _start_time

        self._check_input_files()

        self.config_adapter = ConfigurationAdapter(self._config_manager)

        if hasattr(self.args, 'integrate') and self.args.integrate:
            self.logger.info("Data integration mode enabled")
            self._config_manager.input_config._enable_integration = True

        try:
            self.logger.info("Processing configuration using adapter")
            self.config_adapter.process_configuration()

            all_data_sources = {}
            try:
                if hasattr(self._config_manager,
                           '_pipeline') and self._config_manager._pipeline is not None:
                    all_data_sources = self._config_manager._pipeline.get_all_data_sources()
                else:
                    self.logger.error("Pipeline not initialized properly")
            except Exception as e:
                self.logger.error(f"Error accessing pipeline: {e}")

            if not all_data_sources:
                self.logger.error(
                    "No data sources were loaded. Check if the input files exist and are accessible.")
                for i, entry in enumerate(self._config_manager.app_data.inputs):
                    file_path = os.path.join(entry.get('location', ''),
                                             entry.get('name', ''))
                    self.logger.debug(f"  {i + 1}. {file_path}")
                return

            if hasattr(self.args, 'composite') and self.args.composite:
                composite_args = self.args.composite[0].split(',')
                if len(composite_args) >= 3:
                    field1, field2, operation = composite_args[:3]
                    self.logger.info(
                        f"Creating composite field: {field1} {operation} {field2}")
                    for factory in self.factory_sources:
                        model = factory.create_root_instance(self._config_manager)
                        model.plot_composite_field(field1, field2, operation)
                    return

            for factory in self.factory_sources:
                model = factory.create_root_instance(self._config_manager)

                if hasattr(model, 'set_map_params') and self._config_manager.map_params:
                    self.logger.info(
                        f"Setting map_params with {len(self._config_manager.map_params)} entries")
                    model.set_map_params(self._config_manager.map_params)
                else:
                    self.logger.warning(
                        "No map_params available or model doesn't support set_map_params")

                model()

        finally:
            self.config_adapter.close()

    def set_data(self, input_files) -> None:
        """
        Assign model input files as specified in model config file.
        
        Args:
            input_files (list): Names of input files to be processed.
        """
        config = self._config_manager.input_config
        config.set_input_files(input_files)

    def _check_input_files(self) -> None:
        """
        Check if input files exist and provide warnings for missing files.

        This method verifies the existence of all input files specified in the
        configuration and logs warnings for any files that cannot be found.
        It provides detailed information about missing files to help users
        troubleshoot configuration issues.

        The application will attempt to continue execution even if some files
        are missing, but plotting operations may fail if required data is unavailable.
        """
        if not hasattr(self._config_manager, 'app_data') or not hasattr(
                self._config_manager.app_data, 'inputs'):
            self.logger.warning("No input files specified in configuration.")
            return

        missing_files = []
        for i, entry in enumerate(self._config_manager.app_data.inputs):
            file_path = os.path.join(entry.get('location', ''), entry.get('name', ''))

            if '*' in file_path or '?' in file_path or '[' in file_path:
                matched_files = glob.glob(file_path)
                if not matched_files:
                    missing_files.append((i, file_path))
            else:
                if not os.path.exists(file_path):
                    missing_files.append((i, file_path))

        if missing_files:
            self.logger.warning(f"Found {len(missing_files)} missing input files:")
            for i, file_path in missing_files:
                self.logger.warning(f"  {i + 1}. {file_path}")
            print(f"Warning: {len(missing_files)} input files are missing:")
            for i, file_path in missing_files:
                print(f"  {i + 1}. {file_path}")
            print("The application will attempt to continue, but plotting may fail.")

    def set_output(self, output_dir) -> str:
        """
        Assign model output directory as specified in model config file.
        
        Args:
            output_dir (str): Path to the directory where output files should be saved.
        """
        config = self._config_manager.output_config
        config.set_output_dir(output_dir)
