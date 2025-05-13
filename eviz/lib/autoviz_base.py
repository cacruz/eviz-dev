import os
import logging
import time
from argparse import Namespace
from dataclasses import dataclass, field
from eviz.lib.autoviz.config import Config
from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.lib.autoviz.configuration_adapter import ConfigurationAdapter
from eviz.models.extensions.factory import ModelExtensionFactory
from eviz.models.root_factory import (GenericFactory, 
                              WrfFactory, 
                              LisFactory,
                              AirnowFactory,
                              MopittFactory,
                              LandsatFactory,
                              OmiFactory,
                              FluxnetFactory,
)
import eviz.lib.const as constants


def get_config_path_from_env():
    env_var_name = "EVIZ_CONFIG_PATH"
    return os.environ.get(env_var_name)


def create_config(args):
    source_names = args.sources[0].split(',')
    config_dir = args.config
    config_file = args.configfile
    if config_file:
        config = Config(source_names=source_names, config_files=config_file)
    else:
        if config_dir:
            config_files = [os.path.join(config_dir[0], source_name, f"{source_name}.yaml") for source_name in source_names]
        else:
            config_dir = get_config_path_from_env()
            if not config_dir:
                print(f"Warning: No configuration directory specified. Using default.")
                config_dir = constants.config_path
            config_files = [os.path.join(config_dir, source_name, f"{source_name}.yaml") for source_name in source_names]
        config = Config(source_names=source_names, config_files=config_files)

    # Initialize sub-configurations
    input_config = config.input_config
    output_config = config.output_config
    system_config = config.system_config
    history_config = config.history_config

    # Return a ConfigManager
    return ConfigManager(input_config, output_config, system_config, history_config, config=config)


def get_factory_from_user_input(inputs):
    """ Return subclass associated with user input sources

        Note:
        An entry here associates the specified data source with a unique named configuration
        existing within EVIZ_CONFIG_PATH.
    """
    mappings = {
        "test": GenericFactory(),      # for unit tests
        "generic": GenericFactory(),   # generic is NetCDF
        "geos": GenericFactory(),      # use this for MERRA
        "ccm": GenericFactory(),       # CCM and CF are "special" streams
        "cf": GenericFactory(),        #
        # "crest": CrestFactory(),     #
        "lis": LisFactory(),           # LIS and WRF are generic but require special
        "wrf": WrfFactory(),           # "treatment" due to the "regional" nature of the data
        "airnow": AirnowFactory(),     # CSV
        "fluxnet": FluxnetFactory(),   # CSV
        "omi": OmiFactory(),           # HDF5
        "mopitt": MopittFactory(),     # HDF5
        "landsat": LandsatFactory(),   # HDF4
        # Add other mappings for other subclasses
        # Need MODIS, GRIB, CEDS, EDGAR
    }
    return [mappings[i] for i in inputs]


@dataclass
class Autoviz:
    """ This is the Autoviz class definition. It takes in a list of (source) names and
        creates data-reading-classes (factories) associated with each of those names.

    Parameters:
        source_names (list): source models to process
        factory_models (list): source models to process
        args (Namespace): source models to process
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
        self._config_manager = create_config(self.args)  # Use ConfigManager instead of Config
        # TODO: Associate each model with its corresponding data directory
        #  Note that data can be in local disk or even in a remote locations
        # TODO: enable processing of S3 buckets

    def run(self):
        """ Create plots """
        _start_time = time.time()
        self._config_manager.input_config.start_time = _start_time
        
        # Check if input files exist
        self._check_input_files()
        
        # Create configuration adapter
        self.config_adapter = ConfigurationAdapter(self._config_manager)
        
        # Handle integration options from command line
        if hasattr(self.args, 'integrate') and self.args.integrate:
            self.logger.info("Data integration mode enabled")
            self._config_manager.input_config._enable_integration = True
        
        # Process configuration using the adapter
        try:
            self.logger.info("Processing configuration using adapter")
            self.config_adapter.process_configuration()
            
            # Check if any data sources were loaded
            if not self.config_adapter.get_all_data_sources():
                self.logger.error("No data sources were loaded. Check if the input files exist and are accessible.")
                print("No data sources were loaded. Check if the input files exist and are accessible.")
                print("Input files specified in the configuration:")
                for i, entry in enumerate(self._config_manager.app_data.inputs):
                    file_path = os.path.join(entry.get('location', ''), entry.get('name', ''))
                    print(f"  {i+1}. {file_path}")
                return
            
            # Apply model-specific extensions to data sources
            self._apply_model_extensions()
            
            # Handle composite fields from command line
            if hasattr(self.args, 'composite') and self.args.composite:
                composite_args = self.args.composite[0].split(',')
                if len(composite_args) >= 3:
                    field1, field2, operation = composite_args[:3]
                    self.logger.info(f"Creating composite field: {field1} {operation} {field2}")
                    for factory in self.factory_sources:
                        model = factory.create_root_instance(self._config_manager)
                        # Pass data sources to the model
                        for file_path, data_source in self.config_adapter.get_all_data_sources().items():
                            model.add_data_source(file_path, data_source)
                        model.plot_composite_field(field1, field2, operation)
                    return
            
            # Normal plotting
            for factory in self.factory_sources:
                model = factory.create_root_instance(self._config_manager)
                # Pass data sources to the model
                for file_path, data_source in self.config_adapter.get_all_data_sources().items():
                    model.add_data_source(file_path, data_source)
                
                # Ensure map_params are available to the model
                if hasattr(model, 'set_map_params') and self._config_manager.map_params:
                    self.logger.info(f"Setting map_params with {len(self._config_manager.map_params)} entries")
                    model.set_map_params(self._config_manager.map_params)
                else:
                    self.logger.warning("No map_params available or model doesn't support set_map_params")
                
                model()
                
        finally:
            # Clean up resources
            self.config_adapter.close()
    
    def _apply_model_extensions(self):
        """Apply model-specific extensions to data sources."""
        for source_name in self.source_names:
            # Create model extension
            extension = ModelExtensionFactory.create_extension(source_name, self._config_manager)
            
            # Apply extension to all data sources for this model
            for file_path, data_source in self.config_adapter.get_all_data_sources().items():
                if data_source.model_name == source_name:
                    self.logger.info(f"Applying {source_name} extension to {file_path}")
                    extension.process_data_source(data_source)
            
    def set_data(self, input_files):
        """ Assign model input files as specified in model config file

        Parameters:
            input_files (list): Names of input files
        """
        config = self._config_manager.input_config
        config.set_input_files(input_files)

    def _check_input_files(self):
        """Check if input files exist and provide warnings for missing files."""
        if not hasattr(self._config_manager, 'app_data') or not hasattr(self._config_manager.app_data, 'inputs'):
            self.logger.warning("No input files specified in configuration.")
            return
            
        missing_files = []
        for i, entry in enumerate(self._config_manager.app_data.inputs):
            file_path = os.path.join(entry.get('location', ''), entry.get('name', ''))
            if not os.path.exists(file_path):
                missing_files.append((i, file_path))
                
        if missing_files:
            self.logger.warning(f"Found {len(missing_files)} missing input files:")
            for i, file_path in missing_files:
                self.logger.warning(f"  {i+1}. {file_path}")
            print(f"Warning: {len(missing_files)} input files are missing:")
            for i, file_path in missing_files:
                print(f"  {i+1}. {file_path}")
            print("The application will attempt to continue, but plotting may fail.")
            
    def set_output(self, output_dir):
        """ Assign model output directory as specified in model config file

        Parameters:
            output_dir (str): Name output directory
        """
        config = self._config_manager.output_config
        config.set_output_dir(output_dir)
