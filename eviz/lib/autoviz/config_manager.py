import logging
import os
from eviz.lib.autoviz.config import Config
from eviz.lib.autoviz.input_config import InputConfig
from eviz.lib.autoviz.output_config import OutputConfig
from eviz.lib.autoviz.system_config import SystemConfig
from eviz.lib.autoviz.history_config import HistoryConfig
import eviz.lib.utils as u
from eviz.lib.data.units import Units
from eviz.lib.data.pipeline.integrator import DataIntegrator
from eviz.lib.data.pipeline.pipeline import DataPipeline


class ConfigManager:
    """Centralized manager for all configuration objects."""
    def __init__(self, input_config: InputConfig, output_config: OutputConfig,
                 system_config: SystemConfig, history_config: HistoryConfig, config: Config):
        self.logger.info("Initializing ConfigManager")
        self.input_config = input_config
        self.output_config = output_config
        self.system_config = system_config
        self.history_config = history_config
        self.config = config
        self._units = None  # Placeholder for Units instance
        self._integrator = None  # Placeholder for DataIntegrator instance
        self._pipeline = None  # Placeholder for DataPipeline instance
        self.data_sources = {}  # Maps file paths to data sources
        self.a_list = []
        self.b_list = []        
        self.setup_comparison()

    @property
    def logger(self):
        """Return the logger for this class."""
        return logging.getLogger(__name__)

    @property
    def integrator(self):
        """Lazy initialization of DataIntegrator."""
        if self._integrator is None:
            self._integrator = DataIntegrator(self)
        return self._integrator
        
    @property
    def units(self):
        if self._units is None:
            self._units = Units(self)
        return self._units
        
    @property
    def pipeline(self):
        """Lazy initialization of DataPipeline."""
        if self._pipeline is None:
            self._pipeline = DataPipeline()
        return self._pipeline

    def to_dict(self) -> dict:
        """Return a dictionary representation of all configuration objects."""
        return {
            "input_config": self.input_config.to_dict(),
            "output_config": self.output_config.to_dict(),
            "system_config": self.system_config.to_dict(),
            "history_config": self.history_config.to_dict(),
            "app_data": self.config.app_data,
            "spec_data": self.config.spec_data,
            "map_params": self.config.map_params,
        }

    def __getattr__(self, name):
        """
        Dynamically access attributes from the underlying config objects or the Config object.
        """
        # Check if the attribute exists in this instance directly
        # (This is needed to avoid recursion for attributes like a_list and b_list)
        if name in self.__dict__:
            return self.__dict__[name]

        if hasattr(self.config, name):
            return getattr(self.config, name)

        for config in [self.input_config, self.output_config, self.system_config, self.history_config]:
            if hasattr(config, name):
                return getattr(config, name)

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_model_dim_name(self, dim_name):
        """
        Get model-specific dimension name associated with the source as defined
        in meta_coordinates.yaml.
        
        Args:
            dim_name (str): The generic dimension name to look up
            
        Returns:
            str or None: The model-specific dimension name, or None if not found
        """
        return self._get_model_dim_name(dim_name)
    
    def _get_model_dim_name(self, dim_name):
        """
        Get the model-specific dimension name.
        
        Args:
            dim_name: Generic dimension name
            
        Returns:
            str or None: Model-specific dimension name if available
        """
        source = self.source_names[self.ds_index]
        
        if source not in self.meta_coords.get(dim_name, {}):
            return None
            
        coords = self.meta_coords[dim_name][source]
        
        if ',' in coords:
            coord_candidates = coords.split(',')
            
            # Access reader differently based on its structure
            try:
                # Case 1: Old structure - reader is directly accessed
                if source in self.readers and not isinstance(self.readers[source], dict):
                    reader = self.readers[source]
                    available_dims = list(reader.datasets[self.findex]['dims'].keys())
                
                # Case 2: New structure - reader is in a dictionary by type
                elif source in self.readers and isinstance(self.readers[source], dict):
                    # Try to get a NetCDF reader first
                    readers_dict = self.readers[source]
                    if 'NetCDF' in readers_dict:
                        reader = readers_dict['NetCDF']
                    elif readers_dict:
                        # Fall back to the first available reader
                        reader = next(iter(readers_dict.values()))
                    else:
                        self.logger.warning(f"No reader found for source {source}")
                        return None
                    
                    # Get dimensions from the reader
                    if hasattr(reader, 'datasets') and self.findex in reader.datasets:
                        available_dims = list(reader.datasets[self.findex]['dims'].keys())
                    else:
                        self.logger.warning(f"Reader for {source} has no dataset at index {self.findex}")
                        return None
                else:
                    self.logger.warning(f"Source {source} not found in readers")
                    return None
                
                # Return first matching coordinate
                for coord in coord_candidates:
                    if coord in available_dims:
                        return coord
                        
            except (KeyError, AttributeError) as e:
                self.logger.warning(f"Error accessing dimensions for {source}: {e}")
                return None
                
            return None
        
        return coords
        
    def get_reader_for_file(self, source_name: str, file_path: str):
        """Get the appropriate reader for a file."""
        return self.input_config.get_reader_for_file(source_name, file_path)

    def get_primary_reader(self, source_name):
        """
        Get the primary reader for a source.
        
        Args:
            source_name (str): The name of the data source
            
        Returns:
            The primary reader or None if not found
        """
        # First try to get the reader directly - for backward compatibility
        if source_name in self.readers and not isinstance(self.readers[source_name], dict):
            return self.readers[source_name]
            
        # If readers are stored in a dictionary by type
        if source_name in self.readers:
            readers_dict = self.readers[source_name]
            # Try to get a NetCDF reader first, as it's often the most versatile
            if 'NetCDF' in readers_dict:
                return readers_dict['NetCDF']
            elif readers_dict:
                # Fall back to the first available reader
                return next(iter(readers_dict.values()))
        
        self.logger.warning(f"No reader found for source {source_name}")
        return None    
    
    def setup_comparison(self):
        """
        Set up comparison between datasets based on config settings.
        Creates a_list and b_list for comparing items identified by exp_id.
        """
        self.a_list = []
        self.b_list = []
        
        if not (self.input_config._compare or self.input_config._compare_diff):
            self.logger.debug("Comparison not enabled")
            return
        
        compare_ids = self.input_config._compare_exp_ids
        if not compare_ids or len(compare_ids) < 2:
            self.logger.warning(f"Need at least 2 IDs for comparison, found: {compare_ids}")
            return
        
        id_a, id_b = compare_ids[0].strip(), compare_ids[1].strip()
        
        for i, entry in enumerate(self.app_data.inputs):
            if 'exp_id' in entry and entry['exp_id'] == id_a:
                self.a_list.append(i)
            elif 'exp_id' in entry and entry['exp_id'] == id_b:
                self.b_list.append(i)

        self.logger.debug(f"Comparison setup: a_list={self.a_list}, b_list={self.b_list}")

    def get_file_index(self, filename):
        """ 
        Get the file index associated with the filename.
        
        Args:
            filename: The exact filename to search for
            
        Returns:
            int: Index of the file in app_data.inputs, or 0 if not found
        """
        for i, entry in enumerate(self.app_data.inputs):
            # Use exact matching or path-aware matching
            if filename == entry['filename'] or os.path.basename(filename) == os.path.basename(entry['filename']):
                return i
        return 0
    

    def get_levels(self, to_plot, plot_type):
        """ Get model levels to plot from YAML specs file"""
        levels = u.get_nested_key_value(self.spec_data, [to_plot, plot_type, 'levels'])
        if not levels:
            return []
        else:
            return levels

    def get_file_description(self, file):
        """ Get user-defined file description (default: None)"""
        try:
            return self.file_list[file]['description']
        except (KeyError, IndexError, TypeError) as e:
            self.logger.debug(f'Unable to get file description: {e}')
            return None
        
    def get_file_exp_name(self, i):
        """ Get user-defined experiment name associated with the input file (default None)"""
        try:
            return self.file_list[i]['exp_name']
        except Exception as e:
            self.logger.debug(f'key error {e}, returning default')
            return None

    def get_file_exp_id(self, i):
        """ Get user-defined experiment ID associated with the input file (default None)
        If an expid is set, then it will be used to compare with another expid, as set in compare field
        """
        try:
            return self.file_list[i]['exp_id']
        except Exception as e:
            self.logger.debug(f'key error {e}, returning default')
            return None

    @staticmethod
    def get_dim_names(pid):
        dim1, dim2 = None, None
        if 'yz' in pid:
            dim1, dim2 = 'lat', 'lev'
        elif 'xt' in pid:
            dim1, dim2 = 'time', None
        elif 'tx' in pid:
            dim1, dim2 = 'lon', 'time'
        else:
            dim1, dim2 = 'lon', 'lat'
        return dim1, dim2


    def get_model_attr_name(self, attr_name):
        """ Get model-specific attribute name associated with the source as defined
            in meta_attributes.yaml
        """
        return self.meta_attrs[attr_name][self.source_names[self.ds_index]]

    """Expose Config object attributes"""
    @property
    def app_data(self):
        return self.config.app_data

    @property
    def spec_data(self):
        return self.config.spec_data

    @property
    def map_params(self):
        return self.config.map_params

    @property
    def source_names(self):
        return self.config.source_names

    @property
    def compare(self):
        return self.input_config._compare

    @property
    def compare_diff(self):
        return self.input_config._compare_diff

    @property
    def extra_diff_plot(self):
        return self.input_config._extra_diff_plot

    @property
    def cmap(self):
        return self.input_config._cmap

    @property
    def use_cartopy(self):
        return self.input_config._use_cartopy

    @property
    def have_specs_yaml_file(self):
        return self.config._specs_yaml_exists

    @property
    def ds_index(self):
        return self.config._ds_index

    @property
    def meta_coords(self):
        return self.config.meta_coords

    @property
    def meta_attrs(self):
        return self.config.meta_attrs

    @property
    def species_db(self):
        return self.config.species_db

    @property
    def trop_height_file_list(self):
        return self.input_config._trop_height_file_list

    @property
    def sphum_conv_file_list(self):
        return self.input_config._sphum_conv_file_list

    @property
    def use_trop_height(self):
        return self.input_config._use_trop_height

    @use_trop_height.setter
    def use_trop_height(self, value):
        self.input_config._use_trop_height = value

    @property
    def use_sphum_conv(self):
        return self.input_config._use_sphum_conv

    @property
    def add_logo(self):
        return self._add_logo

    @property
    def print_to_file(self):
        return self.input_config._print_to_file

    @property
    def output_dir(self):
        return self.input_config._output_dir

    @property
    def print_format(self):
        return self.input_config._print_format

    @property
    def make_gif(self):
        return self.input_config._make_gif

    @property
    def gif_fps(self):
        return self.input_config._gif_fps

    @property
    def make_pdf(self):
        return self.input_config._make_pdf

    @property
    def mpl_style(self):
        return self.input_config._mpl_style

    @property
    def print_basic_stats(self):
        return self.input_config._print_basic_stats

    @property
    def use_mp_pool(self):
        return self.input_config._use_mp_pool

    @property
    def archive_web_results(self):
        return self.input_config._archive_web_results

    @property
    def to_plot(self):
        return self.input_config._to_plot

    @property
    def compare_exp_ids(self):
        return self.input_config._compare_exp_ids
