from dataclasses import dataclass, field
import logging
import os
from typing import Optional, List
import eviz.lib.utils as u
from eviz.lib.config.config import Config
from eviz.lib.config.input_config import InputConfig
from eviz.lib.config.output_config import OutputConfig
from eviz.lib.config.system_config import SystemConfig
from eviz.lib.config.history_config import HistoryConfig
from eviz.lib.data.pipeline.integrator import DataIntegrator
from eviz.lib.data.pipeline.pipeline import DataPipeline


@dataclass
class ConfigManager:
    """
    Enhanced configuration manager for the eViz application.
    
    This class extends the base Config class to provide additional functionality specific
    to the eViz application's needs. It serves as the primary interface between the application
    and its configuration system, offering simplified access to configuration parameters and
    adding application-specific features.
    
    The ConfigManager integrates with command-line arguments, manages comparison modes between
    data sources, provides dimension name mapping, and offers runtime configuration updates.
    It also maintains references to key application components like the data processing pipeline.
    
    Attributes:
        All attributes from Config class, plus:
        args: Command-line arguments passed to the application
        compare: Flag indicating if comparison mode is active
        compare_diff: Flag indicating if difference comparison mode is active
        compare_exp_ids: List of experiment IDs for comparison
        a_list: List of indices for the first set of comparison items
        b_list: List of indices for the second set of comparison items
        findex: Current file index being processed
        pindex: Current plot index being processed
        axindex: Current axis index being processed
        level: Current vertical level being processed
        time_level: Current time level being processed
        real_time: Human-readable representation of the current time
        pipeline: Reference to the data processing pipeline
        
    Methods:
        initialize: Complete the initialization process after construction
        get_model_dim_name: Map standard dimension names to model-specific names
        get_dim_names: Get dimension names for a specific plot type
        get_levels: Get level information for a specific field and plot type
        
    Note:
        This class is designed to be instantiated once and used throughout the application
        as the single source of configuration information.
    """
    # Required fields first
    input_config: InputConfig
    output_config: OutputConfig
    system_config: SystemConfig
    history_config: HistoryConfig
    config: Config

    # Fields with default values
    a_list: List[int] = field(default_factory=list)
    b_list: List[int] = field(default_factory=list)
    _findex: int = 0  # Use underscore to avoid name conflicts
    _ds_index: int = 0  # Use underscore to avoid name conflicts

    # Fields not included in __init__
    _units: Optional[object] = field(default=None, init=False)
    _integrator: Optional[DataIntegrator] = field(default=None, init=False)
    _pipeline: Optional[DataPipeline] = field(default=None, init=False)
        
    def __post_init__(self):
        self.logger.info("Start init")
        self.setup_comparison()

    @property
    def paths(self):
        return self.config.paths

    @property
    def app_data(self):
        return self.config.app_data

    @property
    def spec_data(self):
        return self.config.spec_data
    
    @property
    def source_names(self):
        return self.config.source_names

    @property
    def ds_index(self):
        # Return the local value, not delegating to config
        return self._ds_index

    @ds_index.setter
    def ds_index(self, value):
        self._ds_index = value
        # Optionally also update config if needed
        if hasattr(self.config, '_ds_index'):
            self.config._ds_index = value

    @property
    def findex(self):
        return self._findex

    @findex.setter
    def findex(self, value):
        self._findex = value
        # Optionally also update config if needed
        if hasattr(self.config, '_findex'):
            self.config._findex = value    
    @property
    def logger(self):
        """Return the logger for this class."""
        return logging.getLogger(__name__)

    @property
    def integrator(self):
        """Lazy initialization of DataIntegrator."""
        if self._integrator is None:
            self._integrator = DataIntegrator()
        return self._integrator

    @property
    def units(self):
        """Lazy initialization of Units."""
        if self._units is None:
            try:
                from eviz.lib.data.units import Units
                self._units = Units(self)
            except Exception as e:
                self.logger.error(f"Error initializing Units: {e}")
                self._units = None
        return self._units

    @property
    def pipeline(self):
        """Lazy initialization of DataPipeline."""
        if self._pipeline is None:
            self._pipeline = DataPipeline(self)
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
            dim_name (str): The dimension name to look up

        Returns:
            str or None: The model-specific dimension name, or None if not found
        """
        return self._get_model_dim_name(dim_name)

    def _get_model_dim_name(self, dim_name):
        """
        Get the model-specific dimension name.

        Args:
            dim_name: Gridded dimension name

        Returns:
            str or None: Model-specific dimension name if available
        """
        source = self.source_names[self.ds_index]
        
        if source not in self.meta_coords.get(dim_name, {}):
            self.logger.debug(f"No meta_coords mapping for dimension '{dim_name}' and source '{source}'")
            return None

        coords = self.meta_coords[dim_name][source]

        if isinstance(coords, str) and ',' not in coords:
             self.logger.debug(f"Found direct mapping for '{dim_name}' in source '{source}': '{coords}'")
             return coords

        coord_candidates = coords.split(',') if isinstance(coords, str) else [coords.get('dim')] if isinstance(coords, dict) and 'dim' in coords else []

        file_path = None
        try:
            # Assuming app_data.inputs is a list of dictionaries
            if self.findex is not None and self.findex < len(self.app_data.inputs):
                 file_entry = self.app_data.inputs[self.findex]
                 file_path = os.path.join(file_entry.get('location', ''), file_entry.get('name', ''))
        except Exception as e:
            self.logger.debug(f"Could not get file path for ds_index {self.ds_index}, findex {self.findex}: {e}")
            for entry in self.app_data.inputs:
                if entry.get('source_name') == source:
                    file_path = os.path.join(entry.get('location', ''), entry.get('name', ''))
                    break

        if file_path:
            data_source = self.pipeline.get_data_source(file_path)

            if data_source and hasattr(data_source, 'dataset') and data_source.dataset is not None:
                available_dims = list(data_source.dataset.dims.keys())
                self.logger.debug(f"Available dimensions in dataset for {file_path}: {available_dims}")

                self.logger.debug(f"Coordinate candidates for '{dim_name}': {coord_candidates}")
                for coord in coord_candidates:
                    if coord and coord in available_dims:
                        self.logger.debug(f"Found matching coordinate: {coord}")
                        return coord
                return None
            else:
                 self.logger.debug(f"No data source or dataset loaded for file: {file_path}")
                 return None
        else:
            self.logger.debug(f"Could not determine file path for source '{source}' and findex {self.findex}")
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
        if not compare_ids:
            return

        # Create a mapping of exp_ids to their indices
        exp_id_indices = {}
        for i, entry in enumerate(self.app_data.inputs):
            if 'exp_id' in entry:
                exp_id_indices[entry['exp_id']] = i

        # Process all comparison IDs
        for i, exp_id in enumerate(compare_ids):
            exp_id = exp_id.strip()
            if exp_id in exp_id_indices:
                if i == 0:  # First ID goes to a_list
                    self.a_list.append(exp_id_indices[exp_id])
                else:  # All other IDs go to b_list
                    self.b_list.append(exp_id_indices[exp_id])
            else:
                self.logger.warning(f"Could not find entry for exp_id: {exp_id}")

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
            # Ensure 'filename' key exists before accessing
            if 'filename' in entry and (filename == entry['filename'] or os.path.basename(filename) == os.path.basename(entry['filename'])):
                return i
        self.logger.warning(f"File index not found for filename: {filename}, returning 0")
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
            return self.input_config.file_list[file]['description']
        except (KeyError, IndexError, TypeError) as e:
            self.logger.debug(f'Unable to get file description: {e}')
            return None

    def get_file_exp_name(self, i):
        """ Get user-defined experiment name associated with the input file (default None)"""
        try:
            return self.input_config.file_list[i]['exp_name']
        except Exception as e:
            self.logger.debug(f'key error {e}, returning default')
            return None

    def get_file_exp_id(self, i):
        """ Get user-defined experiment ID associated with the input file (default None)
        If an expid is set, then it will be used to compare with another expid, as set in compare field
        """
        try:
            return self.input_config.file_list[i]['exp_id']
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
        if self.ds_index < len(self.source_names):
            source = self.source_names[self.ds_index]
            if attr_name in self.meta_attrs and source in self.meta_attrs[attr_name]:
                return self.meta_attrs[attr_name][source]
            else:
                self.logger.debug(f"No meta_attrs mapping for attribute '{attr_name}' and source '{source}'")
                return None
        else:
            self.logger.debug(f"ds_index {self.ds_index} out of bounds for source_names {self.source_names}")
            return None


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
        # Access ds_index from the config object
        return self.config._ds_index

    @ds_index.setter
    def ds_index(self, value):
        # Set ds_index on the config object
        self.config._ds_index = value

    @property
    def findex(self):
        # Access findex from the config object
        return self.config._findex

    @findex.setter
    def findex(self, value):
        # Set findex on the config object
        self.config._findex = value


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
        # This attribute seems to be missing in the provided code, defaulting to False
        return getattr(self.config, '_add_logo', False)

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

    # Added properties for pindex, axindex, ax_opts, level, time_level, real_time
    # These seem to be state variables used during plotting, better managed elsewhere
    # but keeping them for now to match the original code's usage pattern.
    @property
    def pindex(self):
        return getattr(self.config, '_pindex', 0)

    @pindex.setter
    def pindex(self, value):
        self.config._pindex = value

    @property
    def axindex(self):
        return getattr(self.config, '_axindex', 0)

    @axindex.setter
    def axindex(self, value):
        self.config._axindex = value

    @property
    def ax_opts(self):
        return getattr(self.config, '_ax_opts', {})

    @ax_opts.setter
    def ax_opts(self, value):
        self.config._ax_opts = value

    @property
    def level(self):
        return getattr(self.config, '_level', None)

    @level.setter
    def level(self, value):
        self.config._level = value

    @property
    def time_level(self):
        return getattr(self.config, '_time_level', 0)

    @time_level.setter
    def time_level(self, value):
        self.config._time_level = value

    @property
    def real_time(self):
        return getattr(self.config, '_real_time', None)

    @real_time.setter
    def real_time(self, value):
        self.config._real_time = value
