from dataclasses import dataclass, field
import logging
import os
from typing import Optional, List, Dict, Any
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
        input_config: Configuration for input data sources and parameters
        output_config: Configuration for output settings and file generation
        system_config: Configuration for system-level settings
        history_config: Configuration for tracking history and versioning
        config: Base configuration object containing shared settings
        a_list: List of indices for the first set of comparison items
        b_list: List of indices for the second set of comparison items
        _findex: Current file index being processed
        _ds_index: Current data source index being processed
        _units: Reference to the units conversion system (lazy loaded)
        _integrator: Reference to the data integrator (lazy loaded)
        _pipeline: Reference to the data processing pipeline (lazy loaded)
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
    _findex: int = 0
    _ds_index: int = 0
    current_field_name: str = ""

    # Fields not included in __init__
    _units: Optional[object] = field(default=None, init=False)
    _integrator: Optional[DataIntegrator] = field(default=None, init=False)
    _pipeline: Optional[DataPipeline] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize the ConfigManager after construction."""
        self.input_config.config_manager = self  # CC: Is this necessary?
        self.setup_comparison()

    @property
    def logger(self):
        """Return the logger for this class."""
        return logging.getLogger(__name__)

    @property
    def paths(self):
        """Access to paths configuration."""
        return self.config.paths

    @property
    def app_data(self):
        """Access to application data."""
        return self.config.app_data

    @property
    def spec_data(self):
        """Access to specification data."""
        return self.config.spec_data

    @property
    def source_names(self):
        """Access to source names."""
        return self.config.source_names

    @property
    def ds_index(self):
        """Get the current data source index."""
        return self._ds_index

    @ds_index.setter
    def ds_index(self, value):
        """Set the data source index and update config if needed."""
        self._ds_index = value
        # Also update config if it has the attribute
        if hasattr(self.config, '_ds_index'):
            self.config._ds_index = value

    @property
    def findex(self):
        """Get the current file index."""
        return self._findex

    @findex.setter
    def findex(self, value):
        """Set the file index and update config if needed."""
        self._findex = value
        # Also update config if it has the attribute
        if hasattr(self.config, '_findex'):
            self.config._findex = value

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

    def to_dict(self) -> Dict[str, Any]:
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
        Dynamically access attributes from the underlying config objects.
        
        This method is called only when an attribute is not found through normal lookup.
        It searches for the attribute in the config objects in a specific order.
        
        Args:
            name: The name of the attribute to look up
            
        Returns:
            The value of the attribute if found
            
        Raises:
            AttributeError: If the attribute is not found in any config object
        """
        # Check if the attribute exists in this instance directly
        if name in self.__dict__:
            return self.__dict__[name]

        # Check in config first
        if hasattr(self.config, name):
            return getattr(self.config, name)

        # Then check in other config objects
        for config in [self.input_config, self.output_config, self.system_config,
                       self.history_config]:
            if hasattr(config, name):
                return getattr(config, name)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def should_overlay_plots(self, field_name, plot_type):
        """
        Determine if comparison plots for this field should be overlaid on the same axes.
        
        Args:
            field_name (str): The name of the field
            plot_type (str): The type of plot (e.g., 'yz', 'xt')
            
        Returns:
            bool: True if plots should be overlaid, False otherwise
        """
        # Only consider overlaying for profile plot, box plots,  and time series
        if plot_type not in ['yz', 'xt', 'bo']:
            return False
            
        # Sanity check
        is_profile = False
        if plot_type == 'yz' and field_name in self.spec_data:
            is_profile = 'profile_dim' in self.spec_data[field_name].get('yzplot', {})
        
        # Return True if this is a profile or time series and overlay is requested
        return (is_profile or plot_type == 'xt' or plot_type == 'bo') and self.overlay

    def get_file_format(self, file_path: str) -> Optional[str]:
        """
        Get the format for a specific file.
        
        Args:
            file_path (str): The path to the file
            
        Returns:
            Optional[str]: The format or None if not specified
        """
        if hasattr(self.input_config, 'get_format_for_file'):
            return self.input_config.get_format_for_file(file_path)
        return None

    @property
    def file_formats(self) -> Dict[str, str]:
        """
        Get a dictionary mapping file paths to their formats.
        
        Returns:
            Dict[str, str]: Dictionary mapping file paths to formats
        """
        if hasattr(self.input_config, '_file_format_mapping'):
            return self.input_config._file_format_mapping
        return {}

    def get_model_dim_name(self, dim_name):
        """
        Get model-specific dimension name associated with the source as defined
        in meta_coordinates.yaml.

        Args:
            dim_name (str): The generic dimension name to look up.

        Returns:
            str or None: The model-specific dimension name, or None if not found.
        """
        source = self.source_names[self.ds_index]
        # Try with the current ds_index first
        result = self._get_model_dim_name_for_source(dim_name, source)
        if result:
            return result
        
        # If that fails, try with all sources
        for i, src in enumerate(self.source_names):
            if i != self.ds_index:  # Skip the one we already tried
                result = self._get_model_dim_name_for_source(dim_name, src)
                if result:
                    return result
        
        # If all else fails, try some common dimension names
        common_dims = {
            'xc': ['lon', 'longitude', 'x'],
            'yc': ['lat', 'latitude', 'y'],
            'zc': ['lev', 'level', 'z', 'altitude', 'height', 'plev'],
            'tc': ['time', 't']
        }
        
        if dim_name in common_dims:
            # Get all data sources
            all_sources = self.pipeline.get_all_data_sources()
            for source in all_sources.values():
                if hasattr(source, 'dataset') and source.dataset is not None:
                    dims = list(source.dataset.dims)
                    for common_dim in common_dims[dim_name]:
                        if common_dim in dims:
                            return common_dim
        
        return None

    def _get_model_dim_name_for_source(self, dim_name, source):
        """Helper method to get model dimension name for a specific source."""
        file_path = self._get_current_file_path(source)
        if not file_path:
            return None

        data_source = self._get_data_source_for_file(file_path)
        if not data_source:
            return None

        dims = self._get_available_dimensions(data_source)
        if not dims:
            return None

        if dim_name not in self.meta_coords:
            return None
        if source not in self.meta_coords[dim_name]:
            return None

        if source in ['wrf', 'lis']:
            coords = self.meta_coords[dim_name][source].get('dim', '')
        else:
            coords = self.meta_coords[dim_name][source]

        if ',' in coords:
            coords_list = [c.strip() for c in coords.split(',')]
            for item in coords_list:
                if item in dims:
                    return item
        else:
            return coords if coords in dims else None

        return None

    def _get_current_file_path(self, source):
        """
        Get the file path for the current file index or source name.
        
        Args:
            source (str): The source name to look for if file index is invalid
            
        Returns:
            str or None: The file path if found, None otherwise
        """
        try:
            if self.findex is not None and self.findex < len(self.app_data.inputs):
                file_entry = self.app_data.inputs[self.findex]
                return os.path.join(file_entry.get('location', ''),
                                    file_entry.get('name', ''))
        except Exception as e:
            self.logger.debug(
                f"Could not get file path for ds_index {self.ds_index}, findex {self.findex}: {e}")

        # Fallback to searching by source name
        for entry in self.app_data.inputs:
            if entry.get('source_name') == source:
                return os.path.join(entry.get('location', ''), entry.get('name', ''))

        self.logger.debug(
            f"Could not determine file path for source '{source}' and findex {self.findex}")
        return None

    def _get_data_source_for_file(self, file_path):
        """
        Get the data source for a file path.
        
        Args:
            file_path (str): The file path to get the data source for
            
        Returns:
            object or None: The data source if found, None otherwise
        """
        if not file_path or not self.pipeline:
            return None

        data_source = self.pipeline.get_data_source(file_path)
        if not data_source or not hasattr(data_source,
                                          'dataset') or data_source.dataset is None:
            self.logger.debug(f"No data source or dataset loaded for file: {file_path}")
            return None

        return data_source

    @staticmethod
    def _get_available_dimensions(data_source):
        """
        Get the available dimensions from a data source.
        
        Args:
            data_source (object): The data source to get dimensions from
            
        Returns:
            list or None: The list of available dimensions if found, None otherwise
        """
        if not data_source or not hasattr(data_source,
                                          'dataset') or data_source.dataset is None:
            return None

        available_dims = list(data_source.dataset.dims.keys())
        return available_dims

    def setup_comparison(self):
        """
        Set up comparison between datasets based on config settings.
        Creates a_list and b_list for comparing items identified by exp_id.
        """
        self.a_list = []
        self.b_list = []

        if not (self.input_config._compare or self.input_config._compare_diff or self.input_config._overlay):
            self.logger.debug("Comparison not enabled")
            return

        compare_ids = self.input_config._compare_exp_ids or self.input_config._overlay_exp_ids or []
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
            filename (str): The exact filename to search for

        Returns:
            int: Index of the file in app_data.inputs, or 0 if not found
        """
        if not filename:
            self.logger.warning("Empty filename provided, returning 0")
            return 0

        for i, entry in enumerate(self.app_data.inputs):
            # Check if 'filename' key exists before accessing
            if 'filename' in entry and (filename == entry['filename'] or
                                        os.path.basename(filename) == os.path.basename(
                        entry['filename'])):
                return i

        self.logger.warning(f"File index not found for filename: {filename}, returning 0")
        return 0

    def get_levels(self, to_plot, plot_type):
        """
        Get model levels to plot from YAML specs file.
        
        Args:
            to_plot (str): The field to plot
            plot_type (str): The type of plot
            
        Returns:
            list: The levels to plot, or an empty list if not found
        """
        levels = u.get_nested_key_value(self.spec_data, [to_plot, plot_type, 'levels'])
        return levels if levels else []

    def get_file_description(self, file):
        """
        Get user-defined file description.
        
        Args:
            file (int or str): The file index or name
            
        Returns:
            str or None: The file description if found, None otherwise
        """
        try:
            return self.input_config.file_list[file]['description']
        except (KeyError, IndexError, TypeError) as e:
            self.logger.debug(f'Unable to get file description: {e}')
            return None

    def get_file_exp_name(self, i):
        """
        Get user-defined experiment name associated with the input file.
        
        Args:
            i (int): The file index
            
        Returns:
            str or None: The experiment name if found, None otherwise
        """
        try:
            return self.input_config.file_list[i]['exp_name']
        except Exception as e:
            self.logger.debug(f'Key error {e}, returning default')
            return None

    def get_file_exp_id(self, i):
        """
        Get user-defined experiment ID associated with the input file.
        If an expid is set, then it will be used to compare with another expid, as set in compare field.
        
        Args:
            i (int): The file index
            
        Returns:
            str or None: The experiment ID if found, None otherwise
        """
        try:
            return self.input_config.file_list[i]['exp_id']
        except Exception as e:
            self.logger.debug(f'Key error {e}, returning default')
            return None

    def get_dim_names(self, pid):
        """
        Get dimension names for a specific plot type.
        
        Args:
            pid (str): The plot ID
            
        Returns:
            tuple: A tuple of (dim1, dim2) dimension names
        """
        dim1, dim2 = None, None
        if 'yz' in pid:
            dim1, dim2 =  self.get_model_dim_name('yc'), self.get_model_dim_name('zc')
        elif 'xt' in pid:
            dim1, dim2 = self.get_model_dim_name('tc'), None
        elif 'tx' in pid:
            dim1, dim2 = self.get_model_dim_name('xc'), self.get_model_dim_name('tc')
        else:
            dim1, dim2 = self.get_model_dim_name('xc'), self.get_model_dim_name('yc')
        return dim1, dim2

    def get_model_attr_name(self, attr_name):
        """
        Get model-specific attribute name associated with the source as defined
        in meta_attributes.yaml.
        
        Args:
            attr_name (str): The attribute name to look up
            
        Returns:
            str or None: The model-specific attribute name, or None if not found
        """
        if self.ds_index >= len(self.source_names):
            self.logger.debug(
                f"ds_index {self.ds_index} out of bounds for source_names {self.source_names}")
            return None

        source = self.source_names[self.ds_index]
        if attr_name in self.meta_attrs and source in self.meta_attrs[attr_name]:
            return self.meta_attrs[attr_name][source]
        else:
            self.logger.debug(
                f"No meta_attrs mapping for attribute '{attr_name}' and source '{source}'")
            return None

    def register_plot_type(self, field_name, plot_type):
        """Register the plot type for a field."""
        if not hasattr(self, '_plot_type_registry'):
            self._plot_type_registry = {}
        self._plot_type_registry[field_name] = plot_type
    
    def get_plot_type(self, field_name, default='xy'):
        """Get the plot type for a field."""
        if hasattr(self, '_plot_type_registry') and field_name in self._plot_type_registry:
            return self._plot_type_registry[field_name]
        return default

    def get_file_index_by_filename(self, filename: str) -> int:
        """Return the file_index associated with a filename from map_params.
        
        Args:
            filename: The filename to search for
            
        Returns:
            The file_index if found, or -1 if the filename is not found
        """
        for params in self.config.map_params.values():
            if params['filename'] == filename:
                return params['file_index']
        return -1

    # Properties that delegate to config objects
    # These are defined explicitly to provide better documentation and type hints

    @property
    def map_params(self):
        """Access to map parameters."""
        return self.config.map_params

    @property
    def overlay(self):
        """Flag indicating if overlay mode is active."""
        return self.input_config._overlay

    @property
    def compare(self):
        """Flag indicating if comparison mode is active."""
        return self.input_config._compare

    @property
    def compare_diff(self):
        """Flag indicating if difference comparison mode is active."""
        return self.input_config._compare_diff

    @property
    def extra_diff_plot(self):
        """Flag indicating if extra difference plots should be generated."""
        return self.input_config._extra_diff_plot

    @property
    def shared_cbar(self):
        """Use a shared colorbar in comparison plots."""
        return self.input_config._shared_cbar

    @property
    def pearsonplot(self):
        """The pearsonplot options to use."""
        return self.input_config._pearsonplot

    @property
    def add_legend(self):
        """Add legend to the box plots"""
        return self.input_config._add_legend

    @property
    def box_colors(self):
        """List of colors used in box plots"""
        return self.input_config._box_colors

    @property
    def plot_backend(self):
        """The backend to use for plotting."""
        return self.input_config._plot_backend

    @property
    def cmap(self):
        """The colormap to use for plotting."""
        return self.input_config._cmap

    @property
    def use_cartopy(self):
        """Flag indicating if cartopy should be used for plotting."""
        return self.input_config._use_cartopy

    @property
    def have_specs_yaml_file(self):
        """Flag indicating if a specs YAML file exists."""
        return self.config._specs_yaml_exists

    @property
    def meta_coords(self):
        """Access to metadata for coordinate systems."""
        return self.config.meta_coords

    @property
    def meta_attrs(self):
        """Access to metadata for attributes."""
        return self.config.meta_attrs

    @property
    def species_db(self):
        """Access to the database of chemical species information."""
        return self.config.species_db

    @property
    def trop_height_file_list(self):
        """Access to the list of tropopause height files."""
        return self.input_config._trop_height_file_list

    @property
    def sphum_conv_file_list(self):
        """Access to the list of specific humidity conversion files."""
        return self.input_config._sphum_conv_file_list

    @property
    def use_trop_height(self):
        """Flag indicating if tropopause height should be used."""
        return self.input_config._use_trop_height

    @use_trop_height.setter
    def use_trop_height(self, value):
        """Set the use_trop_height flag."""
        self.input_config._use_trop_height = value

    @property
    def use_sphum_conv(self):
        """Flag indicating if specific humidity conversion should be used."""
        return self.input_config._use_sphum_conv

    @property
    def add_logo(self):
        """Flag indicating if a logo should be added to plots."""
        return self.output_config.add_logo

    @property
    def print_to_file(self):
        """Flag indicating if output should be printed to a file."""
        return self.output_config.print_to_file

    @property
    def output_dir(self):
        """The directory to write output files to."""
        return self.output_config.output_dir

    @property
    def print_format(self):
        """The format to use for printing output."""
        return self.input_config.print_format

    @property
    def make_gif(self):
        """Flag indicating if GIFs should be generated."""
        return self.output_config.make_gif

    @property
    def gif_fps(self):
        """The frames per second to use for GIFs."""
        return self.output_config.gif_fps

    @property
    def make_pdf(self):
        """Flag indicating if PDFs should be generated."""
        return self.output_config.make_pdf

    @property
    def mpl_style(self):
        """The matplotlib style to use for plotting."""
        return self.output_config.mpl_style

    @property
    def print_basic_stats(self):
        """Flag indicating if basic statistics should be printed."""
        return self.output_config.print_basic_stats

    @property
    def use_mp_pool(self):
        """Flag indicating if multiprocessing should be used."""
        return self.system_config.use_mp_pool

    @property
    def archive_web_results(self):
        """Flag indicating if web results should be archived."""
        return self.system_config.archive_web_results

    @property
    def to_plot(self):
        """The fields to plot."""
        return self.input_config._to_plot

    @property
    def overlay_exp_ids(self):
        """The experiment IDs to overlay."""
        return self.input_config._overlay_exp_ids

    @property
    def compare_exp_ids(self):
        """The experiment IDs to compare."""
        return self.input_config._compare_exp_ids

    # State variables used during plotting
    @property
    def pindex(self):
        """The current plot index."""
        return getattr(self.config, '_pindex', 0)

    @pindex.setter
    def pindex(self, value):
        """Set the current plot index."""
        self.config._pindex = value

    @property
    def axindex(self):
        """The current axis index."""
        return getattr(self.config, '_axindex', 0)

    @axindex.setter
    def axindex(self, value):
        """Set the current axis index."""
        self.config._axindex = value

    @property
    def ax_opts(self):
        """The current axis options."""
        return getattr(self.config, '_ax_opts', {})

    @ax_opts.setter
    def ax_opts(self, value):
        """Set the current axis options."""
        self.config._ax_opts = value

    @property
    def level(self):
        """The current vertical level."""
        return getattr(self.config, '_level', None)

    @level.setter
    def level(self, value):
        """Set the current vertical level."""
        self.config._level = value

    @property
    def time_level(self):
        """The current time level."""
        return getattr(self.config, '_time_level', 0)

    @time_level.setter
    def time_level(self, value):
        """Set the current time level."""
        self.config._time_level = value

    @property
    def real_time(self):
        """The human-readable representation of the current time."""
        return getattr(self.config, '_real_time', None)

    @real_time.setter
    def real_time(self, value):
        """Set the human-readable representation of the current time."""
        self.config._real_time = value
