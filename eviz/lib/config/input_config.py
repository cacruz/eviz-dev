import sys
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from eviz.lib.data.sources.base import DataSource
from eviz.lib.utils import join_file_path
from eviz.lib.autoviz.utils import get_subplot_shape
from eviz.lib.config.app_data import AppData
from eviz.lib.data.factory.source_factory import DataSourceFactory

@dataclass
class InputConfig:
    source_names: List[str]
    config_files: List[str]
    app_data: AppData = field(default_factory=AppData)
    file_list: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    readers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compare: bool = False
    compare_diff: bool = False
    overlay: bool = False
    correlation: bool = False
    config_manager: Optional[Any] = None

    # Public state (for external access, e.g. ConfigManager)
    use_trop_height: bool = field(default=False, init=False)
    use_sphum_conv: bool = field(default=False, init=False)
    compare_exp_ids: List[str] = field(default_factory=list, init=False)
    overlay_exp_ids: List[str] = field(default_factory=list, init=False)
    extra_diff_plot: bool = field(default=False, init=False)
    profile: bool = field(default=False, init=False)
    # TODO: remove cmap from here and use value from output_config
    cmap: str = field(default="rainbow", init=False)
    comp_panels: tuple = field(default=(1, 1), init=False)
    subplot_specs: tuple = field(default=(1, 1), init=False)
    use_cartopy: bool = field(default=False, init=False)
    box_colors: List[str] = field(default_factory=list, init=False)
    add_legend: bool = field(default=False, init=False)
    shared_cbar: bool = field(default=False, init=False)

    # Internal state
    _file_reader_mapping: Dict[str, str] = field(default_factory=dict, init=False)
    _file_format_mapping: Dict[str, str] = field(default_factory=dict, init=False)
    _trop_height_file_list: dict = field(default_factory=dict, init=False)
    _sphum_conv_file_list: dict = field(default_factory=dict, init=False)

    logger = logging.getLogger(__name__)

    def initialize(self):
        """Initialize input configuration."""
        self._get_file_list()
        self._init_file_list_to_plot()
        self._init_readers()
        self._init_for_inputs()

    def _get_file_list(self):
        """Get all specified input files from the `inputs` section of the AppData object."""
        if not self.app_data.inputs:
            self.logger.error("The 'inputs' section in the AppData object is empty or missing.")
            sys.exit()
        for i, entry in enumerate(self.app_data.inputs):
            filename = join_file_path(entry.get('location', ''), entry['name'])
            self.file_list[i] = entry
            self.file_list[i]['filename'] = filename
            if 'format' in entry:
                self._file_format_mapping[filename] = entry['format']

    def get_format_for_file(self, file_path: str) -> Optional[str]:
        """
        Get the format specified for a file.
        
        Args:
            file_path (str): The path to the file
            
        Returns:
            Optional[str]: The format or None if not specified
        """
        return self._file_format_mapping.get(file_path)

    def _init_file_list_to_plot(self):
        """Create the list of files that contain the data to be plotted."""
        if getattr(self, '_use_history', False):
            if not getattr(self, '_history_year', None) or not getattr(self, '_history_month', None):
                self.logger.error("You need to provide the year and/or the month.")
                sys.exit()
            return None
        else:
            if len(self.file_list) == 1:
                self.compare = False
                self.compare_diff = False

    def _create_data_source(self, file_path: str, source_name: str, reader_type: Optional[str] = None) -> DataSource:
        """
        Create a data source for the given file path and source name.
        
        Args:
            file_path (str): Path to the data file
            source_name (str): Name of the source
            reader_type (Optional[str]): Optional reader type
            
        Returns:
            DataSource: The created data source
        """
        factory = DataSourceFactory(self.config_manager)
        file_format = self._file_format_mapping.get(file_path)
        return factory.create_data_source(
            file_path=file_path,
            model_name=source_name,
            reader_type=reader_type,
            file_format=file_format
        )

    def _init_reader_structure(self):
        """Initialize the structure for multiple readers per source."""
        for source_name in self.source_names:
            self.readers.setdefault(source_name, {})

    def _determine_reader_types(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Determine the reader types needed for each source and group files by reader type.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary mapping source names to lists of 
            file entries grouped by reader type
        """
        reader_mapping = {source_name: [] for source_name in self.source_names}
        for file_entry in self.file_list.values():
            file_path = file_entry['filename']
            reader_type = self._get_reader_type_for_extension(file_path)
            if reader_type:
                for source_name in self.source_names:
                    reader_mapping[source_name].append({'file_entry': file_entry, 'reader_type': reader_type})
        return reader_mapping

    def _init_readers(self):
        """ Initialize readers based on file extensions, special cases, or explicit reader field.
        """
        self._init_reader_structure()
        file_reader_types = {}
        for file_entry in self.file_list.values():
            file_path = file_entry['filename']
            explicit_reader = file_entry.get('reader')
            explicit_format = file_entry.get('format')
            if explicit_format:
                reader_type = self._get_reader_type_from_format(explicit_format)
                self._file_format_mapping[file_path] = explicit_format
            else:
                reader_type = self._get_reader_type_for_extension(file_path, explicit_reader)
            if reader_type:
                file_reader_types[file_path] = reader_type
                self._file_reader_mapping[file_path] = reader_type
        for source_name in self.source_names:
            processed_extensions = set()
            for file_path, reader_type in file_reader_types.items():
                file_extension = os.path.splitext(file_path)[1]
                if (reader_type, file_extension) in processed_extensions:
                    continue
                if reader_type not in self.readers[source_name]:
                    self.readers[source_name][reader_type] = self._get_reader(source_name, file_extension, reader_type)
                processed_extensions.add((reader_type, file_extension))

    def _get_reader_type_from_format(self, format_str: str) -> str:
        """
        Determine the appropriate reader type based on format string.
        
        Args:
            format_str (str): The format string (e.g., 'netcdf', 'csv', 'hdf5')
            
        Returns:
            str: The reader type identifier
        """
        format_lower = format_str.lower()
        if format_lower in ['netcdf', 'nc', 'nc4']:
            return 'NetCDF'
        elif format_lower in ['csv', 'text', 'txt', 'dat']:
            return 'CSV'
        elif format_lower in ['hdf5', 'h5', 'he5']:
            return 'HDF5'
        elif format_lower in ['hdf4', 'hdf']:
            return 'HDF4'
        elif format_lower in ['zarr']:
            return 'ZARR'
        elif format_lower in ['grib', 'grib2']:
            return 'GRIB'
        else:
            self.logger.warning(f"Unknown format: {format_str}, defaulting to NetCDF")
            return 'NetCDF'

    def _get_reader_type_for_extension(self, file_path: str, explicit_reader: str = None) -> str:
        """
        Determine the appropriate reader type based on file extension
        
        Args:
            file_path (str): The file path which might include wildcards
                
        Returns:
            str: The reader type identifier ('NetCDF', 'CSV', 'HDF5', or 'HDF4') or None if unknown
        """
        if explicit_reader:
            return explicit_reader
        path_lower = file_path.lower()
        if 'wrf' in path_lower or 'wrfout' in path_lower:
            return 'NetCDF'
        if any(keyword in path_lower for keyword in ['hdf5', '.h5', '.he5']):
            return 'HDF5'
        if any(keyword in path_lower for keyword in ['hdf4', '.hdf']):
            if not any(ext_key in path_lower for ext_key in ['.h5', '.he5', 'hdf5']):
                return 'HDF4'
        file_extension = os.path.splitext(path_lower)[1]
        if file_extension in ['.nc', '.nc4', '']:
            return 'NetCDF'
        elif file_extension in ['.csv', '.dat']:
            return 'CSV'
        elif file_extension in ['.h5', '.he5']:
            return 'HDF5'
        elif file_extension == '.hdf':
            return 'HDF4'
        elif file_extension == '.zarr':
            return 'ZARR'
        elif file_extension.startswith('.wrf'):
            return 'NetCDF'
        elif file_extension in ['.grib', '.grib2']:
            return 'GRIB'
        else:
            if any(x in path_lower for x in ['netcdf', 'nc']):
                return 'NetCDF'
            elif any(x in path_lower for x in ['csv', 'data', 'txt']):
                return 'CSV'
        return 'NetCDF'

    def get_reader_for_file(self, source_name: str, file_path: str):
        """
        Get the appropriate reader for a file.
        
        Args:
            source_name (str): The name of the data source
            file_path (str): The path to the file
            
        Returns:
            The appropriate reader or None if not found
        """
        reader_type = self._file_reader_mapping.get(file_path)
        if reader_type and source_name in self.readers and reader_type in self.readers[source_name]:
            return self.readers[source_name][reader_type]
        if source_name in self.readers and self.readers[source_name]:
            return next(iter(self.readers[source_name].values()))
        return None

    def get_primary_reader(self, source_name: str):
        """
        Get the primary reader for a source.
        
        Args:
            source_name (str): The name of the data source
            
        Returns:
            The primary reader or None if not found
        """
        if source_name in self.readers:
            if 'NetCDF' in self.readers[source_name]:
                return self.readers[source_name]['NetCDF']
            elif self.readers[source_name]:
                return next(iter(self.readers[source_name].values()))
        return None

    def _get_reader(self, source_name: str, file_extension: str, reader_type: str = None) -> Any:
        """
        Return the appropriate reader based on file extension using the factory.

        Args:
            source_name (str): The name of the data source.
            file_extension (str): The file extension of the input file.
            reader_type (str): optional reader type to use.

        Returns:
            Any: An instance of the appropriate data source class.
        """
        dummy_path = f"dummy{file_extension}"
        try:
            return self._create_data_source(
                file_path=dummy_path,
                source_name=source_name,
                reader_type=reader_type
            )
        except ValueError as e:
            if 'wrf' in source_name.lower():
                return self._create_data_source("dummy.nc", source_name)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")

    def _init_for_inputs(self) -> None:
        """Initialize parameters in the `for_inputs` section of the AppData object."""
        for_inputs = getattr(self.app_data, 'for_inputs', {})
        self.use_cartopy = for_inputs.get('use_cartopy', False)
        self.comp_panels = for_inputs.get('comp_panels', (1, 1))
        self.subplot_specs = for_inputs.get('subplot_specs', (1, 1))
        self.box_colors = for_inputs.get('box_colors', False)
        self.add_legend = for_inputs.get('add_legend', False)
        self.shared_cbar = for_inputs.get('shared_cbar', False)

        self._parse_for_inputs(for_inputs)
        if self.compare and self.compare_diff:
            self.logger.warning("Both 'compare' and 'compare_diff' are set to True. Setting 'compare' to False.")
            self.compare = False
        if self.compare_diff:
            extra_diff = (
                for_inputs.get('extra_diff') or
                for_inputs.get('compare_diff', {}).get('extra_diff_plot') or
                for_inputs.get('compare_diff', {}).get('extra_diff', False)
            )
            self.extra_diff_plot = extra_diff
            self.profile = for_inputs.get('compare_diff', {}).get('profile', False)
            # TODO: app level cmap: move to outputs?
            self.cmap = for_inputs.get('compare_diff', {}).get('cmap', 'rainbow')
            self.comp_panels = (2, 2) if self.extra_diff_plot else (3, 1)
        elif self.compare:
            self.profile = for_inputs.get('compare', {}).get('profile', False)
            # TODO: app level cmap: move to outputs?
            self.cmap = for_inputs.get('compare', {}).get('cmap', 'rainbow')
            self.comp_panels = get_subplot_shape(len(self.compare_exp_ids))

        if self.overlay and (self.compare or self.compare_diff):
            self.compare = False
            self.compare_diff = False

        self.use_trop_height = 'trop_height' in for_inputs
        if self.use_trop_height:
            self._set_trop_height_file_list()

    def get_all_variables(self, source_name: str) -> Dict[str, Any]:
        """
        Get all available variables across all readers for a source.
        
        Args:
            source_name (str): The name of the data source
            
        Returns:
            Dict[str, Any]: A dictionary of all variables
        """
        all_vars = {}
        if source_name in self.readers:
            for reader_type, reader in self.readers[source_name].items():
                try:
                    vars_dict = reader.get_variables()
                    for var_name, var_data in vars_dict.items():
                        var_data['source_reader'] = reader_type
                        all_vars[var_name] = var_data
                except Exception as e:
                    self.logger.error(f"Error getting variables from {reader_type} reader: {str(e)}")
        return all_vars

    def get_metadata(self, source_name: str, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a specific file.
        
        Args:
            source_name (str): The name of the data source
            file_path (str): The path to the file
            
        Returns:
            Dict[str, Any]: Metadata for the file
        """
        reader = self.get_reader_for_file(source_name, file_path)
        if reader:
            try:
                return reader.get_metadata(file_path)
            except Exception as e:
                self.logger.error(f"Error getting metadata: {str(e)}")
        return {}

    def _parse_for_inputs(self, for_inputs):
        """Parse the for_inputs section of the configuration."""
        if not for_inputs:
            return
        self.correlation = False
        self.overlay = False
        self.compare = False
        self.compare_diff = False
        self.compare_exp_ids = []
        self.overlay_exp_ids = []

        if 'correlation' in for_inputs:
            self.correlation = True
            correlation_config = for_inputs['correlation']
            if 'ids' in correlation_config:
                self.compare_exp_ids = correlation_config['ids'].split(',')
            self.correlation_method = correlation_config.get('method', 'pearson')
            self.space_corr = correlation_config.get('space_corr', False)
            self.time_corr = correlation_config.get('time_corr', False)

        if 'overlay' in for_inputs:
            self.overlay = True
            overlay_config = for_inputs['overlay']
            if 'ids' in overlay_config:
                self.overlay_exp_ids = overlay_config['ids'].split(',')
                self.compare_exp_ids = self.overlay_exp_ids.copy()
            if 'box_colors' in overlay_config:
                self.box_colors = overlay_config['box_colors'].split(',')
            if 'add_legend' in overlay_config:
                self.add_legend = overlay_config['add_legend']

        if 'compare' in for_inputs:
            self.compare = True
            compare_config = for_inputs['compare']
            if 'ids' in compare_config:
                self.compare_exp_ids = compare_config['ids'].split(',')
                
        elif 'compare_diff' in for_inputs:
            self.compare_diff = True
            compare_diff_config = for_inputs['compare_diff']
            if 'ids' in compare_diff_config:
                self.compare_exp_ids = compare_diff_config['ids'].split(',')

    def to_dict(self) -> dict:
        """Return a dictionary representation of the input configuration."""
        return {
            "source_names": self.source_names,
            "config_files": self.config_files,
            "app_data": self.app_data.__dict__,
            "file_list": self.file_list,
            "readers": {key: str(reader) for key, reader in self.readers.items()},
            "overlay": self.overlay,
            "compare": self.compare,
            "compare_diff": self.compare_diff,
            "correlation": self.correlation,
            "overlay_exp_ids": self.overlay_exp_ids,
            "compare_exp_ids": self.compare_exp_ids,
            "extra_diff_plot": self.extra_diff_plot,
            "profile": self.profile,
            "cmap": self.cmap,
            "comp_panels": self.comp_panels,
            "use_trop_height": self.use_trop_height,
            "subplot_specs": self.subplot_specs,
            "use_cartopy": self.use_cartopy,
            "file_formats": self._file_format_mapping,
        }

    def _set_trop_height_file_list(self):
        """ Get all specified tropopause height files from the top level "app" YAML file """
        n_files = len(self.app_data.for_inputs['trop_height'])
        if n_files > 0:
            for i in range(n_files):
                entry = self.app_data.for_inputs['trop_height'][i]
                filename = os.path.join(entry.get('location', '/'), entry['name'])
                self._trop_height_file_list[i] = entry
                self._trop_height_file_list[i]['filename'] = filename
                self._trop_height_file_list[i]['exp_name'] = entry['exp_id']
                self._trop_height_file_list[i]['trop_field_name'] = entry['trop_field_name']
        else:
            self.use_trop_height = False

    def _set_sphum_conv_file_list(self):
        """ Get all specified specific humidity files from the top level "app" YAML file """
        if self.use_sphum_conv and 'sphum_field' in self.app_data.for_inputs:
            n_files = len(self.app_data.for_inputs['sphum_field'])
            for i in range(n_files):
                entry = self.app_data.for_inputs['sphum_field'][i]
                filename = os.path.join(entry.get('location', '/'), entry['name'])
                self._sphum_conv_file_list[i] = entry
                self._sphum_conv_file_list[i]['filename'] = filename
                self._sphum_conv_file_list[i]['exp_name'] = entry['exp_id']
                self._sphum_conv_file_list[i]['trop_field_name'] = entry['sphum_field_name']
        else:
            self.use_sphum_conv = False

    @property
    def trop_height_file_list(self):
        return self._trop_height_file_list

    @property
    def sphum_conv_file_list(self):
        return self._sphum_conv_file_list
