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


# from eviz.lib.utils import log_method


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
    _trop_height_file_list: dict = field(default_factory=dict)
    _sphum_conv_file_list: dict = field(default_factory=dict)
    _use_trop_height: bool = False
    _use_sphum_conv: bool = False
    _file_reader_mapping: Dict[str, str] = field(default_factory=dict)
    _file_format_mapping: Dict[str, str] = field(default_factory=dict)
    config_manager: Optional[Any] = None  # CC: Is this necessary?

    _overlay: bool = field(default=False, init=False)
    _compare: bool = field(default=False, init=False)
    _compare_diff: bool = field(default=False, init=False)
    _overlay_exp_ids: List[str] = field(default_factory=list, init=False)
    _compare_exp_ids: List[str] = field(default_factory=list, init=False)
    _extra_diff_plot: bool = field(default=False, init=False)
    _profile: bool = field(default=False, init=False)
    _cmap: str = field(default="rainbow", init=False)
    _comp_panels: tuple = field(default=(1, 1), init=False)
    _subplot_specs: tuple = field(default=(1, 1), init=False)
    _use_cartopy: bool = field(default=False, init=False)

    # @log_method
    def initialize(self):
        """Initialize input configuration."""
        self._compare = self.compare
        self._compare_diff = self.compare_diff
        self._overlay = self.overlay

        self._get_file_list()
        self._init_file_list_to_plot()
        self._init_readers()
        self._init_for_inputs()  # Initialize for_inputs parameters

    def _get_file_list(self):
        """ Get all specified input files from the `inputs` section of the AppData object.
        """
        if not self.app_data.inputs:
            self.logger.error(
                "The 'inputs' section in the AppData object is empty or missing.")
            sys.exit()

        self.logger.debug(f"Processing {len(self.app_data.inputs)} input file entries.")
        for i, entry in enumerate(self.app_data.inputs):
            filename = join_file_path(entry.get('location', ''), entry['name'])
            self.file_list[i] = entry
            self.file_list[i]['filename'] = filename

            # Store the format if provided
            if 'format' in entry:
                self._file_format_mapping[filename] = entry['format']
                self.logger.debug(f"File format for {filename}: {entry['format']}")

            self.logger.debug(f"file_list[{i}] = {self.file_list[i]}")

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
        if getattr(self, '_use_history', False):  # Check if history is enabled
            if not getattr(self, '_history_year', None) or not getattr(self,
                                                                       '_history_month',
                                                                       None):
                self.logger.error("You need to provide the year and/or the month.")
                self.logger.error("\tEdit the config file and try again.")
                sys.exit()
            return None  # TODO: Implement self.make_dict(self.hist_list_files())
        else:
            if len(self.file_list) == 1:
                self.compare = False
                self.compare_diff = False

    def _create_data_source(self, file_path: str, source_name: str,
                            reader_type: Optional[str] = None) -> DataSource:
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
            if source_name not in self.readers:
                self.readers[source_name] = {}

    def _determine_reader_types(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Determine the reader types needed for each source and group files by reader type.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary mapping source names to lists of 
            file entries grouped by reader type
        """
        reader_mapping = {source_name: [] for source_name in self.source_names}

        for file_idx, file_entry in self.file_list.items():
            file_path = file_entry['filename']
            reader_type = self._get_reader_type_for_extension(file_path)

            if reader_type:
                for source_name in self.source_names:
                    reader_mapping[source_name].append({
                        'file_entry': file_entry,
                        'reader_type': reader_type
                    })

        return reader_mapping

    def _init_readers(self):
        """ Initialize readers based on file extensions, special cases, or explicit reader field.
        """
        self._init_reader_structure()

        file_reader_types = {}
        for file_idx, file_entry in self.file_list.items():
            file_path = file_entry['filename']
            explicit_reader = file_entry.get('reader', None)
            explicit_format = file_entry.get('format', None)

            # Use format to determine reader type if provided
            if explicit_format:
                reader_type = self._get_reader_type_from_format(explicit_format)
                self._file_format_mapping[file_path] = explicit_format
                self.logger.debug(
                    f"Using explicit format for {file_path}: {explicit_format} -> reader_type: {reader_type}")
            else:
                reader_type = self._get_reader_type_for_extension(file_path,
                                                                  explicit_reader)

            if reader_type:
                file_reader_types[file_path] = reader_type
                self._file_reader_mapping[file_path] = reader_type
            else:
                self.logger.warning(f"Could not determine reader type for {file_path}")

        for source_name in self.source_names:
            processed_extensions = set()
            for file_path, reader_type in file_reader_types.items():
                file_extension = os.path.splitext(file_path)[1]
                # One reader per extension type per source
                if (reader_type, file_extension) in processed_extensions:
                    continue
                try:
                    if reader_type not in self.readers[source_name]:
                        self.logger.info(
                            f"Setting up {reader_type} reader for source: {source_name}")
                        self.readers[source_name][reader_type] = self._get_reader(
                            source_name, file_extension, reader_type)
                    processed_extensions.add((reader_type, file_extension))
                except Exception as e:
                    self.logger.error(
                        f"Failed to initialize {reader_type} reader for {file_path}: {str(e)}")

        for source_name, readers in self.readers.items():
            self.logger.info(
                f"Initialized readers for source {source_name}: {list(readers.keys())}")

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

    def _get_reader_type_for_extension(self, file_path: str,
                                       explicit_reader: str = None) -> str:
        """
        Determine the appropriate reader type based on file path.
        
        Args:
            file_path (str): The file path which might include wildcards
                
        Returns:
            str: The reader type identifier ('NetCDF', 'CSV', 'HDF5', or 'HDF4') or None if unknown
        """
        if explicit_reader:
            return explicit_reader  # FIX: Use explicit_reader as is, no .capitalize()

        path_lower = file_path.lower()

        # Handle WRF special cases first, as they are NetCDF
        if 'wrf' in path_lower or 'wrfout' in path_lower:
            self.logger.info(f"Detected WRF file: {file_path}, using NetCDF reader")
            return 'NetCDF'

        # Prioritize HDF5/HDF4 name inference before checking ambiguous extensions
        # This helps with files like 'my_hdf5_data.dat' or 'log_containing_hdf4.log'

        # Check for HDF5 by name or common HDF5 extensions in the name string
        # Note: .he5 is HDF5, .h5 can be HDF5
        if any(keyword in path_lower for keyword in ['hdf5', '.h5', '.he5']):
            # Avoid misclassifying an HDF4 file (e.g. file.hdf) if 'hdf' from 'hdf5' matches
            if not (path_lower.endswith('.hdf') and not any(
                    ext_key in path_lower for ext_key in ['.h5', '.he5'])):
                self.logger.info(f"Inferred HDF5 from filename/keywords: {file_path}")
                return 'HDF5'

        # Check for HDF4 by name or common HDF4 extension in the name string
        # Note: .hdf is HDF4
        if any(keyword in path_lower for keyword in ['hdf4', '.hdf']):
            # Avoid misclassifying an HDF5 file (e.g. file.he5 or file_with_hdf5.hdf)
            if not any(ext_key in path_lower for ext_key in ['.h5', '.he5', 'hdf5']):
                self.logger.info(f"Inferred HDF4 from filename/keywords: {file_path}")
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
        elif file_extension.startswith('.wrf'):  # e.g., .wrf-arw, .wrf-arw-nmp
            return 'NetCDF'
        else:
            self.logger.warning(
                f"Unrecognized extension: {file_extension} in {file_path}, "
                f"attempting to infer type from name as a last resort.")
            if any(x in path_lower for x in ['netcdf', 'nc']):
                return 'NetCDF'
            elif any(x in path_lower for x in ['csv', 'data', 'txt']):
                return 'CSV'

        self.logger.error(
            f"Unsupported file extension: {file_extension} for file {file_path}. "
            f"Defaulting to NetCDF.")
        return 'NetCDF'

    def _determine_data_types(self) -> Dict[str, bool]:
        """Determine the data types needed based on file extensions."""
        data_types_needed = {}
        for i, file_entry in self.file_list.items():
            file_path = file_entry['filename']
            file_extension = os.path.splitext(file_path)[1]

            if 'wrf' in file_path:
                data_types_needed['NetCDF'] = True
            elif 'test' in file_path:
                data_types_needed['NetCDF'] = True
            else:
                if file_extension in ['.nc', '.nc4']:
                    data_types_needed['NetCDF'] = True
                elif file_extension in ['.csv', '.dat']:
                    data_types_needed['CSV'] = True
                elif file_extension in ['.h5', '.he5']:
                    data_types_needed['HDF5'] = True
                elif file_extension == '.zarr':
                    data_types_needed['ZARR'] = True
                elif file_extension == '.hdf':
                    data_types_needed['HDF4'] = True
                elif '*' in file_path:
                    data_types_needed['NetCDF'] = True
                else:
                    self.logger.error(f"Unsupported file type: {file_path}")
                    sys.exit(1)

        return data_types_needed

    def get_reader_for_file(self, source_name: str, file_path: str):
        """
        Get the appropriate reader for a file.
        
        Args:
            source_name (str): The name of the data source
            file_path (str): The path to the file
            
        Returns:
            The appropriate reader or None if not found
        """
        if file_path in self._file_reader_mapping:
            reader_type = self._file_reader_mapping[file_path]
            if source_name in self.readers and reader_type in self.readers[source_name]:
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
        # Try to get a NetCDF reader first...
        if source_name in self.readers:
            if 'NetCDF' in self.readers[source_name]:
                return self.readers[source_name]['NetCDF']
            elif self.readers[source_name]:
                return next(iter(self.readers[source_name].values()))

        return None

    def _get_reader(self, source_name: str, file_extension: str,
                    reader_type: str = None) -> Any:
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
            self.logger.error(f"Error creating data source: {e}")
            # Default to NetCDF for unrecognized extensions when WRF is involved
            if 'wrf' in source_name.lower():
                self.logger.debug(
                    f"Unrecognized extension '{file_extension}' "
                    f"for WRF source, defaulting to NetCDF reader")
                return self._create_data_source("dummy.nc", source_name)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")

    def _init_for_inputs(self) -> None:
        """Initialize parameters in the `for_inputs` section of the AppData object."""
        for_inputs = getattr(self.app_data, 'for_inputs', {})

        self._compare_exp_ids = []
        self._extra_diff_plot = False
        self._profile = False
        self._cmap = 'rainbow'
        self._use_cartopy = for_inputs.get('use_cartopy', False)
        self._comp_panels = for_inputs.get('comp_panels', (1, 1))
        self._subplot_specs = for_inputs.get('subplot_specs', (1, 1))

        # Parse for_inputs to set _compare and _compare_diff
        self._parse_for_inputs(for_inputs)

        # Ensure compare and compare_diff are mutually exclusive
        if self._compare and self._compare_diff:
            self.logger.warning("Both 'compare' and 'compare_diff' are set to True. "
                                "Setting 'compare' to False to make them mutually exclusive.")
            self._compare = False

        if self._compare_diff:
            extra_diff = (
                    for_inputs.get('extra_diff') or
                    for_inputs.get('compare_diff', {}).get(
                        '_extra_diff_plot') or
                    for_inputs.get('compare_diff', {}).get('extra_diff', False)
            )
            self._extra_diff_plot = extra_diff
            self._profile = for_inputs.get('compare_diff', {}).get('profile', False)
            self._cmap = for_inputs.get('compare_diff', {}).get('cmap', 'rainbow')
            self._comp_panels = (2, 2) if self._extra_diff_plot else (3, 1)
            self.logger.debug(
                f"Compare diff settings: extra_diff={extra_diff}, comp_panels={self._comp_panels}")

        elif self._compare:
            self._profile = for_inputs.get('compare', {}).get('profile', False)
            self._cmap = for_inputs.get('compare', {}).get('cmap', 'rainbow')
            self._comp_panels = get_subplot_shape(len(self._compare_exp_ids))

        # Sanity check
        if self.overlay and (self._compare or self._compare_diff):
            self._compare = False
            self._compare_diff = False

        self._use_trop_height = 'trop_height' in for_inputs
        if self._use_trop_height:
            self._set_trop_height_file_list()  # Custom method for trop_height logic

        self.logger.debug(f"Initialized for_inputs with: "
                          f"overlay={self._overlay}, "
                          f"compare={self._compare}, "
                          f"compare_diff={self._compare_diff}, "
                          f"compare_exp_ids={self._compare_exp_ids}, "
                          f"extra_diff_plot={self._extra_diff_plot}, "
                          f"profile={self._profile}, "
                          f"cmap={self._cmap}, "
                          f"comp_panels={self._comp_panels}, "
                          f"use_trop_height={self._use_trop_height}, "
                          f"subplot_specs={self._subplot_specs}, "
                          f"use_cartopy={self._use_cartopy}")

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
                    # Assuming each reader has a method to list variables
                    vars_dict = reader.get_variables()
                    for var_name, var_data in vars_dict.items():
                        var_data['source_reader'] = reader_type
                        all_vars[var_name] = var_data
                except Exception as e:
                    self.logger.error(
                        f"Error getting variables from {reader_type} reader: {str(e)}")

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

        self._overlay = False
        self._compare = False
        self._compare_diff = False
        self._compare_exp_ids = []

        if 'overlay' in for_inputs:
            self._overlay = True
            overlay_config = for_inputs['overlay']

            if 'ids' in overlay_config:
                self._overlay_exp_ids = overlay_config['ids'].split(',')

        if 'compare' in for_inputs:
            self._compare = True
            compare_config = for_inputs['compare']

            if 'ids' in compare_config:
                self._compare_exp_ids = compare_config['ids'].split(',')

        elif 'compare_diff' in for_inputs:
            self._compare_diff = True
            compare_diff_config = for_inputs['compare_diff']

            if 'ids' in compare_diff_config:
                self._compare_exp_ids = compare_diff_config['ids'].split(',')

    @property
    def logger(self):
        """Return the logger for this class."""
        return logging.getLogger(__name__)

    def to_dict(self) -> dict:
        """Return a dictionary representation of the input configuration."""
        result = {
            "source_names": self.source_names,
            "config_files": self.config_files,
            "app_data": self.app_data.__dict__,
            "file_list": self.file_list,
            "readers": {key: str(reader) for key, reader in self.readers.items()},
            "compare": self.compare,
            "compare_diff": self.compare_diff,
            "overlay_exp_ids": getattr(self, "_overlay_exp_ids", None),
            "compare_exp_ids": getattr(self, "_compare_exp_ids", None),
            "extra_diff_plot": getattr(self, "_extra_diff_plot", None),
            "profile": getattr(self, "_profile", None),
            "cmap": getattr(self, "_cmap", None),
            "comp_panels": getattr(self, "_comp_panels", None),
            "use_trop_height": getattr(self, "_use_trop_height", None),
            "subplot_specs": getattr(self, "_subplot_specs", None),
            "use_cartopy": getattr(self, "_use_cartopy", None),
            # Add file formats
            "file_formats": self._file_format_mapping,
        }
        return result

    # TODO: trop_height and sph_conv files are GEOS-specific
    # Therefore, GEOS-specific functionality should be moved to a ConfigGeos class
    def _set_trop_height_file_list(self):
        """ Get all specified tropopause height files from the top level "app" YAML file """
        n_files = len(self.app_data.for_inputs['trop_height'])
        if n_files > 0:
            self.logger.debug(f'Looping over {n_files} trop_height file entries:')
            for i in range(n_files):
                if 'location' in self.app_data.for_inputs['trop_height'][i]:
                    filename = os.path.join(
                        self.app_data.for_inputs['trop_height'][i]['location'],
                        self.app_data.for_inputs['trop_height'][i]['name'])
                else:
                    filename = os.path.join('/',
                                            self.app_data.for_inputs['trop_height'][i][
                                                'name'])
                self._trop_height_file_list[i] = self.app_data.for_inputs['trop_height'][
                    i]
                self._trop_height_file_list[i]['filename'] = filename
                self._trop_height_file_list[i]['exp_name'] = \
                    self.app_data.for_inputs['trop_height'][i]['exp_id']
                self._trop_height_file_list[i]['trop_field_name'] = \
                    self.app_data.for_inputs['trop_height'][i]['trop_field_name']
                self.logger.debug(self.trop_height_file_list[i])
        else:
            self._use_trop_height = False

    def _set_sphum_conv_file_list(self):
        """ Get all specified specific humidity files from the top level "app" YAML file """
        if self._use_sphum_conv:
            if 'sphum_field' in self.app_data.for_inputs:
                n_files = len(self.app_data.for_inputs['sphum_field'])
                self.logger.debug(f'Looping over {n_files} sphum_field file entries:')
                for i in range(n_files):
                    if 'location' in self.app_data.for_inputs['sphum_field'][i]:
                        filename = os.path.join(
                            self.app_data.for_inputs['sphum_field'][i]['location'],
                            self.app_data.for_inputs['sphum_field'][i]['name'])
                    else:
                        filename = os.path.join('/',
                                                self.app_data.for_inputs['sphum_field'][
                                                    i]['name'])
                    self._sphum_conv_file_list[i] = \
                        self.app_data.for_inputs['sphum_field'][i]
                    self._sphum_conv_file_list[i]['filename'] = filename
                    self._sphum_conv_file_list[i]['exp_name'] = \
                        self.app_data.for_inputs['sphum_field'][i]['exp_id']
                    self._sphum_conv_file_list[i]['trop_field_name'] = \
                        self.app_data.for_inputs['sphum_field'][i]['sphum_field_name']
                    self.logger.debug(self.sphum_conv_file_list[i])
            else:
                self._use_sphum_conv = False

    @property
    def trop_height_file_list(self):
        return self._trop_height_file_list

    @property
    def sphum_conv_file_list(self):
        return self._sphum_conv_file_list
