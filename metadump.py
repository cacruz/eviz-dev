#!/usr/bin/env python3
"""
MetaDump - A tool to generate metadata and YAML configuration files for autoviz from NetCDF files.

This module analyzes NetCDF files and generates the necessary configuration files for visualization
with autoviz. It can process single files or pairs of files for comparison, and generates:
- JSON metadata files describing the dataset contents
- YAML specification files for plot configurations
- YAML application files for autoviz execution parameters
"""
import json
import sys
import random
import string
from typing import Optional, Dict, List, Set, Tuple, Any, Union
from dataclasses import dataclass
import logging
import textwrap
import argparse
import yaml
import numpy as np
import xarray as xr

import eviz.lib.utils as u
from eviz.lib.constants import SUPPORTED_MODELS, META_ATTRS_NAME, META_COORDS_NAME, SPECIES_DB_NAME

logger = logging.getLogger(__name__)


@dataclass
class MetadumpConfig:
    """Configuration settings for metadata extraction."""
    filepath_1: str
    filepath_2: Optional[str] = None
    app_output: Optional[str] = None
    specs_output: Optional[str] = None
    json_output: Optional[str] = None
    ignore_vars: Optional[List[str]] = None
    vars: Optional[List[str]] = None
    source: str = 'gridded'

class MetadataExtractor:
    """Main class for extracting metadata and generating configuration files."""
    
    def __init__(self, config: MetadumpConfig):
        """Initialize the metadata extractor with configuration settings."""
        self.config = config
        self.dataset = self._open_dataset(config.filepath_1)
        self.dataset_2 = self._open_dataset(config.filepath_2) if config.filepath_2 else None
        self.meta_coords = u.read_meta_coords()
        self._setup_coordinates()
        
        if self.dataset_2:
            self._validate_datasets()

    def _open_dataset(self, filepath: Optional[str]) -> Optional[xr.Dataset]:
        """Open an xarray dataset from a file."""
        if not filepath:
            return None
        try:
            return xr.open_dataset(filepath, decode_cf=True)
        except Exception as e:
            logger.error(f"Failed to open dataset {filepath}: {e}")
            raise RuntimeError(f"Could not open dataset: {e}")

    def _setup_coordinates(self) -> None:
        """Set up coordinate references based on the dataset."""
        self.tc = self._get_model_dim_name('tc')
        self.xc = self._get_model_dim_name('xc')
        self.yc = self._get_model_dim_name('yc')
        self.zc = self._get_model_dim_name('zc')
        self.space_coords = {self.xc, self.yc}

    def _get_model_dim_name(self, dim_name: str) -> Optional[str]:
        """Get the model-specific dimension name."""
        return get_model_dim_name(self.dataset.dims, dim_name, 
                                self.meta_coords, self.config.source)

    def _validate_datasets(self) -> None:
        """Validate that two datasets are compatible for comparison."""
        vars_ds1 = set(self.dataset.data_vars.keys())
        vars_ds2 = set(self.dataset_2.data_vars.keys())
        if vars_ds1 != vars_ds2:
            raise ValueError("Datasets have different variables")

    def process(self) -> None:
        """Main processing method to generate all required outputs."""
        if self.config.json_output:
            self._generate_json_metadata()
            return

        specs_dict = self._generate_specs_dict()
        app_dict = self._generate_app_dict()

        if self.config.specs_output:
            self._write_specs_yaml(specs_dict)
        if self.config.app_output:
            self._write_app_yaml(app_dict)

        # Print plottable variables if no output files specified
        if not (self.config.specs_output or self.config.app_output):
            filtered_vars = self.get_plottable_vars()
            logger.info(f"Plottable variables: {filtered_vars}")

    def _generate_json_metadata(self) -> None:
        """Generate and save JSON metadata for the dataset."""
        metadata = {
            "global_attributes": self._get_json_compatible_attrs(self.dataset.attrs),
            "variables": {}
        }

        for var_name, da in self.dataset.data_vars.items():
            if self._should_include_var(var_name):
                metadata["variables"][var_name] = {
                    "dimensions": list(da.dims),
                    "data_type": str(da.dtype),
                    "attributes": self._get_json_compatible_attrs(da.attrs)
                }

        with open(self.config.json_output or "ds_metadata.json", "w") as json_file:
            json.dump(metadata, json_file, indent=4)
        logger.debug(f"Saved metadata to {self.config.json_output or 'ds_metadata.json'}")

    def _get_json_compatible_attrs(self, attrs: Dict) -> Dict:
        """Convert attributes to JSON-compatible format."""
        return {k: json_compatible(v) for k, v in attrs.items()}

    def _should_include_var(self, var_name: str) -> bool:
        """Determine if a variable should be included based on configuration."""
        if self.config.vars:
            return var_name in self.config.vars
        if self.config.ignore_vars:
            return not any(substring in var_name for substring in self.config.ignore_vars)
        return True

    def get_plottable_vars(self) -> List[str]:
        """Get list of plottable variables based on configuration."""
        if self.config.vars:
            return self.config.vars

        plottable = [var for var in self.dataset.data_vars 
                    if is_plottable(self.dataset, var, self.space_coords, 
                                  self.zc, self.tc)]
        
        if self.config.ignore_vars:
            plottable = [var for var in plottable 
                        if not any(substring in var 
                                 for substring in self.config.ignore_vars)]
        
        return plottable

    def _generate_specs_dict(self) -> Dict:
        """Generate the specifications dictionary for YAML output."""
        specs_dict = {}
        plottable_vars = self.get_plottable_vars()

        for var_name in plottable_vars:
            var = self.dataset[var_name]
            specs_dict[var_name] = self._process_variable(var_name, var)

        return specs_dict

    def _process_variable(self, var_name: str, var: xr.DataArray) -> Dict:
        """Process a single variable and return its metadata dictionary."""
        temp_dict = {}
        
        # Add metadata if available
        if 'units' in var.attrs:
            temp_dict['units'] = var.attrs['units']
        if 'long_name' in var.attrs:
            temp_dict['name'] = var.attrs['long_name']

        # Determine plot types
        n_non_time_dims = len([dim for dim in var.dims if dim != self.tc])
        if self.zc and len(self.dataset.coords[self.zc]) == 1:
            n_non_time_dims -= 1

        # Add plot configurations
        if has_multiple_time_levels(self.dataset, var_name, self.tc):
            temp_dict['xtplot'] = {
                "time_lev": "all",
                "grid": "yes",
            }

        if n_non_time_dims >= 2:
            default_lev = float(self.dataset.coords[self.zc][0].values) if self.zc else 0
            temp_dict['xyplot'] = dict(levels={default_lev: []})
            if self.tc and self.dataset[self.tc].ndim > 1:
                temp_dict['xyplot']['time_lev'] = 1

        if n_non_time_dims >= 3 and all("soil_layers" not in dim for dim in var.dims):
            temp_dict['yzplot'] = dict(contours=[])
            if self.tc and self.dataset[self.tc].ndim > 1:
                temp_dict['yzplot']['time_lev'] = 1

        return temp_dict

    def _generate_app_dict(self) -> Dict:
        """Generate the application dictionary for YAML output."""
        app_dict = {
            "inputs": [{
                "name": self.config.filepath_1,
                "to_plot": self._get_plot_types()
            }],
            "outputs": {
                "print_to_file": "yes",
                "output_dir": None,
                "print_format": "png",
                "print_basic_stats": True,
                "make_pdf": False
            },
            "system_opts": {
                "use_mp_pool": False,
                "archive_web_results": True
            }
        }

        if self.config.filepath_2:
            self._add_comparison_config(app_dict)

        return app_dict

    def _get_plot_types(self) -> Dict[str, str]:
        """Get plot types for each plottable variable."""
        plot_types = {}
        for var_name in self.get_plottable_vars():
            types = []
            if has_multiple_time_levels(self.dataset, var_name, self.tc):
                types.append("xt")
            if is_plottable(self.dataset, var_name, self.space_coords, self.zc, self.tc):
                types.append("xy")
                if self.zc and "soil_layers" not in var_name:
                    types.append("yz")
            plot_types[var_name] = ",".join(types)
        return plot_types

    def _add_comparison_config(self, app_dict: Dict) -> None:
        """Add comparison configuration for two-file cases."""
        exp_id_1 = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        exp_id_2 = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        
        app_dict['inputs'][0]['exp_id'] = exp_id_1
        app_dict['inputs'][0]['exp_name'] = None
        
        app_dict['inputs'].append({
            "name": self.config.filepath_2,
            "to_plot": {},
            "location": None,
            "exp_id": exp_id_2,
            "exp_name": None
        })
        
        app_dict['for_inputs'] = {
            "compare": {"ids": f"{exp_id_1}, {exp_id_2}"},
            "cmap": "coolwarm"
        }

    def _write_specs_yaml(self, specs_dict: Dict) -> None:
        """Write specifications dictionary to YAML file."""
        with open(self.config.specs_output, 'w') as file:
            for key in sorted(specs_dict.keys()):
                yaml_content = yaml.dump({key: specs_dict[key]}, 
                                       default_flow_style=False)
                yaml_content = yaml_content.replace("'yes'", "yes")
                file.write(yaml_content + '\n')

    def _write_app_yaml(self, app_dict: Dict) -> None:
        """Write application dictionary to YAML file."""
        with open(self.config.app_output, 'w') as file:
            if 'inputs' in app_dict and 'to_plot' in app_dict['inputs'][0]:
                app_dict['inputs'][0]['to_plot'] = \
                    {k: app_dict['inputs'][0]['to_plot'][k] 
                     for k in sorted(app_dict['inputs'][0]['to_plot'])}
            yaml_content = yaml.dump(app_dict, default_flow_style=False)
            yaml_content = yaml_content.replace("'yes'", "yes")
            file.write(yaml_content + '\n')

def json_compatible(value: Any) -> Any:
    """Convert values to JSON-compatible format."""
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.int32, np.int64, np.int16)):
        return int(value)
    elif isinstance(value, np.ndarray):
        return [json_compatible(v) for v in value]
    elif isinstance(value, list):
        return [json_compatible(v) for v in value]
    elif isinstance(value, dict):
        return {k: json_compatible(v) for k, v in value.items()}
    return value

def is_plottable(ds: xr.Dataset, var: str, 
                 space_coords: Set[str], zc: Optional[str], 
                 tc: Optional[str]) -> bool:
    """Determine if a variable is plottable."""
    var_dims = set(ds[var].dims)
    
    # Check for various dimension combinations
    if space_coords.issubset(var_dims) and len(var_dims) == 2:
        return True  # 2D space (lon, lat)
    if space_coords.union({zc}).issubset(var_dims) and len(var_dims) == 3:
        return True  # 3D space (lon, lat, lev)
    if space_coords.union({zc, tc}).issubset(var_dims) and len(var_dims) == 4:
        return True  # 4D space-time (lon, lat, lev, time)
    if space_coords.union({tc}).issubset(var_dims) and len(var_dims) == 3:
        return True  # 2D space-time (lon, lat, time)
    
    return False

def has_multiple_time_levels(ds: xr.Dataset, var: str, tc: Optional[str]) -> bool:
    """Check if variable has multiple time levels."""
    if tc and tc in ds[var].dims:
        time_dim_index = ds[var].dims.index(tc)
        return ds[var].shape[time_dim_index] > 1
    return False

def get_model_dim_name(dims: List[str], dim_name: str, 
                      meta_coords: Dict, source: str = 'gridded') -> Optional[str]:
    """Get the model-specific dimension name."""
    if dim_name not in meta_coords:
        return None

    dim_data = meta_coords[dim_name]
    if source not in dim_data:
        return None

    source_data = dim_data[source]
    coords = source_data.get('dim', '') if isinstance(source_data, dict) else source_data

    if isinstance(coords, str) and ',' in coords:
        coords_list = coords.split(',')
        for item in coords_list:
            if item in dims:
                return item
    elif coords in dims:
        return coords

    return None

def parse_command_line() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate metadata and YAML configuration files for autoviz from NetCDF files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples:
          python metadump.py /path/to/file.nc
          python metadump.py /path/to/file.nc --json
          python metadump.py /path/to/file.nc --app foo.yaml --specs foo_specs.yaml
          python metadump.py /path/to/file.nc --app foo.yaml --specs foo_specs.yaml --ignore Var
          python metadump.py /path/to/file.nc --app foo.yaml --specs foo_specs.yaml --vars var1 var2 var3
          python metadump.py /path/to/file.nc --source wrf
        ''')
    )
    
    parser.add_argument('filepaths', nargs='+', 
                       help='The netCDF file(s) to process. Provide one or two file paths.')
    parser.add_argument('--specs', nargs='?', const=True, default=None,
                       help='Specs file to output to. If not provided, it will be the filename with a _specs.yaml extension.')
    parser.add_argument('--app', nargs='?', const=True, default=None,
                       help='App file to output to. If not provided, it will be the filename with a .yaml extension.')
    parser.add_argument('--json', nargs='?', const='ds_metadata.json', default=None,
                       help='JSON file to output to (default is ds_metadata.json).')
    parser.add_argument('--ignore', nargs='*', default=None,
                       help='Variables to ignore when generating YAML files.')
    parser.add_argument('--vars', nargs='*', default=None,
                       help='Variables to include when generating YAML files. If not provided, all variables are included.')
    parser.add_argument('--source', nargs='?', default='gridded',
                       help='Source name (default is gridded).')
    
    return parser.parse_args()

def main():
    """Main entry point for the metadump tool."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s :: (%(funcName)s:%(lineno)d) : %(message)s"
    )
    
    args = parse_command_line()

    if len(args.filepaths) > 2:
        logger.error("Error: Only one or two file paths are allowed.")
        sys.exit(1)

    try:
        config = MetadumpConfig(
            filepath_1=args.filepaths[0],
            filepath_2=args.filepaths[1] if len(args.filepaths) == 2 else None,
            app_output=args.app,
            specs_output=args.specs,
            json_output=args.json,
            ignore_vars=args.ignore,
            vars=args.vars,
            source=args.source
        )
        
        extractor = MetadataExtractor(config)
        extractor.process()
        
    except Exception as e:
        logger.error(f"Error processing metadata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
