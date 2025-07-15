from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
import logging
from eviz.lib.utils import get_nested_key
import eviz.lib.utils as u


@dataclass
class YAMLParser:
    config_files: List[str]
    source_names: List[str]
    app_data: Dict[str, Any] = field(default_factory=dict)
    spec_data: Dict[str, Any] = field(default_factory=dict)
    _map_params: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    meta_attrs: dict = field(default_factory=dict)
    meta_coords: dict = field(default_factory=dict)
    _ds_index: int = 0
    _specs_yaml_exists: bool = True

    def parse(self):
        """Parse YAML files and populate app_data and spec_data."""
        concat = self._concatenate_yaml()
        self._init_map_params(concat)
        self.meta_coords = u.read_meta_coords()
        self.meta_attrs = u.read_meta_attrs()
        self.species_db = u.read_species_db()

        # for key, value in self.to_dict().items():
        #     self.logger.debug(f"{key} = {value}")

    def _concatenate_yaml(self) -> List[Dict[str, Any]]:
        """Read and merge multiple YAML files and their associated specs."""
        concat = []
        result = {}
        output_dirs = []

        _source_index = 0
        for index, file_path in enumerate(self.config_files):
            self.ds_index = _source_index
            yaml_content = u.load_yaml_simple(file_path)
            yaml_content['source'] = self.source_names[index]
            concat.append(yaml_content)

            if 'inputs' in yaml_content:
                result.setdefault('inputs', []).extend(yaml_content['inputs'])
            else:
                self.logger.error(f"No inputs specified in {file_path}.")
                raise ValueError(f"No inputs specified in {file_path}.")

            if 'for_inputs' in yaml_content:
                result.setdefault('for_inputs', {}).update(yaml_content['for_inputs'])

            if 'system_opts' in yaml_content:
                result.setdefault('system_opts', {}).update(yaml_content['system_opts'])

            if 'outputs' in yaml_content:
                result.setdefault('outputs', {}).update(yaml_content['outputs'])

            if 'geos' in self.source_names and len(set(self.source_names)) == 1:
                if 'history' in yaml_content:
                    result.setdefault('history', {}).update(yaml_content['history'])

            if 'outputs' in yaml_content and 'output_dir' in yaml_content['outputs']:
                output_dirs.append(yaml_content['outputs']['output_dir'])

            specs_file = os.path.join(os.path.dirname(file_path), f"{os.path.splitext(os.path.basename(file_path))[0]}_specs.yaml")
            if os.path.exists(specs_file):
                specs_content = u.load_yaml_simple(specs_file)
                self.spec_data.update(specs_content)
            else:
                self.logger.warning(f"Specs file not found for {file_path}: {specs_file}")
                self._specs_yaml_exists = False
            _source_index += 1

        self.app_data = result
        return concat

    def _init_map_params(self, concat: List[Dict[str, Any]]):
        """Organize data for plotting routines."""
        _maps = {}

        def process_input_dict(in_dict: Dict[str, Any]):
            current_inputs = in_dict.get('inputs', [])
            current_outputs = in_dict.get('outputs', {})
            
            compare_ids = get_nested_key(self.app_data, ['for_inputs', 'compare', 'ids'],
                                         default='')
            compare_diff_ids = get_nested_key(self.app_data,
                                              ['for_inputs', 'compare_diff', 'ids'],
                                              default='')
            
            if isinstance(compare_ids, str) and compare_ids:
                compare_ids = compare_ids.split(',')
            elif not isinstance(compare_ids, list):
                compare_ids = []
                
            if isinstance(compare_diff_ids, str) and compare_diff_ids:
                compare_diff_ids = compare_diff_ids.split(',')
            elif not isinstance(compare_diff_ids, list):
                compare_diff_ids = []

            for i, input_entry in enumerate(current_inputs):
                filename = os.path.join(input_entry.get('location', ''), input_entry.get('name', ''))
                exp_id = input_entry.get('exp_id', '')
                exp_name = input_entry.get('exp_name', exp_id)
                source_name = in_dict.get('source', '')
                source_reader = self.app_data.get(source_name, None)
                description = input_entry.get('description', in_dict.get('description', ''))
                to_ignore = input_entry.get('ignore', in_dict.get('ignore', ''))

                # Handle fields to plot - check both to_plot and variables sections
                current_to_plot = input_entry.get('to_plot', {})
                
                # If to_plot is empty, try to use the variables section (for CCM format)
                if not current_to_plot and 'variables' in input_entry:
                    current_to_plot = {}
                    for var_name, var_config in input_entry['variables'].items():
                        # Default to xy if not specified
                        plot_type = var_config.get('plot_type', 'xy')
                        current_to_plot[var_name] = plot_type
                
                if not current_to_plot:
                    self.logger.warning(f"No fields to plot found for {filename}")
                    continue
                
                source_index = self.source_names.index(source_name)
                for field_name, field_values in current_to_plot.items():
                    field_specs = get_nested_key(self.spec_data, [field_name], default={})
                    
                    is_in_compare = exp_id in compare_ids
                    is_in_compare_diff = exp_id in compare_diff_ids
                    
                    compare_with = []
                    if is_in_compare:
                        compare_with = [cid for cid in compare_ids if cid != exp_id]
                    elif is_in_compare_diff:
                        compare_with = [cid for cid in compare_diff_ids if cid != exp_id]
                    
                    # Handle different formats for field_values
                    to_plot_values = []
                    if isinstance(field_values, str):
                        to_plot_values = field_values.split(',')
                    elif isinstance(field_values, dict) and 'plot_type' in field_values:
                        # Handle CCM format where field_values is a dict with plot_type
                        to_plot_values = [field_values['plot_type']]
                    else:
                        # Default to xy if format is unknown
                        to_plot_values = ['xy']
                    
                    _maps[len(_maps)] = {
                        'source_name': source_name,
                        'source_reader': source_reader,
                        'source_index': source_index,
                        'file_index': i,
                        'field': field_name,
                        'filename': filename,
                        'description': description,
                        'ignore': to_ignore,
                        'exp_id': exp_id,
                        'exp_name': exp_name,
                        'to_plot': to_plot_values,
                        'compare': is_in_compare,
                        'compare_diff': is_in_compare_diff,
                        'compare_with': compare_with,
                        'outputs': current_outputs,
                        'field_specs': field_specs,
                    }

        for input_dict in concat:
            process_input_dict(input_dict)

        self._map_params = _maps

    @property
    def map_params(self) -> Dict[int, Dict[str, Any]]:
        """Return the organized map parameters."""
        return self._map_params

    @property
    def logger(self):
        """Return the logger for this class."""
        return logging.getLogger(__name__)
    
    def to_dict(self) -> dict:
        """Return a dictionary representation of the YAMLParser."""
        return {
            "config_files": self.config_files,
            "source_names": self.source_names,
            "app_data": self.app_data,
            "spec_data": self.spec_data, 
            "map_params": self._map_params,
        }
