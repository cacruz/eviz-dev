import json
import sys
import random
import string

import numpy as np
import xarray as xr
import argparse
import yaml
import logging

import eviz.lib.utils as u

logger = logging.getLogger(__name__)


def is_plottable(ds, var, space_coords, zc, tc):
    var_dims = set(ds[var].dims)
    # Check for 2D space (lon, lat)
    if space_coords.issubset(var_dims) and len(var_dims) == 2:
        return True
    # Check for 3D space (lon, lat, lev)
    if space_coords.union({zc}).issubset(var_dims) and len(var_dims) == 3:
        return True
    # Check for 4D space-time (lon, lat, lev, time)
    if space_coords.union({zc, tc}).issubset(var_dims) and len(var_dims) == 4:
        return True
    # Check for 2D space-time (lon, lat, time)
    if space_coords.union({tc}).issubset(var_dims) and len(var_dims) == 3:
        return True
    # Check for static 2D space (lon, lat)
    if space_coords.issubset(var_dims) and len(var_dims) == 2:
        return True
    # Check for static 3D space (lon, lat, lev)
    if space_coords.union({zc}).issubset(var_dims) and len(var_dims) == 3:
        return True
    return False


def has_multiple_time_levels(ds, var, tc):
    if tc in ds[var].dims:
        # Check if the time dimension has more than one level
        time_dim_index = ds[var].dims.index(tc)
        return ds[var].shape[time_dim_index] > 1
    return False


def attribute_contains(ds, var, substring):
    return any(substring in attr for attr in ds[var].attrs)


def json_compatible(value):
    """ Ensure all values in metadata are JSON-compatible
        Apply recursively to each item as needed
    """
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.int32, np.int64)):
        return int(value)
    elif isinstance(value, (np.int16, np.int64)):
        return int(value)
    elif isinstance(value, np.ndarray):
        return [json_compatible(v) for v in value]
    elif isinstance(value, list):
        return [json_compatible(v) for v in value]
    elif isinstance(value, dict):
        return {k: json_compatible(v) for k, v in value.items()}
    else:
        return value


def get_model_dim_name(dims, dim_name, meta_coords, source='generic'):
    if dim_name not in meta_coords:
        return None  # Ensure dim_name exists in meta_coords

    dim_data = meta_coords[dim_name]

    if source in dim_data:
        source_data = dim_data[source]
        if isinstance(source_data, dict):  # Handle nested dict for wrf, lis, etc.
            coords = source_data.get('dim', '')  # Use 'dim' key
        else:
            coords = source_data  # Directly use the value for sources without nested dicts
    else:
        return None  # Source not found in meta_coords[dim_name]

    if isinstance(coords, str) and ',' in coords:
        coords_list = coords.split(',')
        for item in coords_list:
            if item in dims:
                return item
    elif coords in dims:
        return coords

    return None  # No matching dimension found


def metadump(filepath_1,
             filepath_2=None, app_output=None, specs_output=None,
             json_output=None, ignore_vars=None, vars=None, source='generic'):

    dataset = xr.open_dataset(filepath_1, decode_cf=True)
    if filepath_2 is not None:
        dataset_2 = xr.open_dataset(filepath_2, decode_cf=True)
        vars_ds1 = set(dataset.data_vars.keys())
        vars_ds2 = set(dataset_2.data_vars.keys())
        if vars_ds1 != vars_ds2:
            return False

    meta_coords = u.read_meta_coords()
    tc = get_model_dim_name(dataset.dims, 'tc', meta_coords, source)
    xc = get_model_dim_name(dataset.dims, 'xc', meta_coords, source)
    yc = get_model_dim_name(dataset.dims, 'yc', meta_coords, source)
    zc = get_model_dim_name(dataset.dims, 'zc', meta_coords, source)
    space_coords = {xc, yc}

    if json_output:
        metadata = {
            "global_attributes": {k: json_compatible(v) for k, v in dataset.attrs.items()},
            "variables": {}
        }
        if ignore_vars:
            for var_name, da in dataset.data_vars.items():
                for substring in ignore_vars:
                    if substring not in var_name:
                        metadata["variables"][var_name] = {
                            "dimensions": list(da.dims),
                            "data_type": str(da.dtype),
                            "attributes": {k: json_compatible(v) for k, v in da.attrs.items()}
                        }
        else:
            for var_name, da in dataset.data_vars.items():
                metadata["variables"][var_name] = {
                    "dimensions": list(da.dims),
                    "data_type": str(da.dtype),
                    "attributes": {k: json_compatible(v) for k, v in da.attrs.items()}
                }

        with open("ds_metadata.json", "w") as json_file:
            json.dump(metadata, json_file, indent=4)
        logger.debug("Saved ds_metadata.json")
        return

    specs_dict = {}
    app_dict = {
        "inputs": [{"name": filepath_1, "to_plot": {}}],
        "outputs": {"print_to_file": "yes",
                    "output_dir": None,
                    "print_format": "png",
                    "print_basic_stats": True,
                    "make_pdf": False},
        "system_opts": {"use_mp_pool": False,
                        "archive_web_results": True}
    }

    if filepath_2 is not None:
        characters = string.ascii_uppercase + string.ascii_lowercase + string.digits
        exp_id_1 = ''.join(random.choice(characters) for _ in range(10))
        exp_id_2 = ''.join(random.choice(characters) for _ in range(10))
        app_dict['inputs'].append({"name": filepath_2, "to_plot": {}, "location": None, "exp_id": exp_id_2, "exp_name": None})
        app_dict['inputs'][0]['exp_id'] = exp_id_1
        app_dict['inputs'][0]['exp_name'] = None
        app_dict['for_inputs'] = {"compare":
                                      {"ids": f"{exp_id_1}, {exp_id_2}"},
                                  "cmap": "coolwarm"
                                  }

    # Metadata
    coordinate_variables = list(dataset.coords)
    logger.debug(f"Coordinate variables: {coordinate_variables}")
    data_variables = list(dataset.data_vars)
    logger.debug(f"Data variables: {data_variables}")
    global_attributes = dataset.attrs
    logger.debug(f"Global attributes: {global_attributes}")

    all_variables = list(dataset.variables)
    dimensions = list(dataset.dims)
    non_dimension_variables = [var for var in all_variables if var not in dimensions]
    logger.debug(f"Non-dim variables: {non_dimension_variables}")
    if vars:
        plottable_vars = vars
    else:
        plottable_vars = [var for var in dataset.data_vars if is_plottable(dataset, var, space_coords, zc, tc)]
    logger.debug(f"Plottable variables: {plottable_vars}")

    filtered_vars = []
    for var in plottable_vars:
        should_ignore = False
        if ignore_vars:
            for substring in ignore_vars:
                if substring in var:
                    should_ignore = True
                    break
        if not should_ignore:
            filtered_vars.append(var)

    if not (specs_output or app_output):
        logger.info(f"Plottable variables: {filtered_vars}")
        return

    # filtered_vars = [var for var in plottable_vars if not any(substring in var for substring in ignore_vars)]
    logger.info(f"Filtered variables: {filtered_vars}")

    for var_name in filtered_vars:
        default_lev = float(dataset.coords[zc][0].values) if zc is not None else 0
        if (vars is None or var_name in vars) and (ignore_vars is None or var_name not in ignore_vars):
            var = dataset[var_name]

            n_non_time_dims = len([dim for dim in var.dims if dim != tc])
            # adjust n_non_time_dims for cases where zc is just one value (one level)
            if zc is not None and len(dataset.coords[zc]) == 1:
                    n_non_time_dims -= 1

            temp_dict = {}
            if attribute_contains(dataset, var_name, 'units'):
                units = dataset[var_name].attrs['units']
                temp_dict.update({'units': units})
            if attribute_contains(dataset, var_name, 'long_name'):
                name = dataset[var_name].attrs['long_name']
                temp_dict.update({'name': name})

            plot_types = []
            if has_multiple_time_levels(dataset, var_name, tc):
                temp_dict['xtplot'] = {
                    "time_lev": "all",
                    "grid": "yes",
                }
                plot_types.append("xt")

            if n_non_time_dims >= 2:
                temp_dict['xyplot'] = dict(levels={default_lev: []})
                if tc is not None:
                    if dataset[tc].ndim > 1:
                        temp_dict['xyplot']['time_lev'] = 1
                plot_types.append("xy")

            if n_non_time_dims >= 3 and all("soil_layers" not in dim for dim in var.dims):
                temp_dict['yzplot'] = dict(contours=[])
                if tc is not None:
                    if dataset[tc].ndim > 1:
                        temp_dict['yzplot']['time_lev'] = 1
                plot_types.append("yz")

            specs_dict[var_name] = temp_dict
            app_dict["inputs"][0]["to_plot"][var_name] = ",".join(plot_types)

            if var.attrs.get("long_name", "") != "":
                description = var.attrs.get("long_name", "")
            else:
                description = var.attrs.get("description", "")
            units = var.attrs.get("units", "")
            dimensions = ', '.join(var.dims)

    dataset.close()

    if specs_output:
        with open(specs_output, 'w') as file:
            for key in sorted(specs_dict.keys()):
                value = specs_dict[key]
                yaml_content = yaml.dump({key: value}, default_flow_style=False)
                yaml_content = yaml_content.replace("'yes'", "yes")
                file.write(yaml_content + '\n')

    if app_output:
        with open(app_output, 'w') as file:
            app_dict["inputs"][0]["to_plot"] = \
                {key: app_dict["inputs"][0]["to_plot"][key] for key in sorted(app_dict["inputs"][0]["to_plot"])}
            yaml_content = yaml.dump(app_dict, default_flow_style=False)
            yaml_content = yaml_content.replace("'yes'", "yes")
            file.write(yaml_content + '\n')


def parse_command_line() -> argparse.Namespace:
    """Parse command line arguments.

    Example:

        >>> python metadump /path/to/file.nc
        >>> python metadump /path/to/file.nc --json
        >>> python metadump /path/to/file.nc --app foo.yaml --specs foo_specs.yaml
        >>> python metadump /path/to/file.nc --app foo.yaml --specs foo_specs.yaml --ignore Var
        >>> python metadump /path/to/file.nc --app foo.yaml --specs foo_specs.yaml --vars var1 var2 var3
        >>> python metadump /path/to/file.nc --source wrf

    Notes:
        (1) Prints "filtered" plottable variables to STDOUT
        (2) Writes metadata to json file
        (3) Creates specified app and specs files
        (4) Creates specified app and specs files with ignored subset
        (5) Creates specified app and specs files with specified vars
        (6) Processes a file with a non-'generic' source metadata
            Sources are generic (default), wrf, lis

        Once app and specs files are created one can run autoviz as follows:

           python autoviz.py -s <source_name> -f /path/to/app.yaml

    Returns:
        parser: populated namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Dump xarray file content.',
                                     usage='%(prog)s <netcdf_file> [netcdf_file2] [--specs <specs_output_file>]'
                                           ' [--app [<app_output_file>]] [--ignore var1 var2 ...]'
                                           ' [--vars var1 var2 ...]')
    parser.add_argument('filepaths', nargs='+', help='The netCDF file(s) to dump. Provide one or two file paths.')
    parser.add_argument('--specs', nargs='?', const=True, default=None,
                        help='specs file to output to. If not provided, it will be the filename with a _specs.yaml '
                             'extension.')
    parser.add_argument('--app', nargs='?', const=True, default=None,
                        help='App file to output to. If not provided, it will be the filename with a .yaml extension.')
    parser.add_argument('--json', nargs='?', const='ds_metadata.json', default=None,
                        help='JSON file to output to (default is ds_metadata.json).')
    parser.add_argument('--ignore', nargs='*', default=None,
                        help='Variables to ignore when generating YAML files.')
    parser.add_argument('--vars', nargs='*', default=None,
                        help='Variables to include when generating YAML files. If not provided, all variables are '
                             'included.')
    parser.add_argument('--source', nargs='?', default='generic',
                        help='source name (default is generic).')
    return parser.parse_args()


def main():
    """
    Driver to generate source metadata and, optionally, YAML files required by autoviz

    To run:
        python metadump.py filepath [options] 

    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s :: (%(funcName)s:%(lineno)d) : %(message)s")
    args = parse_command_line()

    if len(args.filepaths) > 2:
        logger.error("Error: Only one or two file paths are allowed.")
        sys.exit(1)

    filepath_1 = args.filepaths[0]
    filepath_2 = args.filepaths[1] if len(args.filepaths) == 2 else None

    if args.specs is True:
        args.specs = args.filepath.split("/")[-1] + "_specs.yaml"

    if args.app is True:
        args.app = args.filepath.split("/")[-1] + "not_used.yaml"

    if args.ignore is not None and args.vars is not None:
        logger.error("Error: --ignore and --vars cannot be used together.")
        sys.exit(1)

    metadump(filepath_1,
             filepath_2=filepath_2, app_output=args.app, specs_output=args.specs,
             json_output=args.json, ignore_vars=args.ignore, vars=args.vars, source=args.source)


if __name__ == "__main__":
    main()
