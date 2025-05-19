import os
import re
import sys
import errno
import yaml
import logging
import pathlib
import subprocess
from functools import wraps
from typing import Any, Dict, List
from yaml import SafeLoader
from datetime import timedelta
from matplotlib.transforms import BboxBase as bbase

from . import const as constants

logger = logging.getLogger(__name__)
path_matcher = re.compile(r'\$\{([^}^{]+)\}')


# ------------------------------
# General Utility Functions
# ------------------------------

def logger_setup(logger_name, log=1, verbose=1):
    """Set up the application logger."""
    verbose_level = logging.INFO
    if verbose == 2:
        verbose_level = logging.DEBUG
    elif verbose == 1:
        verbose_level = logging.INFO
    elif verbose == 0:
        verbose_level = logging.ERROR

    filename = None
    if log == 1:
        filename = str(logger_name) + ".LOG"
    logging.basicConfig(
        filename=filename,
        format="%(levelname)s :: %(module)s (%(funcName)s:%(lineno)d) : %(message)s",
        level=logging.DEBUG,
        filemode="w",
    )
    stdout_log = logging.StreamHandler(sys.stdout)
    stdout_log.setLevel(verbose_level)
    formatter = logging.Formatter("%(levelname)s :: %(module)s (%(funcName)s:%(lineno)d) : %(message)s")
    stdout_log.setFormatter(formatter)
    root = logging.getLogger()
    root.addHandler(stdout_log)


def mkdir_p(path):
    """Create a directory, handling errors gracefully."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            logger.error("Permission denied: cannot create " + path)
            sys.exit()


def get_nested_key_value(dictionary: Dict, keys: List[str]) -> Any:
    """
    Get the value of a nested key in a dictionary.

    Parameters:
    - dictionary: The input dictionary.
    - keys: A list of keys specifying the path to the nested key.

    Returns:
    - The value of the nested key, or None if the key doesn't exist.
    """
    current_dict = dictionary
    for key in keys:
        if key in current_dict:
            current_dict = current_dict[key]
        else:
            return None
    if isinstance(current_dict, str) and ',' in current_dict:
        current_dict = current_dict.split(',')
    return current_dict


def timer(start_time, end_time):
    """Simple timer."""
    return str(timedelta(seconds=(end_time - start_time)))


def get_repo_root_dir(repo_path: str) -> str:
    """Find the root directory of a Git repository."""
    path = pathlib.Path(repo_path)
    if not path.is_dir():
        path = path.parent
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        universal_newlines=True,
        cwd=str(path),
    )
    proc.check_returncode()
    return proc.stdout.strip()


def get_project_root(anchor=".git"):
    """
    Get the top-level project directory by searching for an anchor name

    Parameters:
        anchor: A filename or directory that is unique to the project root (e.g., 'README.md', '.git').

    Returns:
        Path to the project root directory.
    """
    current_dir = pathlib.Path(__file__).resolve().parent

    for parent in current_dir.parents:
        if (parent / anchor).exists():
            return parent  # 'return parent.name' to get name only
    return None


def not_none(*args, default=None, **kwargs):
    """
    Return the first non-``None`` value. This is used with keyword arg aliases and
    for setting default values. Use `kwargs` to issue warnings when multiple passed.
    """
    first = default
    if args and kwargs:
        raise ValueError("not_none can only be used with args or kwargs.")
    elif args:
        for arg in args:
            if arg is not None:
                first = arg
                break
    elif kwargs:
        for name, arg in list(kwargs.items()):
            if arg is not None:
                first = arg
                break
        kwargs = {name: arg for name, arg in kwargs.items() if arg is not None}
        if len(kwargs) > 1:
            logger.warning(
                f"Got conflicting or duplicate keyword arguments: {kwargs}. "
                "Using the first keyword argument."
            )
    return first


# ------------------------------
# YAML Utility Functions
# ------------------------------

def validate_yaml(file_content):
    """Validate a YAML file."""
    try:
        with open(file_content, 'r') as file:
            yaml.safe_load(file)
        return True
    except FileNotFoundError:
        logger.error(f'{file_content} does not exist')
        return None
    except yaml.YAMLError as exc:
        logger.error(f"yaml.YAMLError: {exc}")
        return False


def yaml_path_constructor(loader, node):
    """Extract the matched value, expand env variable, and replace the match."""
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end():]


def load_yaml_simple(file_path: str) -> Dict[str, Any]:
    """Load a YAML file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def load_yaml(yaml_filename):
    """Load a YAML file."""
    my_yaml = os.path.abspath(yaml_filename)
    if not validate_yaml(my_yaml):
        logger.error(f'Error loading yaml file: {my_yaml}')
        sys.exit("YAML validation failure!")
    try:
        yaml.add_implicit_resolver('!path', path_matcher, None, SafeLoader)
        yaml.add_constructor('!path', yaml_path_constructor, SafeLoader)

        logger.debug('loading yaml %s' % my_yaml)
        with open(my_yaml) as f:
            yaml_config = yaml.safe_load(f)
            logger.debug(f'{my_yaml} yaml loaded successfully')
        return yaml_config
    except FileNotFoundError:
        logger.error(f'{my_yaml} does not exist')
        return None


# ------------------------------
# Configuration-Specific Utilities
# ------------------------------

def get_nested_key(data: Dict, keys: List[str], default=None) -> Any:
    """
    Safely retrieve a nested key from a dictionary.

    Args:
        data (dict): The dictionary to retrieve the key from.
        keys (list): A list of keys representing the path to the nested key.
        default: The default value to return if the key is not found.

    Returns:
        The value of the nested key, or the default value if the key is not found.
    """
    try:
        for key in keys:
            data = data[key]
        return data
    except (KeyError, TypeError):
        return default


def join_file_path(base_path: str, file_name: str) -> str:
    """
    Join a base path and file name into a full file path.

    Args:
        base_path (str): The base directory path.
        file_name (str): The file name.

    Returns:
        str: The full file path.
    """
    return os.path.join(base_path, file_name) if base_path else file_name


def log_method(func):
    """
    Decorator to log the start and end of a method.

    Args:
        func (callable): The method to decorate.

    Returns:
        callable: The wrapped method.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        # Extract the file name from the module path
        module_parts = func.__module__.split('.')
        file_name = module_parts[-1] + '.py'
        logger.debug(f"Starting {func.__name__} in {file_name}")
        result = func(*args, **kwargs)
        logger.debug(f"Finished {func.__name__} in {file_name}")
        return result
    return wrapper


# ------------------------------
# Domain-Specific Utilities
# ------------------------------

def read_species_db() -> dict:
    """Read species database YAML file and load into data structure."""
    root_path = os.path.dirname(os.path.abspath('const.py'))
    db_path = constants.species_db_path
    if not os.path.exists(db_path):
        db_path = os.path.join(root_path, constants.species_db_path)
    return load_yaml(db_path)

def read_meta_coords() -> dict:
    """ Read meta coordinates YAML file and load into data structure"""
    root_path = os.path.dirname(os.path.abspath('const.py'))
    coord_file_path = constants.meta_coords_path
    if not os.path.exists(coord_file_path):
        coord_file_path = os.path.join(root_path, constants.meta_coords_path)
    return load_yaml(coord_file_path)


def read_meta_attrs() -> dict:
    """ Read meta attributes YAML file and load into data structure"""
    root_path = os.path.dirname(os.path.abspath('const.py'))
    attr_file_path = constants.meta_attrs_path
    if not os.path.exists(attr_file_path):
        attr_file_path = os.path.join(root_path, constants.meta_attrs_path)
    return load_yaml(attr_file_path)


def read_species_db() -> dict:
    """ Read species database YAML file and load into data structure"""
    root_path = os.path.dirname(os.path.abspath('const.py'))
    db_path = constants.species_db_path
    if not os.path.exists(db_path):
        db_path = os.path.join(root_path, constants.species_db_path)
    return load_yaml(db_path)


def get_reader_from_name(name):
    """ Get reader name (as defined in RootFactory) from a given source name """
    if name in ['gridded', 'geos', 'ccm', 'cf', 'wrf', 'lis']:
        return 'gridded'
    elif name in ['airnow', 'fluxnet']:
        return 'csv'
    elif name in ['omi', 'mopitt', 'landsat']:
        return 'hdf5'


def get_season_from_file(file_name):
    if "ANN" in file_name:
        return "ANN"
    elif "JJA" in file_name:
        return "JJA"
    elif "DJF" in file_name:
        return "DJF"
    elif "SON" in file_name:
        return "SON"
    elif "MAM" in file_name:
        return "MAM"
    else:
        return None
    
def squeeze_fig_aspect(fig, preserve='h'):
    # https://github.com/matplotlib/matplotlib/issues/5463
    preserve = preserve.lower()
    bb = bbase.union([ax.bbox for ax in fig.axes])

    w, h = fig.get_size_inches()
    if preserve == 'h':
        new_size = (h * bb.width / bb.height, h)
    elif preserve == 'w':
        new_size = (w, w * bb.height / bb.width)
    else:
        raise ValueError(
            'preserve must be "h" or "w", not {}'.format(preserve))
    fig.set_size_inches(new_size, forward=True)

