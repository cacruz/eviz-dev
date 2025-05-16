import io
import os
import tempfile

import pytest
import yaml

import eviz.lib.utils as u
import eviz.lib.const as constants


# Write StringIO content to a temporary file
def write_to_temp_file(string_io_obj):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.yaml') as temp_file:
        # Write content from StringIO to the temp file
        temp_file.write(string_io_obj.getvalue())
        # Return the name (path) of the temporary file
        return temp_file.name



# def test_load_yaml_does_not_exist(get_config):
#     with pytest.raises(SystemExit) as excinfo:
#         u.load_yaml('does_not_exist')
#     assert str(excinfo.value) == "YAML validation failure!"


# def test_load_yaml_syntax_error(get_config):
#     yaml_file = os.path.join(constants.ROOT_FILEPATH, 'test/data/test_bad.yaml')
#     with pytest.raises(SystemExit) as excinfo:
#         u.load_yaml(yaml_file)
#     assert str(excinfo.value) == "YAML validation failure!"


@pytest.mark.parametrize(
    ('model_name', 'expected'),
    (
            ('ccm', constants.ccm_yaml_path),
            ('cf', constants.cf_yaml_path),
            ('geos', constants.geos_yaml_path),
            ('lis', constants.lis_yaml_path),
            ('wrf', constants.wrf_yaml_path),
            ('gridded', constants.gridded_yaml_path),
    )
)
def test_get_nested_key_value_1nest(model_name, expected):
    val = u.get_nested_key_value({'key1': 1}, ['key1'])
    assert val == 1


def test_get_nested_key_value2nest():
    val = u.get_nested_key_value({'key1': {'key2': 2}}, ['key1', 'key2'])
    assert val == 2


def test_get_nested_key_value_3nest():
    val = u.get_nested_key_value({'key1': {'key2': {'key3': 3}}}, ['key1', 'key2', 'key3'])
    assert val == 3


def test_get_nested_key_value_2keys():
    val = u.get_nested_key_value({'key0': 0, 'key1': {'key2': 1}}, ['key0'])
    assert val == 0


def test_get_project_root():
    # TODO: this (ugly) test is dependent on the project structure. Remove or redo.
    assert 'eviz' in u.get_project_root().name


def test_yaml_path_constructor():
    yaml.add_implicit_resolver('!path', u.path_matcher, None, yaml.SafeLoader)
    yaml.add_constructor('!path', u.yaml_path_constructor, yaml.SafeLoader)
    data = """
env: ${HOME}/file.txt
other: file.txt
      """
    p = yaml.safe_load(data)
    assert p['env'] == os.path.join(os.environ.get('HOME'), 'file.txt')


def test_valid_yaml():
    valid_yaml = """
    key: value
    another_key:
      - item1
      - item2
    """
    file_content = io.StringIO(valid_yaml)
    temp_file_path = write_to_temp_file(file_content)

    assert u.load_yaml(temp_file_path) is not None


# def test_invalid_yaml_missing_value():
#     invalid_yaml = """
#     key1:
#     key2: value
#     """
#     file_content = io.StringIO(invalid_yaml)
#     assert u.validate_yaml(file_content) is False
#

# def test_invalid_yaml_duplicate_keys():
#     invalid_yaml = """
#     key: value1
#     key: value2  # Duplicate key
#     """
#     file_content = io.StringIO(invalid_yaml)
#     temp_file_path = write_to_temp_file(file_content)
#     # file_content = io.StringIO(invalid_yaml)
#     assert u.load_yaml(temp_file_path) is None


def test_invalid_yaml_non_scalar_key():
    invalid_yaml = """
    [key1, key2]: value  # Invalid key, it should be a scalar
    """
    file_content = io.StringIO(invalid_yaml)
    temp_file_path = write_to_temp_file(file_content)

    with pytest.raises(SystemExit) as excinfo:
        u.load_yaml(temp_file_path)
    assert str(excinfo.value) == "YAML validation failure!"


def test_invalid_yaml_multiline_string():
    invalid_yaml = """
    key: |
      This is a multiline
      string that is valid
     but this line is not properly indented
    """
    file_content = io.StringIO(invalid_yaml)
    temp_file_path = write_to_temp_file(file_content)

    with pytest.raises(SystemExit) as excinfo:
        u.load_yaml(temp_file_path)
    assert str(excinfo.value) == "YAML validation failure!"


def test_invalid_yaml_tabs_instead_of_spaces():
    invalid_yaml = """
    key: value
    another_key:
    \t- item1  # Tabs are not allowed for indentation in YAML
    """
    file_content = io.StringIO(invalid_yaml)
    temp_file_path = write_to_temp_file(file_content)

    with pytest.raises(SystemExit) as excinfo:
        u.load_yaml(temp_file_path)
    assert str(excinfo.value) == "YAML validation failure!"


def test_invalid_yaml_unclosed_quotes():
    invalid_yaml = """
    key: "This is a valid string
    another_key: value
    """
    file_content = io.StringIO(invalid_yaml)
    temp_file_path = write_to_temp_file(file_content)

    with pytest.raises(SystemExit) as excinfo:
        u.load_yaml(temp_file_path)
    assert str(excinfo.value) == "YAML validation failure!"


def test_invalid_yaml_invalid_key():
    invalid_yaml = """
    key: value
    : another_key  # Invalid key due to colon
    """
    file_content = io.StringIO(invalid_yaml)
    temp_file_path = write_to_temp_file(file_content)

    with pytest.raises(SystemExit) as excinfo:
        u.load_yaml(temp_file_path)
    assert str(excinfo.value) == "YAML validation failure!"


def test_invalid_yaml_missing_indentation():
    invalid_yaml = """
key: value
    another_key:
- item1
- item2
    """
    file_content = io.StringIO(invalid_yaml)
    temp_file_path = write_to_temp_file(file_content)

    with pytest.raises(SystemExit) as excinfo:
        u.load_yaml(temp_file_path)
    assert str(excinfo.value) == "YAML validation failure!"


def test_invalid_yaml_missing_indentation2():
    invalid_yaml = """
inputs:
   - name         : filename
 location     : path
     exp_name     : foo
     to_plot:
       T: xy,yz
    """
    file_content = io.StringIO(invalid_yaml)
    temp_file_path = write_to_temp_file(file_content)

    with pytest.raises(SystemExit) as excinfo:
        u.load_yaml(temp_file_path)
    assert str(excinfo.value) == "YAML validation failure!"


def test_invalid_yaml_bad_characters():
    invalid_yaml = """
    key: value
    another_key: !!binary xyz
    """
    file_content = io.StringIO(invalid_yaml)
    temp_file_path = write_to_temp_file(file_content)

    with pytest.raises(SystemExit) as excinfo:
        u.load_yaml(temp_file_path)
    assert str(excinfo.value) == "YAML validation failure!"


def test_invalid_yaml_missing_indentation3():
    invalid_yaml = """
    key: value
    another_key:
      - item1
    - item2
    """
    file_content = io.StringIO(invalid_yaml)
    temp_file_path = write_to_temp_file(file_content)

    with pytest.raises(SystemExit) as excinfo:
        u.load_yaml(temp_file_path)
    assert str(excinfo.value) == "YAML validation failure!"

