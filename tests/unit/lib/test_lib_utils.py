import io
import os
import tempfile

import pytest
import yaml

import eviz.lib.utils as u
import eviz.lib.constants as constants


# Write StringIO content to a temporary file
def write_to_temp_file(string_io_obj):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.yaml') as temp_file:
        # Write content from StringIO to the temp file
        temp_file.write(string_io_obj.getvalue())
        # Return the name (path) of the temporary file
        return temp_file.name


@pytest.mark.parametrize(
    ('model_name', 'expected'),
    (
        ('ccm', constants.CCM_YAML_PATH),
        ('cf', constants.CF_YAML_PATH),
        ('geos', constants.GEOS_YAML_PATH),
        ('lis', constants.LIS_YAML_PATH),
        ('wrf', constants.WRF_YAML_PATH),
        ('gridded', constants.GRIDDED_YAML_PATH),
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


def test_expand_env_vars_simple_string(monkeypatch):
    monkeypatch.setenv("MYTESTVAR", "myvalue")
    input_str = "${MYTESTVAR}/data"
    result = u.expand_env_vars(input_str)
    assert result == "myvalue/data"


def test_expand_env_vars_dict(monkeypatch):
    monkeypatch.setenv("HOME", "/home/testuser")
    input_dict = {
        "path": "${HOME}/project",
        "other": "no_var"
    }
    result = u.expand_env_vars(input_dict)
    assert result["path"] == "/home/testuser/project"
    assert result["other"] == "no_var"


def test_expand_env_vars_list(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/data")
    input_list = ["${DATA_DIR}/a", "${DATA_DIR}/b"]
    result = u.expand_env_vars(input_list)
    assert result == ["/data/a", "/data/b"]


def test_expand_env_vars_nested(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    input_obj = {
        "level1": [
            {"level2": "${FOO}/baz"},
            "plain"
        ]
    }
    result = u.expand_env_vars(input_obj)
    assert result["level1"][0]["level2"] == "bar/baz"
    assert result["level1"][1] == "plain"


def test_expand_env_vars_non_string():
    # Should return non-string types unchanged
    assert u.expand_env_vars(123) == 123
    assert u.expand_env_vars(None) is None
    assert u.expand_env_vars(3.14) == 3.14

