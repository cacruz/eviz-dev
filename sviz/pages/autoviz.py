import os
import streamlit as st
import pandas as pd
from subprocess import run
from json import load
from pathlib import Path
from jinja2 import Template
from datetime import datetime
from time import sleep
import eviz.lib.utils as u

NEW_LOGO = 'sviz/static/resized_logo_image.png'
st.logo(NEW_LOGO)

CURRENT_DIR = Path(__file__).parent
TEMPLATE_PATH = CURRENT_DIR / "../other/template.py"
DEMO_YAML = "sviz/other/options_map.yaml"


def create_new_page(dash_name, dash_config):
    """
    Create a new page in pages for the dashboard.
    Args:
        dash_name (str): Name of the dashboard.
        dash_config (dict): Dashboard parameters.

    Returns:
        True if the dashboard page is successfully created.
        False if the page is not created.

    """
    with TEMPLATE_PATH.open("r") as file:
        template_content = file.read()
    template = Template(template_content)
    populated_content = template.render(config=dash_config)
    new_page_path = CURRENT_DIR / f"./{dash_name}"

    new_script = f"{new_page_path}.py"
    with open(new_script, "w") as file:
        file.write(populated_content)

    if os.path.exists(new_script):
        return True
    return False


def make_nice_dimstr(dimensions):
    """
    Makes the dimensions string more readable
    Args:
        dimensions (): str, dimensions string

    Returns:
        str, formatted dimensions string
    """
    dimensions = dimensions.replace('[', '')
    dimensions = dimensions.replace(']', '')
    dimensions = dimensions.replace(',', ', ')
    dimensions = dimensions.replace("'", '')
    return dimensions


def metadata_to_df(filepath):
    """
    Converts metadata to a pandas DataFrame
    Args:
        filepath (): str, file path to metadata

    Returns:
        df: pandas DataFrame, metadata in a DataFrame
    """
    with open(filepath, 'r') as file:
        metadata = load(file)

    global_attributes = None
    variables = pd.DataFrame()

    # recursively flatten the dictionary
    for key, value in metadata.items():
        if key == 'global_attributes':
            global_attributes = pd.DataFrame(value, index=[0])
        else:
            for k, v in value.items():
                var_tmp = {}
                var_tmp['Variables'] = k
                var_tmp['Dimensions'] = make_nice_dimstr(
                    str(v['dimensions'])) if 'dimensions' in v else ''
                var_tmp['Long Name'] = v['attributes']['long_name'] if 'long_name' in v[
                    'attributes'] else ''
                var_tmp['Units'] = v['attributes']['units'] if 'units' in v[
                    'attributes'] else ''
                var_tmp['Missing Value'] = v['attributes'][
                    'fmissing_value'] if 'fmissing_value' in v['attributes'] else ''
                var_tmp['Min Value'] = v['attributes']['vmin'] if 'vmin' in v[
                    'attributes'] else ''
                var_tmp['Max Value'] = v['attributes']['vmax'] if 'vmax' in v[
                    'attributes'] else ''
                # var_tmp.update(v['attributes'])
                variables = variables._append(var_tmp, ignore_index=True)

    global_attributes.dropna(axis=0, how='all', inplace=True)
    variables.dropna(axis=0, how='all', inplace=True)

    return global_attributes, variables


def get_time_stamped_output(subdir) -> str:
    """ Find and return the most recent time-stamped output

    Returns:
        date_dir string
    """
    dates = os.listdir(subdir)
    dates = {datetime.strptime(date, '%Y%m%d-%H%M%S'): date for date in dates}
    most_recent_date = max(dates.keys())
    date_dir = dates[most_recent_date]

    return date_dir


def demo_options_map(yaml_file) -> tuple:
    """ A demo-specific function to read/load YAML file settings

    Returns:
        A tuple of dicts
    """
    options = u.load_yaml(yaml_file)
    single = options.get('single', {})
    compare = options.get('compare', {})
    return single, compare


def run_metadump():
    """ Run metadump.py upon a dataset box selection
    """
    file_option = st.session_state.select_dataset

    if file_option == "None Selected":
        return
    
    placeholder = st.empty()

    with placeholder.container():
        # Temporary conditional (demo only)
        if 'ANN' in file_option:
            st.write("Your selection will generate comparison plots!")

        st.write(f'Summary for {file_option}...')

        run(['python', os.path.join(u.get_project_root(), 'metadump.py'),
            os.path.join(u.get_project_root(),
                        'demo_data', file_option), '--json', '--ignore', 'Var'])

        global_attributes, variables = metadata_to_df('ds_metadata.json')

        st.dataframe(global_attributes, key="global_attributes")
        st.dataframe(variables, key="variables")


if __name__ == '__main__':
    single, compare = demo_options_map(DEMO_YAML)

    if "file_option" not in st.session_state:
        st.session_state.file_option = "None Selected"

    st.selectbox(
        "Select a dataset",
        (["None Selected"] + list(single.keys()) + list(compare.keys())),
        key='select_dataset',
        on_change=run_metadump
    )

    dashboard_name = st.text_input(
        'Enter the name of the dashboard you would like to create! (e.g. my_viz)',
    )

    if dashboard_name:
        file_option = st.session_state.select_dataset

        if not file_option == "None Selected":
            with st.empty():
                parent_folder = 'single'
                if file_option in compare:
                    parent_folder = 'comparisons'

                autoviz_config = single[file_option] if (file_option in single.keys()) else compare[
                    file_option]
                
                with st.spinner('Creating dashboard...'):
                    make_gif = False
                    if 'wrf' in autoviz_config:
                        run(['python', os.path.join(u.get_project_root(), 'autoviz.py'),
                            '-s', 'wrf', '-f',
                            os.path.join(u.get_project_root(), autoviz_config)])
                        make_gif = True
                    else:
                        run(['python', os.path.join(u.get_project_root(), 'autoviz.py'),
                            '-s', 'gridded', '-f',
                            os.path.join(u.get_project_root(), autoviz_config)])

                    date_dir = get_time_stamped_output(
                        os.path.join(u.get_project_root(), 'demo_output', parent_folder))

                    dash_config = {
                        "parent_folder": parent_folder,
                        "dash_folder": date_dir,
                        "make_gif": make_gif,
                    }

                    created = create_new_page(dashboard_name + '_' + date_dir, dash_config)
                    
                    if created:
                        sleep(1)

                        st.switch_page(f'pages/{dashboard_name}_{date_dir}.py')
