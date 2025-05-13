import os
import json
from pathlib import Path
from datetime import datetime
import streamlit as st
NEW_LOGO = 'sviz/static/resized_logo_image.png'
st.logo(NEW_LOGO)

CURRENT_DIR = Path(__file__).parent
CONFIG = {{ config }}

import eviz.lib.utils as u


def process_json(dir_path, json_file):
    """
    Process image jsons.
    Args:
        dir_path (str): image directory path.
        json_file (str): json file path

    Returns:
        outputs (list): list of image info.
    """
    with open(os.path.join(dir_path, json_file), 'r') as f:
        data = json.load(f)
    output_keys = [str(data["output_findex"])]
    outputs = []
    for output in output_keys:
        file_name = os.path.join(dir_path, data[output]["filename"])
        title = data[output]["title"]
        plot_type = data[output]["plot_type"]
        field_name = data[output]["field_name"]
        model_type = data[output]["model"]
        level = data[output]["level"]
        time_now = datetime.utcfromtimestamp(data["time_now"])
        outputs.append({"filename": file_name, "title": title,
                        "plot_type": plot_type, "field_name": field_name,
                        "level": level, "model": model_type,
                        "time_now": time_now})
    return outputs


def display_vis(date_outs, desc_outs):
    """
    Display images from output files.
    Args:
        outs (list): images from output files.

    Returns:
        None
    """

    if CONFIG['make_gif']:
        import base64
        for date in date_outs.keys():
            with st.expander(date, expanded=True):

                outs = date_outs[date]
                for i, image_url in enumerate(outs):
                    file_ = open(image_url['filename'], "rb")
                    contents = file_.read()
                    data_url = base64.b64encode(contents).decode("utf-8")
                    st.markdown(
                        f'<img src="data:image/gif;base64,{data_url}">',
                        unsafe_allow_html=True,
                    )
    else:
        for date in date_outs.keys():
            with st.expander(date, expanded=True):

                st.markdown(desc_outs[date])
                outs = date_outs[date]

                columns = st.columns(3)
                for i, image_url in enumerate(outs):
                    title = outs[i]['title']
                    time_now = outs[i]['time_now']
                    col_index = i % 3
                    columns[col_index].image(
                        image_url["filename"], use_column_width=True, caption=str(time_now))
            

def sift(date_outs):
    """
    Filter output display based on user selection.
    Args:
        outs (list): sorted images

    Returns:
        outs (list): filtered images
    """
    st.sidebar.title("Filter")

    outs = []
    for date in date_outs.keys():
        outs = outs + date_outs[date]
    
    # filter by model
    model_types = []
    for image in outs:
        model_types.append(image["model"])
    model_types = list(set(model_types))
    model_types.sort()
    model_types.insert(0, "All")
    model_type = st.sidebar.selectbox("Model Type", model_types)
    
    if model_type != "All":
        outs = [image for image in outs if image["model"] == model_type]
    
    # filter by field name
    field_types = []
    for image in outs:
        field_types.append(image["field_name"])
    field_types = list(set(field_types))
    field_types.sort()
    field_types.insert(0, "All")
    field_type = st.sidebar.selectbox("Field Name", field_types)
    if field_type != "All":
        outs = [image for image in outs if image["field_name"] == field_type]
    
    # filter by plot type
    plot_types = []
    for image in outs:
        plot_types.append(image["plot_type"])
    plot_types = list(set(plot_types))
    plot_types.sort()
    plot_types.insert(0, "All")
    plot_type = st.sidebar.selectbox("Plot Type", plot_types)
    if plot_type != "All":
        outs = [image for image in outs if image["plot_type"] == plot_type]
    
    # filter by level
    levels = []
    for image in outs:
        levels.append(image["level"])
    levels = list(set(levels))
    levels.insert(0, "All")
    level = st.sidebar.selectbox("Level", levels)
    if level != "All":
        outs = [image for image in outs if image["level"] == level]
    
    # filter by time with slider
    times = []
    for image in outs:
        times.append(image["time_now"])
    times = list(set(times))
    times.sort()
    times.insert(0, "All")
    tms = st.sidebar.multiselect("Time", times, default="All")
    if not "All" in tms:
        incl = []
        for tm in tms:
            for image in outs:
                if image["time_now"] == tm:
                    incl.append(image)
        outs = incl
    
    # filter by title
    titles = []
    for image in outs:
        titles.append(image["title"])
    titles = list(set(titles))
    titles.insert(0, "All")
    title = st.sidebar.selectbox("Title", titles)
    if title != "All":
        outs = [image for image in outs if image["title"] == title]
    
    date_outs = {}
    for out in outs:
        if out["date_time"] in date_outs:
            date_outs[out["date_time"]].append(out)
        else:
            date_outs[out["date_time"]] = [out]

    return date_outs


if __name__ == "__main__":
    output_path = os.path.join(u.get_project_root(), 'demo_output',
                               CONFIG['parent_folder'],
                               CONFIG['dash_folder'])

    date_outs = {}
    desc_outs = {}
    datetime_object = datetime.strptime(CONFIG['dash_folder'], "%Y%m%d-%H%M%S")
    
    json_files = [f for f in os.listdir(output_path) if f.endswith('.json')]
    
    desc = ""
    if os.path.exists(os.path.join(output_path, 'desc.md')):
        with open(os.path.join(output_path, 'desc.md'), 'r') as f:
            # st.markdown(f.read())
            desc = f.read()
    
    desc_outs[CONFIG['dash_folder']] = desc

    outputs = []
    for file in json_files:
        outputs = outputs + \
            process_json(output_path, file)
        
    for out in outputs:
        out["date_time"] = CONFIG['dash_folder']
    
    date_outs[CONFIG['dash_folder']] = outputs

    # outs = sorted(outputs, key=lambda k: k['time_now'], reverse=True)
    date_outs = sift(date_outs)
    display_vis(date_outs, desc_outs)

