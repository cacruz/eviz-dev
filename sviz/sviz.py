import streamlit as st
import sys
import os
from pathlib import Path

# Add the top-level project directory to sys.path
TOP_LEVEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOP_LEVEL_DIR))

CURRENT_DIR = Path(os.getcwd()).resolve()
if CURRENT_DIR != TOP_LEVEL_DIR:
    st.error("")
    st.markdown(
        """
        <div style="color: red; font-size: 20px; font-weight: bold;">
            Please run sviz from the top-level directory: <br>
            <code>streamlit run sviz/sviz.py</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: blue;'>sViz</h1>",
            unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>A Web-based Approach for Earth System Model Data Visualization</h3>",
            unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: grey;'>(Navigate to different applications using the sidebar)</h6>",
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.image("https://portal.nccs.nasa.gov/datashare/astg/eviz/sample_data/gif/PSFC.gif",
             use_column_width=True)
with col2:
    st.image("https://portal.nccs.nasa.gov/datashare/astg/eviz/sample_data/gif/RI_SEPA.gif",
             use_column_width=True)
col3, col4 = st.columns(2)
with col3:
    st.image("https://portal.nccs.nasa.gov/datashare/astg/eviz/sample_data/gif/T2.gif",
             use_column_width=True)
with col4:
    st.image("https://portal.nccs.nasa.gov/datashare/astg/eviz/sample_data/gif/Q2.gif",
             use_column_width=True)
st.markdown("<h4 style='text-align: center; color: black;'>Visualization of a Hurricane Katrina Simulation</h4>",
            unsafe_allow_html=True)

st.logo("sviz/static/resized_logo_image.png")
