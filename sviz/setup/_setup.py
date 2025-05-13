import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
eviz_eviz_streamlit_dir = os.path.join(parent_dir, 'autoviz')
pages_dir = os.path.join(parent_dir, 'pages')

if eviz_eviz_streamlit_dir not in sys.path:
    sys.path.append(eviz_eviz_streamlit_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

if pages_dir not in sys.path:
    sys.path.append(pages_dir)


