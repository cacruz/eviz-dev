# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
# conf.py
import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock modules that might cause import errors
MOCK_MODULES = ['xarray', 'numpy', 'pandas']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# Add autodoc_type_aliases for modern type annotations
autodoc_type_aliases = {
    'xr.Dataset | None': 'Optional[xarray.Dataset]',
}

# Add autodoc_typehints_format to handle modern type annotations
autodoc_typehints_format = 'fully-qualified'


sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
about = {}
with open('../eviz/__about__.py', "r") as fp:
    exec(fp.read(), about)

project = 'EViz'
author = 'EViz Developers'

# The full version, including alpha/beta/rc tags
release = about["__version__"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosectionlabel',  # link from text to a heading using :ref:
    'sphinx.ext.autodoc',  # autodocument
    'sphinx.ext.napoleon',  # google and numpy doc string support
    'sphinx.ext.mathjax',  # latex rendering of equations using MathJax
    'myst_parser',
    'sphinxcontrib.mermaid',
]
# 'sphinx.ext.viewcode',  # add links to view code

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# -- Napoleon autodoc options -------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_ivar = True
napoleon_include_init_with_doc = True

# -- Other settings -----------------------------------------------------------

# Path to logo image file
html_logo = 'static/ASTG_logo_simple.png'

html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#175762'
}

# Allows to build the docs with a minimal environment without warnings about missing packages
autodoc_mock_imports = [
    'matplotlib',
    'mpl_toolkits',
    'holoviews',
    'cartopy',
    'xesmf',
    'numpy',
    'pyhdf',
    'dask',
    'panel',
    'param',
    'bokeh',
    'geoviews',
    'hvplot',
    'h5py',
    'netcdf4',
    'pandas',
    'scipy',
    'tqdm',
    'yaml',
    'cftime',
    'xarray',
    'pytest',
    'pydap',
    'streamlit',
    'sklearn',
    'PIL',
]

suppress_warnings = ['autosectionlabel.*']
# Ignore duplicate object descriptions
suppress_warnings.append('app.add_directive')
