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
import sys
import pathlib

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())


# -- Project information -----------------------------------------------------

project = 'MIRA'
copyright = '2022, Allen W. Lynch'
author = 'Allen W. Lynch'

# The full version, including alpha/beta/rc tags
release = '1.0.0'
version = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx_copybutton',
    'sphinx_panels',
    'sphinx.ext.viewcode',
]



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
pygments_style = "sphinx"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "auto_*/**.ipynb",
    "auto_*/**.md5",
    "auto_*/**.py",
    "release/changelog/*",
    "**.ipynb_checkpoints",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "_static/mira_logo.png"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Cistrome/MIRA",
            "icon": "fab fa-github-square",
        }
    ],
    "use_edit_page_button": False,
}

html_sidebars = {
    'index' : []
}

html_css_files = ['style.css']

autoclass_content = "both"

autosummary_generate_overwrite = True

autodoc_default_options = {
    'member-order': 'bysource',
}

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "AllenWLynch", # Username
    "github_repo": "MIRA", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/source/", # Path in the checkout to the docs root
    "pygment_light_style": "tango"
}

html_additional_pages = {
    'index': 'landingpage.html',
}