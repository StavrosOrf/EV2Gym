# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('..'))
# sys.path.append(os.path.abspath('../ev2gym'))

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../ev2gym/"))

project = 'EV2Gym'
copyright = '2024, Stavros Orfanoudakis'
author = 'Stavros Orfanoudakis'
release = '4/2024'

source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    # "sphinx.ext.doctest",
    # "nbsphinx_link",
    'autoapi.extension'
]
autodoc_typehints = 'description'

# templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',"**.ipynb_checkpoints",
#                     "*.csv", "*.json", "*.txt", "*.yml", "*.yaml"]

autoapi_options = {
    'members': True,
    'show-inheritance': True,
}

autoapi_dirs = ['../ev2gym/rl_agent',                
                '../ev2gym/utilities',
                '../ev2gym/visuals',
                '../ev2gym/baselines',
                '../ev2gym/models',
                '../ev2gym/example_config_files',
                ]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

def skip_submodules(app, what, name, obj, skip, options):
    if what == "module":
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_submodules)
