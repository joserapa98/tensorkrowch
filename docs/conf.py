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
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'TensorKrowch'
copyright = '2023, José Ramón Pareja Monturiol'
author = 'José Ramón Pareja Monturiol'

# The full version, including alpha/beta/rc tags
with open('../tensorkrowch/__init__.py') as f:
    for line in f.read().splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            __version__ = line.split(delim)[1]
            break

release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = 'contents'

autodoc_typehints = 'none'
autodoc_member_order = 'bysource'
autodoc_typehints_format = 'short'

doctest_global_setup = """
import tensorkrowch as tk
import torch
import torch.nn as nn
"""

copybutton_prompt_text = '>>> |$ '

nbsphinx_execute = 'never'

# Make that the index page does not disappear from sidebar TOC. From
# https://stackoverflow.com/questions/18969093/how-to-include-the-toctree-in-the-sidebar-of-each-page
# html_sidebars = {"**": ['globaltoc.html', 'relations.html',
#                         'sourcelink.html', 'searchbox.html']}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'logo_only': True,
    'repository_url': 'https://github.com/joserapa98/tensorkrowch',
    'use_repository_button': True
}

# html_theme = 'furo'
# html_theme_options = {
#     'sidebar_hide_name': True,
#     'light_css_variables': {
#         # "color-brand-primary": "hsl(45, 80%, 45%)",
#         'color-brand-primary': 'hsl(210, 50%, 50%)',
#         'color-brand-content': 'hsl(210, 50%, 50%)',
#     },
#     'dark_css_variables': {
#         'color-brand-primary': 'hsl(210, 50%, 60%)',
#         'color-brand-content': 'hsl(210, 50%, 60%)',
#     },
#     'light_logo': 'figures/tensorkrowch_logo_light.png',
#     'dark_logo': 'figures/tensorkrowch_logo_dark.png'
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = 'figures/svg/tensorkrowch_logo_light.svg'
html_favicon = 'figures/svg/tensorkrowch_favicon_light.svg'
