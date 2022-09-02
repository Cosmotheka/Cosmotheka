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
from os.path import abspath, dirname, join as pjoin
import sys

this_dir = dirname(abspath(__file__))
root_path = abspath(pjoin(this_dir, '../../'))
if os.path.isdir(root_path):
    sys.path.insert(0, root_path)

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    try:
        from unittest.mock import MagicMock
    except ImportError:
        from mock import Mock as MagicMock

    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
            return MagicMock()

    MOCK_MODULES = ["xcell.mappers",
                    "healpy",
                    "astropy",
                    "yaml"]

    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
# -- Project information -----------------------------------------------------

project = 'xCell'
copyright = '2022, Jaime'
author = 'Jaime'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.ifconfig',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
