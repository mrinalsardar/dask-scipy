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

import packaging.version

from dask_scipy import __version__ as version

sys.path.insert(0, os.path.abspath("../../"))
# sys.path.append(os.path.abspath(".."))
# sys.path.append(os.path.abspath("../../dask_scipy"))


# -- Project information -----------------------------------------------------

project = "Dask-SciPy"
copyright = "2022, Mrinal Kanti Sardar"
author = "Mrinal Kanti Sardar"

# The full version, including alpha/beta/rc tags
release = packaging.version.parse(version).base_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "numpydoc",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": (
        "https://numpy.org/doc/stable/",
        "https://numpy.org/doc/stable/objects.inv",
    ),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "distributed": ("https://distributed.dask.org/en/latest/", None),
}

numpydoc_class_members_toctree = False
autodoc_default_flags = ["members", "inherited-members"]
autosummary_generate = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "dask-ml",
        "dask-ml Documentation",
        author,
        "dask-ml",
        "One line description of project.",
        "Miscellaneous",
    )
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "dask_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {"logo_only": True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
