import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tstrait"
copyright = "2023, Tskit developers"  # noqa A001
author = "Daiki Tagami et al."

import tstrait  # noqa E402

version = str(tstrait.__version__)

release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]

# The encoding of source files.
source_encoding = "utf-8"

exclude_patterns = []

# Coverage
coverage_ignore_functions = []
coverage_ignore_classes = []
coverage_ignore_pyobjects = []


# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pyslim": ("https://tskit.dev/pyslim/docs/latest/", None),
    "tskit": ("https://tskit.dev/tskit/docs/stable/", None),
    "tskit-tutorials": ("https://tskit.dev/tutorials/", None),
    "msprime": ("https://tskit.dev/msprime/docs/stable/", None),
    "stdpopsim": ("https://popsim-consortium.github.io/stdpopsim-docs/stable/", None),
}

# Autosummary
autosummary_generate = True

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/tskit-dev/tstrait",
    "external_links": [{"name": "Community", "url": "https://tskit.dev/community/"}],
    "header_links_before_dropdown": 6,
    "logo": {
        "text": "tstrait",
    },
}

# Show html code
html_show_sourcelink = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]
