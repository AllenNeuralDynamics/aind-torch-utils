"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date
import os
import sys
from os.path import abspath, dirname
from pathlib import Path

# -- Path Setup --------------------------------------------------------------
# Add the project 'src' directory so autodoc can import without installation
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "../../"))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from aind_torch_utils import __version__ as package_version

INSTITUTE_NAME = "Allen Institute for Neural Dynamics"
current_year = date.today().year

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "aind-torch-utils"
copyright = f"{current_year}, {INSTITUTE_NAME}"
author = INSTITUTE_NAME
release = package_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "light_logo": "light-logo.svg",
    "dark_logo": "dark-logo.svg",
}

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False


# -- Auto-generate API docs on build ----------------------------------------
def run_apidoc(_):
    try:
        from sphinx.ext import apidoc

        out_dir = os.path.join(HERE, "api")
        pkg_dir = os.path.join(SRC_DIR, "aind_torch_utils")
        apidoc.main(["-f", "-e", "-o", out_dir, pkg_dir])
    except Exception:
        # Allow docs to build even if apidoc is unavailable
        pass


def setup(app):  # noqa: D401 - Sphinx hook
    """Connect apidoc generation to the build."""
    app.connect("builder-inited", run_apidoc)

