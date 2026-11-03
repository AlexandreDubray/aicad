"""Sphinx configuration of the pyaicad documentation.

Every code cell of the guide is executed when the documentation is built, so a guide that does
not match the library fails the build. The chapters about training need the full build of
pyaicad (the ``learning`` feature) and are skipped when it is not installed.
"""

import pyaicad

project = "pyaicad"
author = "Alexandre Dubray"
release = pyaicad.__version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

html_theme = "furo"
html_title = f"pyaicad {release}"

myst_enable_extensions = ["colon_fence", "deflist"]

# Execute every code cell at each build; a failing cell fails the build.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300
nb_merge_streams = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
