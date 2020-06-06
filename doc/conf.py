# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import re
import sys

sys.path.insert(0, os.path.abspath('..'))
import qit


# -- Project information -----------------------------------------------------

project = 'Quantum Information Toolkit'
copyright = '2011-2020, Ville Bergholm et al.'
author = 'Ville Bergholm et al.'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = qit.__version__

# The short X.Y version.
version = re.match(r'^(\d+\.\d+)', release).expand(r'\1')


# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '3.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinxcontrib.bibtex',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%Y-%m-%d'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'nature'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'qit_manual.tex', u'Quantum Information Toolkit Documentation',
   u'Ville Bergholm et al.', 'manual'),
]

latex_elements = {'papersize': 'a4paper',
                  'preamble' : """\\include{macros}\n
                                  \newcommand{\ket}[1]{\ensuremath{\left| #1 \right \rangle}}"""}

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True

#=========================================================================

# the order in which autodoc lists the documented members
autodoc_member_order = 'bysource'

# documentation source for classes
autoclass_content = 'both'

# latex macros
mathjax_config = {
    'TeX': {
        'Macros': {
            'ket': [r'\left| #1 \right\rangle', 1],
            'bra': [r'\left\langle #1 \right|', 1],
            're': r'\mathrm{Re}',
            'im': r'\mathrm{Im}',
            'trace': r'\mathrm{Tr}',
            'tr': r'\mathrm{Tr}',
            'diag': r'\mathrm{diag}',
            'braket': [r'\langle #1 \rangle', 1],
            'expect': [r'\langle #1 \rangle', 1],
            'hc': r'\text{h.c.}',  # hermitian conjugate
            'cc': r'\text{c.c.}',  # complex conjugate
            'I': r'\mathrm{I}',   # identity operator
        }
    }
}
