===========================
Quantum Information Toolkit
===========================


Introduction
------------

Quantum Information Toolkit (QIT) is a free, open source Python 3 package for various quantum
information and computing -related purposes, released under GNU GPL v3.  It is a descendant of the
MATLAB Quantum Information Toolkit, and has considerably more functionality.

The latest version can be found on `our website <http://qit.sourceforge.net/>`_.

The toolkit is installed from the Python Package Index by

.. code-block:: bash

   $ pip install qit

or by cloning the Git repository, and installing directly from there:

.. code-block:: bash

   $ git clone https://git.code.sf.net/p/qit/code-python qit
   $ cd qit
   $ pip install .

For interactive use, we recommend the `IPython shell <https://ipython.org/>`_.

To get an overview of the features and capabilities of the toolkit, run

.. code-block:: bash

   $ python qit/examples.py


License
-------

QIT is released under the GNU General Public License version 3.
This basically means that you can freely use, share and modify it as
you wish, as long as you give proper credit to the authors and do not
change the terms of the license. See LICENSE.txt for the details.


Design notes
------------

The main design goals for this toolkit are ease of use and comprehensiveness. It is primarily meant
to be used as a tool for experimentation, hypothesis testing, small simulations, and learning, not
for computationally demanding simulations. Hence the efficiency of the algorithms used is not a
number one priority.
However, if you think an algorithm could be improved without compromising accuracy or
maintainability, please let the authors know or become a contributor yourself!


Contributing
------------

QIT is an open source project and your contributions are welcome.
To keep the code readable and maintainable, we ask you to follow these
coding guidelines:

* Fully document all the modules, classes and functions using docstrings
  (purpose, calling syntax, output, approximations used, assumptions made...).
  The docstrings may contain reStructuredText markup for math, citations etc.
  Use the Google docstring style.
* Add relevant literature references to ``docs/refs.bib`` and cite them in the function
  or module docstring using ``sphinxcontrib-bibtex`` syntax.
* Instead of using multiple similar functions, use a single function
  performing multiple related tasks, see e.g. ``qit.state.State.measure``.
* Raise an exception on invalid input.
* Use variables sparingly, give them descriptive (but short) names.
* Use brief comments to explain the logic of your code.
* When you add new functions also add tests for validating
  your code. If you modify existing code, make sure you didn't break
  anything by checking that the tests still run successfully.
