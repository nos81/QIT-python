===========================
Quantum Information Toolkit
===========================


Introduction
============

Quantum Information Toolkit (QIT) is a free, open source Python 3 package for various quantum
information and computing -related purposes, released under GNU GPL.  It is a descendant of the
MATLAB Quantum Information Toolkit, and has considerably more functionality. QIT requires the
following Python libraries:

* `NumPy <http://numpy.org/>`_  1.13.3
* `SciPy <http://scipy.org/>`_  1.0.0
* `matplotlib <http://matplotlib.org/>`_  2.0

For interactive use the `IPython <http://ipython.org/>`_ interactive shell is recommended.

The latest version can be found on `our website <http://qit.sourceforge.net/>`_.

The toolkit is installed by downloading it from the Python Package Index,
or directly from the Git repository. For an interactive session, start
IPython with

.. code-block:: bash

   ipython --pylab

and then import the toolkit using

.. code-block:: python

   from qit import *

To get an overview of the features and capabilities of the toolkit,
run examples.tour()


License
=======

QIT is released under the GNU General Public License version 3.
This basically means that you can freely use, share and modify it as
you wish, as long as you give proper credit to the authors and do not
change the terms of the license. See LICENSE.txt for the details.


Design notes
============

The main design goals for this toolkit are ease of use and comprehensiveness. It is primarily meant
to be used as a tool for experimentation, hypothesis testing, small simulations, and learning, not
for computationally demanding simulations. Hence optimal efficiency of the algorithms used is not a
number one priority.
However, if you think an algorithm could be improved without compromising accuracy or
maintainability, please let the authors know or become a contributor yourself!


Contributing
============

QIT is an open source project and your contributions are welcome.
To keep the code readable and maintainable, we ask you to follow these
coding guidelines:

* Fully document all the modules, classes and functions using docstrings
  (purpose, calling syntax, output, approximations used, assumptions made...).
  The docstrings may contain reStructuredText markup for math, citations etc.
  Use the Google docstring style.
* Add relevant literature references to doc/refs.bib and cite them in the function
  or module docstring using sphinxcontrib-bibtex syntax.
* Instead of using multiple similar functions, use a single function
  performing multiple related tasks, see e.g. :func:`qit.state.state.measure`.
* Raise an exception on invalid input.
* Use variables sparingly, give them descriptive (but short) names.
* Use brief comments to explain the logic of your code.
* When you add new functions also add tests for validating
  your code. If you modify existing code, make sure you didn't break
  anything by checking that the tests still run successfully.


Authors
=======

* Ville Bergholm          2008-2021
* Jacob D. Biamonte       2008-2009
* James D. Whitfield      2009-2010
