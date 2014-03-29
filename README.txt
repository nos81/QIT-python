===========================
Quantum Information Toolkit
===========================



Introduction
============

Quantum Information Toolkit (QIT) is a free, open source
Python 2.7 package for various quantum information and computing
-related purposes, distributed under GPL.
It is a sister project of the MATLAB Quantum Information Toolkit
and has equivalent functionality. QIT requires the following
Python libraries:

* `NumPy <http://numpy.org/>`_  1.7.1
* `SciPy <http://scipy.org/>`_  0.11.0
* `matplotlib <http://matplotlib.org/>`_  1.2

For interactive use the `IPython <http://ipython.org/>`_ interactive shell is recommended.

The latest version can be downloaded from our website,

  http://qit.sourceforge.net/

The toolkit is installed by simply unzipping it, or downloading it
directly from the Git repository. For an interactive session, start
IPython with ::

  ipython --pylab

and then import the toolkit using ::

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

The main design goals for this toolkit are ease of use and
comprehensiveness. It is primarily meant to be used as a tool for
hypothesis testing, small simulations, and learning, not for
computationally demanding simulations. Hence optimal efficiency of the
algorithms used is not a number one priority.
However, if you think an algorithm could be improved without
compromising accuracy or maintainability, please let the authors know
or become a contributor yourself!



Bibliography
============

Some of the source files have literature references relevant to the
algorithms or concepts used. These references use the reStructuredText
citation syntax: each reference is on its own line and starts with the
characters ".. [". One can compile a list of all the references in the
toolkit using the shell command ::

  grep '\.\. \[' *.py



Contributing
============

QIT is an open source project and your contributions are welcome.
To keep the code readable and maintainable, we ask you to follow these
coding guidelines:

* Fully document all the modules, classes and functions using docstrings
  (purpose, calling syntax, output, approximations used, assumptions made...).
  The docstrings may contain reStructuredText markup for math,
  citations etc.
* Add relevant literature references using the reST syntax.
* Instead of using multiple similar functions, use a single function
  performing multiple related tasks, see e.g. :func:`qit.state.state.measure`.
* Raise an exception on invalid input.
* Use variables sparingly, give them descriptive (but short) names.
* Use brief comments to explain the logic of your code.
* When you add new functions also add testing scripts for validating
  your code. If you modify existing code, make sure you didn't break
  anything by checking that the testing scripts still run successfully.



Authors
=======

* Ville Bergholm          2008-2014
* Jacob D. Biamonte       2008-2009
* James D. Whitfield      2009-2010
