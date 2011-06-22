==================================
Python Quantum Information Toolkit
==================================

Version 0.9.10 (beta)
Released 2011-06-??


Introduction
============

Python Quantum Information Toolkit (PyQIT) is a free, open source
Python 2.6 package for various quantum information and computing
-related purposes, distributed under GPL.
It is a sister project of the MATLAB Quantum Information Toolkit
and has equivalent functionality. PyQIT requires the following
Python libraries:

* `NumPy <http://scipy.org/>`_  1.3.0+  (TODO 1.6.0+)
* `SciPy <http://scipy.org/>`_  0.7.2+
* `Matplotlib <http://matplotlib.sourceforge.net/>`_  0.99.3+ (TODO 1.0.0+)

For interactive use the `IPython <http://ipython.scipy.org/>`_ interactive shell is recommended.

The latest version can be downloaded from the project website,

  http://sourceforge.net/projects/qit/

The toolkit is installed by simply unzipping it, or downloading it
directly from the SVN server. For an interactive session, start
IPython with ::

  ipython -pylab

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
citation syntax. Each reference is on its own line and starts with the
characters ".. [". One can compile a list of all the references in the
toolkit using the shell command ::

  grep '\.\. \[' *.py



Contributing
============

QIT is an open source project and your contributions are welcome.
To keep the code readable and maintainable, we ask you to follow these
coding guidelines:

* Fully document all the modules, classes and functions using docstrings
  (purpose, calling syntax, output, approximations used, assumptions made...)
* Instead of using multiple similar functions, use a single function
  performing multiple related tasks, see e.g. state.state.__init__
* Add relevant literature references using the %! syntax.
* Raise an exception on invalid input.
* Use variables sparingly, give them descriptive (but short) names.
* Use brief comments to explain the logic of your code.
* When you add new functions also add testing scripts for validating
  your code. If you modify existing code, make sure you didn't break
  anything by checking that the testing scripts still run flawlessly.



Authors
=======

* Ville Bergholm          2008-2011
* Jacob D. Biamonte       2008-2009
* James D. Whitfield      2009-2010
