# -*- coding: utf-8 -*-
"""Python Quantum Information Toolkit

See the README.txt file included in this distribution
or the project website at http://qit.sourceforge.net/
"""

import scipy.constants as const

from .base import *
from .lmap import *
from .utils import *
from .state import *
from .plot import *
from . import gate, hamiltonian, ho, invariant, markov, seq, examples


# toolkit version number
__version__ = '0.12.0-unreleased'

def version():
    """Returns the QIT version number (as a string)."""
    return __version__



#print('Python Quantum Information Toolkit, version {0}.'.format(__version__))
