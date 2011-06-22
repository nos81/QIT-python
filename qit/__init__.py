# -*- coding: utf-8 -*-
"""Python Quantum Information Toolkit

See the README.txt file included in this distribution
or the project website at http://sf.net/projects/qit/

Authors:
Ville Bergholm 2011
"""

from __future__ import print_function, division

import scipy as sp
import scipy.constants as const
import matplotlib.pyplot as plt

from base import *
from lmap import *
from utils import *
from state import *
from plot import *
import gate
import ho
import invariant
import markov
import seq
import examples


# toolkit version number
version = '0.9.10'

print('Python Quantum Information Toolkit, version ' + version)



def test():
    """Test script for the Quantum Information Toolkit.
    
    Ville Bergholm 2009-2011
    """
    import utils

    lmap.test()
    #utils.test()
    state.test()
    #gate.test()
    ho.test()
    invariant.test()
    markov.test()
    seq.test()
    print('All tests passed.')
