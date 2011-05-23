# -*- coding: utf-8 -*-
"""Python Quantum Information Toolkit

See the README.txt file included in this distribution
or the project website at http://sf.net/projects/qit/

Authors:
Ville Bergholm 2011
"""

from __future__ import print_function, division

#import numpy as np
import scipy as sp
import scipy.constants as const
import matplotlib.pyplot as plt

from base import *
from lmap import *
from state import *
from utils import *
import gate

# toolkit version number
version = '0.9.10'

print('Python Quantum Information Toolkit, version ' + version)



def test():
    """Test script for the Quantum Information Toolkit.
    
    Ville Bergholm 2009-2011
    """
    lmap.test()
    state.test()
    utils.test()
    gate.test()
    #markov.test_markov
    print('All tests passed.')
