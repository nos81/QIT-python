# -*- coding: utf-8 -*-
"Unit tests for qit.seq"
# Ville Bergholm 2011-2014

import unittest
from numpy import pi
from numpy.random import rand, randn
from numpy.linalg import norm

# HACK to import the module in this source distribution, not the installed one!
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from qit import version
from qit.base import sx, sy, tol
from qit.state import state
from qit.utils import rand_positive
from qit.seq import *



class SeqTest(unittest.TestCase):
    def test_funcs(self):
        """Testing the control sequences module."""

        nmr([[3, 2], [1, 2], [-1, 0.3]])

        # pi rotation
        U = seq2prop(nmr([[pi, 0]]))
        self.assertAlmostEqual(norm(U +1j*sx), 0, delta=tol)
        U = seq2prop(nmr([[pi, pi/2]]))
        self.assertAlmostEqual(norm(U +1j*sy), 0, delta=tol)

        # rotation sequences in the absence of errors
        theta = pi * rand()
        phi = 2*pi * rand()
        U = seq2prop(nmr([[theta, phi]]))
        V = seq2prop(bb1(theta, phi, location = rand()))
        self.assertAlmostEqual(norm(U-V), 0, delta=tol)
        V = seq2prop(corpse(theta, phi))
        self.assertAlmostEqual(norm(U-V), 0, delta=tol)
        V = seq2prop(scrofulous(theta, phi))
        self.assertAlmostEqual(norm(U-V), 0, delta=tol)

        cpmg(2.0, 3)

        # equivalent propagations
        s = state(rand_positive(2))
        seq = scrofulous(pi*rand(), 2*pi*rand())
        s1 = s.u_propagate(seq2prop(seq))
        out, t = propagate(s, seq, base_dt=1)
        s2 = out[-1]
        self.assertAlmostEqual((s1-s2).norm(), 0, delta=tol)



if __name__ == '__main__':
    print('Testing QIT version ' + version())
    unittest.main()
