# -*- coding: utf-8 -*-
"Unit tests for qit.seq"
# Ville Bergholm 2011-2016

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

        s = nmr([[3, 2], [1, 2], [-1, 0.3]])

        # pi rotation
        U = nmr([[pi, 0]]).to_prop()
        self.assertAlmostEqual(norm(U +1j*sx), 0, delta=tol)
        U = nmr([[pi, pi/2]]).to_prop()
        self.assertAlmostEqual(norm(U +1j*sy), 0, delta=tol)

        # rotation sequences in the absence of errors
        theta = pi * rand()
        phi = 2*pi * rand()
        U = nmr([[theta, phi]]).to_prop()
        V = bb1(theta, phi, location = rand()).to_prop()
        self.assertAlmostEqual(norm(U-V), 0, delta=tol)
        V = corpse(theta, phi).to_prop()
        self.assertAlmostEqual(norm(U-V), 0, delta=tol)
        V = scrofulous(theta, phi).to_prop()
        self.assertAlmostEqual(norm(U-V), 0, delta=tol)

        s = dd('cpmg', 2.0)

        # equivalent propagations
        s = state(rand_positive(2))
        seq = scrofulous(pi*rand(), 2*pi*rand())
        s1 = s.u_propagate(seq.to_prop())
        out, t = propagate(s, seq, base_dt=1)
        s2 = out[-1]
        self.assertAlmostEqual((s1-s2).norm(), 0, delta=tol)



if __name__ == '__main__':
    print('Testing QIT version ' + version())
    unittest.main()
