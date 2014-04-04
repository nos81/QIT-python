# -*- coding: utf-8 -*-
"Unit tests for qit.ho"
# Ville Bergholm 2009-2014

import unittest
from numpy import eye, diag, ones, mat
from numpy.random import rand, randn
from numpy.linalg import norm

# HACK to import the module in this source distribution, not the installed one!
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from qit import version
from qit.base import tol
from qit.state import state
from qit.ho import *
from qit.utils import boson_ladder, comm



class HOTest(unittest.TestCase):
    def setUp(self):
        pass
        

    def test_funcs(self):
        """Testing the harmonic oscillator module."""

        def randc():
            "Random complex number."
            return randn() + 1j*randn()

        n = 35
        s0 = state(0, n)

        ### displacement
        alpha = randc()
        D = displace(alpha, n=n)
        s = coherent_state(alpha, n=n)
        self.assertAlmostEqual((s -s0.u_propagate(D)).norm(), 0, delta=tol)

        ### squeeezing TODO assert
        z = randc()
        S = squeeze(z, n=n)

        ### position and momentum operators, eigenstates
        Q = position(n)
        P = momentum(n)
        q = randn()
        p = randn()
        sq = position_state(q, n=n)
        sp = momentum_state(p, n=n)
        temp = 1e-1 # the truncation accuracy is not amazing here TODO why?
        # expectation values in eigenstates
        self.assertAlmostEqual(sq.ev(Q), q, delta=temp)
        self.assertAlmostEqual(sp.ev(P), p, delta=temp)
        # [Q, P] = i
        temp = ones(n)
        temp[-1] = -n+1 # truncation...
        self.assertAlmostEqual(norm(comm(Q, P) -1j * diag(temp)), 0, delta=tol)
        # P^2 +Q^2 = 2a^\dagger * a + 1
        a = mat(boson_ladder(n))
        self.assertAlmostEqual(norm(mat(P)**2 +mat(Q)**2 -2 * a.H * a -diag(temp)), 0, delta=tol)



if __name__ == '__main__':
    print('Testing QIT version ' + version())
    unittest.main()
