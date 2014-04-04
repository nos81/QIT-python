# -*- coding: utf-8 -*-
"Unit tests for qit.gate"
# Ville Bergholm 2010-2014

import unittest
from numpy import prod
from numpy.random import rand, randn
from numpy.linalg import norm

# HACK to import the module in this source distribution, not the installed one!
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from qit import version
from qit.base import sx, sy, sz, tol
from qit.gate import *


class GateTest(unittest.TestCase):
    def test_funcs(self):
        """Testing the quantum gates module."""

        dim = (2, 4)
        D = prod(dim)

        I = id(dim)
        S = swap(*dim)
        # swap' * swap = I
        self.assertAlmostEqual((S.ctranspose() * S -I).norm(), 0, delta=tol)

        # TODO test the output
        U = phase(randn(prod(dim)), dim)
        U = walsh(3)
        U = qft(dim)
        U = mod_inc(3, dim, 5)
        V = mod_mul(2, dim, 5)
        U = mod_add(2, 4, 3)
        x = dist(U, V)
        U = controlled(sz, (1, 0), dim)
        cnot = controlled(sx)
        U = single(sy, 0, dim)
        U = two(cnot, (2, 0), (2, 3, 2))

        # test bad input
        self.assertRaises(ValueError, dist, I, S)  # output dimension mismatch
        self.assertRaises(ValueError, phase, rand(D-1), dim)  # dimension mismatch
        self.assertRaises(ValueError, mod_inc, 1, 3, 4)  # N too large
        self.assertRaises(ValueError, mod_mul, 2, 4, 5)  # N too large
        self.assertRaises(ValueError, mod_mul, 2, 4)     # a and N not coprime
        self.assertRaises(ValueError, mod_add, 2, 3, 4)  # N too large
        self.assertRaises(ValueError, controlled, U, (0,), dim)    # ctrl shorter than dim
        self.assertRaises(ValueError, controlled, U, (0, 4), dim)  # ctrl on nonexistant state
        self.assertRaises(ValueError, single, sx, 1, dim)  # input dimension mismatch
        self.assertRaises(ValueError, two, S, (0, 1), (2, 3, 4))   # input dimension mismatch
        self.assertRaises(ValueError, two, S, (-1, 2), (2, 3, 4))  # bad targets
        self.assertRaises(ValueError, two, S, (3, 2), (2, 3, 4))   # bad targets
        self.assertRaises(ValueError, two, S, (0,), (2, 3, 4))     # wrong number of targets



if __name__ == '__main__':
    print('Testing QIT version ' + version())
    unittest.main()
