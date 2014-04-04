# -*- coding: utf-8 -*-
"Unit tests for qit.invariant"
# Ville Bergholm 2010-2014

import unittest
from numpy import kron
from numpy.linalg import norm

# HACK to import the module in this source distribution, not the installed one!
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from qit import version
from qit.base import sx, tol
from qit.utils import rand_U
import qit.gate as gate
from qit.invariant import *


class InvTest(unittest.TestCase):
    def test_funcs(self):
        """Testing the invariants module."""

        U = rand_U(4)  # random two-qubit gate
        L = kron(rand_U(2), rand_U(2))  # random local 2-qubit gate
        cnot = gate.controlled(sx).data
        swap = gate.swap(2, 2).data

        # canonical invariants
        #self.assertAlmostEqual(norm(canonical(L) -[0, 0, 0]), 0, delta=tol) # only point in Berkeley chamber with translation degeneracy, (0,0,0) =^ (1,0,0)
        self.assertAlmostEqual(norm(canonical(cnot) -[0.5, 0, 0]), 0, delta=tol)
        self.assertAlmostEqual(norm(canonical(swap) -[0.5, 0.5, 0.5]), 0, delta=tol)

        # Makhlin invariants
        c = canonical(U)
        g1 = makhlin(c)
        g2 = makhlin(U)
        self.assertAlmostEqual(norm(g1-g2), 0, delta=tol)

        # maximum concurrence
        self.assertAlmostEqual(max_concurrence(L), 0, delta=tol)
        self.assertAlmostEqual(max_concurrence(swap), 0, delta=tol)
        self.assertAlmostEqual(max_concurrence(cnot), 1, delta=tol)

        #plot_weyl_2q()
        #plot_makhlin_2q(25, 25)


if __name__ == '__main__':
    print('Testing QIT version ' + version())
    unittest.main()
