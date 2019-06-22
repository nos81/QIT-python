# -*- coding: utf-8 -*-
"Unit tests for qit.invariant"
# Ville Bergholm 2010-2018

import unittest
import numpy as np
from numpy import kron
from numpy.linalg import norm
from scipy.linalg import expm

from context import (qit, BaseTest, tol)

from qit import version
from qit.base import sx
from qit.utils import rand_U, rand_positive, rand_GL
import qit.gate as gate
from qit.invariant import *
from qit.state import state


class InvTest(BaseTest):
    def test_funcs(self):
        """Testing the invariants module."""

        U = rand_U(4)  # random two-qubit gate
        L = kron(rand_U(2), rand_U(2))  # random local 2-qubit gate
        dim = (2, 2)
        CNOT = gate.controlled(sx).data.todense()
        SWAP = gate.swap(*dim).data.todense()

        # canonical invariants
        #self.assertAlmostEqual(norm(canonical(L) -[0, 0, 0]), 0, delta=tol) # only point in Berkeley chamber with translation degeneracy, (0,0,0) =^ (1,0,0)
        self.assertAlmostEqual(norm(canonical(CNOT) -[0.5, 0, 0]), 0, delta=tol)
        self.assertAlmostEqual(norm(canonical(SWAP) -[0.5, 0.5, 0.5]), 0, delta=tol)

        # Makhlin invariants
        c = canonical(U)
        g1 = makhlin(c)
        g2 = makhlin(U)
        self.assertAlmostEqual(norm(g1-g2), 0, delta=tol)

        # maximum concurrence
        self.assertAlmostEqual(max_concurrence(L), 0, delta=tol)
        self.assertAlmostEqual(max_concurrence(SWAP), 0, delta=tol)
        self.assertAlmostEqual(max_concurrence(CNOT), 1, delta=tol)

        #plot_weyl_2q()
        #plot_makhlin_2q(25, 25)

        # Local unitary invariants of states
        rho = state(rand_positive(4), dim)
        self.assertAlmostEqual(LU(rho, 2, [(), ()]), 1, delta=tol)  # trace of the state
        # invariance under LU maps
        perms = [(), (1,0)]
        self.assertAlmostEqual(LU(rho, 2, perms), LU(rho.u_propagate(L), 2, perms), delta=tol)

        # invariance under LU maps
        perms = [(), (1,2,0)]
        self.assertAlmostEqual(LU(rho, 3, perms), LU(rho.u_propagate(L), 3, perms), delta=tol)

        # orthogonal matrices
        temp = np.random.randn(6,6); temp = temp-temp.T; Z = expm(temp)
        temp = np.random.randn(6,6); temp = temp-temp.T; W = expm(temp)
        W = W[:,:]
        Z = Z[:,:4]

        temp = gate_leakage(CNOT, (2, 2), Z, W)
        temp = gate_leakage(L, (2, 2), Z, W)
        temp = gate_leakage(U, (2, 2), Z, W)


if __name__ == '__main__':
    print('Testing QIT version ' + version())
    unittest.main()
