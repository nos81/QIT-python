# -*- coding: utf-8 -*-
"Unit tests for qit.markov"
# Ville Bergholm 2009-2016

import unittest
from numpy.random import rand, randn
from numpy.linalg import norm

# HACK to import the module in this source distribution, not the installed one!
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from qit import version
from qit.base import tol
from qit.utils import rand_hermitian, superop_lindblad
from qit.markov import *



class SeqTest(unittest.TestCase):
    def test_funcs(self):
        """Testing the Markovian bath module."""

        dim = 6
        TU = 1e-9  # s
        H = rand_hermitian(dim)
        D = [rand_hermitian(dim)/10, rand_hermitian(dim)/10]
        B = [MarkovianBath('ohmic', 'boson', TU, 0.02), MarkovianBath('ohmic', 'fermion', TU, 0.03)]

        # jump operators
        dH, X = ops(H, D)
        self.assertEqual(len(D), X.shape[0])
        self.assertEqual(len(dH), X.shape[1])

        # jump ops should sum to D
        for n, A in enumerate(X):
            temp = 0
            for k in range(len(dH)):
                temp += A[k]
                if dH[k] != 0:
                    temp += A[k].conj().transpose() # A(-omega) == A'(omega)
            self.assertAlmostEqual(norm(temp - D[n]), 0, delta=tol) # Lindblad ops should sum to D

        # equivalence of Lindblad operators and the Liouvillian superoperator
        LL, H_LS = lindblad_ops(H, D, B)
        S1 = superop_lindblad(LL, H + H_LS)
        S2 = superop(H, D, B)
        self.assertAlmostEqual(norm(S1 - S2), 0, delta=tol)


if __name__ == '__main__':
    print('Testing QIT version ' + version())
    unittest.main()
