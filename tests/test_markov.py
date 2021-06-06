"""Unit tests for qit.markov"""
# Ville Bergholm 2009-2020

import pytest
import numpy as np
from numpy.linalg import norm

import qit
from qit.utils import rand_hermitian, superop_lindblad
from qit.markov import *


@pytest.fixture(scope="session")
def dim():
    """Vector dimension."""
    return 6


class TestMarkov:
    """Testing the Markovian bath module."""

    def test_jump_ops(self, dim, tol):
        """Jump operators.
        """
        H = rand_hermitian(dim)
        D = [rand_hermitian(dim)/10, rand_hermitian(dim)/10]
        dH, X = ops(H, D)

        assert len(D) == X.shape[0]
        assert len(dH) == X.shape[1]

        # jump ops should sum to D
        for n, A in enumerate(X):
            temp = 0
            for k in range(len(dH)):
                temp += A[k]
                if dH[k] != 0:
                    temp += A[k].conj().transpose() # A(-omega) == A'(omega)
            assert norm(temp - D[n]) == pytest.approx(0, abs=tol)  # Lindblad ops should sum to D


    def test_lindblad_ops(self, dim, tol):
        """Lindblad operators.
        """
        TU = 1e-9  # s
        H = rand_hermitian(dim)
        D = [rand_hermitian(dim)/10, rand_hermitian(dim)/10]
        B = [MarkovianBath('ohmic', 'boson', TU, 0.02), MarkovianBath('ohmic', 'fermion', TU, 0.03)]

        # equivalence of Lindblad operators and the Liouvillian superoperator
        LL, H_LS = lindblad_ops(H, D, B)
        S1 = superop_lindblad(LL, H + H_LS)
        S2 = superop(H, D, B)
        assert norm(S1 - S2) == pytest.approx(0, abs=tol)
