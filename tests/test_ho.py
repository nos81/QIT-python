"""
Unit tests for qit.ho
"""
# Ville Bergholm 2009-2020

import pytest

import numpy as np
from numpy.linalg import norm

import qit
import qit.ho as ho
from qit.utils import boson_ladder, comm


def randc():
    "Random complex number."
    return np.random.randn() + 1j * np.random.randn()

n = 35  # truncation dimension
s0 = qit.State(0, n)



class TestHO:
    """Testing the harmonic oscillator module.
    """

    def test_displacement(self, tol):
        """Displacement operator.
        """
        # coherent states are displaced zero states
        alpha = randc()
        D = ho.displace(alpha, n=n)
        s = ho.coherent_state(alpha, n=n)
        assert (s -s0.u_propagate(D)).norm() == pytest.approx(0, abs=tol)

    def test_squeeze(self):
        ### squeeezing TODO assert
        z = randc()
        S = ho.squeeze(z, n=n)


    def test_PQ(self, tol):
        """Position and momentum operators.
        """
        Q = ho.position(n)
        P = ho.momentum(n)
        q = np.random.randn()
        p = np.random.randn()
        sq = ho.position_state(q, n=n)
        sp = ho.momentum_state(p, n=n)
        temp = 1e-1 # the truncation accuracy is not amazing here TODO why?

        # expectation values in eigenstates
        assert sq.ev(Q) == pytest.approx(q, abs=temp)
        assert sp.ev(P) == pytest.approx(p, abs=temp)

        # [Q, P] = i
        I = np.ones(n)
        I[-1] = -n+1 # truncation...
        assert norm(comm(Q, P) -1j * np.diag(I)) == pytest.approx(0, abs=tol)
        # P^2 +Q^2 = 2a^\dagger * a + 1
        a = boson_ladder(n)
        assert norm(P @ P + Q @ Q -2 * a.T.conj() @ a -np.diag(I)) == pytest.approx(0, abs=tol)
