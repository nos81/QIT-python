"""
Unit tests for qit.invariant
"""
# Ville Bergholm 2010-2020

import pytest

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm

from qit.base import sx
from qit.utils import rand_U, rand_positive, rand_GL
import qit.gate as gate
import qit.invariant as qi
from qit.state import State


@pytest.fixture(scope="module")
def L():
    """Random local 2-qubit gate."""
    return np.kron(rand_U(2), rand_U(2))

@pytest.fixture(scope="module")
def U():
    """Random 2-qubit gate."""
    return rand_U(4)


dim = (2, 2)
CNOT = gate.controlled(sx).data.A
SWAP = gate.swap(*dim).data.A


class TestInvariants:
    """Testing the invariants module."""

    def test_canonical_inv(self, tol, L, U):
        """Canonical invariants."""

        # canonical invariants
        #assert norm(qi.canonical_inv(L) -[0, 0, 0]) == pytest.approx(0, abs=tol) # only point in Berkeley chamber with translation degeneracy, (0,0,0) =^ (1,0,0)
        assert norm(qi.canonical_inv(CNOT) -[0.5, 0, 0]) == pytest.approx(0, abs=tol)
        assert norm(qi.canonical_inv(SWAP) -[0.5, 0.5, 0.5]) == pytest.approx(0, abs=tol)

    def test_makhlin_inv(self, tol, L, U):
        """Makhlin invariants."""
        c = qi.canonical_inv(U)
        g1 = qi.makhlin_inv(c)
        g2 = qi.makhlin_inv(U)
        assert norm(g1-g2) == pytest.approx(0, abs=tol)
        g = qi.makhlin_inv(L)
        print(g)

    def test_gate_max_concurrence(self, tol, L, U):
        """Maximum concurrence.
        """
        assert qi.gate_max_concurrence(L) == pytest.approx(0, abs=tol)
        assert qi.gate_max_concurrence(SWAP) == pytest.approx(0, abs=tol)
        assert qi.gate_max_concurrence(CNOT) == pytest.approx(1, abs=tol)

        #plot_weyl_2q()
        #plot_makhlin_2q(25, 25)

    def test_state_inv(self, tol, L):
        """Local unitary invariants of states.
        """
        rho = State(rand_positive(4), dim)
        assert qi.state_inv(rho, 2, [(), ()]) == pytest.approx(1, abs=tol)  # trace of the state
        # invariance under LU maps
        perms = [(), (1,0)]
        assert qi.state_inv(rho, 2, perms) == pytest.approx(qi.state_inv(rho.u_propagate(L), 2, perms), abs=tol)

        # invariance under LU maps
        perms = [(), (1,2,0)]
        assert qi.state_inv(rho, 3, perms) == pytest.approx(qi.state_inv(rho.u_propagate(L), 3, perms), abs=tol)

    def test_gate_leakage_inv(self, tol, L, U):

        U = rand_U(4)  # random two-qubit gate

        # orthogonal matrices
        temp = np.random.randn(6,6); temp = temp-temp.T; Z = expm(temp)
        temp = np.random.randn(6,6); temp = temp-temp.T; W = expm(temp)
        W = W[:,:]
        Z = Z[:,:4]

        temp = qi.gate_leakage_inv(CNOT, (2, 2), Z, W)
        temp = qi.gate_leakage_inv(L, (2, 2), Z, W)
        temp = qi.gate_leakage_inv(U, (2, 2), Z, W)
