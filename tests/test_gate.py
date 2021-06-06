"""
Unit tests for qit.gate
"""
# Ville Bergholm 2010-2020

import pytest

import numpy as np
from numpy.linalg import norm

import qit
import qit.gate as gate


dim = (2, 4)


class TestGate:

    def test_dist(self):
        """Test the gate distance function.
        """
        U = gate.qft(dim)
        V = gate.mod_mul(2, dim, 5)
        x = gate.dist(U, V)

        I = gate.id(dim)
        S = gate.swap(*dim)
        # test bad input
        with pytest.raises(ValueError, match=''):
            gate.dist(I, S)  # output dimension mismatch


    def test_walsh(self):
        U = gate.walsh(3)


    def test_phase(self):
        # TODO test the output
        D = np.prod(dim)
        U = gate.phase(np.random.randn(np.prod(dim)), dim)
        with pytest.raises(ValueError, match=''):
            gate.phase(np.random.rand(D - 1), dim)  # dimension mismatch


    def test_mod_mul(self):
        V = gate.mod_mul(2, dim, 5)
        with pytest.raises(ValueError, match=''):
            gate.mod_mul(2, 4, 5)  # N too large
        with pytest.raises(ValueError, match=''):
            gate.mod_mul(2, 4)     # a and N not coprime


    def test_mod_inc(self):
        U = gate.mod_inc(3, dim, 5)
        with pytest.raises(ValueError, match=''):
            gate.mod_inc(1, 3, 4)  # N too large


    def test_mod_add(self):
        U = gate.mod_add(2, 4, 3)
        with pytest.raises(ValueError, match=''):
            gate.mod_add(2, 3, 4)  # N too large


    def test_controlled(self):

        U = gate.controlled(qit.sz, (1, 0), dim)

        with pytest.raises(ValueError, match=''):
            gate.controlled(U, (0,), dim)    # ctrl shorter than dim
        with pytest.raises(ValueError, match=''):
            gate.controlled(U, (0, 4), dim)  # ctrl on nonexistant state


    def test_single(self):
        U = gate.single(qit.sy, 0, dim)
        with pytest.raises(ValueError, match=''):
            gate.single(qit.sx, 1, dim)  # input dimension mismatch


    def test_two(self):
        cnot = gate.controlled(qit.sx)
        S = gate.swap(*dim)
        U = gate.two(cnot, (2, 0), (2, 3, 2))

        with pytest.raises(ValueError, match=''):
            gate.two(S, (0, 1), (2, 3, 4))   # input dimension mismatch
        with pytest.raises(ValueError, match=''):
            gate.two(S, (-1, 2), (2, 3, 4))  # bad targets
        with pytest.raises(ValueError, match=''):
            gate.two(S, (3, 2), (2, 3, 4))   # bad targets
        with pytest.raises(ValueError, match=''):
            gate.two(S, (0,), (2, 3, 4))     # wrong number of targets


    def test_swap(self, tol):
        """Swap gate.
        """
        I1 = gate.id(dim)
        I2 = gate.id(dim[::-1])  # subsystem order reversed
        S = gate.swap(*dim)
        # swap' * swap = I
        assert (S.ctranspose() @ S - I1).norm() == pytest.approx(0, abs=tol)
        assert (S @ S.ctranspose() - I2).norm() == pytest.approx(0, abs=tol)


    def test_dots(self, tol):
        """Copydot and plusdot linear maps.
        """
        n_in = 3
        n_out = 2
        d = 3
        C = gate.copydot(n_in, n_out, d)
        P = gate.plusdot(n_in, n_out, d)
        Q = gate.qft(d)
        # duality transformation using QFT
        temp = Q.tensorpow(n_out) @ C @ Q.tensorpow(n_in)
        assert (temp -P).norm() == pytest.approx(0, abs=tol)
