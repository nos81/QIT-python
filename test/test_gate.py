"""
Unit tests for qit.gate
"""
# Ville Bergholm 2010-2020

import pytest

import numpy as np
from numpy.linalg import norm

import qit.gate as g
from qit.base import sx, sy, sz
from qit.gate import *


dim = (2, 4)


class TestGate:

    def test_dist(self):
        """Test the gate distance function.
        """
        U = g.qft(dim)
        V = g.mod_mul(2, dim, 5)
        x = dist(U, V)

        I = g.id(dim)
        S = g.swap(*dim)
        # test bad input
        with pytest.raises(ValueError, match=''):
            g.dist(I, S)  # output dimension mismatch


    def test_walsh(self):
        U = g.walsh(3)


    def test_phase(self):
        # TODO test the output
        D = np.prod(dim)
        U = g.phase(np.random.randn(np.prod(dim)), dim)
        with pytest.raises(ValueError, match=''):
            g.phase(np.random.rand(D - 1), dim)  # dimension mismatch


    def test_mod_mul(self):
        V = g.mod_mul(2, dim, 5)
        with pytest.raises(ValueError, match=''):
            g.mod_mul(2, 4, 5)  # N too large
        with pytest.raises(ValueError, match=''):
            g.mod_mul(2, 4)     # a and N not coprime


    def test_mod_inc(self):
        U = g.mod_inc(3, dim, 5)
        with pytest.raises(ValueError, match=''):
            g.mod_inc(1, 3, 4)  # N too large


    def test_mod_add(self):
        U = g.mod_add(2, 4, 3)
        with pytest.raises(ValueError, match=''):
            g.mod_add(2, 3, 4)  # N too large


    def test_controlled(self):

        U = g.controlled(sz, (1, 0), dim)

        with pytest.raises(ValueError, match=''):
            g.controlled(U, (0,), dim)    # ctrl shorter than dim
        with pytest.raises(ValueError, match=''):
            g.controlled(U, (0, 4), dim)  # ctrl on nonexistant state


    def test_single(self):
        U = g.single(sy, 0, dim)
        with pytest.raises(ValueError, match=''):
            g.single(sx, 1, dim)  # input dimension mismatch


    def test_two(self):
        cnot = g.controlled(sx)
        S = g.swap(*dim)
        U = g.two(cnot, (2, 0), (2, 3, 2))

        with pytest.raises(ValueError, match=''):
            g.two(S, (0, 1), (2, 3, 4))   # input dimension mismatch
        with pytest.raises(ValueError, match=''):
            g.two(S, (-1, 2), (2, 3, 4))  # bad targets
        with pytest.raises(ValueError, match=''):
            g.two(S, (3, 2), (2, 3, 4))   # bad targets
        with pytest.raises(ValueError, match=''):
            g.two(S, (0,), (2, 3, 4))     # wrong number of targets


    def test_swap(self, tol):
        """Swap gate.
        """
        I = g.id(dim)
        S = g.swap(*dim)
        # swap' * swap = I
        # FIXME with scipy 0.16 we'll have norm for sparse arrays
        assert norm((S.ctranspose() * S -I).data.A) == pytest.approx(0, abs=tol)


    def test_dots(self, tol):
        """Copydot and plusdot linear maps.
        """
        n_in = 3
        n_out = 2
        d = 3
        C = g.copydot(n_in, n_out, d)
        P = g.plusdot(n_in, n_out, d)
        Q = g.qft(d)
        temp = Q.tensorpow(n_out) * C * Q.tensorpow(n_in)
        assert (temp -P).norm() == pytest.approx(0, abs=tol)
