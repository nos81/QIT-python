"""
Unit tests for qit.lmap
"""
# Ville Bergholm 2009-2020

import pytest

import numpy as np
from numpy.random import rand, randn
from numpy.linalg import norm

from qit.lmap  import lmap, tensor
from qit.utils import mkron, rand_GL


@pytest.fixture(scope="session")
def lmaps():
    # generate some random lmaps
    idim = (2, 5, 3)
    odim = (4, 3, 2)
    L = []
    for i, o in zip(idim, odim):
        L.append(lmap(rand(o, i)))
    return L


class TestLmap:
    def test_constructor(self):
        "Test lmap.__init__"

        # 1D
        s = lmap(randn(4, 1))             # ket vector
        s = lmap(randn(4))                # ket vector as 1D array
        s = lmap(randn(1, 3))             # bra vector
        s = lmap(randn(3), ((1,), None))  # bra vector as 1D array
        # 2D
        s = lmap(randn(4, 5), ((2, 2), None))    # input dims inferred
        s = lmap(randn(3, 6), (None, (3, 2)))    # output dims inferred
        s = lmap(randn(6, 6), ((2, 3), (3, 2)))  # all dims given
        temp = lmap(rand_GL(4))                  # both dims inferred

        # copy constructor
        s = lmap(temp, ((1, 4), (2, 2))) # dims reinterpreted
        s = lmap(temp, ((1, 4), None))   # input dims kept

        # bad inputs
        with pytest.raises(ValueError, match='Dimensions of the array do not match the combined dimensions of the subsystems.'):
            lmap(rand(2,3), ((2,), (2, 3)))  # dimension mismatch
        with pytest.raises(ValueError, match='Array dimension must be <= 2.'):
            lmap(rand(2, 2, 2)) # bad array dimesion (3)


    def test_reorder(self, lmaps, tol):
        L = lmaps
        # build a rank-1 tensor out of the "local" lmaps
        T1 = tensor(*L)
        # permute them
        perms = [(2, 0, 1), (1, 0, 2), (2, 1, 0)]
        for p in perms:
            T2 = T1.reorder((p, p))
            tup = (L[i] for i in p)
            assert (tensor(*tup) -T2).norm() == pytest.approx(0, abs=tol)


    def test_tensorpow(self, lmaps, tol):

        ### tensorpow
        n = 3
        for A in lmaps:
            # tensorpow and tensor product must give the same result
            tup = n * (A,)
            assert (A.tensorpow(n) -tensor(*tup)).norm() == pytest.approx(0, abs=tol)
