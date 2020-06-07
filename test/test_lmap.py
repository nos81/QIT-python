"""
Unit tests for qit.lmap
"""
# Ville Bergholm 2009-2020

import pytest

import numpy as np
from numpy.random import rand, randn
from numpy.linalg import norm
import scipy.sparse as sparse

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

        # scalar
        s = lmap(randn())
        assert s.dim == ((1,), (1,))

        # 1D
        s = lmap(randn(4))                # ket vector as 1D array
        assert s.dim == ((4,), (1,))
        s = lmap(randn(4, 1))             # ket vector
        assert s.dim == ((4,), (1,))

        s = lmap(randn(3), ((1,), None))  # bra vector as 1D array
        assert s.dim == ((1,), (3,))
        s = lmap(randn(1, 3))             # bra vector
        assert s.dim == ((1,), (3,))

        # 2D
        s = lmap(randn(4, 5), ((2, 2), None))    # input dims inferred
        assert s.dim == ((2, 2), (5,))
        s = lmap(randn(3, 6), (None, (3, 2)))    # output dims inferred
        assert s.dim == ((3,), (3, 2))
        s = lmap(randn(6, 6), ((2, 3), (3, 2)))  # all dims given
        assert s.dim == ((2, 3,), (3, 2))
        s = lmap(randn(4, 4))                  # both dims inferred
        assert s.dim == ((4,), (4,))
        temp = s

        # copy constructor
        s = lmap(temp, ((1, 4), (2, 2))) # dims reinterpreted
        assert s.dim == ((1, 4), (2, 2))
        s = lmap(temp, ((1, 4), None))   # input dims kept
        assert s.dim == ((1, 4), (4,))

        # sparse lmaps
        s = lmap(sparse.eye(3, 2))    # input dims inferred
        assert s.dim == ((3,), (2,))
        s = lmap(sparse.eye(3, 4), (None, (2, 2)))    # input dims inferred
        assert s.dim == ((3,), (2, 2))

        # bad inputs
        with pytest.raises(ValueError, match='Dimensions of the array do not match the combined dimensions of the subsystems.'):
            lmap(rand(2, 3), ((2,), (2, 3)))  # dimension mismatch
        with pytest.raises(ValueError, match='Array dimension must be <= 2.'):
            lmap(rand(2, 2, 2))  # bad array dimension (3)


    def test_utilities(self):
        """lmap utilities."""

        s = lmap(randn(4), ((1, 1, 2, 1, 2, 1), (1,)))
        assert s.dim == ((1, 1, 2, 1, 2, 1), (1,))
        s.remove_singletons()
        assert s.dim == ((2, 2), (1,))

        t = lmap(randn(4), ((4,), (1,)))
        u = lmap(randn(4), ((2, 2), (1,)))
        assert s.is_compatible(s)
        assert not s.is_compatible(t)
        assert s.is_compatible(u)
        with pytest.raises(TypeError, match='is not an lmap'):
            assert s.is_compatible(randn(4))

        assert s.is_ket()
        t = lmap(randn(4), ((1,), None))
        assert not t.is_ket()
        t = lmap(randn(4, 4))
        assert not t.is_ket()


    def test_algebra(self):
        """Algebraic operations on lmaps."""

        s = lmap(randn(4, 2))
        t = lmap(randn(2, 3))
        u = lmap(randn(3, 3))

        # addition
        r = s + s
        with pytest.raises(ValueError, match='lmaps are not compatible'):
            r = s + t
        with pytest.raises(TypeError, match='is not an lmap'):
            r = s + 1.2

        s += s
        with pytest.raises(TypeError, match='is not an lmap'):
            s += 0.4

        # subtraction
        r = s - s
        with pytest.raises(ValueError, match='lmaps are not compatible'):
            r = s - t
        with pytest.raises(TypeError, match='is not an lmap'):
            r = s - 1.2

        s -= s
        with pytest.raises(TypeError, match='is not an lmap'):
            s -= 0.4

        # scalar multiplication
        r = 2 * s
        r = s * 5
        with pytest.raises(TypeError, match='operator is for scalar multiplication only'):
            r = s * t

        s *= 1.5
        with pytest.raises(TypeError, match='operator is for scalar multiplication only'):
            s *= u

        # scalar division
        r = s / 2.0
        with pytest.raises(TypeError, match='operator is for scalar division only'):
            r = s / t

        s /= 2.5
        with pytest.raises(TypeError, match='operator is for scalar division only'):
            s /= u

        # concatenation
        r = s @ t
        assert r.dim == ((4,), (3,))
        with pytest.raises(ValueError, match='dimensions do not match'):
            r = t @ s
        with pytest.raises(TypeError, match='@ operator is for lmap concatenation only'):
            r = s @ 3.0
        with pytest.raises(TypeError, match='unsupported operand'):
            # we haven't defined __rmatmul__
            r = 1.2 @ s

        # matrix power
        r = u ** 1
        r = u ** 3
        with pytest.raises(TypeError, match='exponent must be an integer'):
            r = u ** 3.3
        with pytest.raises(ValueError, match='The input and output dimensions do not match'):
            r = s ** 2


    def test_trace(self):
        s = lmap(randn(4, 4), ((2, 2), (2, 2)))
        t = lmap(s, ((2, 2), (4,)))
        r = s.trace()
        with pytest.raises(ValueError, match='Trace not defined for non-endomorphisms'):
            r = t.trace()


    def test_norm(self):
        s = lmap(randn(4, 3), ((2, 2), (3,)))
        r = s.norm()
        s = lmap(sparse.eye(4, 3), ((2, 2), (3,)))
        r = s.norm()


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
        n = 3
        for A in lmaps:
            # tensorpow and tensor product must give the same result
            tup = n * (A,)
            assert (A.tensorpow(n) -tensor(*tup)).norm() == pytest.approx(0, abs=tol)

    def test_tensor(self, lmaps):
        u = lmap(randn(4, 3))
        t = lmap(randn(3))
        s = lmap(sparse.eye(4, 3), ((2, 2), (3,)))

        r = tensor(u)
        r = tensor(u, t)
        assert not r.is_sparse()
        r = tensor(u, t, s)
        assert r.is_sparse()
