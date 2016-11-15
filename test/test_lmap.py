# -*- coding: utf-8 -*-
"Unit tests for qit.lmap"
# Ville Bergholm 2009-2016

import unittest
from numpy.random import rand, randn
from numpy.linalg import norm

# HACK to import the module in this source distribution, not the installed one!
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from qit import version
from qit.base  import tol
from qit.lmap  import lmap, tensor
from qit.utils import mkron, rand_GL


class LmapConstructorTest(unittest.TestCase):
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
        self.assertRaises(ValueError, lmap, rand(2,3), ((2,), (2, 3)))  # dimension mismatch
        self.assertRaises(ValueError, lmap, rand(2, 2, 2)) # bad array dimesion (3)



class LmapMethodTest(unittest.TestCase):
    def setUp(self):
        # generate some random lmaps
        self.idim = (2, 5, 3)
        self.odim = (4, 3, 2)
        self.L = []
        for i,o in zip(self.idim, self.odim):
            self.L.append(lmap(rand(o, i)))


    def test_methods(self):
        ### reorder, tensor product
        # build a rank-1 tensor out of the "local" lmaps
        T1 = tensor(*tuple(self.L))
        # permute them
        perms = [(2, 0, 1), (1, 0, 2), (2, 1, 0)]
        for p in perms:
            T2 = T1.reorder((p, p))
            tup = (self.L[i] for i in p)
            self.assertAlmostEqual((tensor(*tup) -T2).norm(), 0, delta=tol)

        ### tensorpow
        n = 3
        for A in self.L:
            # tensorpow and tensor product must give the same result
            tup = n * (A,)
            self.assertAlmostEqual((A.tensorpow(n) -tensor(*tup)).norm(), 0, delta=tol)



if __name__ == '__main__':
    print('Testing QIT version ' + version())
    unittest.main()
