# -*- coding: utf-8 -*-
"Unit tests for qit.lmap"
# Ville Bergholm 2009-2014

import unittest
from numpy.random import rand, randn
from numpy.linalg import norm

# HACK to import the module in this source distribution, not the installed one!
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from qit import version
from qit.base  import tol
from qit.lmap  import lmap, tensor
from qit.utils import mkron


def randn_complex(*arg):
    "Returns an array of random complex numbers, normally distributed."
    return randn(*arg) +1j*randn(*arg)


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
        temp = lmap(randn_complex(4, 4))         # both dims inferred

        # copy constructor
        s = lmap(temp, ((1, 4), (2, 2))) # dims reinterpreted
        s = lmap(temp, ((1, 4), None))   # input dims kept

        # bad inputs
        self.assertRaises(ValueError, lmap, rand(2,3), ((2,), (2, 3)))  # dimension mismatch
        self.assertRaises(ValueError, lmap, rand(2, 2, 2)) # bad array dimesion (3)



class LmapMethodTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_methods(self):
        ### reorder, tensor
        idim = (2, 5, 3)
        odim = (4, 3, 2)
        # generate random local maps
        parts = []
        for k in range(len(idim)):
            parts.append(lmap(rand(odim[k], idim[k])))
        T1 = tensor(*tuple(parts))
        # permute them
        perms = [(2, 0, 1), (1, 0, 2), (2, 1, 0)]
        for p in perms:
            T2 = T1.reorder((p, p))
            tup = (parts[i] for i in p)
            self.assertAlmostEqual((tensor(*tup) -T2).norm(), 0, delta=tol)

        ignore = """
        print(repr(b * c))
        print(repr(tensor(a, b)))
        """



if __name__ == '__main__':
    print('Testing QIT version ' + version())
    unittest.main()
