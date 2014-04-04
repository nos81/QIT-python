# -*- coding: utf-8 -*-
"Unit tests for qit.utils"
# Ville Bergholm 2009-2014

import unittest
from numpy import mat, array, dot, eye, trace
from numpy.random import rand, randn
from numpy.linalg import norm, det, eigvalsh
from scipy.linalg import expm

# HACK to import the module in this source distribution, not the installed one!
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from qit import version
from qit.base  import tol
from qit.utils import *


def randn_complex(*arg):
    "Returns an array of random complex numbers, normally distributed."
    return randn(*arg) +1j*randn(*arg)



class UtilsTest(unittest.TestCase):
    def setUp(self):
        pass


    def assertHermitian(self, H, delta):
        "Make sure H is hermitian."
        dim = len(H)
        H = mat(H)
        self.assertAlmostEqual(norm(H -H.H), 0, delta=delta)


    def assertUnitary(self, U, delta):
        "Make sure U is unitary."
        dim = len(U)
        U = mat(U)
        self.assertAlmostEqual(norm(U * U.H -eye(dim)), 0, delta=delta)
        self.assertAlmostEqual(norm(U.H * U -eye(dim)), 0, delta=delta)
        

    def test_funcs(self):
        "Testing the utils module."

        dim = 10

        ### expv
        A = randn_complex(dim, dim)
        H = rand_hermitian(dim)
        v = randn_complex(dim)
        newtol = 1e2 * tol  # use a larger tolerance here

        # arbitrary matrix with Arnoldi iteration
        w, err, hump = expv(1, A, v, m = dim // 2)
        self.assertAlmostEqual(norm(w - dot(expm(1*A), v)), 0, delta=newtol)
        w, err, hump = expv(1, A, v, m = dim)  # force a happy breakdown
        self.assertAlmostEqual(norm(w - dot(expm(1*A), v)), 0, delta=newtol)

        # Hermitian matrix with Lanczos iteration
        w, err, hump = expv(1, H, v, m = dim // 2, iteration='lanczos')
        self.assertAlmostEqual(norm(w - dot(expm(1*H), v)), 0, delta=newtol)
        w, err, hump = expv(1, H, v, m = dim, iteration = 'lanczos')
        self.assertAlmostEqual(norm(w - dot(expm(1*H), v)), 0, delta=newtol)

        # FIXME why does Lanczos work with nonhermitian matrices?
        w, err, hump = expv(1, A, v, m = dim // 2, iteration='lanczos')
        self.assertAlmostEqual(norm(w - dot(expm(1*A), v)), 0, delta=newtol)
        w, err, hump = expv(1, A, v, m = dim, iteration = 'lanczos')
        self.assertAlmostEqual(norm(w - dot(expm(1*A), v)), 0, delta=newtol)


        dim = 5

        ### random matrices
        H = rand_hermitian(dim)
        self.assertHermitian(H, delta=tol)

        U = rand_U(dim)
        self.assertUnitary(U, delta=tol)

        U = rand_SU(dim)
        self.assertUnitary(U, delta=tol)
        self.assertAlmostEqual(det(U), 1, delta=tol) # det 1

        rho = rand_positive(dim)
        self.assertHermitian(rho, delta=tol)
        self.assertAlmostEqual(trace(rho), 1, delta=tol) # trace 1
        temp = eigvalsh(rho)
        self.assertAlmostEqual(norm(temp.imag), 0, delta=tol) # real eigenvalues
        self.assertAlmostEqual(norm(temp - abs(temp)), 0, delta=tol) # nonnegative eigenvalues


        ### superoperators
        L = mat(rand_U(dim))
        R = mat(rand_U(dim))
        v = vec(array(rho))
        self.assertAlmostEqual(norm(rho -inv_vec(v)), 0, delta=tol)
        self.assertAlmostEqual(norm(L*rho*R -inv_vec(dot(lrmul(L, R), v))), 0, delta=tol)
        self.assertAlmostEqual(norm(L*rho -inv_vec(dot(lmul(L), v))), 0, delta=tol)
        self.assertAlmostEqual(norm(rho*R -inv_vec(dot(rmul(R), v))), 0, delta=tol)


        ### physical operators
        J = angular_momentum(dim)
        self.assertEqual(len(J), 3)  #  3 components
        for A in J:
            self.assertHermitian(A, delta=tol)
        self.assertAlmostEqual(norm(comm(J[0], J[1]) - 1j*J[2]), 0, delta=tol)  # [Jx, Jy] == i Jz

        a = mat(boson_ladder(dim))
        temp = comm(a, a.H)
        self.assertAlmostEqual(norm(temp[:-1, :-1] -eye(dim-1)), 0, delta=tol)  # [a, a'] == I  (truncated, so skip the last row/col!)

        fff = fermion_ladder(3)
        # {f_j, f_k} = 0
        # {f_j, f_k^\dagger} = I \delta_{jk}
        for f in fff:
            fp = f.conj().transpose()
            self.assertAlmostEqual(norm(acomm(f, f)), 0, delta=tol)
            self.assertAlmostEqual(norm(acomm(f, fp) -eye(8)), 0, delta=tol)
            for j in fff:
                if not f is j:
                    self.assertAlmostEqual(norm(acomm(j, f)), 0, delta=tol)
                    self.assertAlmostEqual(norm(acomm(j, fp)), 0, delta=tol)

        ### SU(2) rotations


        ### spectral decomposition
        E, P = spectral_decomposition(H)
        temp = 0
        for k in range(len(E)):
            temp += E[k] * P[k]
        self.assertAlmostEqual(norm(temp -H), 0, delta=tol)


        # tensor bases


        # majorization

        # op_list

        # plots



if __name__ == '__main__':
    print('Testing QIT version ' + version())
    unittest.main()
