# -*- coding: utf-8 -*-
"Unit tests for qit.state"
# Ville Bergholm 2008-2014

import unittest
from numpy import prod, sqrt, log2, kron
from numpy.random import rand, randn
from numpy.linalg import norm

# HACK to import the module in this source distribution, not the installed one!
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from qit import version
from qit.base  import tol
from qit.lmap  import lmap
from qit.state import state
from qit.utils import rand_positive, rand_U, mkron


class StateConstructorTest(unittest.TestCase):
    def test_constructor(self):
        "Test state.__init__"

        # copy constructor
        temp = lmap(randn(4, 4), ((4,), (4,)))
        s = state(temp, (2, 2))
        # strings
        s = state('10011')
        s = state('2014', (3, 2, 3, 5))
        s = state('GHZ')
        s = state('GHZ', (3, 2, 3))
        s = state('W', (2, 3, 2))
        s = state('Bell2')
        # basis kets
        s = state(4, 6)
        s = state(11, (3, 5, 2))
        # kets and state ops
        s = state(rand(5))
        s = state(rand(6), (3, 2))
        s = state(rand(3, 3))
        s = state(rand(4, 4), (2, 2))
        # bad inputs
        temp = lmap(randn(2, 3), ((2,), (3,)))
        self.assertRaises(ValueError, state, temp)          # nonsquare lmap
        self.assertRaises(ValueError, state, 'rubbish')     # unknown state name
        self.assertRaises(ValueError, state, 0)             # missing dimension
        self.assertRaises(ValueError, state, 2, 2)          # ket number too high
        self.assertRaises(ValueError, state, [])            # bad array dimesion (0)
        self.assertRaises(ValueError, state, rand(2, 2, 2)) # bad array dimesion (3)
        self.assertRaises(ValueError, state, randn(3, 4))   # nonsquare array



class StateMethodTest(unittest.TestCase):
    def setUp(self):
        dim = (2, 5, 3)
        self.dim = dim
        # mixed state
        self.rho = state(rand_positive(prod(dim)), dim)
        # pure state
        self.psi = state(0, dim).u_propagate(rand_U(prod(dim)))
        # random unitary
        self.U = rand_U(prod(dim))

    def test_methods(self):
        # TODO concurrence, fix_phase, kraus_propagate, locc_convertible, lognegativity, measure,
        # negativity,

        D = prod(self.dim)
        bipartitions = [[0], [0, 2]]

        ### generalized Bloch vectors.

        temp = self.rho.bloch_vector()
        # round trip
        self.assertAlmostEqual((state.bloch_state(temp) -self.rho).norm(), 0, delta=tol)
        # correlation tensor is real
        self.assertAlmostEqual(norm(temp.imag), 0, delta=tol)
        # state purity limits the Frobenius norm
        self.assertTrue(norm(temp.flat) -sqrt(D) <= tol)
        # state normalization
        self.assertAlmostEqual(temp.flat[0], 1, delta=tol)


        ### entropy

        temp = self.rho.entropy()
        # zero for pure states
        self.assertAlmostEqual(self.psi.entropy(), 0, delta=tol)
        # nonnegative
        self.assertTrue(temp >= -tol)
        # upper limit is log2(D)
        self.assertTrue(temp <= log2(D) +tol)
        # invariant under unitary transformations
        self.assertAlmostEqual(temp, self.rho.u_propagate(self.U).entropy(), delta=tol)


        ### ptrace, ptranspose

        temp = self.rho.trace()
        for sys in bipartitions:
            rho_A = self.rho.ptrace(sys)
            # trace of partial trace equals total trace
            self.assertAlmostEqual(temp, rho_A.trace(), delta=tol)
        # partial trace over all subsystems equals total trace
        self.assertAlmostEqual(temp, self.rho.ptrace(range(self.rho.subsystems())).trace(), delta=tol)

        for sys in bipartitions:
            rho_T = self.rho.ptranspose(sys)
            # two ptransposes cancel
            self.assertAlmostEqual((self.rho -rho_T.ptranspose(sys)).norm(), 0, delta=tol)
            # ptranspose preserves trace
            self.assertAlmostEqual(temp, rho_T.trace(), delta=tol)

        ### schmidt
        return
        # FIXME svdvals causes a crash in schmidt! see if fixed in scipy 0.13.0.
        for sys in bipartitions:
            lambda1, u, v = self.psi.schmidt(sys, full=True)
            lambda2 = self.psi.schmidt(self.psi.invert_selection(sys))
            # squares of schmidt coefficients sum up to unity
            self.assertAlmostEqual(norm(lambda1), 1, delta=tol)
            # both subdivisions have identical schmidt coefficients
            self.assertAlmostEqual(norm(lambda1 -lambda2), 0, delta=tol)

            # decomposition is equal to the original matrix
            temp = 0
            for k in range(len(lambda1)):
                temp += kron(lambda1[k] * u[:, k], v[:, k])
            self.assertAlmostEqual(norm(self.psi.data.ravel() -temp), 0, delta=tol)

        # squared schmidt coefficients equal eigenvalues of partial trace
        #r = state(randn(30) + 1j*randn(30), [5, 6]).normalize()
        #x = r.schmidt([0]) ** 2  # FIXME crashes ipython!
        #temp = r.ptrace([1])
        #y, dummy = eighsort(temp.data)
        #self.assertAlmostEqual(norm(x-y), 0, delta=tol)



class StateMethod2Test(unittest.TestCase):
    def setUp(self):
        pass

    def test_methods(self):
        ### reorder

        dim = (2, 5, 1)
        A = rand(dim[0], dim[0])
        B = rand(dim[1], dim[1])
        C = rand(dim[2], dim[2])

        T1 = state(mkron(A, B, C), dim)
        T2 = T1.reorder([2, 0, 1])
        self.assertAlmostEqual(norm(mkron(C, A, B) - T2.data), 0, delta=tol)
        T2 = T1.reorder([1, 0, 2])
        self.assertAlmostEqual(norm(mkron(B, A, C) - T2.data), 0, delta=tol)
        T2 = T1.reorder([2, 1, 0])
        self.assertAlmostEqual(norm(mkron(C, B, A) - T2.data), 0, delta=tol)



class StateBinaryFuncTest(unittest.TestCase):
    def setUp(self):
        dim = (2, 3)
        # two mixed states
        self.rho1 = state(rand_positive(prod(dim)), dim)
        self.rho2 = state(rand_positive(prod(dim)), dim)
        # two pure states
        p = state(0, dim)
        self.p1 = p.u_propagate(rand_U(prod(dim)))
        self.p2 = p.u_propagate(rand_U(prod(dim)))
        # random unitary
        self.U = rand_U(prod(dim))

    def test_binary_funcs(self):
        from qit.state import fidelity, trace_dist

        fid = fidelity(self.rho1, self.rho2)
        trd = trace_dist(self.rho1, self.rho2)

        # symmetry
        self.assertAlmostEqual(fid, fidelity(self.rho2, self.rho1), delta=tol)
        self.assertAlmostEqual(trd, trace_dist(self.rho2, self.rho1), delta=tol)

        # fidelity with self, distance from self
        self.assertAlmostEqual(fidelity(self.rho1, self.rho1), 1, delta=tol)
        self.assertAlmostEqual(trace_dist(self.rho1, self.rho1), 0, delta=tol)

        # unaffected by unitary transformations
        self.assertAlmostEqual(fid, fidelity(self.rho1.u_propagate(self.U), self.rho2.u_propagate(self.U)), delta=tol)
        self.assertAlmostEqual(trd, trace_dist(self.rho1.u_propagate(self.U), self.rho2.u_propagate(self.U)), delta=tol)

        # for pure states trace_dist and fidelity are equivalent
        self.assertAlmostEqual(trace_dist(self.p1, self.p2) ** 2 +fidelity(self.p1, self.p2) ** 2, 1, delta=tol)

        # for mixed states, these inequalities hold
        self.assertTrue(sqrt(1 -fid ** 2) -trd >= -tol)
        self.assertTrue(1 -fid -trd <= tol)

        # for a pure and a mixed state we get this inequality
        self.assertTrue(1 -fidelity(self.rho1, self.p1) ** 2 -trace_dist(self.rho1, self.p1) <= tol)



if __name__ == '__main__':
    #print(sys.path)
    print('Testing QIT version ' + version())
    unittest.main()
