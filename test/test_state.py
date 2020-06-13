"""
Unit tests for qit.state
"""
# Ville Bergholm 2008-2020


import pytest

import numpy as np
from numpy.random import rand, randn
from numpy.linalg import norm

import qit
from qit.lmap  import Lmap
from qit.state import State, fidelity, trace_dist
from qit.utils import rand_positive, rand_U, mkron

dim = (2, 5, 3)
bipartitions = [0, [0, 2]]

@pytest.fixture(scope="session")
def rho():
    """Mixed state."""
    return State(rand_positive(np.prod(dim)), dim)

@pytest.fixture(scope="session")
def psi():
    """Pure ket state."""
    return State(0, dim).u_propagate(rand_U(np.prod(dim)))



class TestState:
    def test_constructor(self):
        "Test state.__init__"

        # copy constructor
        temp = Lmap(randn(4, 4), ((4,), (4,)))
        s = State(temp, (2, 2))

        # strings
        s = State('10011')
        s = State('2014', (3, 2, 3, 5))
        s = State('GHZ')
        s = State('GHZ', (3, 2, 3))
        s = State('W', (2, 3, 2))
        s = State('Bell2')

        # basis kets
        s = State(4, 6)
        s = State(11, (3, 5, 2))

        # kets and state ops
        s = State(rand(5))
        s = State(rand(6), (3, 2))
        s = State(rand(3, 3))
        s = State(rand(4, 4), (2, 2))

        # bad inputs
        temp = Lmap(randn(2, 3), ((2,), (3,)))
        with pytest.raises(ValueError, match='State operator must be square.'):
            State(temp)          # nonsquare Lmap
        with pytest.raises(ValueError, match="Unknown named state 'rubbish'"):
            State('rubbish')     # unknown state name
        with pytest.raises(ValueError, match='Need system dimension.'):
            State(0)             # missing dimension
        with pytest.raises(ValueError, match='Invalid basis ket.'):
            State(2, 2)          # ket number too high
        #with pytest.raises(ValueError, match='sss'):
        #    State([])            # bad array dimension (0)
        with pytest.raises(ValueError, match='State must be given as a state vector or a state operator.'):
            State(rand(2, 2, 2)) # bad array dimension (3)
        with pytest.raises(ValueError, match='State operator matrix must be square.'):
            State(randn(3, 4))   # nonsquare array

    def test_named_states(self, tol):
        ps = np.array([0, 0.1, 1])
        dims = np.array([2, 3, 4])
        for d in dims:
            for p in ps:
                s = State.werner(p, d)
                t = State.isotropic((2 * p - 1) / d, d)
                #s.check()
                #t.check()
                #assert s.trace() == pytest.approx(1, abs=tol)
                #assert t.trace() == pytest.approx(1, abs=tol)
                assert (s -t.ptranspose(0)).norm() == pytest.approx(0, abs=tol)


    def test_ev_var(self):
        """Expectation values and variances of observables."""
        s = State(0, (2,))  # |0>
        assert s.ev(qit.sz) == pytest.approx(1)
        assert s.var(qit.sz) == pytest.approx(0)
        assert s.ev(qit.sx) == pytest.approx(0)
        assert s.var(qit.sx) == pytest.approx(1)


    def test_methods(self, rho, tol):
        # TODO concurrence, fix_phase, kraus_propagate, locc_convertible, lognegativity, measure,
        # negativity,

        D = np.prod(dim)


        ### generalized Bloch vectors.

        temp = rho.bloch_vector()
        # round trip
        assert (State.bloch_state(temp) -rho).norm() == pytest.approx(0, abs=tol)
        # correlation tensor is real
        assert norm(temp.imag) == pytest.approx(0, abs=tol)
        # state purity limits the Frobenius norm
        assert norm(temp.flat) -np.sqrt(D) <= tol
        # state normalization
        assert temp.flat[0] == pytest.approx(1, abs=tol)


    def test_entropy(self, rho, psi, tol):

        D = np.prod(dim)
        U = rand_U(np.prod(dim))  # random unitary

        temp = rho.entropy()
        # zero for pure states
        assert psi.entropy() == pytest.approx(0, abs=tol)
        # nonnegative
        assert temp >= -tol
        # upper limit is log2(D)
        assert temp <= np.log2(D) +tol
        # invariant under unitary transformations
        assert temp == pytest.approx(rho.u_propagate(U).entropy(), abs=tol)


    def test_ptrace(self, rho, tol):

        temp = rho.trace()
        for sys in bipartitions:
            rho_A = rho.ptrace(sys)
            # trace of partial trace equals total trace
            assert temp == pytest.approx(rho_A.trace(), abs=tol)
        # partial trace over all subsystems equals total trace
        assert temp == pytest.approx(rho.ptrace(range(rho.subsystems())).trace(), abs=tol)

    def test_ptranspose(self, rho, tol):

        temp = rho.trace()
        for sys in bipartitions:
            rho_T = rho.ptranspose(sys)
            # two ptransposes cancel
            assert (rho -rho_T.ptranspose(sys)).norm() == pytest.approx(0, abs=tol)
            # ptranspose preserves trace
            assert temp == pytest.approx(rho_T.trace(), abs=tol)

    def test_schmidt(self, tol):
        return
        # FIXME svdvals causes a crash in schmidt! see if fixed in scipy 0.13.0.
        for sys in bipartitions:
            lambda1, u, v = psi.schmidt(sys, full=True)
            lambda2 = psi.schmidt(psi.invert_selection(sys))
            # squares of schmidt coefficients sum up to unity
            assert norm(lambda1) == pytest.approx(1, abs=tol)
            # both subdivisions have identical schmidt coefficients
            assert norm(lambda1 -lambda2) == pytest.approx(0, abs=tol)

            # decomposition is equal to the original matrix
            temp = 0
            for k in range(len(lambda1)):
                temp += kron(lambda1[k] * u[:, k], v[:, k])
            assert norm(psi.data.ravel() -temp) == pytest.approx(0, abs=tol)

        # squared schmidt coefficients equal eigenvalues of partial trace
        #r = State(randn(30) + 1j*randn(30), [5, 6]).normalize()
        #x = r.schmidt([0]) ** 2  # FIXME crashes ipython!
        #temp = r.ptrace([1])
        #y, dummy = eighsort(temp.data)
        #assert norm(x-y) == pytest.approx(0, abs=tol)



    def test_reorder(self, tol):

        dim = (2, 5, 1)
        A = rand(dim[0], dim[0])
        B = rand(dim[1], dim[1])
        C = rand(dim[2], dim[2])

        T1 = State(mkron(A, B, C), dim)
        T2 = T1.reorder([2, 0, 1])
        assert norm(mkron(C, A, B) - T2.data) == pytest.approx(0, abs=tol)
        T2 = T1.reorder([1, 0, 2])
        assert norm(mkron(B, A, C) - T2.data) == pytest.approx(0, abs=tol)
        T2 = T1.reorder([2, 1, 0])
        assert norm(mkron(C, B, A) - T2.data) == pytest.approx(0, abs=tol)


class TestStateBinaryFuncs:
    def test_distance_funcs(self, tol):

        dim = (2, 3)
        # two mixed states
        rho1 = State(rand_positive(np.prod(dim)), dim)
        rho2 = State(rand_positive(np.prod(dim)), dim)
        # two pure states
        p = State(0, dim)
        self.p1 = p.u_propagate(rand_U(np.prod(dim)))
        self.p2 = p.u_propagate(rand_U(np.prod(dim)))
        # random unitary
        U = rand_U(np.prod(dim))

        fid = fidelity(rho1, rho2)
        trd = trace_dist(rho1, rho2)

        # symmetry
        assert fid == pytest.approx(fidelity(rho2, rho1), abs=tol)
        assert trd == pytest.approx(trace_dist(rho2, rho1), abs=tol)

        # fidelity with self, distance from self
        assert fidelity(rho1, rho1) == pytest.approx(1, abs=tol)
        assert trace_dist(rho1, rho1) == pytest.approx(0, abs=tol)

        # unaffected by unitary transformations
        assert fid == pytest.approx(fidelity(rho1.u_propagate(U), rho2.u_propagate(U)), abs=tol)
        assert trd == pytest.approx(trace_dist(rho1.u_propagate(U), rho2.u_propagate(U)), abs=tol)

        # for pure states trace_dist and fidelity are equivalent
        assert trace_dist(self.p1, self.p2) ** 2 +fidelity(self.p1, self.p2) ** 2 == pytest.approx(1, abs=tol)

        # for mixed states, these inequalities hold
        assert np.sqrt(1 -fid ** 2) -trd >= -tol
        assert 1 -fid -trd <= tol

        # for a pure and a mixed state we get this inequality
        assert 1 -fidelity(rho1, self.p1) ** 2 -trace_dist(rho1, self.p1) <= tol
