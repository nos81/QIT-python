"""
Unit tests for qit.utils
"""
# Ville Bergholm 2009-2020

import pytest

import numpy as np
import scipy.linalg as spl

import qit
from qit.utils import *
from qit.gate  import copydot


@pytest.fixture(scope="session")
def dim():
    """Vector dimension."""
    return 5


def randn_complex(*arg):
    "Returns an array of random complex numbers, normally distributed."
    return np.random.randn(*arg) +1j*np.random.randn(*arg)

def assertHermitian(H, tol):
    "Make sure H is hermitian."
    assert H == pytest.approx(H.T.conj(), abs=tol)

def assertUnitary(U, tol):
    "Make sure U is unitary."
    I = np.eye(len(U))
    temp = U.T.conj()
    assert np.linalg.norm(U @ temp -I) == pytest.approx(0, abs=tol)
    assert np.linalg.norm(temp @ U -I) == pytest.approx(0, abs=tol)


class TestUtils:
    def test_su2_rotations(self, tol):
        theta = 1.3372
        assert Rx(theta) == pytest.approx(spl.expm(-1j * theta/2 * qit.sx), abs=tol)
        assert Ry(theta) == pytest.approx(spl.expm(-1j * theta/2 * qit.sy), abs=tol)
        assert Rz(theta) == pytest.approx(spl.expm(-1j * theta/2 * qit.sz), abs=tol)


    def test_expv(self, tol):
        dim = 10
        v = randn_complex(dim)
        tol = 1e2 * tol  # use a larger tolerance here

        ## arbitrary matrix with Arnoldi iteration
        A = randn_complex(dim, dim)
        res = spl.expm(1*A) @ v

        w, err, hump = expv(1, A, v, m = dim // 2)
        assert np.linalg.norm(w - res) == pytest.approx(0, abs=tol)

        # force a happy breakdown
        w, err, hump = expv(1, A, v, m = dim)
        assert np.linalg.norm(w - res) == pytest.approx(0, abs=tol)

        ## FIXME why does Lanczos work with nonhermitian matrices?
        w, err, hump = expv(1, A, v, m = dim // 2, iteration='lanczos')
        assert np.linalg.norm(w - res) == pytest.approx(0, abs=tol)

        # force a happy breakdown
        w, err, hump = expv(1, A, v, m = dim, iteration = 'lanczos')
        assert np.linalg.norm(w - res) == pytest.approx(0, abs=tol)


        ## Hermitian matrix with Lanczos iteration
        H = rand_hermitian(dim)
        res = spl.expm(1*H) @ v

        w, err, hump = expv(1, H, v, m = dim // 2, iteration='lanczos')
        assert np.linalg.norm(w - res) == pytest.approx(0, abs=tol)

        # force a happy breakdown
        w, err, hump = expv(1, H, v, m = dim, iteration = 'lanczos')
        assert np.linalg.norm(w - res) == pytest.approx(0, abs=tol)


    def test_random_matrices(self, tol, dim):

        ### random matrices
        H = rand_hermitian(dim)
        assert H.shape == (dim, dim)
        assertHermitian(H, tol=tol)

        U = rand_U(dim)
        assert U.shape == (dim, dim)
        assertUnitary(U, tol=tol)

        U = rand_SU(dim)
        assert U.shape == (dim, dim)
        assertUnitary(U, tol=tol)
        assert np.linalg.det(U) == pytest.approx(1, abs=tol)

        rho = rand_positive(dim)
        assert rho.shape == (dim, dim)
        assertHermitian(rho, tol=tol)
        assert np.trace(rho) == pytest.approx(1, abs=tol)
        temp = np.linalg.eigvalsh(rho)
        assert np.linalg.norm(temp.imag) == pytest.approx(0, abs=tol) # real eigenvalues
        assert np.linalg.norm(temp - abs(temp)) == pytest.approx(0, abs=tol) # nonnegative eigenvalues

        G = rand_GL(dim)
        assert G.shape == (dim, dim)

        A = rand_SL(dim)
        assert A.shape == (dim, dim)
        assert np.linalg.det(A) == pytest.approx(1, abs=tol)

    def test_superops(self, tol, dim):

        L = rand_U(dim)
        R = rand_U(dim)
        rho = rand_positive(dim)
        v = vec(rho)

        assert np.linalg.norm(rho -inv_vec(v)) == pytest.approx(0, abs=tol)
        assert np.linalg.norm(L @ rho @ R -inv_vec(lrmul(L, R) @ v)) == pytest.approx(0, abs=tol)
        assert np.linalg.norm(L @ rho -inv_vec(lmul(L) @ v)) == pytest.approx(0, abs=tol)
        assert np.linalg.norm(rho @ R -inv_vec(rmul(R) @ v)) == pytest.approx(0, abs=tol)

        # superop propagators and Choi matrices are equivalent ways of propagating a state
        A = [randn_complex(dim, dim), randn_complex(dim, dim)]
        L = spl.expm(superop_lindblad(A))
        C = superop_to_choi(L)
        wire = np.kron(copydot(0, 2, dim).data.A, np.eye(dim))
        temp = (wire.conj().T @ np.kron(rho, C)) @ wire
        assert np.linalg.norm(temp -inv_vec(L @ v)) == pytest.approx(0, abs=tol)

    def test_angular_momentum(self, tol, dim):
        J = angular_momentum(dim)
        assert len(J) == 3
        for A in J:
            assertHermitian(A, tol=tol)
        assert np.linalg.norm(comm(J[0], J[1]) - 1j * J[2]) == pytest.approx(0, abs=tol)  # [Jx, Jy] == i Jz

    def test_boson_ladder(self, tol, dim):

        a = boson_ladder(dim)
        temp = comm(a, a.T.conj())
        assert np.linalg.norm(temp[:-1, :-1] -np.eye(dim-1)) == pytest.approx(0, abs=tol)  # [a, a'] == I  (truncated, so skip the last row/col!)

        n = a.T.conj() @ a
        assert np.diag(n) == pytest.approx(np.arange(dim), abs=tol)

    def test_fermion_ladder(self, tol):

        fff = fermion_ladder(3)
        # {f_j, f_k} = 0
        # {f_j, f_k^\dagger} = I \delta_{jk}
        for f in fff:
            fp = f.conj().transpose()
            assert np.linalg.norm(acomm(f, f)) == pytest.approx(0, abs=tol)
            assert np.linalg.norm(acomm(f, fp) -np.eye(8)) == pytest.approx(0, abs=tol)
            for j in fff:
                if not f is j:
                    assert np.linalg.norm(acomm(j, f)) == pytest.approx(0, abs=tol)
                    assert np.linalg.norm(acomm(j, fp)) == pytest.approx(0, abs=tol)


    def test_spectral_decomposition(self, tol, dim):

        H = rand_hermitian(dim)
        E, P = spectral_decomposition(H)
        temp = 0
        for k in range(len(E)):
            temp += E[k] * P[k]
        assert np.linalg.norm(temp -H) == pytest.approx(0, abs=tol)


        # tensor bases


        # majorization

        # op_list

        # plots
