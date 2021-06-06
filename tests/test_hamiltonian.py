"""
Unit tests for qit.hamiltonian
"""
# Ville Bergholm 2020

import pytest

import numpy as np
import scipy.linalg as spl

import qit
import qit.hamiltonian as ham



I = np.eye(2)
Z = qit.sz


class TestHamiltonian:
    """Testing the Hamiltonian module.
    """

    def test_magnetic_dipole(self, tol):
        """
        """
        dim = (2, 2)
        H = ham.magnetic_dipole(dim, B=(0, 0, 1))
        D = np.prod(dim)
        assert H.shape == (D, D)
        assert spl.norm(2 * H - np.kron(Z, I) - np.kron(I, Z)) == pytest.approx(0, abs=tol)

    def test_heisenberg(self, tol):
        """Heisenberg model.
        """
        dim = (2, 2)
        H = ham.heisenberg(dim, J=(0, 0, 4))
        assert spl.norm(H - np.kron(Z, Z)) == pytest.approx(0, abs=tol)

    def test_jaynes_cummings(self, tol):
        """Jaynes-Cummings model
        """
        m = 10
        n = 3
        om_atom = np.random.rand(n)
        Omega = np.random.rand(n)
        H, dim = ham.jaynes_cummings(om_atom, Omega, m=m, use_RWA=False)
        assert np.all(dim == (m,) + (2,) * n)

    def test_hubbard(self, tol):
        """Hubbard model.
        """
        n = 3
        C = np.eye(n, n, 1)
        U = 0.7
        mu = 0.4
        H, dim = ham.hubbard(C, U, mu)
        assert np.all(dim == (2,) * (2*n))

    def test_bose_hubbard(self, tol):
        """Bose-Hubbard model.
        """
        n = 3
        C = np.eye(n, n, 1)
        U = 0.7
        mu = 0.4
        m = 8
        H, dim = ham.bose_hubbard(C, U, mu, m=m)
        assert np.all(dim == (m,) * n)

    def test_holstein(self, tol):
        """Holstein model.
        """
        n = 3
        C = np.eye(n, n, 1)
        omega = 1.1
        g = 0.6
        m = 6
        H, dim = ham.holstein(C, omega, g, m=m)
        print(dim)
        assert np.all(dim == (2**n,) + (m,) * n)
