# -*- coding: utf-8 -*-
"""
Model Hamiltonians (:mod:`qit.hamiltonian`)
===========================================

This module has methods that generate several common types of model Hamiltonians used in quantum mechanics.


.. currentmodule:: qit.hamiltonian

Contents
--------

.. autosummary::

   heisenberg
   jaynes_cummings
   hubbard
   bose_hubbard
   holstein
"""
# Ville Bergholm 2014-2016

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np
from numpy import asarray, conj, transpose, dot

from .base import sx, sy, sz
from .utils import (angular_momentum, op_list, boson_ladder, fermion_ladder, cdot)


__all__ = [
    'heisenberg',
    'jaynes_cummings',
    'hubbard',
    'bose_hubbard',
    'holstein']



def magnetic_dipole(dim, B=(0, 0, 1)):
    r"""Local magnetic dipole Hamiltonian.

    .. math::

      H = \sum_i \vec{B}(i) \cdot \vec{S}^{(i)},

    where :math:`S_k^{(i)}` is the k-component of the angular momentum operator of spin i.

    Args:
      dim (tuple): dimensions of the spins, i.e. dim == (2, 2, 2) would be a system of three spin-1/2's.
      B   (tuple): The effective magnetic field the spins locally couple to. Either
                   a 3-tuple (homogeneous field) or a function B(a) that returns a 3-tuple for a site-dependent field.
    Returns:
      array: the Hamiltonian
    """
    if not callable(B):
        if len(B) != 3:
            raise ValueError('B must be either a 3-tuple or a function.')
        Bf = lambda i: B
    else:
        Bf = B

    # local magnetic field terms
    temp = []
    for i in range(len(dim)):
        A = angular_momentum(dim[i])  # spin ops
        temp.append([(cdot(Bf(i), A), i)])
    H = op_list(temp, dim)
    return H


def heisenberg(dim, J=(0, 0, 2), C=None):
    r"""Heisenberg spin network model.

    Args:
      dim (tuple): dimensions of the spins, i.e. dim == (2, 2, 2) would be a system of three spin-1/2's.
      J   (tuple): The form of the spin-spin interaction. Either a 3-tuple or a
                   function J(s, i, j) returning the coefficient of the Hamiltonian term S_s^{(i)} * S_s^{(b)}.
      C   (array): Optional connection matrix of the spin network, where C[i,j]
                   is the coupling strength between spins i and j. Only the upper triangle is used.
    Returns:
      array: the Hamiltonian

    Builds a Heisenberg model Hamiltonian, describing a network of n interacting spins.

    .. math::

      H = \sum_{\langle i,j \rangle} C[i,j] \sum_{k = x,y,z} J(i,j)[k] S_k^{(i)} S_k^{(j)},

    where :math:`S_k^{(i)}` is the k-component of the angular momentum operator of spin i.

    Examples::

      C = np.eye(n, n, 1)  linear n-spin chain
      J = (2, 2, 2)        isotropic Heisenberg coupling
      J = (2, 2, 0)        XX+YY coupling
      J = (0, 0, 2)        Ising ZZ coupling
    """
    # Ville Bergholm 2009-2017

    n = len(dim) # number of spins in the network
    # make J into a function
    if not callable(J):
        if len(J) != 3:
            raise ValueError('J must be either a 3-tuple or a function.')

        if C is None:
            # default: linear chain
            C = np.eye(n, n, 1)

        J = asarray(J)
        Jf = lambda s, i, j: J[s] * C[i, j]
    else:
        Jf = J

    # spin-spin couplings:
    temp = []
    if C is None:
        # loop over the entire upper triangle
        for i in range(n):
            A = angular_momentum(dim[i])  # spin ops for first site
            for j in range(i+1, n):
                B = angular_momentum(dim[j])  # and the second
                for s in range(3):
                    temp.append([(Jf(s, i, j) * A[s], i), (B[s], j)])
    else:
        # loop over nonzero entries of C, only use the upper triangle
        C = np.triu(C)
        for i, j in transpose(C.nonzero()):
            # spin ops for sites i and j
            A = angular_momentum(dim[i])
            B = angular_momentum(dim[j])
            for s in range(3):
                temp.append([(Jf(s, i, j) * A[s], i), (B[s], j)])

    H = op_list(temp, dim)
    return H



def jaynes_cummings(om_atom, Omega, m=10, use_RWA=False):
    r"""Jaynes-Cummings model, one or more two-level atoms coupled to a single-mode cavity.

    :param vector om_atom:  Atom level splittings
    :param vector Omega:       Atom-cavity coupling
    :param int m:           Cavity Hilbert space dimension truncation limit
    :param bool use_RWA:    Should we discard counter-rotating interaction terms?
    :returns tuple (H, dim):  H is the Hamiltonian H and dim the dimension vector for an
      implementation of the Jaynes-Cummings model, describing n two-level atoms coupled
      to a harmonic oscillator (e.g. a single EM field mode in an optical cavity),
      where n == len(om_atom) == len(Omega).

    .. math::
      H/\hbar = -\sum_k \frac{{\omega_a}_k}{2} Z_k +\omega_c a^\dagger a +\sum_k \frac{\Omega_k}{2} (a+a^\dagger) \otimes X_k

    The returned Hamiltonian H has been additionally normalized with :math:`\omega_c`,
    and is thus dimensionless. om_atom[k] = :math:`{\omega_a}_k / \omega_c`,  Omega[k] = :math:`\Omega_k / \omega_c`.

    The order of the subsystems is [cavity, atom_1, ..., atom_n].
    The dimension of the Hilbert space of the bosonic cavity mode (infinite in principle) is truncated to m.
    If use_RWA is true, the Rotating Wave Approximation is applied to the Hamiltonian,
    and the counter-rotating interaction terms are discarded.
    """
    # Ville Bergholm 2014-2016

    n = len(om_atom)
    if len(Omega) != n:
        raise ValueError('The coupling vector Omega must be of the same length as the atom splitting vector om_atom.')

    # dimension vector
    dim = (m,) + (2,)*n
    # operators
    a = boson_ladder(m)
    ax = a.conj().transpose()
    x = a+ax
    sp = 0.5*(sx -1j*sy) # qubit raising operator
    sm = sp.conj().transpose()

    atom = []
    coupling = []
    # loop over atoms
    for k in range(n):
        atom.append([(-0.5*om_atom[k] * sz, k+1)])  # atomic Hamiltonian
        # atom-cavity coupling
        if use_RWA:
            # rotating wave approximation, discard counter-rotating terms
            coupling.append([(a,   0), (0.5*Omega[k] * sp, k+1)])
            coupling.append([(ax,  0), (0.5*Omega[k] * sm, k+1)])
        else:
            coupling.append([(x, 0),   (0.5*Omega[k] * sx, k+1)])

    # cavity
    import pdb; pdb.set_trace()
    Hc    = op_list([[(dot(ax, a), 0)]], dim)
    Ha    = op_list(atom, dim)
    H_int = op_list(coupling, dim)
    H = Hc +Ha +H_int
    return H, dim



def hubbard(C, U=1, mu=0):
    r"""Hubbard model, fermions on a lattice.

    :param array C:    Connection matrix of the interaction graph
    :param float U:    Fermion-fermion interaction strength (normalized)
    :param float mu:   External chemical potential (normalized)
    :returns tuple (H, dim):

    Returns the Hamiltonian H and the dimension vector dim for an
    implementation of the Hubbard model.

    The model consists of spin-1/2 fermions confined in a graph defined by the
    symmetric connection matrix C (only upper triangle is used).
    The fermions interact with other fermions at the same site with interaction strength U,
    as well as with an external chemical potential mu.
    The Hamiltonian has been normalized by the fermion hopping constant t.

    .. math::

      H = -\sum_{\langle i,j \rangle, \sigma} c^\dagger_{i,\sigma} c_{j,\sigma}
        +\frac{U}{t} \sum_i n_{i,up} n_{i,\downarrow} -\frac{\mu}{t} \sum_i (n_{i,up}+n_{i,\downarrow})
    """
    # Ville Bergholm 2010-2016

    n = C.shape[0]
    dim = 2 * np.ones(2*n)  # n sites, two fermionic modes per site

    # fermion annihilation ops f[site, spin]
    f = fermion_ladder(2 * n).reshape((n, 2))
    # NOTE all the f ops have the full Hilbert space dimension

    H = 0j

    for k in range(n):
        # number operators for this site
        n1 = dot(f[k, 0].conj().transpose(), f[k, 0])
        n2 = dot(f[k, 1].conj().transpose(), f[k, 1])
        # on-site interaction
        H += U * dot(n1, n2)
        # chemical potential
        H += -mu * (n1 + n2)

    # fermions hopping: loop over nonzero entries of C
    # only use the upper triangle
    C = np.triu(C)
    for i, j in transpose(C.nonzero()):
        for s in range(2):
            H -= dot(f[i, s].conj().transpose(), f[j, s]) +dot(f[j, s].conj().transpose(), f[i, s])

    return H, dim



def bose_hubbard(C, U=1, mu=0, m=10):
    r"""Bose-Hubbard model, bosons on a lattice.

    Returns the Hamiltonian H and the dimension vector dim for an
    implementation of the Bose-Hubbard model.

    The model consists of spinless bosons confined in a graph defined by the
    symmetric connection matrix C (only upper triangle is used).
    The bosons interact with other bosons at the same site with interaction strength U,
    as well as with an external chemical potential mu.
    The Hamiltonian has been normalized by the boson hopping constant t.

    .. math::

      H = -\sum_{\langle i,j \rangle} b^\dagger_i b_{j} +\frac{U}{2t} \sum_i n_i (n_i-1) -\frac{\mu}{t} \sum_i n_i

    The dimensions of the boson Hilbert spaces (infinite in principle) are truncated to m.
    """
    # Ville Bergholm 2010-2014

    n = len(C)
    dim = m * np.ones(n)

    b = boson_ladder(m)  # boson annihilation op
    b_dagger = b.conj().transpose()  # boson creation op
    nb = dot(b_dagger, b)  # boson number op

    I = np.eye(m)
    A = U/2 * dot(nb, (nb-I)) # on-site interaction
    B = -mu * nb # chemical potential

    temp = []
    for k in range(n):
        temp.append([(A+B, k)])
    H = op_list(temp, dim)

    temp = []
    # bosons hopping: loop over nonzero entries of C
    # only use the upper triangle
    C = np.triu(C)
    for i, j in transpose(C.nonzero()):
        temp.extend([[(b_dagger, i), (b, j)], [(b, i), (b_dagger, j)]])

    H -= op_list(temp, dim)
    return H, dim



def holstein(C, omega=1, g=1, m=10):
    r"""Holstein model, electrons on a lattice coupled to phonons.

    Returns the Hamiltonian H and the dimension vector dim for an implementation of the Holstein model.

    The model consists of spinless electrons confined in a graph defined by the
    symmetric connection matrix C (only upper triangle is used),
    coupled to phonon modes represented by a harmonic oscillator at each site.
    The dimensions of phonon Hilbert spaces (infinite in principle) are truncated to m.

    The order of the subsystems is [e1, ..., en, p1, ..., pn].
    The Hamiltonian has been normalized by the electron hopping constant t.

    .. math::

      H = -\sum_{\langle i,j \rangle} c_i^\dagger c_j  +\frac{\omega}{t} \sum_i b_i^\dagger b_i
        -\frac{g \omega}{t} \sum_i (b_i + b_i^\dagger) c_i^\dagger c_i
    """
    # Ville Bergholm 2010-2014

    n = len(C)
    # Hilbert space: electrons first, then phonons
    dim = np.r_[2**n, m * np.ones(n)]  # Jordan-Wigner clumps all fermion dims together

    c = fermion_ladder(n)  # electron annihilation ops
    b = boson_ladder(m)    # phonon annihilation
    b_dagger = b.conj().transpose()  # phonon creation
    q = b + b_dagger       # phonon position
    nb = dot(b_dagger, b)  # phonon number operator

    temp = []
    for k in range(n):
        # phonon harmonic oscillators
        temp.append([(omega * nb, 1+k)])
        # electron-phonon interaction
        temp.append([(-g * omega * dot(c[k].conj().transpose(), c[k]), 0), (q, 1+k)])
    H = op_list(temp, dim)

    # fermions hopping: loop over nonzero entries of C
    # only use the upper triangle
    C = np.triu(C)
    T = 0j
    for i, j in transpose(C.nonzero()):
        T += dot(c[i].conj().transpose(), c[j]) +dot(c[j].conj().transpose(), c[i])
    H += op_list([[(-T, 0)]], dim)

    # actual dimensions
    #dim = [2*ones(1, n), m*ones(1, n)]
    return H, dim
