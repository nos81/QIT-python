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
"""
# Ville Bergholm 2014

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np
from numpy import asarray, transpose
from numpy.linalg import det, eigvals
from scipy.linalg import norm

from .base import *
from .lmap import *
from .utils import angular_momentum, op_list


__all__ = [
    'heisenberg'
    ]



def _cdot(v, A):
    """Real dot product of a vector and a tuple of operators."""
    res = 0j
    for vv, AA in zip(v, A):
        res += vv * AA
    return res


def heisenberg(dim, C=None, J=(0,0,2), B=(0,0,1)):
    r"""Heisenberg spin network Hamiltonian.

    Returns the Hamiltonian H for the Heisenberg model, describing a network
    of n interacting spins in an external magnetic field.

    dim is an n-tuple of the dimensions of the spins, i.e. dim == (2, 2, 2)
    would be a system of three spin-1/2's.

    C is the :math:`n \times n` connection matrix of the spin network, where C[a,b]
    is the coupling strength between spins a and b. Only the upper triangle is used.

    J defines the form of the spin-spin interaction. It is either a 3-tuple or a
    function J(a, b) returning a 3-tuple for site-dependent interactions.
    Element k of the tuple is the coefficient of the Hamiltonian term S_ka * S_kb,
    where S_ka is the k-component of the angular momentum of spin a.

    B defines the effective magnetic field the spins locally couple to. It's either
    a 3-tuple (homogeneous field) or a function B(a) that returns a 3-tuple for
    site-dependent field.

    .. math::

      H = \sum_{a,b} \sum_{k = x,y,z} J(a,b)[k] S_k^{(a)} S_k^{(b)}  +\sum_a \vec{B}(a) \cdot \vec{S}_{(a)})

    Examples::

      C = np.eye(n, n, 1)  linear n-spin chain
      J = (2, 2, 2)        isotropic Heisenberg coupling
      J = (2, 2, 0)        XX+YY coupling
      J = (0, 0, 2)        Ising ZZ coupling
      B = (0, 0, 1)        homogeneous Z-aligned field
    """
    # Ville Bergholm 2009-2014

    n = len(dim) # number of spins in the network

    if C == None:
        # linear chain
        C = np.eye(n, n, 1)

    # make J and B into functions
    if isinstance(J, tuple):
        if len(J) != 3:
            raise ValueError('J must be either a 3-tuple or a function.')
        J = asarray(J)
        Jf = lambda a,b: C[a,b] * J
    else:
        Jf = J

    if isinstance(B, tuple):
        if len(B) != 3:
            raise ValueError('B must be either a 3-tuple or a function.')
        Bf = lambda a: B
    else:
        Bf = B

    H = 0j #sparse(0j)

    # spin-spin couplings: loop over nonzero entries of C
    # only use the upper triangle
    C = np.triu(C)
    for a,b in transpose(C.nonzero()):
        # spin ops for sites a and b
        Sa = angular_momentum(dim[a])
        Sb = angular_momentum(dim[b])
        temp = []
        # coupling between sites a and b
        c = Jf(a,b)
        for k in range(3):
            temp.append([(c[k] * Sa[k], a), (Sb[k], b)])
        H += op_list(temp, dim)

    # local magnetic field terms
    temp = []
    for a in range(n):
        A = angular_momentum(dim[a])  # spin ops
        temp.append([(_cdot(Bf(a), A), a)])

    H += op_list(temp, dim)
    return H



def test():
    """Test script for Hamiltonian methods."""
    from .utils import assert_o

    # TODO tests
    #assert_o(norm(), 0, tol)
