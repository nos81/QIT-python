# -*- coding: utf-8 -*-
"""
Local invariants (:mod:`qit.invariant`)
=======================================

This module contains tools for computing and plotting the values of
various local gate and state invariants.

.. currentmodule:: qit.invariant

Contents
--------

# TODO these function names are terrible

.. autosummary::
   LU
   canonical
   makhlin
   max_concurrence
   plot_makhlin_2q
   plot_weyl_2q
"""
# Ville Bergholm 2011

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

from .base import sy, Q_Bell
from .lmap import Lmap, tensor
from .utils import tensorbasis, majorize



def LU(rho, k, perms):
    r"""Local unitary polynomial invariants of quantum states.

    Computes the permutation invariant :math:`I_{k; \pi_1, \pi_2, \ldots, \pi_n}` for the state :math:`\rho`,
    defined as :math:`\trace(\rho^{\otimes k} \Pi)`, where :math:`\Pi` permutes all k copies of the i:th subsystem using :math:`\pi_i`.

    Args:
      rho (state): quantum state with n subsystems
      k (int): order of the invariant, ``k >= 1``
      perms (Sequence[Sequence[int]]): Permutations. ``len(perms) == n``, each element must be
          a full k-permutation (or an empty sequence denoting the identity permutation).

    Returns:
      complex: invariant

    Example: :math:`I_{3; (123),(12)}(\rho)` = LU_inv(rho, 3, [(1, 2, 0), (1, 0, 2)])

    This function can be very inefficient for some invariants, since
    it does no partial traces etc. which might simplify the calculation.

    Uses the algorithm in :cite:`BBL2012`.
    """
    # Ville Bergholm 2011-2016

    n = len(perms)
    if n != rho.subsystems():
        raise ValueError('Need one permutation per subsystem.')

    # convert () to identity permutation
    id_perm = np.arange(k)

    # idea: tensor k copies of rho together, then permute all the copies of the i:th subsystem using perms[i] on one side.
    # cols of r correspond to subsystems, rows to copies. the initial order is
    r = np.arange(n * k).reshape((k, n))
    for j, p in enumerate(perms):
        if len(p) == 0:
            p = id_perm
        elif len(p) != k:
            raise ValueError('Permutation #{0} does not have {1} elements.'.format(j, k))
        r[:,j] = r[np.asarray(p),j]  # apply the jth permutation
    r = r.flatten()

    # TODO this could be done much more efficiently
    temp = Lmap(rho.to_op().tensorpow(k))
    return temp.reorder((r, None)).trace()



def canonical(U):
    """Canonical local invariants of a two-qubit gate.

    Computes a vector of three real canonical local invariants for the U(4) matrix U, normalized to the range [0,1].

    Args:
      U (array[complex]): U(4) matrix

    Returns:
      array[float]: canonical local invariants of U

    Uses the algorithm in :cite:`Childs`.
    """
    # Ville Bergholm 2004-2010

    sigma = np.kron(sy, sy)
    U_flip = (sigma @ U.transpose()) @ sigma  # spin flipped U
    temp = (U @ U_flip) / np.sqrt(complex(npl.det(U)))

    Lambda = npl.eigvals(temp) #[exp(i*2*phi_1), etc]
    # logarithm to the branch (-1/2, 3/2]
    Lambda = np.angle(Lambda) / np.pi # divide pi away
    for k in range(len(Lambda)):
        if Lambda[k] <= -0.5:
            Lambda[k] += 2
    S = Lambda / 2
    S = np.sort(S)[::-1]  # descending order

    n = int(round(sum(S)))  # sum(S) must be an integer
    # take away extra translations-by-pi
    S -= np.r_[np.ones(n), np.zeros(4-n)]
    # put the elements in the correct order
    S = np.roll(S, -n)

    M = [[1, 1, 0], [1, 0, 1], [0, 1, 1]] # scaled by factor 2
    c = M @ S[:3]
    # now 0.5 >= c[0] >= c[1] >= |c[2]|
    # and into the Berkeley chamber using a translation and two Weyl reflections
    if c[2] < 0:
        c[0] = 1 - c[0]
        c[2] = -c[2]
    c = np.mod(c, 1)
    return c


def makhlin(U):
    """Makhlin local invariants of a two-qubit gate.

    Computes a vector of the three real Makhlin invariants (see :cite:`Makhlin`) corresponding
    to the U(4) gate U.
    Alternatively, given a vector of canonical invariants normalized to [0, 1],
    returns the corresponding Makhlin invariants (see :cite:`Zhang`).

    Args:
      U (array[complex]): U(4) matrix

    Returns:
      array[float]: Makhlin local invariants of U

    Alternatively, U may be given in terms of a vector of three
    canonical local invariants.
    """
    # Ville Bergholm 2004-2010
    if U.shape[-1] == 3:
        c = U
        # array consisting of vectors of canonical invariants
        c *= np.pi
        g = np.empty(c.shape)

        g[..., 0] = (np.cos(c[..., 0]) * np.cos(c[..., 1]) * np.cos(c[..., 2])) ** 2 -(np.sin(c[..., 0]) * np.sin(c[..., 1]) * np.sin(c[..., 2])) ** 2
        g[..., 1] = 0.25 * np.sin(2 * c[..., 0]) * np.sin(2 * c[..., 1]) * np.sin(2 * c[..., 2])
        g[..., 2] = 4 * g[..., 0] - np.cos(2 * c[..., 0]) * np.cos(2 * c[..., 1]) * np.cos(2*c[..., 2])
    else:
        # U(4) gate matrix
        V = Q_Bell.conj().transpose() @ (U @ Q_Bell)
        M = V.transpose() @ V

        t1 = np.trace(M) ** 2
        t2 = t1 / (16 * npl.det(U))
        g = np.array([t2.real, t2.imag, ((t1 -np.trace(M @ M)) / (4 * npl.det(U))).real])
    return g


def max_concurrence(U):
    """Maximum concurrence generated by a two-qubit gate.

    Returns the maximum concurrence generated by the two-qubit
    gate U (see :cite:`Kraus`), starting from a tensor state.

    Args:
      U (array[complex]): U(4) matrix

    Returns:
      float: maximum concurrence generated by U

    Alternatively, U may be given in terms of a vector of three
    canonical local invariants.
    """
    # Ville Bergholm 2006-2010
    if U.shape[-1] == 4:
        # gate into corresponding invariants
        c = canonical(U)
    else:
        c = U
    temp = np.roll(c, 1, axis=-1)
    return np.max(abs(np.sin(np.pi * np.concatenate((c -temp, c +temp), axis=-1))), axis=-1)


def gate_adjoint_rep(U, dim, only_local=True):
    """Adjoint representation of a unitary gate in the hermitian tensor basis.

    Args:
      U  (array[complex]): unitary gate
      dim (Sequence[int]): dimension vector defining the basis
      only_local (bool): if True, only return the local part of the matrix

    Returns:
      array[float]: adjoint representation of U

    See :cite:`koponen2006`.
    """
    D = len(U)
    if D != np.prod(dim):
        raise ValueError('Dimension of the gate {} does nor match the dimension vector {}.'.format(D, dim))
    # generate the local part of \hat{U}
    B = tensorbasis(dim, d=None, get_locality=False, only_local=only_local)
    n = len(B)
    W = np.empty((n, n), dtype=float)
    for j, y in enumerate(B):
        temp = (U @ y) @ U.T.conj()
        for i, x in enumerate(B):
            # elements of B are hermitian
            W[i, j] = np.trace(x @ temp).real
    return W


def gate_leakage(U, dim, Z=None, W=None):
    """Local degrees of freedom leaked by a unitary gate.

    Args:
      U  (array[complex]): unitary gate
      dim (Sequence[int]): dimension vector
    Returns:
      array[float]: cosines of the principal angles between

    TODO FIXME

    See :cite:`koponen2006`.
    """
    #import pdb; pdb.set_trace()
    # generate the local part of \hat{U}
    ULL = gate_adjoint_rep(U, dim, only_local=True)
    M = ULL
    #M = W.T @ ULL @ Z
    u, s, vh = npl.svd(M, full_matrices=False)
    #_, ref, __ = npl.svd(ULL, full_matrices=False)
    #print(s, ref)
    #print(majorize(s, ref[:len(s)]))
    return s, u, vh


def plot_makhlin_2q(sdiv=31, tdiv=31):
    """Plots the set of two-qubit gates in the space of Makhlin invariants.

    Plots the set of two-qubit gates in the space of Makhlin
    invariants (see :func:`makhlin`), returns the Axes3D object.

    Args:
      sdiv, tdiv (int): number of s and t divisions in the mesh
    Returns:
      Axes3D: plot axes
    """
    # Ville Bergholm 2006-2011

    import matplotlib.cm as cm
    import matplotlib.colors as colors

    s = np.linspace(0, np.pi,   sdiv)
    t = np.linspace(0, np.pi/2, tdiv)

    # more efficient than meshgrid
    #g1 = kron(np.cos(s).^2, np.cos(t).^4) - kron(np.sin(s).^2, np.sin(t).^4)
    #g2 = 0.25*kron(np.sin(2*s), np.sin(2*t).^2)
    #g3 = 4*g1 - kron(np.cos(2*s), np.cos(2*t).^2)
    #S = kron(s, ones(size(t)))
    #T = kron(ones(size(s)), t)

    # canonical coordinate plane (s, t, t) gives the entire surface of the set of gate equivalence classes
    S, T = np.meshgrid(s, t)
    c = np.c_[S.ravel(), T.ravel(), T.ravel()]
    G = makhlin(c).reshape(sdiv, tdiv, 3)
    C = max_concurrence(c).reshape(sdiv, tdiv)

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')

    # mesh, waterfall?
    polyc = ax.plot_surface(G[:, :, 0], G[:, :, 1], G[:, :, 2], rstride = 1, cstride = 1,
        cmap = cm.jet, norm = colors.Normalize(vmin=0, vmax=1, clip=True), alpha = 0.6)
    polyc.set_array(C.ravel() ** 2)  # FIXME colors
    ax.axis('equal')
    #ax.axis([-1, 1, -0.5, 0.5, -3, 3])
    #ax.shading('interp')

    ax.set_xlabel('$g_1$')
    ax.set_ylabel('$g_2$')
    ax.set_zlabel('$g_3$')
    plt.title('Makhlin stingray')

    # labels
    ax.text(1.05, 0, 2.7, 'I')
    ax.text(-1.05, 0, -2.7, 'SWAP')
    ax.text(-0.1, 0, 1.2, 'CNOT')
    ax.text(0.1, 0, -1.2, 'DCNOT')
    ax.text(0.1, 0.26, 0, 'SWAP$^{1/2}$')
    ax.text(0, -0.26, 0, 'SWAP$^{-1/2}$')

    fig.colorbar(polyc, ax = ax)
    plt.show()
    return ax


def plot_weyl_2q(ax=None):
    """Plots the two-qubit Weyl chamber.

    Plots the Weyl chamber for the local invariants
    of 2q gates. See :cite:`Zhang`.

    Returns the Axes3D object.
    """
    # Ville Bergholm 2005-2012

    if ax is None:
        ax = plt.subplot(111, projection='3d')
    ax.hold(True)
    ax.plot_surface(np.array([[0, 0.5, 1], [0, 0.5, 1]]), np.array([[0, 0, 0], [0, 0.5, 0]]), np.array([[0, 0, 0], [0, 0.5, 0]]), alpha=0.2)
    ax.plot_surface(np.array([[0, 0.5], [0, 0.5]]), np.array([[0, 0.5], [0, 0.5]]), np.array([[0, 0], [0, 0.5]]), alpha=0.2)
    ax.plot_surface(np.array([[0.5, 1], [0.5, 1]]), np.array([[0.5, 0], [0.5, 0]]), np.array([[0, 0], [0.5, 0]]), alpha=0.2)
    #axis([0 1 0 0.5 0 0.5])
    ax.axis('equal')
    ax.set_xlabel('$c_1/\\pi$')
    ax.set_ylabel('$c_2/\\pi$')
    ax.set_zlabel('$c_3/\\pi$')
    plt.title('Two-qubit Weyl chamber')

    ax.text(-0.05, -0.05, 0, 'I')
    ax.text(1.05, -0.05, 0, 'I')
    ax.text(0.45, 0.55, 0.55, 'SWAP')
    ax.text(0.45, -0.05, 0, 'CNOT')
    ax.text(0.45, 0.55, -0.05, 'DCNOT')
    ax.text(0.20, 0.25, 0, 'SWAP$^{1/2}$')
    ax.text(0.75, 0.25, 0, 'SWAP$^{-1/2}$')
    return ax
