# -*- coding: utf-8 -*-
"""
Quantum gates (:mod:`qit.gate`)
===============================

This module implements many common types of quantum gates (and some
other useful linear maps). The returned gates are represented as :class:`lmap` instances.
The data type is float unless complex entries are actually needed.

.. currentmodule:: qit.gate


Utilities
---------

.. autosummary::

   single
   two
   controlled
   dist

Unitary gates
-------------

.. autosummary::

   id
   phase
   qft
   walsh
   mod_inc
   mod_mul
   mod_add
   swap

Linear maps
-----------

.. autosummary::

   copydot
   plusdot
   epsilon
"""

import itertools

import numpy as np
import scipy.sparse as sparse

from .lmap import lmap, tensor
from .utils import qubits, op_list, assert_o, copy_memoize, gcd


__all__ = ['dist', 'id', 'phase', 'qft', 'swap', 'walsh',
           'mod_add', 'mod_inc', 'mod_mul',
           'controlled', 'single', 'two',
           'copydot', 'plusdot', 'epsilon']

# TODO reshape will cause problems for sparse matrices!
# TODO utils.op_list too!
# TODO which one is faster in element assignment -style init, dok or lil?
# TODO make input interface consistent, do we want arrays or lmaps?


def dist(A, B):
    r"""Distance between two unitary lmaps.

    Args:
      A (lmap): unitary operator
      B (lmap): unitary operator

    Returns:
      float: squared Frobenius norm distance between the unitaries, disregarding global phase

    .. math::
       \mathrm{dist}(A, B) = \inf_{\phi \in \mathbb{R}} \|A - e^{i \phi} B\|_F^2
       = 2 (\dim_A - |\mathrm{Tr}(A^\dagger B)|)
    """
    # Ville Bergholm 2007-2010

    if not A.is_compatible(B):
        raise ValueError('The lmaps have different dimensions.')

    temp = A.ctranspose() * B
    return 2 * (np.prod(temp.dim[0]) - abs(np.trace(temp.data)))



def id(dim):
    """Identity gate.

    Args:
        dim (tuple[int]): subsystem dimensions

    Returns:
        lmap: identity gate :math:`\I` for the specified system
    """
    if np.isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    return lmap(sparse.eye(np.prod(dim)), (dim, dim))


def mod_add(dim1, dim2, N=None):
    r"""Modular adder gate.

    Args:
        dim1 (tuple[int]): control register dimensions
        dim2 (tuple[int]): target register dimensions
        N (int): cutoff dimension, default is ``prod(dim2)``

    Returns:
        lmap: modular adder gate U

    .. math::
       U: \ket{x, y} \mapsto \ket{x, y+x (\mod N)}

    If N is given we must have ``N <= prod(dim2)``, and U will act trivially on target states ``>= N``.

    Notes:
        The modular subtractor gate can be obtained by taking the Hermitian conjugate of ``mod_add``.
        ``mod_add(2, 2)`` is equal to CNOT.
    """
    # Ville Bergholm 2010

    d1 = np.prod(dim1)
    d2 = np.prod(dim2)
    if N is None:
        N = d2
    elif d2 < N:
        raise ValueError('Target register dimension must be >= N.')

    # NOTE: a real quantum computer would implement this gate using a
    # sequence of reversible arithmetic gates but since we don't have
    # one we might as well cheat
    dim = d1 * d2
    U = sparse.dok_matrix((dim, dim))
    for a in range(d1):
        for b in range(d2):
            y = d2*a + b
            if b < N:
                x = d2*a +np.mod(a+b, N)
            else:
                # U acts trivially for target states >= N
                x = y
            U[x, y] = 1

    dim = (dim1, dim2)
    return lmap(U.tocsr(), (dim, dim))


def mod_inc(x, dim, N=None):
    r"""Modular incrementation gate.

    Args:
        x (int): increment
        dim (tuple[int]): register dimensions
        N (int): cutoff dimension, default is ``prod(dim)``

    Returns:
        lmap: modular incrementation gate U

    .. math::
       U: \ket{y} \mapsto \ket{y+x (mod N)}

    If N is given, we must have ``N <= prod(dim)``, and U will act trivially on computational states ``>= N``.
    """
    # Ville Bergholm 2010

    if np.isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    d = np.prod(dim)
    if N is None:
        N = d
    elif d < N:
        raise ValueError('Gate dimension must be >= N.')

    U = sparse.dok_matrix((d, d))
    for y in range(N):
        U[np.mod(x+y, N), y] = 1
    # U acts trivially for states >= N
    for y in range(N, d):
        U[y, y] = 1

    return lmap(U.tocsr(), (dim, dim))


def mod_mul(x, dim, N=None):
    r"""Modular multiplication gate.

    Args:
        x (int): multiplier
        dim (tuple[int]): register dimensions
        N (int): cutoff dimension, default is ``prod(dim)``

    Returns:
        lmap: modular multiplication gate U

    .. math::
       U: \ket{y} \mapsto \ket{x*y (mod N)}

    ``x`` and ``N`` must be coprime for the operation to be reversible.

    If N is given, we must have ``N <= prod(dim)``, and U will act trivially on computational states ``>= N``.
    """
    # Ville Bergholm 2010-2011

    if np.isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    d = np.prod(dim)
    if N is None:
        N = d
    elif d < N:
        raise ValueError('Gate dimension must be >= N.')

    if gcd(x, N) != 1:
        raise ValueError('x and N must be coprime for the mul operation to be reversible.')

    # NOTE: a real quantum computer would implement this gate using a
    # sequence of reversible arithmetic gates but since we don't have
    # one we might as well cheat
    U = sparse.dok_matrix((d, d))
    for y in range(N):
        U[np.mod(x*y, N), y] = 1
    # U acts trivially for states >= N
    for y in range(N, d):
        U[y, y] = 1

    return lmap(U.tocsr(), (dim, dim))


def phase(theta, dim=None):
    """Diagonal phase shift gate.

    Args:
        theta (vector[float]): phase shift angles
        dim (tuple[int]): register dimensions, default is ``len(theta),``

    Returns:
        lmap: the (diagonal) phase shift gate ``diag(exp(i*theta))``
    """
    # Ville Bergholm 2011

    if np.isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    n = len(theta)
    if dim is None:
        dim = (n,)
    d = np.prod(dim)
    if d != n:
        raise ValueError('Dimension mismatch.')

    return lmap(sparse.diags(np.exp(1j * theta), 0) , (dim, dim))


@copy_memoize
def qft(dim):
    """Quantum Fourier transform gate.

    Args:
        dim (tuple[int]): register dimensions

    Returns:
        lmap: QFT

    Returns the quantum Fourier transform gate for the specified system.
    The returned lmap is dense.
    """
    # Ville Bergholm 2004-2011

    if np.isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    N = np.prod(dim)
    U = np.empty((N, N), complex)  # completely dense, so we don't have to initialize it with zeros
    for j in range(N):
        for k in range(N):
            U[j, k] = np.exp(2j * np.pi * j * k / N) / np.sqrt(N)
    return lmap(U, (dim, dim))


def swap(d1, d2):
    r"""Swap gate.

    Args:
        d1 (int): subsystem 1 dimension
        d2 (int): subsystem 2 dimension

    Returns:
        lmap: SWAP gate which swaps the order of two subsystems with dimensions [d1, d2].

    .. math::

       S: A_1 \otimes A_2 \to A_2 \otimes A_1, \quad
       \ket{x, y} \mapsto \ket{y, x}

    Note:
        The actual subsystem order is swapped as well, not just the states of those subsystems.
        This is only important if ``d1 != d2``.
    """
    # Ville Bergholm 2010

    temp = d1*d2
    U = sparse.dok_matrix((temp, temp))
    for x in range(d1):
        for y in range(d2):
            U[d1*y + x, d2*x + y] = 1
    return lmap(U.tocsr(), ((d2, d1), (d1, d2)))


def walsh(n):
    """Walsh-Hadamard gate.

    Args:
        n (int): number of qubits

    Returns:
        lmap: Walsh-Hadamard gate for n qubits

    The returned lmap is dense.
    """
    # Ville Bergholm 2009-2010

    from .base import H

    U = 1
    for _ in range(n):
        U = np.kron(U, H)
    dim = qubits(n)
    return lmap(U, (dim, dim))


def controlled(U, ctrl=(1,), dim=None):
    r"""Controlled gate.

    Args:
        U (array[complex], lmap): unitary operator
        ctrl (vector[int]): control nodes
        dim (tuple[int]): control subsystem dimensions

    Returns:
        lmap: controlled-U gate

    Returns the ``(t+1)``-qudit controlled-U gate, where ``t == len(ctrl)``.

    ``ctrl`` defines the control nodes. It has one entry k per
    control qudit, denoting the required computational basis state :math:`\ket{k}`
    for that particular qudit. Value k == -1 denotes no control.

    dim is the dimensions vector for the control qudits. If not given, all controls
    are assumed to be qubits.

    Examples:

      * ``controlled(NOT, [1])`` gives the standard CNOT gate.
      * ``controlled(NOT, [1, 1])`` gives the Toffoli gate.
    """
    # Ville Bergholm 2009-2011

    # TODO generalization, uniformly controlled gates?
    if np.isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    t = len(ctrl)
    if dim is None:
        dim = qubits(t) # qubits by default

    if t != len(dim):
        raise ValueError('ctrl and dim vectors have unequal lengths.')

    if any(np.array(ctrl) >= np.array(dim)):
        raise ValueError('Control on non-existant state.')

    yes = 1  # just the diagonal
    for k in range(t):
        if ctrl[k] >= 0:
            temp = np.zeros(dim[k])
            temp[ctrl[k]] = 1  # control on k
            yes = np.kron(yes, temp)
        else:
            yes = np.kron(yes, np.ones(dim[k])) # no control on this qudit

    no = 1 - yes
    T = np.prod(dim)
    dim = list(dim)

    if isinstance(U, lmap):
        d1 = dim + list(U.dim[0])
        d2 = dim + list(U.dim[1])
        U = U.data
    else:
        d1 = dim + [U.shape[0]]
        d2 = dim + [U.shape[1]]

    # controlled gates only make sense for square matrices U (we need an identity transformation for the 'no' cases!)
    U_dim = U.shape[0]
    out = sparse.diags(np.kron(no, np.ones(U_dim)), 0) +sparse.kron(sparse.diags(yes, 0), U)
    return lmap(out, (d1, d2))


def single(L, t, d_in):
    """Single-qudit operator.

    Args:
        L (lmap, array): local (one-subsystem) operator
        t (int): subsystem to which L is applied
        d_in (tuple[int]): input dimensions for the constructed operator

    Returns:
        lmap: L applied to subsystem t (and identity applied to the remaining subsystems)
    """
    # James Whitfield 2010
    # Ville Bergholm 2010

    if isinstance(L, lmap):
        L = L.data  # into ndarray

    d_in = list(d_in)
    if d_in[t] != L.shape[1]:
        raise ValueError('Input dimensions do not match.')
    d_out = d_in
    d_out[t] = L.shape[0]
    return lmap(op_list([[[L, t]]], d_in), (d_out, d_in))


def two(B, t, d_in):
    """Two-qudit operator.

    Args:
        B (lmap, array): two-subsystem operator
        t (tuple[int]): two subsystems to which B is applied
        d_in (tuple[int]): input dimensions for the constructed operator

    Returns:
        lmap: B applied to subsystems t (and identity applied to the remaining subsystems)
    """
    # James Whitfield 2010
    # Ville Bergholm 2010-2011

    if len(t) != 2:
        raise ValueError('Exactly two target subsystems required.')

    n = len(d_in)
    t = np.array(t)
    if any(t < 0) or any(t >= n) or t[0] == t[1]:
        raise ValueError('Bad target subsystem(s).')

    d_in = np.array(d_in)
    if not np.array_equal(d_in[t], B.dim[1]):
        raise ValueError('Input dimensions do not match.')

    # dimensions for the untouched subsystems
    a = min(t)
    b = max(t)
    before    = np.prod(d_in[:a])
    inbetween = np.prod(d_in[a+1:b])
    after     = np.prod(d_in[b+1:])

    # how tensor(B_{01}, I_2) should be reordered
    if t[0] < t[1]:
        p = [0, 2, 1]
    else:
        p = [1, 2, 0]
    U = tensor(B, lmap(np.eye(inbetween))).reorder((p, p), inplace = True)
    U = tensor(lmap(sparse.eye(before)), U, lmap(sparse.eye(after)))

    # restore dimensions
    d_out = d_in.copy()
    d_out[t] = B.dim[0]
    return lmap(U, (d_out, d_in))


def copydot(n_in, n_out, d=2):
    """Copy dot.

    Args:
        n_in (int): number of input legs
        n_out (int): number of ouput legs
        d (int): leg dimension

    Returns:
        lmap: copy dot

    See :cite:`BB2011`
    """
    # Ville Bergholm 2014-2016

    d_in  = d**n_in
    d_out = d**n_out
    C = sparse.dok_matrix((d_out, d_in))
    # compute the strides by summing up 1+d+d^2+...+d^(n-1)
    stride_in  = (d_in -1) / (d-1)
    stride_out = (d_out-1) / (d-1)
    # loop over the sum
    for k in range(d):
        C[k*stride_out, k*stride_in] = 1
    return lmap(C, ((d,)*n_out, (d,)*n_in))


def plusdot(n_in, n_out, d=2):
    """Plus dot.

    Args:
        n_in (int): number of input legs
        n_out (int): number of ouput legs
        d (int): leg dimension

    Returns:
        lmap: plus dot

    See :cite:`BB2011`
    """
    # Ville Bergholm 2014-2016

    # TODO this implementation has small numerical errors from qft, we can do better
    #U = copydot(n_in, n_out, d)
    #Q = qft(d)
    #return Q.tensorpow(n_out) * U * Q.tensorpow(n_in)

    dim_in  = (d,) * n_in
    dim_out = (d,) * n_out
    d_in  = d ** n_in
    d_out = d ** n_out
    P = sparse.dok_matrix((d_out, d_in))
    x = d ** (-(n_in +n_out -2)/2)

    # TODO if scipy could reshape sparse matrices this would be easy... build a n_in==0 plusdot and then reshape it.
    if n_in == 0:
        # loop over all output indices except the last one
        for j in range(int(d_out / d)):
            temp = np.sum(np.unravel_index(k, dim_out[1:]))  # sum of subsystem indices
            last = -temp % d  # the last index must take this value for the sum to be 0 (mod d)
            P[j, d*j +last] = x
    else:
        for j in range(d_out):
            temp = np.sum(np.unravel_index(j, dim_out))
            for k in range(int(d_in / d)):
                temp2 = np.sum(np.unravel_index(k, dim_in[1:]))
                last = -(temp +temp2) % d
            P[j, d*k +last] = x
    return lmap(P, (dim_out, dim_in))


def epsilon(n):
    """Epsilon tensor.

    Args:
        n (int): dimension

    Returns:
        lmap: the fully antisymmetric Levi-Civita symbol in ``n`` dimensions,
            a tensor with ``n`` ``n``-dimensional subsystems
    """
    # Ville Bergholm 2016
    def signum(p):
        """Returns the parity of the permutation p by counting cycle lengths."""
        n = len(p)
        seen = np.zeros(n, dtype=bool)
        sgn = 1;
        k = 0
        while k < n:
            if not seen[k]:
                # a new cycle starts
                cl = 0
                while not seen[k]:
                    cl += 1
                    seen[k] = True
                    k = p[k]
                # even cycle: invert parity
                if cl % 2 == 0:
                    sgn = -sgn
            k += 1
        return sgn

    D = n ** n
    dim = n * (n,)
    U = sparse.dok_matrix((D, 1))
    # loop through all permutations of n indices
    p = itertools.permutations(range(n))
    for k in p:
        ind = np.ravel_multi_index(k, dim)
        U[ind,0] = signum(k)
    return lmap(U, (dim, (1,)))
