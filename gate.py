# -*- coding: utf-8 -*-
"""Quantum gates and other linear maps."""

from __future__ import print_function, division
from copy import deepcopy

from numpy import prod, diag, eye, zeros, trace, exp, sqrt, mod, isscalar, kron, array

from lmap import *
from utils import qubits, op_list, assert_o, copy_memoize, gcd
# TODO use sparse matrices
#import scipy.sparse as sps
# zeros => sps.xxx_matrix
# eye => sps.eye

# all gate funcs return lmaps
# TODO make input interface consistent, do we want arrays or lmaps?



def dist(A, B):
    """Distance between two unitary lmaps.

    Returns \inf_{\phi \in \reals} \|A - e^{i \phi} B\|_F^2
      = 2 (\dim_A - |\trace(A^\dagger B)|)

    Ville Bergholm 2007-2010
    """
    if not A.is_compatible(B):
        raise ValueError('The lmaps have different dimensions.')

    temp = A.ctranspose() * B
    return 2 * (prod(temp.dim[0]) - abs(trace(temp.data)))



def id(dim):
    """Identity gate.

    Returns the identity gate I for the specified system.
    dim is a tuple of subsystem dimensions.
    """
    if isscalar(dim): dim = (dim,)  # scalar into a tuple
    return lmap(eye(prod(dim)), (dim, dim))


def mod_add(dim1, dim2, N=None):
    """Modular adder gate.

    U = mod_add(d1, d2)     N == prod(d2)
    U = mod_add(d1, d2, N)  target register dimension prod(d2) must be >= N

    Returns the gate U, which, operating on the computational state
    |x, y>, produces |x, y+x (mod N)>.
    d1 and d2 are the control and target register dimensions.

    If N is given, U will act trivially on target states >= N.

    Notes:
    The modular subtractor gate can be obtained by taking the
    Hermitian conjugate of mod_add.
    mod_add(2, 2) is equal to CNOT.

    Ville Bergholm 2010
    """
    d1 = prod(dim1)
    d2 = prod(dim2)
    if N == None:
        N = d2
    elif d2 < N:
        raise ValueError('Target register dimension must be >= N.')

    # NOTE: a real quantum computer would implement this gate using a
    # sequence of reversible arithmetic gates but since we don't have
    # one we might as well cheat
    dim = d1 * d2
    U = zeros((dim, dim))
    for a in range(d1):
        for b in range(d2):
            y = d2*a + b
            if b < N:
                x = d2*a +mod(a+b, N)
            else:
                # U acts trivially for target states >= N
                x = y
            U[x, y] = 1

    dim = (dim1, dim2)
    return lmap(U, (dim, dim))


def mod_inc(x, dim, N=None):
    """Modular incrementation gate.

    U = mod_inc(x, dim)    # N == prod(dim)
    U = mod_inc(x, dim, N) % gate dimension prod(dim) must be >= N

    Returns the gate U, which, operating on the computational state
    |y>, increments it by x (mod N):  U |y> = |y+x (mod N)>.

    If N is given, U will act trivially on computational states >= N.
    """
    if isscalar(dim): dim = (dim,)  # scalar into a tuple
    d = prod(dim)
    if N == None:
        N = d
    elif d < N:
        raise ValueError('Gate dimension must be >= N.')

    U = zeros((d, d))
    for y in range(N):
        U[mod(x+y, N), y] = 1
    # U acts trivially for states >= N
    for y in range(N, d):
        U[y, y] = 1

    return lmap(U, (dim, dim))


def mod_mul(x, dim, N=None):
    """Modular multiplication gate.

    U = mod_mul(x, dim)     N == prod(dim)
    U = mod_mul(x, dim, N)  gate dimension prod(dim) must be >= N

    Returns the gate U, which, operating on the computational state
    |y>, multiplies it by x (mod N):  U |y> = |x*y (mod N)>.
    x and N must be coprime for the operation to be reversible.

    If N is given, U will act trivially on computational states >= N.
    """
    if isscalar(dim): dim = (dim,)  # scalar into a tuple
    d = prod(dim)
    if N == None:
        N = d
    elif d < N:
        raise ValueError('Gate dimension must be >= N.')

    if gcd(x, N) != 1:
        raise ValueError('x and N must be coprime for the mul operation to be reversible.')

    # NOTE: a real quantum computer would implement this gate using a
    # sequence of reversible arithmetic gates but since we don't have
    # one we might as well cheat
    U = zeros((d, d))
    for y in range(N):
        U[mod(x*y, N), y] = 1
    # U acts trivially for states >= N
    for y in range(N, d):
        U[y, y] = 1

    return lmap(U, (dim, dim))


def phase(theta, dim=None):
    """Diagonal phase shift gate.

    Returns the (diagonal) phase shift gate U = diag(exp(i*theta)).

    Ville Bergholm 2010
    """
    if isscalar(dim): dim = (dim,)  # scalar into a tuple
    n = len(theta)
    if dim == None:
        dim = (n,)
    elif prod(dim) != n:
        raise ValueError('Dimension mismatch.')

    return lmap(diag(exp(1j * theta)), (dim, dim))


@copy_memoize
def qft(dim):
    """Quantum Fourier transform gate.

    Returns the quantum Fourier transform gate for the specified system.
    dim is a vector of subsystem dimensions.

    Ville Bergholm 2004-2010
    """
    if isscalar(dim): dim = (dim,)  # scalar into a tuple
    n = len(dim)
    N = prod(dim)
    U = zeros((N, N))
    for j in range(N):
        for k in range(N):
            U[j, k] = exp(2j * np.pi * j * k / N) / sqrt(N)
    return lmap(U, (dim, dim))


def swap(d1, d2):
    """Swap gate.

    Returns the swap gate which swaps the order of two subsystems with dimensions [d1, d2].
    S: A_1 \otimes A_2 \to A_2 \otimes A_1,
    S(v_1 \otimes v_2) = v_2 \otimes v_1   for all v_1 \in A_1, v_2 \in A_2.

    Ville Bergholm 2010
    """
    temp = d1*d2
    U = zeros((temp, temp))
    for x in range(d1):
        for y in range(d2):
            U[d1*y + x, d2*x + y] = 1
    return lmap(U, ((d2, d1), (d1, d2)))


def walsh(n):
    """Walsh-Hadamard gate.

    Returns the Walsh-Hadamard gate for n qubits.

    Ville Bergholm 2009-2010
    """
    from base import H

    U = 1
    for k in range(n):
        U = kron(U, H)
    dim = qubits(n)
    return lmap(U, (dim, dim))


def controlled(U, ctrl=(1,), dim=None):
    """Controlled gate.

    Returns the (t+1)-qudit controlled-U gate, where t == length(ctrl).
    ctrl is an integer vector defining the control nodes. It has one entry k per
    control qudit, denoting the required computational basis state |k>
    for that particular qudit. Value k == -1 denotes no control.

    dim is the dimensions vector for the control qudits. If not given, all controls
    are assumed to be qubits.

    Examples:
      controlled(NOT, [1]) gives the standard CNOT gate.
      controlled(NOT, [1, 1]) gives the Toffoli gate.

    Ville Bergholm 2009-2011
    """
    if isscalar(dim): dim = (dim,)  # scalar into a tuple
    t = len(ctrl)
    if dim == None:
        dim = qubits(t) # qubits by default

    if t != len(dim):
        raise ValueError('ctrl and dim vectors have unequal lengths.')

    if any(array(ctrl) >= array(dim)):
        raise ValueError('Control on non-existant state.')

    yes = 1  # just the diagonal
    for k in range(t):
        if ctrl[k] >= 0:
            temp = zeros(dim[k])
            temp[ctrl[k]] = 1  # control on k
            yes = kron(yes, temp) 
        else:
            yes = kron(yes, ones(dim[k])) # no control on this qudit

    no = 1 - yes
    T = prod(dim)
    dim = list(dim)

    if isinstance(U, lmap):
        d1 = dim + list(U.dim[0])
        d2 = dim + list(U.dim[1])
        U = U.data
    else:
        d1 = dim + [U.shape[0]]
        d2 = dim + [U.shape[1]]

    out = kron(diag(no), eye(*(U.shape))) + kron(diag(yes), U)
    return lmap(out, (d1, d2))


def single(L, t, d_in):
    """Single-qudit operator.

    Returns the operator U corresponding to the local operator L applied
    to subsystem t (and identity applied to the remaining subsystems).

    d_in is the input dimension vector for U.

    James Whitfield 2010
    Ville Bergholm 2010
    """
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

    Returns the operator U corresponding to the bipartite operator B applied
    to subsystems t == [t1, t2] (and identity applied to the remaining subsystems).

    d_in is the input dimension vector for U.

    James Whitfield 2010
    Ville Bergholm 2010
    """
    if len(t) != 2:
        raise ValueError('Exactly two target subsystems required.')

    n = len(d_in)
    t = array(t)
    if any(t < 0) or any(t >= n) or t[0] == t[1]:
        raise ValueError('Bad target subsystem(s).')

    d_in = array(d_in)
    if not np.array_equal(d_in[t], B.dim[1]):
        raise ValueError('Input dimensions do not match.')

    # dimensions for the untouched subsystems
    a = min(t)
    b = max(t)
    before    = prod(d_in[:a])
    inbetween = prod(d_in[a+1:b])
    after     = prod(d_in[b+1:])

    # how tensor(B_{01}, I_2) should be reordered
    if t[0] < t[1]:
        p = [0, 2, 1]
    else:
        p = [1, 2, 0]
    U = tensor(B, lmap(eye(inbetween))).reorder((p, p), inplace = True)
    U = tensor(lmap(eye(before)), U, lmap(eye(after)))

    # restore dimensions
    d_out = d_in
    d_out[t] = B.dim[0]
    return lmap(U, (d_out, d_in))


def test():
    """Test script for gates.

    Ville Bergholm 2010
    """
    from numpy.random import randn
    from base import *

    dim = (2, 4)

    I = id(dim)
    U = swap(*dim)
    assert_o((U.ctranspose() * U - I).norm(), 0, tol)  # swap' * swap = I

    U = mod_add(2, 4, 3)
    U = mod_inc(3, dim, 5)
    V = mod_mul(2, dim, 5)
    dist(U, V)
    U = phase(randn(prod(dim)), dim)
    U = qft(dim)
    U = walsh(3)
    U = controlled(sz, (1, 0), dim)
    cnot = controlled(sx)
    U = single(sy, 0, dim)
    U = two(cnot, (2,0), (2,3,2))


    # dist, mod_add, mod_inc, mod_mul, phase, qft, walsh, controlled, single, two
