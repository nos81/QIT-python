# -*- coding: utf-8 -*-
"""Utility functions."""
# Ville Bergholm 2008-2011

from __future__ import division, absolute_import, print_function, unicode_literals

from copy import deepcopy

import numpy as np
from numpy import array, mat, dtype, empty, zeros, ones, eye, prod, sqrt, exp, tanh, dot, sort, diag, trace, kron, pi, r_, c_, inf, isscalar, floor, ceil, log, log10, vdot
from numpy.random import rand, randn, randint
from numpy.linalg import qr, det, eigh, eigvalsh
from scipy.linalg import expm, norm, svdvals

from .base import *

__all__ = ['assert_o', 'copy_memoize', 'gcd', 'lcm', 'rank', 'projector', 'eigsort', 'comm', 'acomm', 'expv',
           'rand_hermitian', 'rand_U', 'rand_SU', 'rand_U1', 'rand_pu', 'rand_positive', 'rand_SL', 
           'vec', 'inv_vec', 'lmul', 'rmul', 'lrmul', 'superop_lindblad',
           'angular_momentum', 'boson_ladder', 'fermion_ladder',
           'R_nmr', 'R_x', 'R_y', 'R_z',
           'spectral_decomposition',
           'gellmann', 'tensorbasis',
           'op_list',
           'qubits', 'majorize', 'mkron']


# the functions in this module return ndarrays, not lmaps, for now


# internal utilities

def assert_o(actual, desired, tolerance):
    """Octave-style assert."""
    if abs(actual - desired) > tolerance:
        raise AssertionError


def copy_memoize(func):
    """Memoization decorator for functions with immutable args, returns deep copies."""
    cache = {}
    def wrapper(*args):
        """Nonsense, this is an election year."""
        if args in cache:
            value = cache[args]
        else:
            value = func(*args)
            cache[args] = value

        return deepcopy(value)

    # so that the help system still works
    wrapper.__name__ = func.__name__
    wrapper.__doc__  = func.__doc__
    return wrapper



# math functions

def gcd(a, b):
    """Greatest common divisor.

    Uses the Euclidean algorithm.
    """
    while b:
        a, b = b, a%b
    return a


def lcm(a, b):
    """Least common multiple."""
    return a * (b // gcd(a, b))


def rank(A, eps=1e-8):
    """Matrix rank."""
    s = svdvals(A)
    return sum(s > eps)


def projector(v):
    """Projector corresponding to vector v."""
    return np.outer(v, v.conj())


def eigsort(A):
    """Returns eigenvalues and eigenvectors of a Hermitian matrix, sorted in nonincreasing order."""
    d, v = eigh(A)
    ind = d.argsort()[::-1]  # nonincreasing real part
    return d[ind], v[:, ind]


def comm(A, B):
    """Array commutator.
    
    Returns [A, B] := A*B - B*A 
    """
    return dot(A, B) - dot(B, A)


def acomm(A, B):
    """Array anticommutator.
    
    Returns {A, B} := A*B + B*A 
    """
    return dot(A, B) + dot(B, A)


def expv(t, A, v, tol=1.0e-7, m=30, iteration='arnoldi'):
    r"""Multiply a vector by an exponentiated matrix.

    Approximates :math:`exp(t A) v` using a Krylov subspace technique.
    Efficient for large sparse matrices.
    The basis for the Krylov subspace is constructed using either Arnoldi or Lanczos iteration.

    Input:
    t           vector of nondecreasing time instances >= 0
    A           n*n matrix (usually sparse) (as an (n,n)-shaped ndarray)
    v           n-dimensional vector (as an (n,)-shaped ndarray)
    tol         tolerance
    m           Krylov subspace dimension, <= n
    iteration   'arnoldi' or 'lanczos'. Lanczos is faster but requires a Hermitian A.

    Output:
    W       result matrix, :math:`W[i,:] \approx \exp(t[i] A) v`
    error   total truncation error estimate
    hump    :math:`\max_{s \in [0, t]}  \| \exp(s A) \|`

    Uses the sparse algorithm from [EXPOKIT]_.

    .. [EXPOKIT] Sidje, R.B., "EXPOKIT: A Software Package for Computing Matrix Exponentials", ACM Trans. Math. Softw. 24, 130 (1998).
    """
    # Ville Bergholm 2009-2012

    # just in case somebody tries to use numpy.matrix instances here
    if isinstance(A, np.matrix) or isinstance(v, np.matrix):
        raise ValueError("A and v must be plain numpy.ndarray instances, not numpy.matrix.")

    n = A.shape[0]
    m = min(n, m)  # Krylov space dimension, m <= n 

    if isscalar(t):
        tt = array([t])
    else:
        tt = t

    a_norm = norm(A, inf)
    v_norm = norm(v)

    happy_tol  = 1.0e-7  # "happy breakdown" tolerance
    min_error = a_norm * np.finfo(float).eps # due to roundoff

    # step size control
    max_stepsize_changes = 10
    # safety factors
    gamma = 0.9
    delta = 1.2
    # initial stepsize
    fact = sqrt(2 * pi * (m + 1)) * ((m + 1) / exp(1)) ** (m + 1)

    def ceil_at_nsd(x, n = 2):
        temp = 10 ** (floor(log10(x))-n+1)
        return ceil(x / temp) * temp

    def update_stepsize(step, err_loc, r):
        step *= gamma  * (tol * step / err_loc) ** (1 / r)
        return ceil_at_nsd(step, 2)

    dt = dtype(complex)
    # TODO don't use complex matrices unless we have to: dt = result_type(t, A, v)

    # TODO shortcuts for Hessenberg matrix exponentiation?
    H = zeros((m+2, m+2), dt) # upper Hessenberg matrix for the Arnoldi process + two extra rows/columns for the error estimate trick
    H[m + 1, m] = 1           # never overwritten!
    V = zeros((n, m+1), dt)   # orthonormal basis for the Krylov subspace + one extra vector

    W = empty((len(tt), len(v)), dt)  # results
    t = 0  # current time
    beta = v_norm
    error = 0  # error estimate
    hump = [[v_norm, t]]
    #v_norm_max = v_norm  # for estimating the hump

    def iterate_lanczos(v, beta):
        """Lanczos iteration, for Hermitian matrices.
        Produces a tridiagonal H, cheaper than Arnoldi.

        Returns the number of basis vectors generated, and a boolean indicating a happy breakdown.
        NOTE that the we _must_not_ change global variables other than V and H here
        """
        # beta_0 and alpha_m are not used in H, beta_m only in a single position for error control 
        prev = 0
        for k in range(0, m):
            vk = (1 / beta) * v
            V[:, k] = vk  # store the now orthonormal basis vector
            # construct the next Krylov vector beta_{k+1} v_{k+1}
            v = dot(A, vk)
            H[k, k] = alpha = vdot(vk, v)
            v += -alpha * vk -beta * prev
            # beta_{k+1}
            beta = norm(v)
            if beta < happy_tol: # "happy breakdown": iteration terminates, Krylov approximation is exact
                return k+1, True
            if k == m-1:
                # v_m and one beta_m for error control (alpha_m not used)
                H[m, m-1] = beta
                V[:, m] = (1 / beta) * v
            else:
                H[k+1, k] = H[k, k+1] = beta
                prev = vk
        return m+1, False

    def iterate_arnoldi(v, beta):
        """Arnoldi iteration, for generic matrices.
        Produces a Hessenberg-form H.
        """
        V[:, 0] = (1 / beta) * v  # the first basis vector v_0 is just v, normalized
        for j in range(1, m+1):
            p = dot(A, V[:, j-1])  # construct the Krylov vector v_j
            # orthogonalize it with the previous ones
            for i in range(j):
                H[i, j-1] = vdot(V[:, i], p)
                p -= H[i, j-1] * V[:, i]
            temp = norm(p) 
            if temp < happy_tol: # "happy breakdown": iteration terminates, Krylov approximation is exact
                return j, True
            # store the now orthonormal basis vector
            H[j, j-1] = temp
            V[:, j] = (1 / temp) * p
        return m+1, False  # one extra vector for error control

    # choose iteration type
    iteration = iteration.lower()
    if iteration == 'lanczos':
        iteration = iterate_lanczos  # only works for Hermitian matrices!
    elif iteration == 'arnoldi':
        iteration = iterate_arnoldi
    else:
        raise ValueError("Only 'arnoldi' and 'lanczos' iterations are supported.")

    # loop over the time instances (which must be in increasing order)
    for kk in range(len(tt)):
        t_end = tt[kk]
        # initial stepsize
        # TODO we should inherit the stepsize from the previous interval
        r = m
        t_step = (1 / a_norm) * ((fact * tol) / (4 * beta * a_norm)) ** (1 / r)
        t_step = ceil_at_nsd(t_step, 2)

        while t < t_end:
            t_step = min(t_end - t, t_step)  # step at most the remaining distance

            # Arnoldi/Lanczos iteration, (re)builds H and V
            j, happy = iteration(v, beta)
            # now V^\dagger A V = H  (just the first m vectors, or j if we had a happy breakdown!)
            # assert(norm(dot(dot(V[:, :m].conj().transpose(), A), V[:, :m]) -H[:m,:m]) < tol)

            # error control
            if happy:
                # "happy breakdown", using j Krylov basis vectors
                t_step = t_end - t  # step all the rest of the way
                F = expm(t_step * H[:j, :j])
                err_loc = happy_tol
                r = m
            else:
                # no happy breakdown, we need the error estimate (using all m+1 vectors)
                av_norm = norm(dot(A, V[:, m]))
                # find a reasonable step size
                for k in range(max_stepsize_changes + 1):
                    F = expm(t_step * H)
                    err1 = beta * abs(F[m, 0])
                    err2 = beta * abs(F[m+1, 0]) * av_norm
                    if err1 > 10 * err2:  # quick convergence
                        err_loc = err2
                        r = m 
                    elif err1 > err2:  # slow convergence
                        err_loc = (err2 * err1) / (err1 - err2)
                        r = m 
                    else:  # asymptotic convergence
                        err_loc = err1
                        r = m-1
                    # should we accept the step?
                    if err_loc <= delta * tol * t_step:
                        break
                    if k >= max_stepsize_changes:
                        raise RuntimeError('Requested tolerance cannot be achieved in {0} stepsize changes.'.format(max_stepsize_changes))
                    t_step = update_stepsize(t_step, err_loc, r)

            # step accepted, update v, beta, error, hump
            v = dot(V[:, :j], beta * F[:j, 0])
            beta = norm(v)
            error += max(err_loc, min_error)
            #v_norm_max = max(v_norm_max, beta)

            t += t_step
            t_step = update_stepsize(t_step, err_loc, r)
            hump.append([beta, t])

        W[kk, :] = v

    hump = array(hump) / v_norm
    return W, error, hump



# random matrices

def rand_hermitian(n):
    """Random Hermitian n*n matrix.

    Returns a random Hermitian matrix of size n*n.
    NOTE: The randomness is not defined in any deeply meaningful sense.
    """
    # Ville Bergholm 2008-2009

    H = (rand(n,n) - 0.5) +1j*(rand(n,n) - 0.5)
    return H + H.conj().transpose() # make it Hermitian


def rand_U(n):
    """Random U(n) matrix.

    Returns a random unitary n*n matrix.
    The matrix is random with respect to the Haar measure.

    Uses the algorithm in [Mezzadri]_.

    .. [Mezzadri] F.Mezzadri, "How to generate random matrices from the classical compact groups", Notices of the AMS 54, 592 (2007). arXiv.org:math-ph/0609050
    """
    # Ville Bergholm 2005-2009

    # sample the Ginibre ensemble, p(Z(i,j)) == 1/pi * exp(-abs(Z(i,j))^2),
    # p(Z) == 1/pi^(n^2) * exp(-trace(Z'*Z))
    Z = (randn(n,n) + 1j*randn(n,n)) / sqrt(2)

    # QR factorization
    Q, R = qr(Z)

    # eliminate multivaluedness in Q
    P = diag(R)
    P /= abs(P)
    return dot(Q, diag(P))


def rand_SU(n):
    """Random SU(n) matrix.

    Returns a random special unitary n*n matrix.
    The matrix is random with respect to the Haar measure.
    """
    # Ville Bergholm 2005-2009

    U = rand_U(n)
    d = det(U) ** (1/n) # *exp(i*2*pi*k/n), not unique FIXME
    return U/d


def rand_U1(n):
    """Random diagonal unitary matrix.

    Returns a random diagonal unitary n*n matrix.
    The matrix is random with respect to the Haar measure.
    """
    # Ville Bergholm 2005-2009

    return diag(exp(2j * pi * rand(n)))


def rand_pu(n):
    r"""Random n-partition of unity.

    Returns the n-partition p, which is random with respect to
    the order-n Dirichlet distribution :math:`Dir(\alpha)`
    with :math:`\alpha = (1, 1, ..., 1)`.
    """
    # Ville Bergholm 2008-2012

    # Sample the Dirichlet distribution using n gamma-distributed
    # random vars. The (shared) scale parameter of the gamma pdfs is irrelevant,
    # and the shape parameters correspond to the Dirichlet \alpha params.
    # Furthermore, Gamma(x; 1,1) = exp(-x), so
    p = -log(rand(n))  # Gamma(1,1) -distributed
    return p / sum(p)  # Dir(1, 1, ..., 1) -distributed

    # TODO this would be a simpler choice, but what's the exact distribution?
    #p = sort(rand(n-1))  # n-1 points in [0,1]
    #d = sort(r_[p, 1] - r_[0, p])  # n deltas between points = partition of unity


def rand_positive(n):
    """Random n*n positive semidefinite matrix.

    Normalized to Tr(A) = 1.
    Since A has all-real eigenvalues, it is Hermitian by construction.
    """
    # Ville Bergholm 2008-2012

    d = rand_pu(n) # random partition of unity
    U = mat(rand_U(n)) # random unitary
    A = U.H * diag(d) * U
    return array((A + A.H) / 2)   # eliminate rounding errors
    # An alternative would be to use an inverse purification, but this would be expensive.


def rand_SL(n):
    """Random SL(n, C) matrix.

    Returns a random special linear n*n matrix.
    NOTE: The randomness is not defined in any deeply meaningful sense.
    """
    # Ville Bergholm 2011

    S = randn(n,n) +1j*randn(n,n)
    d = det(S) ** (1/n)
    return S/d



# superoperators

def vec(rho):
    """Flattens a matrix into a vector.

    Matrix rho is flattened columnwise into a column vector v.

    Used e.g. to convert state operators to superoperator representation.
    """
    # JDW 2009
    # Ville Bergholm 2009

    return rho.flatten('F')  # copy


def inv_vec(v, dim=None):
    r"""Reshapes a vector into a matrix.

    Given dim == (n, m), reshapes vector v (length n*m) into a matrix :math:`\rho` (shape == dim),
    using column-major ordering. If dim is not given, :math:`\rho` is assumed to be square.

    Used e.g. to convert state operators from superoperator representation
    to standard matrix representation.
    """
    # JDW 2009
    # Ville Bergholm 2009

    d = v.size
    if dim == None:
        # assume a square matrix
        n = sqrt(d)
        if floor(n) != n:
            raise ValueError('Length of vector v is not a squared integer.')
        dim = (n, n)
    else:
        if prod(dim) != d:
            raise ValueError('Dimensions n, m are not compatible with given vector v.')
    return v.reshape(dim, order='F').copy()


def lmul(L, q=None):
    r"""Superoperator equivalent for multiplying from the left.

    .. math::

       L \rho = \text{inv\_vec}(\text{lmul}(L) * \text{vec}(\rho))

    Dimensions: L is [m, p], :math:`\rho` is [p, q].
    If q is not given :math:`\rho` is assumed square.
    """
    # Ville Bergholm 2009

    if q == None:
        q = L.shape[1]  # assume target is a square matrix
    return kron(eye(q), L)


def rmul(R, p=None):
    r"""Superoperator equivalent for multiplying from the right.

    .. math::

       \rho R = \text{inv\_vec}(\text{rmul}(R) * \text{vec}(\rho))

    Dimensions: :math:`\rho` is [p, q], R is [q, r].
    If p is not given :math:`\rho` is assumed square.
    """
    # Ville Bergholm 2009

    if p == None:
        p = R.shape[0]  # assume target is a square matrix
    return kron(R.transpose(), eye(p))


def lrmul(L, R):
    r"""Superoperator equivalent for multiplying both from left and right.

    .. math::

       L \rho R = \text{inv\_vec}(\text{lrmul}(L, R) * \text{vec}(\rho))
    """
    # Ville Bergholm 2009-2011

    # L and R fix the shape of rho completely
    return kron(R.transpose(), L)


def superop_lindblad(A, H=0):
    r"""Liouvillian superoperator for a set of Lindblad operators.

    A is a vector of traceless, orthogonal Lindblad operators.
    H is an optional Hamiltonian operator.

    Returns the Liouvillian superoperator L corresponding to the
    diagonal-form Lindblad equation

    .. math:: \dot{\rho} = \text{inv\_vec}(L * \text{vec}(\rho)) = -i [H, \rho] +\sum_k \left(A_k \rho A_k^\dagger -\frac{1}{2} \{A_k^\dagger A_k, \rho\}\right)
    """
    # James D. Whitfield 2009
    # Ville Bergholm 2009-2010

    # Hamiltonian
    iH = 1j * H

    L = zeros(array(H.shape) ** 2, complex)
    acomm = zeros(H.shape, complex)
    for k in A:
        acomm += 0.5 * dot(k.conj().transpose(), k)
        L += lrmul(k, k.conj().transpose()) 

    L += lmul(-acomm -iH) +rmul(-acomm +iH)
    return L



# physical operators

@copy_memoize
def angular_momentum(n):
    r"""Angular momentum matrices.

    (Jx, Jy, Jz) = angular_momentum(d)

    Returns a 3-tuple of angular momentum matrices :math:`\vec{J} / \hbar`
    for the d-dimensional subspace defined by the
    quantum number j == (d-1)/2.
    """
    # Ville Bergholm 2009-2010

    if n < 1:
        raise ValueError('Dimension must be one or greater.')

    j = (n - 1) / 2 # angular momentum quantum number, n == 2*j + 1
    # raising operator in subspace J^2 = j*(j+1)
    m = j
    Jplus = zeros((n, n))
    for k in range(n-1):
        m -= 1
        Jplus[k, k+1] = sqrt(j*(j+1) -m*(m+1))

    # lowering operator
    Jminus = Jplus.conj().transpose()
    # Jplus  = Jx + i*Jy
    # Jminus = Jx - i*Jy
    return (0.5*(Jplus + Jminus), 0.5j*(Jminus - Jplus), diag(np.arange(j, -j-1, -1)))


@copy_memoize
def boson_ladder(n):
    r"""Bosonic ladder operators.

    Returns the n-dimensional approximation of the bosonic
    annihilation operator b for a single bosonic mode in the
    number basis :math:`\{|0\rangle, |1\rangle, ..., |n-1\rangle\}`.

    The corresponding creation operator is :math:`b^\dagger`.
    """
    # Ville Bergholm 2009-2010

    return diag(sqrt(range(1, n)), 1)


@copy_memoize
def fermion_ladder(grouping):
    r"""Fermionic ladder operators.

    Returns a vector of fermionic annihilation operators for a
    system of n fermionic modes in the second quantization.

    The annihilation operators are built using the Jordan-Wigner
    transformation for a chain of n qubits, where the state of each
    qubit denotes the occupation number of the corresponding mode.
    First define annihilation and number operators for a lone fermion mode:

    .. math::

       s &:= (\sigma_x + i \sigma_y)/2   = [[0, 1], [0, 0]],\\
       n &:= s^\dagger s = (I-sz)/2 = [[0, 0], [0, 1]],\\
       &s|0\rangle = 0, \quad s|1\rangle = |0\rangle, \quad n|k\rangle = k|k\rangle.

    Then define a phase operator to keep track of sign changes when
    permuting the order of the operators: :math:`\phi_k := \sum_{j=0}^{k-1} n_j`.
    Now, the fermionic annihilation operators for the n-mode system are given by
    :math:`f_k := (-1)^{\phi_k} s_k`.
    These operators fulfill the required anticommutation relations:

    .. math::

       \{f_k, f_j\}  &= 0,\\
       \{f_k, f_j^\dagger\} &= I \delta_{kj},\\
       f_k^\dagger f_k &= n_k.
    """
    # Ville Bergholm 2009-2010

    n = prod(grouping)
    d = 2 ** n

    # number and phase operators (diagonal, so we store them as such)
    temp = zeros(d)
    phi = [temp]
    for k in range(n-1):  # we don't need the last one
        num = mkron(ones(2 ** k), array([0, 1]), ones(2 ** (n-k-1))) # number operator n_k as a diagonal
        temp += num # sum of number ops up to n_k, diagonal
        phi.append(temp)

    s = array([[0, 1], [0, 0]]) # single annihilation op

    # empty array for the annihilation operators
    f = empty(grouping, object)
    # annihilation operators for a set of fermions (Jordan-Wigner transform)
    for k in range(n):
        f.flat[k] = ((-1) ** array(phi[k])) * mkron(eye(2 ** k), s, eye(2 ** (n-k-1)))
    return f



# SU(2) rotations

def R_nmr(theta, phi):
    r"""SU(2) rotation :math:`\theta_\phi` (NMR notation).

    Returns the one-qubit rotation by angle theta about the unit
    vector :math:`[\cos(\phi), \sin(\phi), 0]`, or :math:`\theta_\phi` in the NMR notation.
    """
    # Ville Bergholm 2009

    return expm(-1j * theta/2 * (cos(phi) * sx + sin(phi) * sy))


def R_x(theta):
    r"""SU(2) x-rotation.

    Returns the one-qubit rotation about the x axis by the angle theta,
    :math:`e^{-i \sigma_x \theta/2}`.
    """
    # Ville Bergholm 2006-2009

    return expm(-1j * theta/2 * sx)


def R_y(theta):
    r"""SU(2) y-rotation.

    Returns the one-qubit rotation about the y axis by the angle theta,
    :math:`e^{-i \sigma_y \theta/2}`.
    """
    # Ville Bergholm 2006-2009

    return expm(-1j * theta/2 * sy)


def R_z(theta):
    r"""SU(2) z-rotation.

    Returns the one-qubit rotation about the z axis by the angle theta,
    :math:`e^{-i \sigma_z \theta/2}`.
    """
    # Ville Bergholm 2006-2009

    return array([[exp(-1j * theta/2), 0], [0, exp(1j * theta/2)]])



# decompositions

def spectral_decomposition(A):
    r"""Spectral decomposition of a Hermitian matrix.

    Returns the unique eigenvalues a and the corresponding projectors P
    for the Hermitian matrix A, such that :math:`A = \sum_k  a_k P_k`.
    """
    # Ville Bergholm 2010

    d, v = eigsort(A)
    d = d.real  # A is assumed Hermitian

    # combine projectors for degenerate eigenvalues
    a = [d[0]]
    P = [projector(v[:, 0])]
    for k in range(1,len(d)):
        temp = projector(v[:, k])
        if abs(d[k] - d[k-1]) > tol:
            # new eigenvalue, new projector
            a.append(d[k])
            P.append(temp)
        else:
            # same eigenvalue, extend current P
            P[-1] += temp
    return array(a), P



# tensor bases

@copy_memoize
def gellmann(n):
    r"""Gell-Mann matrices of dimension n.

    Returns the n**2 - 1 (traceless, Hermitian) Gell-Mann matrices of dimension n,
    normalized such that :math:`\mathrm{Tr}(G_i^\dagger G_j) = \delta_{ij}`.
    """
    # Ville Bergholm 2006-2011

    if n < 1:
        raise ValueError('Dimension must be >= 1.')

    G = []
    # diagonal
    d = zeros(n)
    d[0] = 1
    for k in range(1, n):
        for j in range(0, k):
            # nondiagonal
            temp = zeros((n, n))
            temp[k,j] = 1 / sqrt(2)
            temp[j,k] = 1 / sqrt(2)
            G.append(temp)
  
            temp = zeros((n, n), dtype=complex)
            temp[k,j] = 1j / sqrt(2)
            temp[j,k] = -1j / sqrt(2)
            G.append(temp)

        d[k] = -sum(d)
        G.append(diag(d) / norm(d))
        d[k] = 1 

    return G


# TODO lazy evaluation/cache purging would be nice here to control memory usage
tensorbasis_cache = {}
def tensorbasis(n, d=None, get_locality=False):
    """Hermitian tensor-product basis for End(H).

    Returns a Hermitian basis for linear operators on the Hilbert space H
    which shares H's tensor product structure. The basis elements are tensor products
    of Gell-Mann matrices (which in the case of qubits are equal to Pauli matrices).
    The basis elements are normalized such that :math:`\mathrm{Tr}(b_i^\dagger b_j) = \delta_{ij}`.

    Input is either two scalars, n and d, in which case the system consists of n qu(d)its, :math:`H = C_d^{\otimes n}`,
    or the vector dim, which contains the dimensions of the individual subsystems:
    :math:`H = C_{dim[0]} \otimes ... \otimes C_{dim[n-1]}`.

    In addition to expanding Hermitian operators on H, this basis can be multiplied by
    the imaginary unit to obtain the antihermitian generators of U(prod(dim)).
    """
    # Ville Bergholm 2005-2011

    if d == None:
        # dim vector
        dim = n
        n = len(dim)
    else:
        # n qu(d)its
        dim = ones(n, int) * d

    # check cache first
    dim = tuple(dim)
    if dim in tensorbasis_cache:
        if get_locality:
            # tuple: (matrices, locality)
            return deepcopy(tensorbasis_cache[dim])
        else:
            # just the matrices
            return deepcopy(tensorbasis_cache[dim][0])

    n_elements = array(dim) ** 2    # number of basis elements for each subsystem, incl. identity
    n_all = prod(n_elements) # number of all tensor basis elements, incl. identity

    B = []
    locality = zeros(n_all, dtype = bool)  # logical array, is the corresponding basis element local?
    # create the tensor basis
    for k in range(n_all):  # loop over all basis elements
        inds = np.unravel_index(k, n_elements)
        temp = 1 # basis element being built
        nonid = 0  # number of non-id. matrices included in this element

        for j in range(n):  # loop over subsystems
            ind = inds[j]   # which local basis element to use
            d = dim[j]

            if ind > 0:
                nonid += 1 # using a non-identity matrix
                L = gellmann(d)[ind - 1]  # Gell-Mann basis vector for the subsystem
                # TODO gellmann copying the entire basis for a single matrix is inefficient...
            else:
                L = eye(d) / sqrt(d)  # identity (scaled)
            temp = kron(temp, L)  # tensor in another matrix

        B.append(temp)
        locality[k] = (nonid < 2) # at least two non-identities => nonlocal element

    # store into cache
    tensorbasis_cache[dim] = deepcopy((B, locality))
    if get_locality:
        return (B, locality)
    else:
        return B



# misc

def op_list(G, dim):
    """Operator consisting of k-local terms, given as a list.

    Returns the operator O defined by the connection list G.
    dim is a vector of subsystem dimensions for O.
    G is a list of arrays, :math:`G = [c_1, c_2, ..., c_n]`,
    where each array :math:`c_i` corresponds to a term in O.

    An array that has 2 columns and k rows, :math:`c_i` = [(A1, s1), (A2, s2), ... , (Ak, sk)],
    where Aj are operators and sj subsystem indices, corresponds to the
    k-local term given by the tensor product

    .. math::

       A1_{s1} * A2_{s2} * ... * Ak_{sk}.

    The dimensions of all operators acting on subsystem sj must match dim[sj].

    Alternatively one can think of G as defining a hypergraph, where
    each subsystem corresponds to a vertex and each array c_i in the list
    describes a hyperedge connecting the vertices {s1, s2, ..., sk}.

    Example: The connection list
    G = [[(sz,1)], [(sx,1), (sx,3)], [(sy,1), (sy,3)], [(sz,1), (sz,3)], [(sz,2)], [(A,2), (B+C,3)], [(2*sz,3)]]
    corresponds to the operator

    .. math::

       \sigma_{z1} +\sigma_{z2} +2 \sigma_{z3} +\sigma_{x1} \sigma_{x3} +\sigma_{y1} \sigma_{y3} +\sigma_{z1} \sigma_{z3} +A_2 (B+C)_3.

    """
    # Ville Bergholm 2009-2010

    # TODO we could try to infer dim from the operators
    H = 0
    for spec in G:
        a = -1  # last subsystem taken care of
        term = 1
        for j in spec:
            if len(j) != 2:
                raise ValueError('Malformed term spec {0}.'.format(k))

            b = j[1]  # subsystem number
            if (b <= a):
                raise ValueError('Spec {0} not in ascending order.'.format(k))

            if j[0].shape[1] != dim[b]:
                raise ValueError('The dimension of operator {0} in spec {1} does not match dim.'.format(j, k))

            term = mkron(term, eye(prod(dim[a+1:b])), j[0])
            a = b

        # final identity
        term = mkron(term, eye(prod(dim[a+1:])))
        H += term
    return H


def qubits(n):
    """Dimension vector for an all-qubit system.
    
    For the extemely lazy, returns (2,) * n
    """
    # Ville Bergholm 2010

    return (2,) * n


def majorize(x, y):
    """Majorization partial order of real vectors.

    Returns true iff the real vector x is majorized by the real vector y,
    denoted by :math:`x \preceq y`. This is equivalent to
    .. math::

       \sum_{k=1}^n x^{\downarrow}_k \le \sum_{k=1}^n y^{\downarrow}_k` \quad \text{for all} \quad n \in \{1, 2, \ldots, d\},

    where d is the dimension of the vectors x and y, and :math:`x^{\downarrow}` is the vector x with the elements sorted in nonincreasing order.

    :math:`x \preceq y` if and only if x is in the convex hull of all the coordinate permutations of y.
    """
    # Ville Bergholm 2010-2012

    if x.ndim != 1 or y.ndim != 1 or np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError('Inputs must be real vectors.')

    if len(x) != len(y):
        raise ValueError('The vectors must be of equal length.')

    x = cumsum(sort(x)[::-1])
    y = cumsum(sort(y)[::-1])

    if abs(x[-1] -y[-1]) <= tol:
        # exact majorization
        return all(x-y <= tol)
    else:
        # weak majorization could still be possible, but...
        warn('Vectors have unequal sums.')
        return False


def mkron(*arg):
    """This is how kron should work, dammit.

    Returns the tensor (Kronecker) product :math:`X = A \otimes B \otimes \ldots`
    """
    # Ville Bergholm 2009

    X = 1
    for A in arg:
        X = kron(X, A)
    return X


def test():
    """Test script for the utils module.
    """
    # Ville Bergholm 2009-2011

    dim = 10

    # expv
    A = randn(dim, dim) + 1j * randn(dim, dim)
    H = rand_hermitian(dim)
    v = randn(dim) + 1j * randn(dim)

    # arbitrary matrix with Arnoldi iteration
    w, err, hump = expv(1, A, v, m = dim // 2)
    assert_o(norm(w - dot(expm(1*A), v)), 0, 1e2*tol)
    w, err, hump = expv(1, A, v, m = dim)  # force a happy breakdown
    assert_o(norm(w - dot(expm(1*A), v)), 0, 1e2*tol)

    # Hermitian matrix with Lanczos iteration
    w, err, hump = expv(1, H, v, m = dim // 2, iteration = 'lanczos')
    assert_o(norm(w - dot(expm(1*H), v)), 0, 1e2*tol)
    w, err, hump = expv(1, H, v, m = dim, iteration = 'lanczos')
    assert_o(norm(w - dot(expm(1*H), v)), 0, 1e2*tol)

    # FIXME why does Lanczos work with nonhermitian matrices?
    w, err, hump = expv(1, A, v, m = dim // 2, iteration = 'lanczos')
    assert_o(norm(w - dot(expm(1*A), v)), 0, 1e2*tol)
    w, err, hump = expv(1, A, v, m = dim, iteration = 'lanczos')
    assert_o(norm(w - dot(expm(1*A), v)), 0, 1e2*tol)


    dim = 5

    # random matrices
    H = mat(rand_hermitian(dim))
    assert_o(norm(H - H.H), 0, tol) # hermitian

    U = mat(rand_U(dim))
    assert_o(norm(U * U.H -eye(dim)), 0, tol) # unitary
    assert_o(norm(U.H * U -eye(dim)), 0, tol)

    U = mat(rand_SU(dim))
    assert_o(norm(U * U.H -eye(dim)), 0, tol) # unitary
    assert_o(norm(U.H * U -eye(dim)), 0, tol)
    assert_o(det(U), 1, tol) # det 1

    rho = mat(rand_positive(dim))
    assert_o(norm(rho - rho.H), 0, tol) # hermitian
    assert_o(trace(rho), 1, tol) # trace 1
    temp = eigvalsh(rho)
    assert_o(norm(temp.imag), 0, tol) # real eigenvalues
    assert_o(norm(temp - abs(temp)), 0, tol) # nonnegative eigenvalues


    # superoperators
    L = mat(rand_U(dim))
    R = mat(rand_U(dim))
    v = vec(array(rho))

    assert_o(norm(rho -inv_vec(v)), 0, tol)
    assert_o(norm(L*rho*R -inv_vec(dot(lrmul(L, R), v))), 0, tol)
    assert_o(norm(L*rho -inv_vec(dot(lmul(L), v))), 0, tol)
    assert_o(norm(rho*R -inv_vec(dot(rmul(R), v))), 0, tol)


    # physical operators
    J = angular_momentum(dim)
    assert_o(norm(comm(J[0], J[1]) - 1j*J[2]), 0, tol)  # [Jx, Jy] == i Jz

    a = mat(boson_ladder(dim))
    temp = comm(a, a.H)
    assert_o(norm(temp[:-1,:-1] - eye(dim-1)), 0, tol)  # [a, a'] == I  (truncated, so skip the last row/col!)

    f = fermion_ladder(2)
    temp = f[0].conj().transpose()
    assert_o(norm(acomm(f[0], f[0]) ), 0, tol)  # {f_j, f_k} = 0
    assert_o(norm(acomm(f[0], f[1]) ), 0, tol)
    assert_o(norm(acomm(temp, f[0]) -eye(4)), 0, tol)  # {f_j^\dagger, f_k} = I \delta_{jk}
    assert_o(norm(acomm(temp, f[1]) ), 0, tol)


    # SU(2) rotations


    # spectral decomposition
    E, P = spectral_decomposition(H)
    temp = 0
    for k in range(len(E)):
        temp += E[k] * P[k]

    assert_o(norm(temp - H), 0, tol)


    # tensor bases


    # majorization

    # op_list

    # plots
