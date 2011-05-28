# -*- coding: utf-8 -*-
# Author: Ville Bergholm 2011
"""Utility functions module."""

from __future__ import print_function, division
from copy import deepcopy

import numpy as np
from numpy import array, mat, zeros, ones, eye, prod, sqrt, exp, tanh, dot, sort, diag, trace, kron, pi, r_, c_
from numpy.random import rand, randn, randint
from numpy.linalg import qr, det, eig, eigvals
from scipy.linalg import expm, norm, svdvals
import matplotlib.pyplot as plt

from base import *

# As a rule the functions in this module return ndarrays, not lmaps.


# internal utilities

def assert_o(actual, desired, tolerance):
    """Octave-style assert."""
    if abs(actual - desired) > tolerance:
        raise AssertionError


def gcd(a, b):
    """Greatest common divisor.

    Euclidean algorithm.
    From NumPy source, why isn't this in the API?"""
    while b:
        a, b = b, a%b
    return a


def lcm(a, b):
    """Least common multiple.
    """
    return a * (b // gcd(a, b))


def rank(A, eps=1e-8):
    """Matrix rank."""
    s = svdvals(A)
    return sum(s > eps)


def projector(v):
    """Projector corresponding to vector v."""
    return np.outer(v, v.conj())


def eigsort(A):
    """Returns eigenvalues and eigenvectors sorted with a nonincreasing real part."""
    d, v = eig(A)
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



# random matrices

def rand_hermitian(n):
    """Random Hermitian n*n matrix.

    Returns a random Hermitian matrix of size n*n.
    NOTE: The randomness is not defined in any deeply meaningful sense.

    Ville Bergholm 2008-2009
    """
    H = (rand(n,n) - 0.5) +1j*(rand(n,n) - 0.5)
    return H + H.conj().transpose() # make it Hermitian


def rand_U(n):
    """Random U(n) matrix.

    Returns a random unitary n*n matrix.
    The matrix is random with respect to the Haar measure.

    %! F. Mezzadri, "How to generate random matrices from the classical compact groups", Notices of the AMS 54, 592 (2007). arXiv.org:math-ph/0609050
    Ville Bergholm 2005-2009
    """
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

    Ville Bergholm 2005-2009
    """
    U = rand_U(n)
    d = det(U) ** (1/n) # *exp(i*2*pi*k/n), not unique FIXME
    return U/d


def rand_U1(n):
    """Random diagonal unitary matrix.

    Returns a random diagonal unitary n*n matrix.
    The matrix is random with respect to the Haar measure.

    Ville Bergholm 2005-2009
    """
    return diag(exp(2j * pi * rand(n)))



# randi(n) == np.random.random_integers(n)

def rand_positive(n):
    """Random n*n positive semidefinite matrix.

    Normalized as trace(A) = 1.
    Since the matrix has purely real eigenvalues, it is also
    Hermitian by construction.

    Ville Bergholm 2008-2009
    """
    p = sort(rand(n-1))  # n-1 points in [0,1]
    d = sort(np.r_[p, 1] - np.r_[0, p])  # n deltas between points = partition of unity

    U = mat(rand_U(n)) # random unitary
    A = U.H * diag(d) * U
    return array((A + A.H) / 2)   # eliminate rounding errors



# superoperators

def vec(rho):
    """Flattens a matrix into a vector.

    Matrix rho is flattened columnwise into a column vector v.

    Used e.g. to convert state operators to superoperator representation.

    JDW 2009
    Ville Bergholm 2009
    """
    return rho.flatten('F')  # copy


def inv_vec(v, dim=None):
    """Reshapes a vector into a matrix.
    rho = inv_vec(v) 
    rho = inv_vec(v, [n, m]) 

    Reshapes vector v (length n*m) into a matrix rho (size [n, m]),
    using column-major ordering. If n and m are not given, rho is assumed
    to be square.

    Used e.g. to convert state operators from superoperator representation
    to standard matrix representation.

    JDW 2009
    Ville Bergholm 2009
    """
    d = v.size
    if dim == None:
        # assume a square matrix
        n = sqrt(d)
        if np.floor(n) != n:
            raise ValueError('Length of vector v is not a squared integer.')
        dim = (n, n)
    else:
        if prod(dim) != d:
            raise ValueError('Dimensions n, m are not compatible with given vector v.')
    return v.reshape(dim, order='F')


def lmul(L, q=None):
    """Superoperator equivalent for multiplying from the left.

    L*rho == inv_vec(lmul(L)*vec(rho))

    Dimensions: L is [m, p], rho is [p, q].
    If q is not given rho is assumed square.

    Ville Bergholm 2009
    """
    if q == None:
        q = L.shape[1]  # assume target is a square matrix
    return kron(eye(q), L)


def rmul(R, p=None):
    """Superoperator equivalent for multiplying from the right.

    rho*R == inv_vec(rmul(R)*vec(rho))

    Dimensions: rho is [p, q], R is [q, r].
    If p is not given rho is assumed square.

    Ville Bergholm 2009
    """
    if p == None:
        p = R.shape[0]  # assume target is a square matrix
    return kron(R.transpose(), eye(p))


def lrmul(L, R):
    """Superoperator equivalent for multiplying both from left and right.

    L*rho*R == inv_vec(lrmul(L, R)*vec(rho))

    Ville Bergholm 2009-2010
    """
    # L and R fix the shape of rho completely
    return kron(R.transpose(), L)


def superop_lindblad(A, H=0):
    """Liouvillian superoperator for a set of Lindblad operators.

    A is a vector of traceless, orthogonal Lindblad operators.
    H is an optional Hamiltonian operator.

    Returns the Liouvillian superoperator L corresponding to the
    diagonal-form Lindblad equation

      \dot{\rho} = inv_vec(L * vec(\rho)) =
      = -i [H, \rho] +\sum_k (A_k \rho A_k^\dagger -0.5*\{A_k^\dagger A_k, \rho\})

    James D. Whitfield 2009
    Ville Bergholm 2009-2010
    """
    # Hamiltonian
    iH = 1j * H

    L = 0
    acomm = 0
    for k in A:
        acomm += 0.5 * k.conj().transpose() * k
        L += lrmul(k, k.conj().transpose()) 

    L += lmul(-acomm -iH) +rmul(-acomm +iH)
    return L



# physical operators

@copy_memoize
def angular_momentum(n):
    """Angular momentum matrices.

    (Jx, Jy, Jz) = angular_momentum(d)

    Returns a 3-tuple of angular momentum matrices \vec(J)/\hbar
    for the d-dimensional subspace defined by the
    quantum number j == (d-1)/2.

    Ville Bergholm 2009-2010
    """
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
    """Bosonic ladder operators.

    Returns the n-dimensional approximation of the bosonic
    annihilation operator b for a single bosonic mode in the
    number basis {|0>, |1>, ..., |n-1>}.

    The corresponding creation operator is b.ctranspose().

    Ville Bergholm 2009-2010
    """
    return diag(sqrt(range(1, n)), 1)


@copy_memoize
def fermion_ladder(grouping):
    """Fermionic ladder operators.

    Returns a vector of fermionic annihilation operators for a
    system of n fermionic modes in the second quantization.

    The annihilation operators are built using the Jordan-Wigner
    transformation for a chain of n qubits, where the state of each
    qubit denotes the occupation number of the corresponding mode.

    First define annihilation and number operators for a lone fermion mode:
    s := (sx + i*sy)/2   = [[0, 1], [0, 0]],
    n := s'*s = (I-sz)/2 = [[0, 0], [0, 1]].

    s|0> = 0, s|1> = |0>, n|k> = k|k>

    Then define a phase operator to keep track of sign changes when
    permuting the order of the operators:
    \phi_k := \sum_{j=0}^{k-1} n_j.

    Now, the fermionic annihilation operators for the n-mode system are given by
    f_k := (-1)^{\phi_k} s_k.

    These operators fulfill the required anticommutation relations:
    {f_k, f_j}  = 0,
    {f_k, f_j'} = I \delta_{kj},
    f_k' * f_k  = n_k.

    Ville Bergholm 2009-2010
    """
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
    f = np.empty(grouping, object)
    # annihilation operators for a set of fermions (Jordan-Wigner transform)
    for k in range(n):
        f.flat[k] = ((-1) ** array(phi[k])) * mkron(eye(2 ** k), s, eye(2 ** (n-k-1)))
    return f



# SU(2) rotations

def R_nmr(theta, phi):
    """SU(2) rotation \theta_\phi (NMR notation).

    Returns the one-qubit rotation by angle theta about the unit
    vector [cos(phi), sin(phi), 0], or \theta_\phi in the NMR notation.

    Ville Bergholm 2009
    """
    return expm(-1j * theta/2 * (cos(phi) * sx + sin(phi) * sy))


def R_x(theta):
    """SU(2) x-rotation.

    Returns the one-qubit rotation about the x axis by the angle theta,
    e^(-i \sigma_x theta/2).

    Ville Bergholm 2006-2009
    """
    return expm(-1j * theta/2 * sx)


def R_y(theta):
    """SU(2) y-rotation.

    Returns the one-qubit rotation about the y axis by the angle theta,
    e^(-i \sigma_y theta/2).

    Ville Bergholm 2006-2009
    """
    return expm(-1j * theta/2 * sy)


def R_z(theta):
    """SU(2) z-rotation.

    Returns the one-qubit rotation about the z axis by the angle theta,
    e^(-i \sigma_z theta/2).

    Ville Bergholm 2006-2009
    """
    return array([[exp(-1j * theta/2), 0], [0, exp(1j * theta/2)]])



# decompositions

def spectral_decomposition(A):
    """Spectral decomposition of a Hermitian matrix.

    Returns the unique eigenvalues a and the corresponding projectors P
    for the Hermitian matrix A, such that  A = \sum_k  a_k P_k.

    Ville Bergholm 2010
    """
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
    return a, P



# tensor bases

@copy_memoize
def gellmann(n):
    """Gell-Mann matrices of dimension n.

    Returns the n^2-1 (traceless, Hermitian) Gell-Mann matrices of dimension n,
    normalized such that \trace(G_i.ctranspose() * G_j) = \delta_{ij}.

    Ville Bergholm 2006-2011
    """
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
    B = tensorbasis(n, d)   H = C_d^{\otimes n}.
    B = tensorbasis(dim)    H = C_{dim(1)} \otimes ... \otimes C_{dim(n)}

    Returns a Hermitian basis for linear operators on the Hilbert space H
    which shares H's tensor product structure. The basis elements are tensor products
    of Gell-Mann matrices (which in the case of qubits are equal to Pauli matrices).
    The basis elements are normalized such that \trace(b_i' * b_j) = \delta_{ij}.

    Input is either two scalars, n and d, in which case the system consists of n qu(d)its,
    or the vector dim, which contains the dimensions of the individual subsystems.

    In addition to expanding Hermitian operators on H, this basis can be multiplied by
    the imaginary unit i to obtain the antihermitian generators of U(prod(dim)).

    Ville Bergholm 2005-2011
    """
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



# plots

def plot_bloch_sphere(s=None):
    """Bloch sphere plot.

    Plots a Bloch sphere, a geometrical representation of the state space of a single qubit.
    Pure states are on the surface of the sphere, nonpure states inside it.
    The states |0> and |1> lie on the north and south poles of the sphere, respectively.

    s is a two dimensional state to be plotted.

    Ville Bergholm  2005-2010
    James Whitfield 2010
    """
    X,Y,Z = sphere(40)

    hold('off')
    h = surf(X,Y,Z, 2*ones(41,41))
    hold(on)
    shading('flat')
    alpha(0.2)
    axis('square')
    xlabel('x')
    ylabel('y')
    zlabel('z')
    plot3(0,0,1,'r.')
    plot3(0,0,-1,'b.')

    text(0, 0,  1.2, '$|0\rangle$')
    text(0, 0, -1.2, '$|1\rangle$')

    if s != None:
        v = s.bloch_vector()
        quiver3(0, 0, 0, v(1), v(2), v(3), 0)
    return h


def plot_pcolor(W, a, b, clim=(0, 1)):
    """Easy pseudocolor plot.

    Plots the 2D function given in the matrix W.
    The vectors x and y define the coordinate grid.
    clim is an optional parameter for color limits.

    Returns the plot object.

    Ville Bergholm 2010
    """
    # a and b are quad midpoint coordinates but pcolor wants quad vertices, so
    def to_quad(x):
        return (r_[x, x[-1]] + r_[x[0], x]) / 2

    plt.gcf().clf()  # clear the figure
    p = plt.pcolor(to_quad(a), to_quad(b), W, clim = clim, cmap = asongoficeandfire())
    plt.axis('equal')
    plt.axis('tight')
    #shading('interp')
    plt.colorbar()
    return p


def plot_adiabatic_evolution(t, st, H_func, n=4):
    """Adiabatic evolution plot.

    Input: vector t of time instances, cell vector st of states corresponding
    to the times and time-dependant Hamiltonian function handle H_func.

    Plots the energies of the eigenstates of H_func(t(k)) as a function of t(k),
    and the overlap of st{k} with the n lowest final Hamiltonian eigenstates. 
    Useful for illustrating adiabatic evolution.

    Jacob D. Biamonte 2008
    Ville Bergholm 2009-2010
    """
    T = t[-1]  # final time
    H = H_func(T)

    n = min(n, H.shape[0])
    m = len(t)

    # find the n lowest eigenstates of the final Hamiltonian
    d, v = scipy.sparse.linalg.eigs(H, n, which = 'SR')
    ind = d.argsort()  # increasing real part
    lowest = []
    for j in range(n):
        lowest.append(state(v[:, ind[j]]))
    # TODO with degenerate states these are more or less random linear combinations of the basis states... overlaps are not meaningful

    overlaps = zeros((n, m))
    for k in range(m):
        tt = t[k]
        H = H_func(tt)
        energies[:, k] = sort(real(eig(full(H))), 'ascend')
        for j in range(n):
            overlaps[j, k] = lowest[j].fidelity(st[k]) ** 2 # squared overlap with lowest final states

    plt.subplot(2,1,1)
    plt.plot(t/T, energies)
    plt.grid(True)
    plt.title('Energy spectrum')
    plt.xlabel('Adiabatic time')
    plt.ylabel('Energy')
    plt.axis([0, 1, min(energies), max(energies)])


    plt.subplot(2,1,2)
    plt.plot(t/T, overlaps) #, 'LineWidth', 1.7)
    plt.grid(True)
    plt.title('Squared overlap of current state and final eigenstates')
    plt.xlabel('Adiabatic time')
    plt.ylabel('Probability')
    temp = []
    for k in range(n):
        temp.append('|{0}$\\rangle$'.format(k))
    plt.legend(temp)
    plt.axis([0, 1, 0, 1])
    # axis([0, 1, 0, max(overlaps)])


def makemovie(filename, frameset, plot_func, *arg):
    """Create an AVI movie.
    aviobj = makemovie(filename, frameset, plot_func [, ...])

    Creates an AVI movie file named 'filename.avi' in the current directory.
    Frame k in the movie is obtained from the contents of the
    current figure after calling plot_func(frameset[k]).
    The optional extra parameters are passed directly to avifile.

    Returns the closed avi object handle.

    Example: makemovie('test', cell_vector_of_states, @(x) plot(x))

    James D. Whitfield 2009
    Ville Bergholm 2009-2010
    """
    # create an AVI object
    aviobj = avifile(filename, arg)

    fig = figure('Visible', 'off')
    for k in frameset:
        plot_func(k)
        aviobj = addframe(aviobj, fig)
        #  F = getframe(fig)   
        #  aviobj = addframe(aviobj, F)

    close(fig)
    aviobj = close(aviobj)



# misc

def op_list(G, dim):
    """Operator consisting of k-local terms, given as a list.

    Returns the operator O defined by the connection list G.
    dim is a vector of subsystem dimensions for O.

    G is a list of arrays, G = [c_1, c_2, ..., c_n],
    where each array c_i corresponds to a term in O.

    An array that has 2 columns and k rows, c_i = [(A1, s1), (A2, s2), ... , (Ak, sk)],
    where Aj are operators and sj subsystem indices, corresponds to the
    k-local term given by the tensor product

      A1_{s1} * A2_{s2} * ... * Ak_{sk}.

    The dimensions of all operators acting on subsystem sj must match dim(sj).

    Alternatively one can think of G as defining a hypergraph, where
    each subsystem corresponds to a vertex and each array c_i in the list
    describes a hyperedge connecting the vertices {s1, s2, ..., sk}.

    Example: The connection list
    G = [[(sz,1)], [(sx,1), (sx,3)], [(sy,1), (sy,3)], [(sz,1), (sz,3)],
         [(sz,2)], [(A,2), (B+C,3)], [(2*sz,3)]]

    corresponds to the operator
    O = sz_1 +sz_2 +2*sz_3 +sx_1*sx_3 +sy_1*sy_3 +sz_1*sz_3 +A_2*(B+C)_3.

    Ville Bergholm 2009-2010
    """
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


def asongoficeandfire(n=127):
    """Colormap with blues and reds. Wraps.

    Returns a matplotlib.colors.Colormap object.
    n is the number of color definitions in the map.

    Ville Bergholm 2010-2011
    """
    from matplotlib import colors
    # exponent
    d = 3.1
    p = np.linspace(-1, 1, n)
    # negative values: reds
    x = p[p < 0]
    c = c_[1 -((1+x) ** d), 0.5*(tanh(4*(-x -0.5)) + 1), (-x) ** d]
    # positive values: blues
    x = p[p >= 0]
    c = r_[c, c_[x ** d, 0.5*(tanh(4*(x -0.5)) + 1), 1 -((1-x) ** d)]]
    return colors.ListedColormap(c, name='asongoficeandfire')
    # TODO colors.LinearSegmentedColormap(name, segmentdata, N=256, gamma=1.0)


def qubits(n):
    """Dimension vector for an all-qubit system.
    
    For the extemely lazy, returns (2,) * n

    Ville Bergholm 2010
    """
    return (2,) * n


def majorize(x, y):
    """Majorization partial order of real vectors.

    Returns true iff vector x is majorized by vector y,
    i.e. res = x \preceq y.

    Ville Bergholm 2010
    """
    if x.ndim != 1 or y.ndim != 1 or np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError('Inputs must be real vectors.')

    if len(x) != len(y):
        raise ValueError('The vectors must be of equal length.')

    x = cumsum(sort(x)[::-1])
    y = cumsum(sort(y)[::-1])

    if abs(x[-1] -y[-1]) < tol:
        # exact majorization
        return all(x <= y)
    else:
        # weak majorization could still be possible, but...
        warn('Vectors have unequal sums.')
        return False


def mkron(*arg):
    """This is how kron should work, dammit.

    Returns the tensor (Kronecker) product X = A \otimes B \otimes ...

    Ville Bergholm 2009
    """
    X = 1
    for A in arg:
        X = kron(X, A)
    return X


def test():
    """Test script for the utils module.

    Ville Bergholm 2009-2011
    """
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
    temp = eigvals(rho)
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
