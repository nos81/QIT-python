# -*- coding: utf-8 -*-
# Author: Ville Bergholm 2011
"""Harmonic oscillator module."""

from __future__ import print_function, division

from numpy import array, mat, arange, sqrt, zeros, ones, prod, sqrt, pi, isscalar, linspace, newaxis
from scipy import factorial
from scipy.linalg import expm, norm

from state import *
from utils import boson_ladder


# default truncation limit for number states
default_n = 30

def coherent_state(alpha, n=default_n):
    """Coherent states of a harmonic oscillator.

    Returns the n-dimensional approximation to the
    coherent state \ket{\alpha} in the number basis.

    Ville Bergholm 2010
    """
    k = arange(n)
    ket = (alpha ** k) / sqrt(factorial(k))
    return state(ket, n).normalize()
    #s = state(0, n).u_propagate(expm(alpha * mat(boson_ladder(n)).H))
    #s = state(0, n).u_propagate(displace(alpha, n))
    #s *= exp(-abs(alpha) ** 2 / 2) # normalization


def displace(alpha, n=default_n):
    """Bosonic displacement operator.

    Returns the n-dimensional approximation for the bosonic
    displacement operator D(alpha) in the number basis {|0>, |1>, ..., |n-1>}.

    Ville Bergholm 2010
    """
    if not isscalar(alpha):
        raise TypeError('alpha must be a scalar.')

    a = mat(boson_ladder(n))
    return array(expm(alpha * a.H -alpha.conjugate() * a))


def squeeze(z, n=default_n):
    """Bosonic squeezing operator.

    Returns the n-dimensional approximation for the bosonic
    squeezing operator S(z) in the number basis {|0>, |1>, ..., |n-1>}.

    Ville Bergholm 2010
    """
    if not isscalar(z):
        raise TypeError('z must be a scalar.')

    a = mat(boson_ladder(n))
    return array(expm(0.5 * (z.conjugate() * (a ** 2) - z * (a.H ** 2))))


def position(n=default_n):
    """Position operator.

    Returns the n-dimensional approximation of the
    dimensionless position operator Q in the number basis.

    Q = \sqrt{\frac{m \omega}{\hbar}}   q =    (a+a')/sqrt(2)
    P = \sqrt{\frac{1}{m \hbar \omega}} p = -i*(a-a')/sqrt(2)

    [q, p] = i \hbar,  [Q, P] = i

    H = \frac{p^2}{2m} +\frac{1}{2} m \omega^2 q^2
      = 0.5 \hbar \omega (P^2 +Q^2)
      = \hbar \omega (a'*a +\frac{1}{2})

    Ville Bergholm 2010
    """
    a = mat(boson_ladder(n))
    return array(a + a.H) / sqrt(2)


def momentum(n=default_n):
    """Momentum operator.

    Returns the n-dimensional approximation of the
    dimensionless momentum operator P in the number basis.

    Q = \sqrt{\frac{m \omega}{\hbar}}   q =    (a+a')/sqrt(2)
    P = \sqrt{\frac{1}{m \hbar \omega}} p = -i*(a-a')/sqrt(2)

    [q, p] = i \hbar,  [Q, P] = i

    H = \frac{p^2}{2m} +\frac{1}{2} m \omega^2 q^2
      = 0.5 \hbar \omega (P^2 +Q^2)
      = \hbar \omega (a'*a +\frac{1}{2})

    Ville Bergholm 2010
    """
    a = mat(boson_ladder(n))
    return -1j*array(a - a.H) / sqrt(2)


def position_state(q, n=default_n):
    """Position eigenstates of a harmonic oscillator.

    Returns the n-dimensional approximation of the eigenstate |q>
    of the dimensionless position operator Q in the number basis.

    See position, momentum.

    Difference equation:
      r_1 = \sqrt{2} q r_0
      \sqrt{k+1} r_{k+1} = \sqrt{2} q r_k -\sqrt{k} r_{k-1}, when k >= 1

    Ville Bergholm 2010
    """
    ket = zeros(n, dtype=complex)
    temp = sqrt(2) * q
    ket[0] = 1  # arbitrary nonzero initial value r_0
    ket[1] = temp * ket[0]
    for k in range(2, n):
        ket[k] = temp/sqrt(k) * ket[k - 1] -sqrt((k-1) / k) * ket[k - 2]
    ket /= norm(ket)  # normalize
    return state(ket, n)


def momentum_state(p, n=default_n):
    """Momentum eigenstates of a harmonic oscillator.

    Returns the n-dimensional approximation of the eigenstate |p>
    of the dimensionless momentum operator P in the number basis.

    See position, momentum.

    Difference equation:
      r_1 = i \sqrt{2} p r_0
      \sqrt{k+1} r_{k+1} = i \sqrt{2} p r_k +\sqrt{k} r_{k-1}, when k >= 1

    Ville Bergholm 2010
    """
    ket = zeros(n, dtype=complex)
    temp = 1j * sqrt(2) * p
    ket[0] = 1  # arbitrary nonzero initial value r_0
    ket[1] = temp * ket[0]
    for k in range(2, n):
        ket[k] = temp/sqrt(k) * ket[k - 1] +sqrt((k-1) / k) * ket[k - 2]
    ket /= norm(ket)  # normalize
    return state(ket, n)


def husimi(s, alpha=None, z=0, res=(40, 40), lim=(-2, 2, -2, 2)):
    """Husimi probability distribution.
    H = husimi(s, alpha)
    [H, a, b] = husimi(s, res=xxx, lim=yyy)

    Returns the Husimi probability distribution
    H(Im \alpha, Re \alpha) corresponding to the harmonic
    oscillator state s given in the number basis.

    z is the optional squeezing parameter for the reference state:
      |\alpha, z> = D(\alpha) S(z) |0>

    H(s, \alpha, z) =  1/\pi <\alpha, z| \rho_s |\alpha, z>

    The integral of H is normalized to unity.

    Ville Bergholm 2010
    """
    if alpha == None:
        # return a 2D grid of W values
        a = linspace(lim[0], lim[1], res[0])
        b = linspace(lim[2], lim[3], res[1])
        #a, b = ogrid[lim[0]:lim[1]:1j*res[0], lim[2]:lim[3]:1j*res[1]]
        alpha = a + 1j*b[:, newaxis]
        return_ab = True
    else:
        return_ab = False

    # reference state
    n = prod(s.dims())
    ref = state(0, n).u_propagate(squeeze(z, n))
    ref /= sqrt(pi) # normalization included for convenience

    H = zeros(alpha.shape)
    for k, c in enumerate(alpha.flat):
        temp = ref.u_propagate(displace(c, n))
        H.flat[k] = s.fidelity(temp) ** 2

    if return_ab:
        H = (H, a, b)
    return H


def wigner(s, alpha=None, res=(20, 20), lim=(-2, 2, -2, 2)):
    """Wigner quasi-probability distribution.
    W = wigner(s, alpha)
    W, a, b = wigner(s, res=xxx, lim=yyy)

    Returns the Wigner quasi-probability distribution
    W(Im \alpha, Re \alpha) corresponding to the harmonic
    oscillator state s given in the number basis.

    For a normalized state, the integral of W is normalized to unity.

    NOTE: The truncation of the number state space to a finite dimension
    results in spurious circular ripples in the Wigner function outside
    a given radius. To increase the accuracy, increase the state space dimension.

    Ville Bergholm 2010
    """
    # B = np.broadcast(s, alpha) , (st,a) = B.next() TODO
    if alpha == None:
        # return a grid of W values for a grid of alphas
        a = linspace(lim[0], lim[1], res[0])
        b = linspace(lim[2], lim[3], res[1])
        #a, b = ogrid[lim[0]:lim[1]:1j*res[0], lim[2]:lim[3]:1j*res[1]]
        alpha = a + 1j*b[:, newaxis]
        return_ab = True
    else:
        return_ab = False

    # parity operator (diagonal)
    n = prod(s.dims())
    P = ones(n)
    P[1:n:2] = -1
    P *= 2 / pi  # include Wigner normalization here for convenience

    W = zeros(alpha.shape)
    for k, c in enumerate(alpha.flat):
        temp = s.u_propagate(displace(-c, n))
        W.flat[k] = sum(P * temp.prob().real) # == ev(temp, P).real

    if return_ab:
        W = (W, a, b)
    return W



def test():
    """Testing script for the harmonic oscillator module."""
    from numpy.random import randn
    def randc():
        """Random complex number."""
        return randn() + 1j*randn()

    a = mat(boson_ladder(default_n))

    alpha = randc()
    s = coherent_state(alpha)
    s0 = state(0, default_n)
    D = displace(alpha)
    
    assert_o((s - s0.u_propagate(D)).norm(), 0, tol)  # displacement

    z = randc()
    S = squeeze(z)

    Q = position(); P = momentum()
    q = randn(); p = randn()
    sq = position_state(q)
    sp = momentum_state(p)

    temp = 1e-1 # the truncation accuracy is not amazing here
    assert_o(sq.ev(Q), q, temp)  # Q, P eigenstates
    assert_o(sp.ev(P), p, temp)

    temp = ones(default_n);  temp[-1] = -default_n+1 # truncation...
    assert_o(norm(comm(Q,P) - 1j * diag(temp)), 0, tol) # [Q, P] = i

    assert_o(norm(mat(P)**2 +mat(Q)**2 - 2 * a.H * a -diag(temp)), 0, tol)  # P^2 +Q^2 = 2a' * a + 1
