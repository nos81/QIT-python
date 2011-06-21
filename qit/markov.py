# -*- coding: utf-8 -*-
# Author: Ville Bergholm 2011
"""Born-Markov noise module."""

from __future__ import print_function, division

from numpy import array, sqrt, exp, sin, cos, arctan2, tanh, dot, argsort, pi, r_, linspace, logspace, searchsorted, inf, newaxis, isscalar, unravel_index
from scipy.linalg import norm
from scipy.integrate import quad
import scipy.constants as const

from base import *
from utils import *


class bath(object):
    """Markovian heat bath.

    Currently only one type of bath is supported, a bosonic
    canonical ensemble at absolute temperature T, with a
    single-term coupling to the system.

    The bath spectral density is Ohmic with a cutoff.
      J(omega) = hbar^2 * omega * cut(omega) * heaviside(omega)

    Two types of cutoffs are supported: exponential and sharp.
      cut(omega) = exp(-omega / omega_c)
      cut(omega) = heaviside(omega_c - omega)

      gamma(omega) == 2 * pi / hbar^2 (J(omega) -J(-omega))(1 + n(omega))

    Data members:
    type     Bath type. Currently only 'ohmic' is supported.
    omega0   hbar * omega0 is the unit of energy for all Hamiltonians (omega0 in Hz)
    T        Absolute temperature of the bath (in K).
    scale    Dimensionless temperature scaling parameter hbar * omega0 / (kB * T)
    cut_type    spectral density cutoff type
    cut_limit   spectral density cutoff angular frequency/omega0
    
    # spectral density:
    # J(omega0 * x)/omega0 = \hbar^2 * j(x) * heaviside(x) * cut_func(x)
    j          spectral density profile
    cut_func   cutoff function

    # functions for g and s
    # gamma(omega0 * x) / omega0 = g_func(x) * cut_func(x)
    #    S(omega0 * dH) / omega0 = S_func(dH)
    g_func
    g0
    s0

    # lookup tables for g and s
    # gamma(omega0 * dH[k]) / omega0 = gs_table[k, 0]
    #     S(omega0 * dH[k]) / omega0 = gs_table[k, 1]
    dH
    gs_table

    Ville Bergholm 2009-2011
    """
    def __init__(self, type, omega0, T):
        """constructor
    
        Sets up a descriptor for a heat bath coupled to a quantum system.
        """
        # basic bath parameters
        self.type   = type
        self.omega0 = omega0
        self.T      = T
        # shorthand
        self.scale = const.hbar * omega0 / (const.k * T)

        if type == 'ohmic':
            # Ohmic bosonic bath, canonical ensemble, single-term coupling
            self.g_func = lambda x: 2 * pi * x * (1 + 1 / (exp(self.scale * x) - 1))
            self.g0 = 2 * pi / self.scale  # limit of g at x == 0
            self.j = lambda x: x
        else:
            raise ValueError('Unknown bath type.')

        # defaults, can be changed later
        self.set_cutoff('sharp', 20)


    def __repr__(self):
        """String representation."""
        return """Markovian heat bath.  Spectral density: {sd}, T = {temp:g}, omega0 = {omega:g}""".format(sd = self.type, temp = self.T, omega = self.omega0)


    def build_LUT(self):
        """Build a lookup table for the S integral.
        """
        raise RuntimeError('unused')
        # TODO justify limits for S lookup
        if limit > 10:
            temp = logspace(log10(10.2), log10(limit), 50)
            temp = r_[linspace(0.1, 10, 100), temp] # sampling is denser near zero, where S changes more rapidly
        else:
            temp = linspace(0.1, limit, 10)
        self.dH = r_[-temp[::-1], 0, temp]

        self.s_table = []
        for k in range(len(self.dH)):
            self.s_table[k] = self.S_func(self.dH[k])

        # limits at inifinity
        self.dH      = r_[-inf, self.dH, inf]
        self.s_table = r_[0, self.s_table, 0]

        plot(self.dH, self.s_table, 'k-x')


    def S_func(self, dH):
        """Compute the S function.

        S_func(dH) == S(dH * omega0) / omega0
          == P\int_0^\infty dv J(omega0*v)/(hbar^2*omega0) (dH*coth(scale/2 * v) +v)/(dH^2 -v^2)
        """
        ep = 1e-5 # epsilon for Cauchy principal value
        if abs(dH) <= 1e-8:
            return self.s0
        else:
            # Cauchy principal value, integrand has simple poles at \nu = \pm dH.
            f = lambda nu: self.j(nu) * self.cut_func(nu) * (dH / tanh(self.scale * nu / 2) + nu) / (dH**2 -nu**2)
            a, abserr = quad(f, ep, abs(dH) - ep) 
            b, abserr = quad(f, abs(dH) + ep, inf) # 100 * self.cut_limit)
            return a + b


    def set_cutoff(self, type, lim):
        """Set the spectral density cutoff."""
        self.cut_type = type
        self.cut_limit = lim  # == omega_c/omega0

        # update cutoff function
        if self.cut_type == 'sharp':
            self.cut_func = lambda x: abs(x) <= self.cut_limit  # Heaviside theta cutoff
        elif self.cut_type == 'exp':
            self.cut_func = lambda x: exp(-abs(x) / self.cut_limit)  # exponential cutoff
        else:
            raise ValueError('Unknown cutoff type "{0}"'.format(self.cut_type))

        if self.type == 'ohmic':
            self.s0 = -self.cut_limit  # limit of S at dH == 0
  
        # clear lookup tables, since changing the cutoff requires recalc of S
        # start with a single point precomputed
        self.dH = array([-inf, 0, inf])
        self.gs_table = array([[0, 0], [self.g0, self.s0], [0,0]])



    def corr(self, x):
        """Bath spectral correlation tensor.

        g, s = corr(dH)

        Returns the bath spectral correlation tensor Gamma evaluated at omega0 * dH:
          Gamma(omega0 * dH) / omega0 == 0.5*g +i*s

        Ville Bergholm 2009-2011
        """
        tol = 1e-8
        max_w = 0.1 # maximum interpolation distance, TODO justify

        # assume parameters are set and lookup table computed
        #s = interp1(self.dH, self.s_table, x, 'linear', 0)

        # binary search for the interval [dH_a, dH_b) in which x falls
        b = searchsorted(self.dH, x, side = 'right')
        a = b - 1
        ee = self.dH[[a, b]]
        tt = self.gs_table[[a, b], :]
        # now x is in [ee[0], ee[1])

        gap = ee[1] - ee[0]
        d1 = abs(x - ee[0])
        d2 = abs(x - ee[1])

        def interpolate(ee, tt, x):
            # interp1 does way too many checks
            return tt[0] + ((x - ee[0]) / (ee[1] - ee[0])) * (tt[1] - tt[0])

        # x close enough to either endpoint?
        if d1 <= tol:
            return self.gs_table[a, :]
        elif d2 <= tol:
            return self.gs_table[b, :]
        elif gap <= max_w + tol:  # short enough gap to interpolate?
            return interpolate(ee, tt, x)
        else: # compute a new point p, then interpolate
            if gap <= 2 * max_w:
                p = ee[0] + gap / 2 # gap midpoint
                if x < p:
                    idx = 1 # which ee p will replace
                else:
                    idx = 0
            elif d1 <= max_w: # x within interpolation distance from one of the gap endpoints?
                p = ee[0] + max_w
                idx = 1
            elif d2 <= max_w:
                p = ee[1] - max_w
                idx = 0
            else: # x not near anything, don't interpolate
                p = x
                idx = 0

            # compute new g, s values at p and insert them into the table
            s = self.S_func(p)
            if abs(p) <= tol:
                g = self.g0 # limit at p == 0
            else:
                g = self.g_func(p) * self.cut_func(p)
            temp = array([[g, s]])

            self.dH = r_[self.dH[:b], p, self.dH[b:]]
            self.gs_table = r_[self.gs_table[:b], temp, self.gs_table[b:]]

            # now interpolate the required value
            ee[idx] = p
            tt[idx, :] = temp
            return interpolate(ee, tt, x)



    def fit(self, delta, T1, T2):
        """Qubit-bath coupling that reproduces given decoherence times.

        [H, D] = fit(delta, T1, T2)

        Returns the qubit Hamiltonian H and the qubit-bath coupling operator D
        that reproduce the decoherence times T1 and T2 (in units of 1/omega0)
        for a single-qubit system coupled to the bath.
        delta is the energy splitting for the qubit (in units of hbar*omega0).

        The bath object is not modified in any way.

        Ville Bergholm 2009-2010
        """
        if self.type == 'ohmic':
            # Fitting an ohmic bath to a given set of decoherence times

            iTd = 1 / T2 -0.5 / T1 # inverse pure dephasing time
            if iTd < 0:
                error('Unphysical decoherence times!')
    
            # match bath couplings to T1, T2
            temp = self.scale * delta / 2
            alpha = arctan2(1, sqrt(T1 * iTd / tanh(temp) * temp * self.cut_func(delta)))
            # dimensionless system-bath coupling factor squared
            N = iTd * self.scale / (4 * pi * cos(alpha)**2)

            # qubit Hamiltonian
            H = -delta/2 * sz

            # noise coupling
            D = sqrt(N) * (cos(alpha) * sz + sin(alpha) * sx)

            # decoherence times in scaled time units
            #T1 = 1/(N * sin(alpha)**2 * 2*pi * delta * coth(temp) * self.cut_func(delta))
            #T_dephase = self.scale/(N *4*pi*cos(alpha)**2)
            #T2 = 1/(0.5/T1 +1/T_dephase)
        else:
            raise NotImplementedError('Unknown bath type.')
        return H, D



def ops(H, D):
    """Jump operators for a Born-Markov master equation.

    Builds the jump operators for a Hamiltonian operator H and
    a (Hermitian) interaction operator D.

    Returns dH, a list of the sorted unique nonnegative differences between
    eigenvalues of H, and A, a sequence of the corresponding jump operators.
 
    A_k(dH_i) := A[k][i]

    Since A_k(-dH) = A_k^\dagger(dH), only the nonnegative dH:s and corresponding A:s are returned.

    Ville Bergholm 2009-2011
    """
    E, P = spectral_decomposition(H)
    m = len(E) # unique eigenvalues
    # energy difference matrix is antisymmetric, so we really only need the lower triangle
    deltaE = E[:, newaxis] - E  # deltaE[i,j] == E[i] - E[j]

    # mergesort is a stable sorting algorithm
    ind = argsort(deltaE, axis = None, kind = 'mergesort')
    # index of first lower triangle element
    s = m * (m - 1) / 2
    #assert(ind[s], 0)
    ind = ind[s:] # lower triangle indices only
    deltaE = deltaE.flat[ind] # lower triangle flattened

    if not isinstance(D, (list, tuple)):
        D = [D] # D needs to be a sequence, even if it has just one element
    n_D = len(D) # number of bath coupling ops

    # combine degenerate deltaE, build jump ops
    A = []
    # first dH == 0
    dH = [deltaE[0]]
    r, c = unravel_index(ind[0], (m, m))
    for d in D:
        A.append( [ dot(dot(P[c], d), P[r]) ] )

    for k in range(1, len(deltaE)):
        r, c = unravel_index(ind[k], (m, m))
        if abs(deltaE[k] - deltaE[k-1]) > tol:
            # new omega value, new jump op
            dH.append(deltaE[k])
            for op in range(n_D):
                A[op].append( dot(dot(P[c], D[op]), P[r]) )
        else:
            # extend current op
            for op in range(n_D):
                A[op][-1] += dot(dot(P[c], D[op]), P[r])

    return dH, A



def _check_baths(B):
    """Internal helper."""
    if not isinstance(B, (list, tuple)):
        B = [B] # needs to be a list, even if it has just one element

    # make sure the baths have the same omega0!
    temp = B[0].omega0
    for k in B:
        if k.omega0 != temp:
            raise ValueError('All the baths must have the same energy scale omega0!')
    return B



def lindblad_ops(H, D, B):
    """Lindblad operators for a Born-Markov master equation.

    Builds the Lindblad operators corresponding to a
    base Hamiltonian H and a (Hermitian) interaction operator D
    coupling the system to bath B.

    Returns L == { A_i / omega0 }_i and H_LS / (\hbar * omega0),
    where A_i are the Lindblad operators and H_LS is the Lamb shift.

    B can also be a list of baths, in which case D has to be
    a list of the corresponding interaction operators.

    Ville Bergholm 2009-2011
    """
    B = _check_baths(B)

    # jump ops
    dH, X = ops(H, D)
    H_LS = 0
    L = []
    for n, b in enumerate(B):
        A = X[n] # jump ops for bath/interaction op n

        # dH == 0 terms
        g, s = b.corr(0)
        L.append(sqrt(g) * A[0])
        H_LS += s * dot(A[0].conj().transpose(), A[0])  # Lamb shift

        for k in range(1, len(dH)):
            # first the positive energy shift
            g, s = b.corr(dH[k])
            L.append(sqrt(g) * A[k])
            H_LS += s * dot(A[k].conj().transpose(), A[k])

            # now the corresponding negative energy shift
            g, s = b.corr(-dH[k])
            L.append(sqrt(g) * A[k].conj().transpose())   # note the difference here, A(-omega) = A'(omega)
            H_LS += s * dot(A[k], A[k].conj().transpose()) # here too

    return L, H_LS
    # TODO ops for different baths can be combined into a single basis,
    # N^2-1 ops max in total



def superop(H, D, B):
    """Liouvillian superoperator for a Born-Markov system.

    Builds the Liouvillian superoperator L corresponding to a
    base Hamiltonian H and a (Hermitian) interaction operator D
    coupling the system to bath B.

    Returns L/omega0, which includes the system Hamiltonian, the Lamb shift,
    and the Lindblad dissipator.

    B can also be a list of baths, in which case D has to be
    a list of the corresponding interaction operators.

    Ville Bergholm 2009-2011
    """
    B = _check_baths(B)

    # jump ops
    dH, X = ops(H, D)
    iH_LS = 1j * H  # i * (system Hamiltonian + Lamb-Stark shift)
    acomm = 0
    diss = 0
    for n, b in enumerate(B):
        A = X[n] # jump ops for bath/interaction op n

        # we build the Liouvillian in a funny order to be a bit more efficient
        # dH == 0 terms
        [g, s] = b.corr(0)
        temp = dot(A[0].conj().transpose(), A[0])

        iH_LS += (1j * s) * temp  # Lamb shift
        acomm += (-0.5 * g) * temp # anticommutator
        diss  += lrmul(g * A[0], A[0].conj().transpose()) # dissipator (part)

        for k in range(1, len(dH)):
            # first the positive energy shift
            g, s = b.corr(dH[k])
            temp = dot(A[k].conj().transpose(), A[k])
            iH_LS += (1j * s) * temp
            acomm += (-0.5 * g) * temp
            diss  += lrmul(g * A[k], A[k].conj().transpose())

            # now the corresponding negative energy shift
            g, s = b.corr(-dH[k])
            temp = dot(A[k], A[k].conj().transpose()) # note the difference here, A(-omega) = A'(omega)
            iH_LS += (1j * s) * temp
            acomm += (-0.5 * g) * temp
            diss  += lrmul(g * A[k].conj().transpose(), A[k]) # here too

    return lmul(acomm -iH_LS) +rmul(acomm +iH_LS) +diss



def test():
    """Test script for Born-Markov methods.

    Ville Bergholm 2009-2010
    """
    dim = 6

    H = rand_hermitian(dim)
    D = [rand_hermitian(dim)/10, rand_hermitian(dim)/10]
    B = [bath('ohmic', 1e9, 0.02), bath('ohmic', 1e9, 0.03)]

    # jump operators
    dH, X = ops(H, D)
    assert_o(dH[0], 0, tol)
    for n, A in enumerate(X):
        temp = A[0] # dH[0] == 0
        for k in range(1, len(dH)):
            temp += A[k] +A[k].conj().transpose() # A(-omega) == A'(omega)
        assert_o(norm(temp - D[n]), 0, tol) # Lindblad ops should sum to D

    # equivalence of Lindblad operators and the Liouvillian superoperator
    LL, H_LS = lindblad_ops(H, D, B)
    S1 = superop_lindblad(LL, H + H_LS)
    S2 = superop(H, D, B)
    assert_o(norm(S1 - S2), 0, tol)
