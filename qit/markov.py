# -*- coding: utf-8 -*-
"""
Born-Markov noise (:mod:`qit.markov`)
=====================================


This module simulates the effects of a heat bath coupled to a quantum
system, using the Born-Markov approximation in the weak coupling limit.
This results in a Lindblad-form master equation for the system.

The treatment in this module mostly follows Ref. :cite:`BP`.



Contents
--------

.. currentmodule:: qit.markov

.. autosummary::

   ops
   lindblad_ops
   superop   


:class:`bath` methods
---------------------

.. currentmodule:: qit.markov.bath

.. autosummary::

   desc
   set_cutoff
   setup
   build_LUT
   compute_gs
   corr
   fit
   plot_LUT
   plot_cutoff
   plot_correlation
"""
# Ville Bergholm 2011-2016

from __future__ import division, absolute_import, print_function, unicode_literals

import collections

from numpy import (array, sqrt, exp, log, log10, sin, cos, arctan2, tanh, sign, dot, argsort, pi,
                   r_, linspace, logspace, searchsorted, inf, newaxis, unravel_index, zeros, ones, empty,
                   sinh, cosh)
from scipy.linalg import norm
from scipy.integrate import quad
from scipy.special import expi, shichi
import scipy.constants as const
import matplotlib.pyplot as plt

from .base import sx, sz, tol
from .utils import lmul, rmul, lrmul, rand_hermitian, superop_lindblad, spectral_decomposition


__all__ = ['bath', 'ops', 'lindblad_ops', 'superop']


class bath(object):
    r"""Markovian heat bath.

    Supports bosonic and fermionic canonical ensembles at
    absolute temperature T, with an ohmic spectral density.
    The bath couples to the system via a single-term coupling
    :math:`H_\text{int} = A \otimes \sum_k \lambda_k (a_k +a_k')`.

    The bath spectral density is ohmic with a cutoff:
    :math:`J(\omega) = \omega \: \mathrm{cut}(\omega) \Theta(\omega)`.

    Three types of cutoffs are supported:

    .. math::
       \mathrm{cut}_\text{exp}(x \: \omega_c) &= \exp(-x),\\
       \mathrm{cut}_\text{smooth}(x \: \omega_c) &= \frac{1}{1+x^2},\\
       \mathrm{cut}_\text{sharp}(x \: \omega_c)  &= \Theta(1-x).

    The effects of the bath on the system are contained in the complex spectral correlation tensor :math:`\Gamma`.
    Computing values of this tensor is the main purpose of this class.
    It depends on three parameters:
    the inverse temperature of the bath :math:`s = \beta \hbar`,
    the spectral cutoff frequency of the bath :math:`\omega_c`,
    and the system frequency :math:`\omega`. It has the following scaling property:

    .. math::
       \Gamma_{s,\omega_c}(\omega) = \Gamma_{s/a,\omega_c a}(\omega a) / a.

    Hence we may eliminate the dimensions by choosing a = TU:

    .. math::
       \Gamma_{s,\omega_c}(\omega) = \frac{1}{\text{TU}} \Gamma_{\hat{s}, \hat{\omega_c}}(\hat{\omega}),

    where the hat denotes a dimensionless quantity.
    Since we only have one coupling term, :math:`\Gamma` is a scalar.
    We split it into its hermitian and antihermitian parts:

    .. math::
       \Gamma_{s, \omega_c}(\omega) = \frac{1}{2} \gamma(\omega) +i S(\omega),

    where :math:`\gamma` and :math:`S` are real, and

    .. math::
      \gamma(\omega) &= 2 \pi (1 \pm n(\omega)) \mathrm{cut}(|\omega|)
      \begin{cases}
      \omega   \quad \text{(bosons)}\\
      |\omega| \quad \text{(fermions)}
      \end{cases}\\
      S(\omega) &= P \int_0^\infty \mathrm{d}\nu \frac{J(\nu)}{(\omega-\nu)(\omega+\nu)}
      \begin{cases}
      \omega \coth(\beta \hbar \nu/2) +\nu \quad \text{(bosons)}\\
      \omega +\nu \tanh(\beta \hbar \nu/2) \quad \text{(fermions)}
      \end{cases}

    where :math:`n(\omega) := 1/(e^{\beta \hbar \omega} \mp 1)` is the Planc/Fermi function and :math:`\beta = 1/(k_B T)`. 
    Since :math:`\Gamma` is pretty expensive to compute, we store the computed results into a lookup table which is used to interpolate nearby values.


    Public data:

    ===========  ===========
    Data member  Description
    ===========  ===========
    type         Bath type. Currently only 'ohmic' is supported.
    stat         Bath statistics. Either 'boson' or 'fermion'.
    TU           Time unit (in s). All Hamiltonians have been made dimensionless by multiplying with :math:`\text{TU}/\hbar`.
    T            Absolute temperature of the bath (in K).
    scale        Dimensionless temperature scaling parameter :math:`\hbar / (k_B T \: \text{TU})`.
    cut_type     Spectral density cutoff type (string).
    cut_omega    Spectral density cutoff angular frequency :math:`\omega_c` (in :math:`1/\text{TU}`).
    ===========  ===========    


    Private data (set automatically):

    ===========  ===========
    Data member  Description
    ===========  ===========
    cut_func     Spectral density cutoff function.
    pf           Planck or Fermi function depending on statistics.
    corr_int     Integral transform for the dimensionless bath correlation function.
    g_func       Spectral correlation tensor, real part. :math:`\gamma(\omega/\text{TU}) \: \text{TU} = \mathrm{g\_func}(\omega)`.
    s_func       Spectral correlation tensor, imaginary part.
    g0           :math:`\lim_{\omega \to 0} \gamma(\omega)`.
    s0           :math:`\lim_{\omega \to 0} S(\omega)`.
    omega        Lookup table.
    gs_table     Lookup table. :math:`\gamma(\text{omega[k]}/\text{TU}) \: \text{TU}` = gs_table[k, 0].
    ===========  ===========
    """
    # Ville Bergholm 2009-2016

    def __init__(self, type, stat, TU, T):
        """constructor
    
        Sets up a descriptor for a heat bath coupled to a quantum system.
        """
        # basic bath parameters
        self.type = type
        self.stat = stat
        self.TU   = TU
        self.T    = T

        if type == 'ohmic':
            pass
        else:
            raise ValueError('Unknown bath type.')

        # defaults, can be changed later
        self.set_cutoff('exp', 1)


    def __repr__(self):
        """String representation."""
        return """<Markovian heat bath. Spectral density: {sd}, {st}ic statistics, T = {temp:g} K, TU = {tu:g} s>""".format(sd = self.type, st = self.stat, temp = self.T, tu = self.TU)


    def desc(self, long=True):
        """Bath description string for plots."""
        temp = '{}, {}, inverse T: {:.4g}, cutoff: {}, {:.4g}'.format(self.type, self.stat, self.scale, self.cut_type, self.cut_omega)
        if long:
            return r'Bath correlation tensor $\Gamma = \frac{1}{2} \gamma(\omega) +i S(\omega)$: ' +temp
        return temp


    def set_cutoff(self, type, lim):
        """Set the spectral density cutoff."""
        # We assume that cut_func(0) == 1.

        if type != None:
            self.cut_type = type
        if lim != None:
            self.cut_omega = lim  # == omega_c*TU

        # update cutoff function
        if self.cut_type == 'sharp':
            self.cut_func = lambda x: x <= self.cut_omega  # Heaviside theta cutoff
        elif self.cut_type == 'smooth':
            self.cut_func = lambda x: 1/(1+(x/self.cut_omega)**2)  # rational cutoff
        elif self.cut_type == 'exp':
            self.cut_func = lambda x: exp(-x / self.cut_omega)  # exponential cutoff
        else:
            raise ValueError('Unknown cutoff type "{0}"'.format(self.cut_type))
        self.setup()


    def setup(self):
        """Initializes the g and s functions, and the LUT.
        Must be called after parameters change."""

        # shorthand
        self.scale = const.hbar / (const.k * self.T * self.TU)

        # s_func has simple poles at \nu = \pm x.
        if self.stat == 'boson':
            self.pf = lambda x: 1/(exp(self.scale * x) - 1)
            self.corr_int_real = lambda s,nu: nu * self.cut_func(nu) * cos(nu*s) * (1 +2*self.pf(nu))
            self.corr_int_imag = lambda s,nu: nu * self.cut_func(nu) * -sin(nu*s)
            self.g_func = lambda x: 2*pi * x * self.cut_func(abs(x)) * (1 +self.pf(x))
            #self.s_func = lambda x,nu: nu * self.cut_func(nu) * ((1+self.pf(nu))/(x-nu) +self.pf(nu)/(x+nu))
            self.s_func = lambda x,nu: nu * self.cut_func(nu) * (x / tanh(self.scale * nu/2) +nu) / (x**2 -nu**2)
            self.g0 = 2*pi / self.scale
            temp, abserr = quad(self.cut_func, 0, inf)
        elif self.stat == 'fermion':
            self.pf = lambda x: 1/(exp(self.scale * x) + 1)
            self.corr_int_real = lambda s,nu: nu * self.cut_func(nu) * cos(nu*s)
            self.corr_int_imag = lambda s,nu: nu * self.cut_func(nu) * sin(nu*s) * (-1 +2*self.pf(nu))
            self.g_func = lambda x: 2*pi * abs(x) * self.cut_func(abs(x)) * (1 -self.pf(x))
            #self.s_func = lambda x,nu: nu * self.cut_func(nu) * ((1-self.pf(nu))/(x-nu) +self.pf(nu)/(x+nu))
            self.s_func = lambda x,nu: nu * self.cut_func(nu) * (x +nu * tanh(self.scale * nu/2)) / (x**2 -nu**2)
            self.g0 = 0
            temp, abserr = quad(lambda x: self.cut_func(x) * tanh(x*self.scale/2), 0, inf)
        else:
            raise ValueError('Unknown bath statistics.')
        self.s0 = -temp

        # clear lookup tables, since changing the cutoff requires recalc of S
        self.omega = array([])
        self.gs_table = array([])


    def build_LUT(self, om=None):
        """Build a lookup table for the spectral correlation tensor Gamma.

        :param vector om:  Vector of omegas denoting the points to compute.
        """

        # TODO justify limits for S lookup
        if om == None:
            # Default sampling for the lookup table.
            lim = self.cut_omega
            #lim = log10(10) / 5 / self.scale  # up to boltzmann factor == 10
            om = logspace(log10(1.1 * lim), log10(5 * lim), 20) # logarithmic sampling
            om = r_[linspace(0.05 * lim, 1 * lim, 20), om] # sampling is denser near zero, where S changes more rapidly
            om = r_[-om[::-1], 0, om]  # symmetric around zero

        self.omega = om
        self.gs_table = empty((len(om), 2))
        for k in range(len(om)):
            print(k)
            self.gs_table[k] = self.compute_gs(self.omega[k])

        # limits at infinity
        self.omega    = r_[-inf, self.omega, inf]
        self.gs_table = r_[[[0, 0]], self.gs_table, [[0, 0]]]


    def _plot(self, x, om, q, gs, odd_s, even_s, f=plt.plot):
        """Plotting utility."""

        plt.figure()
        plt.hold(True)
        plt.grid(True)
        plt.plot(x, gs, '-x')
        plt.plot(x, odd_s, 'k-', x, even_s, 'm-')

        # analytical expressions for even and odd s funcs
        if self.cut_type == 'sharp':
            odd_s = log(abs(q**2 / (q**2-1)))
            even_s = log(abs((1+q) / (1-q))) -2/q
        elif self.cut_type == 'smooth':
            odd_s = 2*log(abs(q)) / (1+q**2)
            even_s = -pi/q  / (1+q**2)
        elif self.cut_type == 'exp':
            odd_s = expi(-q)*exp(q) +expi(q)*exp(-q)
            even_s = -(expi(-q)*exp(q) -expi(q)*exp(-q)) -2/q
        else:
            raise ValueError('Unknown cut type.')

        f(x, om * odd_s, 'ko', x, om * even_s, 'mo')
        plt.ylabel('[1/TU]')
        plt.legend(['$\gamma$', 'S', '$S(\omega)-S(-\omega)$', '$S(\omega)+S(-\omega)$',
                    '$S(\omega)-S(-\omega)$ (fermion)', '$S(\omega)+S(-\omega)$ (boson)'])



    def plot_LUT(self):
        """Plot the contents of the spectral correlation tensor LUT as a function of omega."""

        if len(self.omega) == 0:
            self.build_LUT()

        om = self.omega
        # computed values
        s = self.gs_table[:,1]
        temp = s[::-1]  # s(-x)
        odd_s  = s -temp  # s(x)-s(-x)
        even_s = s +temp  # s(x)+s(-x)

        # ratio of Lamb shift to dephasing rate
        #g = self.gs_table[:,0]
        #boltz = exp(-self.scale * om)
        #ratio = odd_s / (g * (boltz+1))

        c = self.cut_omega
        self._plot(om, om, om/c, self.gs_table, odd_s, even_s)
        # cut limits
        a = plt.axis()[2:]  # y limits
        plt.plot([c,c], a, 'k-')
        plt.plot([-c,-c], a, 'k-')

        plt.xlabel(r'$\omega$ [1/TU]')
        plt.title(self.desc())


    def plot_cutoff(self, boltz=0.5):
        r"""Plot spectral correlation tensor components as a function of cutoff frequency.
        boltz is the Boltzmann factor :math:`e^{-\beta \hbar \omega}` that fixes omega.
        """

        orig_scale  = self.scale
        orig_cutoff = self.cut_omega

        om = -log(boltz) / self.scale  # \omega * TU
        # scale with |om|
        temp = abs(om)
        om_scaled = om / temp  # sign of om
        self.scale *= temp     # scale the temperature parameter

        # try different cutoffs relative to omega
        cutoff = logspace(-1.5, 1.5, 50)
        gs  = zeros((len(cutoff), 2))
        gsm = zeros((len(cutoff), 2))
        for k in range(len(cutoff)):
            print(k)
            self.set_cutoff(None, cutoff[k])  # scaled cutoff frequency
            gs[k,:]  = self.compute_gs(om_scaled)
            gsm[k,:] = self.compute_gs(-om_scaled)
        gs  *= temp
        gsm *= temp
        odd_s  = gs[:,1] -gsm[:,1]
        even_s = gs[:,1] +gsm[:,1]

        # restore original bath parameters
        self.scale = orig_scale
        self.set_cutoff(None, orig_cutoff)

        self._plot(cutoff, om, om_scaled/cutoff, gs, odd_s, even_s, plt.semilogx)
        plt.xlabel(r'$\omega_c/\omega$')
        plt.title(self.desc() + ', boltzmann: {:.4g} => omega: {:.4g}'.format(boltz, om))


    def plot_correlation(self):
        r"""Plot the bath correlation function
        :math:`C_{s,\omega_c}(t) = \frac{1}{\hbar^2}\langle B(t) B(0)\rangle`.

        It scales as

        .. math::
           C_{s,\omega_c}(t) = \frac{1}{a^2} C_{s/a,\omega_c a}(t/a).

        Choosing a = TU, we obtain

        .. math::
           C_{s,\omega_c}(t) = \frac{1}{\text{TU}^2} C_{\hat{s}, \hat{\omega_c}}(\hat{t}).
        """

        tol_nu = 1e-5  # approaching the singularity at nu=0 this closely
        c = self.cut_omega
        plt.figure()

        # plot the functions to be transformed
        plt.subplot(1,3,1)
        nu = linspace(tol_nu, 5*c, 500)
        plt.plot(nu, nu * self.cut_func(nu) * self.pf(nu), 'r')
        plt.hold(True)
        if self.stat == 'boson':
            plt.plot(nu, nu * self.cut_func(nu) * (1+self.pf(nu)), 'b')
        else:
            plt.plot(nu, nu * self.cut_func(nu) * (1-self.pf(nu)), 'g')
        plt.grid(True)
        plt.legend(['$n$', '$1 \pm n$'])
        plt.xlabel(r'$\nu$ [1/TU]')
        plt.title('Integrand without exponentials')

        # plot the full integrand
        plt.subplot(1,3,2)
        t = linspace(0, 4/c, 5)
        nu = linspace(tol_nu, 5*c, 100)
        res = empty((len(nu), len(t)), dtype=complex)
        for k in range(len(t)):
            print(k)
            res[:,k] = self.corr_int_real(t[k], nu) +1j*self.corr_int_imag(t[k], nu)
        plt.plot(nu, res.real, '-', nu, res.imag, '--')
        plt.grid(True)
        plt.xlabel(r'$\nu$ [1/TU]')
        plt.title('Integrand')

        # plot the correlation function
        plt.subplot(1,3,3)
        t = linspace(tol_nu, 10/c, 100)  # real part of C(t) is even, imaginary part odd
        res = empty(len(t), dtype=complex)
        # upper limit for integration
        if self.cut_type == 'smooth':
            int_max = 100*c  # HACK, the integrator does not do well otherwise
        else:
            int_max = inf
        for k in range(len(t)):
            print(k)
            #fun = lambda x: x * self.cut_func(x) * (exp(-1j*x*t[k])*(1+self.pf(x))+exp(1j*x*t[k])*self.pf(x))
            fun1 = lambda x: self.corr_int_real(t[k], x)
            fun2 = lambda x: self.corr_int_imag(t[k], x)
            res[k], abserr = quad(fun1, tol_nu, int_max)
            temp, abserr = quad(fun2, tol_nu, int_max)
            res[k] += 1j*temp
        plt.plot(t, res.real, 'k-', t, res.imag, 'k--', t, abs(res), 'k-.')
        a = plt.axis()
        plt.hold(True)
        plt.grid(True)
        plt.xlabel('t [TU]')
        plt.ylabel('[1/TU^2]')
        plt.title('Bath correlation function C(t): ' + self.desc(False))

        # plot analytic high- and low-temp limits
        x = c * t
        if self.cut_type == 'sharp':
            temp = ((1 +1j*x)*exp(-1j*x) -1)/x**2
            hb = 2/self.scale * sin(x)/x
        elif self.cut_type == 'smooth':
            shi, chi = shichi(x)
            temp = sinh(x)*shi -cosh(x)*chi -1j*pi/2*exp(-abs(x))
            hb = pi/self.scale * exp(-abs(x))
        elif self.cut_type == 'exp':
            temp = 1/(1 +1j*x)**2
            hb = 2/self.scale / (1+x**2)
        else:
            raise ValueError('Unknown cutoff type "{0}"'.format(self.cut_type))
        temp *= c**2
        hb   *= c
        if self.stat == 'boson':
            plt.plot(t, temp.real, 'b.')  # real part, cold
            plt.plot(t, hb, 'r.')  # real part, hot
            plt.plot(t, temp.imag, 'ko')  # imag part, every T
            plt.legend(['re C', 'im C', '|C|', 're C (cold, analytic)', 're C (hot, analytic)', 'im C (analytic)'])
        else:
            plt.plot(t, temp.real, 'k.')  # real part, every T
            plt.plot(t, temp.imag, 'bo')  # imag part, cold
            plt.plot(t, 0*t, 'ro')  # imag part, hot
            plt.legend(['re C', 'im C', '|C|', 're C (analytic)', 'im C (cold, analytic)', 'im C (hot, analytic)'])
        plt.axis(a)
        return res


    def compute_gs(self, x):
        r"""Computes the spectral correlation tensor.
        See :func:`corr` for usage.
        """
        ep = 1e-5 # epsilon for Cauchy principal value
        tol_omega0 = 1e-8

        if abs(x) <= tol_omega0:
            return self.g0, self.s0
        else:
            g = self.g_func(x)

            # Cauchy principal value, integrand has simple poles at \nu = \pm x.
            # TODO scipy quad can do these types of simple pole PVs directly...
            f = lambda nu: self.s_func(x, nu)
            a, abserr = quad(f, tol_omega0, abs(x) -ep)
            b, abserr = quad(f, abs(x) +ep, inf)
            return g, a + b


    def corr(self, x):
        r"""Bath spectral correlation tensor, computed or interpolated.

        :param float x:  Angular frequency [1/TU]
        :returns tuple (g,s): Real and imaginary parts of the spectral correlation tensor [1/TU]

        .. math::

           \Gamma(x/\text{TU}) \: \text{TU} = \frac{1}{2} g +i s
        """
        # Ville Bergholm 2009-2011

        tol_omega = 1e-8
        max_ip_omega = 0.1 # maximum interpolation distance, TODO justify

        # assume parameters are set and lookup table computed
        #s = interp1(self.omega, self.gs_table, x, 'linear', 0)

        # TODO omega and gs_table into a single dictionary?
        # binary search for the interval [omega_a, omega_b) in which x falls
        b = searchsorted(self.omega, x, side = 'right')
        a = b - 1
        ee = self.omega[[a, b]]
        tt = self.gs_table[[a, b], :]
        # now x is in [ee[0], ee[1])

        gap = ee[1] - ee[0]
        d1 = abs(x - ee[0])
        d2 = abs(x - ee[1])

        def interpolate(ee, tt, x):
            "Quick interpolation."
            # interp1 does way too many checks
            return tt[0] + ((x - ee[0]) / (ee[1] - ee[0])) * (tt[1] - tt[0])

        # x close enough to either endpoint?
        if d1 <= tol_omega:
            return self.gs_table[a, :]
        elif d2 <= tol_omega:
            return self.gs_table[b, :]
        elif gap <= max_ip_omega + tol_omega:  # short enough gap to interpolate?
            return interpolate(ee, tt, x)
        else: # compute a new point p, then interpolate
            if gap <= 2 * max_ip_omega:
                p = ee[0] + gap / 2 # gap midpoint
                if x < p:
                    idx = 1 # which ee p will replace
                else:
                    idx = 0
            elif d1 <= max_ip_omega: # x within interpolation distance from one of the gap endpoints?
                p = ee[0] + max_ip_omega
                idx = 1
            elif d2 <= max_ip_omega:
                p = ee[1] - max_ip_omega
                idx = 0
            else: # x not near anything, don't interpolate
                p = x
                idx = 0

            # compute new g, s values at p and insert them into the table
            temp = self.compute_gs(p)

            self.omega = r_[self.omega[:b], p, self.omega[b:]]
            self.gs_table = r_[self.gs_table[:b], [temp], self.gs_table[b:]]

            # now interpolate the required value
            ee[idx] = p
            tt[idx, :] = temp
            return interpolate(ee, tt, x)



    def fit(self, delta, T1, T2):
        r"""Qubit-bath coupling that reproduces given decoherence times.

        :param float delta:  Energy splitting for the qubit (in units of :math:`\hbar/TU`)
        :param float T1, T2: Decoherence times T1 and T2 (in units of :math:`TU`)
        :returns tuple (H, D):

        Returns the qubit Hamiltonian H and the qubit-bath coupling operator D
        that reproduce the decoherence times T1 and T2
        for a single-qubit system coupled to the bath.

        The bath object is not modified.
        """
        # Ville Bergholm 2009-2016

        if self.type == 'ohmic':
            # Fitting an ohmic bath to a given set of decoherence times

            iTd = 1/T2 -0.5/T1 # inverse pure dephasing time
            if iTd < 0:
                raise ValueError('Unphysical decoherence times!')
    
            # match bath couplings to T1, T2
            x = self.scale * delta / 2

            if self.stat == 'boson':
                temp = x / tanh(x) * self.cut_func(abs(delta))
                # coupling, ZX angle
                alpha = arctan2(1, sqrt(T1 * iTd *temp))
                # dimensionless system-bath coupling factor squared
                c = iTd * self.scale / (4 * pi * cos(alpha)**2)
                # noise coupling operator
                D = sqrt(c) * (cos(alpha) * sz + sin(alpha) * sx)

                # decoherence times in scaled time units
                #T1 = 1/(c * sin(alpha)**2 * 2*pi * delta * coth(x) * self.cut_func(delta))
                #T_dephase = self.scale/(c *4*pi*cos(alpha)**2)
                #T2 = 1/(0.5/T1 +1/T_dephase)

            elif self.stat == 'fermion':
                if abs(iTd) >= tol:
                    raise ValueError('For a fermionic bath we must have T2 = 2*T1')
                # dimensionless system-bath coupling factor squared
                c = 1/(T1 * 2*pi * abs(delta) * self.cut_func(abs(delta)))
                D = sqrt(c) * sx
            else:
                raise NotImplementedError('Unknown bath statistics.')
        else:
            raise NotImplementedError('Unknown bath type.')

        # qubit Hamiltonian
        H = -delta/2 * sz
        return H, D



def ops(H, D):
    r"""Jump operators for a Born-Markov master equation.

    :param array H: System Hamiltonian
    :param sequence D: Sequence of Hermitian interaction operators
    :returns tuple (dH, A):

    dH is a list of the sorted unique nonnegative differences between
    eigenvalues of H, and A is an array of the corresponding jump operators:
    :math:`A_k(dH_i) = A[k][i]`, where the jump ops A[k] correspond to D[k].

    Since :math:`A_k(-dH) = A_k^\dagger(dH)`, only the nonnegative dH:s and corresponding A:s are returned.
    """
    # Ville Bergholm 2009-2016

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

    if not isinstance(D, collections.Sequence):
        D = [D] # D needs to be a sequence, even if it has just one element
    n_D = len(D) # number of bath coupling ops

    # combine degenerate deltaE, build jump ops
    dH = []
    A = [[] for k in range(n_D)]
    current_dE = inf
    # loop over lower triangle indices
    for k in ind:
        dE = deltaE.flat[k]
        if abs(dE -current_dE) > tol:
            # new omega value, new jump op
            current_dE = dE
            dH.append(dE)
            for op in range(n_D):
                A[op].append(0)
        # extend current jump op
        r, c = unravel_index(k, (m, m))
        for op in range(n_D):
            A[op][-1] += dot(dot(P[c], D[op]), P[r])

    A  = array(A)
    dH = array(dH)
    # find columns in which every A vanishes
    temp = zeros(A.shape[0:2])
    for k in range(len(dH)):
        for op in range(n_D):
            temp[op, k] = norm(A[op, k]) > tol
    temp = temp.any(0)
    # eliminate zero As and corresponding dHs
    A = A[:,temp]
    dH = dH[temp]

    # Are some of the remaining dH differences too low for RWA to hold properly?
    # TODO justify the numerical tolerance used
    for k in range(1, len(dH)):
        if abs(dH[k] -dH[k-1]) < 1e-3:
            print('Warning: Small difference between dH({}) and dH({}) may break the RWA.\n'.format(k-1, k))
    return dH, A



def _check_baths(B):
    """Internal helper."""
    if not isinstance(B, collections.Sequence):
        B = [B] # needs to be a sequence, even if it has just one element

    # make sure the baths have the same TU!
    temp = B[0].TU
    for k in B:
        if k.TU != temp:
            raise ValueError('All the baths must have the same time unit!')
    return B



def lindblad_ops(H, D, B):
    r"""Lindblad operators for a Born-Markov master equation.

    :param array H:  System Hamiltonian
    :param array D:  Hermitian interaction operator
    :param bath B:   :class:`qit.markov.bath` instance
    :returns tuple (L, H_LS):

    Builds the Lindblad operators corresponding to a
    base Hamiltonian H and a (Hermitian) interaction operator D
    coupling the system to bath B.

    Returns a list L of Lindblad operators :math:`A_i` (in units of :math:`1/\sqrt{\text{TU}}`)
    and the Lamb shift Hamiltonian :math:`H_{\text{LS}}` (in units of :math:`\hbar/\text{TU}`).

    B can also be a sequence of baths, in which case D has to be
    a sequence of the corresponding interaction operators.
    """
    # Ville Bergholm 2009-2016

    B = _check_baths(B)

    # jump ops
    dH, X = ops(H, D)
    H_LS = 0
    L = []
    for n, b in enumerate(B):
        A = X[n] # jump ops for bath/interaction op n

        for k in range(len(dH)):
            #NA[k] = norm(A[k], 'fro'); # how significant is this op?

            # first the positive energy shift
            g, s = b.corr(dH[k])
            # is the dissipation significant?
            if abs(g) >= tol:
                L.append(sqrt(g) * A[k])
            H_LS += s * dot(A[k].conj().transpose(), A[k])

            if dH[k] == 0:
                # no negative shift
                continue

            # now the corresponding negative energy shift
            g, s = b.corr(-dH[k])
            if abs(g) >= tol:
                L.append(sqrt(g) * A[k].conj().transpose()) # note the difference here, A(-omega) = A'(omega)
            H_LS += s * dot(A[k], A[k].conj().transpose())  # here too
    return L, H_LS
    # TODO ops for different baths can be combined into a single basis,
    # N^2-1 ops max in total



def superop(H, D, B):
    r"""Liouvillian superoperator for a Born-Markov master equation.

    :param array H:  System Hamiltonian
    :param array D:  Hermitian interaction operator
    :param bath B:   :class:`qit.markov.bath` instance
    :returns array L:

    Builds the Liouvillian superoperator L corresponding to a
    base Hamiltonian H and a (Hermitian) interaction operator D
    coupling the system to bath B.

    L includes the system Hamiltonian, the Lamb shift,
    and the Lindblad dissipator, and is in units of 1/TU.

    B can also be a sequence of baths, in which case D has to be
    a sequence of the corresponding interaction operators.
    """
    # Ville Bergholm 2009-2016
    B = _check_baths(B)

    # jump ops
    dH, X = ops(H, D)
    iH_LS = 1j * H  # i * (system Hamiltonian + Lamb-Stark shift)
    acomm = 0  # anticommutator
    diss = 0   # the rest of the dissipator
    for n, b in enumerate(B):
        A = X[n] # jump ops for bath/interaction op n

        # we build the Liouvillian in a funny order to be a bit more efficient
        for k in range(len(dH)):
            # first the positive energy shift
            g, s = b.corr(dH[k])
            temp = dot(A[k].conj().transpose(), A[k])
            iH_LS += (1j * s) * temp
            acomm += (-0.5 * g) * temp
            diss  += lrmul(g * A[k], A[k].conj().transpose())

            if dH[k] == 0:
                # no negative shift
                continue

            # now the corresponding negative energy shift
            g, s = b.corr(-dH[k])
            temp = dot(A[k], A[k].conj().transpose()) # note the difference here, A(-omega) = A'(omega)
            iH_LS += (1j * s) * temp
            acomm += (-0.5 * g) * temp
            diss  += lrmul(g * A[k].conj().transpose(), A[k]) # here too

    return lmul(acomm -iH_LS) +rmul(acomm +iH_LS) +diss
