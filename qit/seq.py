# -*- coding: utf-8 -*-
r"""
Control sequences (:mod:`qit.seq`)
==================================

.. currentmodule:: qit.seq.seq


.. currentmodule:: qit.seq

Contents
--------

.. autosummary::

   nmr
   corpse
   bb1
   scrofulous
   knill
   dd
   propagate
"""
# Ville Bergholm 2009-2016

from __future__ import division, absolute_import, print_function, unicode_literals

from copy import deepcopy

from numpy import array, arange, sin, cos, arcsin, arccos, pi, asarray, eye, zeros, r_, c_, dot, nonzero, ceil, linspace
from scipy.linalg import expm
from scipy.optimize import brentq

from .base import sx, sy, tol


__all__ = ['seq', 'nmr', 'bb1', 'corpse', 'scrofulous', 'knill', 'dd', 'propagate']


class seq(object):
    r"""
    Piecewise constant control sequences for quantum systems.

    Variables:
    =======  ===============================================================================================
    A        Drift generator (typically :math:`-1j/\hbar` times a Hamiltonian and a time unit of your choice).
    B        List of control generators. c := len(B).
    tau      Vector of durations of the time slices. m := len(tau).
    control  Array, shape (m, c). control[i,j] is the value of control field j during time slice i.
    =======  ===============================================================================================

    The total generator for the time slice j is thus given by

    .. math::

    G_j = A +\sum_k \text{control}_{jk} B_k,

    and the corresponding propagator is

    .. math::

    P_j = \exp(\tau_j G_j).
    """
    def __init__(self, tau=[], control=zeros((0,2))):
        # construct the sequence
        self.A = zeros((2, 2), dtype=complex)
        self.B = [-0.5j * sx, -0.5j * sy]
        self.tau = asarray(tau)
        self.control = asarray(control)

    def __repr__(self):
        # prettyprint the object
        out = 'Control sequence\n'
        out += 'tau: ' + self.tau.__repr__() +'\n'
        out += 'control: ' + self.control.__repr__()
        return out

    def to_prop(self):
        r"""Propagator corresponding to the control sequence.

        Returns the propagator matrix corresponding to the
        action of the control sequence.

        Governing equation: :math:`\dot(X)(t) = (A +\sum_k u_k(t) B_k) X(t) = G(t) X(t)`.
        """
        n = len(self.tau)
        P = eye(self.A.shape[0], dtype=complex)
        for j in range(n):
            G = deepcopy(asarray(self.A, dtype=complex))
            for k, b in enumerate(self.B):
                G += self.control[j, k] * b

            temp = expm(self.tau[j] * G)
            P = dot(temp, P)
        return P


def nmr(a):
    r"""Convert NMR-style rotations into a one-qubit control sequence.

    Returns a one-qubit control sequence corresponding to the array a:

    .. math::

       a = [[\theta_1, \phi_1], [\theta_2, \phi_2], ...]

    Each :math:`\theta, \phi` pair corresponds to a NMR rotation
    of the form :math:`\theta_\phi`,
    or a rotation of the angle :math:`\theta`
    about the unit vector :math:`[\cos(\phi), \sin(\phi), 0]`.

    .. math::

       R_{\vec{a}}(\theta) = \exp(-i \vec{a} \cdot \vec{\sigma} \theta/2) = \exp(-i H t) \quad \Leftarrow \quad
       H = \vec{a} \cdot \vec{\sigma}/2, \quad t = \theta.
    """
    # Ville Bergholm 2006-2016

    a = asarray(a, dtype=float)
    theta = a[:, 0]
    phi   = a[:, 1]

    # find theta angles that are negative, convert them to corresponding positive rotations
    rows = nonzero(theta < 0)[0]
    theta[rows] = -theta[rows]
    phi[rows] = phi[rows] + pi
    return seq(theta, c_[cos(phi), sin(phi)])


def bb1(theta, phi=0, location=0.5):
    r"""Sequence for correcting pulse length errors.

    Returns the Broadband number 1 control sequence for correcting
    proportional errors in pulse length (or amplitude) :cite:`Wimperis`.

    The target rotation is :math:`\theta_\phi` in the NMR notation.
    """
    # Ville Bergholm 2009-2012

    ph1 = arccos(-theta / (4*pi))
    W1  = [[pi, ph1], [2*pi, 3*ph1], [pi, ph1]]
    return nmr([[location * theta, phi]] + W1 + [[(1-location) * theta, phi]])


def corpse(theta, phi=0):
    r"""Sequence for correcting off-resonance errors.

    Returns the CORPSE control sequence for correcting off-resonance
    errors, i.e. ones arising from a constant but unknown
    :math:`\sigma_z` bias in the Hamiltonian :cite:`Cummins`.

    The target rotation is :math:`\theta_\phi` in the NMR notation.

    CORPSE: Compensation for Off-Resonance with a Pulse SEquence
    """
    # Ville Bergholm 2009

    n = [1, 1, 0] # CORPSE
    #n = [0, 1, 0] # short CORPSE

    temp = arcsin(sin(theta / 2) / 2)

    th1 = 2*pi*n[0] +theta/2 -temp
    th2 = 2*pi*n[1] -2*temp
    th3 = 2*pi*n[2] +theta/2 -temp
    return nmr([[th1, phi], [th2, phi+pi], [th3, phi]])


def scrofulous(theta, phi=0):
    r"""Sequence for correcting pulse length errors.

    Returns the SCROFULOUS control sequence for correcting errors
    in pulse duration (or amplitude) :cite:`Cummins`.

    The target rotation is :math:`\theta_\phi` in the NMR notation.

    SCROFULOUS: Short Composite ROtation For Undoing Length Over- and UnderShoot
    """
    # Ville Bergholm 2006-2016

    th1 = brentq(lambda t: (sin(t)/t -(2 / pi) * cos(theta / 2)), 0.1, 4.6)
    ph1 = arccos(-pi * cos(th1) / (2 * th1 * sin(theta / 2))) +phi
    ph2 = ph1 - arccos(-pi / (2 * th1))

    u1 = [[th1, ph1]]
    u2 = [[pi,  ph2]]
    return nmr(u1 + u2 + u1)


def knill(phi=0):
    r"""Sequence for robust pi pulses.

    The target rotation in the NMR notation is \pi_\phi followed by Z_{-\pi/3}.
    In an experimental setting the Z rotation can often be absorbed by a
    reference frame change that does not affect the measurement results :cite:`RHC2010`.

    The Knill pulse is quite robust against off-resonance errors, and somewhat
    robust against pulse strenght errors.
    """
    # Ville Bergholm 2015-2016

    th = pi
    return nmr([[th, pi/6+phi], [th, phi], [th, pi/2+phi], [th, phi], [th, pi/6+phi]])


def dd(name, t, n=1):
    r"""Dynamical decoupling and refocusing sequences.

    name    name of the sequence: hahn, cpmg, uhrig, xy4
    t       total waiting time
    n       order (if applicable)

    The target operation for these sequences is identity.
    See e.g. :cite:`Uhrig2007`.
    """
    # Ville Bergholm 2007-2016

    # Multiplying the pi pulse strength by factor s is equivalent to A -> A/s, t -> t*s.
    # which sequence?
    if name == 'wait':
        # Do nothing, just wait.
        tau = [1]
        phase = []
    elif name == 'hahn':
        # Basic Hahn spin echo
        tau = [0.5, 0.5]
        phase = [0]
    elif name == 'cpmg':
        # Carr-Purcell-Meiboom-Gill
        # The purpose of the CPMG sequence is to facilitate a T_2 measurement
        # under a nonuniform z drift, it is not meant to be a full memory protocol.
        tau = [0.25, 0.5, 0.25]
        phase = [0, 0]
    elif name == 'uhrig':
        # Uhrig's family of sequences
        # n=1: Hahn echo
        # n=2: CPMG
        delta = arange(n+2)
        delta = sin(pi * delta/(2*(n+1))) ** 2
        tau   = delta[1:]-delta[:n+1]  # wait durations
        phase = zeros(n)
    elif name == 'xy4':
        # uncentered version
        tau = np.ones(4) / 4
        phase = [0, pi/2, 0, pi/2]
    else:
        raise ValueError('Unknown sequence.')

    # initialize the sequence struct
    s = seq()
    # waits and pi pulses with given phases
    ind = 1
    for k, p in enumerate(phase):
        # wait
        s.tau = r_[s.tau, t * tau[k]]
        s.control = r_[s.control, zeros((1,2))]
        # pi pulse
        s.tau = r_[s.tau, pi]
        s.control = r_[s.control, array([[cos(p), sin(p)]])]
    if len(tau) > len(phase):
        # final wait
        s.tau = r_[s.tau, t * tau[-1]]
        s.control = r_[s.control, zeros((1,2))]
    return s


def propagate(s, seq, out_func=lambda x: x, base_dt=0.1):
    """Propagate a state in time using a control sequence.
    
    If no output function is given, we use an identity map.
    """
    # Ville Bergholm 2009-2016

    n = len(seq.tau)
    t = [0]  # initial time
    out = [out_func(s)]  # initial state

    # loop over the sequence
    for j in range(n):
        G = deepcopy(asarray(seq.A, dtype=complex))
        for k, b in enumerate(seq.B):
            G += seq.control[j, k] * b

        T = seq.tau[j]  # pulse duration
        n_steps = max(int(ceil(T / base_dt)), 1)
        dt = T / n_steps

        P = expm(G * dt)
        for k in range(n_steps):
            s = s.u_propagate(P)
            out.append(out_func(s))

        temp = t[-1]
        t.extend(list(linspace(temp+dt, temp+T, n_steps)))
    return out, t
