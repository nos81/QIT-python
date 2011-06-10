# -*- coding: utf-8 -*-
# Author: Ville Bergholm 2011
"""Control sequences module."""

from __future__ import print_function, division

from numpy import sin, cos, arcsin, arccos, pi, asarray, eye, zeros, r_, c_, dot, nonzero, ceil, linspace
from scipy.linalg import expm
from scipy.optimize import brentq

from base import *



def nmr(a):
    """Convert NMR-style rotations into a one-qubit control sequence.
    s = NMR([[theta1, phi1], ...])

    Returns a one-qubit control sequence corresponding to NMR rotations
    of the form \theta_\phi.

    [a, theta] ^= R_a(theta) = expm(-i*a*sigma*theta/2) = expm(-i*H*t) => H = a*sigma/2, t = theta

    Ville Bergholm 2006-2009
    """
    a = asarray(a)
    theta = a[:, 0]
    phi   = a[:, 1]

    # find theta angles that are negative, convert them to corresponding positive rotations
    rows = nonzero(theta < 0)[0]
    theta[rows] = -theta[rows]
    phi[rows] = phi[rows] + pi
    return c_[cos(phi), sin(phi), zeros(phi.shape), theta]


def bb1(theta, phi=0, location=0.5):
    """Sequence for fixing off-resonance (\sigma_z bias) errors.

    Returns the Broadband number 1 control sequence for fixing
    errors in pulse length (or amplitude).

    The target rotation is \theta_\phi in the NMR notation.

    #! Cummins et al., "Tackling systematic errors in quantum logic gates with composite rotations", PRA 67, 042308 (2003).
    Ville Bergholm 2009
    """
    ph1 = arccos(-theta / (4*pi))
    W1  = nmr([[pi, ph1], [-2*pi, 3*ph1], [-pi, ph1]])
    return r_[nmr([[location * theta, phi]]), W1, nmr([[(1-location) * theta, phi]])]




def corpse(theta, phi=0):
    """Sequence for fixing off-resonance (\sigma_z bias) errors.

    Returns the CORPSE control sequence for fixing off-resonance
    errors, i.e. ones arising from a constant but unknown
    \sigma_z bias in the Hamiltonian.

    The target rotation is \theta_\phi in the NMR notation.

    CORPSE: Compensation for Off-Resonance with a Pulse SEquence

    #! Cummins et al., "Tackling systematic errors in quantum logic gates with composite rotations", PRA 67, 042308 (2003).
    Ville Bergholm 2009
    """
    n = [1, 1, 0] # CORPSE
    #n = [0, 1, 0] # short CORPSE

    temp = arcsin(sin(theta / 2) / 2)

    th1 = 2*pi*n[0] +theta/2 -temp
    th2 = 2*pi*n[1] -2*temp
    th3 = 2*pi*n[2] +theta/2 -temp
    return nmr([[th1, phi], [th2, phi+pi], [th3, phi]])


def cpmg(t, n):
    """Carr-Purcell-Meiboom-Gill sequence.

    Returns the Carr-Purcell-Meiboom-Gill sequence of n repeats with waiting time t.
    The purpose of the CPMG sequence is to facilitate a T_2 measurement
    under a nonuniform z drift, it is not meant to be a full memory protocol.
    The target operation for this sequence is identity.
    
    Ville Bergholm 2007-2009
    """
    s = nmr([[pi/2, pi/2]]) # initial y rotation

    step = r_[[[0, 0, 0, t]], nmr([[pi, 0]]), [[0, 0, 0, t]]] # wait, pi x rotation, wait

    for k in range(n):
        s = r_[s, step]
    return s


def scrofulous(theta, phi=0):
    """Sequence for fixing errors in pulse length.

    Returns the SCROFULOUS control sequence for fixing errors
    in pulse duration (or amplitude).

    The target rotation is \theta_\phi in the NMR notation.

    SCROFULOUS: Short Composite ROtation For Undoing Length Over- and UnderShoot

    #! Cummins et al., "Tackling systematic errors in quantum logic gates with composite rotations", PRA 67, 042308 (2003).
    Ville Bergholm 2006-2009
    """
    th1 = brentq(lambda t: (sin(t)/t -(2 / pi) * cos(theta / 2)), 0.1, 4.6)
    ph1 = arccos(-pi * cos(th1) / (2 * th1 * sin(theta / 2)))
    ph2 = ph1 - arccos(-pi / (2 * th1))

    u1 = nmr([[th1, ph1]])
    u2 = nmr([[pi, ph2]])
    return r_[u1, u2, u1]


def seq2prop(seq):
    """SU(2) propagator corresponding to a single-qubit control sequence.

    Returns the SU(2) rotation matrix U corresponding to the
    action of the single-qubit control sequence s alone.

    [a, theta] ^= R_a(theta) = expm(-i*a*sigma*theta/2) = expm(-i*H*t) => H = a*sigma/2, t = theta

    Ville Bergholm 2009
    """
    U = eye(2)
    for k in seq:
        H = 0.5 * (sx * k[0] + sy * k[1] + sz * k[2])
        t = k[-1]
        U = dot(expm(-1j * H * t), U)
    return U



def propagate(s, seq, out_func=lambda x: x):
    """Propagate a state in time using a control sequence.
    
    If no output function is given, we use an identity map.

    Ville Bergholm 2009-2010
    """
    base_dt = 0.1
    t = [0]  # initial time
    out = [out_func(s)]  # initial state

    # loop over the sequence
    for q in seq:
        # TODO qudits, gellmann basis
        H = 0.5 * (sx * q[0] +sy * q[1] +sz * q[2])
        T = q[-1]  # pulse duration
    
        n_steps = int(ceil(T / base_dt))
        dt = T / n_steps

        U = expm(-1j * H * dt)
        for j in range(n_steps):
            s = s.u_propagate(U)
            out.append(out_func(s))

        temp = t[-1]
        t.extend(list(linspace(temp+dt, temp+T, n_steps)))
    return out, t



def test():
    """Test script for the control sequences module.

    Ville Bergholm 2011
    """
    from numpy.random import rand
    import state
    from utils import rand_positive, assert_o

    dim = 2
    s = state.state(rand_positive(dim))
    seq = scrofulous(pi*rand(), 2*pi*rand())

    # equivalent propagations
    s1 = s.u_propagate(seq2prop(seq))
    out, t = propagate(s, seq)
    s2 = out[-1]
    assert_o((s1-s2).norm(), 0, tol)
