"""
Unit tests for qit.seq
"""
# Ville Bergholm 2011-2016

import pytest

import numpy as np

from numpy.random import rand, randn
from numpy.linalg import norm

import qit
from qit.base import sx, sy, sz
from qit.utils import rand_positive
import qit.seq as seq



class TestSeq:
    def test_seq_class(self):
        """Sequence construction."""
        s = seq.Seq()
        assert len(s) == 0

        s = seq.Seq([1, 2], [[0.1, 0.2], [0.3, 0.4]])
        assert len(s) == 2
        s.A = -1j * sz

        G = s.generator(0)
        assert G.shape == (2, 2)

    def test_nmr(self, tol):
        """NMR rotations"""

        s = seq.nmr([[3, 2], [1, 2], [-1, 0.3]])

        # pi rotation
        U = seq.nmr([[np.pi, 0]]).to_prop()
        assert norm(U + 1j * sx) == pytest.approx(0, abs=tol)
        U = seq.nmr([[np.pi, np.pi/2]]).to_prop()
        assert norm(U + 1j * sy) == pytest.approx(0, abs=tol)

    def test_correction_sequences(self, tol):

        # rotation sequences in the absence of errors
        theta = np.pi * rand()
        phi = 2 * np.pi * rand()
        U = seq.nmr([[theta, phi]]).to_prop()

        V = seq.bb1(theta, phi, location=rand()).to_prop()
        assert norm(U - V) == pytest.approx(0, abs=tol)

        V = seq.corpse(theta, phi).to_prop()
        assert norm(U - V) == pytest.approx(0, abs=tol)

        V = seq.scrofulous(theta, phi).to_prop()
        assert norm(U - V) == pytest.approx(0, abs=tol)

        #import pdb;pdb.set_trace()
        # pi pulses only
        phi = 0
        U = qit.utils.Rz(-np.pi / 3) @ seq.nmr([[-np.pi, phi]]).to_prop()
        V = seq.knill(phi).to_prop()
        assert norm(U - V) == pytest.approx(0, abs=tol)

        # decoupling
        V = seq.dd('wait', 2.0).to_prop()
        assert norm(qit.I - V) == pytest.approx(0, abs=tol)
        V = seq.dd('hahn', 2.0).to_prop()
        assert norm(-1j * qit.sx - V) == pytest.approx(0, abs=tol)
        V = seq.dd('cpmg', 2.0).to_prop()
        assert norm(-qit.I - V) == pytest.approx(0, abs=tol)
        V = seq.dd('uhrig', 2.0, n=3).to_prop()
        assert norm(1j * qit.sx - V) == pytest.approx(0, abs=tol)
        V = seq.dd('xy4', 2.0).to_prop()
        assert norm(-qit.I - V) == pytest.approx(0, abs=tol)

    def test_propagation(self, tol):
        """Equivalent propagation using two different methods."""

        theta = np.pi * rand()
        phi = 2 * np.pi * rand()

        rho = qit.State(rand_positive(2))
        s = seq.scrofulous(theta, phi)
        rho1 = rho.u_propagate(s.to_prop())
        out, t = seq.propagate(rho, s, base_dt=1)
        rho2 = out[-1]
        assert (rho1 - rho2).norm() == pytest.approx(0, abs=tol)
