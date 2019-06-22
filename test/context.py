"""
Import context for unit tests, basic test class
-----------------------------------------------
"""

import unittest
import os
import sys

import numpy as np

np.set_printoptions(precision=4)

# Always import QIT from the local source tree, see https://docs.python-guide.org/en/latest/writing/structure/#test-suite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import qit

# tolerance for numerical errors
tol = 1e-10


class BaseTest(unittest.TestCase):
    """Base class for Dynamo unit tests."""

    def assertAlmostEq(self, first, second, msg=None):
        """Like assertAlmostEqual, but using an internal tolerance."""
        if first == second:
            return
        if np.abs(first - second) <= self.tol:
            return

        standardMsg = '{} != {} within {} delta'.format(first, second, self.tol)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)


    def assertAllAlmostEq(self, first, second, msg=None):
        """Like assertAlmostEq, but using np.all."""
        if (np.abs(first - second) <= self.tol).all():
            return

        standardMsg = '{} != {} within {} delta'.format(first, second, self.tol)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)
