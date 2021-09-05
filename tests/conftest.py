"""
Shared pytest fixtures etc.
"""
import pytest

import qit


@pytest.fixture(scope='session')
def tol():
    """Numerical tolerance."""
    return qit.TOLERANCE
