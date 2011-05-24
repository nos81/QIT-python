# -*- coding: utf-8 -*-
# Author: Ville Bergholm 2011
"""Basic definitions module."""


from numpy import array, sqrt, finfo


# Pauli matrices
#I  = eye(2);
sx = array([[0, 1], [1, 0]])
sy = array([[0, -1j], [1j, 0]])
sz = array([[1, 0], [0, -1]])

# qubit projectors
p0 = array([[1, 0], [0, 0]])
p1 = array([[0, 0], [0, 1]])

# easy Hadamard
H = array([[1, 1], [1, -1]]) / sqrt(2)

# magic basis
Q_Bell = array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / sqrt(2)


# error tolerance
tol = max(1e-10, finfo(float).eps)

# some relevant physical constants (CODATA 2006)
#hbar = 1.054571628e-34 # J s
#kB   = 1.3806504e-23   # J/K
#eV   = 1.602176487e-19 # J
