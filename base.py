# -*- coding: utf-8 -*-
# Author: Ville Bergholm 2011
"""Basic definitions module."""


import numpy as np


# Pauli matrices
#I  = eye(2);
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

# qubit projectors
p0 = np.array([[1, 0], [0, 0]])
p1 = np.array([[0, 0], [0, 1]])

# easy Hadamard
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

# error tolerance
tol = max(1e-10, np.finfo(float).eps)

# some relevant physical constants (CODATA 2006)
#hbar = 1.054571628e-34 # J s
#kB   = 1.3806504e-23   # J/K
#eV   = 1.602176487e-19 # J
