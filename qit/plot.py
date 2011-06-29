# -*- coding: utf-8 -*-
"""Plots."""
# Ville Bergholm 2011


from __future__ import print_function, division

import numpy as np
from numpy import array, zeros, ones, sin, cos, tanh, dot, sort, pi, r_, c_, linspace, outer
from numpy.linalg import eigvals
import matplotlib.pyplot as plt

from state import *
from utils import copy_memoize, eigsort

__all__ = ['plot_adiabatic_evolution', 'plot_bloch_sphere', 'plot_pcolor',
           'asongoficeandfire', 'sphere']


def sphere(N=15):
    """X, Y, Z coordinate meshes for a unit sphere."""
    theta = linspace(0, pi, N)
    phi = linspace(0, 2*pi, 2*N)
    X = outer(sin(theta), cos(phi))
    Y = outer(sin(theta), sin(phi))
    Z = outer(cos(theta), ones(phi.shape))
    return X, Y, Z


def plot_adiabatic_evolution(t, st, H_func, n=4):
    """Adiabatic evolution plot.

    Input: vector t of time instances, cell vector st of states corresponding
    to the times and time-dependant Hamiltonian function handle H_func.

    Plots the energies of the eigenstates of H_func(t(k)) as a function of t(k),
    and the overlap of st{k} with the n lowest final Hamiltonian eigenstates. 
    Useful for illustrating adiabatic evolution.
    """
    # Jacob D. Biamonte 2008
    # Ville Bergholm 2009-2010

    T = t[-1]  # final time
    H = H_func(T)

    n = min(n, H.shape[0])
    m = len(t)

    # find the n lowest eigenstates of the final Hamiltonian
    #d, v = scipy.sparse.linalg.eigs(H, n, which = 'SR')
    #ind = d.argsort()  # increasing real part
    d,v = eigsort(H)
    lowest = []
    for j in range(n):
        #j = ind[j]
        lowest.append(state(v[:, -j]))
    # TODO with degenerate states these are more or less random linear combinations of the basis states... overlaps are not meaningful

    energies = zeros((m, H.shape[0]))
    overlaps = zeros((m, n))
    for k in range(m):
        tt = t[k]
        H = H_func(tt)
        energies[k, :] = sort(eigvals(H).real)
        for j in range(n):
            overlaps[k, j] = lowest[j].fidelity(st[k]) ** 2 # squared overlap with lowest final states

    plt.subplot(2,1,1)
    plt.plot(t/T, energies)
    plt.grid(True)
    plt.title('Energy spectrum')
    plt.xlabel('Adiabatic time')
    plt.ylabel('Energy')
    plt.axis([0, 1, np.min(energies), np.max(energies)])


    plt.subplot(2,1,2)
    plt.plot(t/T, overlaps) #, 'LineWidth', 1.7)
    plt.grid(True)
    plt.title('Squared overlap of current state and final eigenstates')
    plt.xlabel('Adiabatic time')
    plt.ylabel('Probability')
    temp = []
    for k in range(n):
        temp.append('$|{0}\\rangle$'.format(k))
    plt.legend(temp)
    plt.axis([0, 1, 0, 1])
    # axis([0, 1, 0, max(overlaps)])


def plot_bloch_sphere(s=None, ax=None):
    """Bloch sphere plot.

    Plots a Bloch sphere, a geometrical representation of the state space of a single qubit.
    Pure states are on the surface of the sphere, nonpure states inside it.
    The states |0> and |1> lie on the north and south poles of the sphere, respectively.

    s is a two dimensional state to be plotted.
    """
    # Ville Bergholm  2005-2011
    # James Whitfield 2010

    import mpl_toolkits.mplot3d

    if (ax == None):
        ax = plt.gcf().add_subplot(111, projection='3d')
    plt.hold(True)
    X, Y, Z = sphere()
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, color = 'g', alpha = 0.2, linewidth = 0) #cmap = xxx
    ax.axis('equal')
    ax.scatter(zeros(2), zeros(2), [1, -1], c = ['r', 'b'], marker = 'o')  # poles
    # TODO ax.scatter(*coord_array, c = ['r', 'b'], marker = 'o')  # poles
    # labels
    ax.text(0, 0,  1.1, '$|0\\rangle$')
    ax.text(0, 0, -1.2, '$|1\\rangle$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # plot a given state as well?
    if s != None:
        v = s.bloch_vector()
        ax.scatter(array([v[1]]), [v[2]], [v[3]], c = 'k', marker = 'x')

    plt.show()
    return ax



def plot_pcolor(W, a, b, clim=(0, 1)):
    """Easy pseudocolor plot.

    Plots the 2D function given in the matrix W.
    The vectors x and y define the coordinate grid.
    clim is an optional parameter for color limits.

    Returns the plot object.
    """
    # Ville Bergholm 2010

    # a and b are quad midpoint coordinates but pcolor wants quad vertices, so
    def to_quad(x):
        return (r_[x, x[-1]] + r_[x[0], x]) / 2

    plt.gcf().clf()  # clear the figure
    p = plt.pcolor(to_quad(a), to_quad(b), W, clim = clim, cmap = asongoficeandfire())
    plt.axis('equal')
    plt.axis('tight')
    #shading('interp')
    plt.colorbar()
    return p


def makemovie(filename, frameset, plot_func, *arg):
    """Create an AVI movie. FIXME
    aviobj = makemovie(filename, frameset, plot_func [, ...])

    Creates an AVI movie file named 'filename.avi' in the current directory.
    Frame k in the movie is obtained from the contents of the
    current figure after calling plot_func(frameset[k]).
    The optional extra parameters are passed directly to avifile.

    Returns the closed avi object handle.

    Example: makemovie('test', cell_vector_of_states, @(x) plot(x))
    """
    # James D. Whitfield 2009
    # Ville Bergholm 2009-2010

    # create an AVI object
    aviobj = avifile(filename, arg)

    fig = figure('Visible', 'off')
    for k in frameset:
        plot_func(k)
        aviobj = addframe(aviobj, fig)
        #  F = getframe(fig)   
        #  aviobj = addframe(aviobj, F)

    close(fig)
    aviobj = close(aviobj)


@copy_memoize
def asongoficeandfire(n=127):
    """Colormap with blues and reds. Wraps.

    Returns a matplotlib.colors.Colormap object.
    n is the number of color definitions in the map.
    """
    # Ville Bergholm 2010-2011

    from matplotlib import colors
    # exponent
    d = 3.1
    p = linspace(-1, 1, n)
    # negative values: reds
    x = p[p < 0]
    c = c_[1 -((1+x) ** d), 0.5*(tanh(4*(-x -0.5)) + 1), (-x) ** d]
    # positive values: blues
    x = p[p >= 0]
    c = r_[c, c_[x ** d, 0.5*(tanh(4*(x -0.5)) + 1), 1 -((1-x) ** d)]]
    return colors.ListedColormap(c, name='asongoficeandfire')
    # TODO colors.LinearSegmentedColormap(name, segmentdata, N=256, gamma=1.0)
