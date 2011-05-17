# -*- coding: utf-8 -*-
# Author: Ville Bergholm 2011
"""Multilinear maps module. Serves as a basis for state."""


from __future__ import print_function, division
import sys
import numpy as np

from base import tol


def error(msg, ex=None):
    """Print an error message, raise an exception."""
    print(msg, file=sys.stderr)
    if ex != None:
        raise ex


class lmap(object):
    """Multilinear maps between tensor products of finite-dimensional Hilbert spaces.

    Contains both the order-2 tensor and the dimensional information.

    TODO Another possible interpretation of lmap would be to
    treat each subsystem as an index, with the subsystems within dim{1} and dim{2}
    corresponding to contravariant and covariant indices, respectively?

    Variables:
    data:  ndarray of tensor data
    dim:   tuple of input and output dimension tuples, big-endian: ((out), (in))
    """

# TODO copy constructor, with or without dim change
# TODO make the data members "read-only":  @property def data(self): return self._data
# TODO def __format__(self, format_spec)
# TODO linalg efficiency: copy vs. view vs. reference?
# TODO sparse matrices?

    def __init__(self, s, dim=None):
        """Construct an lmap.

        Calling syntax                     dim
        ==============                     ============
        lmap(rand(a))                      ((a,), (1,))      1D array default: ket
        lmap(rand(a), ((1,), None)         ((1,), (a,))      bra given as a 1D array
        lmap(rand(a,b))                    ((a,), (b,))      2D array, dims inferred
        lmap(rand(4,b), ((2, 2), None))    ((2, 2), (b,))    state op, output: two qubits 
        lmap(rand(a,6), (None, (3, 2)))    ((a,), (3, 2))    state op, input: qutrit+qubit
        lmap(rand(6,6), ((3, 2), (2, 3)))  ((3, 2), (2, 3))  state op, all dims given

        lmap(y)                      (y is an lmap) copy constructor
        lmap(y, dim)                 (y is an lmap) copy constructor, reinterpret dimensions
        """

        # initialize the ndarray part
        self.data = np.array(s)

        # into a 2d array
        if self.data.ndim == 0:
            # scalar
            self.data.resize((1, 1))
        elif self.data.ndim == 1:
            # vector, ket by default
            self.data.resize((self.data.size, 1))
        elif self.data.ndim > 2:
            raise ValueError('Array dimension must be <= 2.')

        # now self.data.ndim == 2, always

        # set the dimensions: dim = ((out), (in))
        if dim == None:
            # infer both dimensions from s
            dim = (None, None)

        # is it a bra, yet given as a ket array?
        if dim[0] == (1,):
            self.data.resize((1, self.data.size))

        self.dim = []
        for k in range(len(dim)):
            if dim[k] == None:
                # not specified, infer from the data
                self.dim.append((self.data.shape[k],))
            else:
                self.dim.append(dim[k])
        self.dim = tuple(self.dim)

        # check dimensions
        if self.data.shape != tuple(map(np.prod, self.dim)):
            raise ValueError('Dimensions of the array do not match the combined dimensions of the subsystems.')


    def __repr__(self):
        """Display the lmap in a neat format."""
        out = ''
        # is it a vector? (a map with a singleton domain or codomain dimension)
        sh = self.data.shape
        if 1 in sh:
            # vector 
            # ket or bra?
            if sh[1] == 1:
                # let scalars be kets too
                dim = self.dim[0]
                is_ket = True
            else:
                dim = self.dim[1]
                is_ket = False

            # loop over all vector elements
            printed = 0
            d = np.prod(dim)
            for ind in range(d):
                # TODO with sparse arrays we could do better
                # sanity check, do not display lmaps with hundreds of terms
                if ind >= 128 or printed >= 20:
                    out += ' ...'
                    break

                temp = self.data.flat[ind]
                # make sure there is something to print
                if abs(temp) < tol:
                    continue

                printed += 1
                if abs(temp.imag) < tol:
                    # just the real part
                    out += ' {0:+.4g}'.format(temp.real)
                elif abs(temp.real) < tol:
                    # just the imaginary part
                    out += ' {0:+.4g}j'.format(temp.imag)
                else:
                    # both
                    out += ' +' + str(temp)

                # ket or bra symbol
                # FIXME this is insane
                temp = str(bytearray(map(lambda(x): x + ord('0'), np.unravel_index(ind, dim))))
                if is_ket:
                    out += ' |' + temp + '>'
                else:
                    out += ' <' + temp + '|'
        else:
            # matrix
            out = self.data.__repr__()

        out += '\ndim: ' + str(self.dim[0]) + ' <- ' + str(self.dim[1])
        return out


# utilities

    def remove_singletons(self):
        """Eliminate unnecessary singleton dimensions.

        NOTE: changes the object itself!
        """
        dd = []
        for d in self.dim[:]:
          temp = filter(lambda(x): x > 1, d)
          if len(temp) == 0:
              temp = (1,)
          dd.append(temp)
        self.dim = tuple(dd)
        return self


    def is_compatible(self, t):
        """True iff the lmaps have equal dimensions and can thus be added."""
        if not isinstance(t, lmap):
            raise TypeError('t is not an lmap.')
        return self.dim == t.dim



# linear algebra

    def conj(self):
        """Complex conjugate."""
        # complex conjugate data
        return lmap(self.data.conj(), self.dim)


    def transpose(self):
        """Transpose."""
        # swap dims
        dim = (self.dim[1], self.dim[0])
        # transpose data
        return lmap(self.data.transpose(), dim)


    def ctranspose(self):
        """Hermitian conjugate."""
        # swap dims
        dim = (self.dim[1], self.dim[0])
        # conjugate transpose data
        return lmap(self.data.conj().transpose(), dim)


    def __mul__(self, t):
        """Multiplication of lmaps by lmaps and scalars."""
        if isinstance(t, lmap):
            if self.dim[1] != t.dim[0]:
                raise ValueError('The dimensions do not match.')
            else:
                return lmap(np.dot(self.data, t.data), (self.dim[0], t.dim[1]))
        else:
            return lmap(self.data * t, self.dim)


    def __rmul__(self, t):
        """Multiplication of lmaps by scalars, reverse."""
        # scalars commute, lmaps already handled by __mul__
        return self.__mul__(t)


    def __truediv__(self, t):
        """Division of lmaps by scalars from the right."""
        return lmap(self.data / t, self.dim)


    def __add__(self, t):
        """Addition of lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The lmaps are not compatible.')
        return lmap(self.data + t.data, self.dim)


    def __sub__(self, t):
        """Subtraction of lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The lmaps are not compatible.')
        return lmap(self.data - t.data, self.dim)


    def __pow__(self, n):
        """Exponentiation of lmaps by integer scalars."""
        if self.dim[0] != self.dim[1]:
            raise ValueError('The dimensions do not match.')
        return lmap(np.linalg.matrix_power(self.data, n), self.dim)


    def __imul__(self, t):
        """In-place multiplication of lmaps by scalars from the right."""
        self.data *= t
        return self


    def __itruediv__(self, t):
        """In-place division of lmaps by scalars from the right."""
        self.data /= t
        return self


    def __iadd__(self, t):
        """In-place addition of lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The lmaps are not compatible.')
        s.data += t.data


    def __isub__(self, t):
        """In-place subtraction of lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The lmaps are not compatible.')
        s.data -= t.data


    def norm(self):
        """Matrix norm of the lmap."""
        return np.linalg.norm(self.data)


# subsystem ordering

    def reorder(self, *perm):
        """Change the relative order of the input and/or output subsystems.

        Returns a copy of the lmap with permuted subsystem order.

        A permutation can be either None (do nothing), a pair (a, b) of subsystems to be swapped,
        or a tuple containing a full permutation of the subsystems.

        reorder(None, (2, 1, 0))   ignore first index, reverse the order of subsystems in the second
        reorder((2, 5))            swap the subsystems 2 and 5 in the first index
        """
        dd = []
        total_d = []
        total_perm = []
        last_used_index = 0
        newdim = list(self.dim)

        # loop over indices
        for k in range(len(perm)):
            this_perm = perm[k]     # requested permutation for this index
            this_dim  = self.dim[k] # subsystem dims for this index

            # number of subsystems
            this_n = len(this_dim)

            # total dimension
            dd.append(np.prod(this_dim))

            temp = range(this_n)
            if this_perm == None:
                # no change
                # let the dimensions vector be, lump all subsystems in this index into one
                this_dim = (dd[k],)
                this_perm = [0]
                this_n = 1

            elif len(this_perm) == 2:
                # swap two subsystems
                temp[this_perm[0]] = this_perm[1]
                temp[this_perm[1]] = this_perm[0]
                this_perm = temp
                # reorder the dimensions vector
                newdim[k] = tuple(np.array(this_dim)[this_perm])

            else:
                # full permutation
                if len(set(temp) ^ set(this_perm)) != 0:
                    raise ValueError('Invalid permutation.')
                # reorder the dimensions vector
                newdim[k] = tuple(np.array(this_dim)[np.array(this_perm)])


            # big-endian ordering is more natural for users, but Matlab funcs
            # prefer little-endian, so we reverse it
            total_d.extend(this_dim)  #fliplr(this_dim)
            total_perm.extend(tuple(last_used_index + np.array(this_perm))) #fliplr(n -this_perm)
            last_used_index += this_n

        # tensor into another tensor which has one index per subsystem, permute dimensions, back into a tensor with the original number of indices
        return lmap(self.data.reshape(total_d).transpose(total_perm).reshape(dd), newdim)


    @staticmethod
    def test():
        """Test script for the lmap module.

        Ville Bergholm 2009-2010
        """
        import copy
        from numpy.testing import assert_almost_equal
        from numpy.random import rand, randn

        decimal = -int(np.log10(tol))

        # reordering subsystems
        idim = (2, 5, 3)
        odim = (4, 3, 2)
        A = lmap(rand(odim[0], idim[0]))
        B = lmap(rand(odim[1], idim[1]))
        C = lmap(rand(odim[2], idim[2]))
        T1 = tensor(A, B, C)

        p = (2, 0, 1)
        T2 = T1.reorder(p, p)
        assert_almost_equal((tensor(C, A, B) - T2).norm(), 0, decimal)

        p = (1, 0, 2)
        T2 = T1.reorder(p, p)
        assert_almost_equal((tensor(B, A, C) - T2).norm(), 0, decimal)

        p = (2, 1, 0)
        T2 = T1.reorder(p, p)
        assert_almost_equal((tensor(C, B, A) - T2).norm(), 0, decimal)

        print('All tests passed.')
        ignore = """
        a = lmap(randn(2,4) +1j*randn(2,4), (None, (2,2)))
        b = lmap(randn(2,2) +1j*randn(2,2))
        c = lmap(randn(2))

        d = lmap(rand(6,6), ((2,3),(2,3)))
        e = copy.deepcopy(d)

        print(repr(b * c))
        print(repr(tensor(a, b)))
        """


def tensor(*arg):
    """Tensor product of lmaps."""
    data = 1
    dout = []
    din  = []

    for k in arg:
        # concatenate dimensions
        dout += k.dim[0]
        din  += k.dim[1]
        # kronecker product of the data
        data = np.kron(data, k.data)

    s = lmap(data, (tuple(dout), tuple(din)))
    return s.remove_singletons()
