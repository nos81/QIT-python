# -*- coding: utf-8 -*-
# Author: Ville Bergholm 2011
"""Quantum states module."""


import numbers


from numpy import array, sort, prod, cumprod, sqrt, trace, dot, vdot, roll, zeros, ones, r_, kron
import scipy as sp  # scipy imports numpy automatically
import scipy.linalg
from scipy.linalg import norm

from lmap import *
from utils import *


def warn(s):
    """Prints a warning."""
    print('Warning: ' + s)


def equal_dims(s, t):
    """True if s and t have equal dimensions."""
    return s.dims() == t.dims()


def index_muls(dim):
    """Index multipliers for C-ordered data"""
    muls = roll(cumprod(dim[::-1]), 1)[::-1]
    muls[-1] = 1  # muls == [d_{n-1}*...*d_1, d_{n-1}*...*d_2, ..., d_{n-1}, 1]
    return muls



class state(lmap):
    """Class for quantum states.

    Describes the state (pure or mixed) of a discrete, composite quantum system.
    The subsystem dimensions can be found in dim[0] (big-endian ordering).

    State class instances are special cases of lmaps. They have exactly two indices.
    If dim[1] == (1,), it is a ket representing a pure state.
    Otherwise both indices must have equal dimensions and the object represents a state operator.

    Does not require the state to be physical (it does not have to be trace-1, Hermitian, or nonnegative).

    Ville Bergholm 2008-2011
    """
    # by default, all state methods leave self unchanged 

    def __init__(self, s, dim=None):
        """Construct a state.
        
        calling syntax            result
        ==============            ======
        state('00101')            standard basis ket |00101> in a five-qubit system
        state('02', (2, 3))       standard basis ket |02> in a qubit+qutrit system
        state(k, (2, 3))          linearized standard basis ket |k> in a qubit+qutrit system, k must be an integer scalar
        state(rand(4))            ket, infer dim[0] == (4,)
        state(rand(4), (2, 2))    ket, two qubits
        state(rand(4,4))          state operator, infer dim[0] == (4,)
        state(rand(6,6), (3, 2))  state operator, qutrit+qubit
        state('GHZ', (2, 2, 2))   named states (in this case the three-qubit GHZ state)

        state(s)                  (s is a state) copy constructor
        state(s, dim)             (s is a state) copy constructor, redefine the dimensions

        The currently supported named states are
          GHZ (Greenberger-Horne-Zeilinger),
          W,
          Bell1, Bell2, Bell3, Bell4 
        """
        dim = tuple(dim)  # convert lists etc.
        if isinstance(s, lmap):
            # copy constructor
            # state vector or operator? (also works with dim == None)
            if s.is_ket():
                dim = (dim, (1,))
            else:
                if s.dim[0] != s.dim[1]:
                    raise ValueError('State operator must be square.')
                dim = (dim, dim)

            # call the lmap copy constructor
            lmap.__init__(self, s, dim)
            return

        elif isinstance(s, basestring):
            # string

            if str.isalpha(s[0]):
                # named state
                name = str.lower(s)

                if dim == None:
                    dim = (2, 2, 2) # default: three-qubit state

                n = len(dim) # subsystems
                s = zeros(prod(dim)) # ket

                if name in ('bell1', 'bell2', 'bell3', 'bell4'):
                    # Bell state
                    Q_Bell = array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / sqrt(2)
                    dim = (2, 2)
                    s = Q_Bell[:, chr(ord(name[-1]) - ord('1'))]
                elif name == 'ghz':
                    # Greenberger-Horne-Zeilinger state
                    s[0] = 1
                    s[-1] = 1
                elif name == 'w':
                    # W state
                    ind = 1
                    for k in reversed(range(n)):
                        s[ind * (dim[k] - 1)] = 1
                        ind *= dim[k]
                else:
                    raise ValueError("Unknown named state '{0}'.".format(name))

                s /= norm(s) # normalize
      
            else:
                # number string defining a standard basis ket
                if dim == None:
                    n = len(s)  # number of subsystems
                    dim = qubits(n)  # assume they're qubits

                # calculate the linear index
                s = numstr_to_array(s)
                if any(s >= dim):
                    raise ValueError('Invalid basis ket.')
                muls = index_muls(dim)
                ind = dot(muls, s)

                s = zeros(prod(dim)) # ket
                s[ind] = 1

            dim = (dim, (1,))  # ket

        elif isinstance(s, numbers.Number):
            # integer defining a standard basis ket
            if dim == None:
                raise ValueError('Need system dimension.')

            ind = s
            temp = prod(dim)  # total number of states
            if ind >= temp:
                raise ValueError('Invalid basis ket.')

            s = zeros(prod(dim)) # ket
            s[ind] = 1
            dim = (dim, (1,))  # ket

        else:
            # valid ndarray initializer representing a state vector or a state op
            s = array(s)
            if not 1 <= s.ndim <= 2:
                raise ValueError('State must be given as a state vector or a state operator.')

            # state vector or operator?
            if s.ndim == 2 and s.shape[1] != 1:
                if s.shape[0] != s.shape[1]:
                    raise ValueError('State operator matrix must be square.')
                dim = (dim, dim)  # op
            else:
                dim = (dim, (1,))  # ket

        # now s is an ndarray
        # call the lmap constructor
        lmap.__init__(self, s, dim)



# utility methods

    def subsystems(self):
        """Number of subsystems in the state."""
        return len(self.dim[0])


    def dims(self):
        """Dimensions of the state."""
        return self.dim[0] # dims of the other index must be equal (or 1)


    def clean_selection(self, sys):
        """Make a subsystem set unique and sorted, return it as an array. TODO valid too?"""
        return array(list(set(range(self.subsystems())).intersection(sys)), int)


    def invert_selection(self, sys):
        """Invert and sort a subsystem set."""
        return array(list(set(range(self.subsystems())).difference(sys)), int)


    def fix_phase(self, inplace=False):
        """Apply a global phase convention to a ket state.

        Returns a copy of the state. Additionally, if the state is represented with
        a ket, the copy has a global phase such that the first nonzero element in the
        state vector is real and positive.

        Ville Bergholm 2009-2011
        """
        s = self.inplacer(inplace)
        if s.is_ket():
            # apply the phase convention: first nonzero element in state vector is real, positive
            v = s.data
            for k in range(v.size):
                temp = v.flat[k]
                if abs(temp) > tol:
                    phase = temp / abs(temp)
                    v /= phase
                    break
        return s


    def normalize(self, inplace=False):
        """Normalize the state."""
        s = self.inplacer(inplace)
        if s.is_ket():
            s.data /= norm(s.data)
        else:
            s.data /= trace(s.data)
        return s


    def purity(self):
        """Purity of the state.

        Returns the purity of a normalized state, p = trace(\rho^2).
        Equivalent to linear entropy, S_l = 1-p.

        Ville Bergholm 2008-2011
        """
        if self.is_ket():
            return 1
        else:
            return trace(dot(self.data, self.data))


    def to_ket(self, inplace=False):
        """Convert the state representation into a ket (if possible).

        If the state is pure returns q, a copy of the state, for which the
        internal representation (q.data) is guaranteed to be a ket vector.

        Ville Bergholm 2009-2010
        """
        s = self.inplacer(inplace)
        if not s.is_ket():
            # state op
            if abs(s.purity() - 1) > tol:
                raise ValueError('The state is not pure, and thus cannot be represented by a ket vector.')

            d, v = eigsort(s.data)
            s.data = v[:, 0]  # corresponds to the highest eigenvalue, i.e. 1
            s.fix_phase(inplace = True)  # clean up global phase
            s.dim = (s.dim[0], (1,))

        return s


    def to_op(self, inplace=False):
        """Convert state representation into a state operator.

        Returns q, a copy of the state for which the internal representation 
        (q.data) is guaranteed to be a state operator.

        Ville Bergholm 2009-2010
        """
        s = self.inplacer(inplace)
        if s.is_ket():
            s.data = np.outer(s.data, s.data.conj())
            s.dim = (s.dim[0], s.dim[0])

        return s


    def trace(self):
        """Trace of the state operator.

        Returns the trace of the state operator of quantum state s.
        For a pure state this is equal to the squared norm of the state vector.

        Ville Bergholm 2008
        """
        if self.is_ket():
            return vdot(self.data, self.data)
        else:
            return trace(self.data)


    def ptrace(self, sys, inplace=False):
        """Partial trace.

        Returns the partial trace of the state
        over the subsystems listed in the vector sys.

        Ville Bergholm 2008-2010
        """
        s = self.to_op(inplace)
        dim = array(s.dims())
        n = s.subsystems()
        sys = s.clean_selection(sys)
        keep = s.invert_selection(sys)

        def tensorsum(a, b):
            #c = log(kron(exp(a), exp(b))) # a perverted way of doing it, the exp overflows...
            c = []
            for k in a:
                c.extend(k + b)
            return array(c)

        # big-endian (C) data ordering
        # we trace over the subsystems in order, starting from the first one
        # partial trace over single system j, performed for every j in sys
        d = list(dim)
        for j in sys:
            muls = index_muls(d)  # muls == [d_{n-1}*...*d_1, d_{n-1}*...*d_2, ..., d_{n-1}, 1]

            # build the index "stencil"
            inds = array([0])
            for k in range(n):
                if k == j:
                    continue
                inds = tensorsum(inds, r_[0 : muls[k] * d[k] : muls[k]])

            stride = muls[j] # stride for moving the stencil while summing
            temp = len(inds)
            res = zeros((temp, temp)) # result
            for k in range(d[j]):
                temp = inds + stride * k
                res += s.data[temp, temp]

            s.data = res # replace data
            d[j] = 1  # remove traced-over dimension.

        dim = dim[keep] # remove traced-over dimensions for good
        if len(dim) == 0:
            dim = (1,) # full trace gives a scalar
        else:
            dim = tuple(dim)

        s.dim = (dim, dim)
        return s


    def ptranspose(self, sys, inplace=False):
        """Partial transpose.

        Returns the partial transpose of the state
        wrt. the subsystems listed in the vector sys.

        Ville Bergholm 2008-2011
        """
        # TODO what about kets? can we do better?
        s = self.to_op(inplace)
        dim = s.dims()
        n = s.subsystems()
        # total dimension
        orig_d = s.data.shape
        # which systems to transpose
        sys = s.clean_selection(sys)

        # swap the transposed dimensions
        perm = np.arange(2 * n)  # identity permutation
        perm[r_[sys, sys + n]] = perm[r_[sys + n, sys]]

        # flat matrix into tensor, partial transpose, back into a flat matrix
        s.data = s.data.reshape(dim + dim).transpose(perm).reshape(orig_d)
        return s


    def reorder(self, perm, inplace=False):
        """Change the relative order of subsystems in a state.
        reorder([2 1 0])  reverse the order of subsystems
        reorder([2 5])    swap subsystems 2 and 5

        Reorders the subsystems of the state according to the permutation perm.

        The permutation vector may consist of either exactly two subsystem numbers
        (to be swapped), or a full permutation of subsystem numbers.

        Ville Bergholm 2010
        """
        # this is just an adapter for lmap.reorder
        if self.is_ket():
            return super(state, self).reorder((perm, None), inplace = inplace)
        else:
            return super(state, self).reorder((perm, perm), inplace = inplace)



# physics methods

    def ev(self, A):
        """Expectation value of an observable in the state.

        Returns the expectation value of the observable A in the state.
        A has to be Hermitian.

        Ville Bergholm 2008
        """
        if self.is_ket():
            # state vector
            x = vdot(self.data, dot(A, self.data))
        else:
            # state operator
            x = trace(dot(A, self.data))
        return x.real # Hermitian observable


    def var(self, A):
        """Variance of an observable in the state.

        Returns the variance of the observable A in the state.
        A has to be Hermitian.

        Ville Bergholm 2009
        """
        return self.ev(A**2) - self.ev(A)**2


    def prob(self):
        """Measurement probabilities of the state in the computational basis.

        Returns a vector of probabilities of finding a system
        in each of the different states of the computational basis.

        Ville Bergholm 2009
        """
        if self.is_ket():
            return np.absolute(self.data) ** 2
        else:
            return diag(self.data)


    def projector(self):
        """Projection operator defined by the state.

        Returns the projection operator P defined by the state.
        TODO remove?
        Ville Bergholm 2009-2010
        """
        s = self.to_op()
        return lmap(s.data, s.dim)


    def u_propagate(self, U):
        """Propagate the state using a unitary.

        Propagates the state using the unitary propagator U,
        returns the resulting state.

        Ville Bergholm 2009-2010
        """
        if isinstance(U, lmap):
            if self.is_ket():
                return state(U * self)
            else:
                return state(U * self * U.ctranspose())
        elif isinstance(U, np.ndarray):
            # U is a matrix, dims do not change. could also construct an lmap here...
            if self.is_ket():
                return state(dot(U, self.data), self.dims())
            else:
                return state(dot(dot(U, self.data), U.conj().transpose()), self.dims())
        else:
            raise TypeError('States can only be propagated using lmaps and arrays.')


    def kraus_propagate(self, E):
        """Apply a quantum operation to the state.

        Applies the quantum operation E to the state.
        E == [E_1, E_2, ...] is a set of Kraus operators.

        Ville Bergholm 2009
        """
        # TODO allow the user to apply E only to some subsystems of s0
        n = len(E)
        # TODO: If n > prod(dims(s))^2, there is a simpler equivalent
        # operation. Should the user be notified?
        #temp = 0
        #for k in E:
        #    temp += dot(k.ctranspose(), k)
        #if norm(temp - eye(temp.shape)) > qit.tol:
        #    warn('Unphysical quantum operation.')
        if self.is_ket():
            if n == 1:
                return self.u_propagate(E[0]) # remains a pure state

        s = self.to_op()
        q = 0
        for k in E:
            q += u_propagate(s, k)
        return q


    def measure(self, M=None, perform=False, discard=False):
        """Quantum measurement.

        [p, res, s]
        = measure(s)                 measure the entire system projectively
        = measure(s, [1 4])          measure subsystems 1 and 4 projectively
        = measure(s, {M1, M2, ...})  perform a general measurement
        = measure(s, A)              measure a Hermitian observable A

        Performs a quantum measurement on the state.

        If no M is given, a full projective measurement in the
        computational basis is performed.

        If a vector of subsystems is given as the second parameter, only
        those subsystems are measured, projectively, in the
        computational basis.

        A general measurement may be performed by giving a complete set
        of measurement operators [M1, M2, ...] as the second parameter.

        Finally, if the second parameter is a Hermitian matrix A, the
        corresponding observable is measured. In this case the second
        column of p contains the eigenvalue of A corresponding to each
        measurement result.

        p = measure(...) returns the vector p, where p[k] is the probability of
        obtaining result k in the measurement. For a projective measurement
        in the computational basis this corresponds to the ket |k>.

        [p, res] = measure(...) additionally returns the index of the result of the
        measurement, res, chosen at random following the probability distribution p.
 
        [p, res, s] = measure(...) additionally gives s, the collapsed state
        corresponding to the measurement result res.

        Ville Bergholm 2009-2010
        """
        def rand_measure(p):
            """Random measurement using the prob. distribution p."""
            return find(rand() <= cumsum(p))[0]


        skip = """

        def build_stencil(j, q, dims, muls):
            ""build projector to state j (diagonal because we project into the computational basis)""
            stencil = ones(dims[0]) # first identity
            for k in range(q):
                temp = sparse(1, dims(2*k))
                temp(mod(floor((j-1)/muls(k)), dims(2*k))+1) = 1 # projector
                stencil = kron(kron(stencil, temp), ones(1, dims(2*k+1))) # identity

            return stencil


        d = self.dims()
        res = s = None

        if M == None:
            # full measurement in the computational basis
            p = self.prob()  # probabilities 
            if perform:
                res = rand_measure(p)
                if collapse:
                    s = state(res, d) # collapsed state
            return p, res, s

        elif isinstance(M, (list, tuple)):
            if isinstance(M[0], numbers.Number):
                # measure a set of subsystems in the computational basis
                sys = self.clean_selection(M)

                # dimensions of selected subsystems and identity ops between them
                # TODO sequential measured subsystems could be concatenated as well
                q = len(sys)
                dims = []
                ppp = 0  # first sys not yet included
                for k in sys:
                    dims.append(prod(d[ppp:k])) # identity
                    dims.append(d[k]) # selected subsys
                    ppp = k+1

                dims.append(prod(d[ppp:])) # last identity

                # big-endian ordering is more natural for users, but little-endian more convenient for calculations
                muls = roll(cumprod(d[sys][::-1]), 1)[::-1]
                m = muls[-1] # number of possible results == prod(d[sys])
                muls[-1] = 1 # now muls == [..., d_s{q-1}*d_s{q}, d_s{q}, 1]

                # sum the probabilities
                born = self.prob()
                for j in range(m)
                    stencil = build_stencil(j, q, dims, muls)
                    p(j) = stencil*born # inner product

                if perform:
                    res = rand_measure(p)
                    if collapse:
                        R = build_stencil(res, q, dims, muls) # each projector is diagonal, hence we only store the diagonal

                        if discard:
                            # discard the measured subsystems from s
                            d(sys) = []
                            keep = find(R)  # indices of elements to keep
        
                            if self.is_ket():
                                s.data = s.data(keep) / sqrt(p(res)) # collapsed state
                            else:
                                s.data = s.data(keep, keep) / p(res) # collapsed state

                            s = state(s, d)
                        else:
                            if self.is_ket():
                                s.data = R.conj().transpose() .* s.data / sqrt(p(res)) # collapsed state
                            else:
                                s.data = (R..conj().transpose()*R) .* s.data / p(res) # collapsed state, HACK
                            end
                        end
                    end
                end

            else:
                # otherwise use set M of measurement operators (assumed complete!)
                m = length(M)

                # probabilities
                if (size(s.data, 2) == 1):
                    # state vector
                    for k=1:m:
                        p(k) = s.data' * M{k}' * M{k} * s.data
                    end
                    if (nargout >= 2):
                        res = rand_measure(p)
                        if (nargout >= 3):
                            s.data = M{res} * s.data / sqrt(p(res)) # collapsed state
                        end
                    end
                else:
                    # state operator
                    for k=1:m:
                        p(k) = trace(M{k}.ctranspose() * M{k} * s.data) # TODO wasteful
                    end
                    if (nargout >= 2):
                        res = rand_measure(p)
                        if (nargout >= 3):
                            s.data = M{res} * s.data * M{res}.ctranspose() / p(res) # collapsed state
                        end
                    end
                end

        elif isinstance(M, np.ndarray):
            # M is a matrix
            # measure the given Hermitian observable
            a, P = spectral_decomposition(M)
            m = len(a)  # number of possible results

            p = zeros((m, 2))
            for k in range(m):
                p[k, 0] = self.ev(P[k])
            p[:, 1] = a  # also return the corresponding results

            if perform:
                res = rand_measure(p)
                if collapse:
                    ppp = P[res]
                    s = deepcopy(self)
                    if self.is_ket():
                        s.data = ppp * s.data / sqrt(p[res], 0) # collapsed state
                    else:
                        s.data = ppp * s.data * ppp / p[res, 0] # collapsed state
        else:
            raise ValueError('Unknown input type.')
"""



# quantum information methods

    def fidelity(self, r):
        """Fidelity of two states.

        Fidelity of two state operators \rho and \sigma is defined as
        $F(\rho, \sigma) = \trace \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}}$.
        For state vectors this is equivalent to the overlap, F = |<a|b>|.

        Fidelity is symmetric in its arguments and bounded in the interval [0,1].

        Ville Bergholm 2009-2010
        %! M.A. Nielsen, I.L. Chuang, "Quantum Computation and Quantum Information" (2000), chapter 9.2.2
        """
        if not isinstance(r, state):
            raise TypeError('Not a state.')

        if self.is_ket():
            if r.is_ket():
                return abs(vdot(self.data, r.data))
            else:
                return sqrt(vdot(self.data, dot(r.data, self.data)).real)
        else:
            if r.is_ket():
                return sqrt(vdot(r.data, dot(self.data, r.data)).real)
            else:
                temp = sp.linalg.sqrtm(self.data)
                return trace(sp.linalg.sqrtm(dot(dot(temp, r.data), temp))).real


    def trace_dist(self, r):
        """Trace distance of two states.

        Trace distance between state operators r and s is defined as
        $D(r, s) = 0.5*\trace(\sqrt{A^\dagger * A})$, where $A = r-s$.

        Equivalently $D(r, s) = 0.5*\sum_k |\lambda_k|$, where $\lambda_k$
        are the eigenvalues of A (since A is Hermitian).

        Ville Bergholm 2009
        %! M.A. Nielsen, I.L. Chuang, "Quantum Computation and Quantum Information" (2000), chapter 9.2.1
        """
        if not isinstance(r, state):
            raise TypeError('Not a state.')

        r = r.to_op()
        s = self.to_op()
        # TODO could use original data
        A = r.data - s.data
        return 0.5 * sum(abs(np.linalg.eigvals(A)))
        #return 0.5*trace(sqrtm(A'*A))



    def schmidt(self, sys=None, full=False):
        """Schmidt decomposition.
        lambda = schmidt(sys)
        lambda, u, v = schmidt(sys, full=True)

        Calculates the Schmidt decomposition of the (pure) state.
        Subsystems listed in vector sys constitute part A, the rest forming part B.
        Vector lambda will contain the Schmidt coefficients.

        If required, matrices u and v will contain the corresponding orthonormal
        Schmidt bases for A and B, respectively, as column vectors, i.e.
        \ket{k}_A = u[:, k], \ket{k}_B = v[:, k].
        The state is then given by \sum_k \lambda_k \ket{k}_A \otimes \ket{k}_B

        %! M.A. Nielsen, I.L. Chuang, "Quantum Computation and Quantum Information" (2000), chapter 2.5.
        Ville Bergholm 2009-2010
        """
        dim = self.dims()
        n = self.subsystems()

        if sys == None:
            if n == 2:
                # reasonable choice
                sys = (0,)
            else:
                raise ValueError('Requires a state and a vector of subsystems.')

        try:
            s = self.to_ket()
        except ValueError:
            raise ValueError('Schmidt decomposition is only defined for pure states.')

        # complement of sys, dimensions of the partitions
        sys = s.clean_selection(sys)
        compl = s.invert_selection(sys)
        d1 = prod(dim[sys])
        d2 = prod(dim[compl])
        perm = r_[sys, compl]

        if all(perm == range(n)):
            # nothing to do
            pass
        else:
            # reorder the system according to the partitioning
            s.reorder(perm, inplace = True)

        # order the coefficients into a matrix, take an svd
        if not full:
            return sp.linalg.svdvals(s.data.reshape(d1, d2))
        else:
            u, s, vh = sp.linalg.svd(s.data.reshape(d1, d2), full_matrices = False)
            # note the definition of vh in svd
            return s, u, vh.transpose()


    def entropy(self, sys=None):
        """Von Neumann entropy of the state.
        S = entropy(s)       entropy
        S = entropy(s, sys)  entropy of entanglement

        Returns the entropy S of the state.

        If a vector of subsystem indices sys is given, returns the
        entropy of entanglement of the state wrt. the partitioning
        defined by sys.

        Entropy of entanglement is only defined for pure states.

        S(\rho) = -\trace(\rho * \log_2(\rho))

        Ville Bergholm 2009-2010
        """
        if sys != None:
            s = ptrace(to_ket(s), sys) # partial trace over one partition

        if self.is_ket():
            return 0
        else:
            p = np.linalg.eigvals(self.data)
            p[p == 0] = 1   # avoid trouble with the logarithm
            return -dot(p, np.log2(p))


    def concurrence(self, sys=None):
        """Concurrence of the state.

        Returns the concurrence of the state s wrt. the partitioning
        given by the listing of subsystems in the vector sys.

        %! W.K. Wootters, "Entanglement of Formation of an Arbitrary State of Two Qubits", PRL 80, 2245 (1998).
        %! R. Horodecki, P. Horodecki, M. Horodecki, K. Horodecki, "Quantum entanglement", arXiv:quant-ph/0702225 (2007).
        Ville Bergholm 2006-2010
        """
        if abs(self.trace() - 1) > tol:
            warn('State not properly normalized.')

        dim = self.dims()

        if sys != None:
            # concurrence between a qubit and a larger system
            if not(len(sys) == 1 and dim[sys] == 2):
                raise ValueError('Concurrence only defined between a qubit and another system.')

            if abs(self.purity() - 1) > tol:
                raise ValueError('Not a pure state.')

            # pure state
            n = len(dim)
            rho_A = self.ptrace(self.invert_selection(sys)) # trace over everything but sys
            C = 2 * sqrt(real(det(rho_A.data))) # = sqrt(2*(1-real(trace(temp*temp))))
            return

        # concurrence between two qubits
        if self.subsystems() != 2 or any(dim != array([2, 2])):
            # not a two-qubit state
            raise ValueError('Not a two-qubit state.')

        X = kron(sy, sy)
        p = self.data
        if self.is_ket():
            # ket
            C = abs(dot(dot(p.transpose(), X), p))

            # find the coefficients a of the state ket in the magic base
            # phi+-, psi+-,  = triplet,singlet
            #bell = [1 i 0 0 0 0 i 1 0 0 i -1 1 -i 0 0]/sqrt(2)
            #a = bell'*p
            #C = abs(sum(a ** 2))
        else:
            # state operator
            temp = dot(p, X)  # X.conj() == X so this works
            temp = dot(temp, temp.conj())  # == p * X * conj(p) * X
            if abs(purity(self) - 1) > tol:
                L = sqrt(sort(np.linalg.eigvals(temp).real)[::-1]).real
                C = max(0, L[1] -L[2] -L[3] -L[4])
            else:
                C = sqrt(trace(temp).real) # same formula as for state vecs


    def negativity(self, sys):
        """Negativity of the state.

        Returns the negativity of the state wrt. the partitioning
        given by the listing of subsystems in the vector sys.

        %! A. Peres, "Separability Criterion for Density Matrices", PRL 77, 1413 (1996).
        %! M. Horodecki et al., "Separability of Mixed States: Necessary and Sufficient Conditions", Physics Letters A 223, 1-8 (1996).
        Ville Bergholm 2008
        """
        s = self.ptranspose(sys)  # partial transpose the state
        x = sp.linalg.svdvals(s.data)  # singular values
        return (sum(sqrt(x)) - 1) / 2


    def lognegativity(self, sys):
        """Logarithmic negativity of the state.

        Returns the logarithmic negativity of the state wrt. the partitioning
        given by the listing of subsystems in the vector sys.

        Ville Bergholm 2008
        """
        return log2(2 * self.negativity(sys) + 1)


    def scott(self, m):
        """Scott's average bipartite entanglement measure.

        Returns the vector Q containing the terms of the Scott entanglement measure
        of the system for partition size m.

        When m = 1 this is coincides with the Meyer-Wallach entanglement measure.

        %! P.J. Love et al., "A characterization of global entanglement", arXiv:quant-ph/0602143 (2006).
        %! A.J. Scott, "Multipartite entanglement, quantum-raise ValueError-correcting codes, and entangling power of quantum evolutions", PRA 69, 052330 (2004).
        %! D.A. Meyer and N.R. Wallach, "Global entanglement in multiparticle systems", J. Math. Phys. 43, 4273 (2002).

        Jacob D. Biamonte 2008
        Ville Bergholm 2008-2010
        """
        dim = self.dims()
        n = self.subsystems()

        def nchoosek(n, k):
            """List of k-combination lists of range(n)."""
            # FIXME probably horribly inefficient, but sp.comb doesn't do this.
            # power set of range(n)
            temp = [[i for i in range(n) if x & (1 << i)] for x in range(2**n)]
            # subset of elements with the correct length
            return [i for i in temp if len(i) == k]

        S = nchoosek(n, m)  # all m-combinations of n subsystems

        D = min(dim) # FIXME correct for arbitrary combinations of qudits??
        C = (D**m / (D**m - 1)) / sp.comb(n, m)  # normalization

        Q = []
        all_systems = set(range(n))
        for sys in S:
            temp = self.ptrace(all_systems.difference(sys))  # trace over everything except S_k
            # NOTE: For pure states, tr(\rho_S^2) == tr(\rho_{\bar{S}}^2),
            # so for them we could just use self.ptrace(sys) here.
            Q.append(C * (1 - trace(np.linalg.matrix_power(temp.data, 2))))
        return Q


    def locc_convertible(self, t, sys):
        """LOCC convertibility of states.

        For bipartite pure states s and t, returns true if s can be converted to t
        using local operations and classical communication (LOCC).
        sys is a vector of subsystems defining the partition.

        Ville Bergholm 2010
        %! M.A. Nielsen, I.L. Chuang, "Quantum Computation and Quantum Information" (2000), chapter 12.5.1
        """
        if not equal_dims(self, t):
            raise ValueError('States must have equal dimensions.')

        try:
            s = self.to_ket()
            t = t.to_ket()
        except ValueError:
            raise ValueError('Not implemented for nonpure states.')

        s.ptrace(sys, inplace = True)
        t.ptrace(sys, inplace = True)
        return majorize(np.linalg.eigvals(s.data), np.linalg.eigvals(t.data))



    def plot(self):
        """State tomography plot.

        Plots the probabilities of finding a system in this state
        in the different computational basis states upon measurement.
        Relative phases are represented by the colors of the bars.

        If the state is nonpure, also plots the coherences.

        Ville Bergholm 2009-2010
        """
        dim = self.dims()
        n = self.subsystems()

        # prepare labels
        m = min(n, 3)  # at most three symbols
        d = dim[:m]
        nd = prod(d)
        rest = '0' * (n-m) # the rest is all zeros
        ticklabels = []
        for k in range(nd):
            temp = array_to_numstr(np.unravel_index(k, d))
            ticklabels.append(array_to_numstr(ket) + rest)

        ntot = prod(dim)
        skip = ntot / nd  # only every skip'th state gets a label to avoid clutter
        ticks = r_[0 : ntot : skip]

        N = s.data.shape[0]
        Ncol = 127 # color resolution (odd to fix zero phase at the center of a color index)
        colormap(circshift(hsv(Ncol), floor(Ncol/6))) # the hsv map wraps (like phase)

        def phases(A):
            """Phase normalized to (0,1]"""
            return 0.5 * ((angle(A) / pi) + 1)

        c = phases(s.data)
        if self.is_ket():
            s = self.fix_phase()

            h = bar(range(N), s.prob())
            xlabel('Basis state')
            ylabel('Probability')
            set(gca,'XTick', ticks)
            set(gca,'XTickLabel', ticklabels)
            axis('tight')

            # color bars using phase data
            ch = get(h,'Children')
            fvd = get(ch,'Faces')
            fvcd = get(ch,'FaceVertexCData')
            for b in range(N):
                fvcd[fvd[b, :]] = c[b] # all four vertices of a bar have same color
            set(ch,'FaceVertexCData',fvcd)
            set(ch,'EdgeColor','k')
        else:
            h = bar3(abs(s.data))
            xlabel('Col state')
            ylabel('Row state')
            zlabel('|\rho|')
            set(gca,'XTick', ticks+1)
            set(gca,'XTickLabel', ticklabels)
            set(gca,'YTick', ticks+1)
            set(gca,'YTickLabel', ticklabels)
            axis('tight')
            #alpha(0.8)

            # color bars using phase data
            for m in range(len(h)):
                # get color data
                cdata = get(h[m], 'Cdata') # [one row of 3d bars * six faces, four vertices per face]
                for k in range(size(cdata, 1) / 6):
                    j = 6*k
                    cdata[j:j+6, :] = c[k, m] # all faces are the same color
                set(h[m], 'Cdata', cdata)

        set(gca, 'CLim', [0, 1]) # color limits

        hcb = colorbar('YTick', linspace(0, 1, 5))
        set(hcb, 'YTickLabel', ['-\pi', '-\pi/2', '0', '\pi/2', '\pi'])


        skip = """
function out = propagate(s, H, t, varargin)
% PROPAGATE  Propagate the state continuously in time.
%
%  out = propagate(s, H, t [, out_func, odeopts])
%  out = propagate(s, L, t [, out_func, odeopts])
%  out = propagate(s, {H, {A_i}}, t [, out_func, odeopts])
%
%  Propagates the state s using the given generator(s) for the time t,
%  returns the resulting state.
%
%  The generator can either be a Hamiltonian matrix H or, for time-dependent
%  Hamiltonians, a function handle H(t) which takes a time instance t
%  as input and return the corresponding H matrix.
%
%  Alternatively, the generator can also be a Liouvillian superoperator, or
%  a list consisting of a Hamiltonian and a list of Lindblad operators.
%
%  If t is a vector of increasing time instances, returns a cell array
%  containing the propagated state for each time given in t.
%
%  Optional parameters (can be given in any order):
%    out_func: If given, for each time instance propagate returns out_func(s(t), H(t)).
%    odeopts:  Options struct for MATLAB ODE solvers from the odeset function.

%  out == expm(-i*H*t)*|s>
%  out == inv_vec(expm(L*t)*vec(\rho_s))

% Ville Bergholm 2008-2010
% James Whitfield 2009


if (nargin < 3)
  raise ValueError('Needs a state, a generator and a time.')
end

out_func = @(x,h) x % if no out_func is given, use a NOP

odeopts = odeset('RelTol', 1e-4, 'AbsTol', 1e-6, 'Vectorized', 'on')

n = length(t) % number of time instances we are interested in
out = cell(1, n)
dim = size(s.data) % system dimension

if (isa(H, 'function_handle'))
  % time dependent
  t_dependent = true
  F = H
  H = F(0)
else
  % time independent
  t_dependent = false
end

if (isnumeric(H))
  % matrix
  dim_H = size(H, 2)

  if (dim_H == dim(1))
    gen = 'H'
  elseif (dim_H == dim(1) ** 2)
    gen = 'L'
    s = to_op(s)
  else
    raise ValueError('Dimension of the generator does not match the dimension of the state.')
  end
  
elseif (iscell(H))
  % list of Lindblad operators
  dim_H = size(H{1}, 2)
  if (dim_H == dim(1))
    gen = 'A'
    s = to_op(s)

    % HACK, in this case we use ode45 anyway
    if (~t_dependent)
      t_dependent = true 
      F = @(t) H % ops stay constant
    end
  else
    raise ValueError('Dimension of the Lindblad ops does not match the dimension of the state.')
  end

else
  raise ValueError(['The second parameter has to be either a matrix, a cell array, '...
         'or a function handle that returns a matrix or a cell array.'])
end

dim = size(s.data) % may have been switched to operator representation


% process optional arguments
for k=1:nargin-3
  switch class(varargin{k})
    case 'function_handle'
      out_func = varargin{k}

    case 'struct'
      odeopts = odeset(odeopts, varargin{k})

    otherwise
      raise ValueError('Unknown optional parameter type.')
  end
end


% derivative functions for the solver

function d = lindblad_fun(t, y, F)
  X = F(t)
  A = X{2}
  A = A(:)

  d = zeros(size(y))
  % lame vectorization
  for loc1_k=1:size(y, 2)
    x = reshape(y(:,loc1_k), dim) % into a matrix
  
    % Hamiltonian
    temp = -1i * (X{1} * x  -x * X{1})
    % Lindblad operators
    for j=1:length(A)
      ac = A{j}'*A{j}
      temp = temp +A{j}*x*A{j}' -0.5*(ac*x + x*ac)
    end
    d(:,loc1_k) = temp(:) % back into a vector
  end
end

function d = mixed_fun(t, y, F)
  H = F(t)
  
  d = zeros(size(y))
  % vectorization
  for loc2_k=1:size(y, 2)
    x = reshape(y(:,loc2_k), dim) % into a matrix
    temp = -1i * (H * x  -x * H)
    d(:,loc2_k) = temp(:) % back into a vector
  end
end


if (t_dependent)
  % time dependent case, use ODE solver

  switch (gen)
    case 'H'
      % Hamiltonian
      if (dim(2) == 1)
        % pure state
        odefun = @(t, y, F) -1i * F(t) * y % derivative function for the solver
      else
        % mixed state
        odefun = @mixed_fun
      end

    case 'L'
      % Liouvillian
      odefun = @(t, y, F) F(t) * y
      %odeopts = odeset(odeopts, 'Jacobian', F)

    case 'A'
      % Hamiltonian and Lindblad operators in a list
      odefun = @lindblad_fun
  end

  skip = 0
  if (t(1) ~= 0)
    t = [0, t] % ODE solver needs to be told that t0 = 0
    skip = 1
  end

  if (length(t) < 3)
    t = [t, t(end)+1e5*eps] % add another time point to get reasonable output from solver
  end

  %odeopts = odeset(odeopts, 'OutputFcn', @(t,y,flag) odeout(t, y, flag, H))

  [t_out, s_out] = ode45(odefun, t, s.data, odeopts, F)
  % s_out has states in columns, row i corresponds to t(i)

  % apply out_func
  for k=1:n
    % this works because ode45 automatically expands input data into a col vector
    s.data = inv_vec(s_out(k+skip,:), dim)
    out{k} = out_func(s, F(t_out(k+skip)))
  end

else
  % time independent case

  switch (gen)
    case 'H'
      if (length(H) < 500)
        % eigendecomposition
        [v, d] = eig(full(H)) % TODO eigs?
        d = diag(d)
        for k=1:n
          U = v * diag(exp(-1i * t(k) * d)) * v'
          out{k} = out_func(u_propagate(s, U), H)
          %out{k} = out_func(u_propagate(s, expm(-i*H*t(k))), H)
        end
      else
        % Krylov subspace method
        [w, err] = expv(-1i*t, H, s.data)
        for k=1:n
          s.data = w(:,k)
          out{k} = out_func(s, H)
        end
      end
      
    case 'L'
      % Krylov subspace method
      [w, err] = expv(t, H, vec(s.data))
      for k=1:n
        s.data = inv_vec(w(:,k))
        out{k} = out_func(s, H)
      end
  end
end

if (n == 1)
  % single output, don't bother with a list
  out = out{1}
end
end


%function status = odeout(t, y, flag, H)
%if isempty(flag)
%  sdfsd
%end
%status = 0
%end






function [out, t] = seq_propagate(s, seq, out_func)
% SEQ_PROPAGATE  Propagate the state in time using a control sequence.
%  [out, t] = propagate(s, seq, out_func)
    
% Ville Bergholm 2009-2010


global qit

if (nargin < 3)
    out_func = @(x) x % no output function given, use a NOP
    
    if (nargin < 2)
        raise ValueError('Needs a stuff')
    end
end


base_dt = 0.1
n = size(seq, 1) % number of pulses
t = [0]
out{1} = out_func(s)

for k=1:n
    H = 0.5*(qit.sx*seq(k, 1) +qit.sy*seq(k, 2) +qit.sz*seq(k, 3))
    T = seq(k, end)
    
    n_steps = ceil(T/base_dt)
    dt = T/n_steps

    U = expm(-i*H*dt)
    for j=1:n_steps
        s = u_propagate(s, U)
        out{end+1} = out_func(s)
    end

    temp = t(end)
    t = [t, linspace(temp+dt, temp+T, n_steps)]
end

"""


# other state representations

    def bloch_vector(self):
        """Generalized Bloch vector.

        Returns the generalized Bloch vector A corresponding to the state.

        For an n-subsystem state the generalized Bloch vector is an order-n correlation
        tensor defined in terms of the standard Hermitian tensor basis B
        corresponding to state dimensions:

        A_{ijk...} == \sqrt(D) * \trace(\rho_s  B_{ijk...}),

        where D = prod(self.dims()). A is always real since \rho_s is Hermitian.
        For valid states norm(A) <= sqrt(D) (e.g. for a qubit system norm(A) <= 2).

        Ville Bergholm 2009-2011
        """
        dim = self.dims()
        G = tensorbasis(dim)
        a = []
        for g in G:
            a.append(self.ev(g))
        a = array(a) * sqrt(prod(dim)) # to match the usual Bloch vector normalization

        # into an array, one dimension per subsystem
        return a.reshape(array(dim) ** 2)




    def tensor(*arg):
        """Tensor product of states.

        Returns the tensor product state of states s1, s2, ...

        Ville Bergholm 2009-2010
        """
        # if all states are kets, keep the result state a ket
        pure = True
        for k in arg:
            if not k.is_ket():
                pure = False
                break

        if not pure:
            # otherwise convert all states to state ops before tensoring
            temp = []
            for k in arg:
                temp.append(k.to_op())
            arg = temp

        return lmap.tensor(arg)


    @staticmethod
    def test():
        """Test script for the state class.

        Ville Bergholm 2008-2011
        """
        #for k in range(5):

        # mixed states
        dim = [2, 2]
        rho1 = state(rand_positive(prod(dim)), dim)
        rho2 = state(rand_positive(prod(dim)), dim)
        U_r = rand_U(prod(dim))

        dim = [2, 3, 5, 2, 2]
        sigma1 = state(rand_positive(prod(dim)), dim)
        U_s = rand_U(prod(dim))

        # pure states
        dim = [2, 2]
        p = state(0, dim)

        p1 = p.u_propagate(rand_SU(prod(dim)))
        p2 = p.u_propagate(rand_SU(prod(dim)))
        U_p = rand_U(prod(dim))


        # TODO concurrence, fix_phase, kraus_propagate, locc_convertible, lognegativity, measure,
        # negativity,

        # generalized Bloch vectors.
        skip = """
        temp = sigma1.bloch_vector()
        assert_o(norm(bloch_state(temp) -sigma1), 0, tol) # need to match
        assert_o(norm(imag(temp)), 0, tol) # correlation tensor is real
        """
        # fidelity, trace_dist

        # symmetric
        assert_o(fidelity(rho1, rho2), fidelity(rho2, rho1), tol)
        assert_o(trace_dist(rho1, rho2), trace_dist(rho2, rho1), tol)

        assert_o(fidelity(sigma1, sigma1), 1, tol) # normalized to unity
        assert_o(trace_dist(sigma1, sigma1), 0, tol) # distance measure

        # unaffected by unitary transformations
        assert_o(fidelity(rho1, rho2), fidelity(rho1.u_propagate(U_r), rho2.u_propagate(U_r)), tol)
        assert_o(trace_dist(rho1, rho2), trace_dist(rho1.u_propagate(U_r), rho2.u_propagate(U_r)), tol)

        # for pure states they're equivalent
        assert_o(trace_dist(p1, p2) ** 2 +fidelity(p1, p2) ** 2, 1, tol)
        # for mixed states, these inequalities hold
        assert(sqrt(1 - fidelity(rho1, rho2) ** 2) - trace_dist(rho1, rho2) >= -tol)
        assert(1 - fidelity(rho1, rho2) - trace_dist(rho1, rho2) <= tol)
        # for a pure and a mixed state we get this inequality
        assert(1 - fidelity(rho1, p1) ** 2 -trace_dist(rho1, p1) <= tol)


        # entropy
        assert_o(p1.entropy(), 0, tol) # zero for pure states
        assert(sigma1.entropy() >= -tol) # nonnegative

        # unaffected by unitary transformations
        assert_o(sigma1.u_propagate(U_s).entropy(), sigma1.entropy(), tol)


        # ptrace, ptranspose

        rho_A = rho1.ptrace([1])
        # trace of partial trace equals total trace
        assert_o(rho1.trace(), rho_A.trace(), tol)
        # partial trace over all subsystems equals total trace
        assert_o(rho1.trace(), rho1.ptrace(range(rho1.subsystems())).trace(), tol)

        rho_X = sigma1.ptrace([0, 2, 3])
        assert_o(sigma1.trace(), rho_X.trace(), tol)
        assert_o(sigma1.trace(), sigma1.ptrace(range(sigma1.subsystems())).trace(), tol)

        rho_pt_B = rho1.ptranspose([1])
        # two ptransposes cancel
        assert_o(trace_dist(rho1, rho_pt_B.ptranspose([1])), 0, tol)
        # ptranspose preserves trace
        assert_o(rho1.trace(), rho_pt_B.trace(), tol)


        # schmidt
        
        lambda1, u, v = p1.schmidt([0], full = True)
        lambda2 = p1.schmidt([1])
        # squares of schmidt coefficients # sum up to unity
        assert_o(norm(lambda1), 1, tol)
        # both subdivisions have identical schmidt coefficients
        assert_o(norm(lambda1-lambda2), 0, tol)

        # decomposition is equal to the original matrix
        temp = 0
        for k in range(len(lambda1)):
            temp += kron(lambda1[k]*u[:,k], v[:,k])
        assert_o(norm(p1.data.flatten() - temp), 0, tol)

        # squared schmidt coefficients equal eigenvalues of partial trace
        r = state(rand(30)-0.5 +1j*(rand(30)-0.5), [5, 6]).normalize()
        #x = r.schmidt([0]) ** 2  # crashes ipython!
        return
        temp = r.ptrace([1])
        y, dummy = eigsort(temp.data)
        assert_o(norm(x-y), 0, tol)


        # reorder

        dim = [2, 5, 1]
        A = rand(dim[0], dim[0])
        B = rand(dim[1], dim[1])
        C = rand(dim[2], dim[2])

        T1 = state(mkron(A, B, C), dim)
        T2 = reorder(T1, [2, 0, 1])
        assert_o(norm(mkron(C, A, B) - T2.data), 0, tol)
        T2 = reorder(T1, [1, 0, 2])
        assert_o(norm(mkron(B, A, C) - T2.data), 0, tol)
        T2 = reorder(T1, [2, 1, 0])
        assert_o(norm(mkron(C, B, A) - T2.data), 0, tol)



# wrappers

def fidelity(s,t):
    """Wrapper for state.fidelity."""
    return s.fidelity(t)


def trace_dist(s,t):
    """Wrapper for state.trace_dist."""
    return s.trace_dist(t)
