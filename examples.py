# -*- coding: utf-8 -*-
# Author: Ville Bergholm 2011
"""Examples and demos."""

from __future__ import print_function, division
from math import asin
from operator import mod

from numpy import floor, ceil, log2, angle, arange
from numpy.linalg import eig, matrix_power
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, hold, plot, bar, title, xlabel, ylabel, axis, legend

from lmap import *
from state import *
from utils import *
import gate
import ho



def bb84(n=50):
    """Bennett-Brassard 1984 quantum key distribution protocol demo.

    Simulate the protocol with n qubits transferred.

    Ville Bergholm 2009
    """
    print('\n\n=== BB84 protocol ===\n')
    print('Using {0} transmitted qubits.\n'.format(n))

    # Alice generates two random bit vectors
    sent    = rand(n) > 0.5
    basis_A = rand(n) > 0.5

    # Bob generates one random bit vector
    basis_B = rand(n) > 0.5
    received = zeros(n, bool)

    # Eve hasn't yet decided her basis
    basis_E = zeros(n, bool)
    eavesdrop = zeros(n, bool)

    print('Alice transmits a sequence of qubits to Bob using a quantum channel.')
    print('For every qubit, she randomly chooses a basis (computational or diagonal)')
    print('and randomly prepares the qubit in either the |0> or the |1> state in that basis.\n')
    print('When Bob receives the qubits, he randomly measures them in either basis')

    temp = state('0')
    for k in range(n):
        # Alice has a source of qubits in the zero state.
        q = temp

        # Should Alice flip the qubit?
        if sent[k]:  q = q.u_propagate(sx)

        # Should Alice apply a Hadamard?
        if basis_A[k]:  q = q.u_propagate(H)

        # Alice now sends the qubit to Bob...
        # ===============================================
        # ...but Eve intercepts it, and conducts an intercept/resend attack

        # Eve might have different strategies here... TODO
        # Eve's strategy (simplest choice, random)
        basis_E[k] = rand() > 0.5
  
        # Eve's choice of basis: Hadamard?
        if basis_E[k]:  q = q.u_propagate(H)

        # Eve measures in the basis she has chosen
        p, res, q = q.measure(do = 'C')
        eavesdrop[k] = res

        # Eve tries to reverse the changes she made...
        if basis_E[k]:  q = q.u_propagate(H)

        # ...and sends the result to Bob.
        # ===============================================

        # Bob's choice of basis
        if basis_B[k]:  q = q.u_propagate(H)

        # Bob measures in the basis he has chosen, and discards the qubit.
        p, res = q.measure()
        received[k] = res

    #sum(xor(sent, eavesdrop))/n
    #sum(xor(sent, received))/n

    print('Now Bob announces on a public classical channel that he has received all the qubits.')
    print('Alice then reveals the bases she used, and Bob compares them to his.')
    print('Whenever the bases match, so should the prepared/measured values unless there\'s an eavesdropper.')

    match = np.logical_not(np.logical_xor(basis_A, basis_B))
    key_A = sent[match]
    key_B = received[match]
    m = len(key_A)
    print('\nMismatch frequency between Alice and Bob: {0}\n'.format(sum(np.logical_xor(key_A, key_B)) / m))

    print('Alice and Bob then sacrifice k bits of their shared key to compare them.')
    print('If an nonmatching bit is found, the reason is either an eavesdropper or a noisy channel.')
    print('Since the probability for each eavesdropped bit to be wrong is 1/4, they will detect')
    print('Eve\'s presence with the probability 1-(3/4)^k.')


def grover_search(n=8):
    """Grover search algorithm demo.

    Simulate the Grover search algorithm formulated using amplitude amplification
    in a system of n qubits.

    %! L.K. Grover, "Quantum Computers Can Search Rapidly by Using Almost Any Transformation", PRL 80, 4329 (1998). doi:10.1103/PhysRevLett.80.4329.   
    Ville Bergholm 2009-2010
    """
    print('\n\n=== Grover search algorithm ===\n')

    A = gate.walsh(n) # Walsh-Hadamard gate for generating uniform superpositions
    N = 2 ** n # number of states

    sol = randint(N)
    reps = int(pi/(4*asin(sqrt(1/N))))

    print('Using {0} qubits.'.format(n))
    print('Probability maximized by {0} iterations.'.format(reps))
    print('Correct solution: {0}'.format(sol))

    # black box oracle capable of recognizing the correct answer (given as the diagonal)
    # TODO an oracle that actually checks the solutions by computing (using ancillas?)
    U_oracle = ones((N,1))
    U_oracle[sol,0] = -1

    U_zeroflip = ones((N,1))
    U_zeroflip[0,0] = -1

    s = state(0, qubits(n))

    # initial superposition
    s = s.u_propagate(A)

    # Grover iteration
    for k in range(reps):
        # oracle phase flip
        s.data = -U_oracle * s.data

        # inversion about the mean
        s = s.u_propagate(A.ctranspose())
        s.data = U_zeroflip * s.data
        s = s.u_propagate(A)

    p, res = s.measure()
    print('\nMeasured {0}.'.format(res))
    return p


def phase_estimation(t, U, s, implicit=False):
    """Quantum phase estimation algorithm.

    Estimate an eigenvalue of unitary operator U using t qubits,
    starting from the state s.

    Returns the state of the index register after the phase estimation
    circuit, but before final measurement.

    To get a result accurate to n bits with probability >= (1-epsilon),
    choose  t >= n + ceil(log2(2+1/(2*epsilon))).

    %! R. Cleve et al., "Quantum Algorithms Revisited", Proc. R. Soc. London A454, 339 (1998).
    %! M.A. Nielsen, I.L. Chuang, "Quantum Computation and Quantum Information" (2000), chapter 5.2.
    Ville Bergholm 2009-2010
    """
    T = 2 ** t
    S = U.shape[0]

    # index register in uniform superposition
    #reg_t = u_propagate(state(0, qubits(t)), gate.walsh(t)) # use Hadamards
    reg_t = state(ones(T) / sqrt(T), qubits(t)) # skip the Hadamards

    # state register (ignore the dimensions)
    reg_s = state(s, S)

    # full register
    reg = reg_t.tensor(reg_s)

    # controlled unitaries
    for k in range(t):
        ctrl = -ones(t)
        ctrl[k] = 1
        temp = gate.controlled(matrix_power(U, 2 ** (t-1-k)), ctrl)
        reg = reg.u_propagate(temp)
    # from this point forward the state register is not used anymore

    if implicit:
        # save memory and CPU: make an implicit measurement of the state reg, discard the results
        dummy, res, reg = reg.measure(t, do = 'D')
        #print('Implicit measurement of state register: {0}\n', res)
    else:
        # more expensive computationally: trace over the state register
        reg = reg.ptrace((t,))

    # do an inverse quantum Fourier transform on the index reg
    QFT = gate.qft(qubits(t))
    return reg.u_propagate(QFT.ctranspose())



def phase_estimation_precision(t, U, u=None):
    """Quantum phase estimation demo.

    Estimate an eigenvalue of unitary operator U using t bits, starting from the state u.
    Plots and returns the probability distribution of the resulting t-bit approximations.
    If u is not given, use a random eigenvector of U.

    %! R. Cleve et al., "Quantum Algorithms Revisited", Proc. R. Soc. London A454, 339 (1998).
    %! M.A. Nielsen, I.L. Chuang, "Quantum Computation and Quantum Information" (2000), chapter 5.2.
    Ville Bergholm 2009-2010
    """
    print('\n\n=== Phase estimation ===\n\n')

    # find eigenstates of the operator
    N = U.shape[0]
    d, v = eig(U)
    if u == None:
        u = state(v[:, 0], N) # exact eigenstate

    print('Use {0} qubits to estimate the phases of the eigenvalues of a U({1}) operator.\n'.format(t, N))
    p = phase_estimation(t, U, u).prob()
    T = 2 ** t
    x = arange(T) / T
    w = 0.8 / T

    # plot probability distribution
    figure()
    hold(True)
    bar(x, p, width = w) # TODO align = 'center' ???
    xlabel('phase / $2\pi$')
    ylabel('probability')
    title('Phase estimation')
    #axis([-1/(T*2), 1-1/(T*2), 0, 1])

    # compare to correct answer
    target = angle(d) / (2*pi) + 1
    target -= floor(target)
    plot(target, 0.5*max(p)*ones(len(target)), 'mo')

    legend(('Target phases', 'Measurement probability distribution'))
    return p



def qft_circuit(dim=(2, 3, 3, 2)):
    """Quantum Fourier transform circuit demo.

    Simulate the quadratic QFT circuit construction.
    dim is the dimension vector of the subsystems.

    NOTE: If dim is not palindromic the resulting circuit also
    reverses the order of the dimensions

    U |x1,x2,...,xn> = 1/sqrt(d) \sum_{ki} |kn,...,k2,k1> exp(i 2 \pi (k1*0.x1x2x3 +k2*0.x2x3 +k3*0.x3))
    = 1/sqrt(d) \sum_{ki} |kn,...,k2,k1> exp(i 2 \pi 0.x1x2x3*(k1 +d1*k2 +d1*d2*k3))

    Ville Bergholm 2010-2011
    """
    print('\n\n=== Quantum Fourier transform using a quadratic circuit ===\n')

    def Rgate(d):
        """R = \sum_{xy} exp(i*2*pi * x*y/prod(dim)) |xy><xy|"""
        temp = kron(arange(d[0]), arange(d[-1])) / prod(d)
        return gate.phase(2 * pi * temp, (d[0], d[-1]))

    n = len(dim)
    U = gate.id(dim)

    for k in range(n):
        H = gate.qft((dim[k],))
        U = gate.single(H, k, dim) * U
        for j in range(k+1, n):
            temp = Rgate(dim[k : j+1])
            U = gate.two(temp, (k, j), dim) * U

    for k in range(n // 2):
        temp = gate.swap(dim[k], dim[n-1-k])
        U = gate.two(temp, (k, n-1-k), dim) * U

    #U.data = todense(U.data) # it's a QFT anyway
    #temp = U - gate.qft(dim)
    #norm(temp.data)
    return U


def shor_factorization(N=9, cheat=False):
    """Shor's factorization algorithm demo.

    Simulates Shor's factorization algorithm, tries to factorize the integer N.
    If cheat is False, simulates the full algorithm. Otherwise avoids the quantum part.

    NOTE: This is a very computationally intensive quantum algorithm
    to simulate classically, and probably will not run for any
    nontrivial value of N (unless you choose to cheat, in which case
    instead of simulating the quantum part we use a more efficient
    classical algorithm for the order-finding).

    %! P.W. Shor, "Algorithms For Quantum Computation: Discrete Logs and Factoring", Proc. 35th Symp. on the Foundations of Comp. Sci., 124 (1994).
    %! M.A. Nielsen, I.L. Chuang, "Quantum Computation and Quantum Information" (2000), chapter 5.3.
    Ville Bergholm 2010-2011
    """
    def find_order_cheat(a, N):
        """Classical order-finding algorithm."""
        for r in range(1, N+1):
            if mod_pow(a, r, N) == 1:
                return r

    def mod_pow(a, x, N):
        """Computes mod(a^x, N) using repeated squaring mod N.
        x must be a positive integer.
        """
        X = numstr_to_array(bin(x)[2:]) # exponent in big-endian binary
        res = 1
        for b in reversed(X): # little-endian
            if b:
                res = mod(res*a, N)
            a = mod(a*a, N) # square a
        return res


    print('\n\n=== Shor\'s factorization algorithm ===\n')
    if cheat:
        print('(cheating)\n')

    # number of bits needed to represent mod N arithmetic:
    m = int(ceil(log2(N)))
    print('Trying to factor N = {0} ({1} bits).'.format(N, m))

    # maximum allowed failure probability for the quantum order-finding part
    epsilon = 0.25

    # number of index qubits required for the phase estimation
    t = 2*m +1 +int(ceil(log2(2 + 1 / (2 * epsilon))))

    print('The quantum order-finding subroutine will need {0} + {1} qubits.\n'.format(t, m))

    # classical reduction of factoring to order-finding
    while True:
        a = randint(2, N) # random integer, 2 <= a < N
        print('Random integer: a = {0}'.format(a))
  
        p = gcd(a, N)
        if p != 1:
            # a and N have a nontrivial common factor p.
            # This becomes extremely unlikely as N grows.
            print('Lucky guess, we found a common factor!')
            break

        print('Trying to find the period of f(x) = a^x mod N')

        if cheat:
            # classical cheating shortcut
            r = find_order_cheat(a, N)
        else:
            while True:
                print('.')
                # ==== quantum part of the algorithm ====
                [s1, r1] = find_order(a, N, t, m)
                [s2, r2] = find_order(a, N, t, m)
                # =============  ends here  =============

                if gcd(s1, s2) == 1:
                    # no common factors
                    r = lcm(r1, r2)
                    break

        print('\n  =>  r = {0}'.format(r))

        # if r is odd, try again
        if gcd(r, 2) == 2:
            # r is even

            x = mod_pow(a, r // 2, N)
            if mod(x, N) != N-1:
                # factor found
                p = gcd(x-1, N) # ok?
                if p == 1:
                    p = gcd(x+1, N) # no, try this
                break
            else:
                print('a^(r/2) = -1 (mod N), try again...\n')
        else:
            print('r is odd, try again...\n')

    print('\nFactor found: {0}'.format(p))
    return p


def find_order(a, N, t, m):
    """Quantum order-finding subroutine.
    Finds the period of the function f(x) = a^x mod N.
    """
    T = 2 ** t # index register dimension
    M = 2 ** m # state register dimension

    # applying f(x) is equivalent to the sequence of controlled modular multiplications in phase estimation
    U = gate.mod_mul(a, M, N)
    U = U.data # FIXME phase_estimation cannot handle lmaps right now

    # state register initialized in the state |1>
    st = state(1, M)

    # run the phase estimation algorithm
    reg = phase_estimation(t, U, st, implicit = True) # use implicit measurement to save memory

    # measure index register
    dummy, num = reg.measure()  # FIXME?

    def find_denominator(x, y, max_den):
        """
        Finds the denominator q for p/q \approx x/y such that q < max_den
        using a continued fraction representation for x.

        We use floor and mod here, which could be both implemented using
        the classical Euclidean algorithm.
        """
        d_2 = 1
        d_1 = 0
        while True:
            a = x // y # integer part == a_n
            temp = a*d_1 +d_2 # n:th convergent denumerator d_n = a_n*d_{n-1} +d_{n-2}
            if temp >= max_den:
                break

            d_2 = d_1
            d_1 = temp
            temp = mod(x, y)  # x - a*y # subtract integer part
            if temp == 0:
            #if (temp/y < 1 / (2*max_den ** 2))
                break

            # invert the remainder (swap numerator and denominator)
            x = y
            y = temp
        return d_1

    # another classical part
    r = find_denominator(num, T, T+1)
    s = num * r // T
    return s, r


def superdense_coding(d=2):
    """Superdense coding demo.

    Simulate Alice sending two d-its of information to Bob using 
    a shared EPR qudit pair.

    Ville Bergholm 2010-2011
    """
    print('\n\n=== Superdense coding ===\n')

    H   = gate.qft(d)        # qft (generalized Hadamard) gate
    add = gate.mod_add(d, d) # modular adder (generalized CNOT) gate
    I   = gate.id(d)

    dim = (d, d)

    # EPR preparation circuit
    U = add * tensor(H, I)

    print('Alice and Bob start with a shared EPR pair.')
    reg = state('00', dim).u_propagate(U)

    # two random d-its
    a = floor(d * rand(2)).astype(int)
    print('Alice wishes to send two d-its of information (d = {0}) to Bob: a = {1}.'.format(d, a))

    Z = H * gate.mod_inc(a[0], d) * H.ctranspose()
    X = gate.mod_inc(-a[1], d)

    print('Alice encodes the d-its to her half of the EPR pair using local transformations,')
    reg = reg.u_propagate(tensor(Z*X, I))

    print('and sends it to Bob. He then disentangles the pair,')
    reg = reg.u_propagate(U.ctranspose())

    p, b = reg.measure()
    b = array(np.unravel_index(b, dim))
    print('and measures both qudits in the computational basis, obtaining the result  b = {0}.'.format(b))

    if all(a == b):
        print('The d-its were transmitted succesfully.')
    else:
        raise RuntimeError('Should not happen.')


def teleportation(d=2):
    """Quantum teleportation demo.

    Simulate the teleportation of a d-dimensional qudit from Alice to Bob.

    Ville Bergholm 2009-2011
    """
    print('\n\n=== Quantum teleportation ===\n')

    H   = gate.qft(d)        # qft (generalized Hadamard) gate
    add = gate.mod_add(d, d) # modular adder (generalized CNOT) gate
    I   = gate.id(d)

    dim = (d, d)

    # EPR preparation circuit
    U = add * tensor(H, I)

    print('Alice and Bob start with a shared EPR pair.')
    epr = state('00', dim).u_propagate(U)

    print('Alice wants to transmit this payload to Bob:')
    payload = state('0', d).u_propagate(rand_SU(d)).fix_phase()
    # choose a nice global phase

    print('The total |payload> \otimes |epr> register looks like')
    reg = payload.tensor(epr)

    print('Now Alice entangles the payload with her half of the EPR pair')
    reg = reg.u_propagate(tensor(U.ctranspose(), I))

    p, b, reg = reg.measure((0, 1), do = 'C')
    b = array(np.unravel_index(b, dim))
    print('and measures her qudits, getting the result {0}.'.format(b))
    print('She then transmits the two d-its to Bob using a classical channel. The shared state is now')
    reg

    print('Since Alice\'s measurement has unentangled the state,')
    print('Bob can ignore her qudits. His qudit now looks like')
    reg_B = reg.ptrace((0, 1)).to_ket() # pure state

    print('Using the two classical d-its of data Alice sent him,')
    print('Bob performs a local transformation on his half of the EPR pair.')
    Z = H * gate.mod_inc(b[0], d) * H.ctranspose()
    X = gate.mod_inc(-b[1], d)
    reg_B = reg_B.u_propagate(Z*X).fix_phase()

    ov = fidelity(payload, reg_B)
    print('The overlap between the resulting state and the original payload state is |<payload|B>| = {0}'.format(ov))
    if abs(ov-1) > tol:
        raise RuntimeError('Should not happen.')
    else:
        print('The payload state was succesfully teleported from Alice to Bob.')
    return reg_B, payload



def tour():
    """Guided tour to the quantum information toolkit.

    Ville Bergholm 2009-2011
    """
    print('This is the guided tour for the Quantum Information Toolkit.')
    print('Between examples, press any key to proceed to the next one.')

    def pause():
        plt.waitforbuttonpress()

    pause()

    teleportation(2)
    pause()

    superdense_coding(2)
    pause()

    #adiabatic_qc_3sat(5, 25)
    #pause()

    phase_estimation_precision(5, rand_U(4))
    title('Phase estimation, eigenstate')
    phase_estimation_precision(5, rand_U(4), state(0, [4]))
    title('Phase estimation, random state')
    pause()

    #nmr_sequences
    #pause()

    #quantum_channels(0.3)
    #pause()

    grover_search(6)
    pause()

    # shor_factorization(9) # TODO need to use sparse matrices to get even this far(!)
    shor_factorization(311*269, cheat = True)
    pause()

    bb84(40)
    pause()

    #markov_decoherence(7e-10, 1e-9)
    #pause()

    #qubit_and_resonator()
    #pause()

    dim = (2,3,2)
    U = qft_circuit(dim)
    (U - gate.qft(dim)).norm()
    
