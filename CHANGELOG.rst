Changelog
=========

for Quantum Information Toolkit (Python version)


0.12.0 (2021-07-30)
-------------------

Features
~~~~~~~~

* Python 2 compatibility dropped.
* Bump dependencies: NumPy 1.20, SciPy 1.6, matplotlib 3.4, Sphinx 4.1
* New project setup with pyproject.toml, setup.cfg.
* Unit tests now use pytest, more unit tests.
* Docstrings now use sphinx.ext.napoleon, math in the docs is rendered using MathJax,
  API docs are generated using sphinx.ext.autosummary.
* Classes renamed to begin with a capital letter: Lmap, State, Seq...
* SU(2) rotations renamed to Rx, Ry, Rz
* seq: sign convention changed, converted into the Seq class.
* utils.tensorbasis also returns the integer locality of each element.
* examples.bb84 prints more info.
* examples.nmr_sequences updated, now also does pulse strength errors.
* hamiltonian.jaynes_cummings updated.
* hamiltonian.heisenberg updated.
* invariant.plot_canonical_2q and invariant.plot_makhlin_2q also plot the set of perfect entanglers.
* markov.MarkovianBath class rewritten, supports also fermionic baths.
* Added:

  + Lmap.tensorpow
  + Lmap.real
  + Lmap.imag
  + State.werner
  + State.isotropic
  + gate.copydot
  + gate.plusdot
  + gate.epsilon
  + utils.cdot
  + utils.tensorsum
  + utils.rand_GL
  + utils.superop_to_choi
  + utils.trisurf
  + seq.knill
  + seq.dd, seq.cpmg merged into seq.dd
  + invariant.gate_leakage_inv
  + invariant.gate_adjoint_rep
  + invariant.canonical_inv_normalize
  + markov.MarkovianBath.plot_bath_correlation
  + markov.MarkovianBath.plot_spectral_correlation
  + markov.MarkovianBath.plot_spectral_correlation_vs_cutoff
  + examples.werner_states
  + examples.quantum_walk
  + ho.rotate
  + ho.beamsplitter
  + ho.cx

Bugfixes
~~~~~~~~

* Lmap.__init__ and Lmap.__repr__ now handle also sparse matrices.
* invariant.LU fixed, renamed invariant.state_inv
* markov.ops warns of possible RWA violations.
* examples.qft_circuit now also works on all nonpalindromic dim vectors.
* ho.squeeze: square the matrix, not the elements



0.11.0 (2014-04-09)
-------------------

Features
~~~~~~~~

* Code now supports both Python 2.7 and Python 3.
* Tests moved into the test subdirectory, rewritten using unittest.
  The tests can be run either individually or all together using
  python -m unittest discover -s test
* Use specific reST syntax for documenting parameters where needed.
* Use sphinxcontrib-bibtex for the references.

Bugfixes
~~~~~~~~

* seq.scrofulous now uses the phi parameter.
* Lots of pylinting.


0.10.0 (2014-03-29)
-------------------

Features
~~~~~~~~

* Bump dependencies: NumPy 1.7.1, SciPy 0.11.0, matplotlib 1.2
* Code cleanup, references updated.
* Use setuptools for packaging.
* examples.nmr_sequences can also plot a user-given sequence.
* plot.correlation_simplex_2q renamed to plot.correlation_simplex
* Added:

  + utils.comm
  + utils.acomm
  + hamiltonian.heisenberg
  + hamiltonian.jaynes_cummings
  + hamiltonian.hubbard
  + hamiltonian.bose_hubbard
  + hamiltonian.holstein

Bugfixes
~~~~~~~~

* utils.fermion_ladder simplified and fixed.
* utils.majorize fixed.
* state.check fixed.
* plot.adiabatic_evolution: indexing bug fixed.


0.9.13 (2014-03-24)
-------------------

Features
~~~~~~~~

* Sphinx module docs moved into the module docstrings.
* Examples beautified.
* state.entropy also gives Rényi entropies.
* utils.gellmann returns a 3-dimensional array instead of a list (in correct order!).
* utils.rank uses a smarter default tolerance.
* utils.eigsort renamed to utils.eighsort
* Added:

  + state.check
  + lmap.__div__ for convenience until we migrate to Python 3.
  + utils.orth
  + utils.nullspace
  + utils.nullspace_hermitian
  + utils.superop_fp

Bugfixes
~~~~~~~~

* Scipy 0.10 fix: scipy.misc.factorial imported from the correct module.
* state.plot 3D plot of mixed states fixed. Color is
  back, better viewpoint, bars follow matrix ordering.
* state.propagate uses deepcopy on the results by default,
  multiple input time instances produce the correct output.
* state.negativity: extra sqrt removed.
* state.projector: should only work on kets.
* state.scott: works again, lexicographic ordering of output terms.


0.9.12 (2012-08-31)
-------------------

Features
~~~~~~~~

* Initialization message removed.
* Better documentation, docstring backslash problems fixed.
* Preparing for Python 3 conversion: added the absolute_import and
  unicode_literals __future__ features. Now we use every non-redundant
  __future__ feature available in Python 2.6.
* examples.bb84 also prints the interesting stuff.
* examples.quantum_channels and examples.nmr_sequences use GridSpec to define the subplots.
* utils.majorize uses tolerances.
* Entire seq module revamped.
* Renamed some functions in the plot module by removing the redundant ``plot_`` prefix.
* plot.bloch_sphere no longer accepts a state as input.
* Added:

  + Bernstein-Vazirani algorithm example.
  + lmap.norm
  + lmap.trace
  + utils.rand_pu, this is how utils.rand_positive gets its partition of unity.
  + utils.rand_SL
  + invariant.LU
  + plot.state_trajectory, plot.correlation_simplex_2q

Bugfixes
~~~~~~~~

* setup.py fixed.
* eig/eigh, eigvals/eigvalsh in state.propagate and elsewhere.
  Now we get orthonormal eigenvectors with degenerate eigenvalues as well.
* utils.expv no longer crashes with a cryptic error message when
  given np.matrix instances as input, now raising an exception instead.


0.9.11 (2012-08-22)
-------------------

Features
~~~~~~~~

* Bump dependencies: matplotlib 1.0.1+, NumPy 1.5.1+, SciPy 0.9.0+.
  This fixes several problems with 3D plotting.
  Sparse matrix handling is improved.
* setup.py, proper distutils packaging.

Bugfixes
~~~~~~~~

* 3D subplots fixed.


0.9.10 (2011-06-26)
-------------------

First public beta, based on the MATLAB QIT version 0.9.9, with several bugfixes.

Features
~~~~~~~~

* Sphinx documentation.
