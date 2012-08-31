Control sequences (:mod:`qit.seq`)
==================================


Piecewise constant control sequences for quantum systems.
Each control sequence is a dictionary with the following keys:

  A
    Drift generator (typically :math:`i/\hbar` times a Hamiltonian and
    a time unit of your choice).

  B
    List of control generators. c := len(B).

  tau
    Vector of durations of the time slices. m := len(tau).

  control
    Array, shape == (m, c). control[i,j] is the value of control
    field j during time slice i.

The total generator for the time slice j is thus given by
.. math::

   G_j = A +\sum_k \text{control}_{jk} B_k,

and the corresponding propagator is
.. math::

   P_j = \exp(-\tau_j G_j).


NOTE the unusual sign convention.


Contents
--------

.. currentmodule:: qit.seq

.. autosummary::

   nmr
   corpse
   bb1
   scrofulous
   cpmg
   seq2prop
   propagate



.. automodule:: qit.seq
   :members:
