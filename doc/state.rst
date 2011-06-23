Quantum states (:mod:`qit.state`)
=================================


In QIT, quantum states are represented by the :class:`state` class,
defined in this module.


.. currentmodule:: qit.state.state


Utilities
---------

.. autosummary::

   subsystems
   dims
   clean_selection
   invert_selection
   fix_phase
   normalize
   purity
   to_ket
   to_op
   trace
   ptrace
   ptranspose
   reorder
   tensor
   plot



Physics
-------

.. autosummary::

   ev
   var
   prob
   projector
   u_propagate
   propagate
   kraus_propagate
   measure
   

Quantum information
-------------------

.. autosummary::

   fidelity
   trace_dist
   schmidt
   entropy
   concurrence
   negativity
   lognegativity
   scott
   locc_convertible


Other state representations
---------------------------

.. autosummary::

   bloch_vector
   bloch_state



.. automodule:: qit.state
   :members:

