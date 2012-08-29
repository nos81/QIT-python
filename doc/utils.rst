Utilities (:mod:`qit.utils`)
============================

.. currentmodule:: qit.utils

This module contains utility functions which do not logically fit anywhere else.


Mathematical utilities
----------------------

.. autosummary::

   comm
   acomm
   gcd
   lcm
   rank
   projector
   eigsort
   expv
   mkron
   majorize


Random matrices
---------------

.. autosummary::

   rand_hermitian
   rand_U
   rand_SU
   rand_U1
   rand_pu
   rand_positive
   rand_SL


Superoperators
--------------

.. autosummary::

   vec
   inv_vec
   lmul
   rmul
   lrmul
   superop_lindblad   


Physics
-------

.. autosummary::

   angular_momentum
   boson_ladder
   fermion_ladder


Bases, decompositions
---------------------

.. autosummary::

   spectral_decomposition
   gellmann
   tensorbasis


Miscellaneous
-------------

.. autosummary::

   assert_o
   copy_memoize
   op_list
   qubits
   R_nmr
   R_x
   R_y
   R_z
   test



.. automodule:: qit.utils
   :members:
