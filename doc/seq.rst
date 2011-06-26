Control sequences (:mod:`qit.seq`)
==================================


One-qubit control sequences. Each control sequence is a list of
vectors of the form [x, y, z, t], each of which corresponds to an
evolution for time t under the Hamiltonian

.. math::

   H = \frac{1}{2} \left(x \sigma_x +y \sigma_y +z \sigma_z \right).


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
