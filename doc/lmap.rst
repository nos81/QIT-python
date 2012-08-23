Linear maps (:mod:`qit.lmap`)
=============================

Bounded finite-dimensional linear maps are represented using :class:`lmap` class instances.
In addition to the matrix representing the map, they contain
the dimension vectors of the domain and codomain vector spaces.
All the usual scalar-map and map-map arithmetic operators are
provided, including the exponentiation of maps by integers.


.. currentmodule:: qit.lmap.lmap


Utilities
---------

.. autosummary::

   remove_singletons
   is_compatible
   is_ket


Linear algebra
--------------

.. autosummary::

   conj
   transpose
   ctranspose
   trace
   norm
   reorder


Non-member functions:

.. currentmodule:: qit.lmap

.. autosummary::
   tensor



.. automodule:: qit.lmap
   :members:
