Formats
=======

.. currentmodule:: tensorkrowch.formats

Formats are lightweight numerical models built directly from PyTorch tensors.
They represent data, functions and operators without TensorKrowch's ``Nodes``,
``Edges`` or graph operations, and can be used by decompositions and numerical
solvers. The graph-based :mod:`~tensorkrowch.models` remain useful for learning
from data with repeated contractions.

:class:`TT` and :class:`TR` represent vectors as open and cyclic chains;
:class:`TTM` and :class:`TRM` are their matrix versions. :class:`QTT`,
:class:`QTR`, :class:`QTTM` and :class:`QTRM` add coordinate quantization.


Cores and bond factors
----------------------

Construct a format from ``cores`` and, optionally, a sequence of diagonal
factors passed as ``bonds``. The format creates its own bond container;
users do not need to construct :class:`BondFactors1D` separately.

.. raw:: html

   <style>
   .formats-flow {
       display: flex;
       align-items: stretch;
       gap: 0.6rem;
       margin: 1.2rem 0;
   }
   .formats-flow .formats-node {
       flex: 1;
       min-width: 0;
       padding: 0.9rem;
       border: 1px solid currentColor;
       border-radius: 0.4rem;
   }
   .formats-flow .formats-node p:last-child { margin-bottom: 0; }
   .formats-flow .formats-arrow {
       align-self: center;
       font-size: 1.6rem;
   }
   @media (max-width: 700px) {
       .formats-flow { flex-direction: column; }
       .formats-flow .formats-arrow { transform: rotate(90deg); }
   }
   </style>

.. container:: formats-flow

   .. container:: formats-node

      **Constructor inputs**

      ``cores``: sequence of tensors.

      ``bonds``: optional sequence of diagonal factors.

   .. container:: formats-arrow

      →

   .. container:: formats-node

      **Tensor format**

      :class:`TT` / :class:`TR` / :class:`TTM` / :class:`TRM`

      Shared interface: :class:`TensorFormat1D`.

      :attr:`TensorFormat1D.cores` stores the core tensors;
      :attr:`TensorFormat1D.bonds` holds the optional bond container.

   .. container:: formats-arrow

      →

   .. container:: formats-node

      **Owned bond container**

      :class:`BondFactors1D`

      :attr:`BondFactors1D.factors`: diagonals between adjacent cores.

      :class:`VidalGauge` also records
      :attr:`VidalGauge.spectra` and :attr:`VidalGauge.powers`.

:meth:`TT.canonicalize_vidal` and :meth:`TTM.canonicalize_vidal` create a
:class:`VidalGauge` in ``format.bonds``. To absorb the stored factors into
cores, use :meth:`TensorFormat1D.materialize_bonds`.


Quantics construction and evaluation
------------------------------------

Quantics formats combine cores with a :class:`QuantizedLayout` and a
:class:`CoordinateMap`. The map defines the domain grid; the layout expands
its integer indices into digits and orders those digits along the chain.

There are three construction options:

* ``cores``, ``n_coordinates``, ``base``, ``level`` and ``domain`` create an
  :class:`AffineCoordinateMap` with ``grid_offset="left"`` and a
  :class:`QuantizedLayout` with ``ordering="interleaved"`` and
  ``digit_order="coarse_to_fine"``.
* Replace ``domain`` with ``grid_coordinates`` to create an
  :class:`ExplicitGridMap`. The number of grid points per coordinate must
  equal ``base ** level``.
* Pass ``cores``, ``n_coordinates``, ``layout`` and ``coordinate_map`` explicitly
  for other choices, including a :class:`FunctionalCoordinateMap`.

For :class:`QTTM` and :class:`QTRM`, specify the input and output layouts and
maps separately through the corresponding ``in_*`` and ``out_*`` arguments.
Both sides must have the same number of digit sites. In every case, each map's
``grid_size`` must match its layout's ``grid_size``.

.. container:: formats-flow

   .. container:: formats-node

      **Domain coordinates**

      :class:`CoordinateMap`

      :class:`AffineCoordinateMap`, :class:`FunctionalCoordinateMap`
      or :class:`ExplicitGridMap`.

      → :meth:`CoordinateMap.to_indices`

      ← :meth:`CoordinateMap.from_indices`

   .. container:: formats-arrow

      →

   .. container:: formats-node

      **Grid indices**

      :class:`QuantizedLayout`

      ``base``, ``level`` and :meth:`QuantizedLayout.sites` define the
      digit-site order.

      → :meth:`QuantizedLayout.encode_indices`

      ← :meth:`QuantizedLayout.decode_digits`

   .. container:: formats-arrow

      →

   .. container:: formats-node

      **Digits and cores**

      :class:`QTT` / :class:`QTR` / :class:`QTTM` / :class:`QTRM`

      :meth:`QTT.evaluate_digits` or :meth:`QTTM.evaluate_digits`
      contract the cores at the selected digits to obtain values.

:meth:`QTT.evaluate_coordinates` and :meth:`QTTM.evaluate_coordinates` perform
this complete path. :meth:`QTT.evaluate_indices` and
:meth:`QTTM.evaluate_indices` start from grid indices. The ring versions provide
the same evaluation methods.


Format interfaces
-----------------

TensorFormat
^^^^^^^^^^^^
.. autoclass:: TensorFormat
    :members:

TensorFormat1D
^^^^^^^^^^^^^^
.. autoclass:: TensorFormat1D
    :members:


Vector and matrix formats
-------------------------

TT
^^
.. autoclass:: TT
    :members:
    :inherited-members: TensorFormat1D

TR
^^
.. autoclass:: TR
    :members:
    :inherited-members: TensorFormat1D

TTM
^^^
.. autoclass:: TTM
    :members:
    :inherited-members: TensorFormat1D

TRM
^^^
.. autoclass:: TRM
    :members:
    :inherited-members: TensorFormat1D


Layouts and coordinate maps
---------------------------

QuantizedLayout
^^^^^^^^^^^^^^^
.. autoclass:: QuantizedLayout
    :members:

CoordinateMap
^^^^^^^^^^^^^
.. autoclass:: CoordinateMap
    :members:

AffineCoordinateMap
^^^^^^^^^^^^^^^^^^^
.. autoclass:: AffineCoordinateMap
    :members:

FunctionalCoordinateMap
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: FunctionalCoordinateMap
    :members:

ExplicitGridMap
^^^^^^^^^^^^^^^
.. autoclass:: ExplicitGridMap
    :members:


Quantics formats
----------------

QTT
^^^
.. autoclass:: QTT
    :members:
    :inherited-members: TensorFormat1D

QTR
^^^
.. autoclass:: QTR
    :members:
    :inherited-members: TensorFormat1D

QTTM
^^^^
.. autoclass:: QTTM
    :members:
    :inherited-members: TensorFormat1D

QTRM
^^^^
.. autoclass:: QTRM
    :members:
    :inherited-members: TensorFormat1D


Bond factors
------------

BondFactors1D
^^^^^^^^^^^^^
.. autoclass:: BondFactors1D
    :members:

VidalGauge
^^^^^^^^^^
.. autoclass:: VidalGauge
    :members:


Gauge orbits
------------

GaugeOrbit
^^^^^^^^^^
.. autoclass:: GaugeOrbit
    :members:

TensorRingOrbit
^^^^^^^^^^^^^^^
.. autoclass:: TensorRingOrbit
    :members:


Blocks
------

BlockLayout
^^^^^^^^^^^
.. autoclass:: BlockLayout
    :members:

SplitBlock
^^^^^^^^^^^
.. autoclass:: SplitBlock
    :members:

split_block
^^^^^^^^^^^
.. autofunction:: split_block


Diagnostics
-----------

SampleError
^^^^^^^^^^^
.. autoclass:: SampleError
    :members:

RoundingInfo
^^^^^^^^^^^^
.. autoclass:: RoundingInfo
    :members:

MinimalCanonicalInfo
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: MinimalCanonicalInfo
    :members:
