Format API reference
====================

.. currentmodule:: tensorkrowch.formats

See :doc:`formats` for construction examples, evaluation, numerical
operations and optional bond or coordinate objects. This page lists the
interfaces, concrete formats and supporting objects.

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
^^^^^^^^^^
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
