Decompositions
==============

.. currentmodule:: tensorkrowch.decompositions

SVD decompositions
------------------

TT-SVD
^^^^^^

.. autoclass:: TTSVD
   :members: fit

.. autofunction:: tt_svd

TTM-SVD
^^^^^^^

.. autoclass:: TTMSVD
   :members: fit

.. autofunction:: ttm_svd

TR-SVD
^^^^^^

.. autoclass:: TRSVD
   :members: fit

.. autofunction:: tr_svd

Lightweight results
-------------------

.. autoclass:: TTDecomposition
   :members:

.. autoclass:: TTMDecomposition
   :members:

.. autoclass:: TRDecomposition
   :members:

Compatibility aliases
---------------------

The historical names below remain available for compatibility and emit a
deprecation warning. New code should use :func:`tt_svd` and :func:`ttm_svd`.

.. autofunction:: vec_to_mps

.. autofunction:: mat_to_mpo

Sketching decompositions
------------------------

.. autofunction:: tt_rss
