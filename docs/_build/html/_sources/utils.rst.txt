Utilities
=========

.. currentmodule:: tensorkrowch.utils

Helpers for names, sequences, tensor construction and numerical operations.


Strings and names
-----------------

.. autofunction:: print_list

.. autofunction:: tab_string

.. autofunction:: check_name_style

.. autofunction:: erase_enum

.. autofunction:: enum_repeated_names


Sequences and permutations
--------------------------

.. autofunction:: permute_list

.. autofunction:: is_permutation

.. autofunction:: inverse_permutation

.. autofunction:: list2slice

.. autofunction:: split_sequence_into_regions


Combinatorics
-------------

.. autofunction:: fact

.. autofunction:: binomial_coeffs


Tensor construction
-------------------

.. autofunction:: stack_unequal_tensors

.. autofunction:: random_unitary


Tensor scaling and normalization
--------------------------------

These operations are shared by formats and decomposition algorithms. They
retain PyTorch broadcasting, devices and autograd. A zero tensor remains zero
under normalization with extracted log-scale ``0``; its mathematical log-norm
is ``-inf``.

.. autofunction:: safe_divide

.. autofunction:: scale_tensor

.. autofunction:: stable_norm

.. autofunction:: log_tensor_norm

.. autofunction:: normalize_tensor

.. autofunction:: ratio_from_log

.. autofunction:: norm_from_log


Singular-value decompositions
-----------------------------

.. autofunction:: accurate_svd

.. autofunction:: truncated_svd
