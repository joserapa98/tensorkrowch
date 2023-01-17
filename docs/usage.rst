Usage
=====

.. _installation:

Installation
------------

To use TensorKrowch, first install it using pip:

.. code-block:: console

    $ pip install tensorkrowch

Here is a ``snippet`` of code

>>> import tensorkrowch as tk
>>> node = tk.randn((2, 3))
>>> node.name
'node'


You can use ``Node`` to create nodes of the network:

.. autoclass:: tensorkrowch.Node


.. autofunction:: tensorkrowch.node_operations._contract_edges_first