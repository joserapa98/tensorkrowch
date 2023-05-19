.. currentmodule:: tensorkrowch
.. _tutorial_1:

=========================================
Creating a Tensor Network in TensorKrowch
=========================================

Tensor networks are structures used in quantum many-body physics to make
simulations or understand properties of some states. At its core, a tensor
network is just a graph where each node is a tensor (a multi-dimensional array),
and each edge represents one of the axes of the tensor. Therefore, two nodes
can be connected if the dimensions of the corresponding edges coincide. This
connection represents a contraction (similar to a matrix multiplication)
between the nodes at the selected axes. Check this `site <https://tensornetwork.org/>`_
to learn more about tensor networks and the tensor diagram notation.


Introduction
============

``TensorKrowch`` enables you to build any tensor network and train it via
gradient descent methods that leverage the automatic differentiation engine
provided by ``PyTorch``.

In this tutorial you will learn how to combine the basic components of
``TensorKrowch`` to create a ``TensorNetwork``.


Setup
=====

Before we begin, we need to install ``tensorkrowch`` if it isn't already available.

::

    $ pip install tensorkrowch


Since ``TensorKrowch`` has a `PyTorch C++ Extension <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_,
it has to be built from source, so make sure you have installed on your system
a C++ compiler compatible with C++14.


Steps
=====

1. Components of a tensor network.
2. How to create Nodes.
3. How to build the TensorNetwork.
4. Different types of Nodes.


1. Components of a Tensor Network
---------------------------------

In ``TensorKrowch`` there are 3 main objects you should be familiarized with:

1. :class:`Nodes <AbstractNode>`:

   These are the elements that make up a :class:`TensorNetwork`. At its most
   basic level, a node is a container for a ``torch.Tensor`` that stores other
   relevant information which enables to build any network and operate nodes
   to contract it (and train it!). Some of the information that is carried by
   the nodes includes:

   a) **Shape**: Every node needs a shape to know if connections with other
      nodes are possible. Even if the tensor is not specified, an empty node
      needs a shape.
     
   b) **Tensor**: The key ingredient of the node. Although the node acts as a
      `container` for the tensor, the node does not `contain` it. Actually,
      for efficiency purposes, the tensors are stored in a sort of memory that
      is shared by all the nodes of the ``TensorNetwork``. Therefore, all
      that nodes `contain` is a memory address.
     
   c) **Edges**: A list of :class:`Edges <Edge>`, one for each dimension of
      the node.
     
   d) **Network**: The :class:`TensorNetwork` to which the node belongs. If
      the network is not specified when creating the node, a new ``TensorNetwork``
      is created to contain the node.

2. :class:`Edges <Edge>`:

   An edge is nothing more than an object that wraps references to the nodes
   it connects. Thus it stores information like the nodes it connects, the
   corresponding nodes' :class:`Axes <Axis>` it is attached to, whether it is
   dangling or batch, its size, etc.
   
   Above all, its importance lies in that edges enable to connect nodes,
   forming any possible graph, and to perform easily :class:`Operations <Operation>`
   like contracting and splitting nodes.

3. :class:`TensorNetwork`:

   ``TensorNetwork`` is the central object of ``TensorKrowch``. It is a subclass
   of ``torch.nn.Module``. Thus, ``TensorNetworks`` are the *trainable objects*
   of ``TensorKrowch``. 
   
   When building a tensor network, all your ``Nodes`` should belong to the
   same ``TensorNetwork`` model. You can also indicate which nodes are
   `trainable` and which ones are fixed.


2. How to create Nodes
----------------------

First of all, let's import the necessary libraries::

    import torch
    import tensorkrowch as tk

To create a node you must provide a shape. As explained before, all nodes must
have a shape, even if the node does not already have a tensor. Thus, creating
an empty ``Node`` is as simple as::
    
    node = tk.Node(shape=(2, 5, 2))

Creating empty nodes can be useful to experiment creating the graph structure. 
However, if we want to train a ``TensorNetwork``, it would be better if the nodes
are not empty. To fill them with a ``torch.Tensor``, we can use different mehtods.

1. We can initialize the node providing a tensor::
    
    tensor = torch.randn(2, 5, 2)
    node = tk.Node(tensor=tensor)

   When doing this, the node's shape is directly inferred from the tensor's shape.

2. We can just provide a shape and specify a method to initialize the tensor::

    node = tk.Node(shape=(2, 5, 2),
                   init_method='randn')

   Actually, to make it even easier (and more similar to ``PyTorch``), the above
   is equivalent to::

    node = tk.randn(shape=(2, 5, 2))

Now that our node has a tensor in it, we can extract it to use it in our code via::

    node.shape   # Returns node's shape
    node.tensor  # Returns node's tensor

For now, all we have done we could have done it directly with vanilla ``PyTorch``.
One feature that comes with ``TensorKrowch`` is naming the axes of a ``Node``, which
facilitates the way you can manipulate, connect and contract nodes simply by using
names rather than having to control the index and order of your axes at all times.

You can name the axes when instantiating the node::

    node = tk.randn(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'))

Actually, each ``Axis`` of the ``Node`` is an object with an index and a name. You
can access any of them and change its name or retrieve its index.

::

    node.get_axis('left').name = 'other_left'  # Changes the axis' name
    idx = node.get_axis_num('other_left')      # Returns index of 'other_left'
                                               # in the axes list

Also, naming axes can define a different behaviour for that ``Axis``.

::

    node.get_axis('other_left').name = 'batch'

If the word ``"batch"`` appears in some axis' name, the corresponding edge will
be used as a *batch* edge. We will explore that in the next section.


3. How to build the TensorNetwork
---------------------------------

To be able to build a network, you must be able to access the ``Edges`` of a
``Node`` in order to connect them to other ``Edges``. Thanks to the naming of
axes, you can do that very easily::

    node['right']

By using the name of each ``Axis``, you can retrieve the corresponding ``Edge``.
What's more, you can get some information from that ``Edge``::

    node['right'].is_dangling()  # Indicates if the edge is not connected
    node['batch'].is_batch()     # Indicates if the edge is a batch edge

    node['input'].size()         # Returns the shape of the node in that axis

Before we start creating the network, let's explore a couple of new options we
have when instatiating a ``Node``. As explained earlier, all nodes must belong
to a ``TensorNetwork``. Hence, when a ``Node`` is created a new ``TensorNetwork``
is also created to contain it. However, one can also create a ``TensorNetwork``
beforehand and then instatiate a ``Node`` in that network.

Besides, we can give our ``Node`` a name, which we will use to access that ``Node``
from the ``TensorNetwork``.

::

    net = tk.TensorNetwork()

    node1 = tk.randn(shape=(2, 5, 2),
                     axes_names=('left', 'input', 'right'),
                     name='node1',
                     network=net)

    assert net['node1'] is node1

We can create another ``Node`` in the same network::

    node2 = tk.randn(shape=(2, 5, 2),
                     axes_names=('left', 'input', 'right'),
                     name='node2',
                     network=net)

Now both nodes belong to the same ``TensorNetwork``, yet they are not related.
They are just two pieces of a ``TensorNetwork`` object. In order to create an
actual graph, we can connect nodes with the operand ``^``::

    node1['right'] ^ node2['left']

Note that if ``node1`` and ``node2`` would have been instantiated in different
``TensorNetworks``, when connecting them, ``node2`` would have been moved to the
``node1``'s network. Therefore, all nodes in a connected component of the graph
always belong to the same network.

You can also disconnect edges, although this is something you should only do
when experimenting creating networks, but not when contracting the network::

    node1['right'].disconnect()

This is equivalent to using the operand ``|``::

    node1['right'] | node2['left']

With this, you already have the basic tools to create any ``TensorNetwork``.
Let's finish showing how you can build a Matrix Product State (MPS)::

    mps = tk.TensorNetwork(name='mps')

    for i in range(100):
        _ = tk.randn(shape=(2, 5, 2),
                     axes_names=('left', 'input', 'right'),
                     name=f'node_({i})',
                     network=mps)
        
    for i in range(100):
        mps[f'node_({i})']['right'] ^ mps[f'node_({(i + 1) % 100})']['left']
