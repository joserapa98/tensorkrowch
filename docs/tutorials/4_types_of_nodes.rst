.. currentmodule:: tensorkrowch
.. _tutorial_4:

=======================================
The different Types of Nodes (ADVANCED)
=======================================

``TensorKrowch`` has different methods to distinguish between types of nodes
that are used for different purposes. These types of nodes are not subclasses,
of ``Node``, but rather labels or names that ``Nodes`` and ``ParamNodes`` can
have in order to indicate the role these nodes play in the model.


Introduction
============

In previous tutorials you learned how to create a ``TensorNetwork``. In this
:ref:`tutorial <tutorial_2>` you learned that some nodes can be used just to
hold the input data tensors, and in this other :ref:`tutorial <tutorial_3>` you
learned how to create uniform tensor networks by using a node that stores the
tensor that will be shared by all other nodes in the network.

In this tutorial you will learn how to create nodes with specific roles, like
``data`` and ``virtual`` nodes. Also, you will learn about a couple of reserved
names that are used in specific situations.


Steps
=====

1. Types of Nodes.
2. Reserved Nodes' Names.


1. Types of Nodes
-----------------

In ``TensorKrowch`` there are **4 excluding types** of nodes that will have
different roles in the ``TensorNetwork``:
    
* **leaf**: These are the nodes that form the ``TensorNetwork`` (together
  with the ``data`` nodes). Usually, these will be the `trainable` nodes.
  These nodes can store their own tensors or use other node's tensor.

  Both ``Nodes`` and ``ParamNodes`` can be ``leaf``. In fact, all nodes
  will be ``leaf`` by default::
    
    import torch
    import tensorkrowch as tk
    
    node = tk.randn(shape=(2, 5, 2))
    assert node.is_leaf()

    paramnode = tk.randn(shape=(2, 5, 2),
                          param_node=True)
    assert paramnode.is_leaf()

  ``leaf`` nodes of the network can be retrieved via::

    net.leaf_nodes
  
* **data**: These are similar to ``leaf`` nodes, but they are never `trainable`,
  and are used to store the temporary tensors coming from input data. These
  nodes can store their own tensors or use other node's tensor.

  ``data`` nodes can be instantiated explicitly or via
  :meth:`~TensorNetwork.set_data_nodes`::

    # Data node instantiated explicitly
    data_node = tk.Node(shape=(100, 5),
                        axes_names=('batch', 'feature'),
                        data=True)

    # Data nodes created specifying to which edges
    # they should be connected
    net = tk.TensorNetwork()
    input_edges = []

    for i in range(100):
        node = tk.randn(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        param_node=True)
        input_edges.append(node['input'])

    net.set_data_nodes(input_edges,
                        num_batch_edges=1)

  ``data`` nodes of the network can be retrieved via::

    net.data_nodes
    assert net['data_0'].is_data()
  
* **virtual**: These nodes are a sort of ancillary, `hidden` nodes that
  accomplish some useful task (e.g. in uniform tensor networks a virtual
  node can store the shared tensor, while all the other nodes in the
  network just have a reference to it). These nodes always store their own
  tensors::

    mps = tk.TensorNetwork(name='mps')
    nodes = []

    uniform_node = tk.Node(shape=(2, 5, 2),
                            axes_names=('left', 'input', 'right'),
                            name='virtual_uniform',
                            network=mps,
                            virtual=True)
    assert uniform_node.is_virtual()

    for i in range(100):
        node = tk.randn(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name=f'node_({i})',
                        network=mps)
        node.set_tensor_from(uniform_node)

        nodes.append(node)
        
    for i in range(100):
        mps[f'node_({i})']['right'] ^ mps[f'node_({(i + 1) % 100})']['left']

    # Check that all nodes share tensor with uniform_node
    for node in nodes:
        assert node.tensor_address() == 'virtual_uniform'

  Giving the ``uniform_node`` the role of ``virtual`` makes more sense,
  since it is a node that one wouldn't desire to see as a ``leaf`` node
  of the network. Instead it is `hidden`.
  
  In the next section you will see that the name ``"virtual_uniform"`` that
  we chose for the ``uniform_node`` is convenient for the case of uniform
  tensor networks.

  ``virtual`` nodes of the network can be retrieved via::

    net.virtual_nodes
  
* **resultant**: These are nodes that result from an :class:`Operation`.
  They are intermediate nodes that (almost always) inherit edges from ``leaf``
  and ``data`` nodes, the ones that really form the network. These nodes can
  store their own tensors or use other node's tensor. The names of the
  ``resultant`` nodes are the name of the ``Operation`` that originated it::

    node1 = tk.randn(shape=(2, 3))
    node2 = tk.randn(shape=(3, 4))
    node1[1] ^ node2[0]

    result = node1 @ node2
    assert result.is_resultant()

  ``resultant`` nodes cannot be instantiated directly, they can only be
  originated from ``Operations``.

  ``resultant`` nodes of the network can be retrieved via::
    
    net.resultant_nodes

To retrieve all the nodes in the network you can do it with::

  net.nodes


2. Reserved Nodes' Names
------------------------

Other thing one should take into account are **reserved nodes' names**:

* **"stack_data_memory"**: Name of the ``virtual`` :class:`StackNode` that
  is created in :meth:`~TensorNetwork.set_data_nodes` to store the whole data
  tensor from which each ``data`` node might take just one `slice`. There should
  be at most one ``"stack_data_memory"`` in the network. To learn more about
  this, see :meth:`~TensorNetwork.set_data_nodes` and
  :meth:`~TensorNetwork.add_data`.

  ::

    net = tk.TensorNetwork()
    input_edges = []

    for i in range(100):
        node = tk.randn(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        param_node=True)
        input_edges.append(node['input'])

    net.set_data_nodes(input_edges,
                        num_batch_edges=1)
    
    # Batch edge has size 1 when created
    assert net['stack_data_memory'].shape == (100, 1, 5)
    
* **"virtual_result"**: Name of the ``virtual`` :class:`ParamStackNode` that
  results from stacking ``ParamNodes`` as the first operation in the network
  contraction, if ``auto_stack`` mode is set to ``True``. There might be as
  much ``"virtual_result"`` nodes as stacks are created from ``ParamNodes``. To
  learn more about this, see :class:`ParamStackNode`. This special name can
  be used for all sort of ``virtual`` nodes that are not part of the network
  explicitly, but are required in some situations.

  ::

    net = tk.TensorNetwork()
    net.auto_stack = True

    nodes = []
    for i in range(100):
        node = tk.randn(shape=(2, 5, 2),
                        network=net,
                        param_node=True)
        nodes.append(node)

    stack_node = tk.stack(nodes)

    # All ParamNodes use a slice of the tensor in stack_node
    for node in nodes:
        assert node.tensor_address() == 'virtual_result'

* **"virtual_uniform"**: Name of the ``virtual`` ``Node`` or ``ParamNode`` that
  is used in uniform (translationally invariant) tensor networks to store the
  tensor that will be shared by all ``leaf`` nodes. There might be as much
  ``"virtual_uniform"`` nodes as shared memories are used for the ``leaf``
  nodes in the network (usually just one). An example of this can be seen in
  the previous section, when ``virtual`` nodes were defined.
    
Although these names can in principle be used for other nodes, this can lead
to undesired behaviour.

The **4 types of nodes** and the **reserved nodes' names** will also play a 
role when :meth:`tracing <TensorNetwork.trace>` of
:meth:`resetting <TensorNetwork.reset>` the ``TensorNetwork``. See the next
:ref:`tutorial <tutorial_5>` to learn more.
