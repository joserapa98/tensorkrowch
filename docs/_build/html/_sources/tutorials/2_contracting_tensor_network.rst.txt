.. currentmodule:: tensorkrowch
.. _tutorial_2:

==================================================
Contracting and Differentiating the Tensor Network
==================================================

In the previous :ref:`tutorial <tutorial_1>` you learned how to build a fixed
tensor network. However, ``TensorKrowch`` is built on top of ``PyTorch`` in
order to be able to train these models as easily as any other ``torch.nn.Module``.
Hence, the next step should be to learn about the components of ``TensorKrowch``
that make it possible to compute `learnable` functions.


Introduction
============

In this tutorial you will learn about the two main classes of nodes in
``TensorKrowch`` and how to operate with them.


Steps
=====

1. Distinguish between Nodes and ParamNodes.
2. Operations between nodes.
3. Contracting a Matrix Product State.


1. Distinguish between Nodes and ParamNodes
-------------------------------------------

In ``TensorKrowch`` there are 2 main classes of nodes: the ones that are fixed
(:class:`Nodes <Node>`) and the ones you can train (:class:`ParamNodes <ParamNode>`).
The main (and almost only difference) is that ``Nodes`` contain a ``torch.Tensor``, 
while ``ParamNodes`` contain a ``torch.nn.Parameter``, the `tensors` of PyTorch
with respect to which gradients are computed.

``ParamNodes`` are initalized in the same fashion as ``Nodes``::

    import torch
    import torch.nn as nn
    import tensorkrowch as tk

    paramnode1 = tk.ParamNode(shape=(2, 5, 2))     # Empty paramnode
    paramnode2 = tk.ParamNode(shape=(2, 5, 2),
                              init_method='randn')
    paramnode3 = tk.randn(shape=(2, 5, 2),
                          param_node=True)  # Indicates if node is ParamNode

Also, if we try to initialize a ``ParamNode`` with an existing ``torch.Tensor``,
this will be first transformed into a ``torch.nn.Parameter``::

    tensor = torch.randn(2, 5, 2)
    paramnode = tk.ParamNode(tensor=tensor)

    assert isinstance(paramnode.tensor, nn.Parameter)

Another important and useful feature of ``TensorKrowch`` is that you can
`parameterize` ``Nodes`` or `de-parameterize` ``ParamNodes`` at any time::

    node = paramnode.parameterize(False)
    assert isinstance(node.tensor, torch.Tensor)
    assert not isinstance(node.tensor, nn.Parameter)

    paramnode = node.parameterize()
    assert isinstance(paramnode.tensor, nn.Parameter)

Be aware that when parameterizing or de-parameterizing, the previous ``Node``
or ``ParamNode`` will be overriden in the network by the new ``ParamNode`` or
``Node``, respectively.

Finally, to check that, effectively, these ``ParamNodes`` can be trained, let's
compute a simple function and differentiate::

    sum = paramnode.sum()  # Sums over all axes of the node
    sum.backward()         # Differentiates sum with respect to paramnode

Now with ``ParamNodes`` we can access directly the gradient of their tensors via::

    paramnode.grad

Although this is insightful to learn the basics of ``ParamNodes``, we want
tools to work with tensor networks. In the next section you will learn
about an important part of ``TensorKrowch``: :class:`Operations <Operation>`.


2. Operations between Nodes
---------------------------

In ``TensorKrowch`` there are some :class:`Operations <Operation>` you can
compute between nodes. We can distinguish between two types of operations:

a) **Tensor-like**: We refer to the operations one can compute using tensors in
   vanilla ``PyTorch`` like :func:`permute` (and the in-place variant
   :func:`permute_`), :func:`tprod` (tensor product), :func:`mul`, :func:`add`
   and :func:`sub`.

b) **Node-like**: We refer to the operations one will need to contract a tensor
   network. These we will explain in more detail in this section.

For both types of operations, the result will always be a ``Node``. That is,
``ParamNodes`` can only be used as the `initial` nodes that define a tensor
network, and with respect to which we will differentiate. But all intermediate
nodes that result from an operation will be non-parametric ``Nodes``.

Regarding the **node-like** operations, these are:

1) :func:`contract_between`: Contracts all connected edges between two nodes.
   The operand ``@`` can be used to perform the contraction::

    node1 = tk.randn(shape=(2, 3),
                     axes_names=('left', 'right'))
    node2 = tk.randn(shape=(2, 3),
                     axes_names=('left', 'right'))
    
    node1['left'] ^ node2['left']
    node1['right'] ^ node2['right']

    result = node1 @ node2

    assert result.shape == ()

   There also variants of this operations. You can contract nodes in-place with
   :func:`contract_between_`), that is, modifying the initial network you defined.
   You can also contract only selected edges with :func:`contract_edges`.

2) :func:`split`: Splits a node in two via Singular Value or QR decompositions.
   The edges that go with each resultant node should be specified::

    node = tk.randn(shape=(2, 3, 4, 5),
                    axes_names=('left1', 'left2', 'right1', 'right2'))
    res1, res2 = node.split(['left1', 'right1'],
                            ['left2', 'right2'],
                            rank=2)

    assert res1.shape == (2, 4, 2)
    assert res2.shape == (2, 3, 5)

   As can be noted, there is also a new edge connecting the resultant nodes.
   Similar to ``contract_between``, there is also an in-place variant :func:`split_`.

3) :func:`stack`: Stacks a list of nodes of the same `type`. That is, only nodes
   with the same number of edges, same axes names and belonging to the same
   network. The sizes of each edge, however, can be different for different nodes::

    net = tk.TensorNetwork()
    nodes = []

    for i in range(100):
        node = tk.randn(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        name=f'node_({i})')
        nodes.append(node)

    stack_node = tk.stack(nodes)

    assert stack_node.shape == (100, 2, 5, 2)

   The resultant ``stack_node`` is actually a different class of node, a
   :class:`StackNode`. These only result from stacking other nodes, and have
   as first edge a special **batch** edge called ``"stack"``. The rest of edges
   are of class :class:`StackEdge`, a new type of edge that collect information
   from all the edges from the nodes that are being stacked. This information
   enables to automatically reconnect nodes to their previous neighbours when
   ``unbinding`` the stack.

   Be aware that stacks **cannot recognize neighbours**. That is, if we create
   two stacks of nodes that were all connected one-to-one, we have to reconnect
   the stacks::

    net = tk.TensorNetwork()
    nodes = []
    data_nodes = []

    for i in range(100):
        node = tk.randn(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        name=f'node_({i})')
        nodes.append(node)

        data_node = tk.randn(shape=(100, 5),
                        axes_names=('batch', 'feature'),
                        network=net,
                        name=f'data_node_({i})')
        data_nodes.append(data_node)

        node['input'] ^ data_node['feature']

    stack_node = tk.stack(nodes)
    stack_data_node = tk.stack(data_nodes)

    stack_node['input'] ^ stack_data_node['feature']

4) :func:`unbind`: Unbinds a :class:`StackNode` and returns a list of nodes that
   are already connected to the corresponding neighbours::

    net = tk.TensorNetwork()
    nodes = []

    for i in range(100):
        node = tk.randn(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        name=f'node_({i})')
        nodes.append(node)

    stack_node = tk.stack(nodes)
    unbinded_nodes = tk.unbind(stack_node)

    assert unbinded_nodes[0].shape == (2, 5, 2)

5) :func:`einsum`: Evaluates the Einstein summation convention on the nodes.
   It is based on `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/autosummary/opt_einsum.contract.html>`_::

    node1 = tk.randn(shape=(10, 15, 100),
                     axes_names=('left', 'right', 'batch'))
    node2 = tk.randn(shape=(15, 7, 100),
                     axes_names=('left', 'right', 'batch'))
    node3 = tk.randn(shape=(7, 10, 100),
                     axes_names=('left', 'right', 'batch'))

    node1['right'] ^ node2['left']
    node2['right'] ^ node3['left']
    node3['right'] ^ node1['left']
    
    result = tk.einsum('ijb,jkb,kib->b', node1, node2, node3)
    
    assert result.shape == (100,)

   There is another variant of ``einsum`` that accepts a sequence of lists of
   nodes and previously stacks each list of nodes in a ``StackNode`` and then
   evaluates a batched version of ``einsum``. This operation is
   :func:`stacked_einsum`.

Some of this operations can also be called from the nodes' edges, like
:func:`contract_` or :func:`svd_`.


3. Contracting a Matrix Product State
-------------------------------------

Now that you know how to construct a :class:`TensorNetwork` with ``ParamNodes``,
and use ``Operations`` between them, let's apply all of this to contract a 
Matrix Product State (MPS) with some input data, and compute gradients of the
result with respect to the MPS nodes::

    mps = tk.TensorNetwork(name='mps')
    nodes = []
    data_nodes = []

    for i in range(100):
        node = tk.randn(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=mps,
                        name=f'node_({i})',
                        param_node=True)
        nodes.append(node)

        data_node = tk.randn(shape=(5,),
                             axes_names=('feature',),
                             network=mps,
                             name=f'data_node_({i})')
        data_nodes.append(data_node)

        node['input'] ^ data_node['feature']
        
    for i in range(100):
        mps[f'node_({i})']['right'] ^ mps[f'node_({(i + 1) % 100})']['left']

With this, we have already created our MPS where nodes can be trained. We have
also added some data nodes that will hold our data (though in this example they
will be filled with random tensors).

To contract all the data nodes with their respective neighbours we can use
``stack`` to perform a single big contraction, instead of a hundread of small
contractions, which will save us some time::

    stack_node = tk.stack(nodes)
    stack_data_node = tk.stack(data_nodes)

    stack_node['input'] ^ stack_data_node['feature']
    
    stack_result = stack_node @ stack_data_node
    unbind_result = tk.unbind(stack_result)

Now we have a list with a bunch of matrices that are all connected to the
previous and next ones, forming a ring. Let's contract all of them with
simple contractions::

    result = unbind_result[0]
    for node in unbind_result[1:]:
        result @= node

    assert result.shape == ()

Since we have contracted the whole network, and no edge is still dangling, the
result is a single number. We can then compute gradients::

    result.tensor.backward()

    for node in nodes:
        assert node.grad is not None

Here we have our desired gradient! Now you can use it to learn a function using
gradient descent methods.
