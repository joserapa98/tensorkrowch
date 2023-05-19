.. currentmodule:: tensorkrowch
.. _tutorial_3:

=============================================
How to save Memory and Time with TensorKrowch
=============================================

Since ``TensorKrowch`` is devoted to construct tensor network models, many of
the operations one can compute between nodes have to keep track of information
of the underlying graph. However, this could be very costly if we had to compute
all these ancillary steps every time during training. That is why ``TensorKrowch``
uses different tricks like managing memory and skipping operations automatically
in order to save some memory and time.


Introduction
============

In this tutorial you will learn how memory is stored in the tensor network and
what are the tricks ``TensorKrowch`` uses to take advantage of that.

Also, you will learn how some steps are skipped in :class:`Operations <Operation>`
in order to save time. Learn more about how to use operations in the previous
:ref:`tutorial <tutorial_2>`.


Steps
=====

1. How Tensors are stored in the TensorNetwork.
2. How TensorKrowch skipps Operations to run faster.
3. Memory Management modes.


1. How Tensors are stored in the TensorNetwork
----------------------------------------------

As explained in the first :ref:`tutorial <tutorial_1>`, although nodes act as a
sort of `containers` for ``torch.Tensor``'s, this is not what happens under the
hood.

Actually, each ``TensorNetwork`` has a unique memory where all tensors are stored.
This memory can be accessed by all nodes to retrieve their respective tensors.
Hence, all that nodes `contain` is just a **memory address** together with some
other information that helps to access the correct tensor.

When a node is instantiated a memory address is created with the name of the
node. That is where its tensor will be stored. Even if the node is empty, that
place is reserved in case we set a tensor in the empty node.

We can check the memory address of our nodes via::

    import torch
    import tensorkrowch as tk

    node1 = tk.Node(shape=(2, 5, 2),
                    name='my_node')
    node1.tensor_address()

For now, there is no tensor stored in that memory address::

    node1.tensor

But we can set a new tensor into the node::

    new_tensor = torch.randn(2, 5, 2)
    node1.tensor = new_tensor  # Same as node1.set_tensor(new_tensor)

Now ``node`` is not empty, there is a tensor stored in its corresponding memory
address::

    node1.tensor

Since nodes only contain memory addresses, we can create a second node that
instead of storing its own memory address, it uses the memory of the first node::

    node2 = tk.Node(shape=(2, 5, 2),
                    name='your_node',
                    network=node1.network)
    node2.set_tensor_from(node1)

    assert node2.tensor_address() == 'my_node'

Of course, to share memory, both nodes need to be in the same network and have
the same shape.

Now, if we change the tensor in ``node1``, ``node2`` will reproduce the same
change::

    node1.tensor = torch.zeros_like(node1.tensor)
    node2.tensor

Furthermore, we can have even more nodes sharing the same memory::

    node3 = tk.Node(shape=(2, 5, 2),
                    name='other_node',
                    network=node1.network)
    node3.set_tensor_from(node2)

    assert node3.tensor_address() == 'my_node'

This feature of ``TensorKrowch`` can be very useful to create uniform or
translationally invariant tensor networks by simply using a node whose memory
is shared by all the nodes in the network. Let's create a Uniform Matrix
Product State::

    mps = tk.TensorNetwork(name='mps')
    nodes = []

    uniform_node = tk.randn(shape=(2, 5, 2),
                            axes_names=('left', 'input', 'right'),
                            name='uniform',
                            network=mps)

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
        assert node.tensor_address() == 'uniform'


2. How TensorKrowch skipps Operations to run faster
---------------------------------------------------

The main purpose of ``TensorKrowch`` is enabling you to experiment
creating and training different tensor networks, only having to worry about
instantiating nodes, making connections and performing contractions.

Because of that, there is much going on under the hood. For instance, say you
want to contract these two nodes::

    node1 = tk.randn(shape=(3, 7, 5))
    node2 = tk.randn(shape=(2, 5, 7))
    node1[1] ^ node2[2]
    node1[2] ^ node2[1]

    result = node1 @ node2

``TensorKrowch`` returns directly the resultant node with its edges in the
correct order. To perform that contraction in vanilla ``PyTorch``, you would
have to permute and reshape both nodes to compute a matrix multiplication, and
then reshape and permute again the result. And for every different node you
would have to think how to do the permutes and reshapes to leave the resultant
edges in the desired order.

``TensorKrowch`` does all of that for you, but it is costly. To avoid having
such overhead compared to just performing a matrix multiplication,
``TensorKrowch`` calculates how the permutes and reshapes should be performed
only during the first time a contraction occurs. Then all the ancillary
information needed to perform the contraction is saved in a sort of cache memory.
In subsequent contractions, ``TensorKrowch`` will behave almost like vanilla
``PyTorch``.


3. Memory Management modes
--------------------------

Now that you know how ``TensorKrowch`` manages memory and skips some steps
when operating with nodes repeatedly, you can learn about **two important modes**
that can be turned on/off for training or inference.

* **auto_stack** (``False`` by default): This mode indicates whether the
  operation :func:`stack` can take control of the memory management of the
  network to skip some steps in future computations. If ``auto_stack`` is set
  to ``True`` and a collection of :class:`ParamNodes <ParamNode>` are
  :func:`stacked <stack>` (as the first operation in which these nodes are
  involved), then those nodes will no longer store their own tensors, but
  rather a ``virtual`` :class:`ParamStackNode` will store the stacked tensor,
  avoiding the computation of that first :func:`stack` in every contraction.
  This behaviour is not possible if ``auto_stack`` is set to ``False``, in
  which case all nodes will always store their own tensors.
  
  Setting ``auto_stack`` to ``True`` will be faster for both **inference** and
  **training**. However, while experimenting with ``TensorNetworks`` one might
  want that all nodes store their own tensors to avoid problems.

  ::
  
      net = tk.TensorNetwork()
      net.auto_stack = True
  
      nodes = []
      for i in range(100):
          node = tk.randn(shape=(2, 5, 2),
                          network=net)
          nodes.append(node)
  
      # First operation is computed
      stack_node = tk.stack(nodes)
  
      # All ParamNodes use a slice of the tensor in stack_node
      for node in nodes:
          assert node.tensor_address() == stack_node.name
  
      # Second operation does nothing
      stack_node = tk.stack(nodes)


* **auto_unbind** (``False`` by default): This mode indicates whether the
  operation :func:`unbind` has to actually `unbind` the stacked tensor or just
  generate a collection of references. That is, if ``auto_unbind`` is set to
  ``False``, :func:`unbind` creates a collection of nodes, each of them storing
  the corresponding slice of the stacked tensor. If ``auto_unbind`` is set to
  ``True``, :func:`unbind` just creates the nodes and gives each of them an
  index to reference the stacked tensor, so that each node's tensor would be
  retrieved by indexing the stack. This avoids performing the operation, since
  these indices will be the same in subsequent iterations.
  
  Setting ``auto_unbind`` to ``True`` will be faster for **inference**, but
  slower for **training**.

  ::
  
      net = tk.TensorNetwork()
      net.auto_unbind = True
  
      nodes = []
      for i in range(100):
          node = tk.randn(shape=(2, 5, 2),
                          network=net)
          nodes.append(node)
  
      stack_node = tk.stack(nodes)
  
      # First operation is computed
      unbinded_nodes = tk.unbind(stack_node)
  
      # All unbinded nodes use a slice of the tensor in stack_node
      for node in unbinded_nodes:
          assert node.tensor_address() == stack_node.name
  
      # Second operation does nothing
      unbinded_nodes = tk.unbind(stack_node)

Once the training algorithm starts, these modes should not be changed (very
often at least), since changing them entails first :meth:`resetting <reset>`
the whole network, which is a costly method.

To learn more about what ``virtual`` and other types of nodes are, check the
next :ref:`tutorial <tutorial_4>`. 
