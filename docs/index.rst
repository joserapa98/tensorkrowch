==========================
TensorKrowch documentation
==========================

**TensorKrowch** is a Python library built on top of **PyTorch** that simplifies
the training of Tensor Networks as machine learning models and their integration
into deep learning pipelines.

The primary goal of **TensorKrowch** is to offer an efficient and user-friendly
framework for constructing and training diverse Tensor Networks. By providing
essential components like ``Nodes``, ``Edges``, and ``TensorNetworks``,
**TensorKrowch** facilitates the creation and training of these models. Notably,
even the included implementations of ``MPS`` or ``PEPS`` only rely on these
fundamental components.

As a result, users who grasp the basic tools of **TensorKrowch** gain the ability
to build a wide range of networks, ranging from simple Matrix Product States to
more intricate architectures.

The true strength of **TensorKrowch** lies in its support for rapid experimentation,
enabling users to create and train different models with just a few lines of code
changes.

It's important to note that while **TensorKrowch** is a versatile library, it
may not always be the fastest option in certain scenarios. However, it excels
as a tool for exploration and identification of the most suitable Tensor Network.
Once the ideal network is determined, users can develop further optimized code
specifically tailored to that network.

Nevertheless, **TensorKrowch** incorporates various optimizations to ensure
efficient training performance.


Requirements
============

* python >= 3.7
* torch >= 1.9
* opt_einsum >= 3.0


Installation
============

.. include:: installation.rst
   :start-line: 3


Example
=======

With **TensorKrowch** you can experiment building Tensor Networks::

   import torch
   import tensorkrowch as tk

   net = tk.TensorNetwork()

   node1 = tk.randn(shape=(7, 5),
                    axes_names=('left', 'right'),
                    name='node1',
                    network=net,
                    param_node=True)
   node2 = tk.randn(shape=(7, 5),
                    axes_names=('left', 'right'),
                    name='node2',
                    network=net,
                    param_node=True)

   node1['left'] ^ node2['left']
   node1['right'] ^ node2['right']

It is also quite easy to contract the network and compute gradients::

   result = node1 @ node2
   result.tensor.backward()

   assert node1.grad is not None
   assert node2.grad is not None

In **TensorKrowch** ``TensorNetworks`` work like **PyTorch** layers. Thus
creating hybrid neural-tensor network models is straightforward::

   import torch.nn as nn
   
   my_model = nn.Sequential(
      tk.models.MPSLayer(n_features=100,
                         in_dim=3,
                         out_dim=10,
                         bond_dim=5),
      nn.ReLU(),
      nn.Linear(10, 10))

   data = torch.randn(500, 100, 3)  # batch x n_features x in_dim
   my_model(data)  # batch x out_dim


Tutorials
=========

To fully grasp the basic components of **TensorKrowch** and harness its potential,
it is highly recommended to explore the available tutorials. These tutorials
provide a detailed introduction to the fundamental elements of the library and
guide you through the process of constructing and training tensor networks.

By immersing yourself in the tutorials, you will become familiar with key
concepts and best practices for using **TensorKrowch**. You will learn how to
define ``Nodes``, create connections between through their ``Edges``, and
configure the ``TensorNetwork`` structure. This hands-on approach will greatly
enhance your understanding and proficiency with **TensorKrowch**.

.. include:: tutorials.rst
   :start-line: 3


License
=======

**TensorKrowch** is licensed under the MIT License. Please see the
`LICENSE <https://github.com/joserapa98/tensorkrowch/blob/master/LICENSE.txt>`_
file for more information.


Citing
======



..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
