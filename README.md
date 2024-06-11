![logo](https://github.com/joserapa98/tensorkrowch/blob/master/docs/figures/svg/tensorkrowch_logo_dark.svg#gh-dark-mode-only)
![logo](https://github.com/joserapa98/tensorkrowch/blob/master/docs/figures/svg/tensorkrowch_logo_light.svg#gh-light-mode-only)

[![DOI](https://zenodo.org/badge/453954432.svg)](https://zenodo.org/badge/latestdoi/453954432)

# Tensor Networks with PyTorch

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


## Documentation

For detailed usage instructions, API reference, and code examples, please refer
to the official **TensorKrowch** [documentation](https://joserapa98.github.io/tensorkrowch).


## Requirements

* python >= 3.8
* torch >= 1.9
* opt_einsum >= 3.0


## Installation

To install the package, run the following command:

```
pip install tensorkrowch
```

You can also install directly from GitHub with:

```
pip install git+https://github.com/joserapa98/tensorkrowch.git@master
```

or download the repository on your computer and run 

```
pip install .
```

in the repository folder.

Tests are written outside the Python module, therefore they are not installed
together with the package. To test the installation, clone the repository and
run, in a Unix terminal

```
python -m pytest -v
```

inside the repository folder.

> [!NOTE]
Certain tests may experience failure as a result of statistical anomalies or 
hardware constraints. We advise reviewing the error messages to determine if 
these failures stem from such occurrences. Should this be the case, consider 
rerunning the tests to ascertain if the errors persist.


## Example

With **TensorKrowch** you can experiment building Tensor Networks:

```python
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
```

It is also quite easy to contract the network and compute gradients:

```python
result = node1 @ node2
result.tensor.backward()

assert node1.grad is not None
assert node2.grad is not None
```

In **TensorKrowch** ``TensorNetworks`` work like **PyTorch** layers. Thus
creating hybrid neural-tensor network models is straightforward:

```python
import torch.nn as nn

my_model = nn.Sequential(
    tk.models.MPSLayer(n_features=101,
                       in_dim=3,
                       out_dim=10,
                       bond_dim=5),
    nn.ReLU(),
    nn.Linear(10, 10))

data = torch.randn(500, 100, 3)  # batch x n_features x in_dim
my_model(data)  # batch x out_dim
```


## Tutorials

To fully grasp the basic components of **TensorKrowch** and harness its potential,
it is highly recommended to explore the available tutorials. These tutorials
provide a detailed introduction to the fundamental elements of the library and
guide you through the process of constructing and training tensor networks.

By immersing yourself in the tutorials, you will become familiar with key
concepts and best practices for using **TensorKrowch**. You will learn how to
define ``Nodes``, create connections between through their ``Edges``, and
configure the ``TensorNetwork`` structure. This hands-on approach will greatly
enhance your understanding and proficiency with **TensorKrowch**.

* [First Steps with TensorKrowch](https://joserapa98.github.io/tensorkrowch/_build/html/tutorials/0_first_steps.html)
* [Creating a Tensor Network in TensorKrowch](https://joserapa98.github.io/tensorkrowch/_build/html/tutorials/1_creating_tensor_network.html)
* [Contracting and Differentiating the Tensor Network](https://joserapa98.github.io/tensorkrowch/_build/html/tutorials/2_contracting_tensor_network.html)
* [How to save Memory and Time with TensorKrowch (ADVANCED)](https://joserapa98.github.io/tensorkrowch/_build/html/tutorials/3_memory_management.html)
* [The different Types of Nodes (ADVANCED)](https://joserapa98.github.io/tensorkrowch/_build/html/tutorials/4_types_of_nodes.html)
* [How to subclass TensorNetwork to build Custom Models](https://joserapa98.github.io/tensorkrowch/_build/html/tutorials/5_subclass_tensor_network.html)
* [Creating a Hybrid Neural-Tensor Network Model](https://joserapa98.github.io/tensorkrowch/_build/html/tutorials/6_mix_with_pytorch.html)


## Example Notebooks

In addition to the informative tutorials, there is also a collection of examples
that serve as practical demonstrations of how to apply **TensorKrowch** in
various contexts, showcasing its versatility.

With the code provided in the examples, you will be able to reproduce key research
findings that bridge the gap between tensor networks and machine learning. These
examples provide a hands-on approach to understanding the intricacies of
**TensorKrowch**, allowing you to explore its potential and adapt it to your
specific needs.

* [Training MPS in different ways](https://joserapa98.github.io/tensorkrowch/_build/html/examples/training_mps.html)
* [Hybrid Tensorial Neural Network model](https://joserapa98.github.io/tensorkrowch/_build/html/examples/hybrid_tnn_model.html)
* [Tensorizing Neural Networks](https://joserapa98.github.io/tensorkrowch/_build/html/examples/tensorizing_nn.html)
* [DMRG-like training of MPS](https://joserapa98.github.io/tensorkrowch/_build/html/examples/mps_dmrg.html)
* [Hybrid DMRG-like training of MPS](https://joserapa98.github.io/tensorkrowch/_build/html/examples/mps_dmrg_hybrid.html)


## License

**TensorKrowch** is licensed under the MIT License. Please see the [LICENSE](https://github.com/joserapa98/tensorkrowch/blob/master/LICENSE.txt) file for more information.


## Citing

If you use TensorKrowch in your work, please cite [TensorKrowch's paper](https://www.arxiv.org/abs/2306.08595):

- J. R. Pareja Monturiol, D. Pérez-García, and A. Pozas-Kerstjens, 
"TensorKrowch: Smooth integration of tensor networks in machine learning", 
Quantum **8**, 1364 (2024), arXiv:2306.08595.

```
@article{pareja2024tensorkrowch,
  title={Tensor{K}rowch: {S}mooth integration of tensor networks in machine learning},
  author={Pareja Monturiol, Jos{\'e} Ram{\'o}n and P{\'e}rez-Garc{\'i}a, David and Pozas-Kerstjens, Alejandro},
  journal={Quantum},
  volume={8},
  pages={1364},
  year={2024},
  publisher={Verein zur F{\"o}rderung des Open Access Publizierens in den Quantenwissenschaften},
  doi = {10.22331/q-2024-06-11-1364},
  archivePrefix = {arXiv},
  eprint = {2306.08595}
}
```
