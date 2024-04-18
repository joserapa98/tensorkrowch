.. currentmodule:: tensorkrowch
.. _tutorial_5:

====================================================
How to subclass TensorNetwork to build Custom Models
====================================================

In order to mimic the ``PyTorch`` convention, a model in ``TensorKrowch``
is always a subclass of ``TensorNetwork``, which is itself a subclass of
``torch.nn.Module``. Therefore, to create models properly and be able to
combine them easily with other ``PyTorch`` layers (like ``torch.nn.Linear``)
one should learn what is the convention for creating subclasses of
``TensorNetwork``.


Introduction
============

In all the previous tutorials you have learned about the pieces of a
``TensorNetwork`` such as ``Nodes`` and ``Edges``. You should have already
built your first `trainable` ``TensorNetwork``, even using nodes with
different **roles** (``leaf``, ``data``, ``virtual``). Not only have you
created a ``TensorNetwork``, but contracted it using ``Operations``.

In this tutorial we will put everything together to build your first custom
``TensorNetwork`` model. You will also learn how to pass data through your
model, ``trace`` it, and ``reset`` it.


Steps
=====

1. The Components of a TensorNetwork Subclass.
2. Putting Everything Together.
3. First Steps with our Custom Model.


1. The Components of a TensorNetwork Subclass
---------------------------------------------

Recall that the common way of defining models out of ``torch.nn.Module`` is
by defining a subclass where the ``__init__`` and ``forward`` methods are
overriden:

* **__init__**: Defines the model itself (its layers, attributes, etc.).

* **forward**: Defines the way the model operates, that is, how the different
    parts of the model might combine to get an output from a particular input.

With ``TensorNetwork``, the workflow is similar, though there are other
methods that should be overriden:

* **__init__**: Defines the graph of the tensor network and initializes the
  tensors of the nodes.
    
* **set_data_nodes** (optional): Creates the data nodes where the data
  tensor(s) will be placed. Usually, it will just select the edges to which
  the ``data`` nodes should be connected, and call the
  :meth:`parent method <TensorNetwork.set_data_nodes>`.
    
* **add_data** (optional): Adds new data tensors that will be stored in ``data``
  nodes. Usually it will not be necessary to override this method, but if one
  wants to customize how data is set into the ``data`` nodes, :meth:`add_data`
  can be overriden.
    
* **contract**: Defines the contraction algorithm of the whole tensor network,
  thus returning a single node. Very much like ``forward`` this is the main
  method that describes how the components of the network are combined. Hence,
  in ``TensorNetwork`` the ``forward`` method shall not be overriden, since it
  will just call ``set_data_nodes``, if needed, ``add_data`` and ``contract``,
  and then it will return the tensor corresponding to the last ``resultant``
  node. Hence, the order in which ``Operations`` are called from ``contract``
  is important. The last operation must be the one returning the final node.


2. Putting Everything Together
------------------------------

Let's create a class for Matrix Product States that we will use to classify
images::

    import torch
    import tensorkrowch as tk

    class MPS(tk.TensorNetwork):

        def __init__(self, image_size, uniform=False):
            """
            In the __init__ method we define the tensor network
            structure and initialize all the nodes
            """
            super().__init__(name='MPS')

            #############
            # Create TN #
            #############
            input_nodes = []

            # Number of input nodes equal to number of pixels
            for _ in range(image_size[0] * image_size[1]):
                node = tk.ParamNode(shape=(10, 3, 10),
                                    axes_names=('left', 'input', 'right'),
                                    name='input_node',
                                    network=self)
                input_nodes.append(node)

            for i in range(len(input_nodes) - 1):
                input_nodes[i]['right'] ^ input_nodes[i + 1]['left']

            # Output node is in the last position,
            # but that could be changed
            output_node = tk.ParamNode(shape=(10, 10, 10),
                                       axes_names=(
                                           'left', 'output', 'right'),
                                       name='output_node',
                                       network=self)
            output_node['right'] ^ input_nodes[0]['left']
            output_node['left'] ^ input_nodes[-1]['right']

            self.input_nodes = input_nodes
            self.output_node = output_node

            # If desired, the MPS can be uniform
            if uniform:
                uniform_memory = tk.ParamNode(shape=(10, 3, 10),
                                              axes_names=(
                                                  'left', 'input', 'right'),
                                              name='virtual_uniform',
                                              network=self,
                                              virtual=True)
                self.uniform_memory = uniform_memory

            ####################
            # Initialize Nodes #
            ####################

            # Input nodes
            if uniform:
                std = 1e-9
                tensor = torch.randn(uniform_memory.shape) * std
                random_eye = torch.randn(
                    tensor.shape[0], tensor.shape[2]) * std
                random_eye = random_eye + \
                    torch.eye(tensor.shape[0], tensor.shape[2])
                tensor[:, 0, :] = random_eye

                uniform_memory.tensor = tensor

                # Memory of each node is just a reference
                # to the uniform_memory tensor
                for node in input_nodes:
                    node.set_tensor_from(self.uniform_memory)

            else:
                std = 1e-9
                for node in input_nodes:
                    tensor = torch.randn(node.shape) * std
                    random_eye = torch.randn(
                        tensor.shape[0], tensor.shape[2]) * std
                    random_eye = random_eye + \
                        torch.eye(tensor.shape[0], tensor.shape[2])
                    tensor[:, 0, :] = random_eye

                    node.tensor = tensor

            # Output node
            eye_tensor = torch.eye(output_node.shape[0], output_node.shape[2])\
                .view([output_node.shape[0], 1, output_node.shape[2]])
            eye_tensor = eye_tensor.expand(output_node.shape)
            tensor = eye_tensor + std * torch.randn(output_node.shape)

            output_node.tensor = tensor

            self.input_nodes = input_nodes
            self.output_node = output_node

        def set_data_nodes(self) -> None:
            """
            This method is optional. If overriden, it should not have
            arguments other than self. Furthermore, we won't have to
            call it explicitly, since it will be called from forward.
            
            If not overriden, it should be explicitly called before
            training.
            """
            # Select input edges where to put data nodes
            input_edges = []
            for node in self.input_nodes:
                input_edges.append(node['input'])

            # num_batch_edges inicates number of batch edges. Usually
            # it will be 1, for the batch of input data. But for
            # convolutional or sequential models it could be 2,
            # one edge for the batch of input data, and one for the
            # patches of sequence
            super().set_data_nodes(input_edges,
                                    num_batch_edges=1)

        def contract(self):
            """
            In this method we define the contraction algorithm
            for the tensor network.

            The last operation computed must be the one returning
            the final node.
            """
            stack_input = tk.stack(self.input_nodes)
            stack_data = tk.stack(list(self.data_nodes.values()))

            stack_input ^ stack_data
            stack_result = stack_input @ stack_data

            stack_result = tk.unbind(stack_result)

            result = stack_result[0]
            for node in stack_result[1:]:
                result @= node
            result @= self.output_node

            return result


3. First Steps with our Custom Model
------------------------------------

Now we can instantiate our model::

    image_size = (10, 10)
    mps = MPS(image_size=image_size)

Since our model is a subclass of ``torch.nn.Module``, we can take advantage of
its methods. For instance, we can easily send the model to the GPU::

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mps = mps.to(device)

Note that only the ``ParamNodes`` are parameters of the model. Thus if your
model has other non-parametric ``Nodes``, these won't be sento to the GPU.
Instead, you should fill them with tensors that are already in the GPU.

Let's get some images to use as our data::

    images = torch.randn(500, image_size[0], image_size[1])

Images are tensors with shape ``batch x height x width``. That is, for each
pixel, for each image in the batch we have just one value. Yet the nodes in
our tensor network are all tensors (vectors, matrices, etc.), so we need to
embed our pixel values into a greater vector space. That is, for each pixel,
for each image in the batch, we should have a vector (of dimension 3 in this
case, the size of ``input`` edges of the nodes in our ``MPS`` class).

::

    images = tk.embeddings.poly(images, axis=1)
    images = images.to(device)

The input data for ``TensorNetwork`` subclasses, when :meth:`~TensorNetwork.add_data`
has not been overriden, should have shape ``batch_1 x ... x batch_n x
n_features x feature_dim``::

    images = images.view(
        500, 3, image_size[0] * image_size[1]).transpose(1, 2)

Before starting training, there are **2 important steps** you have to do
manually. First, set the memory management modes as you wish. Usually, for
training it will be faster to set ``auto_stack`` to ``True`` and ``auto_unbind``
to ``False``. For inference, it will be faster to set both nodes to ``True``.

::

    mps.auto_stack = True
    mps.auto_unbind = False

Secondly, you should :meth:`~TensorNetwork.trace` the ``TensorNetwork``. To
trace the network you need to pass an example input data through your model
in order to perform all the heavy computations (compute all ``Operations`` for
the first time, create all the intermediate ``resultant`` nodes, etc..). The
example can be simply a tensor of zeros with the appropiate shape. In fact,
the batch size can be set to 1::

    example = torch.zeros(1, image_size[0] * image_size[1], 3)
    example = example.to(device)
    mps.trace(example)

Finally everything is ready to train our model::

    # Training loop
    for epoch in range(n_epochs):
        ...
        result = mps(images)
        ...

When the training has finished, and you want to save the model, you will need
to :meth:`~TensorNetwork.reset` it. Note that when the model was traced, a
bunch of ``resultant`` nodes were added to the network. If now you instantiate
``MPS`` again, all those nodes won't be in the new instance. The networks
will have different nodes. Therefore, you need to ``reset`` the network to its
initial state, how it was before tracing::

    mps.reset()

With this, you can save your model and load it later::

    # Save
    torch.save(mps.state_dict(), 'mps.pt')

    # Load
    new_mps = MPS(image_size)
    new_mps.load_state_dict(torch.load('mps.pt'))

Of course, although you can create any tensor network you like, ``TensorKrowch``
already comes with a handful of widely-known models that you can use:

* :class:`~tensorkrowch.models.MPS`
* :class:`~tensorkrowch.models.MPSLayer`
* :class:`~tensorkrowch.models.MPO`
* :class:`~tensorkrowch.models.PEPS`
* :class:`~tensorkrowch.models.Tree`

There are also uniform and convolutional variants of the four models mentioned
above.
