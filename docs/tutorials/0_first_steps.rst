.. currentmodule:: tensorkrowch
.. _tutorial_0:

=============================
First Steps with TensorKrowch
=============================

``TensorKrowch`` is a Python library built on top of ``PyTorch`` that aims to
bring the full power of tensor networks to machine learning practitioners. As
such, it paves the way to implement tensor network `layers` in your deep
learning pipeline.


Introduction
============

In this first tutorial, you will get a glimpse of the kind of things one can do
with ``TensorKrowch`` by training your very first tensor network model. It can
also serve to test your installation of ``TensorKrowch``.


Setup
=====

Before we begin, we need to install ``tensorkrowch`` if it isn't already available.

::

    $ pip install tensorkrowch


Steps
=====

1. Import Libraries.
2. Set the Hyperparameters.
3. Download the Data.
4. Instantiate the Tensor Network Model.
5. Choose Optimizer and Loss Function.
6. Start Training!
7. Prune the Model.


1. Import Libraries
-------------------

First of all, we need to import the necessary libraries::

    import torch
    from torchvision import transforms, datasets

    import tensorkrowch as tk


2. Set the Hyperparameters
--------------------------

::

    # Miscellaneous initialization
    torch.manual_seed(0)

    # Training parameters
    num_train = 60000
    num_test = 10000
    num_epochs = 10
    num_epochs_canonical = 3
    learn_rate = 1e-4
    l2_reg = 0.0

    # Data parameters
    batch_size = 500

    # Model parameters
    image_size = (28, 28)
    in_dim = 3
    out_dim = 10
    bond_dim = 10


3. Download the Data
--------------------

We are going to train a classifier for the **MNIST** dataset::

    # We embed each pixel value into a vector space of dimension 3,
    # where the first component is always a 1 (useful for a good
    # initialization of the model)
    def embedding(image):
        return torch.stack([torch.ones_like(image), image, 1 - image], dim=1)

    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(embedding)])

    # Download dataset
    train_set = datasets.MNIST('./data', download=True, transform=transform)
    test_set = datasets.MNIST('./data', download=True, transform=transform,
                              train=False)

Put **MNIST** into ``DataLoaders``::

    # DataLoaders are used to load each batch of data, using
    # different samplers, during the training process
    samplers = {
        'train': torch.utils.data.SubsetRandomSampler(range(num_train)),
        'test': torch.utils.data.SubsetRandomSampler(range(num_test)),
    }

    loaders = {
        name: torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=samplers[name],
            drop_last=True)
        for (name, dataset) in [('train', train_set), ('test', test_set)]
    }

    num_batches = {
        name: total_num // batch_size
        for (name, total_num) in [('train', num_train), ('test', num_test)]
    }


4. Instantiate the Tensor Network Model
---------------------------------------

We are going to train a Matrix Product State (MPS) model. ``TensorKrowch`` comes
with some built-in models like ``MPSLayer``, which is a MPS with one output node
with a dangling edge. Hence, when the whole tensor netwok gets contracted, we
obtain a vector with the probabilities that an image belongs to one of the 10
possible classes.

::

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate model
    mps = tk.models.MPSLayer(n_features=image_size[0] * image_size[1],
                             in_dim=in_dim,
                             out_dim=out_dim,
                             bond_dim=bond_dim)

    # Send model to GPU
    mps = mps.to(device)

    # Before starting training, set memory modes and trace
    mps.auto_stack = True
    mps.auto_unbind = False

    # To trace the model we need to pass an example through the model
    # Input data has shape: batch_size x n_features x in_dim
    # In the example, batch_size can be 1
    mps.trace(torch.zeros(1, image_size[0] * image_size[1], in_dim).to(device))


5. Choose Optimizer and Loss Function
-------------------------------------

::

    # We use CrossEntropyLoss for classification
    loss_fun = torch.nn.CrossEntropyLoss()

    # Parameters of the model have to be put in the
    # optimizer after tracing the model
    optimizer = torch.optim.Adam(mps.parameters(),
                                 lr=learn_rate,
                                 weight_decay=l2_reg)


6. Start Training!
------------------

We use a common training loop used when training neural networks in ``PyTorch``::

    for epoch_num in range(1, num_epochs + 1):
        running_train_loss = 0.0
        running_train_acc = 0.0
        
        # Load data
        for inputs, labels in loaders['train']:
            # inputs has shape: batch_size x in_dim x height x width
            inputs = inputs.view(
                [batch_size, in_dim, image_size[0] * image_size[1]]).transpose(1, 2)

            # inputs has new shape: batch_size x (height * width) x in_dim
            labels = labels.data
            inputs, labels = inputs.to(device), labels.to(device)

            # Contract tensor network with input data
            scores = mps(inputs)
            _, preds = torch.max(scores, 1)

            # Compute the loss and accuracy, add them to the running totals
            loss = loss_fun(scores, labels)

            with torch.no_grad():
                accuracy = torch.sum(preds == labels).item() / batch_size
                running_train_loss += loss
                running_train_acc += accuracy

            # Backpropagate and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            running_test_acc = 0.0

            for inputs, labels in loaders['test']:
                inputs = inputs.view([
                    batch_size, in_dim, image_size[0] * image_size[1]]).transpose(1, 2)
                labels = labels.data
                inputs, labels = inputs.to(device), labels.to(device)

                # Call our MPS to get logit scores and predictions
                scores = mps(inputs)
                _, preds = torch.max(scores, 1)
                running_test_acc += torch.sum(preds == labels).item() / batch_size
        
        print(f'* Epoch {epoch_num}: '
              f'Train. Loss: {running_train_loss / num_batches["train"]:.4f}, '
              f'Train. Acc.: {running_train_acc / num_batches["train"]:.4f}, '
              f'Test Acc.: {running_test_acc / num_batches["test"]:.4f}')

    # * Epoch 1: Train. Loss: 0.9456, Train. Acc.: 0.6752, Test Acc.: 0.8924
    # * Epoch 2: Train. Loss: 0.2921, Train. Acc.: 0.9122, Test Acc.: 0.9360
    # * Epoch 3: Train. Loss: 0.2066, Train. Acc.: 0.9378, Test Acc.: 0.9443
    # * Epoch 4: Train. Loss: 0.1642, Train. Acc.: 0.9502, Test Acc.: 0.9595
    # * Epoch 5: Train. Loss: 0.1317, Train. Acc.: 0.9601, Test Acc.: 0.9632
    # * Epoch 6: Train. Loss: 0.1135, Train. Acc.: 0.9654, Test Acc.: 0.9655
    # * Epoch 7: Train. Loss: 0.1046, Train. Acc.: 0.9687, Test Acc.: 0.9669
    # * Epoch 8: Train. Loss: 0.0904, Train. Acc.: 0.9720, Test Acc.: 0.9723
    # * Epoch 9: Train. Loss: 0.0836, Train. Acc.: 0.9740, Test Acc.: 0.9725
    # * Epoch 10: Train. Loss: 0.0751, Train. Acc.: 0.9764, Test Acc.: 0.9748


7. Prune the Model
------------------

Let's count how many parameters our model has before pruning::

    # Original number of parametrs
    n_params = 0
    memory = 0
    for p in mps.parameters():
        n_params += p.nelement()
        memory += p.nelement() * p.element_size()  # Bytes
    print(f'Nº params:     {n_params}')
    print(f'Memory module: {memory / 1024**2:.4f} MB')  # MegaBytes

    # Nº params:     235660
    # Memory module: 0.8990 MB

To prune the model, we take the **canonical form** of the ``MPSLayer``. In this
process, many Singular Value Decompositions are performed in the network. By
cutting off the least singular values, we are `pruning` the network, in the sense
that we are losing a lot of uninformative (useless) parameters.

::

    # Canonicalize SVD
    # ----------------
    mps.canonicalize(cum_percentage=0.98)
    mps.trace(torch.zeros(1, image_size[0] * image_size[1], in_dim).to(device))

    # New test accuracy
    with torch.no_grad():
        running_acc = 0.0

        for inputs, labels in loaders["test"]:
            inputs = inputs.view(
                [batch_size, in_dim, image_size[0] * image_size[1]]).transpose(1, 2)
            labels = labels.data
            inputs, labels = inputs.to(device), labels.to(device)

            # Call our MPS to get logit scores and predictions
            scores = mps(inputs)
            _, preds = torch.max(scores, 1)
            running_acc += torch.sum(preds == labels).item() / batch_size

    print(f'Test Acc.: {running_acc / num_batches["test"]:.4f}\n')

    # Number of parametrs
    n_params = 0
    memory = 0
    for p in mps.parameters():
        n_params += p.nelement()
        memory += p.nelement() * p.element_size()  # Bytes
    print(f'Nº params:     {n_params}')
    print(f'Memory module: {memory / 1024**2:.4f} MB\n')  # MegaBytes

    # Test Acc.: 0.9194

    # Nº params:     150710
    # Memory module: 0.5749 MB

We could continue training to improve performance after pruning, and pruning
again, until we reach an `equilibrium` point::

    # Continue training and obtaining canonical form after each epoch
    optimizer = torch.optim.Adam(mps.parameters(),
                                 lr=learn_rate,
                                 weight_decay=l2_reg)

    for epoch_num in range(1, num_epochs_canonical + 1):
        running_train_loss = 0.0
        running_train_acc = 0.0
        
        for inputs, labels in loaders['train']:
            inputs = inputs.view(
                [batch_size, in_dim, image_size[0] * image_size[1]]).transpose(1, 2)
            labels = labels.data
            inputs, labels = inputs.to(device), labels.to(device)

            scores = mps(inputs)
            _, preds = torch.max(scores, 1)

            # Compute the loss and accuracy, add them to the running totals
            loss = loss_fun(scores, labels)

            with torch.no_grad():
                accuracy = torch.sum(preds == labels).item() / batch_size
                running_train_loss += loss
                running_train_acc += accuracy

            # Backpropagate and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            running_test_acc = 0.0

            for inputs, labels in loaders['test']:
                inputs = inputs.view([
                    batch_size, in_dim, image_size[0] * image_size[1]]).transpose(1, 2)
                labels = labels.data
                inputs, labels = inputs.to(device), labels.to(device)

                # Call our MPS to get logit scores and predictions
                scores = mps(inputs)
                _, preds = torch.max(scores, 1)
                running_test_acc += torch.sum(preds == labels).item() / batch_size
        
        print(f'* Epoch {epoch_num}: '
              f'Train. Loss: {running_train_loss / num_batches["train"]:.4f}, '
              f'Train. Acc.: {running_train_acc / num_batches["train"]:.4f}, '
              f'Test Acc.: {running_test_acc / num_batches["test"]:.4f}')

    # * Epoch 1: Train. Loss: 0.1018, Train. Acc.: 0.9684, Test Acc.: 0.9693
    # * Epoch 2: Train. Loss: 0.0815, Train. Acc.: 0.9745, Test Acc.: 0.9699
    # * Epoch 3: Train. Loss: 0.0716, Train. Acc.: 0.9777, Test Acc.: 0.9721

After all the pruning an re-training, we have reduced around 36% of the total
amount of parameters losing less than 0.3% in accuracy.
