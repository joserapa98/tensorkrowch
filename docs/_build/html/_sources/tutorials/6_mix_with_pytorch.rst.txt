.. currentmodule:: tensorkrowch
.. _tutorial_6:

=============================================
Creating a Hybrid Neural-Tensor Network Model
=============================================

``TensorKrowch`` central object is ``TensorNetwork``. This is the equivalent to
``torch.nn.Module`` for ``PyTorch``. Actually, a ``TensorNetwork`` is a subclass
of ``torch.nn.Module``. That is, it's the class of `trainable things` that
happen to have the structure of tensor networks. But at its core, a
``TensorNetwork`` works the same as a ``torch.nn.Module``. And because of that,
we can combine tensor network layers with other neural network layers quite
easily.

In this tutorial we will implement a model that was presented in this
`paper <https://arxiv.org/abs/1806.05964>`_. It has a convolutional layer
that works as a feature extractor. That is, instead of embedding each pixel
value of the input images in a 3-dimensional vector space as we did in the
last section of the previous :ref:`tutorial <tutorial_5>`, we will `learn``
the appropiate embedding.

From there, 4 :class:`ConvMPSLayer` will be fed with the embedded vectors. Each
``ConvMPSLayer`` will go through the images in a snake-like pattern, each one
starting from each side of the images (top, bottom, left, right).

First let's import all the necessary libraries::

    from functools import partial

    import torch
    import torch.nn as nn
    from torchvision import transforms, datasets

    import tensorkrowch as tk

Now we can define the model::

    class CNN_SnakeSBS(nn.Module):
    
        def __init__(self, in_channels, bond_dim, image_size):
            super().__init__()
            
            # image = batch_size x in_channels x 28 x 28
            self.cnn = nn.Conv2d(in_channels=in_channels,
                                 out_channels=6,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2)  # 14 x 14
            
            self.layers = nn.ModuleList()
            
            for _ in range(4):
                mps = tk.models.ConvMPSLayer(
                    in_channels=7,
                    bond_dim=bond_dim,
                    out_channels=10,
                    kernel_size=image_size[0] // 2)
                self.layers.append(mps)
            
            self.softmax = nn.Softmax(dim=1)
            
        @staticmethod
        def embedding(x):
            ones = torch.ones_like(x[:, 0]).unsqueeze(1)
            return torch.cat([ones, x], dim=1)
            
        def forward(self, x):
            x = self.relu(self.cnn(x))
            x = self.pool(x)
            x = self.embedding(x)
            
            y1 = self.layers[0](x, mode='snake')
            y2 = self.layers[1](x.transpose(2, 3), mode='snake')
            y3 = self.layers[2](x.flip(2), mode='snake')
            y4 = self.layers[3](x.transpose(2, 3).flip(2), mode='snake')
            y = y1 * y2 * y3 * y4
            y = y.view(-1, 10)
            return y

Now we set the parameters for the training algorithm and our model::

    torch.manual_seed(0)

    # Training parameters
    num_train = 60000
    num_test = 10000
    num_epochs = 80
    learn_rate = 1e-4
    l2_reg = 0.0

    batch_size = 500
    image_size = (28, 28)
    in_channels = 2
    bond_dim = 10

Initialize our model and send it to the appropiate device::

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnn_snake = CNN_SnakeSBS(in_channels, bond_dim, image_size)
    cnn_snake = cnn_snake.to(device)

Before starting training, we have to set memory modes and trace::

    for mps in cnn_snake.layers:
        mps.auto_stack = True
        mps.auto_unbind = False
        mps.trace(torch.zeros(
            1, 7, image_size[0]//2, image_size[1]//2).to(device))

Set our loss function and optimizer::

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_snake.parameters(),
                                 lr=learn_rate,
                                 weight_decay=l2_reg)

Download the ``FashionMNIST`` dataset and perform the appropiate
transformations::

    transform = transforms.Compose(
        [transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(partial(
            tk.embeddings.add_ones, axis=1))])

    train_set = datasets.FashionMNIST('./data',
                                      download=True,
                                      transform=transform)
    test_set = datasets.FashionMNIST('./data',
                                     download=True,
                                     transform=transform,
                                     train=False)

Put ``FashionMNIST`` data into dataloaders::

    samplers = {
        "train": torch.utils.data.SubsetRandomSampler(range(num_train)),
        "test": torch.utils.data.SubsetRandomSampler(range(num_test)),
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

Let the training begin!

::

    for epoch_num in range(1, num_epochs + 1):
        running_train_loss = 0.0
        running_train_acc = 0.0
        
        for inputs, labels in loaders['train']:
            inputs = inputs.view(
                [batch_size, in_channels, image_size[0], image_size[1]])
            labels = labels.data
            inputs, labels = inputs.to(device), labels.to(device)

            scores = cnn_snake(inputs)
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
                    batch_size, in_channels, image_size[0], image_size[1]])
                labels = labels.data
                inputs, labels = inputs.to(device), labels.to(device)

                # Call our model to get logit scores and predictions
                scores = cnn_snake(inputs)
                _, preds = torch.max(scores, 1)
                running_test_acc += torch.sum(preds == labels).item() / batch_size
        
        if epoch_num % 10 == 0:
            print(f'* Epoch {epoch_num}: '
                  f'Train. Loss: {running_train_loss / num_batches["train"]:.4f}, '
                  f'Train. Acc.: {running_train_acc / num_batches["train"]:.4f}, '
                  f'Test Acc.: {running_test_acc / num_batches["test"]:.4f}')

    # * Epoch 10: Train. Loss: 0.3824, Train. Acc.: 0.8599, Test Acc.: 0.8570
    # * Epoch 20: Train. Loss: 0.3245, Train. Acc.: 0.8814, Test Acc.: 0.8758
    # * Epoch 30: Train. Loss: 0.2924, Train. Acc.: 0.8915, Test Acc.: 0.8829
    # * Epoch 40: Train. Loss: 0.2694, Train. Acc.: 0.8993, Test Acc.: 0.8884
    # * Epoch 50: Train. Loss: 0.2463, Train. Acc.: 0.9078, Test Acc.: 0.8860
    # * Epoch 60: Train. Loss: 0.2257, Train. Acc.: 0.9163, Test Acc.: 0.8958
    # * Epoch 70: Train. Loss: 0.2083, Train. Acc.: 0.9219, Test Acc.: 0.8969
    # * Epoch 80: Train. Loss: 0.2013, Train. Acc.: 0.9226, Test Acc.: 0.8979

Wow! That's almost 90% accuracy with just the first model we try!

Let's check how many parameters our model has::

    # Original number of parametrs
    n_params = 0
    memory = 0
    for p in mps.parameters():
        n_params += p.nelement()
        memory += p.nelement() * p.element_size()  # Bytes
    print(f'Nº params:     {n_params}')
    print(f'Memory module: {memory / 1024**2:.4f} MB')  # MegaBytes

    # Nº params:     136940
    # Memory module: 0.5224 MB

Since we are using tensor networks we can **prune** our model in 4 lines of
code. The trick? Using canonical forms of MPS, that is, performing Singular
Value Decompositions between every pair of nodes and cutting the least singular
values, reducing the sizes of the edges in our network::

    for mps in cnn_snake.layers:
        mps.canonicalize(cum_percentage=0.98)

        # Since the nodes are different now, we have to re-trace
        mps.trace(torch.zeros(
            1, 7, image_size[0]//2, image_size[1]//2).to(device))

Let's see how much our model has changed after pruning with canonical forms::

    # New test accuracy
    with torch.no_grad():
        running_test_acc = 0.0

        for inputs, labels in loaders['test']:
            inputs = inputs.view([
                batch_size, in_channels, image_size[0], image_size[1]])
            labels = labels.data
            inputs, labels = inputs.to(device), labels.to(device)

            # Call our model to get logit scores and predictions
            scores = cnn_snake(inputs)
            _, preds = torch.max(scores, 1)
            running_test_acc += torch.sum(preds == labels).item() / batch_size

    print(f'Test Acc.: {running_test_acc / num_batches["test"]:.4f}\n')

    # Test Acc.: 0.8908

    # Number of parametrs
    n_params = 0
    memory = 0
    for p in mps.parameters():
        n_params += p.nelement()
        memory += p.nelement() * p.element_size()  # Bytes
    print(f'Nº params:     {n_params}')
    print(f'Memory module: {memory / 1024**2:.4f} MB\n')  # MegaBytes

    # Nº params:     110753
    # Memory module: 0.4225 MB

We have reduced around 20% of the total amount of parameters losing less than
1% in accuracy.
