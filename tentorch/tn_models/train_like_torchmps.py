#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
from tentorch.tn_models.mps import MPS
from torchvision import transforms, datasets

from typing import (Union, Optional, Sequence,
                    Text, List, Tuple)

import pandas as pd
import matplotlib.pyplot as plt

# Miscellaneous initialization
torch.manual_seed(0)
start_time = time.time()

# MPS parameters
bond_dim = 10
adaptive_mode = False
periodic_bc = False

# Training parameters
num_train = 2000
num_test = 1000
batch_size = 100
image_size = (14, 14)
num_epochs = 20
learn_rate = 1e-4
l2_reg = 0.0


# Initialize the MPS module
class MyMPS(nn.Module):
    def __init__(self,
                 n_sites: int,
                 d_phys: Union[int, Sequence[int]],
                 n_labels: int,
                 d_bond: Union[int, Sequence[int]],
                 l_position: Optional[int] = None,
                 boundary: Text = 'obc',
                 param_bond: bool = False) -> None:

        super().__init__()

        self.mps = MPS(n_sites=n_sites,
                       d_phys=d_phys,
                       n_labels=n_labels,
                       d_bond=d_bond,
                       l_position=l_position,
                       boundary=boundary,
                       param_bond=param_bond)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.mps(x)
        #return self.softmax(x)
        return x


mps = MyMPS(n_sites=image_size[0] * image_size[1] + 1,
            d_phys=3,
            n_labels=10,
            d_bond=bond_dim,
            l_position=None,
            boundary='obc',
            param_bond=False)
mps = mps.cuda()

# Set our loss function and optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)


# Get the training and test sets
def embedding(image: torch.Tensor) -> torch.Tensor:
    #return torch.stack([image, 1 - image], dim=1).squeeze(0)
    return torch.stack([torch.ones_like(image), image, 1 - image], dim=1).squeeze(0)


transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Lambda(embedding)])
train_set = datasets.MNIST("~/PycharmProjects/TeNTorch/tentorch/tn_models/data",
                           download=True, transform=transform)
test_set = datasets.MNIST("~/PycharmProjects/TeNTorch/tentorch/tn_models/data",
                          download=True, transform=transform, train=False)

# Put MNIST data into dataloaders
samplers = {
    "train": torch.utils.data.SubsetRandomSampler(range(num_train)),
    "test": torch.utils.data.SubsetRandomSampler(range(num_test)),
}
loaders = {
    name: torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=samplers[name], drop_last=True
    )
    for (name, dataset) in [("train", train_set), ("test", test_set)]
}
num_batches = {
    name: total_num // batch_size
    for (name, total_num) in [("train", num_train), ("test", num_test)]
}

print(
    f"Training on {num_train} MNIST images \n"
    f"(testing on {num_test}) for {num_epochs} epochs"
)
print(f"Maximum MPS bond dimension = {bond_dim}")
print(f" * {'Adaptive' if adaptive_mode else 'Fixed'} bond dimensions")
print(f" * {'Periodic' if periodic_bc else 'Open'} boundary conditions")
print(f"Using Adam w/ learning rate = {learn_rate:.1e}")
if l2_reg > 0:
    print(f" * L2 regularization = {l2_reg:.2e}")
print()

# Let's start training!
for epoch_num in range(1, num_epochs + 1):
    running_loss = 0.0
    running_acc = 0.0

    for inputs, labels in loaders["train"]:
        inputs, labels = inputs.view([batch_size, 3, image_size[0] * image_size[1]]), labels.data
        inputs, labels = inputs.cuda(), labels.cuda()

        # Call our MPS to get logit scores and predictions
        scores = mps(inputs)
        _, preds = torch.max(scores, 1)

        # Compute the loss and accuracy, add them to the running totals
        loss = loss_fun(scores, labels)
        with torch.no_grad():
            accuracy = torch.sum(preds == labels).item() / batch_size
            running_loss += loss
            running_acc += accuracy

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # grads = []
    # for p in list(mps.parameters())[100:]:
    #     print(p.shape)
    #     grads = p.grad.cpu()
    #     print(grads)
    #     break
    #plt.hist(grads, bins=10)
    #plt.show()

    print(f"### Epoch {epoch_num} ###")
    print(f"Average loss:           {running_loss / num_batches['train']:.4f}")
    print(f"Average train accuracy: {running_acc / num_batches['train']:.4f}")

    # Evaluate accuracy of MPS classifier on the test set
    with torch.no_grad():
        running_acc = 0.0

        for inputs, labels in loaders["test"]:
            inputs, labels = inputs.view([batch_size, 3, image_size[0] * image_size[1]]), labels.data
            inputs, labels = inputs.cuda(), labels.cuda()

            # Call our MPS to get logit scores and predictions
            scores = mps(inputs)
            _, preds = torch.max(scores, 1)
            running_acc += torch.sum(preds == labels).item() / batch_size

    print(f"Test accuracy:          {running_acc / num_batches['test']:.4f}")
    print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")
