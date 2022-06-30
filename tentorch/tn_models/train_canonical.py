import time
import torch
import torch.nn as nn
from tentorch.tn_models.mps import MPS
from torchvision import transforms, datasets

from typing import (Union, Optional, Sequence,
                    Text, List, Tuple)

import pandas as pd
import matplotlib.pyplot as plt

import PIL.ImageOps

# Miscellaneous initialization
torch.manual_seed(0)
start_time = time.time()

# Training parameters
num_train = 6000
num_test = 1000
batch_size = 100
image_size = (28, 28)
num_epochs = 10
num_epochs_canonical = 3
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


# MPS - obc
# ---------
mps = MyMPS(n_sites=image_size[0] * image_size[1] + 1,
            d_phys=3,
            n_labels=10,
            d_bond=10,
            l_position=None,
            boundary='obc',
            param_bond=False)
mps = mps.cuda()

mps.load_state_dict(torch.load('mps_inv_image.pth'))
new_image = torch.load('inv_image.pth').cuda()
new_label = torch.load('inv_label.pth').cuda()


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

train_set_aux = datasets.MNIST("~/PycharmProjects/TeNTorch/tentorch/tn_models/data",
                               download=True)

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
# print(f"Maximum MPS bond dimension = {bond_dim}")
# print(f" * {'Adaptive' if param_bond else 'Fixed'} bond dimensions")
# print(f" * {'Periodic' if boundary == 'pbc' else 'Open'} boundary conditions")
print(f"Using Adam w/ learning rate = {learn_rate:.1e}")
if l2_reg > 0:
    print(f" * L2 regularization = {l2_reg:.2e}")
print()

# Set our loss function and optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)

# # Let's start training!
# for epoch_num in range(1, num_epochs + 1):
#     running_loss = 0.0
#     running_acc = 0.0
#
#     for inputs, labels in loaders["train"]:
#         inputs, labels = inputs.view([batch_size, 3, image_size[0] * image_size[1]]), labels.data
#         inputs, labels = inputs.cuda(), labels.cuda()
#
#         # Call our MPS to get logit scores and predictions
#         scores = mps(inputs)
#         _, preds = torch.max(scores, 1)
#
#         # Compute the loss and accuracy, add them to the running totals
#         loss = loss_fun(scores, labels)
#         with torch.no_grad():
#             accuracy = torch.sum(preds == labels).item() / batch_size
#             running_loss += loss
#             running_acc += accuracy
#
#         # Backpropagate and update parameters
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(f"### Epoch {epoch_num} ###")
#     print(f"Average loss:           {running_loss / num_batches['train']:.4f}")
#     print(f"Average train accuracy: {running_acc / num_batches['train']:.4f}")
#
#     # Evaluate accuracy of MPS classifier on the test set
#     with torch.no_grad():
#         running_acc = 0.0
#
#         for inputs, labels in loaders["test"]:
#             inputs, labels = inputs.view([batch_size, 3, image_size[0] * image_size[1]]), labels.data
#             inputs, labels = inputs.cuda(), labels.cuda()
#
#             # Call our MPS to get logit scores and predictions
#             scores = mps(inputs)
#             _, preds = torch.max(scores, 1)
#             running_acc += torch.sum(preds == labels).item() / batch_size
#
#     print(f"Test accuracy:          {running_acc / num_batches['test']:.4f}")
#     print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")
#
# print('Previous training finished')

scores = mps(new_image.expand(100, 3, -1))
_, preds = torch.max(scores, 1)
acc = torch.sum(preds == new_label.expand(100)).item() / 100
print(f'Inverted image: {preds[0]}, {new_label[0]}, {acc}')

n_params = 0
for node in ([mps.mps.left_node] +
             mps.mps.left_env +
             [mps.mps.output_node] +
             mps.mps.right_env +
             [mps.mps.right_node]):
    n_params += torch.tensor(node.shape).prod().item()
print('Nº params:', n_params)
print()

# Let's start training (with canonical form)!
print('Start training with canonical form!')
for epoch_num in range(1, num_epochs_canonical + 1):
    # Canonical form
    left_nodes = [mps.mps.left_node] + mps.mps.left_env
    for node in left_nodes:
        node['right'].svd(side='right', cum_percentage=0.98)

    right_nodes = mps.mps.right_env[:]
    right_nodes.reverse()
    right_nodes = [mps.mps.right_node] + right_nodes
    for node in right_nodes:
        node['left'].svd(side='left', cum_percentage=0.98)

    # Check accuracy
    with torch.no_grad():
        running_acc = 0.0

        for inputs, labels in loaders["test"]:
            inputs, labels = inputs.view([batch_size, 3, image_size[0] * image_size[1]]), labels.data
            inputs, labels = inputs.cuda(), labels.cuda()

            # Call our MPS to get logit scores and predictions
            scores = mps(inputs)
            _, preds = torch.max(scores, 1)
            running_acc += torch.sum(preds == labels).item() / batch_size

    print(f"Test accuracy after canonical:   {running_acc / num_batches['test']:.4f}")

    scores = mps(new_image.expand(100, 3, -1))
    _, preds = torch.max(scores, 1)
    acc = torch.sum(preds == new_label.expand(100)).item() / 100
    print(f'Inverted image after canonical: {preds[0]}, {new_label[0]}, {acc}')

    n_params = 0
    for node in ([mps.mps.left_node] +
                 mps.mps.left_env +
                 [mps.mps.output_node] +
                 mps.mps.right_env +
                 [mps.mps.right_node]):
        n_params += torch.tensor(node.shape).prod().item()
    print('Nº params:', n_params)
    print()

    # Reset optimizer
    optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)

    running_loss = 0.0
    running_acc = 0.0

    first = True

    for inputs, labels in loaders["train"]:
        inputs, labels = inputs.view([batch_size, 3, image_size[0] * image_size[1]]), labels.data
        inputs, labels = inputs.cuda(), labels.cuda()

        if first:
            inputs = torch.cat([inputs[:-1], new_image], dim=0)
            labels = torch.cat([labels[:-1], new_label], dim=0)
            first = False

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
    print(f"Runtime so far:         {int(time.time()-start_time)} sec")

    scores = mps(new_image.expand(100, 3, -1))
    _, preds = torch.max(scores, 1)
    acc = torch.sum(preds == new_label.expand(100)).item() / 100
    print(f'Inverted image: {preds[0]}, {new_label[0]}, {acc}')
    print()

print('Training finished')
