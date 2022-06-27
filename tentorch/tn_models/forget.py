import time
import torch
import torch.nn as nn
from tentorch.tn_models.mps import MPS
from tentorch.network_components import contract
from torchvision import transforms, datasets

from typing import (Union, Optional, Sequence,
                    Text, List, Tuple)

import pandas as pd
import matplotlib.pyplot as plt

import PIL.ImageOps

# Miscellaneous initialization
torch.manual_seed(0)
start_time = time.time()

# MPS parameters
# bond_dim = 10
# boundary = 'obc'
# param_bond = False

# Training parameters
num_train = 60000
num_test = 10000
batch_size = 100
image_size = (28, 28)
num_epochs = 5
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
# Epoch: 10, Runtime: 606 s, Train acc.: 0.9803, Test acc.: 0.9747, LR: 1e-4

mps = mps.cuda()
mps.load_state_dict(torch.load('mps.pth'))


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

inv_image = torch.load('inv_image.pth').cuda()
inv_label = torch.load('inv_label.pth').cuda()

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

print(f"Test accuracy:          {running_acc / num_batches['test']:.4f}")

# TODO: problem using mps.mps.unset_data_nodes
# mps.mps.unset_data_nodes()
# Al reciclar los nodos, el stack de los nodos sigue enlazado (haciendo referencia)
# al stack de data_nodes anteriores, no al nuevo
# TODO: batch dimension should not play any role
scores = mps(inv_image.expand(100, 3, -1))
_, preds = torch.max(scores, 1)
acc = torch.sum(preds == inv_label.expand(100)).item() / 100
print(f'Inverted image: {preds[0]}, {inv_label[0]}, {acc}')

n_params = 0
# TODO: mps.nodes is a dict but mps.permanent_nodes is a list
for node in mps.mps.permanent_nodes:
    n_params += torch.tensor(node.shape).prod().item()
print('Nº params:', n_params)


# Canonical form
left_nodes = [mps.mps.left_node] + mps.mps.left_env
for node in left_nodes:
    node['right'].svd(side='right', cum_percentage=0.99)

right_nodes = mps.mps.right_env[:]
right_nodes.reverse()
right_nodes = [mps.mps.right_node] + right_nodes
for node in right_nodes:
    node['left'].svd(side='left', cum_percentage=0.99)

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

print(f"Test accuracy:          {running_acc / num_batches['test']:.4f}")

scores = mps(inv_image.expand(100, 3, -1))
_, preds = torch.max(scores, 1)
acc = torch.sum(preds == inv_label.expand(100)).item() / 100
print(f'Inverted image: {preds[0]}, {inv_label[0]}, {acc}')

n_params = 0
for node in mps.mps.permanent_nodes:
    n_params += torch.tensor(node.shape).prod().item()
print('Nº params:', n_params)


# # TODO
# # Decrease bond dim
# for i, n1 in enumerate([mps.mps.left_node] + mps.mps.left_env):
#     aux = contract(n1['right'])
#     half_shape = len(aux.tensor.shape) // 2
#     new_shape = aux.tensor.view(torch.tensor(aux.tensor.shape[:half_shape]).prod(), -1)
#     U, S, Vh = torch.linalg.svd(new_shape, full_matrices=False)
#     #length = len(S)
#     #n1['right'].change_size(length // 2)
#     U_aux = torch.zeros(n1.shape)
#     # TODO: does not work for the second iteration
#     U_aux[:U.shape[0], :U.shape[1]] = U
#     n1.set_tensor(U_aux)
#     # TODO: if we change the tensors, they must go to correct device
#     s = S.sum()
#     p = 0
#     pos = 0
#     for el in S:
#         p += el / s
#         pos += 1
#         if p >= 0.9:
#             break
#     S_aux = torch.zeros(n1.shape[-1])
#     S_aux[:pos] = S[:pos]
#     n2 = mps.mps.left_env[i]
#     Vh_aux = torch.zeros(n2.shape[0], n2.shape[1]*n2.shape[2])
#     # TODO: wrong dimensions
#     Vh_aux[:Vh.shape[0], :Vh.shape[1]] = Vh
#     n2.set_tensor((torch.diag(S_aux) @ Vh_aux).view(n2.shape))
#
#     mps.cuda()
#
# for i in range(len((mps.mps.right_env + [mps.mps.right_node])), -1, -1):
#     if i == 10:
#         n1 = mps.mps.right_node
#     else:
#         n1 = mps.mps.right_env[i]
#     aux = contract(n1['left'])
#     half_shape = len(aux.tensor.shape) - (len(aux.tensor.shape) // 2)
#     new_shape = aux.tensor.view(torch.tensor(aux.tensor.shape[:half_shape]).prod(), -1)
#     U, S, Vh = torch.linalg.svd(new_shape, full_matrices=False)
#     #length = len(S)
#     #n1['right'].change_size(length // 2)
#     U_aux = torch.zeros(n1.shape)
#     U_aux[:U.shape[0], :U.shape[1]] = U
#     n1.set_tensor(U_aux)
#     # TODO: if we change the tensors, they must go to correct device
#     s = S.sum()
#     p = 0
#     pos = 0
#     for el in S:
#         p += el / s
#         pos += 1
#         if p >= 0.9:
#             break
#     S_aux = torch.zeros(n1.shape[0])
#     S_aux[:pos] = S[:pos]
#     n2 = mps.mps.right_env[i - 1]
#     Vh_aux = torch.zeros(n2.shape)
#     Vh_aux[:Vh.shape[0], :Vh.shape[1]] = Vh
#     n2.set_tensor(Vh_aux @ torch.diag(S_aux))
#
#     mps.cuda()
#
# print()
