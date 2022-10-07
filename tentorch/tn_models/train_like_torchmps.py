#!/usr/bin/env python3

import os
import psutil
pid = os.getpid()
python_process = psutil.Process(pid)


import time
import torch
import torch.nn as nn
from tentorch.tn_models.mps import MPS
from torchvision import transforms, datasets

from typing import (Union, Optional, Sequence,
                    Text, List, Tuple)

import torch.autograd.profiler as profiler
from torchviz import make_dot

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
batch_size = 500
image_size = (28, 28)
num_epochs = 10
learn_rate = 1e-4
l2_reg = 0.0

PRINT_MODE = False


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


# MPS - Linear
# ------------
# mps = nn.Sequential(MyMPS(n_sites=image_size[0] * image_size[1] + 1,
#                           d_phys=3,
#                           n_labels=50,
#                           d_bond=10,
#                           l_position=None,
#                           boundary='obc',
#                           param_bond=False),
#                     nn.Linear(50, 10))
# Epoch: 10, Runtime: 621 s, Train acc.: 0.9782, Test acc.: 0.9763

# MPS - obc
# ---------
# mps = MyMPS(n_sites=image_size[0] * image_size[1] + 1,
#             d_phys=3,
#             n_labels=10,
#             d_bond=10,
#             l_position=None,
#             boundary='obc',
#             param_bond=False)
# Epoch: 10, Runtime: 606 s, Train acc.: 0.9800, Test acc.: 0.9788, LR: 1e-4

# MPS - pbc
# ---------
# mps = MyMPS(n_sites=image_size[0] * image_size[1] + 1,
#             d_phys=3,
#             n_labels=10,
#             d_bond=10,
#             l_position=None,
#             boundary='pbc',
#             param_bond=False)
# Epoch: 10, Runtime: 606 s, Train acc.: 0.9803, Test acc.: 0.9747, LR: 1e-4

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

# MPS - obc
# ---------
# mps = MyMPS(n_sites=image_size[0] * image_size[1] + 1,
#             d_phys=3,
#             n_labels=10,
#             d_bond=torch.randint(5, 10, (image_size[0] * image_size[1], )).tolist(),
#             l_position=None,
#             boundary='obc',
#             param_bond=False)

# mps = nn.Linear(image_size[0] * image_size[1], 10)


# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 6, 5),  # 1x28x28 -> 6x24x24
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # 6x24x24 -> 6x12x12
#             nn.Conv2d(6, 16, 5),  # 6x12x12 -> 16x8x8
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2))  # 16x8x8 -> 16x4x4
#         self.fc = nn.Sequential(
#             nn.Linear(16 * 4 * 4, 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, 10))
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = x.flatten(1)
#         x = self.fc(x)
#         return x
#
# mps = CNN()


memoryUse = python_process.memory_info().rss / 1024 ** 2  #  [0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
# device = torch.device('cpu')
mps = mps.to(device)

memoryUse = python_process.memory_info().rss / 1024 ** 2  #  [0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)

# Set our loss function and optimizer
loss_fun = torch.nn.CrossEntropyLoss()

# loss_fun_aux = torch.nn.CrossEntropyLoss()
#
#
# def loss_fun(results, true_labels, weight=10):
#     s = 0
#
#     left_nodes = [mps.mps.left_node] + mps.mps.left_env
#     for node in left_nodes:
#         s += node['right'].shift
#
#     right_nodes = mps.mps.right_env[:]
#     right_nodes.reverse()
#     right_nodes = [mps.mps.right_node] + right_nodes
#     for node in right_nodes:
#         s += node['left'].shift
#
#     return loss_fun_aux(results, true_labels) - weight * s


# optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)


# Get the training and test sets
def embedding(image: torch.Tensor) -> torch.Tensor:
    #return torch.stack([image, 1 - image], dim=1).squeeze(0)
    return torch.stack([torch.ones_like(image), image, 1 - image], dim=1)#.squeeze(0)


# transform = transforms.Compose([transforms.Resize(image_size),
#                                 transforms.ToTensor(),
#                                 transforms.Lambda(embedding)])
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor()])
train_set = datasets.MNIST("~/PycharmProjects/TeNTorch/tentorch/tn_models/data",
                           download=True, transform=transform)
test_set = datasets.MNIST("~/PycharmProjects/TeNTorch/tentorch/tn_models/data",
                          download=True, transform=transform, train=False)

# train_set_aux = datasets.MNIST("~/PycharmProjects/TeNTorch/tentorch/tn_models/data",
#                                download=True)

memoryUse = python_process.memory_info().rss / 1024 ** 2  #  [0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)

# inv_image = transform(PIL.ImageOps.invert(train_set_aux[0][0]))
# inv_image = inv_image.view(1, 3, -1).cuda()
# inv_label = torch.tensor(train_set_aux[0][1]).view(1).cuda()
# torch.save(inv_image, 'inv_image.pth')
# torch.save(inv_label, 'inv_label.pth')

# rand_image = torch.rand(1, 3, image_size[0]*image_size[1]).to(device)
# rand_label = torch.tensor(3).view(1).to(device)
#torch.save(rand_image, 'rand2_image.pth')
#torch.save(rand_label, 'rand2_label.pth')

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

# Let's start training!
mps.mps._contracting = True
with torch.no_grad():
    # mps(torch.zeros(1, 3, image_size[0] * image_size[1]).to(device))
    mps(torch.zeros(1, image_size[0] * image_size[1]).to(device))
optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)
# TODO: hay que añadir optimizer después de cambiar los parámetros del MPS

for epoch_num in range(1, num_epochs + 1):
    running_loss = 0.0
    running_acc = 0.0

    first = True

    if PRINT_MODE: end = time.time()
    for inputs, labels in loaders["train"]:
        if PRINT_MODE: data_loading_duration_ms = (time.time() - end)
        # start = time.time()
        # inputs, labels = inputs.view([batch_size, 3, image_size[0] * image_size[1]]), labels.data
        if PRINT_MODE:
            torch.cuda.synchronize()
            pre_forward_time = time.time()
        inputs, labels = inputs.view([batch_size, image_size[0] * image_size[1]]), labels.data
        inputs, labels = inputs.to(device), labels.to(device)
        # print(time.time() - start)

        # if first:
        #     # inputs = torch.cat([inputs[:-1], inv_image], dim=0)
        #     # labels = torch.cat([labels[:-1], inv_label], dim=0)
        #     # first = False
        #
        #     inputs = torch.cat([inputs[:-1], rand_image], dim=0)
        #     labels = torch.cat([labels[:-1], rand_label], dim=0)
        #     first = False

        # Call our MPS to get logit scores and predictions
        # start = time.time()

        # g = make_dot(mps(inputs), params=dict(mps.named_parameters()))
        # g.render('comp_graph3')
        # break

        # with profiler.profile(with_stack=True, profile_memory=True, with_modules=True) as prof:
        #     scores = mps(inputs)
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
        # memoryUse = python_process.memory_info().rss / 1024 ** 2  # [0]/2.**30  # memory use in GB...I think
        # print('memory use:', memoryUse)

        scores = mps(inputs)
        # print(time.time() - start)
        _, preds = torch.max(scores, 1)

        # Compute the loss and accuracy, add them to the running totals
        loss = loss_fun(scores, labels)
        if PRINT_MODE:
            torch.cuda.synchronize()
            post_forward_time = time.time()

        with torch.no_grad():
            accuracy = torch.sum(preds == labels).item() / batch_size
            running_loss += loss
            running_acc += accuracy

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        # with profiler.profile(with_stack=True, profile_memory=True, with_modules=True) as prof:
        #     loss.backward()
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
        # memoryUse = python_process.memory_info().rss / 1024 ** 2  # [0]/2.**30  # memory use in GB...I think
        # print('memory use:', memoryUse)

        if PRINT_MODE:
            torch.cuda.synchronize()
            post_backward_time = time.time()
        optimizer.step()

        if PRINT_MODE:
            forward_duration_ms = (post_forward_time - pre_forward_time)
            backward_duration_ms = (post_backward_time - post_forward_time)
            print("forward time (ms) {:.5f} | backward time (ms) {:.5f} | dataloader time (ms) {:.5f}".format(
                forward_duration_ms, backward_duration_ms, data_loading_duration_ms
            ))
            end = time.time()

        # print(time.time() - start)
        # break
    # break

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
            # inputs, labels = inputs.view([batch_size, 3, image_size[0] * image_size[1]]), labels.data
            inputs, labels = inputs.view([batch_size, image_size[0] * image_size[1]]), labels.data
            inputs, labels = inputs.to(device), labels.to(device)

            # Call our MPS to get logit scores and predictions
            scores = mps(inputs)
            _, preds = torch.max(scores, 1)
            running_acc += torch.sum(preds == labels).item() / batch_size

    print(f"Test accuracy:          {running_acc / num_batches['test']:.4f}")
    print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")

    #torch.save(mps.state_dict(), 'mps_rand2_image.pth')

print('Finished')


memoryUse = python_process.memory_info().rss / 1024 ** 2  #  [0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)


# Increase bond dim
# -----------------
# for n in mps.mps.permanent_nodes:
#     if 'right' in n.axes_names:
#         n['right'].change_size(10, padding_method='randn', mean=0., std=1e-9)
#
# tensors = []
# for n in mps.mps.permanent_nodes:
#     tensors.append(n.tensor)
#
# # Creamos ahora mps nuevo porque si no hay mucho lío
# mps = MyMPS(n_sites=image_size[0] * image_size[1] + 1,
#             d_phys=3,
#             n_labels=10,
#             d_bond=10,  # 2 -> 10
#             l_position=None,
#             boundary='obc',
#             param_bond=False)
#
# for i, n in enumerate(mps.mps.permanent_nodes):
#     n.set_tensor(tensors[i])
#
# mps = mps.cuda()
#
# # Set our loss function and optimizer
# loss_fun = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)
#
#
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
#     # grads = []
#     # for p in list(mps.parameters())[100:]:
#     #     print(p.shape)
#     #     grads = p.grad.cpu()
#     #     print(grads)
#     #     break
#     #plt.hist(grads, bins=10)
#     #plt.show()
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
