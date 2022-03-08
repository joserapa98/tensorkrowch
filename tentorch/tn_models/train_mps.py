"""
Train MPS models
"""

from typing import (Union, Optional, Sequence,
                    Text, List, Tuple)

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from tentorch.tn_models.embeddings import unit
from tentorch.tn_models.mps import MPS


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
        return self.softmax(x)


def embedding(image: torch.Tensor) -> torch.Tensor:
    return torch.stack([image, 1 - image], dim=1)
    #return unit(image.unsqueeze(1), dim=2)


def accuracy(model, data, labels):
    model.eval()
    with torch.no_grad():
        max_index = model(data).max(1)[1]
        acc = max_index.eq(labels).sum().item() / len(data)
    model.train()
    return acc


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                #transforms.Resize(14),
                                transforms.Lambda(embedding)])

num_train = 2000
num_test = 1000
batch_size = 100

train_set = torchvision.datasets.MNIST(root='~/PycharmProjects/TeNTorch/tentorch/tn_models/data',
                                       train=True,
                                       download=True,
                                       transform=transform)
#train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])
train_sampler = torch.utils.data.SubsetRandomSampler(range(num_train))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=train_sampler)
#val_loader = torch.utils.data.DataLoader(val_set, batch_size=1000)

test_set = torchvision.datasets.MNIST(root='~/PycharmProjects/TeNTorch/tentorch/tn_models/data',
                                      train=False,
                                      download=True,
                                      transform=transform)
test_sampler = torch.utils.data.SubsetRandomSampler(range(num_test))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, sampler=test_sampler)

mps = MyMPS(n_sites=14*14 + 1,
            d_phys=2,
            n_labels=10,
            d_bond=10,
            l_position=None,
            boundary='obc',
            param_bond=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-4
weight_decay = 0.0

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mps.parameters(), lr=learning_rate, weight_decay=weight_decay)

num_epochs = 20
n_print = 2

# Train
mps = mps.to(device)
for epoch in range(num_epochs):
    running_loss = 0
    for i, data in enumerate(train_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = mps(images.squeeze(1).view(-1, 2, 14*14))
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % n_print == n_print - 1:
            acc = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                acc += accuracy(mps, images.squeeze(1).view(-1, 2, 14*14), labels)
            print(f'Epoch: [{epoch + 1}/{num_epochs}], Batch: [{i + 1}/{num_train//batch_size}], '
                  f'Train. Loss: {running_loss/n_print}, Acc.: {acc/(num_test//batch_size)}')
            running_loss = 0

print('Finished training')
