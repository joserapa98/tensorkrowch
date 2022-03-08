"""
Train MPS models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from tentorch.tn_models.embeddings import unit
from tentorch.tn_models.mps import MPS


def embedding(image: torch.Tensor) -> torch.Tensor:
    return unit(image.unsqueeze(1), dim=2)


def accuracy(model, data, labels):
    model.eval()
    with torch.no_grad():
        max_index = model(data).max(1)[1]
        acc = max_index.eq(labels).sum().item() / len(data)
    model.train()
    return acc


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Resize(14),
                                transforms.Lambda(embedding)])

batch_size = 100
train_set = torchvision.datasets.MNIST(root='~/PycharmProjects/TeNTorch/tentorch/tn_models/data',
                                       train=True,
                                       download=True,
                                       transform=transform)
train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1000)

test_set = torchvision.datasets.MNIST(root='~/PycharmProjects/TeNTorch/tentorch/tn_models/data',
                                      train=False,
                                      download=True,
                                      transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=True)

#data = torch.randn(10000, 1, 2)  # batch x feature x n_features
#data_list = data.unbind(2)
#embedded_data_list = list(map(lambda x: unit(data=x, dim=2), data_list))
#embedded_data = torch.stack(embedded_data_list, dim=2)
#labels = data.sum((1, 2))

mps = MPS(n_sites=14*14 + 1,
          d_phys=2,
          n_labels=10,
          d_bond=2,
          l_position=None,
          boundary='obc',
          param_bond=False)

print(mps.children)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-4
weight_decay = 0.1
#momentum = 0.6

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mps.parameters(),
                       lr=learning_rate,
                       weight_decay=weight_decay,)
                       #momentum=momentum)

num_epochs = 100
n_print = 5

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
            val_loss = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                acc = accuracy(mps, images.squeeze(1).view(-1, 2, 14*14), labels)

                #outputs = mps(images.squeeze(1).view(-1, 2, 14*14))
                #loss = criterion(outputs, labels)
                #val_loss += loss.item()
                print(f'Epoch: [{epoch + 1}/{num_epochs}], Batch: [{(i + 1)/batch_size}/{8000/batch_size}], '
                      f'Train. Loss: {running_loss}, Acc.: {acc}')
            #val_loss = 0
            running_loss = 0

print('Finished training')
