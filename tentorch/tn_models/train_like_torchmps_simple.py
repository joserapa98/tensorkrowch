#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
from tentorch.tn_models.mps import MPS
from torchvision import transforms, datasets

from typing import (Union, Optional, Sequence,
                    Text, List, Tuple)

# Miscellaneous initialization
torch.manual_seed(0)
start_time = time.time()

# MPS parameters
bond_dim = 2
adaptive_mode = False
periodic_bc = False

# Training parameters
batch_size = 100
num_epochs = 50
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
        return x


mps = MyMPS(n_sites=3,
            d_phys=2,
            n_labels=2,
            d_bond=bond_dim,
            l_position=None,
            boundary='obc',
            param_bond=False)
mps = mps.cuda()

# Set our loss function and optimizer
loss_fun = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)


# Get the training and test sets
def embedding(image: torch.Tensor) -> torch.Tensor:
    return torch.stack([image, 1 - image], dim=1)


data = torch.rand((10000, 2))
labels = data[:, 0] * data[:, 1]
embedded_data = embedding(data)

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

    for i in range(0, 10000, batch_size):
        inp = embedded_data[i:i+batch_size, :, :]
        lab = labels[i:i+batch_size]
        inp, lab = inp.cuda(), lab.cuda()

        # Call our MPS to get logit scores and predictions
        scores = mps(inp)

        # Compute the loss and accuracy, add them to the running totals
        loss = loss_fun(scores[:, 0], lab)
        with torch.no_grad():
            running_loss += loss

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"### Epoch {epoch_num} ###")
    print(f"Average loss:           {running_loss / (10000/batch_size):.4f}")
    print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")


test_data = embedding(torch.rand(batch_size, 2)).cuda()
test = mps(test_data)
print(test_data[:10, 0, 0] * test_data[:10, 0, 1])
print(test[:10, 0])
