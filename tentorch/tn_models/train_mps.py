"""
Train MPS models
"""

import torch
import torch.nn as nn
import torch.optim as optim

from tentorch.tn_models.embeddings import unit
from tentorch.tn_models.mps import MPS

data = torch.randn(10000, 1, 10)  # batch x feature x n_features
data_list = data.unbind(2)
embedded_data_list = list(map(lambda x: unit(data=x, dim=2), data_list))
embedded_data = torch.stack(embedded_data_list, dim=2)
label = data.sum((1, 2))

mps = MPS(n_sites=11,
          d_phys=2,
          d_phys_l=10,
          d_bond=2,
          l_position=5,
          boundary='obc')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-4
weight_decay = 0.0
#momentum = 0.6

criterion = nn.MSELoss()
optimizer = optim.Adam(mps.parameters(),
                       lr=learning_rate,
                       weight_decay=weight_decay,)
                       #momentum=momentum)

print(mps(embedded_data)[0, :])

