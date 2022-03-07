"""
Train MPS models
"""

import torch

from tentorch.tn_models.embeddings import unit
from tentorch.tn_models.mps import MPS

data = torch.randn(100, 1, 10)  # batch x feature x n_features
data_list = data.unbind(2)
embedded_data_list = list(map(lambda x: unit(data=x, dim=2), data_list))

mps = MPS(n_sites=11,
          d_phys=2,
          d_phys_l=10,
          d_bond=2,
          l_position=5,
          boundary='obc')


