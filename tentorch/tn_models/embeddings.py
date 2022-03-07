"""
Embedding methods
"""

from math import pi, sqrt
import torch

from tentorch.utils import comb_num


def unit(data: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Parameters
    ----------
    data: input tensor with shape batch x 1 -> batch x dim
    dim: embedding dimension
    """
    embedded_data = torch.empty(data.shape[0], dim)
    for i in range(1, dim + 1):
        aux = sqrt(comb_num(dim - 1, i - 1)) *\
              (pi / 2 * data[:, 0]).cos().pow(dim - i) *\
              (pi / 2 * data[:, 0]).sin().pow(i - 1)
        embedded_data[:, i - 1] = aux
    return embedded_data
