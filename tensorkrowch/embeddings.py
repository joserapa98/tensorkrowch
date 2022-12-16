"""
Embedding methods
"""

from math import pi, sqrt
import torch

from tensorkrowch.utils import comb_num


# TODO: no muy seguro de esto
def unit(data: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Parameters
    ----------
    data: input tensor with shape n_features x batch x 1 -> n_features x batch x dim
    dim: embedding dimension
    """
    # shape = list(data.shape)
    # shape[-1] = dim
    # embedded_data = torch.empty(shape)
    lst_tensors = []
    for i in range(1, dim + 1):
        aux = sqrt(comb_num(dim - 1, i - 1)) * \
                (pi / 2 * data).cos().pow(dim - i) * \
                (pi / 2 * data).sin().pow(i - 1)
        lst_tensors.append(aux)
        # embedded_data[..., i - 1] = aux
    return torch.stack(lst_tensors, dim=-1) # embedded_data


# TODO: esto creo que no lo uso
def ones(data: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.ones_like(data), data], dim=-1)
