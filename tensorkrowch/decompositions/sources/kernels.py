"""
This script contains:

    Kernel functions:
        * gaussian_kernel
"""

from math import log, pi
from typing import Sequence, Union

import torch


def gaussian_kernel(coordinates: torch.Tensor,
                    dataset: torch.Tensor,
                    bandwidth: Union[float, Sequence[float], torch.Tensor] = 1.0
                    ) -> torch.Tensor:
    r"""
    Evaluates normalized Gaussian kernels centered at the dataset samples.

    Each sample contributes a density
    :math:`K_h(x,z)=\exp(-\frac12\sum_i((x_i-z_i)/h_i)^2) /
    ((2\pi)^{d/2}\prod_i h_i)`. Scalar ``bandwidth`` is shared across coordinates.
    Normalization is over the full continuous domain; restricting evaluation
    to a finite grid does not renormalize these densities.

    Parameters
    ----------
    coordinates : torch.Tensor
        Queries with shape ``(batch, coordinates)``.
    dataset : torch.Tensor
        Kernel centers with shape ``(samples, coordinates)``.
    bandwidth : float, sequence[float] or torch.Tensor
        Positive finite width, either scalar or one value per coordinate.
        Its default is ``1.0``; it is not inferred from the samples.

    Returns
    -------
    torch.Tensor
        Kernel values of shape ``(batch, samples)``. Sample weights and their
        reduction belong to the source. Query and parameter gradients remain.

    Examples
    --------
    >>> tk.decompositions.gaussian_kernel(
    ...     torch.tensor([[0.], [1.]]), torch.tensor([[0.]]))
    tensor([[0.3989],
            [0.2420]])
    """
    if not isinstance(coordinates, torch.Tensor) or not isinstance(dataset, torch.Tensor):
        raise TypeError('`coordinates` and `dataset` should be tensors')
    if coordinates.ndim != 2 or dataset.ndim != 2 or coordinates.shape[1] != dataset.shape[1]:
        raise ValueError('`coordinates` and `dataset` should share coordinate dimensions')
    if coordinates.is_complex() or dataset.is_complex():
        raise TypeError('Gaussian coordinates should be real')
    dtype = torch.promote_types(coordinates.dtype, dataset.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    widths = torch.as_tensor(bandwidth, device=coordinates.device, dtype=dtype)
    if widths.ndim == 0:
        widths = widths.expand(coordinates.shape[1])
    if widths.shape != (coordinates.shape[1],) or not torch.isfinite(widths).all() or torch.any(widths <= 0):
        raise ValueError('`bandwidth` should be finite, positive and match coordinates')
    differences = (coordinates.to(dtype)[:, None] -
                   dataset.to(device=coordinates.device, dtype=dtype)[None]) / widths
    exponent = -0.5 * differences.square().sum(-1)
    log_scale = widths.log().sum() + coordinates.shape[1] * 0.5 * log(2 * pi)
    return (exponent - log_scale).exp()


__all__ = ['gaussian_kernel']
