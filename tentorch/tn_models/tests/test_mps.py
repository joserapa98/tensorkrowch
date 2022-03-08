"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tentorch as tn


def test_mps():
    # boundary = obc, param_bond = False
    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='obc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='obc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='obc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='obc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='obc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    # boundary = obc, param_bond = True
    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='obc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='obc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='obc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='obc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='obc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    # boundary = pbc, param_bond = False
    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='pbc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='pbc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='pbc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='pbc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='pbc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    # boundary = pbc, param_bond = True
    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='pbc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='pbc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='pbc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='pbc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='pbc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.nodes) == 21
    assert len(mps.edges) == 1

    # Extreme cases
    mps = tn.MPS(n_sites=2, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='obc', param_bond=True)

    mps = tn.MPS(n_sites=2, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='obc', param_bond=True)

    mps = tn.MPS(n_sites=1, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='pbc', param_bond=True)