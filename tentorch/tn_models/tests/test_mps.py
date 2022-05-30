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
    assert len(mps.permanent_nodes) == 21
    # TODO: It is equal to 13 because it counts Stacknode edges,
    #  should we have also references to the permanent nodes??
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='obc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='obc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='obc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='obc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    # boundary = obc, param_bond = True
    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='obc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='obc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='obc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='obc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='obc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    # boundary = pbc, param_bond = False
    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='pbc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='pbc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='pbc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='pbc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='pbc')

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    # boundary = pbc, param_bond = True
    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='pbc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='pbc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='pbc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='pbc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='pbc', param_bond=True)

    data = torch.randn(1000, 5, 10)
    result = mps.forward(data)
    mean = result.mean(0)
    mean[0].backward()
    std = result.std(0)
    assert len(mps.permanent_nodes) == 21
    #assert len(mps.edges) == 1

    # Extreme cases
    mps = tn.MPS(n_sites=2, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='obc', param_bond=True)

    mps = tn.MPS(n_sites=2, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='obc', param_bond=True)

    mps = tn.MPS(n_sites=1, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='pbc', param_bond=True)


def test_example_mps():
    mps = tn.MPS(n_sites=2, d_phys=2, n_labels=2, d_bond=2, l_position=1, boundary='obc').cuda()

    data = torch.randn(1, 2, 1).cuda()
    result = mps.forward(data)
    result[0, 0].backward()

    I = data.squeeze(2)
    A = mps.left_node.tensor
    B = mps.output_node.tensor

    grad_A = I.t() @ B[:, 0].view(2, 1).t()
    grad_B = (I @ A).t() @ torch.tensor([[1., 0.]]).cuda()

    assert torch.equal(A.grad, grad_A)
    assert torch.equal(B.grad, grad_B)


def test_example2_mps():
    mps = tn.MPS(n_sites=5, d_phys=2, n_labels=2, d_bond=2, boundary='obc')
    for node in mps.nodes.values():
        node.set_tensor(init_method='ones')

    data = torch.ones(1, 4)
    data = torch.stack([data, 1 - data], dim=1)
    result = mps.forward(data)
    result[0, 0].backward()
    result



