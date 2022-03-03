"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tentorch as tn


def test_mps():
    mps = tn.MPS(n_sites=11,
                 d_phys=5,
                 d_phys_l=10,
                 d_bond=2,
                 l_position=5,
                 boundary='pbc',
                 param_bond=True)

    mps.set_data_nodes(batch_sizes=[100])
    mps

