"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tn

import time
import opt_einsum
import dis


def test_svd_edge():
    # TODO: complicado, esto pal futuro
    # TODO: problem, if we have 1 batch edge in each node, but they have different names,
    # I would not be able to build a new shape in contraction, 2 -1 dimensions
    node1 = tn.Node(shape=(2, 5, 3), axes_names=('left', 'contract', 'brratch1'), name='node1', init_method='randn')
    node2 = tn.Node(shape=(3, 5, 7), axes_names=('brratch2', 'contract', 'right'), name='node2', init_method='randn')
    edge = node1['contract'] ^ node2['contract']
    edge.svd(side='left', cum_percentage=0.9)

