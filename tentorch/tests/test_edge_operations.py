"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tentorch as tn

import time
import opt_einsum
import dis


def test_svd_edge():
    # TODO: complicado, esto pal futuro
    node1 = tn.Node(shape=(2, 5, 3), axes_names=('left', 'contract', 'batch1'), name='node1', init_method='randn')
    node2 = tn.Node(shape=(3, 5, 7), axes_names=('batch2', 'contract', 'right'), name='node2', init_method='randn')
    edge = node1['contract'] ^ node2['contract']
    edge.svd(side='left', cum_percentage=0.9)

