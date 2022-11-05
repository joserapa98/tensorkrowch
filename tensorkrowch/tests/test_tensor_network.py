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


# TODO: funcion de copiar Tn entera
# TODO: test aplicar mismas contracciones varias veces, se reutilizan nodos, pero no se optimiza memoria
# TODO: test definir subclase de TN, y definir y usar forward, se reutilizan nodos y se optimiza memoria
