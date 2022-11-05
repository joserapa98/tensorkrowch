"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tn


class TestMPS:
    
    def test_mps1(self):
        # boundary = obc, param_bond = False
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='obc')

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21 # TODO: no uso permanent_nodes
        # TODO: It is equal to 13 because it counts Stacknode edges,
        #  should we have also references to the _leaf nodes??
        assert len(mps.edges) == 1
        
    def test_mps2(self):
        # boundary = obc, param_bond = False
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='obc')

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1
        
    def test_mps3(self):
        # boundary = obc, param_bond = False
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='obc')

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1
        
    def test_mps4(self):
        # boundary = obc, param_bond = False
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='obc')

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1
        
    def test_mps5(self):
        # boundary = obc, param_bond = False
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='obc')

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1
        
    def test_mps6(self):
        # boundary = obc, param_bond = True
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='obc', param_bond=True)

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1
        
    def test_mps7(self):
        # boundary = obc, param_bond = True
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='obc', param_bond=True)

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1
        
    def test_mps8(self):
        # boundary = obc, param_bond = True
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='obc', param_bond=True)

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1
    
    def test_mps9(self):
        # boundary = obc, param_bond = True
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='obc', param_bond=True)

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps10(self):
        # boundary = obc, param_bond = True
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='obc', param_bond=True)

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps11(self):
        # boundary = pbc, param_bond = False
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='pbc')

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps12(self):
        # boundary = pbc, param_bond = False
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='pbc')

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps13(self):
        # boundary = pbc, param_bond = False
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='pbc')

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps14(self):
        # boundary = pbc, param_bond = False
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='pbc')

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps15(self):
        # boundary = pbc, param_bond = False
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='pbc')

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps16(self):
        # boundary = pbc, param_bond = True
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=5, boundary='pbc', param_bond=True)

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps17(self):
        # boundary = pbc, param_bond = True
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=0, boundary='pbc', param_bond=True)

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps18(self):
        # boundary = pbc, param_bond = True
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=1, boundary='pbc', param_bond=True)

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps19(self):
        # boundary = pbc, param_bond = True
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=9, boundary='pbc', param_bond=True)

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_mps20(self):
        # boundary = pbc, param_bond = True
        mps = tn.MPS(n_sites=11, d_phys=5, n_labels=10, d_bond=2, l_position=10, boundary='pbc', param_bond=True)

        data = torch.randn(1000, 5, 10)
        result = mps.forward(data)
        mean = result.mean(0)
        mean[0].backward()
        std = result.std(0)
        assert len(mps.permanent_nodes) == 21
        assert len(mps.edges) == 1

    def test_extreme_cases(self):
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
    grad_A1 = mps.left_node.grad
    grad_B1 = mps.output_node.grad

    grad_A2 = I.t() @ B[:, 0].view(2, 1).t()
    grad_B2 = (I @ A).t() @ torch.tensor([[1., 0.]]).cuda()

    assert torch.equal(grad_A1, grad_A2)
    assert torch.equal(grad_B1, grad_B2)


def test_example2_mps():
    mps = tn.MPS(n_sites=5, d_phys=2, n_labels=2, d_bond=2, boundary='obc')
    for node in mps.nodes.values():
        node.set_tensor(init_method='ones')

    data = torch.ones(1, 4)
    data = torch.stack([data, 1 - data], dim=1)
    result = mps.forward(data)
    result[0, 0].backward()
    result


def test_convnode():
    image = torch.randn(1, 28, 28)  # batch x height x width
    
    def embedding(image: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.ones_like(image),
                            image,
                            1 - image], dim=1)
        
    image = embedding(image)
    print(image.shape)  # batch x channels x height x width

    n_channels = 3

    kh, kw = 3, 3 # kernel size
    dh, dw = 3, 3 # stride

    # Manual approach
    patches = image.unfold(2, kh, dh).unfold(3, kw, dw)
    print(patches.shape) # batch_size, channels, h_windows, w_windows, kh, kw

    patches = patches.contiguous().view(1, n_channels, -1, kh, kw)
    print(patches.shape) # batch_size, channels, windows, kh, kw

    nb_windows = patches.size(2)

    # Now we have to shift the windows into the batch dimension.
    # Maybe there is another way without .permute, but this should work
    patches = patches.permute(0, 2, 1, 3, 4)
    print(patches.shape) # batch_size, nb_windows, channels, kh, kw

    patches = patches.view((*patches.shape[:-2], -1))
    print(patches.shape) # batch_size, nb_windows, channels, num_input

    patches = patches.permute(3, 0, 1, 2)
    print(patches.shape) # num_input, batch_size, nb_windows, channels

    class ConvNode(tn.TensorNetwork):
        
        def __init__(self, kernel_size, input_dim, output_dim):
            super().__init__(name='ConvTN')
            
            num_input = kernel_size[0] * kernel_size[1]
            node = tn.ParamNode(shape=(*([input_dim]*num_input), output_dim),
                                axes_names=(*(['input']*num_input), 'output'),
                                network=self,
                                name='node')
            
            tensor = 1e-9 * torch.randn(node.shape)
            tensor[(0,)*(num_input + 1)] = 1.
            node.tensor = tensor
            
        def set_data_nodes(self) -> None:
            input_edges = []
            for edge in self.edges:
                if edge.axis1.name != 'output':
                    input_edges.append(edge)
                    
            super().set_data_nodes(input_edges, 2)
            for data_node in self.data_nodes.values():
                data_node.axes[1].name = 'stack'
        
        def contract(self):
            result = self['node']
            for node in self.data_nodes.values():
                result @= node
            return result
        
    convnode = ConvNode(kernel_size=(kh, kw),
                        input_dim=n_channels,
                        output_dim=5)

    result = convnode(patches)

