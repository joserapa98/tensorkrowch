"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tk

from typing import Sequence
import time


class TestMPS:
    
    def test_all_algorithms(self):
        example = torch.randn(10, 1, 5)
        data = torch.randn(10, 100, 5)
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                for l_position in [0, 1, 5, 9, 10]:
                    
                    mps = tk.MPS(n_sites=11,
                                 d_phys=5,
                                 n_labels=12,
                                 d_bond=2,
                                 l_position=l_position,
                                 boundary=boundary,
                                 param_bond=param_bond)
                    
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    mps.automemory = automemory
                                    mps.unbind_mode = unbind_mode
                                    mps.inline_input = inline_input
                                    mps.inline_mats = inline_mats
                                    
                                    mps.trace(example)
                                    result = mps(data)
                                    
                                    assert result.shape == (100, 12)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 11
                                    assert len(mps.data_nodes) == 10
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 4
                                    else:
                                        assert len(mps.virtual_nodes) == 3
                                        
    def test_all_algorithms_diff_d_phys(self):
        d_phys = torch.randint(low=2, high=7, size=(10,)).tolist()
        example = [torch.randn(1, d) for d in d_phys]
        data = [torch.randn(100, d) for d in d_phys]
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                for l_position in [0, 1, 5, 9, 10]:
                    
                    mps = tk.MPS(n_sites=11,
                                 d_phys=d_phys,
                                 n_labels=12,
                                 d_bond=2,
                                 l_position=l_position,
                                 boundary=boundary,
                                 param_bond=param_bond)
                    
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    mps.automemory = automemory
                                    mps.unbind_mode = unbind_mode
                                    mps.inline_input = inline_input
                                    mps.inline_mats = inline_mats
                                    
                                    mps.trace(example)
                                    result = mps(data)
                                    
                                    assert result.shape == (100, 12)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 11
                                    assert len(mps.data_nodes) == 10
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 1
                                    else:
                                        assert len(mps.virtual_nodes) == 0
                                        
    def test_all_algorithms_diff_d_bond(self):
        d_bond = torch.randint(low=2, high=7, size=(11,)).tolist()
        example = torch.randn(10, 1, 5)
        data = torch.randn(10, 100, 5)
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                for l_position in [0, 1, 5, 9, 10]:
                    
                    mps = tk.MPS(n_sites=11,
                                 d_phys=5,
                                 n_labels=12,
                                 d_bond=d_bond[:-1] if boundary == 'obc' else d_bond,
                                 l_position=l_position,
                                 boundary=boundary,
                                 param_bond=param_bond)
                    
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    print(boundary, param_bond, l_position,
                                          automemory, unbind_mode, inline_input, inline_mats)
                                    mps.automemory = automemory
                                    mps.unbind_mode = unbind_mode
                                    mps.inline_input = inline_input
                                    mps.inline_mats = inline_mats
                                    
                                    mps.trace(example)
                                    result = mps(data)
                                    
                                    assert result.shape == (100, 12)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 11
                                    assert len(mps.data_nodes) == 10
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 4
                                    else:
                                        assert len(mps.virtual_nodes) == 3
                                        
    def test_all_algorithms_diff_d_phys_d_bond(self):
        d_phys = torch.randint(low=2, high=7, size=(10,)).tolist()
        d_bond = torch.randint(low=2, high=7, size=(11,)).tolist()
        example = [torch.randn(1, d) for d in d_phys]
        data = [torch.randn(100, d) for d in d_phys]
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                for l_position in [0, 1, 5, 9, 10]:
                    
                    mps = tk.MPS(n_sites=11,
                                 d_phys=d_phys,
                                 n_labels=12,
                                 d_bond=d_bond[:-1] if boundary == 'obc' else d_bond,
                                 l_position=l_position,
                                 boundary=boundary,
                                 param_bond=param_bond)
                    
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    print(boundary, param_bond, l_position,
                                          automemory, unbind_mode, inline_input, inline_mats)
                                    mps.automemory = automemory
                                    mps.unbind_mode = unbind_mode
                                    mps.inline_input = inline_input
                                    mps.inline_mats = inline_mats
                                    
                                    mps.trace(example)
                                    result = mps(data)
                                    
                                    assert result.shape == (100, 12)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 11
                                    assert len(mps.data_nodes) == 10
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 1
                                    else:
                                        assert len(mps.virtual_nodes) == 0

    def test_extreme_cases(self):
        # Left node + Outpt node
        mps = tk.MPS(n_sites=2,
                     d_phys=5,
                     n_labels=12,
                     d_bond=2,
                     l_position=0,
                     boundary='obc',
                     param_bond=True)
        data = torch.randn(1, 100, 5)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.automemory = automemory
                        mps.unbind_mode = unbind_mode
                        mps.inline_input = inline_input
                        mps.inline_mats = inline_mats
                        
                        mps.trace(data[:, 0, :].view(1, 1, 5))
                        result = mps(data)
                        
                        assert result.shape == (100, 12)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 3

        # Output node + Right node
        mps = tk.MPS(n_sites=2,
                     d_phys=5,
                     n_labels=12,
                     d_bond=2,
                     l_position=1,
                     boundary='obc',
                     param_bond=True)
        data = torch.randn(1, 100, 5)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.automemory = automemory
                        mps.unbind_mode = unbind_mode
                        mps.inline_input = inline_input
                        mps.inline_mats = inline_mats
                        
                        mps.trace(data[:, 0, :].view(1, 1, 5))
                        result = mps(data)
                        
                        assert result.shape == (100, 12)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 3

        # Outpt node
        mps = tk.MPS(n_sites=1,
                     d_phys=5,
                     n_labels=12,
                     d_bond=2,
                     l_position=0,
                     boundary='pbc',
                     param_bond=True)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.automemory = automemory
                        mps.unbind_mode = unbind_mode
                        mps.inline_input = inline_input
                        mps.inline_mats = inline_mats
                        
                        # We shouldn't pass any data since the MPS only
                        # has the output node
                        mps.trace()
                        result = mps()  # Same as mps.contract()
                        
                        assert result.shape == (12,)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 0
                        assert len(mps.virtual_nodes) == 0

    def test_check_grad(self):
        mps = tk.MPS(n_sites=2,
                     d_phys=2,
                     n_labels=2,
                     d_bond=2,
                     l_position=1,
                     boundary='obc').cuda()

        data = torch.randn(1, 1, 2).cuda()
        result = mps.forward(data)
        result[0, 0].backward()

        I = data.squeeze(0)
        A = mps.left_node.tensor
        B = mps.output_node.tensor
        grad_A1 = mps.left_node.grad
        grad_B1 = mps.output_node.grad

        grad_A2 = I.t() @ B[:, 0].view(2, 1).t()
        grad_B2 = (I @ A).t() @ torch.tensor([[1., 0.]]).cuda()

        assert torch.equal(grad_A1, grad_A2)
        assert torch.equal(grad_B1, grad_B2)
        
        

    def test_example2_mps(self):
        mps = tk.MPS(n_sites=5,
                     d_phys=2,
                     n_labels=2,
                     d_bond=2,
                     boundary='obc')
        
        for node in mps.nodes.values():
            node.set_tensor(init_method='ones')

        data = torch.ones(4, 1)
        data = torch.stack([data, 1 - data], dim=2)
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

    class ConvNode(tk.TensorNetwork):
        
        def __init__(self, kernel_size, input_dim, output_dim):
            super().__init__(name='ConvTN')
            
            num_input = kernel_size[0] * kernel_size[1]
            node = tk.ParamNode(shape=(*([input_dim]*num_input), output_dim),
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
            # for data_node in self.data_nodes.values():
            #     data_node.axes[1].name = 'stack'
        
        def contract(self):
            result = self['node']
            for node in self.data_nodes.values():
                result @= node
            return result
        
    convnode = ConvNode(kernel_size=(kh, kw),
                        input_dim=n_channels,
                        output_dim=5)

    result = convnode(patches)


def test_param_non_leaf():
    node1 = tk.randn(shape=(2, 3, 2))
    node2 = tk.randn(shape=(2, 3, 2))
    
    node1[2] ^ node2[0]
    node3 = node1 @ node2
    
    node3.name = 'paramnode3'
    paramnode3 = node3.parameterize()
    
    print()


def test_conv_mps():
    class MPSLayer(tk.TensorNetwork):
    
        def __init__(self, in_channels, out_channels, kernel_size, bond_dim=10):
            super().__init__(name='MPS')
            
            input_nodes = []
            for _ in range(kernel_size[0] * kernel_size[1]):
                node = tk.ParamNode(shape=(bond_dim, in_channels, bond_dim),
                                    axes_names=('left', 'input', 'right'),
                                    name='input_node',
                                    network=self)
                input_nodes.append(node)
                
            for i in range(len(input_nodes) - 1):
                input_nodes[i]['right'] ^ input_nodes[i + 1]['left']
                
            output_node = tk.ParamNode(shape=(bond_dim, out_channels, bond_dim),
                                    axes_names=('left', 'output', 'right'),
                                    name='output_node',
                                    network=self)
            output_node['right'] ^ input_nodes[0]['left']
            output_node['left'] ^ input_nodes[-1]['right']
            
            
            std = 1e-9
            for node in input_nodes:
                tensor = torch.randn(node.shape) * std
                random_eye = torch.randn(tensor.shape[0], tensor.shape[2]) * std
                random_eye  = random_eye + torch.eye(tensor.shape[0], tensor.shape[2])
                tensor[:, 0, :] = random_eye
                
                node.tensor = tensor
                    
            eye_tensor = torch.eye(output_node.shape[0], output_node.shape[2])
            eye_tensor = eye_tensor.view([output_node.shape[0], 1, output_node.shape[2]])
            eye_tensor = eye_tensor.expand(output_node.shape)
            tensor = eye_tensor + std * torch.randn(output_node.shape)
            
            output_node.tensor = tensor
            
            self.input_nodes = input_nodes
            self.output_node = output_node
            
            
        def set_data_nodes(self) -> None:
                input_edges = []
                for node in self.input_nodes:
                    input_edges.append(node['input'])
                        
                super().set_data_nodes(input_edges, 2)
                # for data_node in self.data_nodes.values():
                #     data_node.axes[1].name = 'stack_patches'
        
        def contract(self):
            stack_input = tk.stack(self.input_nodes)
            stack_data = tk.stack(list(self.data_nodes.values()))
            
            stack_input['input'] ^ stack_data['feature']
            stack_result = stack_input @ stack_data
            
            # stack_result = stack_result.permute((0, 3, 1, 2))
            stack_result = tk.unbind(stack_result)
            
            result = stack_result[0]
            for node in stack_result[1:]:
                result @= node
            result @= self.output_node
            
            return result
        
        
    class ConvNodeLayer(nn.Module):
    
        def __init__(self,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    example_dims=(28, 28)):
            super().__init__()
            
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            elif not isinstance(kernel_size, Sequence):
                raise TypeError('`kernel_size` must be int or Sequence')
            
            if isinstance(stride, int):
                stride = (stride, stride)
            elif not isinstance(stride, Sequence):
                raise TypeError('`stride` must be int or Sequence')
            
            if isinstance(padding, int):
                padding = (padding, padding)
            elif not isinstance(padding, Sequence):
                raise TypeError('`padding` must be int or Sequence')
            
            if isinstance(dilation, int):
                dilation = (dilation, dilation)
            elif not isinstance(dilation, Sequence):
                raise TypeError('`dilation` must be int or Sequence')
            
            self.in_channels = in_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            
            self.unfold = nn.Unfold(kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation)
            
            # self.nodelayer = NodeLayer(in_channels=in_channels,
            #                            out_channels=out_channels,
            #                            kernel_size=kernel_size)
            self.nodelayer = MPSLayer(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size)
            
            # Trace TN
            example = torch.zeros(1, in_channels, *example_dims)
            patches = self.unfold(example).transpose(1, 2)
            patches = patches.view(*patches.shape[:-1], self.in_channels, -1)
            patches = patches.permute(3, 0, 1, 2)
            
            self.nodelayer.trace(patches)
            
        def forward(self, image):
            # Input image shape: batch_size x in_channels x height x width
            
            # batch_size x nb_windows x (in_channels * nb_pixels)
            patches = self.unfold(image).transpose(1, 2)
            
            # batch_size x nb_windows x in_channels x nb_pixels
            patches = patches.view(*patches.shape[:-1], self.in_channels, -1)
            
            # nb_pixels x batch_size x nb_windows x in_channels
            patches = patches.permute(3, 0, 1, 2)
            
            # batch_size x nb_windows x out_channels
            result = self.nodelayer(patches)
            
            # batch_size x out_channels x nb_windows
            result = result.transpose(1, 2)
            
            h_in = image.shape[2]
            w_in = image.shape[3]
            
            h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] * \
                (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
            w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] * \
                        (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
            
            # batch_size x out_channels x height_out x width_out
            result = result.view(*result.shape[:-1], h_out, w_out)
            
            return result
        
    
    image = torch.randn(500, 14, 14)  # batch_size x height x width

    def embedding(image: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.ones_like(image),
                            image], dim=1)
        
    image = embedding(image)
    print(image.shape)  # batch_size x in_channels x height x width


    model = ConvNodeLayer(2, 5, (3, 3), stride=2, padding=1, example_dims=(14, 14))

    model.nodelayer.automemory = True
    model.nodelayer.unbind_mode = True

    start = time.time()
    result = model(image)
    print(time.time() - start, result.shape)

    start = time.time()
    result = model(image)
    print(time.time() - start, result.shape)