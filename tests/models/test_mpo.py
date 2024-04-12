"""
Tests for mpo:

    * TestMPO
    * TestUMPO
"""

import pytest

import torch
import tensorkrowch as tk


class TestMPO:
    
    def test_initialize_with_tensors(self):
        for n in [1, 2, 5]:
            # PBC
            tensors = [torch.randn(10, 2, 10, 2) for _ in range(n)]
            mpo = tk.models.MPO(tensors=tensors)
            assert mpo.n_features == n
            assert mpo.boundary == 'pbc'
            assert mpo.in_dim == [2] * n
            assert mpo.out_dim == [2] * n
            assert mpo.bond_dim == [10] * n
            
            # OBC
            tensors = [torch.randn(10, 2, 10, 2) for _ in range(n)]
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0, :]
            mpo = tk.models.MPO(tensors=tensors)
            assert mpo.n_features == n
            assert mpo.boundary == 'obc'
            assert mpo.in_dim == [2] * n
            assert mpo.out_dim == [2] * n
            assert mpo.bond_dim == [10] * (n - 1)
    
    def test_initialize_with_tensors_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for n in [1, 2, 5]:
            # PBC
            tensors = [torch.randn(10, 2, 10, 2, device=device) for _ in range(n)]
            mpo = tk.models.MPO(tensors=tensors)
            assert mpo.n_features == n
            assert mpo.boundary == 'pbc'
            assert mpo.in_dim == [2] * n
            assert mpo.out_dim == [2] * n
            assert mpo.bond_dim == [10] * n
            for node in mpo.mats_env:
                assert node.device == device
            
            # OBC
            tensors = [torch.randn(10, 2, 10, 2, device=device) for _ in range(n)]
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0, :]
            mpo = tk.models.MPO(tensors=tensors)
            assert mpo.n_features == n
            assert mpo.boundary == 'obc'
            assert mpo.in_dim == [2] * n
            assert mpo.out_dim == [2] * n
            assert mpo.bond_dim == [10] * (n - 1)
            for node in mpo.mats_env:
                assert node.device == device
    
    def test_initialize_with_tensors_ignore_rest(self):
        tensors = [torch.randn(10, 2, 10, 2) for _ in range(10)]
        mpo = tk.models.MPO(tensors=tensors,
                            boundary='obc',
                            n_features=3,
                            in_dim=4,
                            out_dim=3,
                            bond_dim=7)
        assert mpo.boundary == 'pbc'
        assert mpo.n_features == 10
        assert mpo.in_dim == [2] * 10
        assert mpo.out_dim == [2] * 10
        assert mpo.bond_dim == [10] * 10
        
    def test_initialize_with_tensors_errors(self):
        # Tensors should be at most rank-4 tensors
        tensors = [torch.randn(10, 2, 10, 2, 5) for _ in range(10)]
        with pytest.raises(ValueError):
            mpo = tk.models.MPO(tensors=tensors)
        
        # First and last tensors should have the same rank
        tensors = [torch.randn(10, 2, 10, 2) for _ in range(10)]
        tensors[0] = tensors[0][0]
        with pytest.raises(ValueError):
            mpo = tk.models.MPO(tensors=tensors)
        
        # First and last bond dims should coincide
        tensors = [torch.randn(10, 2, 10, 2) for _ in range(10)]
        tensors[0] = tensors[0][:5]
        tensors[-1] = tensors[-1][..., :3, 0]
        with pytest.raises(ValueError):
            mpo = tk.models.MPO(tensors=tensors)
    
    def test_initialize_init_method(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mpo = tk.models.MPO(boundary='pbc',
                                    n_features=n,
                                    in_dim=2,
                                    out_dim=2,
                                    bond_dim=10,
                                    init_method=init_method)
                assert mpo.n_features == n
                assert mpo.boundary == 'pbc'
                assert mpo.in_dim == [2] * n
                assert mpo.out_dim == [2] * n
                assert mpo.bond_dim == [10] * n
                
                # OBC
                mpo = tk.models.MPO(boundary='obc',
                                    n_features=n,
                                    in_dim=2,
                                    out_dim=2,
                                    bond_dim=10,
                                    init_method=init_method)
                assert mpo.n_features == n
                assert mpo.boundary == 'obc'
                assert mpo.in_dim == [2] * n
                assert mpo.out_dim == [2] * n
                assert mpo.bond_dim == [10] * (n - 1)
                
                assert torch.equal(mpo.left_node.tensor[0],
                                   torch.ones_like(mpo.left_node.tensor)[0])
                assert torch.equal(mpo.left_node.tensor[1:],
                                   torch.zeros_like(mpo.left_node.tensor)[1:])
                
                assert torch.equal(mpo.right_node.tensor[0],
                                   torch.ones_like(mpo.right_node.tensor)[0])
                assert torch.equal(mpo.right_node.tensor[1:],
                                   torch.zeros_like(mpo.right_node.tensor)[1:])
    
    def test_initialize_init_method_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mpo = tk.models.MPO(boundary='pbc',
                                    n_features=n,
                                    in_dim=2,
                                    out_dim=2,
                                    bond_dim=10,
                                    init_method=init_method,
                                    device=device)
                assert mpo.n_features == n
                assert mpo.boundary == 'pbc'
                assert mpo.in_dim == [2] * n
                assert mpo.out_dim == [2] * n
                assert mpo.bond_dim == [10] * n
                for node in mpo.mats_env:
                    assert node.device == device
                
                # OBC
                mpo = tk.models.MPO(boundary='obc',
                                    n_features=n,
                                    in_dim=2,
                                    out_dim=2,
                                    bond_dim=10,
                                    init_method=init_method,
                                    device=device)
                assert mpo.n_features == n
                assert mpo.boundary == 'obc'
                assert mpo.in_dim == [2] * n
                assert mpo.out_dim == [2] * n
                assert mpo.bond_dim == [10] * (n - 1)
                for node in mpo.mats_env:
                    assert node.device == device
                
                assert torch.equal(mpo.left_node.tensor[0],
                                   torch.ones_like(mpo.left_node.tensor)[0])
                assert torch.equal(mpo.left_node.tensor[1:],
                                   torch.zeros_like(mpo.left_node.tensor)[1:])
                
                assert torch.equal(mpo.right_node.tensor[0],
                                   torch.ones_like(mpo.right_node.tensor)[0])
                assert torch.equal(mpo.right_node.tensor[1:],
                                   torch.zeros_like(mpo.right_node.tensor)[1:])
    
    def test_in_out_dims(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                in_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                out_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                
                mpo = tk.models.MPO(n_features=n_features,
                                    in_dim=in_dim,
                                    out_dim=out_dim,
                                    bond_dim=10,
                                    boundary=boundary)
                
                assert mpo.in_dim == in_dim
                assert mpo.out_dim == out_dim
    
    def test_in_out_dims_error(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                in_dim = torch.randint(low=2, high=10, size=(n_features + 1,)).tolist()
                out_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                
                # in_dim should have n_features elements
                with pytest.raises(ValueError):
                    mpo = tk.models.MPO(n_features=n_features,
                                        in_dim=in_dim,
                                        out_dim=out_dim,
                                        bond_dim=10,
                                        boundary=boundary)
                
                in_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                out_dim = torch.randint(low=2, high=10, size=(n_features + 1,)).tolist()
                
                # out_dim should have n_features elements
                with pytest.raises(ValueError):
                    mpo = tk.models.MPO(n_features=n_features,
                                        in_dim=in_dim,
                                        out_dim=out_dim,
                                        bond_dim=10,
                                        boundary=boundary)
    
    def test_bond_dims(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                
                mpo = tk.models.MPO(n_features=n_features,
                                    in_dim=5,
                                    out_dim=5,
                                    bond_dim=bond_dim,
                                    boundary=boundary)
                
                assert mpo.in_dim == [5] * n_features
                assert mpo.out_dim == [5] * n_features
                assert mpo.bond_dim == bond_dim
                
                extended_bond_dim = [mpo.mats_env[0].shape[0]] + \
                    [node.shape[2] for node in mpo.mats_env]
                
                if boundary == 'obc':
                    if n_features == 1:
                        assert extended_bond_dim == [1, 1]
                    else:
                        assert extended_bond_dim == [bond_dim[0]] + bond_dim + [bond_dim[-1]]
                else:
                    assert extended_bond_dim == [bond_dim[-1]] + bond_dim
    
    def test_copy(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                for share_tensors in [True, False]:
                    in_dim = torch.randint(low=2,
                                             high=12,
                                             size=(n_features,)).tolist()
                    out_dim = torch.randint(low=2,
                                             high=12,
                                             size=(n_features,)).tolist()
                    bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                    bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                    
                    mpo = tk.models.MPO(n_features=n_features,
                                        in_dim=in_dim,
                                        out_dim=out_dim,
                                        bond_dim=bond_dim,
                                        boundary=boundary)
                    
                    copied_mpo = mpo.copy(share_tensors=share_tensors)
                    
                    assert isinstance(copied_mpo, tk.models.MPO)
                    assert mpo.n_features == copied_mpo.n_features
                    assert mpo.in_dim == copied_mpo.in_dim
                    assert mpo.out_dim == copied_mpo.out_dim
                    assert mpo.bond_dim == copied_mpo.bond_dim
                    assert mpo.boundary == copied_mpo.boundary
                    assert mpo.n_batches == copied_mpo.n_batches
                    
                    for node, copied_node in zip(mpo.mats_env, copied_mpo.mats_env):
                        if share_tensors:
                            assert node.tensor is copied_node.tensor
                        else: 
                            assert node.tensor is not copied_node.tensor
    
    def test_all_algorithms(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                # batch x n_features x feature_dim
                example = torch.randn(1, n_features, 2)
                data = torch.randn(100, n_features, 2)
                
                mpo = tk.models.MPO(n_features=n_features,
                                    in_dim=2,
                                    out_dim=2,
                                    bond_dim=10,
                                    boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mpo.auto_stack = auto_stack
                                mpo.auto_unbind = auto_unbind

                                mpo.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mpo(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == tuple([100] + [2] * n_features)
                                assert len(mpo.edges) == n_features
                                if boundary == 'obc':
                                    assert len(mpo.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mpo.leaf_nodes) == n_features
                                assert len(mpo.data_nodes) == n_features
                                if not inline_input and auto_stack:
                                    assert len(mpo.virtual_nodes) == 2
                                else:
                                    assert len(mpo.virtual_nodes) == 1
                                
                                result.sum().backward()
                                for node in mpo.mats_env:
                                    assert node.grad is not None
    
    def test_all_algorithms_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                # batch x n_features x feature_dim
                example = torch.randn(1, n_features, 2, device=device)
                data = torch.randn(100, n_features, 2, device=device)
                
                mpo = tk.models.MPO(n_features=n_features,
                                    in_dim=2,
                                    out_dim=2,
                                    bond_dim=10,
                                    boundary=boundary)
                mpo = mpo.to(device)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mpo.auto_stack = auto_stack
                                mpo.auto_unbind = auto_unbind

                                mpo.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mpo(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == tuple([100] + [2] * n_features)
                                assert len(mpo.edges) == n_features
                                if boundary == 'obc':
                                    assert len(mpo.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mpo.leaf_nodes) == n_features
                                assert len(mpo.data_nodes) == n_features
                                if not inline_input and auto_stack:
                                    assert len(mpo.virtual_nodes) == 2
                                else:
                                    assert len(mpo.virtual_nodes) == 1
                                
                                result.sum().backward()
                                for node in mpo.mats_env:
                                    assert node.grad is not None
    
    def test_mpo_mps_data_manually(self):
        mpo = tk.models.MPO(n_features=10,
                            in_dim=2,
                            out_dim=2,
                            bond_dim=10,
                            boundary='obc')
        mps_data = tk.models.MPSData(n_features=10,
                                     phys_dim=2,
                                     bond_dim=5)
        tensors = [torch.randn(100, 5, 2, 5) for _ in range(10)]
        tensors[0] = tensors[0][:, 0]
        tensors[-1] = tensors[-1][..., 0]
        mps_data.add_data(tensors)
            
        for mps_node, mpo_node in zip(mps_data.mats_env, mpo.mats_env):
            mps_node['feature'] ^ mpo_node['input']
        
        def contract():
            mps_nodes = mps_data.mats_env[:]
            mps_nodes[0] = mps_data.left_node @ mps_nodes[0]
            mps_nodes[-1] = mps_nodes[-1] @ mps_data.right_node
            
            mpo_nodes = mpo.mats_env[:]
            mpo_nodes[0] = mpo.left_node @ mpo_nodes[0]
            mpo_nodes[-1] = mpo_nodes[-1] @ mpo.right_node
            
            result = mpo_nodes[0]
            for i in range(mpo.n_features - 1):
                result = mps_nodes[i] @ result
                result = result @ mpo_nodes[i + 1]
            result = mps_nodes[-1] @ result
            
            return result
        
        mpo.contract = contract
        
        mpo.trace()
        result = mpo()
        
        assert result.shape == tuple([100] + [2] * 10)
    
    def test_mpo_mps_data_all_algorithms(self):
        for n_features in [1, 2, 4, 10]:
            for mpo_boundary in ['obc', 'pbc']:
                for mps_boundary in ['obc', 'pbc']:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            phys_dim = torch.randint(low=2, high=10,
                                                     size=(n_features,)).tolist()
                            bond_dim = torch.randint(low=2, high=8,
                                                     size=(n_features,)).tolist()
                            
                            mpo = tk.models.MPO(
                                n_features=n_features,
                                in_dim=phys_dim,
                                out_dim=2,
                                bond_dim=10,
                                boundary=mpo_boundary)
                            
                            mps_data = tk.models.MPSData(
                                n_features=n_features,
                                phys_dim=phys_dim,
                                bond_dim=bond_dim[:-1] \
                                    if mps_boundary == 'obc' else bond_dim,
                                boundary=mps_boundary)
                            
                            # To ensure MPSData nodes go to MPO network, and
                            # not the other way
                            mps_data.mats_env[0].move_to_network(mpo)
                            
                            for mps_node, mpo_node in zip(mps_data.mats_env,
                                                          mpo.mats_env):
                                mps_node['feature'] ^ mpo_node['input']
                            
                            for _ in range(10):
                                # MPO needs to be reset each time if inline_input
                                # or inline_mats are False, since the bond dims
                                # of MPSData may change, and therefore the
                                # computation of stacks may fail
                                if not inline_input or not inline_mats:
                                    mpo.reset()
                                
                                bond_dim = torch.randint(low=2, high=8,
                                                         size=(n_features,)).tolist()
                                tensors = [
                                    torch.randn(10,
                                                bond_dim[i - 1],
                                                phys_dim[i],
                                                bond_dim[i]) 
                                    for i in range(n_features)]
                                if mps_boundary == 'obc':
                                    tensors[0] = tensors[0][:, 0]
                                    tensors[-1] = tensors[-1][..., 0]
                                
                                mps_data.add_data(tensors)
                                result = mpo(mps=mps_data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)
                                
                                assert result.shape == tuple([10] + [2] * n_features)


class TestUMPO:
    
    def test_initialize_with_tensors(self):
        for n in [1, 2, 5]:
            # PBC
            tensor = torch.randn(10, 2, 10, 2)
            mpo = tk.models.UMPO(n_features=n,
                                 tensor=tensor)
            assert mpo.n_features == n
            assert mpo.boundary == 'pbc'
            assert mpo.in_dim == [2] * n
            assert mpo.out_dim == [2] * n
            assert mpo.bond_dim == [10] * n
    
    def test_initialize_with_tensors_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for n in [1, 2, 5]:
            # PBC
            tensor = torch.randn(10, 2, 10, 2, device=device)
            mpo = tk.models.UMPO(n_features=n,
                                 tensor=tensor)
            assert mpo.n_features == n
            assert mpo.boundary == 'pbc'
            assert mpo.in_dim == [2] * n
            assert mpo.out_dim == [2] * n
            assert mpo.bond_dim == [10] * n
            for node in mpo.mats_env:
                assert node.device == device
        
    def test_initialize_with_tensors_errors(self):
        # Tensor should be at most rank-4 tensor
        tensor = torch.randn(10, 2, 10, 2, 5)
        with pytest.raises(ValueError):
            mpo = tk.models.UMPO(n_features=5,
                                 tensor=tensor)
        
        # Bond dimensions should coincide
        tensor = torch.randn(10, 2, 7, 2)
        with pytest.raises(ValueError):
            mpo = tk.models.UMPO(n_features=5,
                                 tensor=tensor)
        
        # First and last bond dims should coincide
        tensors = torch.randn(5, 2, 3, 2)
        with pytest.raises(ValueError):
            mpo = tk.models.UMPO(n_features=1,
                                 tensor=tensor)
    
    def test_initialize_init_method(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mpo = tk.models.UMPO(n_features=n,
                                     in_dim=2,
                                     out_dim=2,
                                     bond_dim=10,
                                     init_method=init_method)
                assert mpo.n_features == n
                assert mpo.boundary == 'pbc'
                assert mpo.in_dim == [2] * n
                assert mpo.out_dim == [2] * n
                assert mpo.bond_dim == [10] * n
    
    def test_initialize_init_method_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mpo = tk.models.UMPO(n_features=n,
                                     in_dim=2,
                                     out_dim=2,
                                     bond_dim=10,
                                     init_method=init_method,
                                     device=device)
                assert mpo.n_features == n
                assert mpo.boundary == 'pbc'
                assert mpo.in_dim == [2] * n
                assert mpo.out_dim == [2] * n
                assert mpo.bond_dim == [10] * n
                for node in mpo.mats_env:
                    assert node.device == device
    
    def test_in_out_dims(self):
        for n_features in [1, 2, 3, 4, 6]:
            in_dim = torch.randint(low=2, high=10, size=(1,)).item()
            out_dim = torch.randint(low=2, high=10, size=(1,)).item()
            
            mpo = tk.models.UMPO(n_features=n_features,
                                 in_dim=in_dim,
                                 out_dim=out_dim,
                                 bond_dim=10)
            
            assert mpo.in_dim == [in_dim] * n_features
            assert mpo.out_dim == [out_dim] * n_features
    
    def test_in_out_dims_error(self):
        for n_features in [1, 2, 3, 4, 6]:
            in_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
            out_dim = torch.randint(low=2, high=10, size=(1,)).item()
            
            # in_dim should be int
            with pytest.raises(TypeError):
                mpo = tk.models.UMPO(n_features=n_features,
                                     in_dim=in_dim,
                                     out_dim=out_dim,
                                     bond_dim=10)
                
            in_dim = torch.randint(low=2, high=10, size=(1,)).item()
            out_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
            
            # out_dim should be int
            with pytest.raises(TypeError):
                mpo = tk.models.UMPO(n_features=n_features,
                                     in_dim=in_dim,
                                     out_dim=out_dim,
                                     bond_dim=10)
    
    def test_copy(self):
        for n_features in [1, 2, 3, 4, 6]:
            for share_tensors in [True, False]:
                in_dim = torch.randint(low=2, high=12, size=(1,)).item()
                out_dim = torch.randint(low=2, high=12, size=(1,)).item()
                bond_dim = torch.randint(low=2, high=10, size=(1,)).item()
                
                mpo = tk.models.UMPO(n_features=n_features,
                                     in_dim=in_dim,
                                     out_dim=out_dim,
                                     bond_dim=bond_dim,)
                
                copied_mpo = mpo.copy(share_tensors=share_tensors)
                
                assert isinstance(copied_mpo, tk.models.UMPO)
                assert mpo.n_features == copied_mpo.n_features
                assert mpo.in_dim == copied_mpo.in_dim
                assert mpo.out_dim == copied_mpo.out_dim
                assert mpo.bond_dim == copied_mpo.bond_dim
                assert mpo.boundary == copied_mpo.boundary
                assert mpo.n_batches == copied_mpo.n_batches
                
                for node, copied_node in zip(mpo.mats_env, copied_mpo.mats_env):
                    if share_tensors:
                        assert node.tensor is copied_node.tensor
                    else: 
                        assert node.tensor is not copied_node.tensor
    
    def test_all_algorithms(self):
        for n_features in [1, 2, 3, 4, 10]:
            # batch x n_features x feature_dim
            example = torch.randn(1, n_features, 2)
            data = torch.randn(100, n_features, 2)
            
            mpo = tk.models.UMPO(n_features=n_features,
                                 in_dim=2,
                                 out_dim=2,
                                 bond_dim=10)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            mpo.auto_stack = auto_stack
                            mpo.auto_unbind = auto_unbind

                            mpo.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats)
                            result = mpo(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats)

                            assert result.shape == tuple([100] + [2] * n_features)
                            assert len(mpo.edges) == n_features
                            assert len(mpo.leaf_nodes) == n_features
                            assert len(mpo.data_nodes) == n_features
                            assert len(mpo.virtual_nodes) == 2
                            
                            result.sum().backward()
                            for node in mpo.mats_env:
                                assert node.grad is not None
