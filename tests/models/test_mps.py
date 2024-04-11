"""
Tests for mps:

    * TestMPS
    * TestUMPS
    * TestMPSLayer
    * TestUMPSLayer
    * TestMPSData
    * TestConvModels
"""

import pytest

import torch
import tensorkrowch as tk


class TestMPS:  # MARK: TestMPS
    
    def test_initialize_with_tensors(self):
        for n in [1, 2, 5]:
            # PBC
            tensors = [torch.randn(10, 2, 10) for _ in range(n)]
            mps = tk.models.MPS(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * n
            
            # OBC
            tensors = [torch.randn(10, 2, 10) for _ in range(n)]
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]
            mps = tk.models.MPS(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * (n - 1)
    
    def test_initialize_with_tensors_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for n in [1, 2, 5]:
            # PBC
            tensors = [torch.randn(10, 2, 10, device=device) for _ in range(n)]
            mps = tk.models.MPS(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * n
            for node in mps.mats_env:
                assert node.device == device
            
            # OBC
            tensors = [torch.randn(10, 2, 10, device=device) for _ in range(n)]
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]
            mps = tk.models.MPS(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * (n - 1)
            for node in mps.mats_env:
                assert node.device == device
    
    def test_initialize_with_tensors_ignore_rest(self):
        tensors = [torch.randn(10, 2, 10) for _ in range(10)]
        mps = tk.models.MPS(tensors=tensors,
                            boundary='obc',
                            n_features=3,
                            phys_dim=4,
                            bond_dim=7)
        assert mps.boundary == 'pbc'
        assert mps.n_features == 10
        assert mps.phys_dim == [2] * 10
        assert mps.bond_dim == [10] * 10
        assert mps.in_features == list(range(10))
        
    def test_initialize_with_tensors_errors(self):
        # Tensors should be at most rank-3 tensors
        tensors = [torch.randn(10, 2, 10, 2) for _ in range(10)]
        with pytest.raises(ValueError):
            mps = tk.models.MPS(tensors=tensors)
        
        # First and last tensors should have the same rank
        tensors = [torch.randn(10, 2, 10) for _ in range(10)]
        tensors[0] = tensors[0][0]
        with pytest.raises(ValueError):
            mps = tk.models.MPS(tensors=tensors)
        
        # First and last bond dims should coincide
        tensors = [torch.randn(10, 2, 10) for _ in range(10)]
        tensors[0] = tensors[0][:5]
        tensors[-1] = tensors[-1][..., :3]
        with pytest.raises(ValueError):
            mps = tk.models.MPS(tensors=tensors)
    
    def test_initialize_init_method(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn', 'randn_eye']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mps = tk.models.MPS(boundary='pbc',
                                    n_features=n,
                                    phys_dim=2,
                                    bond_dim=2,
                                    init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [2] * n
                
                # OBC
                mps = tk.models.MPS(boundary='obc',
                                    n_features=n,
                                    phys_dim=2,
                                    bond_dim=2,
                                    init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'obc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [2] * (n - 1)
                
                assert torch.equal(mps.left_node.tensor[0],
                                   torch.ones_like(mps.left_node.tensor)[0])
                assert torch.equal(mps.left_node.tensor[1:],
                                   torch.zeros_like(mps.left_node.tensor)[1:])
                
                assert torch.equal(mps.right_node.tensor[0],
                                   torch.ones_like(mps.right_node.tensor)[0])
                assert torch.equal(mps.right_node.tensor[1:],
                                   torch.zeros_like(mps.right_node.tensor)[1:])
    
    def test_initialize_init_method_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn', 'randn_eye']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mps = tk.models.MPS(boundary='pbc',
                                    n_features=n,
                                    phys_dim=2,
                                    bond_dim=2,
                                    init_method=init_method,
                                    device=device)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [2] * n
                for node in mps.mats_env:
                    assert node.device == device
                
                # OBC
                mps = tk.models.MPS(boundary='obc',
                                    n_features=n,
                                    phys_dim=2,
                                    bond_dim=2,
                                    init_method=init_method,
                                    device=device)
                assert mps.n_features == n
                assert mps.boundary == 'obc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [2] * (n - 1)
                for node in mps.mats_env:
                    assert node.device == device
                
                assert torch.equal(mps.left_node.tensor[0],
                                   torch.ones_like(mps.left_node.tensor)[0])
                assert torch.equal(mps.left_node.tensor[1:],
                                   torch.zeros_like(mps.left_node.tensor)[1:])
                
                assert torch.equal(mps.right_node.tensor[0],
                                   torch.ones_like(mps.right_node.tensor)[0])
                assert torch.equal(mps.right_node.tensor[1:],
                                   torch.zeros_like(mps.right_node.tensor)[1:])
    
    def test_initialize_with_unitaries(self):
        for n in [1, 2, 5]:
            # PBC
            mps = tk.models.MPS(boundary='pbc',
                                n_features=n,
                                phys_dim=2,
                                bond_dim=2,
                                init_method='unit')
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * n
            
            # For PBC norm does not have to be 1. always
            
            # OBC
            mps = tk.models.MPS(boundary='obc',
                                n_features=n,
                                phys_dim=2,
                                bond_dim=2,
                                init_method='unit')
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * (n - 1)
            
            # Check it has norm == 1
            assert mps.norm().isclose(torch.tensor(1.))
            # Norm is close to 1. if bond dimension is <= than
            # physical dimension, otherwise, it will not be exactly 1.
    
    def test_initialize_with_unitaries_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for n in [1, 2, 5]:
            # PBC
            mps = tk.models.MPS(boundary='pbc',
                                n_features=n,
                                phys_dim=2,
                                bond_dim=2,
                                init_method='unit',
                                device=device)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * n
            for node in mps.mats_env:
                assert node.device == device
            
            # For PBC norm does not have to be 1. always
            
            # OBC
            mps = tk.models.MPS(boundary='obc',
                                n_features=n,
                                phys_dim=2,
                                bond_dim=2,
                                init_method='unit',
                                device=device)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * (n - 1)
            for node in mps.mats_env:
                assert node.device == device
            
            # Check it has norm == 1
            assert mps.norm().isclose(torch.tensor(1.))
            # Norm is close to 1. if bond dimension is <= than
            # physical dimension, otherwise, it will not be exactly 1.
    
    def test_in_and_out_features(self):
        tensors = [torch.randn(10, 2, 10) for _ in range(10)]
        mps = tk.models.MPS(tensors=tensors,
                            in_features=[0, 1, 4, 5])
        
        assert mps.in_features == [0, 1, 4, 5]
        assert mps.out_features == [2, 3, 6, 7, 8, 9]
        assert mps.in_regions == [[0, 1], [4, 5]]
        assert mps.out_regions == [[2, 3], [6, 7, 8, 9]]
        
        # Change output features affects input features
        mps.out_features = [0, 2, 2, 3, 7, 8]  # Ignores repeated elements
        assert mps.in_features == [1, 4, 5, 6, 9]
        assert mps.out_features == [0, 2, 3, 7, 8]
        assert mps.in_regions == [[1], [4, 5, 6], [9]]
        assert mps.out_regions == [[0], [2, 3], [7, 8]]
        
        # Raises error if in_features and out_features are not complementary
        with pytest.raises(ValueError):
            mps = tk.models.MPS(tensors=tensors,
                                in_features=[0, 1, 4, 5],
                                out_features=[0, 2, 3, 6, 7, 8, 9])
        
        # Raises error if in_features or out_features are out of range
        with pytest.raises(ValueError):
            mps = tk.models.MPS(tensors=tensors,
                                in_features=[0, 7, 15])
        with pytest.raises(ValueError):
            mps = tk.models.MPS(tensors=tensors,
                                out_features=[-1])
    
    def test_phys_dims(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                phys_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=phys_dim,
                                    bond_dim=10,
                                    boundary=boundary)
                
                assert mps.phys_dim == phys_dim
    
    def test_phys_dims_error(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                phys_dim = torch.randint(low=2, high=10, size=(n_features + 1,)).tolist()
                
                # phys_dim should have n_features elements
                with pytest.raises(ValueError):
                    mps = tk.models.MPS(n_features=n_features,
                                        phys_dim=phys_dim,
                                        bond_dim=10,
                                        boundary=boundary)
    
    def test_bond_dims(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=bond_dim,
                                    boundary=boundary)
                
                assert mps.phys_dim == [5] * n_features
                assert mps.bond_dim == bond_dim
                
                extended_bond_dim = [mps.mats_env[0].shape[0]] + \
                    [node.shape[-1] for node in mps.mats_env]
                
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
                    phys_dim = torch.randint(low=2,
                                             high=12,
                                             size=(n_features,)).tolist()
                    bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                    bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                    
                    mps = tk.models.MPS(n_features=n_features,
                                        phys_dim=phys_dim,
                                        bond_dim=bond_dim,
                                        boundary=boundary)
                    
                    copied_mps = mps.copy(share_tensors=share_tensors)
                    
                    assert isinstance(copied_mps, tk.models.MPS)
                    assert mps.n_features == copied_mps.n_features
                    assert mps.phys_dim == copied_mps.phys_dim
                    assert mps.bond_dim == copied_mps.bond_dim
                    assert mps.boundary == copied_mps.boundary
                    assert mps.in_features == copied_mps.in_features
                    assert mps.out_features == copied_mps.out_features
                    assert mps.n_batches == copied_mps.n_batches
                    
                    for node, copied_node in zip(mps.mats_env, copied_mps.mats_env):
                        if share_tensors:
                            assert node.tensor is copied_node.tensor
                        else: 
                            assert node.tensor is not copied_node.tensor
        
    def test_all_algorithms(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                example = torch.randn(1, n_features, 5) # batch x n_features x feature_dim
                data = torch.randn(100, n_features, 5)
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100,)
                                assert len(mps.edges) == 0
                                if boundary == 'obc':
                                    assert len(mps.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mps.leaf_nodes) == n_features
                                assert len(mps.data_nodes) == n_features
                                if not inline_input and auto_stack:
                                    assert len(mps.virtual_nodes) == 2
                                else:
                                    assert len(mps.virtual_nodes) == 1
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None
    
    def test_all_algorithms_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                # batch x n_features x feature_dim
                example = torch.randn(1, n_features, 5, device=device)
                data = torch.randn(100, n_features, 5, device=device)
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary)
                mps = mps.to(device)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100,)
                                assert len(mps.edges) == 0
                                if boundary == 'obc':
                                    assert len(mps.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mps.leaf_nodes) == n_features
                                assert len(mps.data_nodes) == n_features
                                if not inline_input and auto_stack:
                                    assert len(mps.virtual_nodes) == 2
                                else:
                                    assert len(mps.virtual_nodes) == 1
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None

    def test_all_algorithms_diff_in_dim(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                phys_dim = torch.randint(low=2,
                                         high=12,
                                         size=(n_features,)).tolist()
                # To ensure all dims are different
                while len(phys_dim) > len(set(phys_dim)):
                    phys_dim = torch.randint(low=2,
                                             high=10,
                                             size=(n_features,)).tolist()
                example = [torch.randn(1, d) for d in phys_dim]
                data = [torch.randn(100, d) for d in phys_dim]
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=phys_dim,
                                    bond_dim=2,
                                    boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100,)
                                assert len(mps.edges) == 0
                                if boundary == 'obc':
                                    assert len(mps.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mps.leaf_nodes) == n_features
                                assert len(mps.data_nodes) == n_features
                                if not inline_input and auto_stack:
                                    if n_features == 1:
                                        assert len(mps.virtual_nodes) == 2
                                    else:
                                        assert len(mps.virtual_nodes) == 1
                                else:
                                    if n_features == 1:
                                        assert len(mps.virtual_nodes) == 1
                                    else:
                                        assert len(mps.virtual_nodes) == 0
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None

    def test_all_algorithms_diff_bond_dim(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                example = torch.randn(1, n_features, 5)  # batch x n_features x feature_dim
                data = torch.randn(100, n_features, 5)
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=bond_dim,
                                    boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100,)
                                assert len(mps.edges) == 0
                                if boundary == 'obc':
                                    assert len(mps.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mps.leaf_nodes) == n_features
                                assert len(mps.data_nodes) == n_features
                                if not inline_input and auto_stack:
                                    assert len(mps.virtual_nodes) == 2
                                else:
                                    assert len(mps.virtual_nodes) == 1
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None

    def test_all_algorithms_diff_in_dim_bond_dim(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                phys_dim = torch.randint(low=2,
                                         high=12,
                                         size=(n_features,)).tolist()
                # To ensure all dims are different
                while len(phys_dim) > len(set(phys_dim)):
                    phys_dim = torch.randint(low=2,
                                             high=10,
                                             size=(n_features,)).tolist()
                bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                example = [torch.randn(1, d) for d in phys_dim]
                data = [torch.randn(100, d) for d in phys_dim]
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=phys_dim,
                                    bond_dim=bond_dim,
                                    boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100,)
                                assert len(mps.edges) == 0
                                
                                if boundary == 'obc':
                                    assert len(mps.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mps.leaf_nodes) == n_features
                                    
                                assert len(mps.data_nodes) == n_features
                                
                                if not inline_input and auto_stack:
                                    if n_features == 1:
                                        assert len(mps.virtual_nodes) == 2
                                    else:
                                        assert len(mps.virtual_nodes) == 1
                                else:
                                    if n_features == 1:
                                        assert len(mps.virtual_nodes) == 1
                                    else:
                                        assert len(mps.virtual_nodes) == 0
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None
    
    def test_all_algorithms_marginalize(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                example = torch.randn(1, n_features // 2, 5) # batch x n_features x feature_dim
                data = torch.randn(100, n_features // 2, 5)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats,
                                          marginalize_output=True)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats,
                                             marginalize_output=True)

                                if in_features:
                                    assert result.shape == (100,)
                                    
                                    if not inline_input and auto_stack:
                                        assert len(mps.virtual_nodes) == \
                                            (2 + len(mps.out_features))
                                    else:
                                        assert len(mps.virtual_nodes) == \
                                            (1 + len(mps.out_features))
                                        
                                else:
                                    assert result.shape == tuple()
                                    assert len(mps.virtual_nodes) == \
                                        len(mps.out_features)
                                
                                if boundary == 'obc':
                                    assert len(mps.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mps.leaf_nodes) == n_features
                                    
                                assert len(mps.data_nodes) == len(in_features)
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None
    
    def test_all_algorithms_marginalize_with_list_matrices(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                example = torch.randn(1, n_features // 2, 5) # batch x n_features x feature_dim
                data = torch.randn(100, n_features // 2, 5)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features)
                
                embedding_matrices = [torch.randn(5, 5)
                                      for _ in range(len(mps.out_features))]

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats,
                                          marginalize_output=True,
                                          embedding_matrices=embedding_matrices)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats,
                                             marginalize_output=True,
                                             embedding_matrices=embedding_matrices)

                                if in_features:
                                    assert result.shape == (100,)
                                    
                                    if not inline_input and auto_stack:
                                        assert len(mps.virtual_nodes) == \
                                            (2 + 2 * len(mps.out_features))
                                    else:
                                        assert len(mps.virtual_nodes) == \
                                            (1 + 2 * len(mps.out_features))
                                        
                                else:
                                    assert result.shape == tuple()
                                    assert len(mps.virtual_nodes) == \
                                        2 * len(mps.out_features)
                                
                                if boundary == 'obc':
                                    assert len(mps.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mps.leaf_nodes) == n_features
                                    
                                assert len(mps.data_nodes) == len(in_features)
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None
    
    def test_all_algorithms_marginalize_with_matrix(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                example = torch.randn(1, n_features // 2, 5) # batch x n_features x feature_dim
                data = torch.randn(100, n_features // 2, 5)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features)
                
                embedding_matrix = torch.randn(5, 5)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats,
                                          marginalize_output=True,
                                          embedding_matrices=embedding_matrix)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats,
                                             marginalize_output=True,
                                             embedding_matrices=embedding_matrix)

                                if in_features:
                                    assert result.shape == (100,)
                                    
                                    if not inline_input and auto_stack:
                                        assert len(mps.virtual_nodes) == \
                                            (2 + 2 * len(mps.out_features))
                                    else:
                                        assert len(mps.virtual_nodes) == \
                                            (1 + 2 * len(mps.out_features))
                                        
                                else:
                                    assert result.shape == tuple()
                                    assert len(mps.virtual_nodes) == \
                                        2 * len(mps.out_features)
                                
                                if boundary == 'obc':
                                    assert len(mps.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mps.leaf_nodes) == n_features
                                    
                                assert len(mps.data_nodes) == len(in_features)
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None
    
    def test_all_algorithms_marginalize_with_matrix_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                # batch x n_features x feature_dim
                example = torch.randn(1, n_features // 2, 5, device=device)
                data = torch.randn(100, n_features // 2, 5, device=device)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features)
                mps = mps.to(device)
                
                embedding_matrix = torch.randn(5, 5, device=device)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats,
                                          marginalize_output=True,
                                          embedding_matrices=embedding_matrix)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats,
                                             marginalize_output=True,
                                             embedding_matrices=embedding_matrix)

                                if in_features:
                                    assert result.shape == (100,)
                                    
                                    if not inline_input and auto_stack:
                                        assert len(mps.virtual_nodes) == \
                                            (2 + 2 * len(mps.out_features))
                                    else:
                                        assert len(mps.virtual_nodes) == \
                                            (1 + 2 * len(mps.out_features))
                                        
                                else:
                                    assert result.shape == tuple()
                                    assert len(mps.virtual_nodes) == \
                                        2 * len(mps.out_features)
                                
                                if boundary == 'obc':
                                    assert len(mps.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mps.leaf_nodes) == n_features
                                    
                                assert len(mps.data_nodes) == len(in_features)
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None
                                    
    def test_all_algorithms_no_marginalize(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                example = torch.randn(1, n_features // 2, 5) # batch x n_features x feature_dim
                data = torch.randn(100, n_features // 2, 5)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                aux_shape = [5] * len(mps.out_features)
                                if in_features:
                                    aux_shape = [100] + aux_shape
                                    assert result.shape == tuple(aux_shape)
                                    
                                    if not inline_input and auto_stack:
                                        assert len(mps.virtual_nodes) == 2
                                    else:
                                        assert len(mps.virtual_nodes) == 1
                                        
                                else:
                                    assert result.shape == tuple(aux_shape)
                                    assert len(mps.virtual_nodes) == 0
                                    
                                assert len(mps.edges) == len(mps.out_features)
                                
                                if boundary == 'obc':
                                    assert len(mps.leaf_nodes) == n_features + 2
                                else:
                                    assert len(mps.leaf_nodes) == n_features
                                    
                                assert len(mps.data_nodes) == len(in_features)
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None
    
    def test_norm(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                example = torch.randn(1, n_features // 2, 5) # batch x n_features x feature_dim
                if example.numel() == 0:
                    example = None
                
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                in_features.sort()
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features)
                mps.trace(example)
                
                assert mps.resultant_nodes
                if in_features:
                    assert mps.data_nodes
                assert mps.in_features == in_features
                
                # MPS has to be rese, otherwise norm automatically calls
                # the forward method that was traced when contracting the MPS
                # with example
                mps.reset()
                norm = mps.norm()
                assert mps.resultant_nodes
                assert not mps.data_nodes
                assert mps.in_features == []
                assert mps.out_features == list(range(n_features))
                
                norm.sum().backward()
                for node in mps.mats_env:
                    assert node.grad is not None
    
    def test_canonicalize(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                for oc in range(n_features):
                    for mode in ['svd', 'svdr', 'qr']:
                        for renormalize in [True, False]:
                            mps = tk.models.MPS(n_features=n_features,
                                                phys_dim=2,
                                                bond_dim=10,
                                                boundary=boundary,
                                                in_features=[])
                            
                            mps_tensor = mps()
                            assert mps_tensor.shape == (2,) * n_features
                            
                            mps.out_features = []
                            example = torch.randn(1, n_features, 2)
                            mps.trace(example)
                            
                            if mps.boundary == 'obc':
                                assert len(mps.leaf_nodes) == n_features + 2
                            else:
                                assert len(mps.leaf_nodes) == n_features
                            assert len(mps.data_nodes) == n_features
                            
                            # Canonicalize
                            rank = torch.randint(5, 11, (1,)).item()
                            mps.canonicalize(oc=oc,
                                             mode=mode,
                                             rank=rank,
                                             cum_percentage=0.98,
                                             cutoff=1e-5,
                                             renormalize=renormalize)
                            
                            if mps.bond_dim and mode != 'qr':
                                if mps.boundary == 'obc':
                                    assert (torch.tensor(mps.bond_dim) <= rank).all()
                                else:
                                    assert (torch.tensor(mps.bond_dim[:-1]) <= rank).all()
                            
                            if mps.boundary == 'obc':
                                assert len(mps.leaf_nodes) == n_features + 2
                            else:
                                assert len(mps.leaf_nodes) == n_features
                            assert len(mps.data_nodes) == n_features
                            
                            mps.unset_data_nodes()
                            mps.in_features = []
                            approx_mps_tensor = mps()
                            assert approx_mps_tensor.shape == (2,) * n_features
    
    def test_canonicalize_univocal(self):
        for n_features in [1, 2, 3, 4, 10]:
            mps = tk.models.MPS(n_features=n_features,
                                phys_dim=2,
                                bond_dim=10,
                                boundary='obc',
                                in_features=[],
                                init_method='unit')
            
            mps_tensor = mps()
            assert mps_tensor.shape == (2,) * n_features
            
            mps.out_features = []
            example = torch.randn(1, n_features, 2)
            mps.trace(example)
            
            assert len(mps.leaf_nodes) == n_features + 2
            assert len(mps.data_nodes) == n_features
            
            # Canonicalize
            mps.canonicalize_univocal()
            
            assert len(mps.leaf_nodes) == n_features + 2
            assert len(mps.data_nodes) == n_features
            
            mps.unset_data_nodes()
            mps.in_features = []
            approx_mps_tensor = mps()
            assert approx_mps_tensor.shape == (2,) * n_features
            
            assert torch.allclose(mps_tensor, approx_mps_tensor,
                                  rtol=1e-2, atol=1e-4)
    
    def test_canonicalize_univocal_diff_dims(self):
        for n_features in [1, 2, 3, 4, 10]:
            phys_dim = torch.arange(2, 2 + n_features).int().tolist()
            bond_dim = torch.arange(2, 1 + n_features).int().tolist()
            mps = tk.models.MPS(n_features=n_features,
                                phys_dim=phys_dim,
                                bond_dim=bond_dim,
                                boundary='obc',
                                in_features=[],
                                init_method='unit')
            
            mps_tensor = mps()
            assert mps_tensor.shape == tuple(phys_dim)
            
            mps.out_features = []
            example = [torch.randn(1, d) for d in phys_dim]
            mps.trace(example)
            
            assert len(mps.leaf_nodes) == n_features + 2
            assert len(mps.data_nodes) == n_features
            
            # Canonicalize
            mps.canonicalize_univocal()
            
            assert len(mps.leaf_nodes) == n_features + 2
            assert len(mps.data_nodes) == n_features
            
            mps.unset_data_nodes()
            mps.in_features = []
            approx_mps_tensor = mps()
            assert approx_mps_tensor.shape == tuple(phys_dim)
            
            assert torch.allclose(mps_tensor, approx_mps_tensor,
                                  rtol=1e-2, atol=1e-4)
    
    def test_canonicalize_univocal_bond_greater_than_phys(self):
        for n_features in [1, 2, 3, 4, 10]:
            mps = tk.models.MPS(n_features=n_features,
                                phys_dim=2,
                                bond_dim=100,
                                boundary='obc',
                                in_features=[],
                                init_method='unit')
            
            mps_tensor = mps()
            assert mps_tensor.shape == (2,) * n_features
            
            mps.out_features = []
            example = torch.randn(1, n_features, 2)
            mps.trace(example)
            
            assert len(mps.leaf_nodes) == n_features + 2
            assert len(mps.data_nodes) == n_features
            
            # Canonicalize
            mps.canonicalize_univocal()
            
            assert len(mps.leaf_nodes) == n_features + 2
            assert len(mps.data_nodes) == n_features
            
            mps.unset_data_nodes()
            mps.in_features = []
            approx_mps_tensor = mps()
            assert approx_mps_tensor.shape == (2,) * n_features
            
            assert torch.allclose(mps_tensor, approx_mps_tensor,
                                  rtol=1e-2, atol=1e-4)


class TestUMPS:  # MARK: TestUMPS
    
    def test_initialize_with_tensors(self):
        for n in [1, 2, 5]:
            tensor = torch.randn(10, 2, 10)
            mps = tk.models.UMPS(n_features=n,
                                 tensor=tensor)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * n
    
    def test_initialize_with_tensors_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for n in [1, 2, 5]:
            tensor = torch.randn(10, 2, 10, device=device)
            mps = tk.models.UMPS(n_features=n,
                                 tensor=tensor)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * n
            for node in mps.mats_env:
                assert node.device == device
    
    def test_initialize_with_tensors_errors(self):
        # Tensor should be at most rank-3 tensor
        tensor = torch.randn(10, 2, 7, 3)
        with pytest.raises(ValueError):
            mps = tk.models.UMPS(n_features=5,
                                 tensor=tensor)
        
        # Bond dimensions should coincide
        tensor = torch.randn(10, 2, 7)
        with pytest.raises(ValueError):
            mps = tk.models.UMPS(n_features=5,
                                 tensor=tensor)
        
        # First and last bond dims should coincide
        tensors = torch.randn(5, 2, 3)
        with pytest.raises(ValueError):
            mps = tk.models.UMPS(n_features=1,
                                 tensor=tensor)
    
    def test_initialize_init_method(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn', 'randn_eye']
        for n in [1, 2, 5]:
            for init_method in methods:
                mps = tk.models.UMPS(n_features=n,
                                     phys_dim=2,
                                     bond_dim=2,
                                     init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [2] * n
    
    def test_initialize_init_method_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn', 'randn_eye']
        for n in [1, 2, 5]:
            for init_method in methods:
                mps = tk.models.UMPS(n_features=n,
                                     phys_dim=2,
                                     bond_dim=2,
                                     init_method=init_method,
                                     device=device)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [2] * n
                for node in mps.mats_env:
                    assert node.device == device
    
    def test_initialize_with_unitaries(self):
        for n in [1, 2, 5]:
            mps = tk.models.UMPS(n_features=n,
                                 phys_dim=2,
                                 bond_dim=2,
                                 init_method='unit')
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * n
    
    def test_initialize_with_unitaries_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for n in [1, 2, 5]:
            mps = tk.models.UMPS(n_features=n,
                                 phys_dim=2,
                                 bond_dim=2,
                                 init_method='unit',
                                 device=device)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * n
            for node in mps.mats_env:
                assert node.device == device
    
    def test_in_and_out_features(self):
        tensor = torch.randn(10, 2, 10)
        mps = tk.models.UMPS(n_features=10,
                             tensor=tensor,
                             in_features=[0, 1, 4, 5])
        
        assert mps.in_features == [0, 1, 4, 5]
        assert mps.out_features == [2, 3, 6, 7, 8, 9]
        assert mps.in_regions == [[0, 1], [4, 5]]
        assert mps.out_regions == [[2, 3], [6, 7, 8, 9]]
        
        # Change output features affects input features
        mps.out_features = [0, 2, 3, 7, 8]
        assert mps.in_features == [1, 4, 5, 6, 9]
        assert mps.out_features == [0, 2, 3, 7, 8]
        assert mps.in_regions == [[1], [4, 5, 6], [9]]
        assert mps.out_regions == [[0], [2, 3], [7, 8]]
        
        # Raises error if in_features and out_features are not complementary
        with pytest.raises(ValueError):
            mps = tk.models.UMPS(n_features=10,
                                 tensor=tensor,
                                 in_features=[0, 1, 4, 5],
                                 out_features=[0, 2, 3, 6, 7, 8, 9])
        
        # Raises error if in_features or out_features are out of range
        with pytest.raises(ValueError):
            mps = tk.models.UMPS(n_features=10,
                                 tensor=tensor,
                                 in_features=[0, 7, 15])
        with pytest.raises(ValueError):
            mps = tk.models.UMPS(n_features=10,
                                 tensor=tensor,
                                 out_features=[-1])
    
    def test_copy(self):
        for n_features in [1, 2, 3, 4, 6]:
            for share_tensors in [True, False]:
                phys_dim = torch.randint(low=2,
                                         high=12,
                                         size=(1,)).item()
                bond_dim = torch.randint(low=2, high=10, size=(1,)).item()
                
                mps = tk.models.UMPS(n_features=n_features,
                                     phys_dim=phys_dim,
                                     bond_dim=bond_dim)
                
                copied_mps = mps.copy(share_tensors=share_tensors)
                
                assert isinstance(copied_mps, tk.models.UMPS)
                assert mps.n_features == copied_mps.n_features
                assert mps.phys_dim == copied_mps.phys_dim
                assert mps.bond_dim == copied_mps.bond_dim
                assert mps.boundary == copied_mps.boundary
                assert mps.in_features == copied_mps.in_features
                assert mps.out_features == copied_mps.out_features
                assert mps.n_batches == copied_mps.n_batches
                
                for node, copied_node in zip(mps.mats_env, copied_mps.mats_env):
                    if share_tensors:
                        assert node.tensor is copied_node.tensor
                    else: 
                        assert node.tensor is not copied_node.tensor

    def test_all_algorithms(self):
        for n_features in [1, 2, 3, 4, 10]:
            example = torch.randn(1, n_features, 5) # batch x n_features x feature_dim
            data = torch.randn(100, n_features, 5)
            
            mps = tk.models.UMPS(n_features=n_features,
                                 phys_dim=5,
                                 bond_dim=2)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats)

                            assert result.shape == (100,)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == n_features
                            assert len(mps.data_nodes) == n_features
                            assert len(mps.virtual_nodes) == 2
                            
                            result.sum().backward()
                            for node in mps.mats_env:
                                assert node.grad is not None
                            assert mps.uniform_memory.grad is not None
    
    def test_all_algorithms_marginalize(self):
        for n_features in [1, 2, 3, 4, 10]:
            example = torch.randn(1, n_features // 2, 5) # batch x n_features x feature_dim
            data = torch.randn(100, n_features // 2, 5)
            
            if example.numel() == 0:
                example = None
                data = None
            
            in_features = torch.randint(low=0,
                                        high=n_features,
                                        size=(n_features // 2,)).tolist()
            in_features = list(set(in_features))
            
            mps = tk.models.UMPS(n_features=n_features,
                                 phys_dim=5,
                                 bond_dim=2,
                                 in_features=in_features)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      marginalize_output=True)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         marginalize_output=True)

                            if in_features:
                                assert result.shape == (100,)
                                assert len(mps.virtual_nodes) == \
                                    (2 + len(mps.out_features))  
                            else:
                                assert result.shape == tuple()
                                assert len(mps.virtual_nodes) == \
                                    (1 + len(mps.out_features))
                            
                            assert len(mps.leaf_nodes) == n_features
                            assert len(mps.data_nodes) == len(in_features)
                            
                            result.sum().backward()
                            for node in mps.mats_env:
                                assert node.grad is not None
                            assert mps.uniform_memory.grad is not None
                                    
    def test_all_algorithms_no_marginalize(self):
        for n_features in [1, 2, 3, 4, 10]:
            example = torch.randn(1, n_features // 2, 5) # batch x n_features x feature_dim
            data = torch.randn(100, n_features // 2, 5)
            
            if example.numel() == 0:
                example = None
                data = None
            
            in_features = torch.randint(low=0,
                                        high=n_features,
                                        size=(n_features // 2,)).tolist()
            in_features = list(set(in_features))
            
            mps = tk.models.UMPS(n_features=n_features,
                                 phys_dim=5,
                                 bond_dim=2,
                                 in_features=in_features)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats)

                            aux_shape = [5] * len(mps.out_features)
                            if in_features:
                                aux_shape = [100] + aux_shape
                                assert result.shape == tuple(aux_shape)
                                assert len(mps.virtual_nodes) == 2    
                            else:
                                assert result.shape == tuple(aux_shape)
                                assert len(mps.virtual_nodes) == 1
                                
                            assert len(mps.edges) == len(mps.out_features)
                            assert len(mps.leaf_nodes) == n_features
                            assert len(mps.data_nodes) == len(in_features)
                            
                            result.sum().backward()
                            for node in mps.mats_env:
                                assert node.grad is not None
                            assert mps.uniform_memory.grad is not None
    
    def test_norm(self):
        for n_features in [1, 2, 3, 4, 10]:
            example = torch.randn(1, n_features // 2, 5) # batch x n_features x feature_dim
            if example.numel() == 0:
                example = None
            
            in_features = torch.randint(low=0,
                                        high=n_features,
                                        size=(n_features // 2,)).tolist()
            in_features = list(set(in_features))
            in_features.sort()
            
            mps = tk.models.UMPS(n_features=n_features,
                                 phys_dim=5,
                                 bond_dim=2,
                                 in_features=in_features)
            mps.trace(example)
            
            assert mps.resultant_nodes
            if in_features:
                assert mps.data_nodes
            assert mps.in_features == in_features
            
            # MPS has to be rese, otherwise norm automatically calls
            # the forward method that was traced when contracting the MPS
            # with example
            mps.reset()
            norm = mps.norm()
            assert mps.resultant_nodes
            assert not mps.data_nodes
            assert mps.in_features == []
            assert mps.out_features == list(range(n_features))
            
            norm.sum().backward()
            for node in mps.mats_env:
                assert node.grad is not None
            assert mps.uniform_memory.grad is not None
    
    def test_canonicalize_error(self):
        mps = tk.models.UMPS(n_features=10,
                             phys_dim=2,
                             bond_dim=10,
                             in_features=[])
        
        with pytest.raises(NotImplementedError):
            mps.canonicalize()
    
    def test_canonicalize_univocal_error(self):
        mps = tk.models.UMPS(n_features=10,
                             phys_dim=2,
                             bond_dim=10,
                             in_features=[])
        
        with pytest.raises(NotImplementedError):
            mps.canonicalize_univocal()


class TestMPSLayer:  # MARK: TestMPSLayer
    
    def test_initialize_with_tensors(self):
        for n in [1, 2, 5]:
            # PBC
            tensors = [torch.randn(10, 2, 10) for _ in range(n)]
            mps = tk.models.MPSLayer(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.in_dim == [2] * (n - 1)
            assert mps.out_dim == 2
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * n
            
            # OBC
            tensors = [torch.randn(10, 2, 10) for _ in range(n)]
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]
            mps = tk.models.MPSLayer(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.in_dim == [2] * (n - 1)
            assert mps.out_dim == 2
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * (n - 1)
    
    def test_initialize_with_tensors_ignore_rest(self):
        tensors = [torch.randn(10, 2, 10) for _ in range(10)]
        mps = tk.models.MPSLayer(tensors=tensors,
                                 boundary='obc',
                                 n_features=3,
                                 in_dim=4,
                                 out_dim=5,
                                 bond_dim=7)
        assert mps.boundary == 'pbc'
        assert mps.n_features == 10
        assert mps.in_dim == [2] * 9
        assert mps.out_dim == 2
        assert mps.phys_dim == [2] * 10
        assert mps.bond_dim == [10] * 10
    
    def test_initialize_init_method(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn', 'randn_eye']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mps = tk.models.MPSLayer(boundary='pbc',
                                         n_features=n,
                                         in_dim=2,
                                         out_dim=10,
                                         bond_dim=2,
                                         init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 10
                assert mps.bond_dim == [2] * n
                
                # OBC
                mps = tk.models.MPSLayer(boundary='obc',
                                         n_features=n,
                                         in_dim=2,
                                         out_dim=10,
                                         bond_dim=2,
                                         init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'obc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 10
                assert mps.bond_dim == [2] * (n - 1)
                
                assert torch.equal(mps.left_node.tensor[0],
                                   torch.ones_like(mps.left_node.tensor)[0])
                assert torch.equal(mps.left_node.tensor[1:],
                                   torch.zeros_like(mps.left_node.tensor)[1:])
                
                assert torch.equal(mps.right_node.tensor[0],
                                   torch.ones_like(mps.right_node.tensor)[0])
                assert torch.equal(mps.right_node.tensor[1:],
                                   torch.zeros_like(mps.right_node.tensor)[1:])
    
    def test_initialize_init_method_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn', 'randn_eye']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mps = tk.models.MPSLayer(boundary='pbc',
                                         n_features=n,
                                         in_dim=2,
                                         out_dim=10,
                                         bond_dim=2,
                                         init_method=init_method,
                                         device=device)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 10
                assert mps.bond_dim == [2] * n
                for node in mps.mats_env:
                    assert node.device == device
                
                # OBC
                mps = tk.models.MPSLayer(boundary='obc',
                                         n_features=n,
                                         in_dim=2,
                                         out_dim=10,
                                         bond_dim=2,
                                         init_method=init_method,
                                         device=device)
                assert mps.n_features == n
                assert mps.boundary == 'obc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 10
                assert mps.bond_dim == [2] * (n - 1)
                for node in mps.mats_env:
                    assert node.device == device
                
                assert torch.equal(mps.left_node.tensor[0],
                                   torch.ones_like(mps.left_node.tensor)[0])
                assert torch.equal(mps.left_node.tensor[1:],
                                   torch.zeros_like(mps.left_node.tensor)[1:])
                
                assert torch.equal(mps.right_node.tensor[0],
                                   torch.ones_like(mps.right_node.tensor)[0])
                assert torch.equal(mps.right_node.tensor[1:],
                                   torch.zeros_like(mps.right_node.tensor)[1:])
    
    def test_in_and_out_features(self):
        tensors = [torch.randn(10, 2, 10) for _ in range(10)]
        mps = tk.models.MPSLayer(tensors=tensors,
                                 out_position=5)
        
        assert mps.in_features == [0, 1, 2, 3, 4, 6, 7, 8, 9]
        assert mps.out_features == [5]
        assert mps.in_regions == [[0, 1, 2, 3, 4], [6, 7, 8, 9]]
        assert mps.out_regions == [[5]]
        
        # Change output features affects input features
        mps.out_features = [0, 2, 3, 7, 8]
        assert mps.in_features == [1, 4, 5, 6, 9]
        assert mps.out_features == [0, 2, 3, 7, 8]
        assert mps.in_regions == [[1], [4, 5, 6], [9]]
        assert mps.out_regions == [[0], [2, 3], [7, 8]]
        
        # But out_position is still 5
        assert mps.out_position == 5
    
    def test_in_out_dims(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                in_dim = torch.randint(low=2, high=10, size=(n_features - 1,)).tolist()
                
                mps = tk.models.MPSLayer(n_features=n_features,
                                         in_dim=in_dim,
                                         out_dim=2,
                                         bond_dim=10,
                                         boundary=boundary)
                
                out_position = n_features // 2
                assert mps.out_position == out_position
                
                assert mps.in_dim == in_dim
                assert mps.out_dim == 2
                assert mps.phys_dim == in_dim[:out_position] + [2] + in_dim[out_position:]
    
    def test_in_dims_error(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                in_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                
                # in_dim should have (n_features - 1) elements
                with pytest.raises(ValueError):
                    mps = tk.models.MPSLayer(n_features=n_features,
                                             in_dim=in_dim,
                                             out_dim=2,
                                             bond_dim=10,
                                             boundary=boundary)
    
    def test_copy(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                for share_tensors in [True, False]:
                    in_dim = torch.randint(low=2,
                                           high=12,
                                           size=(n_features - 1,)).tolist()
                    out_dim = torch.randint(low=2,
                                            high=12,
                                            size=(1,)).item()
                    bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                    bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                    
                    mps = tk.models.MPSLayer(n_features=n_features,
                                             in_dim=in_dim,
                                             out_dim=out_dim,
                                             bond_dim=bond_dim,
                                             boundary=boundary)
                    
                    copied_mps = mps.copy(share_tensors=share_tensors)
                    
                    assert isinstance(copied_mps, tk.models.MPSLayer)
                    assert mps.n_features == copied_mps.n_features
                    assert mps.phys_dim == copied_mps.phys_dim
                    assert mps.bond_dim == copied_mps.bond_dim
                    assert mps.boundary == copied_mps.boundary
                    assert mps.in_features == copied_mps.in_features
                    assert mps.out_features == copied_mps.out_features
                    assert mps.n_batches == copied_mps.n_batches
                    assert mps.out_position == copied_mps.out_position
                    
                    for node, copied_node in zip(mps.mats_env, copied_mps.mats_env):
                        if share_tensors:
                            assert node.tensor is copied_node.tensor
                        else: 
                            assert node.tensor is not copied_node.tensor


class TestUMPSLayer:  # MARK: TestUMPSLayer
    
    def test_initialize_with_tensors(self):
        for n in [1, 2, 5]:
            tensors = [torch.randn(10, 2, 10),  # uniform memory
                       torch.randn(10, 2, 10)]  # output tensor
            mps = tk.models.UMPSLayer(n_features=n,
                                      tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.in_dim == [2] * (n - 1)
            assert mps.out_dim == 2
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * n
            
            # Output node has a different tensor than the uniform tensor
            assert mps.out_node.tensor is not mps.uniform_memory.tensor
    
    def test_initialize_init_method(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn', 'randn_eye']
        for n in [1, 2, 5]:
            for init_method in methods:
                mps = tk.models.UMPSLayer(n_features=n,
                                          in_dim=2,
                                          out_dim=5,
                                          bond_dim=2,
                                          init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 5
                assert mps.bond_dim == [2] * n
                
                # Output node has a different tensor than the uniform tensor
                assert mps.out_node.tensor is not mps.uniform_memory.tensor
    
    def test_initialize_with_unitaries(self):
        for n in [1, 2, 5]:
            mps = tk.models.UMPSLayer(n_features=n,
                                      in_dim=2,
                                      out_dim=5,
                                      bond_dim=2,
                                      init_method='unit')
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.in_dim == [2] * (n - 1)
            assert mps.out_dim == 5
            assert mps.bond_dim == [2] * n
            
            # Output node has a different tensor than the uniform tensor
            assert mps.out_node.tensor is not mps.uniform_memory.tensor
    
    def test_in_and_out_features(self):
        tensors = [torch.randn(10, 2, 10),  # uniform memory
                   torch.randn(10, 2, 10)]  # output tensor
        mps = tk.models.UMPSLayer(n_features=10,
                                  tensors=tensors,
                                  out_position=5)
        
        assert mps.in_features == [0, 1, 2, 3, 4, 6, 7, 8, 9]
        assert mps.out_features == [5]
        assert mps.in_regions == [[0, 1, 2, 3, 4], [6, 7, 8, 9]]
        assert mps.out_regions == [[5]]
        
        # Change output features affects input features
        mps.out_features = [0, 2, 3, 7, 8]
        assert mps.in_features == [1, 4, 5, 6, 9]
        assert mps.out_features == [0, 2, 3, 7, 8]
        assert mps.in_regions == [[1], [4, 5, 6], [9]]
        assert mps.out_regions == [[0], [2, 3], [7, 8]]
        
        # But out_position is still 5
        assert mps.out_position == 5
        
        # Output node has a different tensor than the uniform tensor
        assert mps.out_node.tensor is not mps.uniform_memory.tensor
    
    def test_in_out_dims(self):
        for n_features in [2, 3, 4, 6]:
            in_dim = torch.randint(low=2, high=10, size=(n_features - 1,)).tolist()
            
            # in_dim should be int
            with pytest.raises(TypeError):
                mps = tk.models.UMPSLayer(n_features=n_features,
                                          in_dim=in_dim,
                                          out_dim=2,
                                          bond_dim=10)
    
    def test_copy(self):
        for n_features in [1, 2, 3, 4, 6]:
            for share_tensors in [True, False]:
                in_dim = torch.randint(low=2,
                                       high=12,
                                       size=(1,)).item()
                out_dim = torch.randint(low=2,
                                        high=12,
                                        size=(1,)).item()
                bond_dim = torch.randint(low=2, high=10, size=(1,)).item()
                
                mps = tk.models.UMPSLayer(n_features=n_features,
                                          in_dim=in_dim,
                                          out_dim=out_dim,
                                          bond_dim=bond_dim)
                
                copied_mps = mps.copy(share_tensors=share_tensors)
                
                assert isinstance(copied_mps, tk.models.UMPSLayer)
                assert mps.n_features == copied_mps.n_features
                assert mps.phys_dim == copied_mps.phys_dim
                assert mps.bond_dim == copied_mps.bond_dim
                assert mps.boundary == copied_mps.boundary
                assert mps.in_features == copied_mps.in_features
                assert mps.out_features == copied_mps.out_features
                assert mps.n_batches == copied_mps.n_batches
                assert mps.out_position == copied_mps.out_position
                
                for node, copied_node in zip(mps.mats_env, copied_mps.mats_env):
                    if share_tensors:
                        assert node.tensor is copied_node.tensor
                    else: 
                        assert node.tensor is not copied_node.tensor
    
    def test_canonicalize_error(self):
        mps = tk.models.UMPSLayer(n_features=10,
                                  in_dim=2,
                                  out_dim=2,
                                  bond_dim=10)
        
        with pytest.raises(NotImplementedError):
            mps.canonicalize()
    
    def test_canonicalize_univocal_error(self):
        mps = tk.models.UMPSLayer(n_features=10,
                                  in_dim=2,
                                  out_dim=2,
                                  bond_dim=10)
        
        with pytest.raises(NotImplementedError):
            mps.canonicalize_univocal()


class TestMPSData:   # MARK: TestMPSData
    
    def test_initialize_init_method(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn']
        for n_features in [1, 2, 5]:
            for boundary in ['obc', 'pbc']:
                for n_batches in [1, 2, 3]:
                    for init_method in methods:
                        mps = tk.models.MPSData(boundary=boundary,
                                                n_features=n_features,
                                                phys_dim=2,
                                                bond_dim=2,
                                                n_batches=n_batches,
                                                init_method=init_method)
                        
                        assert mps.n_features == n_features
                        assert mps.boundary == boundary
                        assert mps.phys_dim == [2] * n_features
                        
                        if boundary == 'obc':
                            assert mps.bond_dim == [2] * (n_features - 1)
                        else:
                            assert mps.bond_dim == [2] * n_features
                        
                        if (n_features == 1) and (boundary == 'obc'):
                            node = mps.mats_env[0]
                            assert node.shape == tuple([1] * n_batches + [1, 2, 1])
                        else:  
                            for node in mps.mats_env:
                                assert node.shape == tuple([1] * n_batches + [2] * 3)
    
    def test_initialize_add_data(self):
        for n_features in [1, 2, 5]:
            for boundary in ['obc', 'pbc']:
                for n_batches in [1, 2, 3]:
                    mps = tk.models.MPSData(boundary=boundary,
                                            n_features=n_features,
                                            phys_dim=2,
                                            bond_dim=5,
                                            n_batches=n_batches)
                    
                    tensors = [torch.randn(*([10] * n_batches),
                                           5, 2, 5) for _ in range(n_features)]
                    if boundary == 'obc':
                        tensors[0] = tensors[0][..., 0, :, :]
                        tensors[-1] = tensors[-1][..., 0]
                    
                    mps.add_data(tensors)
                    
                    assert mps.n_features == n_features
                    assert mps.boundary == boundary
                    assert mps.phys_dim == [2] * n_features
                    
                    if boundary == 'obc':
                        assert mps.bond_dim == [5] * (n_features - 1)
                    else:
                        assert mps.bond_dim == [5] * n_features
                    
                    if (n_features == 1) and (boundary == 'obc'):
                        node = mps.mats_env[0]
                        assert node.shape == tuple([10] * n_batches + [1, 2, 1])
                    else:  
                        for node in mps.mats_env:
                            assert node.shape == tuple([10] * n_batches + [5, 2, 5])


class TestConvModels:  # MARK: TestConvModels
    
    def test_conv_mps_all_algorithms(self):
        for height in [1, 2, 10]:
            for width in [1, 2, 10]:
                for boundary in ['obc', 'pbc']:
                    # batch  x in_channels x height x width
                    example = torch.randn(1, 5, height, width)
                    data = torch.randn(100, 5, height, width)
                    
                    mps = tk.models.ConvMPS(in_channels=5,
                                            bond_dim=2,
                                            kernel_size=(height, width),
                                            boundary=boundary)

                    for auto_stack in [True, False]:
                        for auto_unbind in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats)

                                    assert result.shape == (100, 1, 1)
                                    assert len(mps.edges) == 0
                                    if boundary == 'obc':
                                        assert len(mps.leaf_nodes) == height * width + 2
                                    else:
                                        assert len(mps.leaf_nodes) == height * width
                                    assert len(mps.data_nodes) == height * width
                                    if not inline_input and auto_stack:
                                        assert len(mps.virtual_nodes) == 2
                                    else:
                                        assert len(mps.virtual_nodes) == 1
                                    
                                    result.sum().backward()
                                    for node in mps.mats_env:
                                        assert node.grad is not None
    
    def test_conv_umps_all_algorithms(self):
        for height in [1, 2, 10]:
            for width in [1, 2, 10]:
                # batch  x in_channels x height x width
                example = torch.randn(1, 5, height, width)
                data = torch.randn(100, 5, height, width)
                
                mps = tk.models.ConvUMPS(in_channels=5,
                                         bond_dim=2,
                                         kernel_size=(height, width))

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100, 1, 1)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == height * width
                                assert len(mps.data_nodes) == height * width
                                assert len(mps.virtual_nodes) == 2
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None
                                assert mps.uniform_memory.grad is not None
    
    def test_conv_mps_layer_all_algorithms(self):
        for height in [1, 2, 10]:
            for width in [1, 2, 10]:
                for boundary in ['obc', 'pbc']:
                    # batch  x in_channels x height x width
                    example = torch.randn(1, 5, height, width)
                    data = torch.randn(100, 5, height, width)
                    
                    mps = tk.models.ConvMPSLayer(in_channels=5,
                                                 out_channels=10,
                                                 bond_dim=2,
                                                 kernel_size=(height, width),
                                                 boundary=boundary)

                    for auto_stack in [True, False]:
                        for auto_unbind in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats)

                                    assert result.shape == (100, 10, 1, 1)
                                    assert len(mps.edges) == 1
                                    if boundary == 'obc':
                                        assert len(mps.leaf_nodes) == height * width + 3
                                    else:
                                        assert len(mps.leaf_nodes) == height * width + 1
                                    assert len(mps.data_nodes) == height * width
                                    if not inline_input and auto_stack:
                                        assert len(mps.virtual_nodes) == 2
                                    else:
                                        assert len(mps.virtual_nodes) == 1
                                    
                                    result.sum().backward()
                                    for node in mps.mats_env:
                                        assert node.grad is not None
    
    def test_conv_umps_layer_all_algorithms(self):
        for height in [1, 2, 10]:
            for width in [1, 2, 10]:
                # batch  x in_channels x height x width
                example = torch.randn(1, 5, height, width)
                data = torch.randn(100, 5, height, width)
                
                mps = tk.models.ConvUMPSLayer(in_channels=5,
                                              out_channels=10,
                                              bond_dim=2,
                                              kernel_size=(height, width))

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100, 10, 1, 1)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == height * width + 1
                                assert len(mps.data_nodes) == height * width
                                assert len(mps.virtual_nodes) == 2
                                
                                result.sum().backward()
                                for node in mps.mats_env:
                                    assert node.grad is not None
                                assert mps.uniform_memory.grad is not None
    
    def test_copy_conv_mps(self):
        for height in [1, 2, 10]:
            for width in [1, 2, 10]:
                for boundary in ['obc', 'pbc']:
                    for share_tensors in [True, False]:
                        phys_dim = torch.randint(low=2, high=12,
                                                 size=(height * width,)).tolist()
                        bond_dim = torch.randint(low=2, high=10,
                                                 size=(height * width,)).tolist()
                        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                        
                        mps = tk.models.ConvMPS(in_channels=phys_dim,
                                                bond_dim=bond_dim,
                                                kernel_size=(height, width),
                                                boundary=boundary)
                        
                        copied_mps = mps.copy(share_tensors=share_tensors)
                        
                        assert isinstance(copied_mps, tk.models.ConvMPS)
                        assert mps.n_features == copied_mps.n_features
                        assert mps.phys_dim == copied_mps.phys_dim
                        assert mps.bond_dim == copied_mps.bond_dim
                        assert mps.boundary == copied_mps.boundary
                        assert mps.in_features == copied_mps.in_features
                        assert mps.out_features == copied_mps.out_features
                        assert mps.n_batches == copied_mps.n_batches
                        
                        for node, copied_node in zip(mps.mats_env, copied_mps.mats_env):
                            if share_tensors:
                                assert node.tensor is copied_node.tensor
                            else: 
                                assert node.tensor is not copied_node.tensor
    
    def test_copy_conv_umps(self):
        for height in [1, 2, 10]:
            for width in [1, 2, 10]:
                for share_tensors in [True, False]:
                    phys_dim = torch.randint(low=2, high=12, size=(1,)).item()
                    bond_dim = torch.randint(low=2, high=10, size=(1,)).item()
                    
                    mps = tk.models.ConvUMPS(in_channels=phys_dim,
                                             bond_dim=bond_dim,
                                             kernel_size=(height, width))
                    
                    copied_mps = mps.copy(share_tensors=share_tensors)
                    
                    assert isinstance(copied_mps, tk.models.ConvUMPS)
                    assert mps.n_features == copied_mps.n_features
                    assert mps.phys_dim == copied_mps.phys_dim
                    assert mps.bond_dim == copied_mps.bond_dim
                    assert mps.boundary == copied_mps.boundary
                    assert mps.in_features == copied_mps.in_features
                    assert mps.out_features == copied_mps.out_features
                    assert mps.n_batches == copied_mps.n_batches
                    
                    for node, copied_node in zip(mps.mats_env, copied_mps.mats_env):
                        if share_tensors:
                            assert node.tensor is copied_node.tensor
                        else: 
                            assert node.tensor is not copied_node.tensor
    
    def test_copy_conv_mps_layer(self):
        for height in [1, 2, 10]:
            for width in [1, 2, 10]:
                for boundary in ['obc', 'pbc']:
                    for share_tensors in [True, False]:
                        phys_dim = torch.randint(low=2, high=12,
                                                 size=(height * width,)).tolist()
                        bond_dim = torch.randint(low=2, high=10,
                                                 size=(height * width + 1,)).tolist()
                        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                        
                        mps = tk.models.ConvMPSLayer(in_channels=phys_dim,
                                                     out_channels=10,
                                                     bond_dim=bond_dim,
                                                     kernel_size=(height, width),
                                                     boundary=boundary)
                        
                        copied_mps = mps.copy(share_tensors=share_tensors)
                        
                        assert isinstance(copied_mps, tk.models.ConvMPSLayer)
                        assert mps.n_features == copied_mps.n_features
                        assert mps.phys_dim == copied_mps.phys_dim
                        assert mps.bond_dim == copied_mps.bond_dim
                        assert mps.boundary == copied_mps.boundary
                        assert mps.in_features == copied_mps.in_features
                        assert mps.out_features == copied_mps.out_features
                        assert mps.n_batches == copied_mps.n_batches
                        
                        for node, copied_node in zip(mps.mats_env, copied_mps.mats_env):
                            if share_tensors:
                                assert node.tensor is copied_node.tensor
                            else: 
                                assert node.tensor is not copied_node.tensor
    
    def test_copy_conv_umps_layer(self):
        for height in [1, 2, 10]:
            for width in [1, 2, 10]:
                for share_tensors in [True, False]:
                    phys_dim = torch.randint(low=2, high=12, size=(1,)).item()
                    bond_dim = torch.randint(low=2, high=10, size=(1,)).item()
                    
                    mps = tk.models.ConvUMPSLayer(in_channels=phys_dim,
                                                  out_channels=10,
                                                  bond_dim=bond_dim,
                                                  kernel_size=(height, width))
                    
                    copied_mps = mps.copy(share_tensors=share_tensors)
                    
                    assert isinstance(copied_mps, tk.models.ConvUMPSLayer)
                    assert mps.n_features == copied_mps.n_features
                    assert mps.phys_dim == copied_mps.phys_dim
                    assert mps.bond_dim == copied_mps.bond_dim
                    assert mps.boundary == copied_mps.boundary
                    assert mps.in_features == copied_mps.in_features
                    assert mps.out_features == copied_mps.out_features
                    assert mps.n_batches == copied_mps.n_batches
                    
                    for node, copied_node in zip(mps.mats_env, copied_mps.mats_env):
                        if share_tensors:
                            assert node.tensor is copied_node.tensor
                        else: 
                            assert node.tensor is not copied_node.tensor
        


# TODO: test if I have 2 nodes from 2 networks, and set a.tensor = b.tensor,
# the tensor is not copied, they are sharing the same tensor. If it is a Parameter,
# I can compute gradients with respect to 1 node but update both

# import tensorkrowch as tk

# a = tk.randn((2,2)).parameterize()
# b = tk.randn((2,2)).parameterize()

# b.tensor = a.tensor

# a, b

# opt = optim.SGD(a.network.parameters())

# a.sum().backward()
# opt.step()

# a, b


# TODO: test if I connect two MPS, names will get a _0, _1, appended. When I
# disconnect both networks, and send each of them to a different network,
# names return to its original naming

# TODO: test, if I connect two MPs and then disconnect them, they still live in
# the same MPS, I have to send one of them back to its original network object
# Override reset to introduce the case when MPS is connected to another MPS