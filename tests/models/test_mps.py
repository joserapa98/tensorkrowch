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
    
    def test_initialize_with_tensors_complex(self):
        for n in [1, 2, 5]:
            # PBC
            tensors = [torch.randn(10, 2, 10,
                                   dtype=torch.complex64) for _ in range(n)]
            mps = tk.models.MPS(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * n
            for node in mps.mats_env:
                assert node.dtype == torch.complex64
                assert node.is_complex()
            
            # OBC
            tensors = [torch.randn(10, 2, 10,
                                   dtype=torch.complex64) for _ in range(n)]
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]
            mps = tk.models.MPS(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * (n - 1)
            for node in mps.mats_env:
                assert node.dtype == torch.complex64
                assert node.is_complex()
    
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
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn',
                   'randn_eye', 'unit', 'canonical']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mps = tk.models.MPS(boundary='pbc',
                                    n_features=n,
                                    phys_dim=2,
                                    bond_dim=5,
                                    init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [5] * n
                
                # OBC
                mps = tk.models.MPS(boundary='obc',
                                    n_features=n,
                                    phys_dim=2,
                                    bond_dim=5,
                                    init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'obc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [5] * (n - 1)
                
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
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn',
                   'randn_eye', 'unit', 'canonical']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mps = tk.models.MPS(boundary='pbc',
                                    n_features=n,
                                    phys_dim=2,
                                    bond_dim=5,
                                    init_method=init_method,
                                    device=device)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [5] * n
                for node in mps.mats_env:
                    assert node.device == device
                
                # OBC
                mps = tk.models.MPS(boundary='obc',
                                    n_features=n,
                                    phys_dim=2,
                                    bond_dim=5,
                                    init_method=init_method,
                                    device=device)
                assert mps.n_features == n
                assert mps.boundary == 'obc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [5] * (n - 1)
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
    
    def test_initialize_init_method_complex(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn',
                   'randn_eye', 'unit', 'canonical']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mps = tk.models.MPS(boundary='pbc',
                                    n_features=n,
                                    phys_dim=2,
                                    bond_dim=5,
                                    init_method=init_method,
                                    dtype=torch.complex64)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [5] * n
                for node in mps.mats_env:
                    assert node.dtype == torch.complex64
                    assert node.is_complex()
                
                # OBC
                mps = tk.models.MPS(boundary='obc',
                                    n_features=n,
                                    phys_dim=2,
                                    bond_dim=5,
                                    init_method=init_method,
                                    dtype=torch.complex64)
                assert mps.n_features == n
                assert mps.boundary == 'obc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [5] * (n - 1)
                for node in mps.mats_env:
                    assert node.dtype == torch.complex64
                    assert node.is_complex()
                
                assert torch.equal(mps.left_node.tensor[0],
                                   torch.ones_like(mps.left_node.tensor)[0])
                assert torch.equal(mps.left_node.tensor[1:],
                                   torch.zeros_like(mps.left_node.tensor)[1:])
                
                assert torch.equal(mps.right_node.tensor[0],
                                   torch.ones_like(mps.right_node.tensor)[0])
                assert torch.equal(mps.right_node.tensor[1:],
                                   torch.zeros_like(mps.right_node.tensor)[1:])
    
    def test_initialize_canonical(self):
        for n in [1, 2, 5]:
            # PBC
            mps = tk.models.MPS(boundary='pbc',
                                n_features=n,
                                phys_dim=2,
                                bond_dim=2,
                                init_method='canonical')
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * n
            
            # For PBC norm does not have to be 2**n always
            
            # OBC
            mps = tk.models.MPS(boundary='obc',
                                n_features=n,
                                phys_dim=2,
                                bond_dim=2,
                                init_method='canonical')
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * (n - 1)
            
            # Check it has norm == 2**n
            norm = mps.norm()
            assert mps.norm().isclose(torch.tensor(2. ** n).sqrt())
            # Norm is close to 2**n if bond dimension is <= than
            # physical dimension, otherwise, it will not be exactly 2**n
    
    def test_initialize_canonical_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n in [1, 2, 5]:
            # PBC
            mps = tk.models.MPS(boundary='pbc',
                                n_features=n,
                                phys_dim=2,
                                bond_dim=2,
                                init_method='canonical',
                                device=device)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * n
            for node in mps.mats_env:
                assert node.device == device
            
            # For PBC norm does not have to be 2**n always
            
            # OBC
            mps = tk.models.MPS(boundary='obc',
                                n_features=n,
                                phys_dim=2,
                                bond_dim=2,
                                init_method='canonical',
                                device=device)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * (n - 1)
            for node in mps.mats_env:
                assert node.device == device
            
            # Check it has norm == 2**n
            assert mps.norm().isclose(torch.tensor(2. ** n).sqrt())
            # Norm is close to 2**n if bond dimension is <= than
            # physical dimension, otherwise, it will not be exactly 2**n
    
    def test_initialize_canonical_complex(self):
        for n in [1, 2, 5]:
            # PBC
            mps = tk.models.MPS(boundary='pbc',
                                n_features=n,
                                phys_dim=2,
                                bond_dim=2,
                                init_method='canonical',
                                dtype=torch.complex64)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * n
            for node in mps.mats_env:
                assert node.dtype == torch.complex64
                assert node.is_complex()
            
            # For PBC norm does not have to be 2**n always
            
            # OBC
            mps = tk.models.MPS(boundary='obc',
                                n_features=n,
                                phys_dim=2,
                                bond_dim=2,
                                init_method='canonical',
                                dtype=torch.complex64)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [2] * (n - 1)
            for node in mps.mats_env:
                assert node.dtype == torch.complex64
                assert node.is_complex()
            
            # Check it has norm == 2**n
            assert mps.norm().isclose(torch.tensor(2. ** n).sqrt())
            # Norm is close to 2**n if bond dimension is <= than
            # physical dimension, otherwise, it will not be exactly 2**n
    
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
                    phys_dim = torch.randint(low=2, high=12,
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
    
    def test_deparameterize(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                for override in [True, False]:
                    phys_dim = torch.randint(low=2, high=12,
                                             size=(n_features,)).tolist()
                    bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                    bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                    
                    mps = tk.models.MPS(n_features=n_features,
                                        phys_dim=phys_dim,
                                        bond_dim=bond_dim,
                                        boundary=boundary)
                    
                    non_param_mps = mps.parameterize(set_param=False,
                                                     override=override)
                    
                    if override:
                        assert non_param_mps is mps
                    else:
                        assert non_param_mps is not mps
                    
                    new_nodes = non_param_mps.mats_env
                    if boundary == 'obc':
                        new_nodes += [non_param_mps.left_node,
                                      non_param_mps.right_node]
                        
                    for node in new_nodes:
                        assert isinstance(node, tk.Node)
                        assert not isinstance(node.tensor, torch.nn.Parameter)
    
    def test_update_bond_dim(self):
        mps = tk.models.MPS(n_features=100,
                            phys_dim=2,
                            bond_dim=10,
                            boundary='obc',
                            init_method='randn')
        
        mps.canonicalize(rank=3, renormalize=True)
        assert mps.bond_dim == [3] * 99
        assert (mps.left_node.tensor == torch.tensor([1., 0., 0.])).all()
        assert (mps.right_node.tensor == torch.tensor([1., 0., 0.])).all()
        
        mps.canonicalize(rank=5, renormalize=True)
        assert mps.bond_dim == [5] * 99
        assert (mps.left_node.tensor == torch.tensor([1., 0., 0. , 0., 0.])).all()
        assert (mps.right_node.tensor == torch.tensor([1., 0., 0. , 0., 0.])).all()
        
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
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize)

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
                                    boundary=boundary,
                                    device=device)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize)

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
    
    def test_all_algorithms_complex(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                # batch x n_features x feature_dim
                example = torch.randn(1, n_features, 5, dtype=torch.complex64)
                data = torch.randn(100, n_features, 5, dtype=torch.complex64)
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    dtype=torch.complex64)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize)

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
                                    
                                    result.sum().abs().backward()
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
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize)

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
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize)

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
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize)

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
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5)
                data = torch.randn(100, len(in_features), 5)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize,
                                              marginalize_output=True)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize,
                                                 marginalize_output=True)

                                    if in_features:
                                        assert result.shape == (100, 100)
                                        
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
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5)
                data = torch.randn(100, len(in_features), 5)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
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
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize,
                                              marginalize_output=True,
                                              embedding_matrices=embedding_matrices)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize,
                                                 marginalize_output=True,
                                                 embedding_matrices=embedding_matrices)

                                    if in_features:
                                        assert result.shape == (100, 100)
                                        
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
    
    def test_all_algorithms_marginalize_with_list_matrices_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5, device=device)
                data = torch.randn(100, len(in_features), 5, device=device)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features,
                                    device=device)
                
                embedding_matrices = [torch.randn(5, 5, device=device)
                                      for _ in range(len(mps.out_features))]

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize,
                                              marginalize_output=True,
                                              embedding_matrices=embedding_matrices)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize,
                                                 marginalize_output=True,
                                                 embedding_matrices=embedding_matrices)

                                    if in_features:
                                        assert result.shape == (100, 100)
                                        
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
    
    def test_all_algorithms_marginalize_with_list_matrices_complex(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5, dtype=torch.complex64)
                data = torch.randn(100, len(in_features), 5, dtype=torch.complex64)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features,
                                    dtype=torch.complex64)
                
                embedding_matrices = [torch.randn(5, 5, dtype=torch.complex64)
                                      for _ in range(len(mps.out_features))]

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize,
                                              marginalize_output=True,
                                              embedding_matrices=embedding_matrices)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize,
                                                 marginalize_output=True,
                                                 embedding_matrices=embedding_matrices)

                                    if in_features:
                                        assert result.shape == (100, 100)
                                        
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
                                    
                                    result.sum().abs().backward()
                                    for node in mps.mats_env:
                                        assert node.grad is not None
    
    def test_all_algorithms_marginalize_with_matrix(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5)
                data = torch.randn(100, len(in_features), 5)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
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
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize,
                                              marginalize_output=True,
                                              embedding_matrices=embedding_matrix)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize,
                                                 marginalize_output=True,
                                                 embedding_matrices=embedding_matrix)

                                    if in_features:
                                        assert result.shape == (100, 100)
                                        
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
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5, device=device)
                data = torch.randn(100, len(in_features), 5, device=device)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features,
                                    device=device)
                
                embedding_matrix = torch.randn(5, 5, device=device)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize,
                                              marginalize_output=True,
                                              embedding_matrices=embedding_matrix)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize,
                                                 marginalize_output=True,
                                                 embedding_matrices=embedding_matrix)

                                    if in_features:
                                        assert result.shape == (100, 100)
                                        
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
    
    def test_all_algorithms_marginalize_with_matrix_complex(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5, dtype=torch.complex64)
                data = torch.randn(100, len(in_features), 5, dtype=torch.complex64)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features,
                                    dtype=torch.complex64)
                
                embedding_matrix = torch.randn(5, 5, dtype=torch.complex64)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize,
                                              marginalize_output=True,
                                              embedding_matrices=embedding_matrix)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize,
                                                 marginalize_output=True,
                                                 embedding_matrices=embedding_matrix)

                                    if in_features:
                                        assert result.shape == (100, 100)
                                        
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
                                    
                                    result.sum().abs().backward()
                                    for node in mps.mats_env:
                                        assert node.grad is not None
    
    def test_all_algorithms_marginalize_with_mpo(self):
        for n_features in [1, 2, 3, 4, 10]:
            for mps_boundary in ['obc', 'pbc']:
                for mpo_boundary in ['obc', 'pbc']:
                    in_features = torch.randint(low=0,
                                                high=n_features,
                                                size=(n_features // 2,)).tolist()
                    in_features = list(set(in_features))
                    
                    # batch x n_features x feature_dim
                    example = torch.randn(1, len(in_features), 5)
                    data = torch.randn(100, len(in_features), 5)
                    
                    if example.numel() == 0:
                        example = None
                        data = None
                    
                    mps = tk.models.MPS(n_features=n_features,
                                        phys_dim=5,
                                        bond_dim=2,
                                        boundary=mps_boundary,
                                        in_features=in_features)
                    
                    mpo = tk.models.MPO(n_features=n_features - len(in_features),
                                        in_dim=5,
                                        out_dim=5,
                                        bond_dim=2,
                                        boundary=mpo_boundary)
                    
                    # De-parameterize MPO nodes to only train MPS nodes
                    mpo = mpo.parameterize(set_param=False, override=True)

                    for auto_stack in [True, False]:
                        for auto_unbind in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    for renormalize in [True, False]:
                                        mps.auto_stack = auto_stack
                                        mps.auto_unbind = auto_unbind

                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats,
                                                  renormalize=renormalize,
                                                  marginalize_output=True,
                                                  mpo=mpo)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats,
                                                     renormalize=renormalize,
                                                     marginalize_output=True,
                                                     mpo=mpo)
                                        
                                        if in_features:
                                            assert result.shape == (100, 100)
                                        else:
                                            assert result.shape == tuple()
                                        
                                        if mps_boundary == 'obc':
                                            if mpo_boundary == 'obc':
                                                leaf = (n_features + 2) + \
                                                    (n_features - len(in_features) + 2)
                                                assert len(mps.leaf_nodes) == leaf
                                            else:
                                                leaf = (n_features + 2) + \
                                                    (n_features - len(in_features))
                                                assert len(mps.leaf_nodes) == leaf
                                        else:
                                            if mpo_boundary == 'obc':
                                                leaf = n_features + \
                                                    (n_features - len(in_features) + 2)
                                                assert len(mps.leaf_nodes) == leaf
                                            else:
                                                leaf = n_features + \
                                                    (n_features - len(in_features))
                                                assert len(mps.leaf_nodes) == leaf
                                        
                                        result.sum().backward()
                                        for node in mps.mats_env:
                                            assert node.grad is not None
                                        for node in mpo.mats_env:
                                            assert node.tensor.grad is None
    
    def test_all_algorithms_marginalize_with_mpo_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 10]:
            for mps_boundary in ['obc', 'pbc']:
                for mpo_boundary in ['obc', 'pbc']:
                    in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5, device=device)
                data = torch.randn(100, len(in_features), 5, device=device)
                
                if example.numel() == 0:
                    example = None
                    data = None
                    
                    mps = tk.models.MPS(n_features=n_features,
                                        phys_dim=5,
                                        bond_dim=2,
                                        boundary=mps_boundary,
                                        in_features=in_features,
                                        device=device)
                    
                    mpo = tk.models.MPO(n_features=n_features - len(in_features),
                                        in_dim=5,
                                        out_dim=5,
                                        bond_dim=2,
                                        boundary=mpo_boundary,
                                        device=device)
                    
                    # Send mpo to cuda before deparameterizing, so that all
                    # nodes are still in the state_dict of the model and are
                    # automatically sent to cuda
                    # mpo = mpo.to(device)
                    
                    # De-parameterize MPO nodes to only train MPS nodes
                    mpo = mpo.parameterize(set_param=False, override=True)

                    for auto_stack in [True, False]:
                        for auto_unbind in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    for renormalize in [True, False]:
                                        mps.auto_stack = auto_stack
                                        mps.auto_unbind = auto_unbind

                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats,
                                                  renormalize=renormalize,
                                                  marginalize_output=True,
                                                  mpo=mpo)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats,
                                                     renormalize=renormalize,
                                                     marginalize_output=True,
                                                     mpo=mpo)
                                        
                                        if in_features:
                                            assert result.shape == (100, 100)
                                        else:
                                            assert result.shape == tuple()
                                        
                                        if mps_boundary == 'obc':
                                            if mpo_boundary == 'obc':
                                                leaf = (n_features + 2) + \
                                                    (n_features - len(in_features) + 2)
                                                assert len(mps.leaf_nodes) == leaf
                                            else:
                                                leaf = (n_features + 2) + \
                                                    (n_features - len(in_features))
                                                assert len(mps.leaf_nodes) == leaf
                                        else:
                                            if mpo_boundary == 'obc':
                                                leaf = n_features + \
                                                    (n_features - len(in_features) + 2)
                                                assert len(mps.leaf_nodes) == leaf
                                            else:
                                                leaf = n_features + \
                                                    (n_features - len(in_features))
                                                assert len(mps.leaf_nodes) == leaf
                                        
                                        result.sum().backward()
                                        for node in mps.mats_env:
                                            assert node.grad is not None
                                        for node in mpo.mats_env:
                                            assert node.tensor.grad is None
    
    def test_all_algorithms_marginalize_with_mpo_complex(self):
        for n_features in [1, 2, 3, 4, 10]:
            for mps_boundary in ['obc', 'pbc']:
                for mpo_boundary in ['obc', 'pbc']:
                    in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5, dtype=torch.complex64)
                data = torch.randn(100, len(in_features), 5, dtype=torch.complex64)
                
                if example.numel() == 0:
                    example = None
                    data = None
                    
                    mps = tk.models.MPS(n_features=n_features,
                                        phys_dim=5,
                                        bond_dim=2,
                                        boundary=mps_boundary,
                                        in_features=in_features,
                                        dtype=torch.complex64)
                    
                    mpo = tk.models.MPO(n_features=n_features - len(in_features),
                                        in_dim=5,
                                        out_dim=5,
                                        bond_dim=2,
                                        boundary=mpo_boundary,
                                        dtype=torch.complex64)
                    
                    # Send mpo to cuda before deparameterizing, so that all
                    # nodes are still in the state_dict of the model and are
                    # automatically sent to cuda
                    # mpo = mpo.to(device)
                    
                    # De-parameterize MPO nodes to only train MPS nodes
                    mpo = mpo.parameterize(set_param=False, override=True)

                    for auto_stack in [True, False]:
                        for auto_unbind in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    for renormalize in [True, False]:
                                        mps.auto_stack = auto_stack
                                        mps.auto_unbind = auto_unbind

                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats,
                                                  renormalize=renormalize,
                                                  marginalize_output=True,
                                                  mpo=mpo)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats,
                                                     renormalize=renormalize,
                                                     marginalize_output=True,
                                                     mpo=mpo)
                                        
                                        if in_features:
                                            assert result.shape == (100, 100)
                                        else:
                                            assert result.shape == tuple()
                                        
                                        if mps_boundary == 'obc':
                                            if mpo_boundary == 'obc':
                                                leaf = (n_features + 2) + \
                                                    (n_features - len(in_features) + 2)
                                                assert len(mps.leaf_nodes) == leaf
                                            else:
                                                leaf = (n_features + 2) + \
                                                    (n_features - len(in_features))
                                                assert len(mps.leaf_nodes) == leaf
                                        else:
                                            if mpo_boundary == 'obc':
                                                leaf = n_features + \
                                                    (n_features - len(in_features) + 2)
                                                assert len(mps.leaf_nodes) == leaf
                                            else:
                                                leaf = n_features + \
                                                    (n_features - len(in_features))
                                                assert len(mps.leaf_nodes) == leaf
                                        
                                        result.sum().abs().backward()
                                        for node in mps.mats_env:
                                            assert node.grad is not None
                                        for node in mpo.mats_env:
                                            assert node.tensor.grad is None
                                    
    def test_all_algorithms_no_marginalize(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5)
                data = torch.randn(100, len(in_features), 5)
                
                if example.numel() == 0:
                    example = None
                    data = None
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for renormalize in [True, False]:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              renormalize=renormalize)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 renormalize=renormalize)

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
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                in_features.sort()
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5)
                
                if example.numel() == 0:
                    example = None
                
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
                
                norms = []
                for log_scale in [True, False]:
                    # MPS has to be reset, otherwise norm automatically calls
                    # the forward method that was traced when contracting the MPS
                    # with example
                    mps.reset()
                    norm = mps.norm(log_scale=log_scale)
                    assert mps.resultant_nodes
                    assert not mps.data_nodes
                    assert mps.in_features == in_features
                    assert len(norm.shape) == 0
                    
                    norm.sum().backward()
                    for node in mps.mats_env:
                        assert node.grad is not None
                    
                    # Repeat norm
                    norm = mps.norm(log_scale=log_scale)
                    
                    norms.append(norm)
                    
                assert torch.isclose(norms[0].exp(), norms[1])
    
    def test_norm_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']: 
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                in_features.sort()
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5, device=device)
                
                if example.numel() == 0:
                    example = None
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features,
                                    device=device)
                mps.trace(example)
                
                assert mps.resultant_nodes
                if in_features:
                    assert mps.data_nodes
                assert mps.in_features == in_features
                
                norms = []
                for log_scale in [True, False]:
                    # MPS has to be reset, otherwise norm automatically calls
                    # the forward method that was traced when contracting the MPS
                    # with example
                    mps.reset()
                    norm = mps.norm(log_scale=log_scale)
                    assert mps.resultant_nodes
                    assert not mps.data_nodes
                    assert mps.in_features == in_features
                    assert len(norm.shape) == 0
                    
                    norm.sum().backward()
                    for node in mps.mats_env:
                        assert node.grad is not None
                    
                    # Repeat norm
                    norm = mps.norm(log_scale=log_scale)
                    
                    norms.append(norm)
                    
                assert torch.isclose(norms[0].exp(), norms[1])
    
    def test_norm_complex(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']: 
                in_features = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                in_features = list(set(in_features))
                in_features.sort()
                
                # batch x n_features x feature_dim
                example = torch.randn(1, len(in_features), 5, dtype=torch.complex64)
                
                if example.numel() == 0:
                    example = None
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=5,
                                    bond_dim=2,
                                    boundary=boundary,
                                    in_features=in_features,
                                    dtype=torch.complex64)
                mps.trace(example)
                
                assert mps.resultant_nodes
                if in_features:
                    assert mps.data_nodes
                assert mps.in_features == in_features
                
                norms = []
                for log_scale in [True, False]:
                    # MPS has to be reset, otherwise norm automatically calls
                    # the forward method that was traced when contracting the MPS
                    # with example
                    mps.reset()
                    norm = mps.norm(log_scale=log_scale)
                    assert mps.resultant_nodes
                    assert not mps.data_nodes
                    assert mps.in_features == in_features
                    assert len(norm.shape) == 0
                    
                    norm.sum().abs().backward()
                    for node in mps.mats_env:
                        assert node.grad is not None
                    
                    # Repeat norm
                    norm = mps.norm(log_scale=log_scale)
                    
                    norms.append(norm)
                    
                assert torch.isclose(norms[0].exp(), norms[1])
     
    def test_reduced_density(self):
        for n_features in [1, 2, 3, 4, 5]:
            for boundary in ['obc', 'pbc']:
                phys_dim = torch.randint(low=2, high=12,
                                         size=(n_features,)).tolist()
                bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                
                trace_sites = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=phys_dim,
                                    bond_dim=bond_dim,
                                    boundary=boundary,
                                    in_features=trace_sites)
                
                in_dims = [phys_dim[i] for i in mps.in_features]
                example = [torch.randn(1, d) for d in in_dims]
                if example == []:
                    example = None
                
                mps.trace(example)
                
                assert mps.resultant_nodes
                if trace_sites:
                    assert mps.data_nodes
                assert set(mps.in_features) == set(trace_sites)
                
                # MPS has to be reset, otherwise reduced_density automatically
                # calls the forward method that was traced when contracting the
                # MPS with example
                mps.reset()
                
                # Here, trace_sites are now the out_features,
                # not the in_features
                density = mps.reduced_density(trace_sites)
                assert mps.resultant_nodes
                assert mps.data_nodes
                assert set(mps.out_features) == set(trace_sites)
                
                assert density.shape == \
                    tuple([phys_dim[i] for i in mps.in_features] * 2)
                
                density.sum().backward()
                for node in mps.mats_env:
                    assert node.grad is not None
                
                # Repeat density
                density = mps.reduced_density(trace_sites)
    
    def test_reduced_density_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 5]:
            for boundary in ['obc', 'pbc']:
                phys_dim = torch.randint(low=2, high=12,
                                         size=(n_features,)).tolist()
                bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                
                trace_sites = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=phys_dim,
                                    bond_dim=bond_dim,
                                    boundary=boundary,
                                    in_features=trace_sites,
                                    device=device)
                
                in_dims = [phys_dim[i] for i in mps.in_features]
                example = [torch.randn(1, d, device=device) for d in in_dims]
                if example == []:
                    example = None
                
                mps.trace(example)
                
                assert mps.resultant_nodes
                if trace_sites:
                    assert mps.data_nodes
                assert set(mps.in_features) == set(trace_sites)
                
                # MPS has to be reset, otherwise reduced_density automatically
                # calls the forward method that was traced when contracting the
                # MPS with example
                mps.reset()
                
                # Here, trace_sites are now the out_features,
                # not the in_features
                density = mps.reduced_density(trace_sites)
                assert mps.resultant_nodes
                assert mps.data_nodes
                assert set(mps.out_features) == set(trace_sites)
                
                assert density.shape == \
                    tuple([phys_dim[i] for i in mps.in_features] * 2)
                
                density.sum().backward()
                for node in mps.mats_env:
                    assert node.grad is not None
                
                # Repeat density
                density = mps.reduced_density(trace_sites)
    
    def test_reduced_density_complex(self):
        for n_features in [1, 2, 3, 4, 5]:
            for boundary in ['obc', 'pbc']:
                phys_dim = torch.randint(low=2, high=12,
                                         size=(n_features,)).tolist()
                bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                
                trace_sites = torch.randint(low=0,
                                            high=n_features,
                                            size=(n_features // 2,)).tolist()
                
                mps = tk.models.MPS(n_features=n_features,
                                    phys_dim=phys_dim,
                                    bond_dim=bond_dim,
                                    boundary=boundary,
                                    in_features=trace_sites,
                                    dtype=torch.complex64)
                
                in_dims = [phys_dim[i] for i in mps.in_features]
                example = [torch.randn(1, d, dtype=torch.complex64) for d in in_dims]
                if example == []:
                    example = None
                
                mps.trace(example)
                
                assert mps.resultant_nodes
                if trace_sites:
                    assert mps.data_nodes
                assert set(mps.in_features) == set(trace_sites)
                
                # MPS has to be reset, otherwise reduced_density automatically
                # calls the forward method that was traced when contracting the
                # MPS with example
                mps.reset()
                
                # Here, trace_sites are now the out_features,
                # not the in_features
                density = mps.reduced_density(trace_sites)
                assert mps.resultant_nodes
                assert mps.data_nodes
                assert set(mps.out_features) == set(trace_sites)
                
                assert density.shape == \
                    tuple([phys_dim[i] for i in mps.in_features] * 2)
                
                density.sum().abs().backward()
                for node in mps.mats_env:
                    assert node.grad is not None
                
                # Repeat density
                density = mps.reduced_density(trace_sites)
    
    def test_entropy(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                for middle_site in range(n_features - 1):
                    bond_dim = torch.randint(low=2, high=10,
                                             size=(n_features,)).tolist()
                    bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                    
                    mps = tk.models.MPS(n_features=n_features,
                                        phys_dim=2,
                                        bond_dim=bond_dim,
                                        boundary=boundary,
                                        in_features=[],
                                        init_method='canonical')
                    
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
                    
                    # Mutual Information
                    scaled_entropy, log_norm = mps.entropy(middle_site=middle_site,
                                                           renormalize=True)
                    entropy = mps.entropy(middle_site=middle_site,
                                          renormalize=False)
                    
                    assert all([mps.bond_dim[i] <= bond_dim[i]
                                for i in range(len(bond_dim))])
                    
                    sq_norm = log_norm.exp().pow(2)
                    approx_entropy = sq_norm * scaled_entropy - sq_norm * 2 * log_norm
                    assert torch.isclose(entropy, approx_entropy,
                                         rtol=1e-03, atol=1e-05)
                    
                    if mps.boundary == 'obc':
                        assert len(mps.leaf_nodes) == n_features + 2
                    else:
                        assert len(mps.leaf_nodes) == n_features
                    assert len(mps.data_nodes) == n_features
                    
                    mps.unset_data_nodes()
                    mps.in_features = []
                    approx_mps_tensor = mps()
                    assert approx_mps_tensor.shape == (2,) * n_features
    
    def test_entropy_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                for middle_site in range(n_features - 1):
                    bond_dim = torch.randint(low=2, high=10,
                                             size=(n_features,)).tolist()
                    bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                                
                    mps = tk.models.MPS(n_features=n_features,
                                        phys_dim=2,
                                        bond_dim=bond_dim,
                                        boundary=boundary,
                                        in_features=[],
                                        device=device,
                                        init_method='canonical')
                    
                    mps_tensor = mps()
                    assert mps_tensor.shape == (2,) * n_features
                    
                    mps.out_features = []
                    example = torch.randn(1, n_features, 2, device=device)
                    mps.trace(example)
                    
                    if mps.boundary == 'obc':
                        assert len(mps.leaf_nodes) == n_features + 2
                    else:
                        assert len(mps.leaf_nodes) == n_features
                    assert len(mps.data_nodes) == n_features
                    
                    # Mutual Information
                    scaled_entropy, log_norm = mps.entropy(middle_site=middle_site,
                                                           renormalize=True)
                    entropy = mps.entropy(middle_site=middle_site,
                                          renormalize=False)
                    
                    assert all([mps.bond_dim[i] <= bond_dim[i]
                                for i in range(len(bond_dim))])
                    
                    sq_norm = log_norm.exp().pow(2)
                    approx_entropy = sq_norm * scaled_entropy - sq_norm * 2 * log_norm
                    assert torch.isclose(entropy, approx_entropy,
                                         rtol=1e-03, atol=1e-05)
                    
                    if mps.boundary == 'obc':
                        assert len(mps.leaf_nodes) == n_features + 2
                    else:
                        assert len(mps.leaf_nodes) == n_features
                    assert len(mps.data_nodes) == n_features
                    
                    mps.unset_data_nodes()
                    mps.in_features = []
                    approx_mps_tensor = mps()
                    assert approx_mps_tensor.shape == (2,) * n_features
    
    def test_entropy_complex(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                for middle_site in range(n_features - 1):
                    bond_dim = torch.randint(low=2, high=10,
                                             size=(n_features,)).tolist()
                    bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                                
                    mps = tk.models.MPS(n_features=n_features,
                                        phys_dim=2,
                                        bond_dim=bond_dim,
                                        boundary=boundary,
                                        in_features=[],
                                        dtype=torch.complex64,
                                        init_method='canonical')
                    
                    mps_tensor = mps()
                    assert mps_tensor.shape == (2,) * n_features
                    
                    mps.out_features = []
                    example = torch.randn(1, n_features, 2,
                                          dtype=torch.complex64)
                    mps.trace(example)
                    
                    if mps.boundary == 'obc':
                        assert len(mps.leaf_nodes) == n_features + 2
                    else:
                        assert len(mps.leaf_nodes) == n_features
                    assert len(mps.data_nodes) == n_features
                    
                    # Mutual Information
                    scaled_entropy, log_norm = mps.entropy(middle_site=middle_site,
                                                           renormalize=True)
                    entropy = mps.entropy(middle_site=middle_site,
                                          renormalize=False)
                    
                    assert all([mps.bond_dim[i] <= bond_dim[i]
                                for i in range(len(bond_dim))])
                    
                    sq_norm = log_norm.exp().pow(2)
                    approx_entropy = sq_norm * scaled_entropy - sq_norm * 2 * log_norm
                    assert torch.isclose(entropy, approx_entropy,
                                         rtol=1e-03, atol=1e-05)
                    
                    if mps.boundary == 'obc':
                        assert len(mps.leaf_nodes) == n_features + 2
                    else:
                        assert len(mps.leaf_nodes) == n_features
                    assert len(mps.data_nodes) == n_features
                    
                    mps.unset_data_nodes()
                    mps.in_features = []
                    approx_mps_tensor = mps()
                    assert approx_mps_tensor.shape == (2,) * n_features
    
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
    
    def test_canonicalize_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                for oc in range(n_features):
                    for mode in ['svd', 'svdr', 'qr']:
                        for renormalize in [True, False]:
                            mps = tk.models.MPS(n_features=n_features,
                                                phys_dim=2,
                                                bond_dim=10,
                                                boundary=boundary,
                                                in_features=[],
                                                device=device)
                            
                            mps_tensor = mps()
                            assert mps_tensor.shape == (2,) * n_features
                            
                            mps.out_features = []
                            example = torch.randn(1, n_features, 2,
                                                  device=device)
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
    
    def test_canonicalize_complex(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                for oc in range(n_features):
                    for mode in ['svd', 'svdr', 'qr']:
                        for renormalize in [True, False]:
                            mps = tk.models.MPS(n_features=n_features,
                                                phys_dim=2,
                                                bond_dim=10,
                                                boundary=boundary,
                                                in_features=[],
                                                dtype=torch.complex64)
                            
                            mps_tensor = mps()
                            assert mps_tensor.shape == (2,) * n_features
                            
                            mps.out_features = []
                            example = torch.randn(1, n_features, 2,
                                                  dtype=torch.complex64)
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
    
    def test_canonicalize_diff_bond_dims(self):
        for n_features in [1, 2, 3, 4, 10]:
            for boundary in ['obc', 'pbc']:
                for oc in range(n_features):
                    for mode in ['svd', 'svdr']:
                        for renormalize in [True, False]:
                            bond_dim = torch.randint(low=2, high=10,
                                                     size=(n_features,)).tolist()
                            bond_dim = bond_dim[:-1] if boundary == 'obc' \
                                else bond_dim
                
                            mps = tk.models.MPS(n_features=n_features,
                                                phys_dim=2,
                                                bond_dim=bond_dim,
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
                            mps.canonicalize(oc=oc,
                                             mode=mode,
                                             renormalize=renormalize)
                            
                            assert all([mps.bond_dim[i] <= bond_dim[i]
                                        for i in range(len(bond_dim))])
                            
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
                                  rtol=1e-2, atol=1e-3)
    
    def test_canonicalize_univocal_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 10]:
            mps = tk.models.MPS(n_features=n_features,
                                phys_dim=2,
                                bond_dim=10,
                                boundary='obc',
                                in_features=[],
                                init_method='unit',
                                device=device)
            
            mps_tensor = mps()
            assert mps_tensor.shape == (2,) * n_features
            
            mps.out_features = []
            example = torch.randn(1, n_features, 2, device=device)
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
    
    def test_canonicalize_univocal_complex(self):
        for n_features in [1, 2, 3, 4, 10]:
            mps = tk.models.MPS(n_features=n_features,
                                phys_dim=2,
                                bond_dim=10,
                                boundary='obc',
                                in_features=[],
                                init_method='unit',
                                dtype=torch.complex64)
            
            mps_tensor = mps()
            assert mps_tensor.shape == (2,) * n_features
            
            mps.out_features = []
            example = torch.randn(1, n_features, 2, dtype=torch.complex64)
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
    
    def test_save_load_model(self):
        mps = tk.models.MPS(n_features=100,
                            phys_dim=2,
                            bond_dim=10,
                            boundary='obc',
                            init_method='randn')
        
        mps.canonicalize(rank=5, renormalize=True)
        assert mps.bond_dim == [5] * 99
        
        # Save state_dict
        mps_state_dict = mps.state_dict()
        
        # Load new model from state_dict
        new_mps = tk.models.MPS(n_features=100,
                                phys_dim=2,
                                bond_dim=5,
                                boundary='obc')
        new_mps.load_state_dict(mps_state_dict)
    
    def test_save_load_model_univocal(self):
        mps = tk.models.MPS(n_features=100,
                            phys_dim=2,
                            bond_dim=10,
                            boundary='obc',
                            init_method='randn')
        
        mps.canonicalize_univocal()
        new_bond_dim = mps.bond_dim
        
        # Save state_dict
        mps_state_dict = mps.state_dict()
        
        # Load new model from state_dict
        new_mps = tk.models.MPS(n_features=100,
                                phys_dim=2,
                                bond_dim=new_bond_dim,
                                boundary='obc')
        new_mps.load_state_dict(mps_state_dict)


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
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn',
                   'randn_eye', 'unit', 'canonical']
        for n in [1, 2, 5]:
            for init_method in methods:
                mps = tk.models.UMPS(n_features=n,
                                     phys_dim=2,
                                     bond_dim=5,
                                     init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [5] * n
    
    def test_initialize_init_method_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn',
                   'randn_eye', 'unit', 'canonical']
        for n in [1, 2, 5]:
            for init_method in methods:
                mps = tk.models.UMPS(n_features=n,
                                     phys_dim=2,
                                     bond_dim=5,
                                     init_method=init_method,
                                     device=device)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [5] * n
                for node in mps.mats_env:
                    assert node.device == device
    
    def test_initialize_init_method_complex(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn',
                   'randn_eye', 'unit', 'canonical']
        for n in [1, 2, 5]:
            for init_method in methods:
                mps = tk.models.UMPS(n_features=n,
                                     phys_dim=2,
                                     bond_dim=5,
                                     init_method=init_method,
                                     dtype=torch.complex64)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.phys_dim == [2] * n
                assert mps.bond_dim == [5] * n
                for node in mps.mats_env:
                    assert node.dtype == torch.complex64
                    assert node.is_complex()
    
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
                phys_dim = torch.randint(low=2, high=12, size=(1,)).item()
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
    
    def test_deparameterize(self):
        for n_features in [1, 2, 3, 4, 6]:
            for override in [True, False]:
                phys_dim = torch.randint(low=2, high=12, size=(1,)).item()
                bond_dim = torch.randint(low=2, high=10, size=(1,)).item()
                
                mps = tk.models.UMPS(n_features=n_features,
                                     phys_dim=phys_dim,
                                     bond_dim=bond_dim)
                
                non_param_mps = mps.parameterize(set_param=False,
                                                 override=override)
                
                if override:
                    assert non_param_mps is mps
                else:
                    assert non_param_mps is not mps
                    
                for node in non_param_mps.mats_env:
                    assert isinstance(node, tk.Node)
                    assert not isinstance(node.tensor, torch.nn.Parameter)
                    assert node.tensor_address() == 'virtual_uniform'

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
                            for renormalize in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats,
                                          renormalize=renormalize)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats,
                                             renormalize=renormalize)

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
            in_features = torch.randint(low=0,
                                        high=n_features,
                                        size=(n_features // 2,)).tolist()
            in_features = list(set(in_features))
            
            # batch x n_features x feature_dim
            example = torch.randn(1, len(in_features), 5)
            data = torch.randn(100, len(in_features), 5)
            
            if example.numel() == 0:
                example = None
                data = None
            
            mps = tk.models.UMPS(n_features=n_features,
                                 phys_dim=5,
                                 bond_dim=2,
                                 in_features=in_features)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            for renormalize in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats,
                                          renormalize=renormalize,
                                          marginalize_output=True)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats,
                                             renormalize=renormalize,
                                             marginalize_output=True)

                                if in_features:
                                    assert result.shape == (100, 100)
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
            in_features = torch.randint(low=0,
                                        high=n_features,
                                        size=(n_features // 2,)).tolist()
            in_features = list(set(in_features))
            
            # batch x n_features x feature_dim
            example = torch.randn(1, len(in_features), 5)
            data = torch.randn(100, len(in_features), 5)
            
            if example.numel() == 0:
                example = None
                data = None
            
            mps = tk.models.UMPS(n_features=n_features,
                                 phys_dim=5,
                                 bond_dim=2,
                                 in_features=in_features)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            for renormalize in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         renormalize=renormalize)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats,
                                             renormalize=renormalize)

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
            in_features = torch.randint(low=0,
                                        high=n_features,
                                        size=(n_features // 2,)).tolist()
            in_features = list(set(in_features))
            in_features.sort()
            
            # batch x n_features x feature_dim
            example = torch.randn(1, len(in_features), 5)
            
            if example.numel() == 0:
                example = None
            
            mps = tk.models.UMPS(n_features=n_features,
                                 phys_dim=5,
                                 bond_dim=2,
                                 in_features=in_features)
            mps.trace(example)
            
            assert mps.resultant_nodes
            if in_features:
                assert mps.data_nodes
            assert mps.in_features == in_features
            
            norms = []
            for log_scale in [True, False]:
                # MPS has to be reset, otherwise norm automatically calls
                # the forward method that was traced when contracting the MPS
                # with example
                mps.reset()
                norm = mps.norm(log_scale=log_scale)
                assert mps.resultant_nodes
                assert not mps.data_nodes
                assert mps.in_features == in_features
                assert len(norm.shape) == 0
                
                norm.sum().backward()
                for node in mps.mats_env:
                    assert node.grad is not None
                
                # Repeat norm
                norm = mps.norm(log_scale=log_scale)
                
                norms.append(norm)
                
            assert torch.isclose(norms[0].exp(), norms[1])
    
    def test_reduced_density(self):
        for n_features in [1, 2, 3, 4, 5]:
            phys_dim = torch.randint(low=2, high=12, size=(1,)).item()
            bond_dim = torch.randint(low=2, high=10, size=(1,)).item()
            
            trace_sites = torch.randint(low=0,
                                        high=n_features,
                                        size=(n_features // 2,)).tolist()
            
            mps = tk.models.UMPS(n_features=n_features,
                                 phys_dim=phys_dim,
                                 bond_dim=bond_dim,
                                 in_features=trace_sites)
            
            # batch x n_features x feature_dim
            example = torch.randn(1, n_features // 2, phys_dim)
            if example.numel() == 0:
                example = None
            
            mps.trace(example)
            
            assert mps.resultant_nodes
            if trace_sites:
                assert mps.data_nodes
            assert set(mps.in_features) == set(trace_sites)
            
            # MPS has to be reset, otherwise reduced_density automatically
            # calls the forward method that was traced when contracting the
            # MPS with example
            mps.reset()
            
            # Here, trace_sites are now the out_features,
            # not the in_features
            density = mps.reduced_density(trace_sites)
            assert mps.resultant_nodes
            assert mps.data_nodes
            assert set(mps.out_features) == set(trace_sites)
            
            assert density.shape == (phys_dim,) * 2 * len(mps.in_features)
            
            density.sum().backward()
            for node in mps.mats_env:
                assert node.grad is not None
            
            # Repeat density
            density = mps.reduced_density(trace_sites)
    
    def test_reduced_density_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n_features in [1, 2, 3, 4, 5]:
            phys_dim = torch.randint(low=2, high=12, size=(1,)).item()
            bond_dim = torch.randint(low=2, high=10, size=(1,)).item()
            
            trace_sites = torch.randint(low=0,
                                        high=n_features,
                                        size=(n_features // 2,)).tolist()
            
            mps = tk.models.UMPS(n_features=n_features,
                                 phys_dim=phys_dim,
                                 bond_dim=bond_dim,
                                 in_features=trace_sites,
                                 device=device)
            
            # batch x n_features x feature_dim
            example = torch.randn(1, n_features // 2, phys_dim, device=device)
            if example.numel() == 0:
                example = None
            
            mps.trace(example)
            
            assert mps.resultant_nodes
            if trace_sites:
                assert mps.data_nodes
            assert set(mps.in_features) == set(trace_sites)
            
            # MPS has to be reset, otherwise reduced_density automatically
            # calls the forward method that was traced when contracting the
            # MPS with example
            mps.reset()
            
            # Here, trace_sites are now the out_features,
            # not the in_features
            density = mps.reduced_density(trace_sites)
            assert mps.resultant_nodes
            assert mps.data_nodes
            assert set(mps.out_features) == set(trace_sites)
            
            assert density.shape == (phys_dim,) * 2 * len(mps.in_features)
            
            density.sum().backward()
            for node in mps.mats_env:
                assert node.grad is not None
            
            # Repeat density
            density = mps.reduced_density(trace_sites)
    
    def test_reduced_density_complex(self):
        for n_features in [1, 2, 3, 4, 5]:
            phys_dim = torch.randint(low=2, high=12, size=(1,)).item()
            bond_dim = torch.randint(low=2, high=10, size=(1,)).item()
            
            trace_sites = torch.randint(low=0,
                                        high=n_features,
                                        size=(n_features // 2,)).tolist()
            
            mps = tk.models.UMPS(n_features=n_features,
                                 phys_dim=phys_dim,
                                 bond_dim=bond_dim,
                                 in_features=trace_sites,
                                 dtype=torch.complex64)
            
            # batch x n_features x feature_dim
            example = torch.randn(1, n_features // 2, phys_dim,
                                  dtype=torch.complex64)
            if example.numel() == 0:
                example = None
            
            mps.trace(example)
            
            assert mps.resultant_nodes
            if trace_sites:
                assert mps.data_nodes
            assert set(mps.in_features) == set(trace_sites)
            
            # MPS has to be reset, otherwise reduced_density automatically
            # calls the forward method that was traced when contracting the
            # MPS with example
            mps.reset()
            
            # Here, trace_sites are now the out_features,
            # not the in_features
            density = mps.reduced_density(trace_sites)
            assert mps.resultant_nodes
            assert mps.data_nodes
            assert set(mps.out_features) == set(trace_sites)
            
            assert density.shape == (phys_dim,) * 2 * len(mps.in_features)
            
            density.sum().abs().backward()
            for node in mps.mats_env:
                assert node.grad is not None
            
            # Repeat density
            density = mps.reduced_density(trace_sites)
    
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
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn',
                   'randn_eye', 'unit', 'canonical']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mps = tk.models.MPSLayer(boundary='pbc',
                                         n_features=n,
                                         in_dim=2,
                                         out_dim=10,
                                         bond_dim=5,
                                         init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 10
                assert mps.bond_dim == [5] * n
                
                # OBC
                mps = tk.models.MPSLayer(boundary='obc',
                                         n_features=n,
                                         in_dim=2,
                                         out_dim=10,
                                         bond_dim=5,
                                         init_method=init_method)
                assert mps.n_features == n
                assert mps.boundary == 'obc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 10
                assert mps.bond_dim == [5] * (n - 1)
                
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
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn',
                   'randn_eye', 'unit', 'canonical']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mps = tk.models.MPSLayer(boundary='pbc',
                                         n_features=n,
                                         in_dim=2,
                                         out_dim=10,
                                         bond_dim=5,
                                         init_method=init_method,
                                         device=device)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 10
                assert mps.bond_dim == [5] * n
                for node in mps.mats_env:
                    assert node.device == device
                
                # OBC
                mps = tk.models.MPSLayer(boundary='obc',
                                         n_features=n,
                                         in_dim=2,
                                         out_dim=10,
                                         bond_dim=5,
                                         init_method=init_method,
                                         device=device)
                assert mps.n_features == n
                assert mps.boundary == 'obc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 10
                assert mps.bond_dim == [5] * (n - 1)
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
    
    def test_initialize_init_method_complex(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn',
                   'randn_eye', 'unit', 'canonical']
        for n in [1, 2, 5]:
            for init_method in methods:
                # PBC
                mps = tk.models.MPSLayer(boundary='pbc',
                                         n_features=n,
                                         in_dim=2,
                                         out_dim=10,
                                         bond_dim=5,
                                         init_method=init_method,
                                         dtype=torch.complex64)
                assert mps.n_features == n
                assert mps.boundary == 'pbc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 10
                assert mps.bond_dim == [5] * n
                for node in mps.mats_env:
                    assert node.dtype == torch.complex64
                    assert node.is_complex()
                
                # OBC
                mps = tk.models.MPSLayer(boundary='obc',
                                         n_features=n,
                                         in_dim=2,
                                         out_dim=10,
                                         bond_dim=5,
                                         init_method=init_method,
                                         dtype=torch.complex64)
                assert mps.n_features == n
                assert mps.boundary == 'obc'
                assert mps.in_dim == [2] * (n - 1)
                assert mps.out_dim == 10
                assert mps.bond_dim == [5] * (n - 1)
                for node in mps.mats_env:
                    assert node.dtype == torch.complex64
                    assert node.is_complex()
                
                assert torch.equal(mps.left_node.tensor[0],
                                   torch.ones_like(mps.left_node.tensor)[0])
                assert torch.equal(mps.left_node.tensor[1:],
                                   torch.zeros_like(mps.left_node.tensor)[1:])
                
                assert torch.equal(mps.right_node.tensor[0],
                                   torch.ones_like(mps.right_node.tensor)[0])
                assert torch.equal(mps.right_node.tensor[1:],
                                   torch.zeros_like(mps.right_node.tensor)[1:])
    
    def test_initialize_canonical(self):
        for n in [1, 2, 5]:
            # PBC
            mps = tk.models.MPSLayer(boundary='pbc',
                                     n_features=n,
                                     in_dim=10,
                                     out_dim=10,
                                     bond_dim=10,
                                     init_method='canonical')
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [10] * n
            assert mps.bond_dim == [10] * n
            
            # For PBC norm does not have to be 10**n always
            
            # OBC
            mps = tk.models.MPSLayer(boundary='obc',
                                     n_features=n,
                                     in_dim=10,
                                     out_dim=10,
                                     bond_dim=10,
                                     init_method='canonical')
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [10] * n
            assert mps.bond_dim == [10] * (n - 1)
            
            # Check it has norm == 10**n
            assert mps.norm().isclose(torch.tensor(10. ** n).sqrt())
            # Norm is close to 10**n if bond dimension is <= than
            # physical dimension, otherwise, it will not be exactly 10**n
    
    def test_initialize_canonical_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for n in [1, 2, 5]:
            # PBC
            mps = tk.models.MPSLayer(boundary='pbc',
                                     n_features=n,
                                     in_dim=10,
                                     out_dim=10,
                                     bond_dim=10,
                                     init_method='canonical',
                                     device=device)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [10] * n
            assert mps.bond_dim == [10] * n
            for node in mps.mats_env:
                assert node.device == device
            
            # For PBC norm does not have to be 10**n always
            
            # OBC
            mps = tk.models.MPSLayer(boundary='obc',
                                     n_features=n,
                                     in_dim=10,
                                     out_dim=10,
                                     bond_dim=10,
                                     init_method='canonical',
                                     device=device)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [10] * n
            assert mps.bond_dim == [10] * (n - 1)
            for node in mps.mats_env:
                assert node.device == device
            
            # Check it has norm == 10**n
            assert mps.norm().isclose(torch.tensor(10. ** n).sqrt())
            # Norm is close to 10**n if bond dimension is <= than
            # physical dimension, otherwise, it will not be exactly 10**n
    
    def test_initialize_canonical_complex(self):
        for n in [1, 2, 5]:
            # PBC
            mps = tk.models.MPSLayer(boundary='pbc',
                                     n_features=n,
                                     in_dim=10,
                                     out_dim=10,
                                     bond_dim=10,
                                     init_method='canonical',
                                     dtype=torch.complex64)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [10] * n
            assert mps.bond_dim == [10] * n
            for node in mps.mats_env:
                assert node.dtype == torch.complex64
                assert node.is_complex()
            
            # For PBC norm does not have to be 10**n always
            
            # OBC
            mps = tk.models.MPSLayer(boundary='obc',
                                     n_features=n,
                                     in_dim=10,
                                     out_dim=10,
                                     bond_dim=10,
                                     init_method='canonical',
                                     dtype=torch.complex64)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [10] * n
            assert mps.bond_dim == [10] * (n - 1)
            for node in mps.mats_env:
                assert node.dtype == torch.complex64
                assert node.is_complex()
            
            # Check it has norm == 10**n
            assert mps.norm().isclose(torch.tensor(10. ** n).sqrt())
            # Norm is close to 10**n if bond dimension is <= than
            # physical dimension, otherwise, it will not be exactly 10**n
    
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
                    in_dim = torch.randint(low=2, high=12,
                                           size=(n_features - 1,)).tolist()
                    out_dim = torch.randint(low=2, high=12, size=(1,)).item()
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
    
    def test_deparameterize(self):
        for n_features in [1, 2, 3, 4, 6]:
            for boundary in ['obc', 'pbc']:
                for override in [True, False]:
                    in_dim = torch.randint(low=2, high=12,
                                           size=(n_features - 1,)).tolist()
                    out_dim = torch.randint(low=2, high=12, size=(1,)).item()
                    bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
                    bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
                    
                    mps = tk.models.MPSLayer(n_features=n_features,
                                             in_dim=in_dim,
                                             out_dim=out_dim,
                                             bond_dim=bond_dim,
                                             boundary=boundary)
                    
                    non_param_mps = mps.parameterize(set_param=False,
                                                     override=override)
                    
                    if override:
                        assert non_param_mps is mps
                    else:
                        assert non_param_mps is not mps
                    
                    new_nodes = non_param_mps.mats_env
                    if boundary == 'obc':
                        new_nodes += [non_param_mps.left_node,
                                      non_param_mps.right_node]
                        
                    for node in new_nodes:
                        assert isinstance(node, tk.Node)
                        assert not isinstance(node.tensor, torch.nn.Parameter)


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
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn',
                   'randn_eye', 'unit', 'canonical']
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
                in_dim = torch.randint(low=2, high=12, size=(1,)).item()
                out_dim = torch.randint(low=2, high=12, size=(1,)).item()
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
    
    def test_deparameterize(self):
        for n_features in [1, 2, 3, 4, 6]:
            for override in [True, False]:
                in_dim = torch.randint(low=2, high=12, size=(1,)).item()
                out_dim = torch.randint(low=2, high=12, size=(1,)).item()
                bond_dim = torch.randint(low=2, high=10, size=(1,)).item()
                
                mps = tk.models.UMPSLayer(n_features=n_features,
                                          in_dim=in_dim,
                                          out_dim=out_dim,
                                          bond_dim=bond_dim)
                
                non_param_mps = mps.parameterize(set_param=False,
                                                 override=override)
                
                if override:
                    assert non_param_mps is mps
                else:
                    assert non_param_mps is not mps
                    
                for node in non_param_mps.mats_env:
                    assert isinstance(node, tk.Node)
                    assert not isinstance(node.tensor, torch.nn.Parameter)
                    assert node.tensor_address() == 'virtual_uniform'
    
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
    
    def test_initialize_with_tensors(self):
        for n in [1, 2, 5]:
            # PBC
            tensors = [torch.randn(20, 10, 2, 10) for _ in range(n)]
            mps = tk.models.MPSData(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * n
            
            # OBC
            tensors = [torch.randn(20, 10, 2, 10) for _ in range(n)]
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]
            mps = tk.models.MPSData(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * (n - 1)
    
    def test_initialize_with_tensors_cuda(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for n in [1, 2, 5]:
            # PBC
            tensors = [torch.randn(20, 10, 2, 10, device=device) for _ in range(n)]
            mps = tk.models.MPSData(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'pbc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * n
            for node in mps.mats_env:
                assert node.device == device
            
            # OBC
            tensors = [torch.randn(20, 10, 2, 10, device=device) for _ in range(n)]
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]
            mps = tk.models.MPSData(tensors=tensors)
            assert mps.n_features == n
            assert mps.boundary == 'obc'
            assert mps.phys_dim == [2] * n
            assert mps.bond_dim == [10] * (n - 1)
            for node in mps.mats_env:
                assert node.device == device
    
    def test_initialize_with_tensors_ignore_rest(self):
        tensors = [torch.randn(20, 10, 2, 10) for _ in range(10)]
        mps = tk.models.MPSData(tensors=tensors,
                                boundary='obc',
                                n_features=3,
                                phys_dim=4,
                                bond_dim=7)
        assert mps.boundary == 'pbc'
        assert mps.n_features == 10
        assert mps.phys_dim == [2] * 10
        assert mps.bond_dim == [10] * 10
        
    def test_initialize_with_tensors_errors(self):
        # Number of batches should coincide with n_batches
        tensors = [torch.randn(20, 20, 5, 2, 5) for _ in range(10)]
        with pytest.raises(ValueError):
            mps = tk.models.MPSData(tensors=tensors,
                                    n_batches=1)
        
        # First and last tensors should have the same rank
        tensors = [torch.randn(20, 10, 2, 10) for _ in range(10)]
        tensors[0] = tensors[0][:, 0]
        with pytest.raises(ValueError):
            mps = tk.models.MPSData(tensors=tensors)
        
        # First and last bond dims should coincide
        tensors = [torch.randn(20, 10, 2, 10) for _ in range(10)]
        tensors[0] = tensors[0][:, :5]
        tensors[-1] = tensors[-1][..., :3]
        with pytest.raises(ValueError):
            mps = tk.models.MPSData(tensors=tensors)
    
    def test_initialize_init_method(self):
        methods = ['zeros', 'ones', 'copy', 'rand', 'randn']
        for n_features in [1, 2, 5]:
            for boundary in ['obc', 'pbc']:
                for n_batches in [1, 2, 3]:
                    for init_method in methods:
                        mps = tk.models.MPSData(boundary=boundary,
                                                n_features=n_features,
                                                phys_dim=2,
                                                bond_dim=5,
                                                n_batches=n_batches,
                                                init_method=init_method)
                        
                        assert mps.n_features == n_features
                        assert mps.boundary == boundary
                        assert mps.phys_dim == [2] * n_features
                        
                        if boundary == 'obc':
                            assert mps.bond_dim == [5] * (n_features - 1)
                        else:
                            assert mps.bond_dim == [5] * n_features
                        
                        if (n_features == 1) and (boundary == 'obc'):
                            node = mps.mats_env[0]
                            assert node.shape == tuple([1] * n_batches + [1, 2, 1])
                        else:  
                            for node in mps.mats_env:
                                assert node.shape == tuple([1] * n_batches + [5, 2, 5])
    
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