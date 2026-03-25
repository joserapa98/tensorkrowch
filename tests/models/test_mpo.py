"""
Tests for mpo:

    * TestMPO
    * TestUMPO
"""

import pytest

import torch
import tensorkrowch as tk

AUTO_BOOL_CASES = [True, False]
N_FEATURES_CASES = [1, 2, 3, 4, 10]
SMALL_N_FEATURES_CASES = [1, 2, 4]
BOUNDARY_CASES = ['obc', 'pbc']
DEVICE_RUNTIME_CASES = ['default', 'cuda']
ALGORITHM_RUNTIME_CASES = ['default', 'cuda']
INIT_N_CASES = [1, 2, 5]
INIT_METHODS = ['zeros', 'ones', 'copy', 'rand', 'randn']
MODEL_N_FEATURES_CASES = [1, 2, 3, 4, 6]
CANONICALIZE_MODES = ['svd', 'svdr', 'qr']
CANONICALIZE_CASES = [
    (n_features, boundary, oc, mode, renormalize)
    for n_features in SMALL_N_FEATURES_CASES
    for boundary in BOUNDARY_CASES
    for oc in range(n_features)
    for mode in CANONICALIZE_MODES
    for renormalize in AUTO_BOOL_CASES
]
MPO_MPS_DATA_ALGORITHM_CASES = [
    (n_features, mpo_boundary, mps_boundary, inline_input, inline_mats, renormalize)
    for n_features in SMALL_N_FEATURES_CASES
    for mpo_boundary in BOUNDARY_CASES
    for mps_boundary in BOUNDARY_CASES
    for inline_input in AUTO_BOOL_CASES
    for inline_mats in AUTO_BOOL_CASES
    for renormalize in AUTO_BOOL_CASES
]
UMPO_MPS_DATA_ALGORITHM_CASES = [
    (n_features, mps_boundary, inline_input, inline_mats)
    for n_features in SMALL_N_FEATURES_CASES
    for mps_boundary in BOUNDARY_CASES
    for inline_input in AUTO_BOOL_CASES
    for inline_mats in AUTO_BOOL_CASES
]


def _runtime_kwargs(runtime, device):
    if runtime == 'cuda':
        return {'device': device}
    return {}


def _assert_nodes_runtime(nodes, runtime, device):
    if runtime == 'cuda':
        for node in nodes:
            assert node.device == device


def _assert_boundary_vector(node):
    assert torch.equal(node.tensor[0], torch.ones_like(node.tensor)[0])
    assert torch.equal(node.tensor[1:], torch.zeros_like(node.tensor)[1:])


def _assert_copied_mpo(mpo, copied_mpo, share_tensors):
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


def _assert_deparameterized_nodes(nodes, tensor_address=None):
    for node in nodes:
        assert isinstance(node, tk.Node)
        assert not isinstance(node.tensor, torch.nn.Parameter)
        if tensor_address is not None:
            assert node.tensor_address() == tensor_address


def _run_umpo_mps_data_case(n_features, mps_boundary, inline_input, inline_mats):
    phys_dim = torch.randint(low=2, high=6, size=(1,)).item()
    bond_dim = torch.randint(low=2, high=5, size=(n_features,)).tolist()

    mpo = tk.models.UMPO(n_features=n_features,
                         in_dim=phys_dim,
                         out_dim=2,
                         bond_dim=10)

    mps_data = tk.models.MPSData(
        n_features=n_features,
        phys_dim=phys_dim,
        bond_dim=bond_dim[:-1] if mps_boundary == 'obc' else bond_dim,
        boundary=mps_boundary)

    # Reuse the same UMPO across several MPSData payloads to exercise reset logic.
    for _ in range(3):
        if not inline_input or not inline_mats:
            mpo.reset()

        bond_dim = torch.randint(low=2, high=5, size=(n_features,)).tolist()
        tensors = [
            torch.randn(5, bond_dim[i - 1], phys_dim, bond_dim[i])
            for i in range(n_features)
        ]
        if mps_boundary == 'obc':
            tensors[0] = tensors[0][:, 0]
            tensors[-1] = tensors[-1][..., 0]

        mps_data.add_data(tensors)
        result = mpo(mps=mps_data,
                     inline_input=inline_input,
                     inline_mats=inline_mats)

        assert result.shape == tuple([5] + [2] * n_features)

    for i, node in enumerate(mpo.mats_env):
        assert node.is_connected_to(mps_data.mats_env[i])

    mpo.unset_data_nodes()
    for i, node in enumerate(mpo.mats_env):
        assert not node.is_connected_to(mps_data.mats_env[i])
        assert mps_data.mats_env[i].network is None


class TestMPO:  # MARK: TestMPO

    def _assert_canonicalized_mpo_bond_dim(self, mpo, rank, mode):
        # QR keeps the decomposition exact, so the rank cap only applies to
        # the SVD-based variants.
        if not mpo.bond_dim or mode == 'qr':
            return
        if mpo.boundary == 'obc':
            assert (torch.tensor(mpo.bond_dim) <= rank).all()
        else:
            assert (torch.tensor(mpo.bond_dim[:-1]) <= rank).all()

    def _assert_mpo_leaf_nodes(self, mpo, n_features):
        # Boundary conditions determine whether the boundary vectors are exposed
        # as extra leaves after canonicalization.
        if mpo.boundary == 'obc':
            assert len(mpo.leaf_nodes) == n_features + 2
        else:
            assert len(mpo.leaf_nodes) == n_features

    def _run_mpo_mps_data_case(self, n_features, mpo_boundary, mps_boundary,
                               inline_input, inline_mats, renormalize):
        phys_dim = torch.randint(low=2, high=6, size=(n_features,)).tolist()
        bond_dim = torch.randint(low=2, high=5, size=(n_features,)).tolist()

        mpo = tk.models.MPO(n_features=n_features,
                            in_dim=phys_dim,
                            out_dim=2,
                            bond_dim=10,
                            boundary=mpo_boundary)

        mps_data = tk.models.MPSData(
            n_features=n_features,
            phys_dim=phys_dim,
            bond_dim=bond_dim[:-1] if mps_boundary == 'obc' else bond_dim,
            boundary=mps_boundary)

        # Reuse the same MPO/MPSData a few times because this is exactly the
        # scenario that used to stress stack recomputation across calls.
        for _ in range(3):
            if not inline_input or not inline_mats:
                mpo.reset()

            bond_dim = torch.randint(low=2, high=5, size=(n_features,)).tolist()
            tensors = [
                torch.randn(5, bond_dim[i - 1], phys_dim[i], bond_dim[i])
                for i in range(n_features)
            ]
            if mps_boundary == 'obc':
                tensors[0] = tensors[0][:, 0]
                tensors[-1] = tensors[-1][..., 0]

            mps_data.add_data(tensors)
            result = mpo(mps=mps_data,
                         inline_input=inline_input,
                         inline_mats=inline_mats,
                         renormalize=renormalize)

            assert result.shape == tuple([5] + [2] * n_features)

        for i, node in enumerate(mpo.mats_env):
            assert node.is_connected_to(mps_data.mats_env[i])

        # Unsetting data nodes must detach the borrowed MPSData structure.
        mpo.unset_data_nodes()
        for i, node in enumerate(mpo.mats_env):
            assert not node.is_connected_to(mps_data.mats_env[i])
            assert mps_data.mats_env[i].network is None

    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_with_tensors(self, n, boundary):
        tensors = [torch.randn(10, 2, 10, 2) for _ in range(n)]
        if boundary == 'obc':
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0, :]

        mpo = tk.models.MPO(tensors=tensors)
        assert mpo.n_features == n
        assert mpo.boundary == boundary
        assert mpo.in_dim == [2] * n
        assert mpo.out_dim == [2] * n
        assert mpo.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
    
    @pytest.mark.parametrize('runtime', DEVICE_RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_with_tensors_runtime(self, runtime, n, boundary):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tensor_kwargs = _runtime_kwargs(runtime, device)
        tensors = [torch.randn(10, 2, 10, 2, **tensor_kwargs) for _ in range(n)]
        if boundary == 'obc':
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0, :]

        mpo = tk.models.MPO(tensors=tensors)
        assert mpo.n_features == n
        assert mpo.boundary == boundary
        assert mpo.in_dim == [2] * n
        assert mpo.out_dim == [2] * n
        assert mpo.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
        _assert_nodes_runtime(mpo.mats_env, runtime, device)
    
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
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('init_method', INIT_METHODS)
    def test_initialize_init_method(self, n, boundary, init_method):
        mpo = tk.models.MPO(boundary=boundary,
                            n_features=n,
                            in_dim=2,
                            out_dim=2,
                            bond_dim=10,
                            init_method=init_method)
        assert mpo.n_features == n
        assert mpo.boundary == boundary
        assert mpo.in_dim == [2] * n
        assert mpo.out_dim == [2] * n
        assert mpo.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
        if boundary == 'obc':
            _assert_boundary_vector(mpo.left_node)
            _assert_boundary_vector(mpo.right_node)
    
    @pytest.mark.parametrize('runtime', DEVICE_RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('init_method', INIT_METHODS)
    def test_initialize_init_method_runtime(self, runtime, n, boundary,
                                            init_method):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_kwargs = _runtime_kwargs(runtime, device)
        mpo = tk.models.MPO(boundary=boundary,
                            n_features=n,
                            in_dim=2,
                            out_dim=2,
                            bond_dim=10,
                            init_method=init_method,
                            **model_kwargs)
        assert mpo.n_features == n
        assert mpo.boundary == boundary
        assert mpo.in_dim == [2] * n
        assert mpo.out_dim == [2] * n
        assert mpo.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
        _assert_nodes_runtime(mpo.mats_env, runtime, device)
        if boundary == 'obc':
            _assert_boundary_vector(mpo.left_node)
            _assert_boundary_vector(mpo.right_node)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_in_out_dims(self, n_features, boundary):
        in_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        out_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()

        mpo = tk.models.MPO(n_features=n_features,
                            in_dim=in_dim,
                            out_dim=out_dim,
                            bond_dim=10,
                            boundary=boundary)

        assert mpo.in_dim == in_dim
        assert mpo.out_dim == out_dim
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_in_out_dims_error(self, n_features, boundary):
        # in_dim should have n_features elements.
        in_dim = torch.randint(low=2, high=10, size=(n_features + 1,)).tolist()
        out_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        with pytest.raises(ValueError):
            tk.models.MPO(n_features=n_features,
                          in_dim=in_dim,
                          out_dim=out_dim,
                          bond_dim=10,
                          boundary=boundary)

        # out_dim should have n_features elements.
        in_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        out_dim = torch.randint(low=2, high=10, size=(n_features + 1,)).tolist()
        with pytest.raises(ValueError):
            tk.models.MPO(n_features=n_features,
                          in_dim=in_dim,
                          out_dim=out_dim,
                          bond_dim=10,
                          boundary=boundary)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_bond_dims(self, n_features, boundary):
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
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy(self, n_features, boundary, share_tensors):
        in_dim = torch.randint(low=2, high=12, size=(n_features,)).tolist()
        out_dim = torch.randint(low=2, high=12, size=(n_features,)).tolist()
        bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        mpo = tk.models.MPO(n_features=n_features,
                            in_dim=in_dim,
                            out_dim=out_dim,
                            bond_dim=bond_dim,
                            boundary=boundary)

        copied_mpo = mpo.copy(share_tensors=share_tensors)

        assert isinstance(copied_mpo, tk.models.MPO)
        _assert_copied_mpo(mpo, copied_mpo, share_tensors)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('override', AUTO_BOOL_CASES)
    def test_deparameterize(self, n_features, boundary, override):
        in_dim = torch.randint(low=2, high=12, size=(n_features,)).tolist()
        out_dim = torch.randint(low=2, high=12, size=(n_features,)).tolist()
        bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        mpo = tk.models.MPO(n_features=n_features,
                            in_dim=in_dim,
                            out_dim=out_dim,
                            bond_dim=bond_dim,
                            boundary=boundary)

        non_param_mpo = mpo.parameterize(set_param=False, override=override)

        if override:
            assert non_param_mpo is mpo
        else:
            assert non_param_mpo is not mpo

        new_nodes = non_param_mpo.mats_env[:]
        if boundary == 'obc':
            new_nodes += [non_param_mpo.left_node, non_param_mpo.right_node]

        _assert_deparameterized_nodes(new_nodes)
    
    def test_update_bond_dim(self):
        mpo = tk.models.MPO(n_features=100,
                            in_dim=2,
                            out_dim=2,
                            bond_dim=10,
                            boundary='obc',
                            init_method='randn')
        
        mpo.canonicalize(rank=3, renormalize=True)
        assert mpo.bond_dim == [3] * 99
        assert (mpo.left_node.tensor == torch.tensor([1., 0., 0.])).all()
        assert (mpo.right_node.tensor == torch.tensor([1., 0., 0.])).all()
        
        mpo.canonicalize(rank=5, renormalize=True)
        assert mpo.bond_dim == [5] * 99
        assert (mpo.left_node.tensor == torch.tensor([1., 0., 0. , 0., 0.])).all()
        assert (mpo.right_node.tensor == torch.tensor([1., 0., 0. , 0., 0.])).all()
    
    def _run_all_algorithms_case(self, n_features, boundary, auto_stack,
                                 auto_unbind, inline_input, inline_mats,
                                 renormalize, device=None):
        tensor_kwargs = {}
        model_kwargs = {}
        if device is not None:
            tensor_kwargs['device'] = device
            model_kwargs['device'] = device

        example = torch.randn(1, n_features, 2, **tensor_kwargs)
        data = torch.randn(100, n_features, 2, **tensor_kwargs)

        mpo = tk.models.MPO(n_features=n_features,
                            in_dim=2,
                            out_dim=2,
                            bond_dim=10,
                            boundary=boundary,
                            **model_kwargs)
        mpo.auto_stack = auto_stack
        mpo.auto_unbind = auto_unbind

        mpo.trace(example,
                  inline_input=inline_input,
                  inline_mats=inline_mats,
                  renormalize=renormalize)
        result = mpo(data,
                     inline_input=inline_input,
                     inline_mats=inline_mats,
                     renormalize=renormalize)

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

    @pytest.mark.parametrize('runtime', ALGORITHM_RUNTIME_CASES)
    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms(self, runtime, n_features, boundary, auto_stack,
                            auto_unbind, inline_input, inline_mats,
                            renormalize):
        device = None
        if runtime == 'cuda':
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._run_all_algorithms_case(n_features, boundary, auto_stack,
                                      auto_unbind, inline_input,
                                      inline_mats, renormalize, device=device)
    
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
    
    @pytest.mark.parametrize(
        'n_features,mpo_boundary,mps_boundary,inline_input,inline_mats,renormalize',
        MPO_MPS_DATA_ALGORITHM_CASES,
    )
    def test_mpo_mps_data_all_algorithms(self, n_features, mpo_boundary,
                                         mps_boundary, inline_input,
                                         inline_mats, renormalize):
        self._run_mpo_mps_data_case(n_features, mpo_boundary, mps_boundary,
                                    inline_input, inline_mats, renormalize)
    
    @pytest.mark.parametrize(
        'n_features,boundary,oc,mode,renormalize',
        CANONICALIZE_CASES,
    )
    def test_canonicalize(self, n_features, boundary, oc, mode, renormalize):
        # Canonicalization should respect the requested rank constraint and
        # preserve the expected boundary bookkeeping.
        mpo = tk.models.MPO(n_features=n_features,
                            in_dim=2,
                            out_dim=2,
                            bond_dim=10,
                            boundary=boundary)

        rank = torch.randint(3, 7, (1,)).item()
        mpo.canonicalize(oc=oc,
                         mode=mode,
                         rank=rank,
                         cum_percentage=0.98,
                         cutoff=1e-5,
                         renormalize=renormalize)

        self._assert_canonicalized_mpo_bond_dim(mpo, rank, mode)
        self._assert_mpo_leaf_nodes(mpo, n_features)

    @pytest.mark.parametrize(
        'n_features,boundary,oc,mode,renormalize',
        CANONICALIZE_CASES,
    )
    def test_canonicalize_diff_bond_dims(self, n_features, boundary, oc,
                                         mode, renormalize):
        # Same canonicalization checks, but starting from heterogeneous bond dims.
        bond_dim = torch.randint(low=2, high=6, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        mpo = tk.models.MPO(n_features=n_features,
                            in_dim=2,
                            out_dim=2,
                            bond_dim=bond_dim,
                            boundary=boundary)

        rank = torch.randint(3, 7, (1,)).item()
        mpo.canonicalize(oc=oc,
                         mode=mode,
                         rank=rank,
                         cum_percentage=0.98,
                         cutoff=1e-5,
                         renormalize=renormalize)

        self._assert_canonicalized_mpo_bond_dim(mpo, rank, mode)
        self._assert_mpo_leaf_nodes(mpo, n_features)
    
    def test_save_load_model(self):
        mpo = tk.models.MPO(n_features=100,
                            in_dim=2,
                            out_dim=2,
                            bond_dim=10,
                            boundary='obc',
                            init_method='randn')
        mpo.canonicalize(rank=5, renormalize=True)
        
        assert mpo.bond_dim == [5] * 99
        
        mpo_state_dict = mpo.state_dict()
        
        # Load new model from state_dict
        new_mpo = tk.models.MPO(n_features=100,
                                in_dim=2,
                                out_dim=2,
                                bond_dim=5,
                                boundary='obc')
        new_mpo.load_state_dict(mpo_state_dict)


class TestUMPO:  # MARK: TestUMPO
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    def test_initialize_with_tensors(self, n):
        tensor = torch.randn(10, 2, 10, 2)
        mpo = tk.models.UMPO(n_features=n, tensor=tensor)
        assert mpo.n_features == n
        assert mpo.boundary == 'pbc'
        assert mpo.in_dim == [2] * n
        assert mpo.out_dim == [2] * n
        assert mpo.bond_dim == [10] * n
    
    @pytest.mark.parametrize('runtime', DEVICE_RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    def test_initialize_with_tensors_runtime(self, runtime, n):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tensor_kwargs = _runtime_kwargs(runtime, device)
        tensor = torch.randn(10, 2, 10, 2, **tensor_kwargs)
        mpo = tk.models.UMPO(n_features=n, tensor=tensor)
        assert mpo.n_features == n
        assert mpo.boundary == 'pbc'
        assert mpo.in_dim == [2] * n
        assert mpo.out_dim == [2] * n
        assert mpo.bond_dim == [10] * n
        _assert_nodes_runtime(mpo.mats_env, runtime, device)
        
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
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('init_method', INIT_METHODS)
    def test_initialize_init_method(self, n, init_method):
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
    
    @pytest.mark.parametrize('runtime', DEVICE_RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('init_method', INIT_METHODS)
    def test_initialize_init_method_runtime(self, runtime, n, init_method):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_kwargs = _runtime_kwargs(runtime, device)
        mpo = tk.models.UMPO(n_features=n,
                             in_dim=2,
                             out_dim=2,
                             bond_dim=10,
                             init_method=init_method,
                             **model_kwargs)
        assert mpo.n_features == n
        assert mpo.boundary == 'pbc'
        assert mpo.in_dim == [2] * n
        assert mpo.out_dim == [2] * n
        assert mpo.bond_dim == [10] * n
        _assert_nodes_runtime(mpo.mats_env, runtime, device)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    def test_in_out_dims(self, n_features):
        in_dim = torch.randint(low=2, high=10, size=(1,)).item()
        out_dim = torch.randint(low=2, high=10, size=(1,)).item()

        mpo = tk.models.UMPO(n_features=n_features,
                             in_dim=in_dim,
                             out_dim=out_dim,
                             bond_dim=10)

        assert mpo.in_dim == [in_dim] * n_features
        assert mpo.out_dim == [out_dim] * n_features
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    def test_in_out_dims_error(self, n_features):
        # in_dim should be int.
        in_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        out_dim = torch.randint(low=2, high=10, size=(1,)).item()
        with pytest.raises(TypeError):
            tk.models.UMPO(n_features=n_features,
                           in_dim=in_dim,
                           out_dim=out_dim,
                           bond_dim=10)

        # out_dim should be int.
        in_dim = torch.randint(low=2, high=10, size=(1,)).item()
        out_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        with pytest.raises(TypeError):
            tk.models.UMPO(n_features=n_features,
                           in_dim=in_dim,
                           out_dim=out_dim,
                           bond_dim=10)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy(self, n_features, share_tensors):
        in_dim = torch.randint(low=2, high=12, size=(1,)).item()
        out_dim = torch.randint(low=2, high=12, size=(1,)).item()
        bond_dim = torch.randint(low=2, high=10, size=(1,)).item()

        mpo = tk.models.UMPO(n_features=n_features,
                             in_dim=in_dim,
                             out_dim=out_dim,
                             bond_dim=bond_dim)

        copied_mpo = mpo.copy(share_tensors=share_tensors)

        assert isinstance(copied_mpo, tk.models.UMPO)
        _assert_copied_mpo(mpo, copied_mpo, share_tensors)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('override', AUTO_BOOL_CASES)
    def test_deparameterize(self, n_features, override):
        in_dim = torch.randint(low=2, high=12, size=(1,)).item()
        out_dim = torch.randint(low=2, high=12, size=(1,)).item()
        bond_dim = torch.randint(low=2, high=10, size=(1,)).item()

        mpo = tk.models.UMPO(n_features=n_features,
                             in_dim=in_dim,
                             out_dim=out_dim,
                             bond_dim=bond_dim)

        non_param_mpo = mpo.parameterize(set_param=False, override=override)

        if override:
            assert non_param_mpo is mpo
        else:
            assert non_param_mpo is not mpo

        _assert_deparameterized_nodes(non_param_mpo.mats_env,
                                      tensor_address='virtual_uniform')
    
    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    def test_all_algorithms(self, n_features, auto_stack,
                            auto_unbind, inline_input, inline_mats):
        example = torch.randn(1, n_features, 2)
        data = torch.randn(100, n_features, 2)

        mpo = tk.models.UMPO(n_features=n_features,
                             in_dim=2,
                             out_dim=2,
                             bond_dim=10)
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
    
    @pytest.mark.parametrize(
        'n_features,mps_boundary,inline_input,inline_mats',
        UMPO_MPS_DATA_ALGORITHM_CASES,
    )
    def test_mpo_mps_data_all_algorithms(self, n_features, mps_boundary,
                                         inline_input, inline_mats):
        _run_umpo_mps_data_case(n_features, mps_boundary,
                                inline_input, inline_mats)
