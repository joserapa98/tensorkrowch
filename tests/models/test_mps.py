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

AUTO_BOOL_CASES = [True, False]
N_FEATURES_CASES = [1, 2, 3, 4, 10]
DIFF_N_FEATURES_CASES = [1, 2, 3, 4, 6]
BOUNDARY_CASES = ['obc', 'pbc']
RUNTIME_CASES = ['default', 'cuda', 'complex']
DEVICE_RUNTIME_CASES = ['default', 'cuda']
SMALL_N_FEATURES_CASES = [1, 2, 4]
SMALL_SPATIAL_CASES = [1, 2, 4]
INIT_N_CASES = [1, 2, 5]
MPS_INIT_METHODS = [
    'zeros', 'ones', 'copy', 'rand', 'randn', 'randn_eye', 'unit',
    'canonical'
]
MODEL_N_FEATURES_CASES = [1, 2, 3, 4, 6]
MPSDATA_INIT_METHODS = ['zeros', 'ones', 'copy', 'rand', 'randn']
CANONICALIZE_MODES = ['svd', 'svdr', 'qr']
CANONICALIZE_DIFF_BOND_MODES = ['svd', 'svdr']
CANONICALIZE_CASES = [
    (n_features, boundary, oc, mode, renormalize)
    for n_features in SMALL_N_FEATURES_CASES
    for boundary in BOUNDARY_CASES
    for oc in range(n_features)
    for mode in CANONICALIZE_MODES
    for renormalize in AUTO_BOOL_CASES
]
CANONICALIZE_DIFF_BOND_CASES = [
    (n_features, boundary, oc, mode, renormalize)
    for n_features in SMALL_N_FEATURES_CASES
    for boundary in BOUNDARY_CASES
    for oc in range(n_features)
    for mode in CANONICALIZE_DIFF_BOND_MODES
    for renormalize in AUTO_BOOL_CASES
]
ENTROPY_CASES = [
    (n_features, boundary, middle_site)
    for n_features in SMALL_N_FEATURES_CASES
    for boundary in BOUNDARY_CASES
    for middle_site in range(n_features - 1)
]
REDUCED_DENSITY_N_FEATURES_CASES = [1, 2, 3, 4, 5]
REDUCED_DENSITY_CASES = [
    (n_features, boundary)
    for n_features in REDUCED_DENSITY_N_FEATURES_CASES
    for boundary in BOUNDARY_CASES
]
NORM_CASES = [
    (n_features, boundary)
    for n_features in SMALL_N_FEATURES_CASES
    for boundary in BOUNDARY_CASES
]
UNIVOCAL_N_FEATURES_CASES = [1, 2, 3, 4, 5]
UNIVOCAL_RUNTIME_CASES = ['default', 'cuda', 'complex']
COPY_CONV_MPS_CASES = [
    (height, width, boundary, share_tensors)
    for height in SMALL_SPATIAL_CASES
    for width in SMALL_SPATIAL_CASES
    for boundary in BOUNDARY_CASES
    for share_tensors in AUTO_BOOL_CASES
]
COPY_CONV_UMPS_CASES = [
    (height, width, share_tensors)
    for height in SMALL_SPATIAL_CASES
    for width in SMALL_SPATIAL_CASES
    for share_tensors in AUTO_BOOL_CASES
]


def _runtime_kwargs(runtime, device):
    if runtime == 'cuda':
        return {'device': device}
    if runtime == 'complex':
        return {'dtype': torch.complex64}
    return {}


def _assert_nodes_runtime(nodes, runtime, device):
    for node in nodes:
        if runtime == 'cuda':
            assert node.device == device
        elif runtime == 'complex':
            assert node.dtype == torch.complex64
            assert node.is_complex()


def _assert_copied_mps(mps, copied_mps, share_tensors):
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


def _assert_boundary_vector(node):
    assert torch.equal(node.tensor[0], torch.ones_like(node.tensor)[0])
    assert torch.equal(node.tensor[1:], torch.zeros_like(node.tensor)[1:])


def _assert_deparameterized_nodes(nodes, tensor_address=None):
    for node in nodes:
        assert isinstance(node, tk.Node)
        assert not isinstance(node.tensor, torch.nn.Parameter)
        if tensor_address is not None:
            assert node.tensor_address() == tensor_address


def _deparameterize_even_nodes(model):
    expected_param_flags = []
    for i, node in enumerate(model.mats_env):
        set_param = (i % 2) == 1
        model._mats_env[i] = node.parameterize(set_param=set_param)
        expected_param_flags.append(set_param)
    return expected_param_flags


def _assert_parameterization_pattern(nodes, expected_param_flags):
    assert len(nodes) == len(expected_param_flags)
    for node, is_param in zip(nodes, expected_param_flags):
        assert isinstance(node, tk.ParamNode) == is_param
        assert isinstance(node.tensor, torch.nn.Parameter) == is_param


def _assert_nodes_device_and_dtype(nodes, device, dtype):
    for node in nodes:
        assert node.device == device
        assert node.dtype == dtype


def _assert_obc_boundary_nodes_are_non_parametric(model):
    assert isinstance(model.left_node, tk.Node)
    assert isinstance(model.right_node, tk.Node)
    assert not isinstance(model.left_node, tk.ParamNode)
    assert not isinstance(model.right_node, tk.ParamNode)


def _assert_obc_boundary_runtime(model, device, dtype):
    _assert_obc_boundary_nodes_are_non_parametric(model)
    _assert_nodes_device_and_dtype([model.left_node, model.right_node],
                                   device,
                                   dtype)


def _assert_mps_data_node_shapes(mps, n_batches, batch_size):
    if (mps.n_features == 1) and (mps.boundary == 'obc'):
        assert mps.mats_env[0].shape == tuple([batch_size] * n_batches + [1, 2, 1])
        return

    for node in mps.mats_env:
        assert node.shape == tuple([batch_size] * n_batches + [5, 2, 5])


class TestMPS:  # MARK: TestMPS

    def _get_runtime_kwargs(self, runtime, device):
        # Reuse the same test body for eager, CUDA and complex runs.
        return _runtime_kwargs(runtime, device)

    def _assert_mps_trace_state(self, mps, n_features):
        # After tracing with full input, the contracted MPS should keep the
        # same leaf/data node bookkeeping regardless of the contraction mode.
        if mps.boundary == 'obc':
            assert len(mps.leaf_nodes) == n_features + 2
        else:
            assert len(mps.leaf_nodes) == n_features
        assert len(mps.data_nodes) == n_features

    def _trace_mps_for_canonicalize(self,
                                    mps,
                                    n_features,
                                    runtime,
                                    inline_input=False,
                                    inline_mats=False):
        # Canonicalize/entropy tests always start from a fully traced MPS.
        example_kwargs = self._get_runtime_kwargs(runtime, mps.mats_env[0].device)

        mps_tensor = mps()
        assert mps_tensor.shape == (2,) * n_features

        mps.out_features = []
        example = torch.randn(1, n_features, 2, **example_kwargs)
        mps.trace(example,
                  inline_input=inline_input,
                  inline_mats=inline_mats)
        self._assert_mps_trace_state(mps, n_features)

    def _assert_canonicalized_bond_dim(self, mps, rank, mode):
        # QR does not enforce the same singular-value truncation bound.
        if not mps.bond_dim or mode == 'qr':
            return
        if mps.boundary == 'obc':
            assert (torch.tensor(mps.bond_dim) <= rank).all()
        else:
            assert (torch.tensor(mps.bond_dim[:-1]) <= rank).all()

    def _finalize_mps_canonicalize(self, mps, n_features):
        # Canonicalization must not leave the network in an unusable state.
        self._assert_mps_trace_state(mps, n_features)
        mps.unset_data_nodes()
        mps.in_features = []
        approx_mps_tensor = mps()
        assert approx_mps_tensor.shape == (2,) * n_features

    def _backward_maybe_complex(self, tensor, runtime):
        if runtime == 'complex':
            tensor.sum().abs().backward()
        else:
            tensor.sum().backward()

    def _trace_mps_with_features(self, mps, phys_dim, runtime):
        # Reduced-density tests trace only the selected input features.
        in_dims = [phys_dim[i] for i in mps.in_features]
        example_kwargs = self._get_runtime_kwargs(runtime, mps.mats_env[0].device)
        example = [torch.randn(1, d, **example_kwargs) for d in in_dims]
        if example == []:
            example = None

        mps.trace(example)

    def _assert_reduced_density(self, mps, phys_dim, trace_sites):
        # The reduced density should expose the traced sites as output features
        # and have the expected doubled Hilbert-space shape.
        assert mps.resultant_nodes
        if trace_sites:
            assert mps.data_nodes
        assert set(mps.in_features) == set(trace_sites)

        # MPS has to be reset, otherwise reduced_density automatically calls
        # the forward method that was traced when contracting the MPS with example
        mps.reset()

        density = mps.reduced_density(trace_sites)
        assert mps.resultant_nodes
        assert mps.data_nodes
        assert set(mps.out_features) == set(trace_sites)
        assert density.shape == tuple([phys_dim[i] for i in mps.in_features] * 2)
        return density

    def _sample_unique_features(self, n_features):
        in_features = torch.randint(low=0,
                                    high=n_features,
                                    size=(n_features // 2,)).tolist()
        in_features = sorted(set(in_features))
        return in_features

    def _trace_norm_mps(self, mps, in_features, runtime):
        # Norm tests operate on a partially traced MPS.
        example_kwargs = self._get_runtime_kwargs(runtime, mps.mats_env[0].device)
        example = torch.randn(1, len(in_features), 5, **example_kwargs)
        if example.numel() == 0:
            example = None

        mps.trace(example)
        assert mps.resultant_nodes
        if in_features:
            assert mps.data_nodes
        assert mps.in_features == in_features

    def _compute_norm(self, mps, in_features, log_scale, runtime):
        # MPS has to be reset, otherwise norm automatically calls
        # the forward method that was traced when contracting the MPS with example
        mps.reset()
        norm = mps.norm(log_scale=log_scale)
        assert mps.resultant_nodes
        assert not mps.data_nodes
        assert mps.in_features == in_features
        assert len(norm.shape) == 0

        self._backward_maybe_complex(norm, runtime)
        for node in mps.mats_env:
            assert node.grad is not None

        norm = mps.norm(log_scale=log_scale)
        return norm

    def _run_univocal_case(self, n_features, runtime, phys_dim, bond_dim,
                           atol):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        runtime_kwargs = self._get_runtime_kwargs(runtime, device)
        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=phys_dim,
                            bond_dim=bond_dim,
                            boundary='obc',
                            in_features=[],
                            init_method='unit',
                            **runtime_kwargs)

        expected_shape = tuple(phys_dim) if isinstance(phys_dim, list) \
            else (phys_dim,) * n_features
        mps_tensor = mps()
        assert mps_tensor.shape == expected_shape

        mps.out_features = []
        if isinstance(phys_dim, list):
            example = [torch.randn(1, d, **runtime_kwargs) for d in phys_dim]
        else:
            example = torch.randn(1, n_features, phys_dim, **runtime_kwargs)
        mps.trace(example)

        assert len(mps.leaf_nodes) == n_features + 2
        assert len(mps.data_nodes) == n_features

        mps.canonicalize_univocal()

        assert len(mps.leaf_nodes) == n_features + 2
        assert len(mps.data_nodes) == n_features

        mps.unset_data_nodes()
        mps.in_features = []
        approx_mps_tensor = mps()
        assert approx_mps_tensor.shape == expected_shape
        assert torch.allclose(mps_tensor, approx_mps_tensor,
                              rtol=1e-2, atol=atol)

    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_with_tensors(self, n, boundary):
        # Cover both boundary conventions with the same tensor construction.
        tensors = [torch.randn(10, 2, 10) for _ in range(n)]
        if boundary == 'obc':
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]

        mps = tk.models.MPS(tensors=tensors)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
    
    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_with_tensors_runtime(self, runtime, n, boundary):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tensor_kwargs = _runtime_kwargs(runtime, device)
        tensors = [torch.randn(10, 2, 10, **tensor_kwargs) for _ in range(n)]
        if boundary == 'obc':
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]

        mps = tk.models.MPS(tensors=tensors)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
        _assert_nodes_runtime(mps.mats_env, runtime, device)
    
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
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('init_method', MPS_INIT_METHODS)
    def test_initialize_init_method(self, n, boundary, init_method):
        # All init methods should preserve the requested metadata.
        mps = tk.models.MPS(boundary=boundary,
                            n_features=n,
                            phys_dim=2,
                            bond_dim=5,
                            init_method=init_method)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [5] * (n - 1 if boundary == 'obc' else n)
        if boundary == 'obc':
            _assert_obc_boundary_nodes_are_non_parametric(mps)
            _assert_boundary_vector(mps.left_node)
            _assert_boundary_vector(mps.right_node)
    
    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('init_method', MPS_INIT_METHODS)
    def test_initialize_init_method_runtime(self, runtime, n, boundary,
                                            init_method):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_kwargs = _runtime_kwargs(runtime, device)
        mps = tk.models.MPS(boundary=boundary,
                            n_features=n,
                            phys_dim=2,
                            bond_dim=5,
                            init_method=init_method,
                            **model_kwargs)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [5] * (n - 1 if boundary == 'obc' else n)
        _assert_nodes_runtime(mps.mats_env, runtime, device)
        if boundary == 'obc':
            _assert_obc_boundary_nodes_are_non_parametric(mps)
            _assert_boundary_vector(mps.left_node)
            _assert_boundary_vector(mps.right_node)
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_canonical(self, n, boundary):
        # Canonical init keeps the requested shape and, for OBC, the expected norm.
        mps = tk.models.MPS(boundary=boundary,
                            n_features=n,
                            phys_dim=2,
                            bond_dim=2,
                            init_method='canonical')
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [2] * (n - 1 if boundary == 'obc' else n)
        if boundary == 'obc':
            assert mps.norm().isclose(torch.tensor(2. ** n).sqrt())
    
    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_canonical_runtime(self, runtime, n, boundary):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_kwargs = _runtime_kwargs(runtime, device)
        mps = tk.models.MPS(boundary=boundary,
                            n_features=n,
                            phys_dim=2,
                            bond_dim=2,
                            init_method='canonical',
                            **model_kwargs)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [2] * (n - 1 if boundary == 'obc' else n)
        _assert_nodes_runtime(mps.mats_env, runtime, device)
        if boundary == 'obc':
            assert mps.norm().isclose(torch.tensor(2. ** n).sqrt())
    
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
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_phys_dims(self, n_features, boundary):
        phys_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=phys_dim,
                            bond_dim=10,
                            boundary=boundary)

        assert mps.phys_dim == phys_dim
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_phys_dims_error(self, n_features, boundary):
        # phys_dim should have n_features elements.
        phys_dim = torch.randint(low=2, high=10, size=(n_features + 1,)).tolist()
        with pytest.raises(ValueError):
            tk.models.MPS(n_features=n_features,
                          phys_dim=phys_dim,
                          bond_dim=10,
                          boundary=boundary)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_bond_dims(self, n_features, boundary):
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
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy(self, n_features, boundary, share_tensors):
        phys_dim = torch.randint(low=2, high=12, size=(n_features,)).tolist()
        bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=phys_dim,
                            bond_dim=bond_dim,
                            boundary=boundary)

        copied_mps = mps.copy(share_tensors=share_tensors)

        assert isinstance(copied_mps, tk.models.MPS)
        _assert_copied_mps(mps, copied_mps, share_tensors)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('override', AUTO_BOOL_CASES)
    def test_deparameterize(self, n_features, boundary, override):
        phys_dim = torch.randint(low=2, high=12, size=(n_features,)).tolist()
        bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=phys_dim,
                            bond_dim=bond_dim,
                            boundary=boundary)

        non_param_mps = mps.parameterize(set_param=False, override=override)

        if override:
            assert non_param_mps is mps
        else:
            assert non_param_mps is not mps

        new_nodes = non_param_mps.mats_env[:]
        if boundary == 'obc':
            new_nodes += [non_param_mps.left_node, non_param_mps.right_node]

        _assert_deparameterized_nodes(new_nodes)

    @pytest.mark.parametrize('n_features', INIT_N_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy_preserves_boundary_dtype(self, n_features, share_tensors):
        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=3,
                            bond_dim=4,
                            boundary='obc',
                            dtype=torch.complex64)

        copied_mps = mps.copy(share_tensors=share_tensors)

        assert copied_mps.left_node.dtype == torch.complex64
        assert copied_mps.right_node.dtype == torch.complex64
        assert copied_mps.mats_env[0].dtype == torch.complex64

    @pytest.mark.parametrize('n_features', INIT_N_CASES)
    def test_deparameterize_preserves_boundary_runtime(self, n_features):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dtype = torch.complex64

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=3,
                            bond_dim=4,
                            boundary='obc').to(device=device, dtype=dtype)

        non_param_mps = mps.parameterize(set_param=False, override=False)

        _assert_nodes_device_and_dtype(non_param_mps.mats_env, device, dtype)
        _assert_obc_boundary_runtime(non_param_mps, device, dtype)

    @pytest.mark.parametrize('n_features', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_to(self, n_features, boundary):
        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=5,
                            bond_dim=2,
                            boundary=boundary)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float64

        returned_mps = mps.to(device=device, dtype=dtype)
        assert returned_mps is mps

        _assert_nodes_device_and_dtype(mps.mats_env, device, dtype)
        if boundary == 'obc':
            _assert_obc_boundary_nodes_are_non_parametric(mps)
            _assert_nodes_device_and_dtype([mps.left_node, mps.right_node],
                                           device,
                                           dtype)

    @pytest.mark.parametrize('n_features', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_to_contracted(self, n_features, boundary):
        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=5,
                            bond_dim=2,
                            boundary=boundary)

        example = torch.randn(1, n_features, 5)
        data = torch.randn(2, n_features, 5)

        mps.trace(example)
        _ = mps(data)

        assert mps.resultant_nodes
        tensor_nodes = [node for node in mps.nodes.values()
                        if node.tensor is not None]
        assert any(node.is_resultant() for node in tensor_nodes)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float64

        mps.to(device=device, dtype=dtype)

        _assert_nodes_device_and_dtype(tensor_nodes, device, dtype)
        _assert_nodes_device_and_dtype(mps.mats_env, device, dtype)
        if boundary == 'obc':
            _assert_obc_boundary_nodes_are_non_parametric(mps)
            _assert_nodes_device_and_dtype([mps.left_node, mps.right_node],
                                           device,
                                           dtype)
    
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

    @staticmethod
    def _runtime_kwargs(runtime):
        tensor_kwargs = {}
        model_kwargs = {}
        is_complex = runtime == 'complex'

        if runtime == 'cuda':
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            tensor_kwargs['device'] = device
            model_kwargs['device'] = device
        elif is_complex:
            tensor_kwargs['dtype'] = torch.complex64
            model_kwargs['dtype'] = torch.complex64

        return tensor_kwargs, model_kwargs, is_complex

    @staticmethod
    def _backward(result, is_complex):
        if is_complex:
            result.sum().abs().backward()
        else:
            result.sum().backward()

    @staticmethod
    def _sample_unique_phys_dim(n_features):
        phys_dim = torch.randint(low=2, high=12, size=(n_features,)).tolist()
        while len(phys_dim) > len(set(phys_dim)):
            phys_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        return phys_dim

    @staticmethod
    def _sample_in_features(n_features):
        return list(set(torch.randint(low=0,
                                      high=n_features,
                                      size=(n_features // 2,)).tolist()))

    @staticmethod
    def _make_inputs(phys_dim, batch_size, tensor_kwargs):
        if isinstance(phys_dim, int):
            return torch.randn(batch_size, 0 if batch_size is None else 1, 1)
        return [torch.randn(batch_size, d, **tensor_kwargs) for d in phys_dim]

    @staticmethod
    def _leaf_nodes(boundary, n_features):
        return n_features + 2 if boundary == 'obc' else n_features

    @staticmethod
    def _virtual_nodes_standard(auto_stack, inline_input):
        return 2 if not inline_input and auto_stack else 1

    @staticmethod
    def _virtual_nodes_diff_in_dim(n_features, auto_stack, inline_input):
        if not inline_input and auto_stack:
            return 2 if n_features == 1 else 1
        return 1 if n_features == 1 else 0

    @staticmethod
    def _virtual_nodes_marginalize(in_features, out_features_len,
                                   auto_stack, inline_input,
                                   embedding_multiplier):
        if in_features:
            base = 2 if not inline_input and auto_stack else 1
            return base + embedding_multiplier * out_features_len
        return embedding_multiplier * out_features_len

    @staticmethod
    def _virtual_nodes_no_marginalize(in_features, auto_stack, inline_input):
        if not in_features:
            return 0
        return 2 if not inline_input and auto_stack else 1

    @staticmethod
    def _tensor_inputs(n_features, phys_dim, tensor_kwargs):
        example = torch.randn(1, n_features, phys_dim, **tensor_kwargs)
        data = torch.randn(100, n_features, phys_dim, **tensor_kwargs)
        return example, data

    @staticmethod
    def _list_inputs(phys_dim, tensor_kwargs):
        example = [torch.randn(1, d, **tensor_kwargs) for d in phys_dim]
        data = [torch.randn(100, d, **tensor_kwargs) for d in phys_dim]
        return example, data

    @staticmethod
    def _partial_inputs(length, tensor_kwargs):
        example = torch.randn(1, length, 5, **tensor_kwargs)
        data = torch.randn(100, length, 5, **tensor_kwargs)
        if example.numel() == 0:
            return None, None
        return example, data

    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
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
        tensor_kwargs, model_kwargs, is_complex = self._runtime_kwargs(runtime)
        example, data = self._tensor_inputs(n_features, 5, tensor_kwargs)

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=5,
                            bond_dim=2,
                            boundary=boundary,
                            **model_kwargs)
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
        assert len(mps.leaf_nodes) == self._leaf_nodes(boundary, n_features)
        assert len(mps.data_nodes) == n_features
        assert len(mps.virtual_nodes) == self._virtual_nodes_standard(
            auto_stack, inline_input
        )

        self._backward(result, is_complex)
        for node in mps.mats_env:
            assert node.grad is not None

    @pytest.mark.parametrize('n_features', DIFF_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms_diff_in_dim(self, n_features, boundary,
                                        auto_stack, auto_unbind,
                                        inline_input, inline_mats,
                                        renormalize):
        phys_dim = self._sample_unique_phys_dim(n_features)
        example, data = self._list_inputs(phys_dim, {})

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=phys_dim,
                            bond_dim=2,
                            boundary=boundary)
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
        assert len(mps.leaf_nodes) == self._leaf_nodes(boundary, n_features)
        assert len(mps.data_nodes) == n_features
        assert len(mps.virtual_nodes) == self._virtual_nodes_diff_in_dim(
            n_features, auto_stack, inline_input
        )

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None

    @pytest.mark.parametrize('n_features', DIFF_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms_diff_bond_dim(self, n_features, boundary,
                                          auto_stack, auto_unbind,
                                          inline_input, inline_mats,
                                          renormalize):
        bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
        example, data = self._tensor_inputs(n_features, 5, {})

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=5,
                            bond_dim=bond_dim,
                            boundary=boundary)
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
        assert len(mps.leaf_nodes) == self._leaf_nodes(boundary, n_features)
        assert len(mps.data_nodes) == n_features
        assert len(mps.virtual_nodes) == self._virtual_nodes_standard(
            auto_stack, inline_input
        )

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None

    @pytest.mark.parametrize('n_features', DIFF_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms_diff_in_dim_bond_dim(self, n_features, boundary,
                                                 auto_stack, auto_unbind,
                                                 inline_input, inline_mats,
                                                 renormalize):
        phys_dim = self._sample_unique_phys_dim(n_features)
        bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim
        example, data = self._list_inputs(phys_dim, {})

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=phys_dim,
                            bond_dim=bond_dim,
                            boundary=boundary)
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
        assert len(mps.leaf_nodes) == self._leaf_nodes(boundary, n_features)
        assert len(mps.data_nodes) == n_features
        assert len(mps.virtual_nodes) == self._virtual_nodes_diff_in_dim(
            n_features, auto_stack, inline_input
        )

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None

    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms_marginalize(self, n_features, boundary,
                                        auto_stack, auto_unbind,
                                        inline_input, inline_mats,
                                        renormalize):
        in_features = self._sample_in_features(n_features)
        example, data = self._partial_inputs(len(in_features), {})

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=5,
                            bond_dim=2,
                            boundary=boundary,
                            in_features=in_features)
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
        else:
            assert result.shape == tuple()

        assert len(mps.virtual_nodes) == self._virtual_nodes_marginalize(
            in_features, len(mps.out_features), auto_stack, inline_input, 1
        )
        assert len(mps.leaf_nodes) == self._leaf_nodes(boundary, n_features)
        assert len(mps.data_nodes) == len(in_features)

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None

    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms_marginalize_with_list_matrices(
            self, runtime, n_features, boundary, auto_stack, auto_unbind,
            inline_input, inline_mats, renormalize):
        tensor_kwargs, model_kwargs, is_complex = self._runtime_kwargs(runtime)
        in_features = self._sample_in_features(n_features)
        example, data = self._partial_inputs(len(in_features), tensor_kwargs)

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=5,
                            bond_dim=2,
                            boundary=boundary,
                            in_features=in_features,
                            **model_kwargs)
        embedding_matrices = [
            torch.randn(5, 5, **tensor_kwargs) for _ in range(len(mps.out_features))
        ]
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
        else:
            assert result.shape == tuple()

        assert len(mps.virtual_nodes) == self._virtual_nodes_marginalize(
            in_features, len(mps.out_features), auto_stack, inline_input, 2
        )
        assert len(mps.leaf_nodes) == self._leaf_nodes(boundary, n_features)
        assert len(mps.data_nodes) == len(in_features)

        self._backward(result, is_complex)
        for node in mps.mats_env:
            assert node.grad is not None

    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms_marginalize_with_matrix(
            self, runtime, n_features, boundary, auto_stack, auto_unbind,
            inline_input, inline_mats, renormalize):
        tensor_kwargs, model_kwargs, is_complex = self._runtime_kwargs(runtime)
        in_features = self._sample_in_features(n_features)
        example, data = self._partial_inputs(len(in_features), tensor_kwargs)

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=5,
                            bond_dim=2,
                            boundary=boundary,
                            in_features=in_features,
                            **model_kwargs)
        embedding_matrix = torch.randn(5, 5, **tensor_kwargs)
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
        else:
            assert result.shape == tuple()

        assert len(mps.virtual_nodes) == self._virtual_nodes_marginalize(
            in_features, len(mps.out_features), auto_stack, inline_input, 2
        )
        assert len(mps.leaf_nodes) == self._leaf_nodes(boundary, n_features)
        assert len(mps.data_nodes) == len(in_features)

        self._backward(result, is_complex)
        for node in mps.mats_env:
            assert node.grad is not None

    @staticmethod
    def _leaf_nodes_with_mpo(n_features, in_features_len,
                             mps_boundary, mpo_boundary):
        mps_leaf_nodes = n_features + 2 if mps_boundary == 'obc' else n_features
        mpo_n_features = n_features - in_features_len
        mpo_leaf_nodes = mpo_n_features + 2 if mpo_boundary == 'obc' else mpo_n_features
        return mps_leaf_nodes + mpo_leaf_nodes

    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    @pytest.mark.parametrize('mps_boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('mpo_boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms_marginalize_with_mpo(
            self, runtime, n_features, mps_boundary, mpo_boundary,
            auto_stack, auto_unbind, inline_input, inline_mats,
            renormalize):
        tensor_kwargs, model_kwargs, is_complex = self._runtime_kwargs(runtime)
        in_features = self._sample_in_features(n_features)
        example, data = self._partial_inputs(len(in_features), tensor_kwargs)

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=5,
                            bond_dim=2,
                            boundary=mps_boundary,
                            in_features=in_features,
                            **model_kwargs)
        mpo = tk.models.MPO(n_features=n_features - len(in_features),
                            in_dim=5,
                            out_dim=5,
                            bond_dim=2,
                            boundary=mpo_boundary,
                            **model_kwargs)
        mpo = mpo.parameterize(set_param=False, override=True)
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

        assert len(mps.leaf_nodes) == self._leaf_nodes_with_mpo(
            n_features, len(in_features), mps_boundary, mpo_boundary
        )

        self._backward(result, is_complex)
        for node in mps.mats_env:
            assert node.grad is not None
        for node in mpo.mats_env:
            assert node.tensor.grad is None

    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms_no_marginalize(self, n_features, boundary,
                                           auto_stack, auto_unbind,
                                           inline_input, inline_mats,
                                           renormalize):
        in_features = self._sample_in_features(n_features)
        example, data = self._partial_inputs(len(in_features), {})

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=5,
                            bond_dim=2,
                            boundary=boundary,
                            in_features=in_features)
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

        assert len(mps.virtual_nodes) == self._virtual_nodes_no_marginalize(
            in_features, auto_stack, inline_input
        )
        assert len(mps.edges) == len(mps.out_features)
        assert len(mps.leaf_nodes) == self._leaf_nodes(boundary, n_features)
        assert len(mps.data_nodes) == len(in_features)

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None
    
    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n_features,boundary', NORM_CASES)
    def test_norm(self, runtime, n_features, boundary):
        # Check both raw norm and log-norm paths on the same partially traced MPS.
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        runtime_kwargs = self._get_runtime_kwargs(runtime, device)
        in_features = self._sample_unique_features(n_features)

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=5,
                            bond_dim=2,
                            boundary=boundary,
                            in_features=in_features,
                            **runtime_kwargs)
        self._trace_norm_mps(mps, in_features, runtime)

        norms = [
            self._compute_norm(mps, in_features, log_scale, runtime)
            for log_scale in AUTO_BOOL_CASES
        ]
        assert torch.isclose(norms[0].exp(), norms[1])
     
    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n_features,boundary', REDUCED_DENSITY_CASES)
    def test_reduced_density(self, runtime, n_features, boundary):
        # Exercise reduced-density on eager/CUDA/complex tensors with random traced sites.
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        runtime_kwargs = self._get_runtime_kwargs(runtime, device)
        phys_dim = torch.randint(low=2, high=6, size=(n_features,)).tolist()
        bond_dim = torch.randint(low=2, high=4, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        trace_sites = torch.randint(low=0,
                                    high=n_features,
                                    size=(n_features // 2,)).tolist()

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=phys_dim,
                            bond_dim=bond_dim,
                            boundary=boundary,
                            in_features=trace_sites,
                            **runtime_kwargs)

        self._trace_mps_with_features(mps, phys_dim, runtime)
        density = self._assert_reduced_density(mps, phys_dim, trace_sites)
        self._backward_maybe_complex(density, runtime)
        for node in mps.mats_env:
            assert node.grad is not None

        density = mps.reduced_density(trace_sites)

    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n_features,boundary,middle_site', ENTROPY_CASES)
    def test_entropy(self, runtime, n_features, boundary, middle_site):
        # Compare the renormalized entropy output with the non-renormalized one.
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        runtime_kwargs = self._get_runtime_kwargs(runtime, device)
        bond_dim = torch.randint(low=2, high=6, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=2,
                            bond_dim=bond_dim,
                            boundary=boundary,
                            in_features=[],
                            init_method='canonical',
                            **runtime_kwargs)

        self._trace_mps_for_canonicalize(mps, n_features, runtime)

        scaled_entropy, log_norm = mps.entropy(middle_site=middle_site,
                                               renormalize=True)
        entropy = mps.entropy(middle_site=middle_site, renormalize=False)

        assert all(mps.bond_dim[i] <= bond_dim[i] for i in range(len(bond_dim)))

        sq_norm = log_norm.exp().pow(2)
        approx_entropy = sq_norm * scaled_entropy - sq_norm * 2 * log_norm
        assert torch.isclose(entropy, approx_entropy, rtol=1e-03, atol=1e-05)

        self._finalize_mps_canonicalize(mps, n_features)

    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n_features,boundary,middle_site', ENTROPY_CASES)
    def test_entropy_preserves_mixed_parameterization(self,
                                                      runtime,
                                                      n_features,
                                                      boundary,
                                                      middle_site):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        runtime_kwargs = self._get_runtime_kwargs(runtime, device)

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=2,
                            bond_dim=6,
                            boundary=boundary,
                            in_features=[],
                            **runtime_kwargs)

        expected_param_flags = _deparameterize_even_nodes(mps)
        self._trace_mps_for_canonicalize(mps,
                                         n_features,
                                         runtime,
                                         inline_input=True,
                                         inline_mats=True)

        mps.entropy(middle_site=middle_site, renormalize=False)

        _assert_parameterization_pattern(mps.mats_env, expected_param_flags)
    
    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize(
        'n_features,boundary,oc,mode,renormalize',
        CANONICALIZE_CASES,
    )
    def test_canonicalize(self, runtime, n_features, boundary, oc,
                          mode, renormalize):
        # Cover all canonicalization modes and ensure the post-state can still
        # be evaluated after unsetting traced data.
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        runtime_kwargs = {}
        if runtime == 'cuda':
            runtime_kwargs['device'] = device
        elif runtime == 'complex':
            runtime_kwargs['dtype'] = torch.complex64

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=2,
                            bond_dim=6,
                            boundary=boundary,
                            in_features=[],
                            **runtime_kwargs)

        self._trace_mps_for_canonicalize(mps, n_features, runtime)

        rank = torch.randint(3, 7, (1,)).item()
        mps.canonicalize(oc=oc,
                         mode=mode,
                         rank=rank,
                         cum_percentage=0.98,
                         cutoff=1e-5,
                         renormalize=renormalize)

        self._assert_canonicalized_bond_dim(mps, rank, mode)
        self._finalize_mps_canonicalize(mps, n_features)

    @pytest.mark.parametrize(
        'runtime,n_features,boundary,oc,mode,renormalize',
        [
            (runtime, n_features, boundary, oc, mode, renormalize)
            for runtime in RUNTIME_CASES
            for n_features, boundary, oc, mode, renormalize in CANONICALIZE_CASES
        ],
    )
    def test_canonicalize_preserves_mixed_parameterization(self,
                                                           runtime,
                                                           n_features,
                                                           boundary,
                                                           oc,
                                                           mode,
                                                           renormalize):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        runtime_kwargs = self._get_runtime_kwargs(runtime, device)

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=2,
                            bond_dim=6,
                            boundary=boundary,
                            in_features=[],
                            **runtime_kwargs)

        expected_param_flags = _deparameterize_even_nodes(mps)
        self._trace_mps_for_canonicalize(mps,
                                         n_features,
                                         runtime,
                                         inline_input=True,
                                         inline_mats=True)

        rank = torch.randint(3, 7, (1,)).item()
        mps.canonicalize(oc=oc,
                         mode=mode,
                         rank=rank,
                         cum_percentage=0.98,
                         cutoff=1e-5,
                         renormalize=renormalize)

        _assert_parameterization_pattern(mps.mats_env, expected_param_flags)
        self._assert_canonicalized_bond_dim(mps, rank, mode)

    @pytest.mark.parametrize(
        'n_features,boundary,oc,mode,renormalize',
        CANONICALIZE_DIFF_BOND_CASES,
    )
    def test_canonicalize_diff_bond_dims(self, n_features, boundary, oc,
                                         mode, renormalize):
        # The variable-bond case should never increase the original bond sizes.
        bond_dim = torch.randint(low=2, high=6, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=2,
                            bond_dim=bond_dim,
                            boundary=boundary,
                            in_features=[])

        self._trace_mps_for_canonicalize(mps, n_features, 'default')

        mps.canonicalize(oc=oc,
                         mode=mode,
                         renormalize=renormalize)

        assert all(mps.bond_dim[i] <= bond_dim[i] for i in range(len(bond_dim)))
        self._finalize_mps_canonicalize(mps, n_features)

    def test_canonicalize_linalg_error_breaks_tensors_access(self):
        # Non-finite values make ``torch.linalg.svd`` fail, which currently
        # leaves OBC boundary tensors unusable for ``mps.tensors`` afterwards.
        mps = tk.models.MPS(n_features=3,
                            phys_dim=2,
                            bond_dim=4,
                            boundary='obc',
                            in_features=[])
        self._trace_mps_for_canonicalize(mps, 3, 'default')

        with torch.no_grad():
            mps.mats_env[1].tensor[0, 0, 0] = float('nan')

        with pytest.raises(torch.linalg.LinAlgError):
            mps.canonicalize(mode='svd')
        
        # This currently breaks because canonicalize contracts and splits nodes
        # in-place. If the SVD fails after an in-place contraction, the original
        # tensors are no longer fully recoverable from the current state.
        with pytest.raises(TypeError):
            _ = mps.tensors
    
    @pytest.mark.parametrize('runtime', UNIVOCAL_RUNTIME_CASES)
    @pytest.mark.parametrize('n_features', UNIVOCAL_N_FEATURES_CASES)
    def test_canonicalize_univocal(self, runtime, n_features):
        atol = 1e-3 if runtime == 'default' else 1e-4
        self._run_univocal_case(n_features, runtime, 2, 10, atol)

    @pytest.mark.parametrize('n_features', UNIVOCAL_N_FEATURES_CASES)
    def test_canonicalize_univocal_diff_dims(self, n_features):
        phys_dim = torch.arange(2, 2 + n_features).int().tolist()
        bond_dim = torch.arange(2, 1 + n_features).int().tolist()
        self._run_univocal_case(n_features, 'default', phys_dim, bond_dim, 1e-4)

    @pytest.mark.parametrize('n_features', UNIVOCAL_N_FEATURES_CASES)
    def test_canonicalize_univocal_bond_greater_than_phys(self, n_features):
        self._run_univocal_case(n_features, 'default', 2, 100, 1e-4)
    
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
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    def test_initialize_with_tensors(self, n):
        tensor = torch.randn(10, 2, 10)
        mps = tk.models.UMPS(n_features=n, tensor=tensor)
        assert mps.n_features == n
        assert mps.boundary == 'pbc'
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [10] * n
    
    @pytest.mark.parametrize('runtime', DEVICE_RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    def test_initialize_with_tensors_runtime(self, runtime, n):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tensor_kwargs = _runtime_kwargs(runtime, device)
        tensor = torch.randn(10, 2, 10, **tensor_kwargs)
        mps = tk.models.UMPS(n_features=n, tensor=tensor)
        assert mps.n_features == n
        assert mps.boundary == 'pbc'
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [10] * n
        _assert_nodes_runtime(mps.mats_env, runtime, device)
    
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
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('init_method', MPS_INIT_METHODS)
    def test_initialize_init_method(self, n, init_method):
        mps = tk.models.UMPS(n_features=n,
                             phys_dim=2,
                             bond_dim=5,
                             init_method=init_method)
        assert mps.n_features == n
        assert mps.boundary == 'pbc'
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [5] * n
    
    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('init_method', MPS_INIT_METHODS)
    def test_initialize_init_method_runtime(self, runtime, n, init_method):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_kwargs = _runtime_kwargs(runtime, device)
        mps = tk.models.UMPS(n_features=n,
                             phys_dim=2,
                             bond_dim=5,
                             init_method=init_method,
                             **model_kwargs)
        assert mps.n_features == n
        assert mps.boundary == 'pbc'
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [5] * n
        _assert_nodes_runtime(mps.mats_env, runtime, device)
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    def test_initialize_with_unitaries(self, n):
        mps = tk.models.UMPS(n_features=n,
                             phys_dim=2,
                             bond_dim=2,
                             init_method='unit')
        assert mps.n_features == n
        assert mps.boundary == 'pbc'
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [2] * n
    
    @pytest.mark.parametrize('runtime', DEVICE_RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    def test_initialize_with_unitaries_runtime(self, runtime, n):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_kwargs = _runtime_kwargs(runtime, device)
        mps = tk.models.UMPS(n_features=n,
                             phys_dim=2,
                             bond_dim=2,
                             init_method='unit',
                             **model_kwargs)
        assert mps.n_features == n
        assert mps.boundary == 'pbc'
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [2] * n
        _assert_nodes_runtime(mps.mats_env, runtime, device)
    
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
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy(self, n_features, share_tensors):
        phys_dim = torch.randint(low=2, high=12, size=(1,)).item()
        bond_dim = torch.randint(low=2, high=10, size=(1,)).item()

        mps = tk.models.UMPS(n_features=n_features,
                             phys_dim=phys_dim,
                             bond_dim=bond_dim)

        copied_mps = mps.copy(share_tensors=share_tensors)

        assert isinstance(copied_mps, tk.models.UMPS)
        _assert_copied_mps(mps, copied_mps, share_tensors)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('override', AUTO_BOOL_CASES)
    def test_deparameterize(self, n_features, override):
        phys_dim = torch.randint(low=2, high=12, size=(1,)).item()
        bond_dim = torch.randint(low=2, high=10, size=(1,)).item()

        mps = tk.models.UMPS(n_features=n_features,
                             phys_dim=phys_dim,
                             bond_dim=bond_dim)

        non_param_mps = mps.parameterize(set_param=False, override=override)

        if override:
            assert non_param_mps is mps
        else:
            assert non_param_mps is not mps

        _assert_deparameterized_nodes(non_param_mps.mats_env,
                                      tensor_address='virtual_uniform')

    @pytest.mark.parametrize('n_features', INIT_N_CASES)
    def test_to(self, n_features):
        mps = tk.models.UMPS(n_features=n_features,
                             phys_dim=5,
                             bond_dim=2)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float64

        mps.to(device=device, dtype=dtype)

        _assert_nodes_device_and_dtype(mps.mats_env, device, dtype)
        assert mps.uniform_memory.device == device
        assert mps.uniform_memory.dtype == dtype

    @pytest.mark.parametrize('n_features', INIT_N_CASES)
    def test_to_contracted(self, n_features):
        mps = tk.models.UMPS(n_features=n_features,
                             phys_dim=5,
                             bond_dim=2)

        example = torch.randn(1, n_features, 5)
        data = torch.randn(2, n_features, 5)

        mps.trace(example)
        _ = mps(data)

        assert mps.resultant_nodes
        tensor_nodes = [node for node in mps.nodes.values()
                        if node.tensor is not None]
        assert any(node.is_resultant() for node in tensor_nodes)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float64

        mps.to(device=device, dtype=dtype)

        _assert_nodes_device_and_dtype(tensor_nodes, device, dtype)
        _assert_nodes_device_and_dtype(mps.mats_env, device, dtype)
        assert mps.uniform_memory.device == device
        assert mps.uniform_memory.dtype == dtype

    @staticmethod
    def _sample_in_features(n_features):
        return list(set(torch.randint(low=0,
                                      high=n_features,
                                      size=(n_features // 2,)).tolist()))

    @staticmethod
    def _partial_inputs(length):
        example = torch.randn(1, length, 5)
        data = torch.randn(100, length, 5)
        if example.numel() == 0:
            return None, None
        return example, data

    @staticmethod
    def _runtime_kwargs(runtime, device):
        if runtime == 'cuda':
            return {'device': device}
        if runtime == 'complex':
            return {'dtype': torch.complex64}
        return {}

    @staticmethod
    def _backward_maybe_complex(tensor, runtime):
        if runtime == 'complex':
            tensor.sum().abs().backward()
        else:
            tensor.sum().backward()

    def _trace_umps_norm(self, mps, in_features, runtime):
        example_kwargs = self._runtime_kwargs(runtime, mps.mats_env[0].device)
        example = torch.randn(1, len(in_features), 5, **example_kwargs)
        if example.numel() == 0:
            example = None

        mps.trace(example)
        assert mps.resultant_nodes
        if in_features:
            assert mps.data_nodes
        assert mps.in_features == in_features

    def _compute_umps_norm(self, mps, in_features, log_scale, runtime):
        # Reset before calling norm to avoid reusing the traced forward graph.
        mps.reset()
        norm = mps.norm(log_scale=log_scale)
        assert mps.resultant_nodes
        assert not mps.data_nodes
        assert mps.in_features == in_features
        assert len(norm.shape) == 0

        self._backward_maybe_complex(norm, runtime)
        for node in mps.mats_env:
            assert node.grad is not None

        return mps.norm(log_scale=log_scale)

    def _trace_umps_density(self, mps, phys_dim, n_trace_sites, runtime):
        example_kwargs = self._runtime_kwargs(runtime, mps.mats_env[0].device)
        example = torch.randn(1, n_trace_sites, phys_dim, **example_kwargs)
        if example.numel() == 0:
            example = None
        mps.trace(example)

    def _assert_umps_reduced_density(self, mps, phys_dim, trace_sites):
        assert mps.resultant_nodes
        if trace_sites:
            assert mps.data_nodes
        assert set(mps.in_features) == set(trace_sites)

        # Reset before reduced_density because it traces its own contraction path.
        mps.reset()
        density = mps.reduced_density(trace_sites)
        assert mps.resultant_nodes
        assert mps.data_nodes
        assert set(mps.out_features) == set(trace_sites)
        assert density.shape == (phys_dim,) * 2 * len(mps.in_features)
        return density

    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms(self, n_features, auto_stack, auto_unbind,
                            inline_input, inline_mats, renormalize):
        example = torch.randn(1, n_features, 5)
        data = torch.randn(100, n_features, 5)

        mps = tk.models.UMPS(n_features=n_features,
                             phys_dim=5,
                             bond_dim=2)
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

    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms_marginalize(self, n_features, auto_stack,
                                        auto_unbind, inline_input,
                                        inline_mats, renormalize):
        in_features = self._sample_in_features(n_features)
        example, data = self._partial_inputs(len(in_features))

        mps = tk.models.UMPS(n_features=n_features,
                             phys_dim=5,
                             bond_dim=2,
                             in_features=in_features)
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
            assert len(mps.virtual_nodes) == 2 + len(mps.out_features)
        else:
            assert result.shape == tuple()
            assert len(mps.virtual_nodes) == 1 + len(mps.out_features)

        assert len(mps.leaf_nodes) == n_features
        assert len(mps.data_nodes) == len(in_features)

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None
        assert mps.uniform_memory.grad is not None

    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('renormalize', AUTO_BOOL_CASES)
    def test_all_algorithms_no_marginalize(self, n_features, auto_stack,
                                           auto_unbind, inline_input,
                                           inline_mats, renormalize):
        in_features = self._sample_in_features(n_features)
        example, data = self._partial_inputs(len(in_features))

        mps = tk.models.UMPS(n_features=n_features,
                             phys_dim=5,
                             bond_dim=2,
                             in_features=in_features)
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
            assert len(mps.virtual_nodes) == 2
        else:
            assert len(mps.virtual_nodes) == 1
        assert result.shape == tuple(aux_shape)
        assert len(mps.edges) == len(mps.out_features)
        assert len(mps.leaf_nodes) == n_features
        assert len(mps.data_nodes) == len(in_features)

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None
        assert mps.uniform_memory.grad is not None
    
    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n_features', N_FEATURES_CASES)
    def test_norm(self, runtime, n_features):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        runtime_kwargs = self._runtime_kwargs(runtime, device)
        in_features = sorted(set(torch.randint(low=0,
                                               high=n_features,
                                               size=(n_features // 2,)).tolist()))

        mps = tk.models.UMPS(n_features=n_features,
                             phys_dim=5,
                             bond_dim=2,
                             in_features=in_features,
                             **runtime_kwargs)
        self._trace_umps_norm(mps, in_features, runtime)

        norms = [
            self._compute_umps_norm(mps, in_features, log_scale, runtime)
            for log_scale in AUTO_BOOL_CASES
        ]
        assert torch.isclose(norms[0].exp(), norms[1])

    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n_features', REDUCED_DENSITY_N_FEATURES_CASES)
    def test_reduced_density(self, runtime, n_features):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        runtime_kwargs = self._runtime_kwargs(runtime, device)
        phys_dim = torch.randint(low=2, high=6, size=(1,)).item()
        bond_dim = torch.randint(low=2, high=4, size=(1,)).item()
        trace_sites = torch.randint(low=0,
                                    high=n_features,
                                    size=(n_features // 2,)).tolist()

        mps = tk.models.UMPS(n_features=n_features,
                             phys_dim=phys_dim,
                             bond_dim=bond_dim,
                             in_features=trace_sites,
                             **runtime_kwargs)

        self._trace_umps_density(mps, phys_dim, len(set(trace_sites)), runtime)
        density = self._assert_umps_reduced_density(mps, phys_dim, trace_sites)
        self._backward_maybe_complex(density, runtime)
        for node in mps.mats_env:
            assert node.grad is not None

        density = mps.reduced_density(trace_sites)
    
    def test_canonicalize_error(self):
        mps = tk.models.UMPS(n_features=10,
                             phys_dim=2,
                             bond_dim=10,
                             in_features=[])
        
        with pytest.raises(NotImplementedError):
            mps.canonicalize()
    
    def test_canonicalize_univocal_error(self):
        mps = tk.models.UMPS(n_features=6,
                             phys_dim=2,
                             bond_dim=10,
                             in_features=[])
        
        with pytest.raises(NotImplementedError):
            mps.canonicalize_univocal()


class TestMPSLayer:  # MARK: TestMPSLayer
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_with_tensors(self, n, boundary):
        tensors = [torch.randn(10, 2, 10) for _ in range(n)]
        if boundary == 'obc':
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]

        mps = tk.models.MPSLayer(tensors=tensors)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.in_dim == [2] * (n - 1)
        assert mps.out_dim == 2
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
    
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
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('init_method', MPS_INIT_METHODS)
    def test_initialize_init_method(self, n, boundary, init_method):
        mps = tk.models.MPSLayer(boundary=boundary,
                                 n_features=n,
                                 in_dim=2,
                                 out_dim=10,
                                 bond_dim=5,
                                 init_method=init_method)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.in_dim == [2] * (n - 1)
        assert mps.out_dim == 10
        assert mps.bond_dim == [5] * (n - 1 if boundary == 'obc' else n)
        if boundary == 'obc':
            _assert_boundary_vector(mps.left_node)
            _assert_boundary_vector(mps.right_node)
    
    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('init_method', MPS_INIT_METHODS)
    def test_initialize_init_method_runtime(self, runtime, n, boundary,
                                            init_method):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_kwargs = _runtime_kwargs(runtime, device)
        mps = tk.models.MPSLayer(boundary=boundary,
                                 n_features=n,
                                 in_dim=2,
                                 out_dim=10,
                                 bond_dim=5,
                                 init_method=init_method,
                                 **model_kwargs)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.in_dim == [2] * (n - 1)
        assert mps.out_dim == 10
        assert mps.bond_dim == [5] * (n - 1 if boundary == 'obc' else n)
        _assert_nodes_runtime(mps.mats_env, runtime, device)
        if boundary == 'obc':
            _assert_boundary_vector(mps.left_node)
            _assert_boundary_vector(mps.right_node)
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_canonical(self, n, boundary):
        mps = tk.models.MPSLayer(boundary=boundary,
                                 n_features=n,
                                 in_dim=10,
                                 out_dim=10,
                                 bond_dim=10,
                                 init_method='canonical')
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.phys_dim == [10] * n
        assert mps.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
        if boundary == 'obc':
            assert mps.norm().isclose(torch.tensor(10. ** n).sqrt())
    
    @pytest.mark.parametrize('runtime', RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_canonical_runtime(self, runtime, n, boundary):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_kwargs = _runtime_kwargs(runtime, device)
        mps = tk.models.MPSLayer(boundary=boundary,
                                 n_features=n,
                                 in_dim=10,
                                 out_dim=10,
                                 bond_dim=10,
                                 init_method='canonical',
                                 **model_kwargs)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.phys_dim == [10] * n
        assert mps.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
        _assert_nodes_runtime(mps.mats_env, runtime, device)
        if boundary == 'obc':
            assert mps.norm().isclose(torch.tensor(10. ** n).sqrt())
    
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
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_in_out_dims(self, n_features, boundary):
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
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_in_dims_error(self, n_features, boundary):
        # in_dim should have (n_features - 1) elements.
        in_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        with pytest.raises(ValueError):
            tk.models.MPSLayer(n_features=n_features,
                               in_dim=in_dim,
                               out_dim=2,
                               bond_dim=10,
                               boundary=boundary)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy(self, n_features, boundary, share_tensors):
        in_dim = torch.randint(low=2, high=12, size=(n_features - 1,)).tolist()
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
        _assert_copied_mps(mps, copied_mps, share_tensors)
        assert mps.out_position == copied_mps.out_position
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('override', AUTO_BOOL_CASES)
    def test_deparameterize(self, n_features, boundary, override):
        in_dim = torch.randint(low=2, high=12, size=(n_features - 1,)).tolist()
        out_dim = torch.randint(low=2, high=12, size=(1,)).item()
        bond_dim = torch.randint(low=2, high=10, size=(n_features,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        mps = tk.models.MPSLayer(n_features=n_features,
                                 in_dim=in_dim,
                                 out_dim=out_dim,
                                 bond_dim=bond_dim,
                                 boundary=boundary)

        non_param_mps = mps.parameterize(set_param=False, override=override)

        if override:
            assert non_param_mps is mps
        else:
            assert non_param_mps is not mps

        new_nodes = non_param_mps.mats_env[:]
        if boundary == 'obc':
            new_nodes += [non_param_mps.left_node, non_param_mps.right_node]

        _assert_deparameterized_nodes(new_nodes)

    @pytest.mark.parametrize('n_features', INIT_N_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy_preserves_boundary_dtype(self, n_features, share_tensors):
        mps = tk.models.MPSLayer(n_features=n_features,
                                 in_dim=[] if n_features == 1 else [3] * (n_features - 1),
                                 out_dim=2,
                                 bond_dim=4,
                                 boundary='obc',
                                 dtype=torch.complex64)

        copied_mps = mps.copy(share_tensors=share_tensors)

        assert copied_mps.left_node.dtype == torch.complex64
        assert copied_mps.right_node.dtype == torch.complex64
        assert copied_mps.mats_env[0].dtype == torch.complex64

    @pytest.mark.parametrize('n_features', INIT_N_CASES)
    def test_deparameterize_preserves_boundary_runtime(self, n_features):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dtype = torch.complex64

        mps = tk.models.MPSLayer(n_features=n_features,
                                 in_dim=[] if n_features == 1 else [3] * (n_features - 1),
                                 out_dim=2,
                                 bond_dim=4,
                                 boundary='obc').to(device=device, dtype=dtype)

        non_param_mps = mps.parameterize(set_param=False, override=False)

        _assert_nodes_device_and_dtype(non_param_mps.mats_env, device, dtype)
        _assert_obc_boundary_runtime(non_param_mps, device, dtype)


class TestUMPSLayer:  # MARK: TestUMPSLayer
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    def test_initialize_with_tensors(self, n):
        tensors = [torch.randn(10, 2, 10), torch.randn(10, 2, 10)]
        mps = tk.models.UMPSLayer(n_features=n, tensors=tensors)
        assert mps.n_features == n
        assert mps.boundary == 'pbc'
        assert mps.in_dim == [2] * (n - 1)
        assert mps.out_dim == 2
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [10] * n
        assert mps.out_node.tensor is not mps.uniform_memory.tensor
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('init_method', MPS_INIT_METHODS)
    def test_initialize_init_method(self, n, init_method):
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
        assert mps.out_node.tensor is not mps.uniform_memory.tensor
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    def test_initialize_with_unitaries(self, n):
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
    
    @pytest.mark.parametrize('n_features', [2, 3, 4, 6])
    def test_in_out_dims(self, n_features):
        in_dim = torch.randint(low=2, high=10, size=(n_features - 1,)).tolist()

        # in_dim should be int.
        with pytest.raises(TypeError):
            tk.models.UMPSLayer(n_features=n_features,
                                in_dim=in_dim,
                                out_dim=2,
                                bond_dim=10)
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy(self, n_features, share_tensors):
        in_dim = torch.randint(low=2, high=12, size=(1,)).item()
        out_dim = torch.randint(low=2, high=12, size=(1,)).item()
        bond_dim = torch.randint(low=2, high=10, size=(1,)).item()

        mps = tk.models.UMPSLayer(n_features=n_features,
                                  in_dim=in_dim,
                                  out_dim=out_dim,
                                  bond_dim=bond_dim)

        copied_mps = mps.copy(share_tensors=share_tensors)

        assert isinstance(copied_mps, tk.models.UMPSLayer)
        _assert_copied_mps(mps, copied_mps, share_tensors)
        assert mps.out_position == copied_mps.out_position
    
    @pytest.mark.parametrize('n_features', MODEL_N_FEATURES_CASES)
    @pytest.mark.parametrize('override', AUTO_BOOL_CASES)
    def test_deparameterize(self, n_features, override):
        in_dim = torch.randint(low=2, high=12, size=(1,)).item()
        out_dim = torch.randint(low=2, high=12, size=(1,)).item()
        bond_dim = torch.randint(low=2, high=10, size=(1,)).item()

        mps = tk.models.UMPSLayer(n_features=n_features,
                                  in_dim=in_dim,
                                  out_dim=out_dim,
                                  bond_dim=bond_dim)

        non_param_mps = mps.parameterize(set_param=False, override=override)

        if override:
            assert non_param_mps is mps
        else:
            assert non_param_mps is not mps

        _assert_deparameterized_nodes(non_param_mps.mats_env,
                                      tensor_address='virtual_uniform')
    
    def test_canonicalize_error(self):
        mps = tk.models.UMPSLayer(n_features=10,
                                  in_dim=2,
                                  out_dim=2,
                                  bond_dim=10)
        
        with pytest.raises(NotImplementedError):
            mps.canonicalize()
    
    def test_canonicalize_univocal_error(self):
        mps = tk.models.UMPSLayer(n_features=6,
                                  in_dim=2,
                                  out_dim=2,
                                  bond_dim=10)
        
        with pytest.raises(NotImplementedError):
            mps.canonicalize_univocal()


class TestMPSData:   # MARK: TestMPSData
    
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_with_tensors(self, n, boundary):
        tensors = [torch.randn(20, 10, 2, 10) for _ in range(n)]
        if boundary == 'obc':
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]

        mps = tk.models.MPSData(tensors=tensors)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
    
    @pytest.mark.parametrize('runtime', DEVICE_RUNTIME_CASES)
    @pytest.mark.parametrize('n', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    def test_initialize_with_tensors_runtime(self, runtime, n, boundary):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tensor_kwargs = _runtime_kwargs(runtime, device)
        tensors = [torch.randn(20, 10, 2, 10, **tensor_kwargs) for _ in range(n)]
        if boundary == 'obc':
            tensors[0] = tensors[0][0]
            tensors[-1] = tensors[-1][..., 0]

        mps = tk.models.MPSData(tensors=tensors)
        assert mps.n_features == n
        assert mps.boundary == boundary
        assert mps.phys_dim == [2] * n
        assert mps.bond_dim == [10] * (n - 1 if boundary == 'obc' else n)
        _assert_nodes_runtime(mps.mats_env, runtime, device)
    
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
    
    @pytest.mark.parametrize('n_features', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('n_batches', [1, 2, 3])
    @pytest.mark.parametrize('init_method', MPSDATA_INIT_METHODS)
    def test_initialize_init_method(self, n_features, boundary, n_batches,
                                    init_method):
        mps = tk.models.MPSData(boundary=boundary,
                                n_features=n_features,
                                phys_dim=2,
                                bond_dim=5,
                                n_batches=n_batches,
                                init_method=init_method)

        assert mps.n_features == n_features
        assert mps.boundary == boundary
        assert mps.phys_dim == [2] * n_features
        assert mps.bond_dim == [5] * (n_features - 1 if boundary == 'obc'
                                      else n_features)
        _assert_mps_data_node_shapes(mps, n_batches, 1)
    
    @pytest.mark.parametrize('n_features', INIT_N_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('n_batches', [1, 2, 3])
    def test_initialize_add_data(self, n_features, boundary, n_batches):
        mps = tk.models.MPSData(boundary=boundary,
                                n_features=n_features,
                                phys_dim=2,
                                bond_dim=5,
                                n_batches=n_batches)

        tensors = [torch.randn(*([10] * n_batches), 5, 2, 5)
                   for _ in range(n_features)]
        if boundary == 'obc':
            tensors[0] = tensors[0][..., 0, :, :]
            tensors[-1] = tensors[-1][..., 0]

        mps.add_data(tensors)

        assert mps.n_features == n_features
        assert mps.boundary == boundary
        assert mps.phys_dim == [2] * n_features
        assert mps.bond_dim == [5] * (n_features - 1 if boundary == 'obc'
                                      else n_features)
        _assert_mps_data_node_shapes(mps, n_batches, 10)


class TestConvModels:  # MARK: TestConvModels
    
    @staticmethod
    def _expected_leaf_nodes(boundary, n_features, output_edge):
        base = n_features + 2 if boundary == 'obc' else n_features
        return base + output_edge

    @pytest.mark.parametrize('height', SMALL_SPATIAL_CASES)
    @pytest.mark.parametrize('width', SMALL_SPATIAL_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    def test_conv_mps_all_algorithms(self, height, width, boundary,
                                     auto_stack, auto_unbind,
                                     inline_input, inline_mats):
        example = torch.randn(1, 5, height, width)
        data = torch.randn(20, 5, height, width)

        mps = tk.models.ConvMPS(in_channels=5,
                                bond_dim=2,
                                kernel_size=(height, width),
                                boundary=boundary)
        mps.auto_stack = auto_stack
        mps.auto_unbind = auto_unbind

        mps.trace(example, inline_input=inline_input, inline_mats=inline_mats)
        result = mps(data, inline_input=inline_input, inline_mats=inline_mats)

        assert result.shape == (20, 1, 1)
        assert len(mps.edges) == 0
        assert len(mps.leaf_nodes) == self._expected_leaf_nodes(
            boundary, height * width, 0
        )
        assert len(mps.data_nodes) == height * width
        assert len(mps.virtual_nodes) == (2 if not inline_input and auto_stack else 1)

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None

    @pytest.mark.parametrize('height', SMALL_SPATIAL_CASES)
    @pytest.mark.parametrize('width', SMALL_SPATIAL_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    def test_conv_umps_all_algorithms(self, height, width, auto_stack,
                                      auto_unbind, inline_input, inline_mats):
        example = torch.randn(1, 5, height, width)
        data = torch.randn(20, 5, height, width)

        mps = tk.models.ConvUMPS(in_channels=5,
                                 bond_dim=2,
                                 kernel_size=(height, width))
        mps.auto_stack = auto_stack
        mps.auto_unbind = auto_unbind

        mps.trace(example, inline_input=inline_input, inline_mats=inline_mats)
        result = mps(data, inline_input=inline_input, inline_mats=inline_mats)

        assert result.shape == (20, 1, 1)
        assert len(mps.edges) == 0
        assert len(mps.leaf_nodes) == height * width
        assert len(mps.data_nodes) == height * width
        assert len(mps.virtual_nodes) == 2

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None
        assert mps.uniform_memory.grad is not None

    @pytest.mark.parametrize('height', SMALL_SPATIAL_CASES)
    @pytest.mark.parametrize('width', SMALL_SPATIAL_CASES)
    @pytest.mark.parametrize('boundary', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    def test_conv_mps_layer_all_algorithms(self, height, width, boundary,
                                           auto_stack, auto_unbind,
                                           inline_input, inline_mats):
        example = torch.randn(1, 5, height, width)
        data = torch.randn(20, 5, height, width)

        mps = tk.models.ConvMPSLayer(in_channels=5,
                                     out_channels=10,
                                     bond_dim=2,
                                     kernel_size=(height, width),
                                     boundary=boundary)
        mps.auto_stack = auto_stack
        mps.auto_unbind = auto_unbind

        mps.trace(example, inline_input=inline_input, inline_mats=inline_mats)
        result = mps(data, inline_input=inline_input, inline_mats=inline_mats)

        assert result.shape == (20, 10, 1, 1)
        assert len(mps.edges) == 1
        assert len(mps.leaf_nodes) == self._expected_leaf_nodes(
            boundary, height * width, 1
        )
        assert len(mps.data_nodes) == height * width
        assert len(mps.virtual_nodes) == (2 if not inline_input and auto_stack else 1)

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None

    @pytest.mark.parametrize('height', SMALL_SPATIAL_CASES)
    @pytest.mark.parametrize('width', SMALL_SPATIAL_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline_mats', AUTO_BOOL_CASES)
    def test_conv_umps_layer_all_algorithms(self, height, width, auto_stack,
                                            auto_unbind, inline_input,
                                            inline_mats):
        example = torch.randn(1, 5, height, width)
        data = torch.randn(20, 5, height, width)

        mps = tk.models.ConvUMPSLayer(in_channels=5,
                                      out_channels=10,
                                      bond_dim=2,
                                      kernel_size=(height, width))
        mps.auto_stack = auto_stack
        mps.auto_unbind = auto_unbind

        mps.trace(example, inline_input=inline_input, inline_mats=inline_mats)
        result = mps(data, inline_input=inline_input, inline_mats=inline_mats)

        assert result.shape == (20, 10, 1, 1)
        assert len(mps.edges) == 1
        assert len(mps.leaf_nodes) == height * width + 1
        assert len(mps.data_nodes) == height * width
        assert len(mps.virtual_nodes) == 2

        result.sum().backward()
        for node in mps.mats_env:
            assert node.grad is not None
        assert mps.uniform_memory.grad is not None
    
    @pytest.mark.parametrize('height,width,boundary,share_tensors',
                             COPY_CONV_MPS_CASES)
    def test_copy_conv_mps(self, height, width, boundary, share_tensors):
        phys_dim = torch.randint(low=2, high=8, size=(height * width,)).tolist()
        bond_dim = torch.randint(low=2, high=6, size=(height * width,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        mps = tk.models.ConvMPS(in_channels=phys_dim,
                                bond_dim=bond_dim,
                                kernel_size=(height, width),
                                boundary=boundary)

        copied_mps = mps.copy(share_tensors=share_tensors)
        assert isinstance(copied_mps, tk.models.ConvMPS)
        _assert_copied_mps(mps, copied_mps, share_tensors)

    @pytest.mark.parametrize('height,width', [(1, 1), (2, 2)])
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy_conv_mps_preserves_boundary_dtype(self,
                                                    height,
                                                    width,
                                                    share_tensors):
        mps = tk.models.ConvMPS(in_channels=3,
                                bond_dim=4,
                                kernel_size=(height, width),
                                boundary='obc',
                                dtype=torch.complex64)

        copied_mps = mps.copy(share_tensors=share_tensors)

        assert copied_mps.left_node.dtype == torch.complex64
        assert copied_mps.right_node.dtype == torch.complex64
        assert copied_mps.mats_env[0].dtype == torch.complex64

    @pytest.mark.parametrize('height,width,share_tensors', COPY_CONV_UMPS_CASES)
    def test_copy_conv_umps(self, height, width, share_tensors):
        phys_dim = torch.randint(low=2, high=8, size=(1,)).item()
        bond_dim = torch.randint(low=2, high=6, size=(1,)).item()

        mps = tk.models.ConvUMPS(in_channels=phys_dim,
                                 bond_dim=bond_dim,
                                 kernel_size=(height, width))

        copied_mps = mps.copy(share_tensors=share_tensors)
        assert isinstance(copied_mps, tk.models.ConvUMPS)
        _assert_copied_mps(mps, copied_mps, share_tensors)

    @pytest.mark.parametrize('height,width,boundary,share_tensors',
                             COPY_CONV_MPS_CASES)
    def test_copy_conv_mps_layer(self, height, width, boundary, share_tensors):
        phys_dim = torch.randint(low=2, high=8, size=(height * width,)).tolist()
        bond_dim = torch.randint(low=2, high=6,
                                 size=(height * width + 1,)).tolist()
        bond_dim = bond_dim[:-1] if boundary == 'obc' else bond_dim

        mps = tk.models.ConvMPSLayer(in_channels=phys_dim,
                                     out_channels=10,
                                     bond_dim=bond_dim,
                                     kernel_size=(height, width),
                                     boundary=boundary)

        copied_mps = mps.copy(share_tensors=share_tensors)
        assert isinstance(copied_mps, tk.models.ConvMPSLayer)
        _assert_copied_mps(mps, copied_mps, share_tensors)

    @pytest.mark.parametrize('height,width', [(1, 1), (2, 2)])
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy_conv_mps_layer_preserves_boundary_dtype(self,
                                                          height,
                                                          width,
                                                          share_tensors):
        mps = tk.models.ConvMPSLayer(in_channels=3,
                                     out_channels=10,
                                     bond_dim=4,
                                     kernel_size=(height, width),
                                     boundary='obc',
                                     dtype=torch.complex64)

        copied_mps = mps.copy(share_tensors=share_tensors)

        assert copied_mps.left_node.dtype == torch.complex64
        assert copied_mps.right_node.dtype == torch.complex64
        assert copied_mps.mats_env[0].dtype == torch.complex64

    @pytest.mark.parametrize('height,width,share_tensors', COPY_CONV_UMPS_CASES)
    def test_copy_conv_umps_layer(self, height, width, share_tensors):
        phys_dim = torch.randint(low=2, high=8, size=(1,)).item()
        bond_dim = torch.randint(low=2, high=6, size=(1,)).item()

        mps = tk.models.ConvUMPSLayer(in_channels=phys_dim,
                                      out_channels=10,
                                      bond_dim=bond_dim,
                                      kernel_size=(height, width))

        copied_mps = mps.copy(share_tensors=share_tensors)
        assert isinstance(copied_mps, tk.models.ConvUMPSLayer)
        _assert_copied_mps(mps, copied_mps, share_tensors)
        


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
