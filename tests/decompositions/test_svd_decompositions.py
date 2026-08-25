"""
Tests for svd decompositions:

    * TestSVDDecompositions
    * TestSVDKernelCallers
"""

import pytest

import torch
import tensorkrowch as tk


TRUNCATION_CASES = [
    ({}, 4),
    ({'rank': 2}, 2),
    ({'cutoff': 1.0}, 2),
    ({'atol': 1.05}, 2),
    ({'rtol': 0.03}, 2),
    ({'cum_percentage': 0.97}, 2),
    ({'rank': 3, 'cutoff': 1.0, 'atol': 1.05}, 2),
    ({'rank': 4, 'rtol': 0.03, 'cum_percentage': 0.97}, 2),
]
DEVICE_CASES = ['cpu', 'cuda', 'mps']
SVD_METHOD_CASES = ['svd', 'qr_svd']


def _make_tensor(shape, dtype, scale):
    numel = 1
    for dim in shape:
        numel *= dim

    base = torch.arange(1, numel + 1, dtype=torch.float32).reshape(shape)
    base = scale * base / numel

    if dtype in (torch.complex64, torch.complex128):
        imag = scale * base.flip(-1) / (2 * numel)
        return base.to(dtype) + 1j * imag.to(dtype)
    return base.to(dtype)


def _device(device_name):
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA is not available')
    elif device_name == 'mps':
        if not getattr(torch.backends, 'mps', None) or \
                not torch.backends.mps.is_available():
            pytest.skip('MPS is not available')
    return torch.device(device_name)


def _contract_tt_cores(tensors, n_batches=0):
    """Contracts TT cores using only batched PyTorch matrix products."""
    result = tensors[0]
    if len(tensors) == 1:
        return result

    for i, tensor in enumerate(tensors[1:], 1):
        batch_shape = result.shape[:n_batches]
        prev_phys_dims = result.shape[n_batches:-1]
        prev_rank = result.shape[-1]
        result = result.reshape(*batch_shape, -1, prev_rank)

        if i < (len(tensors) - 1):
            phys_dims = tensor.shape[(n_batches + 1):-1]
            rank = tensor.shape[-1]
            tensor = tensor.reshape(*batch_shape, prev_rank, -1)
            result = (result @ tensor).reshape(
                *batch_shape, *prev_phys_dims, *phys_dims, rank)
        else:
            phys_dims = tensor.shape[(n_batches + 1):]
            tensor = tensor.reshape(*batch_shape, prev_rank, -1)
            result = (result @ tensor).reshape(
                *batch_shape, *prev_phys_dims, *phys_dims)

    return result


def _contract_ttm_cores(tensors):
    """Contracts TTM cores into an interleaved input/output tensor."""
    if len(tensors) == 1:
        return tensors[0]

    # Move each right rank behind the physical input/output dimensions.
    result = tensors[0].permute(0, 2, 1)
    for tensor in tensors[1:-1]:
        tensor = tensor.permute(0, 1, 3, 2)
        result = torch.tensordot(result, tensor, dims=([-1], [0]))
    return torch.tensordot(result, tensors[-1], dims=([-1], [0]))


class TestSVDDecompositions:  # MARK: TestSVDDecompositions

    @pytest.mark.parametrize('renormalize', [True, False],
                             ids=['renorm', 'no-renorm'])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64],
                             ids=['float32', 'complex64'])
    @pytest.mark.parametrize(
        'dims, n_batches',
        [
            ((3,), 0),
            ((2, 3, 4), 0),
            ((2, 3, 4), 1),
            ((2, 3, 4, 5), 2),
        ],
        ids=['one-site', 'mps', 'mpsdata-1-batch', 'mpsdata-2-batches'],
    )
    def test_vec_to_mps(self, renormalize, dtype, dims, n_batches):
        vec = _make_tensor(dims, dtype, scale=1e-5)
        tensors = tk.decompositions.vec_to_mps(vec=vec,
                                               n_batches=n_batches,
                                               rank=5,
                                               renormalize=renormalize)

        phys_dims = list(dims[n_batches:])
        expected_num_tensors = max(1, len(phys_dims))
        assert len(tensors) == expected_num_tensors

        for tensor in tensors:
            assert tensor.shape[:n_batches] == dims[:n_batches]
            assert tensor.dtype == dtype

        bond_dims = [tensor.shape[-1] for tensor in tensors[:-1]]
        for bond_dim in bond_dims:
            assert bond_dim <= 5

        if n_batches == 0:
            mps = tk.models.MPS(tensors=tensors)
            assert mps.phys_dim == phys_dims
            assert mps.bond_dim == bond_dims
        else:
            mps = tk.models.MPSData(tensors=tensors, n_batches=n_batches)
            assert mps.phys_dim == phys_dims
            assert mps.bond_dim == bond_dims

    @pytest.mark.parametrize('renormalize', [True, False],
                             ids=['renorm', 'no-renorm'])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64],
                             ids=['float32', 'complex64'])
    @pytest.mark.parametrize(
        'dims, n_batches',
        [
            ((2, 3, 4), 0),
            ((2, 3, 4), 1),
            ((2, 3, 4, 2), 2),
        ],
        ids=['mps', 'mpsdata-1-batch', 'mpsdata-2-batches'],
    )
    def test_vec_to_mps_accuracy(self, renormalize, dtype, dims, n_batches):
        vec = _make_tensor(dims, dtype, scale=1e-1)
        tensors = tk.decompositions.vec_to_mps(vec=vec,
                                               n_batches=n_batches,
                                               cum_percentage=0.9999,
                                               renormalize=renormalize)

        approx_vec = _contract_tt_cores(tensors, n_batches=n_batches)
        diff = vec - approx_vec
        assert diff.norm() < 1e-1

    @pytest.mark.parametrize('n_batches', [0, 1],
                             ids=['no-batches', 'one-batch'])
    @pytest.mark.parametrize(
        'kwargs, expected_rank',
        TRUNCATION_CASES,
        ids=[
            'full-rank',
            'rank',
            'cutoff',
            'atol',
            'rtol',
            'cum_percentage',
            'rank-cutoff-atol',
            'rank-rtol-cum_percentage',
        ],
    )
    def test_vec_to_mps_truncation_criteria(self,
                                            n_batches,
                                            kwargs,
                                            expected_rank):
        diag = torch.diag(torch.tensor([5.0, 3.0, 1.0, 0.1]))
        vec = diag if n_batches == 0 else torch.stack([diag, diag])

        tensors = tk.decompositions.vec_to_mps(vec=vec,
                                               n_batches=n_batches,
                                               **kwargs)

        assert len(tensors) == 2
        assert tensors[0].shape[-1] == expected_rank
        assert tensors[1].shape[-2] == expected_rank

        if n_batches == 0:
            mps = tk.models.MPS(tensors=tensors)
        else:
            mps = tk.models.MPSData(tensors=tensors, n_batches=n_batches)

        assert mps.phys_dim == [4, 4]
        assert mps.bond_dim == [expected_rank]

    @pytest.mark.parametrize('renormalize', [True, False],
                             ids=['renorm', 'no-renorm'])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64],
                             ids=['float32', 'complex64'])
    @pytest.mark.parametrize(
        'dims',
        [
            (2, 3),
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6, 7),
        ],
        ids=['one-site', 'two-sites', 'three-sites'],
    )
    def test_mat_to_mpo(self, renormalize, dtype, dims):
        mat = _make_tensor(dims, dtype, scale=1e-5)
        tensors = tk.decompositions.mat_to_mpo(mat=mat,
                                               rank=5,
                                               renormalize=renormalize)

        expected_num_tensors = max(1, len(dims) // 2)
        assert len(tensors) == expected_num_tensors

        bond_dims = [tensor.shape[-2] for tensor in tensors[:-1]]
        for bond_dim in bond_dims:
            assert bond_dim <= 5

        mpo = tk.models.MPO(tensors=tensors)
        assert mpo.in_dim == list(dims[::2])
        assert mpo.out_dim == list(dims[1::2])
        assert mpo.bond_dim == bond_dims

    @pytest.mark.parametrize('renormalize', [True, False],
                             ids=['renorm', 'no-renorm'])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64],
                             ids=['float32', 'complex64'])
    @pytest.mark.parametrize(
        'dims',
        [
            (2, 3, 4, 5),
            (2, 3, 4, 5, 2, 3),
        ],
        ids=['two-sites', 'three-sites'],
    )
    def test_mat_to_mpo_accuracy(self, renormalize, dtype, dims):
        mat = _make_tensor(dims, dtype, scale=1e-1)
        tensors = tk.decompositions.mat_to_mpo(mat=mat,
                                               cum_percentage=0.9999,
                                               renormalize=renormalize)

        approx_mat = _contract_ttm_cores(tensors)
        diff = mat - approx_mat
        assert diff.norm() < 1e-1

    @pytest.mark.parametrize('renormalize', [True, False],
                             ids=['renorm', 'no-renorm'])
    @pytest.mark.parametrize(
        'in_dims, out_dims',
        [
            ((2, 2, 2, 2), (3, 3, 3, 3)),
            ((2, 3, 4, 2), (3, 5, 7, 2)),
        ],
        ids=['same-dims', 'different-dims'],
    )
    def test_mat_to_mpo_permuted_accuracy(self, renormalize, in_dims, out_dims):
        dims = list(in_dims) + list(out_dims)
        mat = _make_tensor(dims, torch.float32, scale=1e-1)

        permute_order = []
        for idx in range(len(in_dims)):
            permute_order.extend([idx, len(in_dims) + idx])
        aux_mat = mat.permute(*permute_order)

        tensors = tk.decompositions.mat_to_mpo(mat=aux_mat,
                                               cum_percentage=0.9999,
                                               renormalize=renormalize)

        mpo = tk.models.MPO(tensors=tensors)
        assert mpo.in_dim == list(in_dims)
        assert mpo.out_dim == list(out_dims)

        approx_mat = _contract_ttm_cores(tensors)
        inverse_permute = tuple(range(0, 2 * len(in_dims), 2)) + \
            tuple(range(1, 2 * len(in_dims), 2))
        approx_mat = approx_mat.permute(*inverse_permute)

        diff = mat - approx_mat
        assert diff.norm() < 1e-1

    @pytest.mark.parametrize(
        'kwargs, expected_rank',
        TRUNCATION_CASES,
        ids=[
            'full-rank',
            'rank',
            'cutoff',
            'atol',
            'rtol',
            'cum_percentage',
            'rank-cutoff-atol',
            'rank-rtol-cum_percentage',
        ],
    )
    def test_mat_to_mpo_truncation_criteria(self, kwargs, expected_rank):
        singular_values = torch.tensor([5.0, 3.0, 1.0, 0.1])
        mat = torch.zeros(2, 2, 2, 2, 2, 2)
        for idx, value in enumerate(singular_values):
            pair_idx = ((idx // 2), (idx % 2))
            mat[pair_idx[0], pair_idx[1],
                pair_idx[0], pair_idx[1],
                pair_idx[0], pair_idx[1]] = value

        tensors = tk.decompositions.mat_to_mpo(mat=mat, **kwargs)

        assert len(tensors) == 3
        assert tensors[0].shape == (2, expected_rank, 2)
        assert tensors[1].shape[0] == expected_rank
        assert tensors[1].shape[1] == 2
        assert tensors[1].shape[2] <= expected_rank
        assert tensors[1].shape[3] == 2
        assert tensors[2].shape[0] == tensors[1].shape[2]
        assert tensors[2].shape[1:] == (2, 2)

        mpo = tk.models.MPO(tensors=tensors)
        assert mpo.in_dim == [2, 2, 2]
        assert mpo.out_dim == [2, 2, 2]
        assert mpo.bond_dim[0] == expected_rank
        assert mpo.bond_dim[1] <= expected_rank

    @pytest.mark.parametrize('svd_method', SVD_METHOD_CASES)
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_one_site_dense_oracles(self, svd_method, dtype):
        vec = _make_tensor((5,), dtype, scale=1e-2)
        mat = _make_tensor((3, 4), dtype, scale=1e-2)

        with tk.svd_method(svd_method):
            tt_cores = tk.decompositions.vec_to_mps(vec)
            ttm_cores = tk.decompositions.mat_to_mpo(mat)

        assert len(tt_cores) == 1
        assert len(ttm_cores) == 1
        assert torch.equal(_contract_tt_cores(tt_cores), vec)
        assert torch.equal(_contract_ttm_cores(ttm_cores), mat)

    @pytest.mark.parametrize('svd_method', SVD_METHOD_CASES)
    @pytest.mark.parametrize('renormalize', [True, False])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('n_batches', [0, 1])
    def test_vec_to_mps_exact_dense_oracle(self,
                                           svd_method,
                                           renormalize,
                                           dtype,
                                           n_batches):
        shape = (2, 3, 4) if n_batches == 0 else (2, 2, 3, 4)
        vec = _make_tensor(shape, dtype, scale=1e-2)

        with tk.svd_method(svd_method):
            tensors = tk.decompositions.vec_to_mps(
                vec=vec,
                n_batches=n_batches,
                renormalize=renormalize)

        result = _contract_tt_cores(tensors, n_batches=n_batches)
        assert torch.allclose(result, vec, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize('svd_method', SVD_METHOD_CASES)
    @pytest.mark.parametrize('renormalize', [True, False])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_mat_to_mpo_exact_dense_oracle(self,
                                           svd_method,
                                           renormalize,
                                           dtype):
        mat = _make_tensor((2, 3, 4, 2, 3, 2), dtype, scale=1e-2)

        with tk.svd_method(svd_method):
            tensors = tk.decompositions.mat_to_mpo(
                mat=mat,
                renormalize=renormalize)

        result = _contract_ttm_cores(tensors)
        assert torch.allclose(result, mat, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize('svd_method', SVD_METHOD_CASES)
    @pytest.mark.parametrize('device_name', DEVICE_CASES)
    def test_decompositions_preserve_device(self, svd_method, device_name):
        device = _device(device_name)
        vec = torch.randn(2, 3, 4, device=device)
        mat = torch.randn(2, 3, 4, 5, device=device)

        with tk.svd_method(svd_method):
            tt_cores = tk.decompositions.vec_to_mps(vec)
            ttm_cores = tk.decompositions.mat_to_mpo(mat)

        assert all(tensor.device == device for tensor in tt_cores)
        assert all(tensor.device == device for tensor in ttm_cores)
        assert torch.allclose(_contract_tt_cores(tt_cores), vec,
                              rtol=1e-5, atol=1e-6)
        assert torch.allclose(_contract_ttm_cores(ttm_cores), mat,
                              rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize('svd_method', SVD_METHOD_CASES)
    @pytest.mark.parametrize('decomposition', ['tt', 'ttm'])
    def test_decompositions_gradcheck(self, svd_method, decomposition):
        generator = torch.Generator().manual_seed(0)
        if decomposition == 'tt':
            tensor = torch.randn(2, 3, 4, dtype=torch.float64,
                                 generator=generator,
                                 requires_grad=True)

            def reconstruct(value):
                tensors = tk.decompositions.vec_to_mps(value)
                return _contract_tt_cores(tensors)
        else:
            tensor = torch.randn(2, 3, 2, 3, dtype=torch.float64,
                                 generator=generator,
                                 requires_grad=True)

            def reconstruct(value):
                tensors = tk.decompositions.mat_to_mpo(value)
                return _contract_ttm_cores(tensors)

        with tk.svd_method(svd_method):
            assert torch.autograd.gradcheck(
                reconstruct,
                (tensor,),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3)

    @pytest.mark.parametrize(
        'decomposition, kwargs, error_type, match',
        [
            ('tt', {'vec': [1, 2]}, TypeError,
             '`vec` should be torch.Tensor type'),
            ('tt', {'vec': torch.ones(2, 3), 'n_batches': 3}, ValueError,
             '`n_batches` should be between 0 and the rank of `vec`'),
            ('ttm', {'mat': [1, 2]}, TypeError,
             '`mat` should be torch.Tensor type'),
            ('ttm', {'mat': torch.ones(2, 3, 4)}, ValueError,
             '`mat` have an even number of dimensions'),
        ],
    )
    def test_public_argument_errors(self,
                                    decomposition,
                                    kwargs,
                                    error_type,
                                    match):
        function = (tk.decompositions.vec_to_mps
                    if decomposition == 'tt'
                    else tk.decompositions.mat_to_mpo)
        with pytest.raises(error_type, match=match):
            function(**kwargs)


class TestSVDKernelCallers:  # MARK: TestSVDKernelCallers

    @pytest.mark.parametrize('svd_method', SVD_METHOD_CASES)
    def test_vec_to_mps_and_mat_to_mpo_backend(self, svd_method):
        generator = torch.Generator().manual_seed(0)
        vec = torch.randn(2, 3, 4, dtype=torch.float64, generator=generator)
        with tk.svd_method(svd_method):
            tensors = tk.decompositions.vec_to_mps(vec=vec)
        assert torch.allclose(
            _contract_tt_cores(tensors), vec, rtol=1e-10, atol=1e-12)

        mat = torch.randn(
            2, 3, 4, 5, dtype=torch.float64, generator=generator)
        with tk.svd_method(svd_method):
            tensors = tk.decompositions.mat_to_mpo(mat=mat)
        assert torch.allclose(
            _contract_ttm_cores(tensors), mat, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize('svd_method', SVD_METHOD_CASES)
    @pytest.mark.parametrize('operation', ['split', 'svd', 'svdr'])
    def test_node_operations_backend(self, operation, svd_method):
        generator = torch.Generator().manual_seed(1)
        if operation == 'split':
            tensor = torch.randn(
                5, 7, dtype=torch.float64, generator=generator)
            node = tk.Node(
                tensor=tensor,
                axes_names=('left', 'right'))
            with tk.svd_method(svd_method):
                node1, node2 = node.split(
                    node1_axes=['left'],
                    node2_axes=['right'])
        else:
            tensor1 = torch.randn(
                5, 3, dtype=torch.float64, generator=generator)
            tensor2 = torch.randn(
                3, 7, dtype=torch.float64, generator=generator)
            tensor = tensor1 @ tensor2
            node1 = tk.Node(
                tensor=tensor1,
                axes_names=('left', 'bond'))
            node2 = tk.Node(
                tensor=tensor2,
                axes_names=('bond', 'right'),
                network=node1.network)
            edge = node1['bond'] ^ node2['bond']
            if operation == 'svd':
                with tk.svd_method(svd_method):
                    node1, node2 = tk.svd(edge)
            else:
                with torch.random.fork_rng():
                    torch.manual_seed(2)
                    with tk.svd_method(svd_method):
                        node1, node2 = tk.svdr(edge)

        assert torch.allclose(
            node1.tensor @ node2.tensor,
            tensor,
            rtol=1e-10,
            atol=1e-12)

    @pytest.mark.parametrize('svd_method', SVD_METHOD_CASES)
    def test_tt_rss_trimming_backend(self, svd_method):
        domain = torch.tensor([0.0, 1.0])
        sketch_samples = torch.cartesian_prod(domain, domain, domain)

        def function(data):
            return data.prod(dim=1, keepdim=True)

        def embedding(data):
            return torch.stack([data, 1 - data], dim=-1)

        with torch.random.fork_rng():
            torch.manual_seed(0)
            with tk.svd_method(svd_method):
                cores, info = tk.decompositions.tt_rss(
                    function=function,
                    embedding=embedding,
                    sketch_samples=sketch_samples,
                    domain=domain,
                    rank=2,
                    batch_size=8,
                    verbose=False,
                    return_info=True)

        assert [tuple(core.shape) for core in cores] == [
            (2, 2), (2, 2, 2), (2, 2)]
        assert all(torch.isfinite(core).all() for core in cores)
        assert info['val_eps'] < 1e-6
