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


def _contract_mps(mps):
    result = mps.left_node
    for node in mps.mats_env + [mps.right_node]:
        result @= node
    return result.tensor


def _contract_mpo(mpo):
    result = mpo.left_node
    for node in mpo.mats_env + [mpo.right_node]:
        result @= node
    return result.tensor


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

        if n_batches == 0:
            mps = tk.models.MPS(tensors=tensors)
        else:
            mps = tk.models.MPSData(tensors=tensors, n_batches=n_batches)

        approx_vec = _contract_mps(mps)
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
    @pytest.mark.parametrize(
        'dims',
        [
            (2, 3),
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6, 7),
        ],
        ids=['one-site', 'two-sites', 'three-sites'],
    )
    def test_mat_to_mpo(self, renormalize, dims):
        mat = _make_tensor(dims, torch.float32, scale=1e-5)
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
    @pytest.mark.parametrize(
        'dims',
        [
            (2, 3, 4, 5),
            (2, 3, 4, 5, 2, 3),
        ],
        ids=['two-sites', 'three-sites'],
    )
    def test_mat_to_mpo_accuracy(self, renormalize, dims):
        mat = _make_tensor(dims, torch.float32, scale=1e-1)
        tensors = tk.decompositions.mat_to_mpo(mat=mat,
                                               cum_percentage=0.9999,
                                               renormalize=renormalize)

        mpo = tk.models.MPO(tensors=tensors)
        approx_mat = _contract_mpo(mpo)
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

        approx_mat = _contract_mpo(mpo)
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


class TestSVDKernelCallers:  # MARK: TestSVDKernelCallers

    @pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
    def test_vec_to_mps_and_mat_to_mpo_backend(self, svd_method):
        generator = torch.Generator().manual_seed(0)
        vec = torch.randn(2, 3, 4, dtype=torch.float64, generator=generator)
        with tk.svd_method(svd_method):
            tensors = tk.decompositions.vec_to_mps(vec=vec)
        mps = tk.models.MPS(tensors=tensors)
        assert torch.allclose(
            _contract_mps(mps), vec, rtol=1e-10, atol=1e-12)

        mat = torch.randn(
            2, 3, 4, 5, dtype=torch.float64, generator=generator)
        with tk.svd_method(svd_method):
            tensors = tk.decompositions.mat_to_mpo(mat=mat)
        mpo = tk.models.MPO(tensors=tensors)
        assert torch.allclose(
            _contract_mpo(mpo), mat, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
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

    @pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
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
