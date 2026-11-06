"""Per-link SVD caps, source adaptation and rank-first interfaces."""

import pytest
import torch

import tensorkrowch as tk


d = tk.decompositions


@pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
@pytest.mark.parametrize('rank', [[], [2], [2, 3, 1, 2]])
@pytest.mark.parametrize('n_batches', [0, 1])
@pytest.mark.parametrize('renormalize', [False, True])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_tt_recovers_nonuniform_profiles(
        svd_method, rank, n_batches, renormalize, dtype):
    generator = torch.Generator().manual_seed(503)
    batch_shape = (2,) if n_batches else ()
    cores = []
    for site in range(len(rank) + 1):
        left = 1 if site == 0 else rank[site - 1]
        right = 1 if site == len(rank) else rank[site]
        core = torch.randn(*batch_shape, left, 4, right,
                           dtype=dtype, generator=generator)
        if not rank:
            core = core[..., 0, :, 0]
        elif site == 0:
            core = core.squeeze(-3)
        elif site == len(rank):
            core = core.squeeze(-1)
        cores.append(core)
    tensor = tk.formats.TT(cores, n_batches=n_batches).contract_dense()
    source = d.DenseTensorSource(tensor, n_batches=n_batches)
    with tk.svd_method(svd_method):
        result = d.TTSVD(source).fit(
            rank, renormalize=renormalize, collect_metrics=True)
    assert result.rank == rank
    assert result.batch_shape == batch_shape
    assert len(result.metrics.truncations) == len(rank)
    torch.testing.assert_close(result.contract_dense(), tensor,
                               atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
@pytest.mark.parametrize('renormalize', [False, True])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('dimensions,rank,center,padding', [
    ((7, 9), [4, 2], 1, 1),
    ((7, 9), [2, 4], 1, 1),
    ((2, 3, 4, 5), [4, 3, 10, 2], 2, 0),
    ((2, 3, 4, 5), [6, 2, 8, 3], 2, 0),
    ((2,) * 6, [4, 8, 4, 8, 4, 2], 3, 0),
])
def test_tr_asymmetric_caps_and_exact_reconstruction(
        svd_method, renormalize, dtype, dimensions, rank, center, padding):
    tensor = torch.randn(dimensions, dtype=dtype,
                         generator=torch.Generator().manual_seed(504))
    with tk.svd_method(svd_method):
        result = d.TRSVD(tensor).fit(
            rank, center=center, renormalize=renormalize,
            collect_metrics=True)
    assert result.rank == rank
    assert result.metadata['rank_mode'] == 'per_link'
    assert result.metadata['initial_padding'] == padding
    records = result.metrics.truncations
    initial = next(record for record in records
                   if record.phase == 'initial_bipartition')
    assert initial.selected_rank <= rank[-1] * rank[center - 1]
    for record in records:
        if record.phase != 'initial_bipartition':
            assert record.selected_rank <= rank[record.site]
    torch.testing.assert_close(result.contract_dense(), tensor,
                               atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('center', [1, 2, 3])
def test_tr_caps_follow_original_bonds_when_center_changes(center):
    tensor = torch.randn(2, 3, 4, 5, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(505))
    caps = [4, 6, 10, 2]
    result = d.TRSVD(tensor, center=1).fit(caps, center=center)
    assert result.metadata['center'] == center
    assert all(rank <= cap for rank, cap in zip(result.rank, caps))
    torch.testing.assert_close(result.contract_dense(), tensor,
                               atol=1e-10, rtol=1e-10)


def test_rank_profiles_remain_caps_without_padding_or_forced_retention():
    tensor = torch.randn(2, 3, 4, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(506))
    result = d.tt_svd(tensor, [20, 20])
    assert result.rank == [2, 4]
    torch.testing.assert_close(result.contract_dense(), tensor)

    tensor = torch.zeros(7, 9, dtype=torch.float64)
    tensor[range(4), range(4)] = tensor.new_tensor([4., 2., 1., 1e-8])
    result = d.tr_svd(tensor, [4, 2], rtol=1e-12)
    assert result.rank == [3, 1]
    assert (result.contract_dense() - tensor).norm() < 1.1e-8


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_matrix_profiles_count_paired_sites(cyclic, ordering):
    rank = [8, 8, 8, 2] if cyclic else [4, 16, 4]
    tensor = torch.randn((2,) * 8, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(507))
    stored = tensor.permute(0, 2, 4, 6, 1, 3, 5, 7) \
        if ordering == 'grouped' else tensor
    source = d.DenseTensorSource.matrix(stored, ordering=ordering)
    algorithm = d.TRMSVD if cyclic else d.TTMSVD
    function = d.trm_svd if cyclic else d.ttm_svd
    result = algorithm(source).fit(rank)
    direct = function(source, rank=rank)
    assert result.rank == rank
    assert result.n_sites == 4
    assert result.in_dim == result.out_dim == (2,) * 4
    torch.testing.assert_close(result.contract_dense(), tensor,
                               atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(direct.contract_dense(), tensor,
                               atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_quantics_profiles_count_digit_sites(cyclic, ordering):
    values = torch.randn(4, 4, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(508))
    layout = tk.formats.QuantizedLayout(2, 2, 2, ordering=ordering)
    coordinates = tk.formats.AffineCoordinateMap((0., 1.), layout.grid_size)

    def function(points):
        indices = coordinates.to_indices(points)
        return values[tuple(indices.T)]

    source = d.CallableQuanticsSource(
        function, 2, layout=layout, coordinate_map=coordinates,
        out_shape=(), dtype=values.dtype)
    algorithm = d.qtr_svd if cyclic else d.qtt_svd
    rank = [4, 2, 4, 2] if cyclic else [2, 4, 2]
    result = algorithm(source, rank=rank)
    assert result.rank == rank
    assert result.layout == layout
    torch.testing.assert_close(result.contract_dense_grid_axes(), values,
                               atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('matrix', [False, True])
def test_svd_adapts_raw_models_like_format_sources(cyclic, matrix):
    model = tk.models.MPO(n_features=3, in_dim=2, out_dim=3, bond_dim=2,
                          boundary='obc', dtype=torch.float64) if matrix \
        else tk.models.MPS(n_features=3, phys_dim=4, bond_dim=2,
                           boundary='obc', dtype=torch.float64)
    algorithm = (d.TRMSVD if cyclic else d.TTMSVD) if matrix \
        else (d.TRSVD if cyclic else d.TTSVD)
    engine = algorithm(model)
    assert isinstance(engine.source, d.FormatSource)
    result = engine.fit()
    expected = engine.source.contract_dense()
    torch.testing.assert_close(result.contract_dense(), expected,
                               atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('rank,error', [
    ([1], ValueError),
    ([1, 0, 1], ValueError),
    ([1, True, 1], TypeError),
    ('2', TypeError),
    ({1, 2}, TypeError),
])
def test_invalid_rank_profiles(cyclic, rank, error):
    algorithm = d.TRSVD if cyclic else d.TTSVD
    with pytest.raises(error, match='rank'):
        algorithm(torch.ones(4, 4, 4)).fit(rank)


def test_tr_rank_is_first_positional_fit_and_function_argument():
    tensor = torch.randn(2, 3, 4, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(509))
    result = d.TRSVD(tensor, center=1).fit(2)
    direct = d.tr_svd(tensor, 2, center=1)
    assert result.metadata['center'] == 1
    assert max(result.rank) <= 2
    assert result.rank == direct.rank
    torch.testing.assert_close(result.contract_dense(),
                               direct.contract_dense())


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda', 'mps'])
def test_rank_profiles_preserve_source_device(cyclic, device):
    if (device == 'cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    if (device == 'mps') and not torch.backends.mps.is_available():
        pytest.skip('MPS is not available')
    dtype = torch.float32 if device == 'mps' else torch.float64
    generator = torch.Generator().manual_seed(510)
    if cyclic:
        tensor = torch.randn(2, 3, 4, 5, dtype=dtype, generator=generator)
        rank = [4, 3, 10, 2]
    else:
        cores = [torch.randn(*shape, dtype=dtype, generator=generator)
                 for shape in ((4, 2), (2, 4, 3), (3, 4))]
        tensor = tk.formats.TT(cores).contract_dense()
        rank = [2, 3]
    source = d.DenseTensorSource(tensor.to(device))
    algorithm = d.TRSVD if cyclic else d.TTSVD
    result = algorithm(source, out_device=None).fit(rank, collect_metrics=True)
    assert result.device == source.device
    assert result.rank == rank
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(result.contract_dense().cpu(), tensor,
                               atol=tolerance, rtol=tolerance)
