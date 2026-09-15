"""Tests for lightweight tensor decomposition results."""

import inspect

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.decompositions._runtime import _RuntimePolicy


def _product_tt(vectors):
    if len(vectors) == 1:
        return [vectors[0]]
    cores = [vectors[0].unsqueeze(-1)]
    cores.extend(vector.reshape(1, -1, 1) for vector in vectors[1:-1])
    cores.append(vectors[-1].unsqueeze(0))
    return cores


def _product_tr(vectors):
    return [vector.reshape(1, -1, 1) for vector in vectors]


def _outer(vectors):
    result = vectors[0]
    for vector in vectors[1:]:
        result = torch.tensordot(result, vector, dims=0)
    return result


def _qtt_factor(values):
    """Returns a two-digit TT factor with its connector at the endpoint."""
    tensor = values.reshape(2, 2, values.shape[-1])
    return tk.decompositions.TTSVD(
        tensor, out_device=None).fit(rank=4)


class TestTensorDecompositionResults:  # MARK: TestTensorDecompositionResults

    def test_result_hierarchy_and_placeholders(self):
        result_types = tk.decompositions

        assert issubclass(result_types.TensorDecomposition1D,
                          result_types.TensorDecomposition)
        assert issubclass(result_types.TensorDecomposition2D,
                          result_types.TensorDecomposition)
        assert issubclass(result_types.TTDecomposition,
                          result_types.TensorDecomposition1D)
        assert issubclass(result_types.TRDecomposition,
                          result_types.TensorDecomposition1D)
        assert issubclass(result_types.TTMDecomposition,
                          result_types.TensorDecomposition1D)
        assert issubclass(result_types.TRMDecomposition,
                          result_types.TensorDecomposition1D)
        assert issubclass(result_types.PEPSDecomposition,
                          result_types.TensorDecomposition2D)
        assert issubclass(result_types.PEPODecomposition,
                          result_types.TensorDecomposition2D)

        for result_type in [result_types.TensorDecomposition,
                            result_types.TensorDecomposition1D,
                            result_types.TensorDecomposition2D,
                            result_types.PEPSDecomposition,
                            result_types.PEPODecomposition]:
            assert inspect.isabstract(result_type)

        for result_type in [result_types.TTDecomposition,
                            result_types.TRDecomposition,
                            result_types.TTMDecomposition,
                            result_types.TRMDecomposition]:
            assert not inspect.isabstract(result_type)

    def test_boolean_n_batches_is_rejected(self):
        with pytest.raises(TypeError, match='`n_batches` should be int type'):
            tk.decompositions.TTDecomposition(
                [torch.ones(2, 1), torch.ones(1, 3)],
                n_batches=True)

    def test_tt_validation_rank_and_dense_contraction(self):
        generator = torch.Generator().manual_seed(0)
        cores = [
            torch.randn(2, 3, dtype=torch.float64, generator=generator),
            torch.randn(3, 4, 5, dtype=torch.float64,
                        generator=generator),
            torch.randn(5, 2, dtype=torch.float64, generator=generator),
        ]
        result = tk.decompositions.TTDecomposition(cores)

        assert result.rank == [3, 5]
        assert result.input_dim == (2, 4, 2)
        assert result.output_dim is None

        expected = torch.einsum('ia,ajb,bk->ijk', *cores)
        assert torch.allclose(result.contract_dense(), expected)

    def test_batched_tt_dense_contraction(self):
        cores = [
            torch.randn(3, 2, 4, dtype=torch.float64),
            torch.randn(3, 4, 5, 6, dtype=torch.float64),
            torch.randn(3, 6, 2, dtype=torch.float64),
        ]
        result = tk.decompositions.TTDecomposition(cores, n_batches=1)

        expected = torch.einsum('xia,xajb,xbk->xijk', *cores)
        assert result.batch_shape == (3,)
        assert torch.allclose(result.contract_dense(), expected)

    def test_tr_validation_rank_and_dense_contraction(self):
        generator = torch.Generator().manual_seed(0)
        cores = [
            torch.randn(2, 3, 4, dtype=torch.float64,
                        generator=generator),
            torch.randn(4, 5, 3, dtype=torch.float64,
                        generator=generator),
            torch.randn(3, 2, 2, dtype=torch.float64,
                        generator=generator),
        ]
        result = tk.decompositions.TRDecomposition(cores)

        assert result.rank == [4, 3, 2]
        assert result.input_dim == (3, 5, 2)
        assert result.output_dim is None
        expected = torch.einsum('aib,bjc,cka->ijk', *cores)
        assert torch.allclose(result.contract_dense(), expected)

    def test_batched_tr_dense_contraction(self):
        generator = torch.Generator().manual_seed(0)
        cores = [
            torch.randn(3, 2, 3, 4,
                        dtype=torch.float64,
                        generator=generator),
            torch.randn(3, 4, 5, 2,
                        dtype=torch.float64,
                        generator=generator),
        ]
        result = tk.decompositions.TRDecomposition(cores, n_batches=1)

        expected = torch.einsum('xaib,xbja->xij', *cores)
        assert result.batch_shape == (3,)
        assert torch.allclose(result.contract_dense(), expected)

    def test_trm_validation_rank_and_dense_contraction(self):
        generator = torch.Generator().manual_seed(3)
        cores = [
            torch.randn(2, 3, 4, 5, dtype=torch.float64,
                        generator=generator),
            torch.randn(4, 6, 2, 7, dtype=torch.float64,
                        generator=generator),
        ]
        result = tk.decompositions.TRMDecomposition(cores)

        expected = torch.einsum('aibo,bjap->iojp', *cores)
        assert result.rank == [4, 2]
        assert result.in_dim == (3, 6)
        assert result.out_dim == (5, 7)
        assert torch.allclose(result.contract_dense(), expected)
        converted = result.to(dtype=torch.float32, copy=True)
        assert converted is not result
        assert converted.dtype == torch.float32
        assert converted.rank == result.rank

    def test_batched_trm_dense_contraction(self):
        generator = torch.Generator().manual_seed(4)
        cores = [
            torch.randn(3, 2, 3, 4, 5, dtype=torch.float64,
                        generator=generator),
            torch.randn(3, 4, 6, 2, 7, dtype=torch.float64,
                        generator=generator),
        ]
        result = tk.decompositions.TRMDecomposition(cores, n_batches=1)

        expected = torch.einsum('caibo,cbjap->ciojp', *cores)
        assert result.batch_shape == (3,)
        assert torch.allclose(result.contract_dense(), expected)

    def test_one_site_trm_closes_cyclic_rank(self):
        core = torch.arange(24., dtype=torch.float64).reshape(2, 3, 2, 2)
        result = tk.decompositions.TRMDecomposition([core])
        expected = torch.einsum('aiao->io', core)

        assert result.rank == [2]
        assert result.in_dim == (3,)
        assert result.out_dim == (2,)
        assert torch.equal(result.contract_dense(), expected)

        in_samples = torch.tensor([[0], [2]])
        out_samples = torch.tensor([[1], [0]])
        assert torch.equal(
            result.evaluate(in_samples, out_samples),
            expected[in_samples[:, 0], out_samples[:, 0]])
        assert torch.equal(
            result.apply(in_samples).contract_dense(),
            expected[in_samples[:, 0]])

    def test_trm_evaluate_and_apply_match_dense_contraction(self):
        generator = torch.Generator().manual_seed(5)
        cores = [
            torch.randn(2, 3, 2, 4, dtype=torch.float64,
                        generator=generator),
            torch.randn(2, 5, 2, 6, dtype=torch.float64,
                        generator=generator),
        ]
        result = tk.decompositions.TRMDecomposition(cores)
        in_samples = torch.tensor([[0, 1], [2, 4]])
        out_samples = torch.tensor([[1, 3], [3, 5]])
        dense = result.contract_dense()

        expected_entries = torch.stack([
            dense[0, 1, 1, 3],
            dense[2, 3, 4, 5],
        ])
        assert torch.allclose(
            result.evaluate(in_samples, out_samples), expected_entries)

        applied = result.apply(in_samples)
        expected_applied = torch.stack([
            dense[0, :, 1, :],
            dense[2, :, 4, :],
        ])
        assert isinstance(applied, tk.decompositions.TRDecomposition)
        assert applied.n_batches == 1
        assert torch.allclose(applied.contract_dense(), expected_applied)

    def test_trm_embedded_evaluate_and_apply_match_dense_contraction(self):
        generator = torch.Generator().manual_seed(6)
        cores = [
            torch.randn(2, 3, 2, 4, dtype=torch.float64,
                        generator=generator),
            torch.randn(2, 5, 2, 6, dtype=torch.float64,
                        generator=generator),
        ]
        result = tk.decompositions.TRMDecomposition(cores)
        in_vectors = [
            torch.randn(7, dim, dtype=torch.float64, generator=generator)
            for dim in result.in_dim
        ]
        out_vectors = [
            torch.randn(7, dim, dtype=torch.float64, generator=generator)
            for dim in result.out_dim
        ]
        dense = result.contract_dense()

        expected_entries = torch.einsum(
            'bi,bo,bj,bp,iojp->b',
            in_vectors[0],
            out_vectors[0],
            in_vectors[1],
            out_vectors[1],
            dense)
        assert torch.allclose(
            result.evaluate(in_vectors, out_vectors), expected_entries)

        expected_applied = torch.einsum(
            'bi,bj,iojp->bop', in_vectors[0], in_vectors[1], dense)
        applied = result.apply(in_vectors)
        assert torch.allclose(applied.contract_dense(), expected_applied)

    def test_trm_keeps_core_and_data_batches_separate(self):
        generator = torch.Generator().manual_seed(7)
        cores = [
            torch.randn(2, 2, 3, 2, 4, dtype=torch.float64,
                        generator=generator),
            torch.randn(2, 2, 5, 2, 6, dtype=torch.float64,
                        generator=generator),
        ]
        result = tk.decompositions.TRMDecomposition(cores, n_batches=1)
        in_samples = torch.stack([
            torch.randint(0, dim, (3, 4), generator=generator)
            for dim in result.in_dim
        ], dim=-1)
        out_samples = torch.stack([
            torch.randint(0, dim, (3, 4), generator=generator)
            for dim in result.out_dim
        ], dim=-1)
        dense = result.contract_dense()

        expected_entries = torch.stack([
            dense[batch][
                in_samples[..., 0], out_samples[..., 0],
                in_samples[..., 1], out_samples[..., 1]]
            for batch in range(dense.shape[0])
        ])
        assert torch.allclose(
            result.evaluate(
                in_samples, out_samples, n_batches=2),
            expected_entries)

        expected_applied = torch.stack([
            torch.stack([
                dense[batch][
                    in_samples[i, j, 0], :, in_samples[i, j, 1], :]
                for i in range(in_samples.shape[0])
                for j in range(in_samples.shape[1])
            ]).reshape(*in_samples.shape[:2], *result.out_dim)
            for batch in range(dense.shape[0])
        ])
        applied = result.apply(in_samples, n_batches=2)
        assert applied.n_batches == 3
        assert applied.batch_shape == (2, 3, 4)
        assert torch.allclose(applied.contract_dense(), expected_applied)

    @pytest.mark.parametrize(
        'cores, n_batches, match',
        [
            ([torch.ones(2, 3, 2)], 0,
             'left rank, input, right rank and output'),
            ([torch.ones(2, 3, 4, 5),
              torch.ones(3, 6, 2, 7)], 0,
             'Adjacent TRM ranks should match'),
            ([torch.ones(2, 3, 4, 5),
              torch.ones(4, 6, 3, 7)], 0,
             'last and first cyclic TRM ranks should match'),
            ([torch.ones(2, 2, 3, 4, 5),
              torch.ones(3, 4, 6, 2, 7)], 1,
             'same batch shape'),
        ],
    )
    def test_trm_validation_errors(self, cores, n_batches, match):
        with pytest.raises(ValueError, match=match):
            tk.decompositions.TRMDecomposition(
                cores, n_batches=n_batches)

    @pytest.mark.parametrize(
        'result_type, cores, boundary',
        [
            (
                tk.decompositions.TTDecomposition,
                [torch.arange(12.).reshape(3, 4),
                 torch.arange(20.).reshape(4, 5)],
                'obc',
            ),
            (
                tk.decompositions.TRDecomposition,
                [torch.arange(24.).reshape(2, 3, 4),
                 torch.arange(40.).reshape(4, 5, 2)],
                'pbc',
            ),
        ],
    )
    def test_state_result_initializes_mps(self,
                                          result_type,
                                          cores,
                                          boundary):
        result = result_type(cores)
        mps = tk.models.MPS(
            tensors=result.cores,
            parameterized=False)

        assert mps.boundary == boundary
        assert mps.phys_dim == list(result.input_dim)
        assert mps.bond_dim == result.rank
        assert all(torch.allclose(model_core, result_core)
                   for model_core, result_core
                   in zip(mps.tensors, result.cores))

    @pytest.mark.parametrize(
        'result',
        [
            tk.decompositions.TTDecomposition([
                torch.arange(24.).reshape(3, 2, 4),
                torch.arange(60.).reshape(3, 4, 5),
            ], n_batches=1),
            tk.decompositions.TRDecomposition([
                torch.arange(120.).reshape(3, 2, 4, 5),
                torch.arange(180.).reshape(3, 5, 6, 2),
            ], n_batches=1),
        ],
    )
    def test_batched_state_result_initializes_mps_data(self, result):
        mps_data = tk.models.MPSData(
            tensors=result.cores,
            n_batches=result.n_batches)

        assert mps_data.boundary == ('obc' if result.topology == 'tt'
                                     else 'pbc')
        assert mps_data.n_batches == result.n_batches
        assert mps_data.phys_dim == list(result.input_dim)
        assert mps_data.bond_dim == result.rank

    def test_ttm_validation_dense_contraction_and_apply(self):
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        second = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        cores = [first.unsqueeze(1), second.unsqueeze(0)]
        result = tk.decompositions.TTMDecomposition(cores)

        expected_dense = torch.einsum('io,jp->iojp', first, second)
        assert result.rank == [1]
        assert result.input_dim == (2, 2)
        assert result.output_dim == (2, 3)
        assert torch.allclose(result.contract_dense(), expected_dense)

        inputs = torch.tensor([[0, 1], [1, 0]])
        expected_apply = torch.stack([
            torch.einsum('i,j->ij', first[0], second[1]),
            torch.einsum('i,j->ij', first[1], second[0]),
        ])
        applied = result.apply(inputs)
        assert isinstance(applied, tk.decompositions.TTDecomposition)
        assert applied.n_batches == 1
        assert torch.allclose(applied.contract_dense(), expected_apply)
        assert torch.allclose(
            result(inputs).contract_dense(), expected_apply)

        out_samples = torch.tensor([[1, 2], [0, 1]])
        expected_evaluate = expected_dense[
            inputs[:, 0], out_samples[:, 0],
            inputs[:, 1], out_samples[:, 1]]
        assert torch.allclose(
            result.evaluate(inputs, out_samples), expected_evaluate)
        assert torch.allclose(
            result(inputs, out_samples), expected_evaluate)

    def test_ttm_evaluate_and_apply_with_nontrivial_ranks(self):
        generator = torch.Generator().manual_seed(2)
        cores = [
            torch.randn(2, 3, 3, dtype=torch.float64,
                        generator=generator),
            torch.randn(3, 2, 4, 2, dtype=torch.float64,
                        generator=generator),
            torch.randn(4, 3, 2, dtype=torch.float64,
                        generator=generator),
        ]
        result = tk.decompositions.TTMDecomposition(cores)
        dense = result.contract_dense()
        in_samples = torch.stack([
            torch.randint(0, dim, (2, 3), generator=generator)
            for dim in result.in_dim
        ], dim=-1)
        out_samples = torch.stack([
            torch.randint(0, dim, (2, 3), generator=generator)
            for dim in result.out_dim
        ], dim=-1)
        expected_evaluate = dense[
            in_samples[..., 0], out_samples[..., 0],
            in_samples[..., 1], out_samples[..., 1],
            in_samples[..., 2], out_samples[..., 2]]

        assert torch.allclose(
            result.evaluate(in_samples, out_samples, n_batches=2),
            expected_evaluate)

        in_vectors = [
            torch.nn.functional.one_hot(
                in_samples[..., site], num_classes=site_in_dim).to(torch.float64)
            for site, site_in_dim in enumerate(result.in_dim)
        ]
        assert torch.allclose(
            result.evaluate(in_vectors, out_samples, n_batches=2),
            expected_evaluate)
        out_vectors = [
            torch.nn.functional.one_hot(
                out_samples[..., site], num_classes=site_out_dim).to(torch.float64)
            for site, site_out_dim in enumerate(result.out_dim)
        ]
        assert torch.allclose(
            result.evaluate(in_samples, out_vectors, n_batches=2),
            expected_evaluate)
        assert torch.allclose(
            result.evaluate(in_vectors, out_vectors, n_batches=2),
            expected_evaluate)

        applied = result.apply(in_samples, n_batches=2)
        expected_apply = torch.stack([
            dense[tuple(
                index
                for site, sample in enumerate(configuration)
                for index in (sample, slice(None)))]
            for configuration in in_samples.reshape(-1, 3)
        ]).reshape(2, 3, *result.out_dim)
        assert applied.n_batches == 2
        assert torch.allclose(applied.contract_dense(), expected_apply)

    def test_one_site_ttm_evaluate_and_apply(self):
        core = torch.arange(6., dtype=torch.float64).reshape(2, 3)
        result = tk.decompositions.TTMDecomposition([core])
        in_samples = torch.tensor([[0], [1]])
        out_samples = torch.tensor([[2], [0]])

        assert torch.equal(
            result.evaluate(in_samples, out_samples),
            core[in_samples[:, 0], out_samples[:, 0]])

        applied = result.apply(in_samples)
        assert applied.n_batches == 1
        assert torch.equal(applied.contract_dense(), core[in_samples[:, 0]])

        single = result.apply(torch.tensor([1]), n_batches=0)
        assert single.n_batches == 0
        assert torch.equal(single.contract_dense(), core[1])

    def test_ttm_result_initializes_mpo(self):
        result = tk.decompositions.TTMSVD(
            torch.randn(2, 3, 4, 5), out_device=None).fit(rank=2)
        mpo = tk.models.MPO(
            tensors=result.cores,
            parameterized=False)

        assert mpo.boundary == 'obc'
        assert mpo.in_dim == list(result.input_dim)
        assert mpo.out_dim == list(result.output_dim)
        assert mpo.bond_dim == result.rank
        assert all(torch.allclose(model_core, result_core)
                   for model_core, result_core
                   in zip(mpo.tensors, result.cores))

    def test_trm_result_initializes_periodic_mpo(self):
        result = tk.decompositions.TRMSVD(
            torch.randn(2, 3, 4, 5), out_device=None).fit(rank=2)
        mpo = tk.models.MPO(
            tensors=result.cores,
            parameterized=False)

        assert mpo.boundary == 'pbc'
        assert mpo.in_dim == list(result.in_dim)
        assert mpo.out_dim == list(result.out_dim)
        assert mpo.bond_dim == result.rank
        assert all(torch.allclose(model_core, result_core)
                   for model_core, result_core
                   in zip(mpo.tensors, result.cores))

    @pytest.mark.parametrize(
        'result_type, upper_type',
        [
            (tk.decompositions.QTTTuckerDecomposition,
             tk.decompositions.TTDecomposition),
            (tk.decompositions.QTRTuckerDecomposition,
             tk.decompositions.TRDecomposition),
        ],
    )
    def test_quantized_tucker_evaluation_and_flatten(
            self, result_type, upper_type):
        dtype = torch.float64
        first = torch.tensor(
            [[1., 0.], [0., 1.], [1., 1.], [2., -1.]], dtype=dtype)
        second = torch.tensor(
            [[1., 2.], [2., 1.], [0., 1.], [1., -1.]], dtype=dtype)
        connector = torch.tensor(
            [[2., -1.], [0.5, 3.]], dtype=dtype)
        factors = (_qtt_factor(first), _qtt_factor(second))
        if upper_type is tk.decompositions.TTDecomposition:
            upper = tk.decompositions.TTSVD(
                connector, out_device=None).fit(rank=2)
        else:
            upper = upper_type([
                torch.eye(2, dtype=dtype).reshape(1, 2, 2),
                connector.reshape(2, 2, 1),
            ])
        layout = tk.decompositions.QuantizedLayout(
            n_variables=2, base=2, level=2)
        result = result_type(
            upper,
            factors,
            layout,
            tk.decompositions.UniformCoordinateMap(),
            torch.tensor([[0., 1.], [0., 1.]], dtype=dtype))
        indices = torch.tensor([[0, 0], [1, 2], [3, 1]])
        points = indices.to(dtype) / 3
        expected = torch.einsum(
            'bi,ij,bj->b', first[indices[:, 0]], connector,
            second[indices[:, 1]])

        assert torch.allclose(result.evaluate(points), expected)
        assert torch.allclose(result.evaluate_indices(indices), expected)
        assert result.input_dim == (2, 2, 2, 2)
        assert result.output_shape == ()
        assert result.flatten().topology == (
            'tt' if upper_type is tk.decompositions.TTDecomposition else 'tr')
        assert torch.allclose(
            result.flatten().contract_dense().reshape(4, 4),
            torch.einsum('ai,ij,bj->ab', first, connector, second))
        assert result.to(dtype=torch.float32).dtype == torch.float32

    def test_quantized_tucker_retains_upper_output_axes(self):
        dtype = torch.float64
        first = torch.tensor(
            [[1., 0.], [0., 1.], [1., 1.], [2., -1.]], dtype=dtype)
        second = torch.tensor(
            [[1., 2.], [2., 1.], [0., 1.], [1., -1.]], dtype=dtype)
        output_core = torch.arange(12., dtype=dtype).reshape(2, 3, 2)
        upper = tk.decompositions.TTDecomposition([
            torch.eye(2, dtype=dtype),
            output_core,
            torch.eye(2, dtype=dtype),
        ])
        result = tk.decompositions.QTTTuckerDecomposition(
            upper,
            (_qtt_factor(first), _qtt_factor(second)),
            tk.decompositions.QuantizedLayout(
                n_variables=2, base=2, level=2),
            tk.decompositions.UniformCoordinateMap(),
            torch.tensor([[0., 1.], [0., 1.]], dtype=dtype),
            variable_positions=(0, 2))
        indices = torch.tensor([[1, 3], [2, 0]])
        expected = torch.einsum(
            'bi,ioj,bj->bo', first[indices[:, 0]], output_core,
            second[indices[:, 1]])

        assert result.output_shape == (3,)
        assert result.input_dim == (2, 2, 3, 2, 2)
        assert torch.allclose(result.evaluate_indices(indices), expected)
        assert torch.allclose(
            result.flatten().contract_dense().reshape(4, 3, 4),
            torch.einsum('ai,ioj,bj->aob', first, output_core, second))

    def test_norm_overlap_fidelity_and_phase(self):
        vectors = [
            torch.tensor([1.0, 2.0], dtype=torch.complex128),
            torch.tensor([2.0, -1.0], dtype=torch.complex128),
        ]
        reference = tk.decompositions.TTDecomposition(_product_tt(vectors))

        phase = torch.exp(torch.tensor(0.7j, dtype=torch.complex128))
        phased_cores = _product_tt(vectors)
        phased_cores[0] = 3 * phase * phased_cores[0]
        phased = tk.decompositions.TTDecomposition(phased_cores)

        assert torch.allclose(reference.norm(),
                              torch.linalg.vector_norm(_outer(vectors)))
        assert torch.allclose(reference.normalized_overlap(phased), phase)
        assert torch.allclose(reference.fidelity(phased),
                              torch.ones((), dtype=torch.float64))

    def test_orthogonal_states(self):
        zero = torch.tensor([1.0, 0.0])
        one = torch.tensor([0.0, 1.0])
        state_00 = tk.decompositions.TTDecomposition(
            _product_tt([zero, zero]))
        state_01 = tk.decompositions.TTDecomposition(
            _product_tt([zero, one]))

        assert torch.allclose(state_00.normalized_overlap(state_01),
                              torch.tensor(0.0))
        assert torch.allclose(state_00.fidelity(state_01), torch.tensor(0.0))

    def test_tt_tr_overlap(self):
        vectors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, -1.0])]
        tt = tk.decompositions.TTDecomposition(_product_tt(vectors))
        tr = tk.decompositions.TRDecomposition(_product_tr(vectors))

        assert torch.allclose(tt.normalized_overlap(tr), torch.tensor(1.0))
        assert torch.allclose(tr.fidelity(tt), torch.tensor(1.0))

    @pytest.mark.parametrize('pair', ['tt-tt', 'tr-tr', 'tt-tr'])
    def test_overlap_matches_dense_contraction(self, pair):
        generator = torch.Generator().manual_seed(3)
        if pair == 'tt-tt':
            first = tk.decompositions.TTDecomposition([
                torch.randn(2, 2, dtype=torch.complex128,
                            generator=generator),
                torch.randn(2, 3, dtype=torch.complex128,
                            generator=generator),
            ])
            second = tk.decompositions.TTDecomposition([
                torch.randn(2, 3, dtype=torch.complex128,
                            generator=generator),
                torch.randn(3, 3, dtype=torch.complex128,
                            generator=generator),
            ])
        elif pair == 'tr-tr':
            first = tk.decompositions.TRDecomposition([
                torch.randn(2, 2, 3, dtype=torch.complex128,
                            generator=generator),
                torch.randn(3, 3, 2, dtype=torch.complex128,
                            generator=generator),
            ])
            second = tk.decompositions.TRDecomposition([
                torch.randn(1, 2, 2, dtype=torch.complex128,
                            generator=generator),
                torch.randn(2, 3, 1, dtype=torch.complex128,
                            generator=generator),
            ])
        else:
            first = tk.decompositions.TTDecomposition([
                torch.randn(2, 2, dtype=torch.complex128,
                            generator=generator),
                torch.randn(2, 3, dtype=torch.complex128,
                            generator=generator),
            ])
            second = tk.decompositions.TRDecomposition([
                torch.randn(2, 2, 3, dtype=torch.complex128,
                            generator=generator),
                torch.randn(3, 3, 2, dtype=torch.complex128,
                            generator=generator),
            ])

        first_dense = first.contract_dense().flatten()
        second_dense = second.contract_dense().flatten()
        expected = torch.vdot(first_dense, second_dense) / (
            first_dense.norm() * second_dense.norm())

        assert torch.allclose(first.normalized_overlap(second), expected)
        assert torch.allclose(first.fidelity(second), expected.abs().square())

    def test_batched_overlap_matches_dense_contraction(self):
        first = tk.decompositions.TTDecomposition([
            torch.randn(3, 2, 2, dtype=torch.float64),
            torch.randn(3, 2, 4, dtype=torch.float64),
        ], n_batches=1)
        second = tk.decompositions.TTDecomposition([
            torch.randn(3, 2, 3, dtype=torch.float64),
            torch.randn(3, 3, 4, dtype=torch.float64),
        ], n_batches=1)

        first_dense = first.contract_dense().flatten(1)
        second_dense = second.contract_dense().flatten(1)
        expected = (first_dense.conj() * second_dense).sum(1) / (
            first_dense.norm(dim=1) * second_dense.norm(dim=1))

        assert torch.allclose(first.normalized_overlap(second), expected)

    def test_ttm_norm_matches_dense(self):
        cores = [torch.randn(2, 3, 2), torch.randn(3, 2, 4)]
        result = tk.decompositions.TTMDecomposition(cores)

        expected = torch.linalg.vector_norm(result.contract_dense())
        assert torch.allclose(result.norm(), expected)

    def test_zero_norm_fidelity_error(self):
        zero = tk.decompositions.TTDecomposition(
            _product_tt([torch.zeros(2), torch.ones(2)]))
        nonzero = tk.decompositions.TTDecomposition(
            _product_tt([torch.ones(2), torch.ones(2)]))

        with pytest.raises(ValueError, match='zero-norm'):
            zero.fidelity(nonzero)

    def test_evaluate_and_sample_error(self):
        cores = [torch.randn(2, 3), torch.randn(3, 2)]
        result = tk.decompositions.TTDecomposition(cores)
        dense = result.contract_dense()
        samples = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = dense[tuple(samples.t())]

        assert torch.allclose(result.evaluate(samples), expected)
        assert torch.allclose(
            result(torch.tensor([1, 0]), n_batches=0), dense[1, 0])
        assert torch.allclose(
            result([torch.tensor(1), torch.tensor(0)], n_batches=0),
            dense[1, 0])

        record = result.error(
            lambda values: dense[tuple(values.t())], samples)
        assert record.kind == 'samples'
        assert record.absolute == pytest.approx(0.0, abs=1e-6)
        assert record.relative == pytest.approx(0.0, abs=1e-6)

    def test_tr_evaluate_matches_dense(self):
        cores = [torch.randn(2, 3, 4), torch.randn(4, 2, 2)]
        result = tk.decompositions.TRDecomposition(cores)
        dense = result.contract_dense()
        samples = torch.tensor([[0, 0], [1, 1], [2, 0]])

        expected = dense[tuple(samples.t())]
        assert torch.allclose(result.evaluate(samples), expected)

    @pytest.mark.parametrize('topology', ['tt', 'tr'])
    def test_evaluate_keeps_core_and_data_batches(self, topology):
        generator = torch.Generator().manual_seed(1)
        if topology == 'tt':
            cores = [
                torch.randn(2, 3, 2, 2, dtype=torch.float64,
                            generator=generator),
                torch.randn(2, 3, 2, 3, dtype=torch.float64,
                            generator=generator),
            ]
            result = tk.decompositions.TTDecomposition(cores, n_batches=2)
        else:
            cores = [
                torch.randn(2, 3, 2, 2, 2, dtype=torch.float64,
                            generator=generator),
                torch.randn(2, 3, 2, 3, 2, dtype=torch.float64,
                            generator=generator),
            ]
            result = tk.decompositions.TRDecomposition(cores, n_batches=2)

        indices = torch.randint(
            0, 2, (4, 5, 2), generator=generator)
        indices[..., 1] = torch.randint(
            0, 3, (4, 5), generator=generator)
        dense = result.contract_dense()
        expected = torch.stack([
            torch.stack([
                dense[i, j][indices[..., 0], indices[..., 1]]
                for j in range(dense.shape[1])
            ])
            for i in range(dense.shape[0])
        ])

        assert torch.allclose(
            result.evaluate(indices, n_batches=2), expected)
        assert torch.allclose(
            result([indices[..., 0], indices[..., 1]], n_batches=2),
            expected)

        vectors = [
            torch.nn.functional.one_hot(
                indices[..., site], num_classes=site_in_dim).to(torch.float64)
            for site, site_in_dim in enumerate(result.in_dim)
        ]
        assert torch.allclose(result(vectors, n_batches=2), expected)

    def test_evaluate_with_tensor_and_site_vectors(self):
        generator = torch.Generator().manual_seed(0)
        cores = [
            torch.randn(2, 3, dtype=torch.float64, generator=generator),
            torch.randn(3, 2, dtype=torch.float64, generator=generator),
        ]
        result = tk.decompositions.TTDecomposition(cores)
        samples = torch.tensor(
            [[0.2, 0.4], [0.5, 0.7]], dtype=torch.float64)
        calls = []

        def embedding(values):
            calls.append(values.shape)
            return torch.stack([torch.ones_like(values), values], dim=-1)

        vectors = [
            torch.stack([torch.ones_like(samples[:, site]),
                         samples[:, site]], dim=-1)
            for site in range(2)
        ]
        expected = torch.einsum(
            'xi,ia,aj,xj->x', vectors[0], cores[0], cores[1], vectors[1])

        embedded = embedding(samples)
        assert torch.allclose(result.evaluate(embedded), expected)
        assert calls == [samples.shape]

        assert torch.allclose(result.evaluate(vectors), expected)

    def test_to_cpu_and_as_info(self):
        result = tk.decompositions.TTDecomposition(
            _product_tt([torch.ones(2), torch.ones(3)]),
            metadata={'source': 'test'})

        assert result.to() is result
        copied = result.to(dtype=torch.float64, copy=True)
        assert copied is not result
        assert copied.dtype == torch.float64
        assert copied.cpu().device.type == 'cpu'
        assert copied.as_info()['rank'] == [1]
        assert copied.as_info()['input_dim'] == [2, 3]
        assert copied.as_info()['output_dim'] is None
        assert copied.as_info()['metadata'] == {'source': 'test'}

    @pytest.mark.parametrize(
        'result_type, cores, match',
        [
            (tk.decompositions.TTDecomposition,
             [torch.randn(2, 3), torch.randn(4, 2)],
             'Adjacent TT ranks'),
            (tk.decompositions.TRDecomposition,
             [torch.randn(2, 3, 4), torch.randn(4, 2, 3)],
             'cyclic TR ranks'),
            (tk.decompositions.TTMDecomposition,
             [torch.randn(2, 3, 2), torch.randn(4, 2, 2)],
             'Adjacent TTM ranks'),
        ],
    )
    def test_core_validation(self, result_type, cores, match):
        with pytest.raises(ValueError, match=match):
            result_type(cores)

    def test_core_dtype_validation(self):
        cores = [torch.randn(2, 3), torch.randn(3, 2, dtype=torch.float64)]
        with pytest.raises(ValueError, match='same dtype'):
            tk.decompositions.TTDecomposition(cores)


class TestRuntimePolicy:  # MARK: TestRuntimePolicy

    def test_inference_prepare_finalize_and_timer(self):
        tensor = torch.ones(2, dtype=torch.float32)
        runtime = _RuntimePolicy.from_tensor(tensor, dtype=torch.float64)

        prepared = runtime.prepare(tensor)
        finalized = runtime.finalize(prepared)
        with runtime.timer() as timer:
            _ = prepared.square()

        assert runtime.device == tensor.device
        assert runtime.out_device == torch.device('cpu')
        assert runtime.dtype == torch.float64
        assert prepared.dtype == torch.float64
        assert finalized.device.type == 'cpu'
        assert timer.elapsed is not None
        assert timer.elapsed >= 0

    def test_output_device_none_keeps_tensor(self):
        tensor = torch.ones(2)
        runtime = _RuntimePolicy.from_tensor(tensor, out_device=None)

        assert runtime.finalize(tensor) is tensor

    def test_invalid_dtype(self):
        with pytest.raises(TypeError):
            _RuntimePolicy(dtype='float64')
