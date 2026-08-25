"""Tests for lightweight tensor decomposition results."""

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


class TestTensorDecompositionResults:  # MARK: TestTensorDecompositionResults

    def test_tt_validation_rank_and_dense_contraction(self):
        cores = [
            torch.randn(2, 3),
            torch.randn(3, 4, 5),
            torch.randn(5, 2),
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
        cores = [
            torch.randn(2, 3, 4),
            torch.randn(4, 5, 3),
            torch.randn(3, 2, 2),
        ]
        result = tk.decompositions.TRDecomposition(cores)

        assert result.rank == [4, 3, 2]
        assert result.input_dim == (3, 5, 2)
        assert result.output_dim is None
        expected = torch.einsum('aib,bjc,cka->ijk', *cores)
        assert torch.allclose(result.contract_dense(), expected)

    def test_batched_tr_dense_contraction(self):
        cores = [
            torch.randn(3, 2, 3, 4),
            torch.randn(3, 4, 5, 2),
        ]
        result = tk.decompositions.TRDecomposition(cores, n_batches=1)

        expected = torch.einsum('xaib,xbja->xij', *cores)
        assert result.batch_shape == (3,)
        assert torch.allclose(result.contract_dense(), expected)

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
        assert torch.allclose(result.apply(inputs), expected_apply)

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

    def test_evaluate_with_shared_and_site_embeddings(self):
        cores = [torch.randn(2, 3), torch.randn(3, 2)]
        result = tk.decompositions.TTDecomposition(cores)
        samples = torch.tensor([[0.2, 0.4], [0.5, 0.7]])

        def embedding(values):
            return torch.stack([torch.ones_like(values), values], dim=-1)

        vectors = [embedding(samples[:, site]) for site in range(2)]
        expected = torch.einsum(
            'xi,ia,aj,xj->x', vectors[0], cores[0], cores[1], vectors[1])

        assert torch.allclose(result.evaluate(samples, embedding), expected)
        assert torch.allclose(
            result.evaluate(samples, [embedding, embedding]), expected)

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
        assert runtime.output_device == torch.device('cpu')
        assert runtime.dtype == torch.float64
        assert prepared.dtype == torch.float64
        assert finalized.device.type == 'cpu'
        assert timer.elapsed is not None
        assert timer.elapsed >= 0

    def test_output_device_none_keeps_tensor(self):
        tensor = torch.ones(2)
        runtime = _RuntimePolicy.from_tensor(tensor, output_device=None)

        assert runtime.finalize(tensor) is tensor

    def test_invalid_dtype(self):
        with pytest.raises(TypeError):
            _RuntimePolicy(dtype='float64')
