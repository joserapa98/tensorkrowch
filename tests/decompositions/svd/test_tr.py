"""Tests for TR-SVD classes and functional interfaces."""

import pytest

import torch
import tensorkrowch as tk

import tensorkrowch.decompositions.svd.tr as tr_module


SVD_METHODS = ['svd', 'qr_svd']
DEVICE_NAMES = ['cpu', 'cuda', 'mps']


def _device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    if name == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is not available')
    return torch.device(name)


class TestTRSVD:  # MARK: TestTRSVD

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('renormalize', [False, True])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('center', [1, 2, 3])
    def test_rank_discovery_reconstructs_dense_tensor(
            self, svd_method, renormalize, dtype, center):
        generator = torch.Generator().manual_seed(0)
        tensor = torch.randn(
            2, 3, 2, 2,
            dtype=dtype,
            generator=generator)

        with tk.svd_method(svd_method):
            result = tk.decompositions.TRSVD(
                tensor,
                center=center,
                output_device=None).fit(
                    renormalize=renormalize,
                    collect_metrics=True)

        assert isinstance(result, tk.decompositions.TRDecomposition)
        assert result.input_dim == tuple(tensor.shape)
        assert result.dtype == dtype
        assert result.metadata['center'] == center
        assert result.metadata['rank_mode'] == 'discovery'
        assert result.metadata['truncation_errors'] == 'local_diagnostics'
        assert result.metrics.errors == []
        assert [record.site for record in result.metrics.truncations] == [
            0, 1, 2]
        assert all(record.global_relative_contribution is None
                   for record in result.metrics.truncations)
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)

    def test_leftmost_center_uses_first_input_dimension(self):
        tensor = torch.randn(2, 5, 3, dtype=torch.float64)

        result = tk.decompositions.TRSVD(
            tensor,
            center=1,
            output_device=None).fit()

        assert result.cores[0].shape[1] == tensor.shape[0]
        assert result.input_dim == tuple(tensor.shape)
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-10, atol=1e-12)

    def test_default_center_is_middle_cut(self):
        decomposer = tk.decompositions.TRSVD(
            torch.randn(2, 2, 2, 2, 2))

        result = decomposer.fit()

        assert decomposer.center == 2
        assert result.metadata['center'] == 2

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_shared_rank_is_a_global_cap(self, svd_method):
        generator = torch.Generator().manual_seed(1)
        tensor = torch.randn(
            3, 4, 2, 3,
            dtype=torch.float64,
            generator=generator)

        with tk.svd_method(svd_method):
            result = tk.decompositions.TRSVD(
                tensor,
                output_device=None).fit(
                    rank=2,
                    rtol=0.1,
                    cum_percentage=0.95,
                    collect_metrics=True)

        assert max(result.rank) <= 2
        assert result.metadata['rank_mode'] == 'shared'
        assert result.metadata['requested_rank'] == 2
        assert result.metrics.errors == []
        assert {record.phase for record in result.metrics.truncations} == {
            'left_subchain',
            'initial_bipartition',
            'right_subchain',
        }
        assert 'local diagnostics' in result.metrics.warnings[-1]

    def test_rank_sequence_caps_every_tr_rank(self):
        tensor = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        requested_rank = (2, 3, 2, 4)

        result = tk.decompositions.TRSVD(
            tensor,
            center=2,
            output_device=None).fit(rank=requested_rank)

        assert all(actual <= cap
                   for actual, cap in zip(result.rank, requested_rank))
        assert result.metadata['requested_rank'] == requested_rank
        assert result.metadata['cycle_rank'] == 2
        assert result.metadata['center_rank'] == 3
        assert result.metadata['initial_selected_rank'] == 6
        assert result.metadata['initial_capacity'] == 6
        assert result.metadata['structural_padding'] == 0

    def test_prime_rank_discovery_uses_closest_exact_factors(self):
        result = tk.decompositions.TRSVD(
            torch.eye(5, dtype=torch.float64),
            output_device=None).fit(collect_metrics=True)

        assert result.rank == [5, 1]
        assert result.metadata['initial_selected_rank'] == 5
        assert result.metadata['cycle_rank'] == 1
        assert result.metadata['center_rank'] == 5
        assert result.metadata['initial_capacity'] == 5
        assert result.metadata['structural_padding'] == 0
        assert not any('structural zero' in warning
                       for warning in result.metrics.warnings)
        assert torch.allclose(
            result.contract_dense(),
            torch.eye(5, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12)

    def test_nondivisible_selected_rank_under_shared_cap(self):
        tensor = torch.diag(torch.tensor(
            [5.0, 3.0, 1.0, 0.0, 0.0], dtype=torch.float64))

        result = tk.decompositions.TRSVD(
            tensor,
            output_device=None).fit(rank=2, cutoff=0)

        assert result.rank == [2, 2]
        assert result.metadata['initial_selected_rank'] == 3
        assert result.metadata['initial_capacity'] == 4
        assert result.metadata['structural_padding'] == 1

    def test_asymmetric_rank_sequence_caps(self):
        tensor = torch.eye(5, dtype=torch.float64)

        result = tk.decompositions.TRSVD(
            tensor,
            output_device=None).fit(rank=(2, 3))

        assert result.rank == [2, 3]
        assert result.metadata['initial_selected_rank'] == 5
        assert result.metadata['initial_capacity'] == 6
        assert result.metadata['structural_padding'] == 1
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_zero_tensor_is_finite(self, svd_method, dtype):
        tensor = torch.zeros(2, 3, 4, dtype=dtype)

        with tk.svd_method(svd_method):
            result = tk.decompositions.TRSVD(
                tensor,
                output_device=None).fit(
                    rank=1,
                    cutoff=0,
                    renormalize=True,
                    collect_metrics=True)

        assert result.rank == [1, 1, 1]
        assert all(torch.isfinite(core).all() for core in result.cores)
        assert torch.equal(result.contract_dense(), tensor)
        assert result.metrics.errors == []
        assert all(record.local_absolute_error == 0
                   for record in result.metrics.truncations)

    def test_repeated_fits_have_independent_results(self):
        tensor = torch.randn(3, 4, 5, 2, dtype=torch.float64)
        decomposer = tk.decompositions.TRSVD(
            tensor,
            output_device=None)

        rank_one = decomposer.fit(rank=1)
        rank_three = decomposer.fit(rank=3)

        assert rank_one.rank == [1, 1, 1, 1]
        assert max(rank_three.rank) <= 3
        assert rank_one.metrics is not rank_three.metrics
        assert rank_one.cores[0] is not rank_three.cores[0]
        assert torch.linalg.vector_norm(
            tensor - rank_three.contract_dense()) <= torch.linalg.vector_norm(
                tensor - rank_one.contract_dense())

    def test_reuses_tt_svd_for_initial_and_subchain_decompositions(
            self, monkeypatch):
        tensor = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        fit_shapes = []
        original_fit = tr_module.TTSVD.fit

        def tracked_fit(self, *args, **kwargs):
            fit_shapes.append(tuple(self.tensor.shape))
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(tr_module.TTSVD, 'fit', tracked_fit)
        tk.decompositions.TRSVD(
            tensor,
            center=2,
            output_device=None).fit()

        assert len(fit_shapes) == 3
        assert fit_shapes[0] == (6, 20)

    @pytest.mark.parametrize('device_name', DEVICE_NAMES)
    def test_output_device_policy(self, device_name):
        device = _device(device_name)
        tensor = torch.randn(2, 3, 4, 2, device=device)

        cpu_result = tk.decompositions.TRSVD(tensor).fit(rank=2)
        active_result = tk.decompositions.TRSVD(
            tensor,
            output_device=None).fit(rank=2)

        assert all(core.device.type == 'cpu' for core in cpu_result.cores)
        assert all(core.device == device for core in active_result.cores)
        assert torch.allclose(
            active_result.contract_dense(),
            cpu_result.contract_dense().to(device),
            rtol=1e-5,
            atol=1e-6)

    def test_fast_path_skips_diagnostics(self, monkeypatch):
        def unexpected_timer(*args, **kwargs):
            pytest.fail('The fast path should not start synchronized timers')

        def unexpected_event(*args, **kwargs):
            pytest.fail('The fast path should not construct observer events')

        monkeypatch.setattr(
            tr_module._RuntimePolicy, 'timer', unexpected_timer)
        monkeypatch.setattr(tr_module, 'DecompositionEvent', unexpected_event)
        result = tk.decompositions.TRSVD(
            torch.randn(2, 3, 4, 2),
            output_device=None).fit(rank=2)

        assert result.metrics.as_info() == {
            'errors': [],
            'truncations': [],
            'timings': [],
            'fidelities': [],
            'warnings': [],
        }

    def test_history_and_console_observers(self, capsys):
        history = tk.decompositions.HistoryObserver()

        result = tk.decompositions.TRSVD(
            torch.randn(2, 3, 4, 2),
            output_device=None).fit(
                rank=2,
                verbose=3,
                observer=history)

        output = capsys.readouterr().out
        assert 'TR-SVD\n======' in output
        assert 'Summary\n-------' in output
        assert history.metrics is result.metrics
        assert [event.name for event in history.events] == [
            'start',
            'site_complete',
            'site_complete',
            'site_complete',
            'summary',
            'core',
            'core',
            'core',
            'core',
        ]

    @pytest.mark.parametrize(
        'constructor, kwargs, error_type, match',
        [
            (([1, 2],), {}, TypeError,
             '`tensor` should be torch.Tensor type'),
            ((torch.ones(2),), {}, ValueError,
             'at least two TR sites'),
            ((torch.ones(2, 3),), {'center': 0}, ValueError,
             '1 <= center < tensor.ndim'),
            ((torch.ones(2, 3),), {'center': 2}, ValueError,
             '1 <= center < tensor.ndim'),
            ((torch.ones(2, 3),), {'center': True}, TypeError,
             '`center` should be int type'),
        ],
    )
    def test_constructor_errors(self,
                                constructor,
                                kwargs,
                                error_type,
                                match):
        with pytest.raises(error_type, match=match):
            tk.decompositions.TRSVD(*constructor, **kwargs)

    @pytest.mark.parametrize(
        'kwargs, error_type, match',
        [
            ({'rank': 0}, ValueError, '`rank` should be a positive integer'),
            ({'rank': True}, TypeError, '`rank` should be int or a sequence'),
            ({'rank': (1, 2)}, ValueError, 'one value per site'),
            ({'rank': (1, 2.0, 3)}, TypeError, 'contain only ints'),
            ({'rank': (1, 0, 3)}, ValueError, 'contain positive values'),
            ({'center': 0}, ValueError, '1 <= center < tensor.ndim'),
            ({'rtol': 2}, ValueError, '`rtol` should be a number between'),
            ({'renormalize': 1}, TypeError,
             '`renormalize` should be bool type'),
            ({'collect_metrics': 1}, TypeError,
             '`collect_metrics` should be bool type'),
            ({'verbose': 4}, ValueError,
             '`verbose` should be between 0 and 3'),
            ({'observer': object()}, TypeError,
             '`observer` should implement'),
        ],
    )
    def test_fit_errors(self, kwargs, error_type, match):
        decomposer = tk.decompositions.TRSVD(torch.ones(2, 3, 4))
        with pytest.raises(error_type, match=match):
            decomposer.fit(**kwargs)

    def test_gradcheck(self):
        tensor = torch.randn(
            2, 3, 2,
            dtype=torch.float64,
            requires_grad=True)

        def reconstruct(value):
            return tk.decompositions.TRSVD(
                value,
                output_device=None).fit().contract_dense()

        with tk.svd_method('qr_svd'):
            assert torch.autograd.gradcheck(
                reconstruct,
                (tensor,),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3)


class TestTRSVDFunction:  # MARK: TestTRSVDFunction

    def test_function_returns_cores_and_optional_info(self, monkeypatch):
        tensor = torch.randn(2, 3, 4, 2)
        collect_metrics_calls = []
        original_fit = tr_module.TRSVD.fit

        def tracked_fit(self, *args, **kwargs):
            collect_metrics_calls.append(kwargs['collect_metrics'])
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(tr_module.TRSVD, 'fit', tracked_fit)
        cores = tk.decompositions.tr_svd(tensor, rank=2)
        info_cores, info = tk.decompositions.tr_svd(
            tensor,
            rank=2,
            return_info=True)

        assert collect_metrics_calls == [False, True]
        assert info['topology'] == 'tr'
        assert info['rank'] == [2, 2, 2, 2]
        assert info['input_dim'] == [2, 3, 4, 2]
        assert info['metadata']['algorithm'] == 'tr_svd'
        assert info['metrics']['errors'] == []
        assert len(info['metrics']['truncations']) == 3
        assert len(cores) == len(info_cores) == 4

    def test_return_info_validation(self):
        with pytest.raises(TypeError, match='`return_info` should be bool type'):
            tk.decompositions.tr_svd(
                torch.ones(2, 3), return_info=1)
