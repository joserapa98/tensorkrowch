"""Tests for the canonical SVD API and deprecated compatibility wrappers."""

import inspect
import warnings

import pytest

import torch
import tensorkrowch as tk

import tensorkrowch.decompositions.svd.tt as tt_module
import tensorkrowch.decompositions.svd.ttm as ttm_module


class TestSVDPublicAPI:  # MARK: TestSVDPublicAPI

    def test_canonical_exports(self):
        assert tk.decompositions.TTSVD is tt_module.TTSVD
        assert tk.decompositions.TTMSVD is ttm_module.TTMSVD
        assert tk.decompositions.tt_svd is tt_module.tt_svd
        assert tk.decompositions.ttm_svd is ttm_module.ttm_svd

    def test_public_aliases_export_canonical_wrappers(self):
        assert tk.decompositions.vec_to_mps is tt_module.vec_to_mps
        assert tk.decompositions.mat_to_mpo is ttm_module.mat_to_mpo

    @pytest.mark.parametrize(
        'function, tensor, message',
        [
            (tk.decompositions.vec_to_mps,
             torch.ones(2, 3),
             '`vec_to_mps` is deprecated; use `tt_svd` instead'),
            (tk.decompositions.mat_to_mpo,
             torch.ones(2, 3),
             '`mat_to_mpo` is deprecated; use `ttm_svd` instead'),
        ],
    )
    def test_legacy_warning_is_unique_and_points_to_caller(
            self, function, tensor, message):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            function(tensor)

        matching = [warning for warning in caught
                    if str(warning.message) == message]
        assert len(matching) == 1
        assert matching[0].category is FutureWarning
        assert matching[0].filename == __file__

    def test_vec_to_mps_preserves_keywords_and_delegates(self, monkeypatch):
        tensor = torch.randn(2, 3, 4)
        calls = []

        def fake_tt_svd(**kwargs):
            calls.append(kwargs)
            return 'tt-result'

        monkeypatch.setattr(tt_module, 'tt_svd', fake_tt_svd)
        with pytest.warns(FutureWarning, match='`vec_to_mps` is deprecated'):
            result = tk.decompositions.vec_to_mps(
                vec=tensor,
                n_batches=1,
                rank=2,
                cutoff=0.1,
                atol=0.2,
                rtol=0.3,
                cum_percentage=0.8,
                renormalize=True,
                verbose=2,
                return_info=True)

        assert result == 'tt-result'
        assert calls == [{
            'tensor': tensor,
            'n_batches': 1,
            'rank': 2,
            'cutoff': 0.1,
            'atol': 0.2,
            'rtol': 0.3,
            'cum_percentage': 0.8,
            'renormalize': True,
            'output_device': None,
            'verbose': 2,
            'return_info': True,
        }]

    def test_mat_to_mpo_preserves_keywords_and_delegates(self, monkeypatch):
        tensor = torch.randn(2, 3, 4, 5)
        calls = []

        def fake_ttm_svd(**kwargs):
            calls.append(kwargs)
            return 'ttm-result'

        monkeypatch.setattr(ttm_module, 'ttm_svd', fake_ttm_svd)
        with pytest.warns(FutureWarning, match='`mat_to_mpo` is deprecated'):
            result = tk.decompositions.mat_to_mpo(
                mat=tensor,
                rank=2,
                cutoff=0.1,
                atol=0.2,
                rtol=0.3,
                cum_percentage=0.8,
                renormalize=True,
                verbose=2,
                return_info=True)

        assert result == 'ttm-result'
        assert calls == [{
            'tensor': tensor,
            'layout': 'interleaved',
            'rank': 2,
            'cutoff': 0.1,
            'atol': 0.2,
            'rtol': 0.3,
            'cum_percentage': 0.8,
            'renormalize': True,
            'output_device': None,
            'verbose': 2,
            'return_info': True,
        }]

    @pytest.mark.parametrize(
        'function, first_parameter',
        [
            (tk.decompositions.vec_to_mps, 'vec'),
            (tk.decompositions.mat_to_mpo, 'mat'),
        ],
    )
    def test_legacy_signatures_keep_historical_tensor_keyword(
            self, function, first_parameter):
        parameters = inspect.signature(function).parameters

        assert next(iter(parameters)) == first_parameter
        assert 'return_info' in parameters

    def test_legacy_return_info_matches_canonical_results(self):
        tensor = torch.randn(2, 3, 4, 2)
        matrix = torch.randn(2, 3, 4, 5)

        with pytest.warns(FutureWarning, match='`vec_to_mps` is deprecated'):
            _, tt_info = tk.decompositions.vec_to_mps(
                tensor,
                rank=2,
                return_info=True)
        with pytest.warns(FutureWarning, match='`mat_to_mpo` is deprecated'):
            _, ttm_info = tk.decompositions.mat_to_mpo(
                matrix,
                rank=2,
                return_info=True)

        assert tt_info['topology'] == 'tt'
        assert tt_info['metadata']['algorithm'] == 'tt_svd'
        assert len(tt_info['metrics']['truncations']) == 3
        assert ttm_info['topology'] == 'ttm'
        assert ttm_info['metadata']['algorithm'] == 'ttm_svd'
        assert len(ttm_info['metrics']['truncations']) == 1

    def test_canonical_functions_do_not_emit_deprecation_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            tk.decompositions.tt_svd(torch.ones(2, 3))
            tk.decompositions.ttm_svd(torch.ones(2, 3))

        messages = [str(warning.message) for warning in caught]
        assert not any('deprecated' in message for message in messages)
