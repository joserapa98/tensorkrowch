"""Characterization tests for the legacy TT-RSS implementation."""

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.decompositions.tt_decompositions import (
    create_projector,
    extend_with_output,
    sketching,
)


_DEVICE_CASES = [
    torch.device('cpu'),
    pytest.param(
        torch.device('cuda'),
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='CUDA is unavailable')),
    pytest.param(
        torch.device('mps'),
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason='MPS is unavailable')),
]


def _binary_problem(dtype=torch.float64):
    """Returns a small scalar problem exactly represented at rank two."""
    domain = torch.tensor([0., 1.], dtype=torch.float64)
    samples = torch.cartesian_prod(domain, domain, domain)

    def function(data):
        values = 1 + data.prod(dim=1, keepdim=True)
        if dtype.is_complex:
            values = values + 1j * data.sum(dim=1, keepdim=True)
        return values.to(dtype)

    def embedding(data):
        return torch.stack((1 - data, data), dim=-1)

    return function, embedding, samples, domain


def _vector_problem():
    """Returns a vector function whose input/output dimensions differ."""
    domain = torch.tensor([-1., 0., 1.], dtype=torch.float64)
    samples = torch.cartesian_prod(domain, domain, domain)

    def function(data):
        return torch.stack(
            (1 + data.sum(dim=1), 2 + data.prod(dim=1)), dim=1)

    def embedding(data):
        return torch.stack((torch.ones_like(data), data, data.square()), dim=-1)

    labels = (samples[:, 0] > 0).long()
    return function, embedding, samples, domain, labels


def _input_dim(cores):
    """Extracts standard TT input dimensions from raw open-boundary cores."""
    if len(cores) == 1:
        return [cores[0].shape[0]]
    return [cores[0].shape[0], *(
        core.shape[1] for core in cores[1:])]


class TestLegacyTTRSSWorkflow:  # MARK: TestLegacyTTRSSWorkflow

    def test_scalar_function_returns_mps_compatible_cores_and_info(self):
        function, embedding, samples, domain = _binary_problem()
        cores, info = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            generator=torch.Generator().manual_seed(700),
            verbose=False,
            return_info=True)
        model = tk.models.MPS(tensors=cores, parameterized=False)

        assert [tuple(core.shape) for core in cores] == [
            (2, 2), (2, 2, 2), (2, 2)]
        assert model.phys_dim == [2, 2, 2]
        assert info['total_time'] >= 0
        assert info['val_eps'] < 1e-10

    @pytest.mark.parametrize('out_position', [0, 2, 3])
    def test_vector_output_first_middle_last_with_explicit_labels(
            self, out_position):
        function, embedding, samples, domain, labels = _vector_problem()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            labels=labels,
            domain=domain,
            out_position=out_position,
            rank=4,
            generator=torch.Generator().manual_seed(701),
            verbose=False)
        model = tk.models.MPSLayer(
            tensors=cores,
            out_position=out_position,
            parameterized=False)

        expected = [3, 3, 3]
        expected.insert(out_position, 2)
        assert _input_dim(cores) == expected
        assert model.phys_dim == expected
        assert model.out_position == out_position

    def test_default_vector_output_is_equally_centered_for_one_axis(self):
        function, embedding, samples, domain, labels = _vector_problem()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            labels=labels,
            domain=domain,
            rank=4,
            generator=torch.Generator().manual_seed(702),
            verbose=False)

        assert _input_dim(cores) == [3, 3, 2, 3]

    @pytest.mark.parametrize('domain_kind', ['shared', 'per_site', 'inferred'])
    def test_shared_per_site_and_inferred_domains(self, domain_kind):
        function, embedding, samples, shared = _binary_problem()
        if domain_kind == 'shared':
            domain = shared
        elif domain_kind == 'per_site':
            domain = [shared.clone() for _ in range(samples.shape[1])]
        else:
            domain = None
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            generator=torch.Generator().manual_seed(703),
            verbose=False)

        assert [tuple(core.shape) for core in cores] == [
            (2, 2), (2, 2, 2), (2, 2)]
        assert all(torch.isfinite(core).all() for core in cores)

    def test_vector_coordinate_samples_and_domains(self):
        domain = torch.tensor(
            [[0., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)
        ids = torch.cartesian_prod(*(torch.arange(3) for _ in range(3)))
        samples = torch.stack(
            [domain.index_select(0, ids[:, site]) for site in range(3)],
            dim=1)

        def function(data):
            return (1 + data.sum(dim=(1, 2))).unsqueeze(1)

        def embedding(data):
            return torch.cat((torch.ones_like(data[..., :1]), data), dim=-1)

        cores, info = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=3,
            generator=torch.Generator().manual_seed(704),
            verbose=False,
            return_info=True)

        assert [tuple(core.shape) for core in cores] == [
            (3, 3), (3, 3, 3), (3, 3)]
        assert info['val_eps'] < 1e-10

    @pytest.mark.parametrize(
        'truncation',
        [
            {'cutoff': 0.0},
            {'atol': 0.0},
            {'rtol': 0.0},
            {'cum_percentage': 1.0},
        ])
    def test_modern_truncation_options_reach_truncated_svd(self, truncation):
        function, embedding, samples, domain = _binary_problem()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            generator=torch.Generator().manual_seed(705),
            verbose=False,
            **truncation)

        assert all(torch.isfinite(core).all() for core in cores)
        assert all(max(core.shape) <= 2 for core in cores)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_real_and_complex_dtype(self, dtype):
        function, embedding, samples, domain = _binary_problem(dtype)
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            device=torch.device('cpu'),
            dtype=dtype,
            generator=torch.Generator().manual_seed(706),
            verbose=False)

        assert all(core.dtype == dtype for core in cores)
        assert all(core.device.type == 'cpu' for core in cores)
        assert all(torch.isfinite(core).all() for core in cores)

    @pytest.mark.parametrize('device', _DEVICE_CASES)
    def test_explicit_compute_device_returns_final_cpu_cores(self, device):
        function, embedding, samples, domain = _binary_problem(torch.float32)
        samples = samples.float()
        domain = domain.float()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            device=device,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(706),
            verbose=False)

        assert all(core.dtype == torch.float32 for core in cores)
        assert all(core.device.type == 'cpu' for core in cores)
        assert all(torch.isfinite(core).all() for core in cores)

    def test_generator_is_reproducible_and_does_not_touch_global_rng(self):
        function, embedding, samples, _ = _vector_problem()[:4]
        torch.manual_seed(707)
        initial_state = torch.random.get_rng_state().clone()
        results = []
        for _ in range(2):
            results.append(tk.decompositions.tt_rss(
                function=function,
                embedding=embedding,
                sketch_samples=samples,
                domain=None,
                rank=3,
                generator=torch.Generator().manual_seed(708),
                verbose=False))

        assert torch.equal(torch.random.get_rng_state(), initial_state)
        assert all(torch.equal(first, second)
                   for first, second in zip(*results))


class TestLegacySketchKernels:  # MARK: TestLegacySketchKernels

    def test_sketching_matches_explicit_cartesian_product(self):
        left = torch.tensor([[0.], [1.]], dtype=torch.float64)
        right = torch.tensor([[10.], [20.], [30.]], dtype=torch.float64)

        def function(data):
            return data.sum(dim=1, keepdim=True)

        result = sketching(
            function=function,
            tensors_list=[left, right],
            out_position=-1,
            batch_size=2,
            device=torch.device('cpu'),
            dtype=torch.float64)

        assert torch.equal(result, left + right.mT)

    def test_create_projector_matches_prefix_lookup_oracle(self):
        parent = torch.tensor([
            [0, 0], [0, 1], [1, 0], [1, 1]])
        child = torch.tensor([
            [0, 0, 2], [0, 0, 3], [0, 1, 4],
            [1, 0, 5], [1, 1, 6], [1, 1, 7]])

        prefix_ids, new_values = create_projector(parent, child)

        assert torch.equal(prefix_ids, torch.tensor([0, 0, 1, 2, 3, 3]))
        assert torch.equal(new_values, child[:, -1:])
        assert torch.equal(
            torch.cat((parent.index_select(0, prefix_ids), new_values), dim=1),
            child)

    def test_explicit_and_sampled_labels_extend_the_requested_position(self):
        samples = torch.tensor([[0., 1.], [1., 0.], [1., 1.]])

        def function(data):
            return torch.tensor(
                [[1., 0.], [0., 1.], [1., 1.]], dtype=data.dtype)[:len(data)]

        explicit, explicit_values = extend_with_output(
            function,
            samples,
            labels=torch.tensor([1, 0, 1]),
            out_position=1,
            batch_size=2,
            device=torch.device('cpu'))
        sampled_first = extend_with_output(
            function,
            samples,
            labels=None,
            out_position=1,
            batch_size=3,
            device=torch.device('cpu'),
            generator=torch.Generator().manual_seed(709))
        sampled_second = extend_with_output(
            function,
            samples,
            labels=None,
            out_position=1,
            batch_size=3,
            device=torch.device('cpu'),
            generator=torch.Generator().manual_seed(709))

        assert torch.equal(explicit[:, 1], torch.tensor([1., 0., 1.]))
        assert torch.equal(explicit_values, torch.ones(3, 1))
        assert torch.equal(sampled_first[0], sampled_second[0])
        assert torch.equal(sampled_first[1], sampled_second[1])


class TestDocumentedLegacyLimitations:  # MARK: TestDocumentedLegacyLimitations

    @pytest.mark.xfail(
        strict=True,
        reason='RSS-10 will accept scalar outputs with shape (batch,)')
    def test_scalar_output_without_artificial_axis(self):
        function, embedding, samples, domain = _binary_problem()
        tk.decompositions.tt_rss(
            function=lambda data: function(data).squeeze(1),
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            verbose=False)

    @pytest.mark.xfail(
        strict=True,
        reason='RSS-01/RSS-10 will support one embedding per input site')
    def test_heterogeneous_embeddings(self):
        function, embedding, samples, domain = _binary_problem()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=[embedding, embedding, embedding],
            sketch_samples=samples,
            domain=domain,
            rank=2,
            verbose=False)
        assert len(cores) == 3

    @pytest.mark.xfail(
        strict=True,
        reason='RSS-01/RSS-10 will split every tensor-output axis')
    def test_multiple_output_axes(self):
        _, embedding, samples, domain = _binary_problem()

        def function(data):
            scalar = 1 + data.sum(dim=1)
            return torch.stack((scalar, scalar + 1, scalar + 2, scalar + 3),
                               dim=1).reshape(-1, 2, 2)

        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            verbose=False)
        assert len(cores) == 5

    @pytest.mark.xfail(
        strict=True,
        reason='RSS-07/RSS-10 will make range projection optional')
    def test_random_projection_can_be_disabled(self):
        function, embedding, samples, domain = _binary_problem()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            random_projection=False,
            verbose=False)
        assert len(cores) == 3

    @pytest.mark.xfail(
        strict=True,
        reason='RSS-08/RSS-10 will provide structured verbosity levels')
    def test_verbosity_level_two_does_not_print_full_cores(self, capsys):
        function, embedding, samples, domain = _binary_problem()
        tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=1,
            verbose=2)
        assert 'tensor(' not in capsys.readouterr().out
