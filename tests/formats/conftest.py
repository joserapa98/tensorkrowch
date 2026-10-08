"""Small raw-tensor fixtures and independent dense oracles."""

import pytest
import torch
import tensorkrowch as tk


@pytest.fixture(params=[
    pytest.param((device, dtype), id=f'{device}-{str(dtype).split(".")[-1]}')
    for device in ('cpu', 'cuda', 'mps')
    for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
    if device != 'mps' or dtype in (torch.float32, torch.complex64)
])
def device_dtype(request):
    device, dtype = request.param
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is unavailable')
    return device, dtype


@pytest.fixture
def assert_close(device_dtype):
    _, dtype = device_dtype
    low_precision = dtype in (torch.float32, torch.complex64)
    rtol, atol = (5e-5, 5e-6) if low_precision else (1e-9, 1e-10)

    def check(actual, expected):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return check


@pytest.fixture
def make_format():
    def build(topology='tt', n_sites=3, n_batches=0, dtype=torch.float64,
              seed=31, in_dim=None, out_dim=None, quantized=False, device='cpu'):
        generator = torch.Generator().manual_seed(seed)
        cyclic = topology.startswith('tr')
        matrix = topology.endswith('m')
        batch = (2,) * n_batches
        in_dim = (tuple(in_dim) if in_dim is not None else
                  tuple(2 + site % 2 for site in range(n_sites)))
        out_dim = (tuple(out_dim) if out_dim is not None else
                   tuple(3 - site % 2 for site in range(n_sites)))
        ranks = [2 + site % 2 for site in range(n_sites + 1)]
        ranks[-1] = ranks[0] if cyclic else 1
        ranks[0] = ranks[0] if cyclic else 1
        cores = []
        for site in range(n_sites):
            shape = (*batch, ranks[site], in_dim[site], ranks[site + 1])
            if matrix:
                shape += (out_dim[site],)
            core = torch.randn(shape, dtype=dtype, generator=generator)
            if not cyclic:
                if site == 0:
                    core = core.squeeze(n_batches)
                if site == n_sites - 1:
                    core = core.squeeze(-2 if matrix else -1)
            cores.append(core)
        classes = {'tt': tk.formats.TT, 'tr': tk.formats.TR,
                   'ttm': tk.formats.TTM,
                   'trm': tk.formats.TRM}
        if quantized:
            layout = tk.formats.QuantizedLayout(n_sites, base=in_dim)
            coordinate_map = tk.formats.AffineCoordinateMap(
                [0., 1.], layout.grid_size)
            if matrix:
                output_layout = tk.formats.QuantizedLayout(n_sites, base=out_dim)
                output_map = tk.formats.AffineCoordinateMap(
                    [0., 1.], output_layout.grid_size)
                cls = tk.formats.QTRM if cyclic else tk.formats.QTTM
                return cls(cores, n_sites, n_sites,
                           in_layout=layout, out_layout=output_layout,
                           in_coordinate_map=coordinate_map,
                           out_coordinate_map=output_map, n_batches=n_batches).to(device)
            cls = tk.formats.QTR if cyclic else tk.formats.QTT
            return cls(cores, n_sites, layout=layout,
                       coordinate_map=coordinate_map, n_batches=n_batches).to(device)
        return classes[topology](cores, n_batches=n_batches).to(device)
    return build


@pytest.fixture
def dense_cores():
    """Factors small reference tensors using PyTorch, independently of decompositions."""
    def build(tensor, in_dim, out_dim=None, cyclic=False):
        physical = (tuple(in_dim) if out_dim is None else
                    tuple(i * o for i, o in zip(in_dim, out_dim)))
        state = tensor.reshape(*physical)
        standard = []
        left = 1
        for dimension in physical[:-1]:
            u, values, vh = torch.linalg.svd(
                state.reshape(left * dimension, -1), full_matrices=False)
            standard.append(u.reshape(left, dimension, -1))
            left = values.numel()
            state = values[:, None] * vh
        standard.append(state.reshape(left, physical[-1], 1))

        cores = []
        for site, core in enumerate(standard):
            if out_dim is not None:
                core = core.reshape(core.shape[0], in_dim[site],
                                    out_dim[site], core.shape[-1]).transpose(-1, -2)
            if not cyclic:
                if site == 0:
                    core = core.squeeze(0)
                if site == len(standard) - 1:
                    core = core.squeeze(-2 if out_dim is not None else -1)
            cores.append(core)
        return cores
    return build


@pytest.fixture
def matrix_view():
    """Reorders interleaved dense operator axes into a usual row/column matrix."""
    def build(format):
        dense = format.contract_dense()
        batch = format.n_batches
        axes = [*range(batch),
                *range(batch + 1, dense.ndim, 2),
                *range(batch, dense.ndim, 2)]
        return dense.permute(axes).reshape(
            *format.batch_shape, int(torch.tensor(format.out_dim).prod()),
            int(torch.tensor(format.in_dim).prod()))
    return build


@pytest.fixture
def exponential_format():
    """Constructs a rank-one Quantics exponential analytically, without fitting."""
    def build(cyclic=False, dtype=torch.float64, ordering='interleaved', device='cpu'):
        layout = tk.formats.QuantizedLayout(
            2, base=(2, 3), level=(2, 2), ordering=ordering)
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        domain = torch.tensor([[-1., 1.], [0., 2.]], dtype=real_dtype, device=device)
        coordinate_map = tk.formats.AffineCoordinateMap(domain, layout.grid_size)
        coefficients = torch.tensor([0.3, -0.2], dtype=dtype, device=device)
        if dtype.is_complex:
            coefficients = coefficients + 1j * torch.tensor(
                [0.2, 0.1], dtype=real_dtype, device=device)
        cores = []
        for coordinate, digit in layout.sites():
            stride = layout.base[coordinate] ** (layout.level[coordinate] - digit - 1)
            step = (domain[coordinate, 1] - domain[coordinate, 0]) / layout.grid_size[coordinate]
            values = torch.arange(layout.base[coordinate], dtype=real_dtype,
                                  device=device)
            core = (coefficients[coordinate] * step * stride * values).exp()
            cores.append(core.reshape(1, -1, 1))
        prefactor = (coefficients * domain[:, 0]).sum().exp()
        cores[0] = prefactor * cores[0]
        if not cyclic:
            cores[0] = cores[0].squeeze(0)
            cores[-1] = cores[-1].squeeze(-1)
        cls = tk.formats.QTR if cyclic else tk.formats.QTT
        format = cls(cores, 2, layout=layout, coordinate_map=coordinate_map)

        def function(coordinates, scale=1.):
            return scale * (coordinates * coefficients).sum(-1).exp()
        return format, function
    return build
