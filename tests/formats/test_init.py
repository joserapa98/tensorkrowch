"""Public format exports and decomposition-result parity."""

import pytest
import torch

import tensorkrowch as tk


# Public API parity


@pytest.mark.skipif(not hasattr(tk, 'decompositions'),
                    reason='Decompositions are excluded from the isolated formats run')
@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
def test_result_parity(make_format, topology, n_sites, device_dtype, assert_close):
    device, dtype = device_dtype
    format = make_format(topology, n_sites, dtype=dtype, device=device)
    result_class = {'tt': tk.decompositions.TTDecomposition,
                    'tr': tk.decompositions.TRDecomposition,
                    'ttm': tk.decompositions.TTMDecomposition,
                    'trm': tk.decompositions.TRMDecomposition}[topology]
    result = result_class(format.cores)
    dense = result.contract_dense()
    assert_close(format.contract_dense(), dense)
    assert_close(format.norm(), dense.norm())
    assert_close(format.inner(format), dense.abs().square().sum().to(dtype))
    assert_close(format.fidelity(format), dense.real.new_ones(()))
    inputs = torch.zeros(5, n_sites, dtype=torch.long, device=device)
    if format.out_dim is None:
        assert_close(format.evaluate(inputs), result.evaluate(inputs))
    else:
        outputs = torch.ones_like(inputs)
        assert_close(format.evaluate(inputs, outputs), result.evaluate(inputs, outputs))
        assert_close(format.apply(inputs).contract_dense(), result.apply(inputs).contract_dense())


def test_public_exports():
    assert len(tk.formats.__all__) == len(set(tk.formats.__all__))
    for name in tk.formats.__all__:
        assert getattr(tk.formats, name) is not None
    assert not hasattr(tk.formats, 'QTTTucker')
    assert not hasattr(tk.formats, 'QTRTucker')


@pytest.mark.parametrize('name', ['TT', 'TR', 'TTM', 'TRM',
                                  'QTT', 'QTR', 'QTTM', 'QTRM'])
def test_formats_use_raw_tensors(name):
    cls = getattr(tk.formats, name)
    assert issubclass(cls, tk.formats.TensorFormat1D)
    assert not issubclass(cls, tk.TensorNetwork)


@pytest.mark.parametrize('module_name', ['base', 'bonds', 'formats1d',
                                        'orbits', 'quantization', 'quantics'])
def test_documented_examples(module_name):
    import doctest
    import importlib
    import io

    module = importlib.import_module('tensorkrowch.formats.' + module_name)
    output = io.StringIO()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    examples = doctest.DocTestFinder().find(
        module, extraglobs={'tk': tk, 'torch': torch, 'nn': torch.nn})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        for example in examples:
            runner.run(example, out=output.write)
    result = runner.summarize()
    assert result.failed == 0, output.getvalue()
