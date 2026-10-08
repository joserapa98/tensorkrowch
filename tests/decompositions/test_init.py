"""Tests for init."""

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize('package', ['', 'als', 'ring', 'sketching', 'sources', 'svd'])
def test_decomposition_exports_and_source_test_structure(package):
    name = 'tensorkrowch.decompositions' + ('.' + package if package else '')
    module = importlib.import_module(name)
    assert len(module.__all__) == len(set(module.__all__))
    for export in module.__all__:
        assert hasattr(module, export)
        assert 'tucker' not in export.lower()
    source = Path(module.__file__).parent
    tests = Path(__file__).parent / package
    for file in source.glob('*.py'):
        expected = 'test_init.py' if file.stem == '__init__' else 'test_' + file.name
        assert (tests / expected).is_file(), str(file)
