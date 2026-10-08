"""
Quantization interfaces used by recursive sketching.

    * QuantizedLayout
    * CoordinateMap
    * AffineCoordinateMap
    * FunctionalCoordinateMap
    * ExplicitGridMap
    * QuantizedSourceAdapter

Layouts and maps belong to ``formats``. The adapter belongs to
``decompositions.sources`` and evaluates a source on digit configurations.
"""

from tensorkrowch.formats.quantization import (AffineCoordinateMap,
                                               CoordinateMap,
                                               ExplicitGridMap,
                                               FunctionalCoordinateMap,
                                               QuantizedLayout)

from tensorkrowch.decompositions.sources.quantization import (QuantizedSourceAdapter)


__all__ = [
    'QuantizedLayout',
    'CoordinateMap',
    'AffineCoordinateMap',
    'FunctionalCoordinateMap',
    'ExplicitGridMap',
    'QuantizedSourceAdapter',
]
