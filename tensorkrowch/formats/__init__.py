"""
This module contains:

    Format interfaces:
        * TensorFormat, TensorFormat1D

    Formats:
        * TT, TR, TTM, TRM
        * QTT, QTR, QTTM, QTRM

    Bonds and gauges:
        * BondFactors1D, VidalGauge
        * GaugeOrbit, TensorRingOrbit

    Layouts and coordinate maps:
        * QuantizedLayout, CoordinateMap
        * AffineCoordinateMap, FunctionalCoordinateMap, ExplicitGridMap

    Blocks:
        * BlockLayout, SplitBlock
        * split_block

    Diagnostics:
        * SampleError, RoundingInfo, MinimalCanonicalInfo

Construction:

    cores + optional diagonal factors ─> TT / TR / TTM / TRM
        cores ─> format.cores
        factors ─> format.bonds (BondFactors1D)
        canonicalize_vidal ─> format.bonds (VidalGauge, TT / TTM)

    Quantics:
        base + level + domain ─> QuantizedLayout + AffineCoordinateMap
        base + level + grid_coordinates ─> QuantizedLayout + ExplicitGridMap
        cores + layout + coordinate_map ─> QTT / QTR
        cores + input/output layouts and maps ─> QTTM / QTRM

Evaluation:

    domain coordinates ─> CoordinateMap.to_indices ─> grid indices
    grid indices ─> QuantizedLayout.encode_indices ─> digits in site order
    digits ─> evaluate_digits ─> Quantics values

Numerical workflows:

    block ─> BlockLayout; split_block ─> SplitBlock; unblock ─> restored sites
    rounding(return_info=True) ─> format + RoundingInfo
    error ─> SampleError
    canonicalize_minimal(return_info=True) ─> format + MinimalCanonicalInfo
    formats <─> models (explicit conversion methods)
    decompositions ─> results inheriting formats and adding fit diagnostics
"""

from tensorkrowch.formats.base import (RoundingInfo, SampleError, BlockLayout,
                                       SplitBlock, TensorFormat)
from tensorkrowch.formats.bonds import BondFactors1D, VidalGauge
from tensorkrowch.formats.formats1d import (TensorFormat1D,
                                            TT, TR, TTM, TRM,
                                            split_block)
from tensorkrowch.formats.orbits import (GaugeOrbit,
                                         TensorRingOrbit,
                                         MinimalCanonicalInfo)
from tensorkrowch.formats.quantics import QTT, QTR, QTTM, QTRM
from tensorkrowch.formats.quantization import (QuantizedLayout,
                                               CoordinateMap,
                                               AffineCoordinateMap,
                                               FunctionalCoordinateMap,
                                               ExplicitGridMap)


__all__ = [
    'TensorFormat',
    'TensorFormat1D',
    'SampleError',

    'TT',
    'TR',
    'TTM',
    'TRM',
    'QTT',
    'QTR',
    'QTTM',
    'QTRM',

    'BondFactors1D',
    'VidalGauge',
    'RoundingInfo',

    'BlockLayout',
    'SplitBlock',
    'split_block',

    'GaugeOrbit',
    'TensorRingOrbit',
    'MinimalCanonicalInfo',

    'QuantizedLayout',
    'CoordinateMap',
    'AffineCoordinateMap',
    'FunctionalCoordinateMap',
    'ExplicitGridMap',
]
