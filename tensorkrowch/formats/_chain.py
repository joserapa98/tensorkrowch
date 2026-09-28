"""Shared raw-tensor chain operations, independent of decomposition engines."""

from abc import abstractmethod
from copy import copy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.formats.base import TensorFormat, SampleError
from tensorkrowch.formats.bonds import BondFactors, VidalGauge


EvaluationData = Union[torch.Tensor, Sequence[torch.Tensor]]

_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64
)


class _CoreList(list):
    """Fixed-length core container with immediate validation on replacement."""

    def __init__(self, cores: Sequence[torch.Tensor], owner: 'TensorFormat1D') -> None:
        """Stores core references and their owning format."""
        super().__init__(cores)
        self._owner = owner

    def __setitem__(self, key, value):
        """Validates a controlled replacement before accepting it."""
        values = list(value) if isinstance(key, slice) else [value]

        if not all(isinstance(core, torch.Tensor) for core in values):
            raise TypeError('`cores` should contain torch.Tensor objects')
        if isinstance(key, slice) and (len(values) != len(self[key])):
            raise ValueError('Core slice replacement should preserve length')

        previous_cores = self[key]
        previous_owner = self._owner.__dict__.copy()
        super().__setitem__(key, values if isinstance(key, slice) else value)
        try:
            self._owner.validate()
        except (TypeError, ValueError):
            super().__setitem__(key, previous_cores)
            self._owner.__dict__.clear()
            self._owner.__dict__.update(previous_owner)
            raise

        self._owner._orth_center = None
        if isinstance(self._owner._bonds, VidalGauge):
            self._owner._bonds._valid = False

    def _structural_error(self, *args, **kwargs):
        """Rejects changes that bypass controlled structural replacement."""
        raise TypeError('Use the full cores setter to change the format structure')

    append = extend = insert = pop = remove = clear = _structural_error
    reverse = sort = __delitem__ = __iadd__ = __imul__ = _structural_error


class TensorFormat1D(TensorFormat):
    r"""Compact chain of raw cores, with cached dimensions and bond ranks.

    The constructor copies the container and shares tensor storage. Element
    and same-length slice replacement validate immediately and refresh metadata.
    Invalid replacements leave the format unchanged. Tensor value updates
    preserve dimensions. Shape changes through ``tensor.resize_`` are outside
    this contract.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores in the endpoint layout of the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    """

    _family = 'tensor'
    _topology = 'tensor'
    _cyclic = False

    def __init__(self, cores: Sequence[torch.Tensor], n_batches: int = 0) -> None:
        """Initializes the stored tensor references and validates construction."""
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        self._n_batches = n_batches
        self._orth_center = None
        self._bonds = None
        self.cores = cores

    @property
    def cores(self):
        """Mutable, fixed-length core sequence."""
        return self._cores

    @cores.setter
    def cores(self, cores: Sequence[torch.Tensor]):
        r"""Replaces the complete core container, validating before acceptance.

        Invalid structure restores previous cores and metadata. Valid manual
        replacement clears canonical state. For coupled rank changes, replace
        neighboring cores together.

        Parameters
        ----------
        cores : sequence of torch.Tensor
            Raw cores with the shapes required by the concrete format. The
            container is copied and tensor storage is shared; inputs retain
            autograd.
        """
        self._replace_cores(cores, self._bonds)
        if isinstance(self._bonds, VidalGauge):
            self._bonds._valid = False

    def _replace_cores(self, cores: Sequence[torch.Tensor], bonds):
        """Replaces cores and bonds together, restoring state on invalid input."""
        if isinstance(cores, torch.Tensor):
            raise TypeError('`cores` should be a sequence of torch.Tensor objects')

        cores = list(cores)
        previous = self.__dict__.copy()
        self._cores = _CoreList(cores, self)
        self._bonds = None if bonds is None else bonds._with_owner(self)

        try:
            self.validate()
        except (TypeError, ValueError):
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise

        self._orth_center = None

    @property
    def n_batches(self) -> int:
        """Number of leading structural batch axes."""
        return self._n_batches

    @abstractmethod
    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates cores and returns rank, batch, input and output dims."""

    def validate(self):
        r"""Validates cores and bonds and refreshes structural metadata.

        Called when constructing or replacing cores. Ordinary queries and
        contractions do not call it. Invalid structure raises TypeError or
        ValueError. Direct tensor shape changes remain outside the mutation
        contract.

        Returns
        -------
        TensorFormat1D
            The current format.
        """
        if not self._cores:
            raise ValueError('`cores` should contain at least one tensor')
        if not all(isinstance(core, torch.Tensor) for core in self._cores):
            raise TypeError('`cores` should contain torch.Tensor objects')

        device, dtype = self._cores[0].device, self._cores[0].dtype
        for core in self._cores:
            if any(dim < 1 for dim in core.shape):
                raise ValueError('Core dimensions should be positive')
            if core.device != device:
                raise ValueError('All cores should be on the same device')
            if core.dtype != dtype:
                raise ValueError('All cores should have the same dtype')

        rank, batch_shape, in_dim, out_dim = self._validate_cores()

        self._rank = tuple(rank)
        self._batch_shape = batch_shape

        self._in_dim = in_dim
        self._out_dim = out_dim
        self._same_in_dim = all(dim == in_dim[0] for dim in in_dim)
        self._same_out_dim = out_dim is None or all(
            dim == out_dim[0] for dim in out_dim)

        self.validate_bonds()
        return self

    def validate_bonds(self):
        r"""Checks stored diagonal factors against current core dimensions.

        Checks factor count, ranks, batch shapes, device and compatible dtype.
        Called on controlled bond replacement; it does not certify Vidal
        spectra.

        Returns
        -------
        TensorFormat1D
            The current format.
        """
        if self._bonds is not None:
            self._bonds.validate(self._raw_standard_cores(), self._cyclic)
        return self

    def _map_tensors(self, function):
        """Maps stored tensors while preserving concrete container semantics."""
        result = copy(self)
        result._cores = _CoreList([function(core) for core in self._cores], result)
        if self._bonds is not None:
            result._bonds = self._bonds._map_tensors(function)._with_owner(result)
        return result

    def _same_aux_tensors(self, other):
        """Checks whether auxiliary tensor references are unchanged."""
        return True

    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           copy: bool = False):
        r"""Returns a device/dtype conversion, preserving the concrete format.

        PyTorch device errors propagate without a CPU fallback. Autograd is
        retained.

        Parameters
        ----------
        device : str or torch.device, optional
            Target device. None preserves the current device.
        dtype : torch.dtype, optional
            Target dtype. None preserves the current dtype. Coordinate grids and
            Schmidt spectra remain real when cores are complex.
        copy : bool
            If True, copies tensors even when device and dtype are unchanged. If
            False, an unchanged conversion may return self.

        Returns
        -------
        TensorFormat
            Converted format; self when no conversion is needed and copy is
            False.

        Examples
        --------
        >>> format = tk.formats.TT([torch.ones(2)])
        >>> format.to() is format
        True
        >>> format.to(dtype=torch.float64).dtype == torch.float64
        True
        """
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type')
        if not isinstance(copy, bool):
            raise TypeError('`copy` should be bool type')

        result = self._map_tensors(lambda tensor: tensor.to(
            device=device, dtype=dtype, copy=copy))

        same_cores = all(
            new is old for new, old in zip(result._cores, self._cores))
        same_bonds = self._bonds is None or all(
            new is old for new, old in zip(result._bonds.values, self._bonds.values))

        if not copy and same_cores and same_bonds and self._same_aux_tensors(result):
            return self

        result.validate()
        return result

    def clone(self):
        r"""Clones the structural tensors, preserving autograd.

        Returns
        -------
        TensorFormat
            Independent tensor storage with the same represented tensor.
        """
        return self._map_tensors(lambda tensor: tensor.clone())

    def detach(self):
        r"""Returns a detached format sharing tensor storage.

        Returns
        -------
        TensorFormat
            Separate containers with detached tensor references. Value edits to
            shared storage affect both formats.
        """
        return self._map_tensors(lambda tensor: tensor.detach())

    def detach_(self):
        r"""Detaches structural tensors in-place by replacing references.

        Returns
        -------
        TensorFormat
            The current format. Tensor shapes and canonical metadata are
            preserved.
        """
        self._cores = _CoreList([core.detach() for core in self._cores], self)
        if self._bonds is not None:
            self._bonds = self._bonds._map_tensors(
                lambda tensor: tensor.detach())._with_owner(self)
        return self

    @abstractmethod
    def _raw_standard_cores(self) -> List[torch.Tensor]:
        """Returns (*batch, left, physical, right) cores without bond factors."""

    def _standard_cores(self):
        """Returns standard fused cores including stored diagonal factors."""
        cores = self._raw_standard_cores()
        if self._bonds is not None:
            cores = list(cores)
            for site, value in enumerate(self._bonds.values):
                if value is not None:
                    cores[site] = cores[site] * value[..., None, None, :]
        return cores

    @property
    def bonds(self):
        """Optional diagonal factors, with one entry per virtual bond."""
        return self._bonds

    @bonds.setter
    def bonds(self, value):
        r"""Assigns compatible diagonal factors to the format.

        Factor count, ranks, batches and runtime are checked before acceptance.

        Parameters
        ----------
        value : BondFactors or None
            Factors to attach, or None to remove them. A separate container is
            owned by this format while tensor references are shared.
        """
        if value is not None:
            if not isinstance(value, BondFactors):
                raise TypeError('`bonds` should be BondFactors type or None')
            value.validate(self._raw_standard_cores(), self._cyclic)
        self._bonds = None if value is None else value._with_owner(self)
        self._orth_center = None

    def materialize_bonds(self, orth_center: Optional[int] = None):
        r"""Absorbs stored factors into the cores in-place.

        A valid Vidal gauge redistributes its spectra towards the selected
        center, accounting for previously absorbed powers, including inverse
        Vidal. Generic or invalidated factors are absorbed once into the
        adjacent core. A mixed canonical form is obtained only when the initial
        Vidal gauge is valid.

        Parameters
        ----------
        orth_center : int, optional
            Orthogonality center in [0, n_sites - 1]. None selects the last
            site.

        Returns
        -------
        TensorFormat1D
            The current format, with bonds set to None.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> _ = format.canonicalize_vidal(mode='explicit')
        >>> _ = format.materialize_bonds(orth_center=0)
        >>> format.bonds is None
        True
        >>> torch.allclose(format.contract_dense(), torch.eye(2))
        True
        """
        from tensorkrowch.formats.canonical import materialize_bonds

        return materialize_bonds(self, orth_center)

    def _set_standard_cores(self, cores: Sequence[torch.Tensor], bonds=None):
        """Restores compact core layouts and publishes cores and bonds together."""
        from tensorkrowch.formats.operations import _restore_cores

        cores = _restore_cores(cores, self._in_dim, self._out_dim,
                               self._n_batches, self._cyclic)
        self._replace_cores(cores, bonds)

    def canonicalize(self,
                     orth_center: Optional[int] = None,
                     renormalize: bool = False):
        r"""Performs QR/RQ sweeps in-place without truncation.

        Open chains become left-isometric before the center and right-isometric
        after it. For rings, this is a local gauge relative to the stored cut,
        without a global Schmidt interpretation. Stored factors are materialized
        before sweeping.

        Parameters
        ----------
        orth_center : int, optional
            Orthogonality center in [0, n_sites - 1]. None selects the last
            site.
        renormalize : bool
            Rescales intermediate factors to reduce numerical overflow or
            underflow and restores the accumulated scale in the final cores. The
            represented tensor retains its global scale.

        Returns
        -------
        TensorFormat1D
            The current format, with its selected orthogonality center recorded.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> _ = format.canonicalize(orth_center=0, renormalize=True)
        >>> torch.allclose(format.contract_dense(), torch.eye(2))
        True
        """
        from tensorkrowch.formats.canonical import canonicalize

        return canonicalize(self, orth_center, renormalize)

    def canonicalize_vidal(self,
                           mode: str = 'implicit',
                           inverse_positions: Optional[Sequence[int]] = None,
                           remaining_mode: str = 'implicit',
                           inverse_cutoff: float = 0.0):
        r"""Builds or redistributes an open-chain Vidal gauge in-place.

        Only TT/TTM admit this global Schmidt representation. Invalidated gauges
        are recomputed; valid gauges can be redistributed without another SVD.
        No truncation is performed to manufacture an inverse.

        Parameters
        ----------
        mode : {"implicit", "explicit", "inverse"}
            Absorbs spectrum powers (0.5, 0.5), (0, 0), or (1, 1) into
            neighboring cores, respectively. The remaining diagonal factor has
            power 1 minus their sum.
        inverse_positions : sequence of int, optional
            Distinct virtual bonds to use in inverse form. Cannot be combined
            with mode="inverse". When supplied, remaining_mode controls every
            other bond.
        remaining_mode : {"implicit", "explicit"}
            Distribution for bonds not selected by inverse_positions.
        inverse_cutoff : float
            Finite non-negative threshold. Every spectrum value used in an
            inverse should be strictly greater than this value; values are not
            truncated to create an inverse.

        Returns
        -------
        TensorFormat1D
            The current format with a valid VidalGauge.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> _ = format.canonicalize_vidal(inverse_positions=[0])
        >>> format.bonds.powers
        [(1, 1)]
        >>> _ = format.canonicalize_vidal(mode='implicit')
        >>> format.bonds.powers
        [(0.5, 0.5)]
        >>> torch.allclose(format.contract_dense(), torch.eye(2))
        True
        """
        from tensorkrowch.formats.canonical import canonicalize_vidal

        return canonicalize_vidal(self, mode, inverse_positions,
                                  remaining_mode, inverse_cutoff)

    def rounding(self,
                 rank: Optional[int] = None,
                 cutoff: Optional[float] = None,
                 atol: Optional[float] = None,
                 rtol: Optional[float] = None,
                 cum_percentage: Optional[float] = None,
                 renormalize: bool = False,
                 *,
                 rel_error: Optional[float] = None,
                 return_info: bool = False):
        r"""Compresses bond ranks in-place with one QR/SVD execution.

        Open chains use left QR followed by sitewise right SVD. Rings use
        Algorithm 4 of Mickelin and Karaman, `On Algorithms for and Computing
        with the Tensor Ring Decomposition <https://arxiv.org/pdf/1807.02513>`_,
        including closure reduction. Ring ranks need not become minimal, especially after
        block-diagonal sums or products. Multiple criteria use the most
        restrictive retained rank.

        Parameters
        ----------
        rank : int, optional
            Maximum number of singular values to keep.
        cutoff : float, optional
            Minimum singular value to keep. It must be finite and non-negative.
            Singular values <= cutoff are removed.
        atol : float, optional
            Absolute tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded
            while the accumulated sum of squares is <= atol. It must be finite
            and non-negative.
        rtol : float, optional
            Relative tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded
            while the tail sum of squares divided by the total sum of squares is
            <= rtol. It must be finite and in [0, 1].
        cum_percentage : float, optional
            Minimum fraction of squared singular-value mass to keep. Equivalent
            to setting rtol = 1 - cum_percentage. It must be finite and in [0,
            1].
        renormalize : bool
            Rescales intermediate factors to reduce numerical overflow or
            underflow and restores the accumulated scale in the final cores. The
            represented tensor retains its global scale.
        rel_error : float, optional
            Finite non-negative relative Frobenius error budget. It is
            distributed over local cuts. Other truncation constraints can exceed
            it and then emit a warning.
        return_info : bool
            If True, returns the format together with the operation-specific
            information record.

        Returns
        -------
        TensorFormat1D or tuple[TensorFormat1D, RoundingInfo]
            The current format, optionally with discarded energies and an
            absolute error bound. The bound is not a measured reconstruction
            error.

        Examples
        --------
        >>> format = tk.formats.TT([
        ...     torch.diag(torch.tensor([4., 1.])), torch.eye(2)])
        >>> _, info = format.rounding(rank=1, return_info=True)
        >>> info.rank
        (1,)
        >>> torch.allclose(format.contract_dense(), torch.diag(torch.tensor([4., 0.])))
        True
        """
        from tensorkrowch.formats.rounding import rounding

        return rounding(self, rank, cutoff, atol, rtol, cum_percentage,
                        renormalize, rel_error, return_info)

    def canonicalize_minimal(self, max_iter: int = 200, lr: float = 0.05,
                             tol: float = 1e-8, *, return_info: bool = False):
        r"""Uses implicit Vidal for trains or experimental gauge balancing for
        rings.

        Ring optimization minimizes half the sum of squared core norms using
        Hermitian exponential gauges and retains the best finite iterate. It may
        stop without convergence and does not certify a global minimum. Batched
        rings share one gauge per bond. Input core gradients are not accumulated
        by gauge optimization. The objective is inspired by `The minimal
        canonical form of a tensor network <https://arxiv.org/pdf/2209.14358>`_.

        Parameters
        ----------
        max_iter : int
            Positive maximum number of gauge optimization iterations for a ring.
        lr : float
            Finite positive learning rate of the Adam gauge optimizer.
        tol : float
            Finite positive stopping tolerance for the maximum absolute gauge-
            parameter gradient.
        return_info : bool
            If True, returns the format together with the operation-specific
            information record.

        Returns
        -------
        TensorFormat1D or tuple[TensorFormat1D, MinimalCanonicalInfo]
            The current format, optionally with iteration count, convergence
            status and final ring Gram imbalance. Trains report zero iterations
            and no imbalance.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> _, info = format.canonicalize_minimal(return_info=True)
        >>> (info.iterations, info.converged)
        (0, True)
        >>> torch.allclose(format.contract_dense(), torch.eye(2))
        True
        """
        from tensorkrowch.formats.orbits import canonicalize_minimal

        return canonicalize_minimal(self, max_iter, lr, tol, return_info)

    def block(self, groups: Sequence[int], return_info: bool = False):
        r"""Returns a format with consecutive sites contracted into blocks.

        Does not modify self. Matrix input and output axes are grouped
        separately within a block. Internal factors are contracted and external
        factors are retained.

        Parameters
        ----------
        groups : sequence of int
            Positive numbers of consecutive sites in each block. Their sum
            should equal n_sites.
        return_info : bool
            If True, returns the format together with the operation-specific
            information record.

        Returns
        -------
        TensorFormat1D or tuple[TensorFormat1D, BlockLayout]
            Blocked format, optionally with original dimensions. Layout is also
            stored on the result for unblock().

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> blocked, layout = format.block([2], return_info=True)
        >>> blocked.in_dim
        (4,)
        >>> restored = blocked.unblock(layout)
        >>> torch.allclose(restored.contract_dense(), format.contract_dense())
        True
        """
        from tensorkrowch.formats.blocking import block

        return block(self, groups, return_info)

    def unblock(self, info=None, **kwargs):
        r"""Restores individual sites from a blocked format.

        Parameters
        ----------
        info : BlockLayout, optional
            Original dimensions and group sizes. None uses the layout stored by
            block().
        **kwargs : keyword arguments
            Local splitting options passed to :func:`~tensorkrowch.formats.split_block`: rank, cutoff, atol,
            rtol, cum_percentage, mode and renormalize. Without truncation
            criteria the reconstruction is exact up to numerical precision.

        Returns
        -------
        TensorFormat1D
            Separate unblocked format. The source is unchanged.
        """
        from tensorkrowch.formats.blocking import unblock

        return unblock(self, info, **kwargs)

    def contract_block(self, first, last):
        r"""Contracts a contiguous region with both external ranks left open.

        Parameters
        ----------
        first : int
            First site of the region, included.
        last : int
            Last site of the region, included. Should be at least first.

        Returns
        -------
        torch.Tensor
            Tensor with shape (*core_batch, left, *physical, right). Matrix
            physical axes are interleaved. Internal factors are included and
            external factors excluded.
        """
        from tensorkrowch.formats.blocking import contract_block

        return contract_block(self, first, last)

    def split_block(self, block: torch.Tensor, first, last, **kwargs):
        r"""Splits a local tensor while preserving the selected external interfaces.

        Parameters
        ----------
        block : torch.Tensor
            Local tensor with shape (*core_batch, left, *physical, right).
            Matrix physical axes are interleaved in site order.
        first : int
            First site of the region, included.
        last : int
            Last site of the region, included. Should be at least first.
        **kwargs : keyword arguments
            Options passed to :func:`~tensorkrowch.formats.split_block`: rank, cutoff, atol, rtol,
            cum_percentage, mode and renormalize.

        Returns
        -------
        SplitBlock
            Standard fused cores and local factors. The current format is
            unchanged. Spectra are local singular values, not certified global
            Schmidt values.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> local = format.split_block(format.contract_block(0, 1), 0, 1)
        >>> _ = format.replace_block(0, 1, local)
        >>> torch.allclose(format.contract_dense(), torch.eye(2))
        True
        """
        from tensorkrowch.formats.blocking import split_block
        if any(isinstance(site, bool) or not isinstance(site, int)
               for site in (first, last)):
            raise TypeError('Block endpoints should be integers')
        if not 0 <= first <= last < self.n_sites:
            raise ValueError('Block endpoints should select an ordered region')
        outputs = None if self._out_dim is None else self._out_dim[first:last + 1]
        return split_block(block, self._in_dim[first:last + 1], outputs,
                           self._n_batches, **kwargs)

    def replace_block(self, first, last, replacement):
        r"""Replaces a contiguous region in-place, preserving external ranks.

        Parameters
        ----------
        first : int
            First site of the region, included.
        last : int
            Last site of the region, included. Should be at least first.
        replacement : SplitBlock or sequence of torch.Tensor
            Standard fused cores for the selected sites, optionally with
            internal factors. Physical dimensions and both external ranks should
            match the region.

        Returns
        -------
        TensorFormat1D
            The current format. Invalid replacement leaves it unchanged.
        """
        from tensorkrowch.formats.blocking import replace_block

        return replace_block(self, first, last, replacement)

    def absorb_bond(self, bond, side: str = 'left'):
        r"""Absorbs one diagonal factor into a neighboring core in-place.

        Parameters
        ----------
        bond : int
            Index of the right virtual bond of a core. The last bond closes a
            cyclic format.
        side : {"left", "right"}
            Neighbor receiving the factor. A valid Vidal gauge redistributes
            existing absorption powers; an ordinary or invalidated gauge absorbs
            its stored diagonal once.

        Returns
        -------
        TensorFormat1D
            The current format, with the selected explicit factor removed.
        """
        from tensorkrowch.formats.blocking import absorb_bond

        return absorb_bond(self, bond, side)

    def redistribute_bond(self, bond: int, mode: str = 'implicit',
                          inverse_cutoff: float = 0.0):
        r"""Redistributes one valid Vidal spectrum without another SVD.

        Requires a valid stored Vidal gauge. Manual edits invalidate it;
        recompute canonicalize_vidal() before redistributing spectra.

        Parameters
        ----------
        bond : int
            Index of the right virtual bond of a core. The last bond closes a
            cyclic format.
        mode : {"implicit", "explicit", "inverse", "left", "right"}
            New powers absorbed into the left/right neighbors: (0.5, 0.5), (0,
            0), (1, 1), (1, 0) or (0, 1), respectively.
        inverse_cutoff : float
            Finite non-negative threshold. Every spectrum value used in an
            inverse should be strictly greater than this value; values are not
            truncated to create an inverse.

        Returns
        -------
        TensorFormat1D
            The current format. Other bonds retain their distribution.
        """
        from tensorkrowch.formats.blocking import redistribute_bond

        return redistribute_bond(self, bond, mode, inverse_cutoff)

    def add(self, other, method='stacked'):
        r"""Returns the exact sum of compatible formats.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with compatible local input/output dimensions,
            structural batches and device.
        method : {"stacked", "block_diagonal"}
            Cyclic sum construction. Stacked endpoints favor subsequent
            compression; block_diagonal uses the usual construction at every
            site. For open chains both choices use the ordinary endpoint
            construction.

        Returns
        -------
        TensorFormat1D
            New format without implicit truncation. Either cyclic operand
            produces a cyclic result. Inputs are unchanged.
        """
        from tensorkrowch.formats.operations import add

        return add(self, other, method=method)

    def sub(self, other, method='stacked'):
        r"""Returns the exact difference of compatible formats.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with compatible local input/output dimensions,
            structural batches and device.
        method : {"stacked", "block_diagonal"}
            Cyclic sum construction. Stacked endpoints favor subsequent
            compression; block_diagonal uses the usual construction at every
            site. For open chains both choices use the ordinary endpoint
            construction.

        Returns
        -------
        TensorFormat1D
            New format without implicit truncation. Either cyclic operand
            produces a cyclic result. Inputs are unchanged.
        """
        from tensorkrowch.formats.operations import add

        return add(self, other, method=method, coefficient=-1)

    def hadamard(self, other):
        r"""Returns an exact element-wise product.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with compatible local input/output dimensions,
            structural batches and device.

        Returns
        -------
        TensorFormat1D
            New format with product bond ranks, before any explicit rounding. A
            cyclic operand produces a cyclic result.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> torch.equal((format * format).contract_dense(), torch.eye(2))
        True
        """
        from tensorkrowch.formats.operations import hadamard

        return hadamard(self, other)

    def __add__(self, other):
        """Returns the exact sum with another format."""
        return self.add(other)

    def __sub__(self, other):
        """Returns the exact difference with another format."""
        return self.sub(other)

    def __neg__(self):
        """Returns the format with its represented tensor negated."""
        return self * -1

    def __mul__(self, other):
        """Returns a Hadamard product or scalar-scaled format."""
        from tensorkrowch.formats.operations import scale

        return self.hadamard(other) if isinstance(
            other, TensorFormat1D) else scale(self, other)

    def __rmul__(self, other):
        """Returns a scalar-scaled format or Hadamard product."""
        return self * other

    def __matmul__(self, other):
        """Returns the exact matrix application or matrix product."""
        from tensorkrowch.formats.operations import apply

        return apply(self, other)

    def apply(self, other):
        r"""Applies a matrix format from the right, equivalent to self @ other.

        Parameters
        ----------
        other : TTM or TRM
            Matrix whose output dimensions match the vector input dimensions.
            This contracts the matrix output axes, unlike matrix @ vector, which
            contracts matrix input axes.

        Returns
        -------
        TT or TR
            Exact vector result; cyclic when either operand is cyclic. Quantics
            subclasses retain compatible coordinate semantics.
        """
        return self @ other

    def conj(self):
        r"""Returns the conjugate cores and bond factors.

        Returns
        -------
        TensorFormat1D
            Separate format with conjugated tensor references; storage may be
            shared.
        """
        return self._map_tensors(lambda tensor: tensor.conj())

    def contract_dense(self) -> torch.Tensor:
        r"""Contracts the full represented tensor explicitly.

        Returns
        -------
        torch.Tensor
            Dense tensor with shape (*core_batch, *in_dim) for vectors and
            interleaved (in_0, out_0, ...) axes for matrices. Intended for small
            tensors, not large-grid evaluation.
        """
        cores = self._standard_cores()
        closing = cores[0].shape[-3]
        result = cores[0]
        dimensions = [result.shape[-2]]
        for core in cores[1:]:
            result = result.reshape(*self._batch_shape, closing, -1, result.shape[-1])
            result = torch.einsum('...apr,...rqb->...apqb', result, core)
            dimensions.append(core.shape[-2])
        result = result.reshape(*self._batch_shape, closing, *
                                dimensions, cores[-1].shape[-1])
        result = result.diagonal(dim1=self._n_batches, dim2=-1).sum(-1)
        if self._out_dim is not None:
            dimensions = [dim for pair in zip(
                self._in_dim, self._out_dim) for dim in pair]
            result = result.reshape(*self._batch_shape, *dimensions)
        return result

    def inner(self, other) -> torch.Tensor:
        r"""Contracts the conjugate of self with another format.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with matching input/output dimensions and structural
            batch shape, on the same device. Dtypes may be promoted; boundary
            topologies may differ.

        Returns
        -------
        torch.Tensor
            Overlap tensor with shape core_batch, preserving complex phase.
        """
        phase, log_magnitude = self._log_overlap(other)
        return phase * log_magnitude.exp()

    @property
    def device(self) -> torch.device:
        """Device shared by all cores."""
        return self.cores[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Data type shared by all cores."""
        return self.cores[0].dtype

    @property
    def rank(self) -> List[int]:
        """Bond ranks inferred from the cores."""
        return list(self._rank)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Batch dimensions shared by the cores."""
        return self._batch_shape

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension associated with every site."""
        return self._in_dim

    @property
    def n_sites(self) -> int:
        """Number of sites in the stored core network.

        Each stored core represents one site. A hierarchical quantized result
        counts its upper-network sites; :meth:`flatten` constructs a separate
        result whose sites include the digit factors.
        """
        return len(self.cores)

    @property
    def out_dim(self) -> Optional[Tuple[int, ...]]:
        """Output dimension per site, when the decomposition has one."""
        return self._out_dim

    @property
    def topology(self) -> str:
        """Topology identifier used in serialized result information."""
        return self._topology

    def _normalize_data(
            self,
            data: EvaluationData,
            dimensions: Sequence[int],
            same_dim: bool,
            n_batches: int
    ) -> Tuple[List[torch.Tensor], bool, Tuple[int, ...]]:
        """Normalizes discrete indices or embedded vectors by site."""
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        n_sites = len(dimensions)
        if isinstance(data, torch.Tensor):
            if data.ndim == (n_batches + 1):
                if data.shape[-1] != n_sites:
                    raise ValueError(
                        'The last dimension of discrete inputs should equal '
                        'the number of sites')
                if data.dtype not in _INTEGER_DTYPES:
                    raise TypeError(
                        'Discrete inputs should have an integer dtype')
                site_data = list(data.to(device=self.device).unbind(-1))
                discrete = True
            elif data.ndim == (n_batches + 2):
                if not same_dim:
                    raise ValueError(
                        'Embedded inputs should be provided as a sequence when '
                        'site dimensions differ')
                if data.shape[-2:] != (n_sites, dimensions[0]):
                    raise ValueError(
                        'Embedded inputs should end in `(n_sites, site_dim)`')
                site_data = list(
                    data.to(device=self.device, dtype=self.dtype).unbind(-2))
                discrete = False
            else:
                raise ValueError(
                    '`data` has an incompatible number of batch dimensions')
        else:
            try:
                site_data = list(data)
            except TypeError as exc:
                raise TypeError(
                    '`data` should be a tensor or a sequence of tensors') \
                    from exc
            if len(site_data) != n_sites:
                raise ValueError(
                    '`data` should contain one tensor per site')
            if not all(isinstance(item, torch.Tensor)
                       for item in site_data):
                raise TypeError('Every site input should be a torch.Tensor')

            if all(item.ndim == n_batches for item in site_data):
                if not all(item.dtype in _INTEGER_DTYPES
                           for item in site_data):
                    raise TypeError(
                        'Discrete inputs should have an integer dtype')
                discrete = True
            elif all(item.ndim == (n_batches + 1)
                     for item in site_data):
                for item, site_dim in zip(site_data, dimensions):
                    if item.shape[-1] != site_dim:
                        raise ValueError(
                            'The last dimension of each embedded input should '
                            'match its site dimension')
                discrete = False
            else:
                raise ValueError(
                    'Site inputs should all be discrete indices or embedded '
                    'vectors')

            target_dtype = None if discrete else self.dtype
            site_data = [
                item.to(device=self.device, dtype=target_dtype)
                for item in site_data
            ]

        batch_shape = tuple(site_data[0].shape[:n_batches])
        if any(tuple(item.shape[:n_batches]) != batch_shape
               for item in site_data[1:]):
            raise ValueError('All site inputs should have the same batch shape')

        if discrete:
            for indices, site_dim in zip(site_data, dimensions):
                if torch.any(indices < 0) or torch.any(indices >= site_dim):
                    raise ValueError(
                        'Discrete indices should lie in each site dimension')

        return site_data, discrete, batch_shape

    @staticmethod
    def _contract_open_chain(
            matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts an open chain of matrices along adjacent ranks."""
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result

    def _check_overlap_compatibility(
            self, other: 'TensorFormat1D') -> None:
        """Validates topology, shapes and runtime for an overlap."""
        if not isinstance(other, TensorFormat1D):
            raise TypeError('`other` should be TensorFormat1D type')
        if self._family != other._family:
            raise ValueError('The decomposition families are incompatible')
        if len(self.cores) != len(other.cores):
            raise ValueError('Decompositions should have the same number of sites')
        if self._in_dim != other._in_dim:
            raise ValueError('Decompositions should have matching input dimensions')
        if self._out_dim != other._out_dim:
            raise ValueError('Decompositions should have matching output dimensions')
        if self._batch_shape != other._batch_shape:
            raise ValueError('Decompositions should have matching batch shapes')
        if self.device != other.device:
            raise ValueError('Decompositions should be on the same device')

    def _log_overlap(
            self, other: 'TensorFormat1D') -> Tuple[torch.Tensor,
                                                    torch.Tensor]:
        """Returns overlap phase and log-magnitude using scaled environments."""
        self._check_overlap_compatibility(other)
        dtype = torch.promote_types(self.dtype, other.dtype)
        self_cores = self._standard_cores()
        other_cores = self_cores if other is self else other._standard_cores()

        self_eye = torch.eye(
            self_cores[0].shape[-3], device=self.device, dtype=dtype)
        other_eye = torch.eye(
            other_cores[0].shape[-3], device=self.device, dtype=dtype)
        environment = torch.einsum('ai,bj->abij', self_eye, other_eye)
        if self.n_batches:
            environment = environment.reshape(
                *((1,) * self.n_batches), *environment.shape)
            environment = environment.expand(
                *self._batch_shape, *environment.shape[self.n_batches:])

        real_dtype = torch.empty((), dtype=dtype).real.dtype
        log_scale = torch.zeros(
            self._batch_shape, device=self.device, dtype=real_dtype)

        for self_core, other_core in zip(self_cores, other_cores):
            self_core = self_core.to(dtype=dtype)
            self_scale = self_core.abs().amax(dim=(-3, -2, -1))
            self_scale = torch.where(self_scale > 0, self_scale,
                                     torch.ones_like(self_scale))
            self_core = self_core / self_scale[..., None, None, None]
            if other is self:
                other_scale, other_core = self_scale, self_core
            else:
                other_core = other_core.to(dtype=dtype)
                other_scale = other_core.abs().amax(dim=(-3, -2, -1))
                other_scale = torch.where(
                    other_scale > 0, other_scale, torch.ones_like(other_scale))
                other_core = other_core / other_scale[..., None, None, None]
            log_scale = log_scale + self_scale.log() + other_scale.log()
            environment = torch.einsum(
                '...xyab,...apr->...xybpr',
                environment,
                self_core.conj())
            environment = torch.einsum(
                '...xybpr,...bps->...xyrs',
                environment,
                other_core)

            scale = torch.linalg.vector_norm(
                environment, dim=(-4, -3, -2, -1))
            nonzero = scale > 0
            safe_scale = torch.where(nonzero, scale, torch.ones_like(scale))
            environment = environment / safe_scale[..., None, None, None, None]
            log_scale = log_scale + torch.where(
                nonzero, safe_scale.log(), torch.zeros_like(safe_scale))

        overlap = torch.einsum('...ijij->...', environment)
        magnitude = overlap.abs()
        nonzero = magnitude > 0
        safe_magnitude = torch.where(
            nonzero, magnitude, torch.ones_like(magnitude))
        phase = torch.where(nonzero,
                            overlap / safe_magnitude,
                            torch.zeros_like(overlap))
        log_magnitude = torch.where(
            nonzero,
            safe_magnitude.log() + log_scale,
            torch.full_like(log_scale, -torch.inf))
        return phase, log_magnitude

    def norm(self) -> torch.Tensor:
        r"""Returns the Frobenius norm using scaled double-layer contractions.

        Returns
        -------
        torch.Tensor
            Real tensor with shape core_batch, or a scalar for an unbatched
            format.
        """
        _, log_squared_norm = self._log_overlap(self)
        return torch.exp(log_squared_norm / 2)

    def normalized_overlap(
            self, other: 'TensorFormat1D') -> torch.Tensor:
        r"""Returns <self, other> / (||self|| ||other||), preserving phase.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with matching input/output dimensions and structural
            batch shape, on the same device. Dtypes may be promoted; boundary
            topologies may differ.

        Returns
        -------
        torch.Tensor
            Normalized overlap with shape core_batch. Zero-norm operands raise
            ValueError.
        """
        phase, log_overlap = self._log_overlap(other)
        _, log_self = self._log_overlap(self)
        _, log_other = other._log_overlap(other)

        if torch.any(torch.isneginf(log_self)) or \
                torch.any(torch.isneginf(log_other)):
            raise ValueError(
                'Normalized overlap is undefined for a zero-norm decomposition')

        log_denominator = (log_self + log_other) / 2
        return phase * torch.exp(log_overlap - log_denominator)

    def fidelity(self, other: 'TensorFormat1D') -> torch.Tensor:
        r"""Returns the squared magnitude of the normalized overlap.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with matching input/output dimensions and structural
            batch shape, on the same device. Dtypes may be promoted; boundary
            topologies may differ.

        Returns
        -------
        torch.Tensor
            Real tensor with shape core_batch. Zero-norm operands raise
            ValueError.
        """
        return self.normalized_overlap(other).abs().square()


class _VectorFormat1D(TensorFormat1D):
    """Shared raw-tensor vector operations."""

    _family = 'state'

    def __call__(self,
                 data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        r"""Calls :meth:`evaluate` with the same input conventions.

        Parameters
        ----------
        data : torch.Tensor or sequence of torch.Tensor
            Integer configurations of shape (*data_batch, n_sites), or embedded
            vectors of shape (*data_batch, n_sites, in_dim) for uniform
            dimensions. For heterogeneous dimensions, pass one tensor per site.
            Each site tensor has shape (*data_batch,) for indices or
            (*data_batch, in_dim[site]) for embeddings.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch). Structural batches and
            data batches remain independent.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> format.evaluate(torch.tensor([[0, 0], [0, 1]])).tolist()
        [1.0, 0.0]
        >>> embeddings = torch.ones(1, 2, 2)
        >>> format.evaluate(embeddings).tolist()
        [2.0]
        """
        return self.evaluate(data, n_batches=n_batches)

    def _local_matrices(
            self,
            site_data: Sequence[torch.Tensor],
            discrete: bool,
            data_batch_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Contracts TT/TR cores with one input at each site."""
        core_batch_size = int(torch.Size(self._batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        matrices = []

        for core, site_value in zip(self._standard_cores(), site_data):
            left_rank, site_in_dim, right_rank = core.shape[-3:]
            core = core.reshape(
                core_batch_size, left_rank, site_in_dim, right_rank)

            if discrete:
                indices = site_value.reshape(data_batch_size).to(torch.long)
                matrix = core[:, :, indices, :].permute(0, 2, 1, 3)
            else:
                vectors = site_value.reshape(data_batch_size, site_in_dim)
                matrix = torch.einsum('dp,clpr->cdlr', vectors, core)

            matrices.append(matrix.reshape(
                *self._batch_shape,
                *data_batch_shape,
                left_rank,
                right_rank))

        return matrices

    def evaluate(self,
                 data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        r"""Evaluates integer configurations or local embedded vectors.

        Parameters
        ----------
        data : torch.Tensor or sequence of torch.Tensor
            Integer configurations of shape (*data_batch, n_sites), or embedded
            vectors of shape (*data_batch, n_sites, in_dim) for uniform
            dimensions. For heterogeneous dimensions, pass one tensor per site.
            Each site tensor has shape (*data_batch,) for indices or
            (*data_batch, in_dim[site]) for embeddings.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch). Structural batches and
            data batches remain independent.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> format.evaluate(torch.tensor([[0, 0], [0, 1]])).tolist()
        [1.0, 0.0]
        >>> embeddings = torch.ones(1, 2, 2)
        >>> format.evaluate(embeddings).tolist()
        [2.0]
        """
        site_data, discrete, data_batch_shape = self._normalize_data(
            data, self._in_dim, self._same_in_dim, n_batches)
        matrices = self._local_matrices(
            site_data, discrete, data_batch_shape)
        return self._contract_local_matrices(matrices)

    def error(self,
              function: Callable[..., torch.Tensor],
              samples: torch.Tensor,
              data: Optional[EvaluationData] = None,
              n_batches: int = 1,
              **kwargs: Any) -> SampleError:
        r"""Measures absolute and relative errors on a sample set.

        Parameters
        ----------
        function : callable
            Target callable invoked as function(samples, **kwargs), returning
            one value per data configuration. Its output is broadcast over
            structural batches stored in the format.
        samples : torch.Tensor
            Sample coordinates or indices passed to the target callable, with
            n_batches leading batch axes.
        data : torch.Tensor or sequence of torch.Tensor, optional
            Inputs used to evaluate the format. None evaluates samples directly;
            supply embedded inputs when target coordinates differ from format
            configurations.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.
        **kwargs : keyword arguments
            Additional arguments passed only to the target callable.

        Returns
        -------
        SampleError
            Scalar error record retaining autograd. Absolute and target norms
            aggregate all data and structural batches. Relative error is zero
            when both norms vanish and infinity when only the target norm
            vanishes.
        """
        if not callable(function):
            raise TypeError('`function` should be callable')
        if not isinstance(samples, torch.Tensor):
            raise TypeError('`samples` should be torch.Tensor type')
        if samples.ndim != (n_batches + 1):
            raise ValueError(
                '`samples` has an incompatible number of batch dimensions')

        samples = samples.to(device=self.device)
        approximation = self.evaluate(
            samples if data is None else data,
            n_batches=n_batches)
        target = function(samples, **kwargs)
        if not isinstance(target, torch.Tensor):
            raise TypeError('`function` should return a torch.Tensor')
        target = target.to(device=approximation.device,
                           dtype=approximation.dtype)
        target = target.reshape(samples.shape[:n_batches])
        if self.n_batches:
            target = target.reshape(
                *((1,) * self.n_batches), *target.shape)
            target = target.expand(*self._batch_shape, *target.shape[self.n_batches:])

        absolute = torch.linalg.vector_norm(approximation - target)
        denominator = torch.linalg.vector_norm(target)
        if denominator > 0:
            relative = absolute / denominator
        elif absolute == 0:
            relative = torch.zeros_like(absolute)
        else:
            relative = torch.full_like(absolute, torch.inf)

        size = int(torch.Size(samples.shape[:n_batches]).numel())
        return SampleError(
            kind='samples',
            absolute=absolute,
            relative=relative,
            size=size,
            denominator=denominator)

    @abstractmethod
    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts input-selected local matrices along the ranks."""


class _MatrixFormat1D(TensorFormat1D):
    """Shared raw-tensor matrix operations."""

    _family = 'matrix'

    def _operator_cores(self):
        """Returns matrix cores with separate input and output axes."""
        return [core.reshape(*self._batch_shape, core.shape[-3],
                             self._in_dim[site], self._out_dim[site],
                             core.shape[-1]).transpose(-1, -2)
                for site, core in enumerate(self._standard_cores())]

    def transpose(self):
        r"""Swaps local matrix input/output axes without reversing sites.

        Returns
        -------
        TTM or TRM
            Separate matrix format; tensor views may share storage. Quantics
            subclasses exchange input/output coordinate semantics.
        """
        from tensorkrowch.formats.operations import _build_network

        standard = []
        for site, core in enumerate(self._raw_standard_cores()):
            core = core.reshape(*self._batch_shape, core.shape[-3],
                                self._in_dim[site], self._out_dim[site], core.shape[-1])
            core = core.transpose(-3, -2)
            standard.append(core.reshape(*self._batch_shape,
                            core.shape[-4], -1, core.shape[-1]))
        result = _build_network(standard, self._out_dim, self._in_dim,
                                self._n_batches, self._cyclic)
        result.bonds = self._bonds
        return result

    def adjoint(self):
        r"""Returns the conjugate transpose, including explicit factors.

        Returns
        -------
        TTM or TRM
            Separate matrix format; tensor views may share storage. Quantics
            subclasses exchange input/output coordinate semantics.
        """
        return self.transpose().conj()

    @property
    def T(self):
        """Matrix transpose, without reversing the chain."""
        return self.transpose()

    @property
    def H(self):
        """Matrix adjoint, including conjugation of bond factors."""
        return self.adjoint()

    def trace(self) -> torch.Tensor:
        r"""Contracts the operator trace without forming the dense matrix.

        Returns
        -------
        torch.Tensor
            Trace resolved by structural batch. Every site should have matching
            input/output dimensions; global squareness alone is insufficient.
        """
        if self._in_dim != self._out_dim:
            raise ValueError('Trace requires matching local input/output dimensions')
        matrices = [core.diagonal(dim1=-3, dim2=-1).sum(-1)
                    for core in self._operator_cores()]
        return self._contract_open_chain(matrices).diagonal(dim1=-2, dim2=-1).sum(-1)

    def __call__(
            self,
            in_data: EvaluationData,
            out_data: Optional[EvaluationData] = None,
            n_batches: int = 1
    ) -> Union[torch.Tensor, TensorFormat1D]:
        r"""Calls :meth:`evaluate` with the same input conventions.

        Parameters
        ----------
        in_data : torch.Tensor or sequence of torch.Tensor
            Integer configurations of shape (*data_batch, n_sites), or embedded
            vectors of shape (*data_batch, n_sites, in_dim) for uniform
            dimensions. For heterogeneous dimensions, pass one tensor per site.
            Each site tensor has shape (*data_batch,) for indices or
            (*data_batch, in_dim[site]) for embeddings.
        out_data : torch.Tensor or sequence of torch.Tensor
            Output configurations or embeddings in the same layouts as in_data,
            using out_dim instead of in_dim. Both data batch shapes should
            match.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch). Embedded contractions
            use supplied vectors directly, without implicit conjugation.
        """
        if out_data is None:
            return self.apply(in_data, n_batches=n_batches)
        return self.evaluate(in_data, out_data, n_batches=n_batches)

    @abstractmethod
    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts entry-selected matrices with the topology closure."""

    @abstractmethod
    def _build_applied_decomposition(
            self,
            cores: List[torch.Tensor],
            n_batches: int) -> TensorFormat1D:
        """Builds the vector-like result produced by :meth:`apply`."""

    def _entry_matrices(
            self,
            in_data_by_site: Sequence[torch.Tensor],
            out_data_by_site: Sequence[torch.Tensor],
            in_discrete: bool,
            out_discrete: bool,
            data_batch_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Builds local matrices for paired matrix-entry evaluation."""
        core_batch_size = int(torch.Size(self._batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        matrices = []

        for core, in_site_data, out_site_data in zip(
                self._operator_cores(), in_data_by_site, out_data_by_site):
            left_rank, in_dim, right_rank, out_dim = core.shape[-4:]
            core = core.reshape(
                core_batch_size, left_rank, in_dim, right_rank, out_dim)

            if in_discrete and out_discrete:
                in_indices = in_site_data.reshape(data_batch_size).to(torch.long)
                out_indices = out_site_data.reshape(data_batch_size).to(torch.long)
                fused = in_indices * out_dim + out_indices
                matrix = core.permute(0, 1, 3, 2, 4).reshape(
                    core_batch_size, left_rank, right_rank, in_dim * out_dim)
                matrix = matrix[..., fused].permute(0, 3, 1, 2)
            elif in_discrete:
                indices = in_site_data.reshape(data_batch_size).to(torch.long)
                vectors = out_site_data.reshape(data_batch_size, out_dim)
                selected = core.index_select(2, indices)
                matrix = torch.einsum('cldro,do->cdlr', selected, vectors)
            elif out_discrete:
                indices = out_site_data.reshape(data_batch_size).to(torch.long)
                vectors = in_site_data.reshape(data_batch_size, in_dim)
                selected = core.index_select(4, indices)
                matrix = torch.einsum('clird,di->cdlr', selected, vectors)
            else:
                in_vectors = in_site_data.reshape(data_batch_size, in_dim)
                out_vectors = out_site_data.reshape(data_batch_size, out_dim)
                matrix = torch.einsum(
                    'di,do,cliro->cdlr', in_vectors, out_vectors, core)

            matrices.append(matrix.reshape(
                *self._batch_shape,
                *data_batch_shape,
                left_rank,
                right_rank))

        return matrices

    def evaluate(self,
                 in_data: EvaluationData,
                 out_data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        r"""Evaluates paired matrix configurations or embedded vectors.

        Parameters
        ----------
        in_data : torch.Tensor or sequence of torch.Tensor
            Integer configurations of shape (*data_batch, n_sites), or embedded
            vectors of shape (*data_batch, n_sites, in_dim) for uniform
            dimensions. For heterogeneous dimensions, pass one tensor per site.
            Each site tensor has shape (*data_batch,) for indices or
            (*data_batch, in_dim[site]) for embeddings.
        out_data : torch.Tensor or sequence of torch.Tensor
            Output configurations or embeddings in the same layouts as in_data,
            using out_dim instead of in_dim. Both data batch shapes should
            match.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch). Embedded contractions
            use supplied vectors directly, without implicit conjugation.
        """
        in_data_by_site, in_discrete, data_batch_shape = self._normalize_data(
            in_data, self._in_dim, self._same_in_dim, n_batches)
        out_data_by_site, out_discrete, out_batch_shape = self._normalize_data(
            out_data, self._out_dim, self._same_out_dim, n_batches)
        if out_batch_shape != data_batch_shape:
            raise ValueError(
                'Input and output data should have the same batch shape')

        matrices = self._entry_matrices(
            in_data_by_site,
            out_data_by_site,
            in_discrete,
            out_discrete,
            data_batch_shape)
        return self._contract_local_matrices(matrices)

    def apply(self,
              data: EvaluationData,
              n_batches: int = 1) -> TensorFormat1D:
        r"""Applies the operator to product data or another format.

        Parameters
        ----------
        data : torch.Tensor, sequence of torch.Tensor or TensorFormat1D
            Product input data accepted by vector evaluate(), or a TT/TR vector
            or TTM/TRM matrix. With a vector, contracts in_dim; with a matrix,
            contracts self.in_dim with data.out_dim.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        TensorFormat1D
            Vector or matrix format, according to the operand. Cyclic if either
            operand is cyclic. Product data become structural batches in the
            returned vector; applying a global dense vector does not
            automatically factor it into TT cores.

        Examples
        --------
        >>> operator = tk.formats.TTM([torch.eye(2)])
        >>> vector = tk.formats.TT([torch.tensor([2., 3.])])
        >>> torch.equal((operator @ vector).contract_dense(), vector.contract_dense())
        True
        >>> operator.apply(torch.tensor([[0], [1]])).contract_dense().tolist()
        [[1.0, 0.0], [0.0, 1.0]]
        >>> torch.equal((operator @ operator).contract_dense(), torch.eye(2))
        True
        """
        if isinstance(data, TensorFormat1D):
            from tensorkrowch.formats.operations import apply

            return apply(self, data)
        site_data, discrete, data_batch_shape = self._normalize_data(
            data, self._in_dim, self._same_in_dim, n_batches)
        core_batch_size = int(torch.Size(self._batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        output_cores = []

        for core, site_value in zip(self._operator_cores(), site_data):
            left_rank, in_dim, right_rank, out_dim = core.shape[-4:]
            core = core.reshape(
                core_batch_size, left_rank, in_dim, right_rank, out_dim)
            if discrete:
                indices = site_value.reshape(data_batch_size).to(torch.long)
                output_core = core.index_select(2, indices)
                output_core = output_core.permute(0, 2, 1, 4, 3)
            else:
                vectors = site_value.reshape(data_batch_size, in_dim)
                output_core = torch.einsum(
                    'di,cliro->cdlor', vectors, core)
            output_cores.append(output_core.reshape(
                *self._batch_shape,
                *data_batch_shape,
                left_rank,
                out_dim,
                right_rank))

        # NOTE: A future `apply_tt` could apply the matrix to a general TT
        return self._build_applied_decomposition(
            output_cores, self.n_batches + n_batches)
