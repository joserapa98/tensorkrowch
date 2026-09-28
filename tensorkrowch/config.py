"""
This script contains:

    SVD backend configuration:
        * get_svd_method
        * set_svd_method
        * get_svd_refinement
        * set_svd_refinement
        * svd_method

The initial SVD backend can be selected with the environment variable
``TENSORKROWCH_SVD_METHOD``. Runtime configuration takes precedence over that
initial value.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional, Text


_SVD_METHODS = ('svd', 'qr_svd')
_SVD_METHOD_OVERRIDE = ContextVar(
    'tensorkrowch_svd_method_override',
    default=None)
_SVD_REFINEMENT_OVERRIDE = ContextVar(
    'tensorkrowch_svd_refinement_override',
    default=None)
_DEFAULT_SVD_REFINEMENT = False


def _validate_svd_method(method: Text) -> Text:
    """Validates and returns an exact SVD backend name."""
    if not isinstance(method, str):
        raise TypeError('`method` should be str type')
    if method not in _SVD_METHODS:
        raise ValueError('`method` can only be "svd" or "qr_svd"')
    return method


_DEFAULT_SVD_METHOD = _validate_svd_method(
    os.getenv('TENSORKROWCH_SVD_METHOD', 'svd'))


def get_svd_method() -> Text:
    """
    Returns the active exact SVD backend.

    Returns
    -------
    str
        Either ``"svd"`` or ``"qr_svd"``.
    """
    override = _SVD_METHOD_OVERRIDE.get()
    if override is not None:
        return override
    return _DEFAULT_SVD_METHOD


def set_svd_method(method: Text) -> None:
    """
    Sets the process-wide default exact SVD backend.

    Parameters
    ----------
    method : {"svd", "qr_svd"}
        Backend used by subsequent SVD-based operations outside a temporary
        :func:`svd_method` context.
    """
    global _DEFAULT_SVD_METHOD
    _DEFAULT_SVD_METHOD = _validate_svd_method(method)


def get_svd_refinement() -> bool:
    """
    Returns whether recursive refinement of small singular values is active.

    Returns
    -------
    bool
        The temporary override, when present, or the process-wide default.
    """
    override = _SVD_REFINEMENT_OVERRIDE.get()
    if override is not None:
        return override
    return _DEFAULT_SVD_REFINEMENT


def set_svd_refinement(refine: bool) -> None:
    """
    Sets the process-wide default for recursive SVD refinement.

    Parameters
    ----------
    refine : bool
        Whether SVD-based operations refine small singular values with
        :func:`~tensorkrowch.utils.accurate_svd`. Initially False. Refinement
        is independent of the exact backend selected by :func:`set_svd_method`.
    """
    if not isinstance(refine, bool):
        raise TypeError('`refine` should be bool type')
    global _DEFAULT_SVD_REFINEMENT
    _DEFAULT_SVD_REFINEMENT = refine


@contextmanager
def svd_method(method: Text,
               refine: Optional[bool] = None) -> Iterator[None]:
    """
    Temporarily selects the exact SVD backend in the current context.

    The previous backend is restored even if the context exits with an
    exception. Nested contexts are supported and the override is local to the
    current thread or asynchronous context.

    Parameters
    ----------
    method : {"svd", "qr_svd"}
        Backend used inside the context.
    refine : bool, optional
        Whether to recursively refine small singular values, independently of
        the backend. If omitted, inherits the active refinement setting.

    Examples
    --------
    >>> with svd_method('qr_svd'):
    ...     active_method = get_svd_method()
    >>> active_method
    'qr_svd'
    """
    method = _validate_svd_method(method)
    if (refine is not None) and not isinstance(refine, bool):
        raise TypeError('`refine` should be bool type or None')
    token = _SVD_METHOD_OVERRIDE.set(method)
    refinement_token = _SVD_REFINEMENT_OVERRIDE.set(
        get_svd_refinement() if refine is None else refine)
    try:
        yield
    finally:
        _SVD_METHOD_OVERRIDE.reset(token)
        _SVD_REFINEMENT_OVERRIDE.reset(refinement_token)
