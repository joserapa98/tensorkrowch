"""
This script contains:

    SVD backend configuration:
        * get_svd_method
        * set_svd_method
        * svd_method

The initial SVD backend can be selected with the environment variable
``TENSORKROWCH_SVD_METHOD``. Runtime configuration takes precedence over that
initial value.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Text


_SVD_METHODS = ('svd', 'qr_svd')
_SVD_METHOD_OVERRIDE = ContextVar(
    'tensorkrowch_svd_method_override',
    default=None)


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


@contextmanager
def svd_method(method: Text) -> Iterator[None]:
    """
    Temporarily selects the exact SVD backend in the current context.

    The previous backend is restored even if the context exits with an
    exception. Nested contexts are supported and the override is local to the
    current thread or asynchronous context.

    Parameters
    ----------
    method : {"svd", "qr_svd"}
        Backend used inside the context.

    Examples
    --------
    >>> with svd_method('qr_svd'):
    ...     active_method = get_svd_method()
    >>> active_method
    'qr_svd'
    """
    token = _SVD_METHOD_OVERRIDE.set(_validate_svd_method(method))
    try:
        yield
    finally:
        _SVD_METHOD_OVERRIDE.reset(token)
