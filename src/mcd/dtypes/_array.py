"""Array data type module."""

import functools
import typing

import numpy as np
import numpy.typing as npt
import pydantic
from numpy._typing import _shape


def _check_array_shape[T: np.generic](
    v: npt.NDArray[T], shape: _shape._ShapeLike
) -> npt.NDArray[T]:
    """Check if the input array has the specified shape.

    Parameters
    ----------
    v : np.ndarray
        Input array.
    shape : _shape._ShapeLike
        Expected shape of the array.

    Returns
    -------
    np.ndarray
        The input array if it has the expected shape.

    Raises
    ------
    ValueError
        If the input array does not have the expected shape.
    """
    if v.shape != shape:
        message = f"Input array must have shape {shape}."
        raise ValueError(message)
    return v


type NpArray1dType[T: np.generic] = typing.Annotated[
    npt.NDArray[T],
    pydantic.AfterValidator(functools.partial(_check_array_shape, shape=())),
]
"""1D numpy array type."""

type NpArray3x3Type[T: np.generic] = typing.Annotated[
    npt.NDArray[T],
    pydantic.AfterValidator(functools.partial(_check_array_shape, shape=(3, 3))),
]
"""3x3 numpy array type."""

type NpArray4x4Type[T: np.generic] = typing.Annotated[
    npt.NDArray[T],
    pydantic.AfterValidator(functools.partial(_check_array_shape, shape=(4, 4))),
]
"""4x4 numpy array type."""


def _check_batch_array_shape[T: np.generic](
    v: npt.NDArray[T], shape: _shape._ShapeLike
) -> npt.NDArray[T]:
    """Check if the input array has the specified shape.

    Parameters
    ----------
    v : np.ndarray
        Input array.
    shape : _shape._ShapeLike
        Expected shape of the array.

    Returns
    -------
    np.ndarray
        The input array if it has the expected shape.

    Raises
    ------
    ValueError
        If the input array does not have the expected shape.
    """
    if v.shape[1:] != shape:
        message = f"Input array must have shape (batch_size, {shape})."
        raise ValueError(message)
    return v


type NpArrayNx1Type[T: np.generic] = typing.Annotated[
    npt.NDArray[T],
    pydantic.AfterValidator(functools.partial(_check_batch_array_shape, shape=(1,))),
]
"""Nx1 numpy array type."""

type NpArrayNx2Type[T: np.generic] = typing.Annotated[
    npt.NDArray[T],
    pydantic.AfterValidator(functools.partial(_check_batch_array_shape, shape=(2,))),
]
"""Nx2 numpy array type."""

type NpArrayNx3Type[T: np.generic] = typing.Annotated[
    npt.NDArray[T],
    pydantic.AfterValidator(functools.partial(_check_batch_array_shape, shape=(3,))),
]
"""Nx3 numpy array type."""

type NpArrayNx8Type[T: np.generic] = typing.Annotated[
    npt.NDArray[T],
    pydantic.AfterValidator(functools.partial(_check_batch_array_shape, shape=(8,))),
]
"""Nx8 numpy array type."""
