"""Image data type module."""

import typing

import numpy as np
import numpy.typing as npt
import pydantic

type ImageId = str
"""Type for image identifier."""


def _validate_grayscale_image[T: np.generic](
    v: npt.NDArray[T],
) -> "GrayscaleImageType[T]":
    """Validate if the input is a grayscale image.

    Parameters
    ----------
    v : np.ndarray, shape (H, W)
        Input image.

    Returns
    -------
    GrayscaleImageType
        The validated grayscale image.

    Raises
    ------
    ValueError
        If the input is not a 2D array.
    """
    expected_dimensions = 2
    if v.ndim != expected_dimensions:
        message = "Input image must be a 2D array."
        raise ValueError(message)
    return v


type GrayscaleImageType[T: np.generic] = typing.Annotated[
    npt.NDArray[T], pydantic.AfterValidator(_validate_grayscale_image)
]
"""Grayscale image type."""


def _validate_colored_image[T: np.generic](v: npt.NDArray[T]) -> "ColoredImageType[T]":
    """Validate if the input is a colored image.

    Parameters
    ----------
    v : np.ndarray, shape (H, W, 3)
        Input image.

    Returns
    -------
    ColoredImageType
        The validated colored image.

    Raises
    ------
    ValueError
        If the input is not a 3D array with 3 channels.
    """
    num_channels = 3
    if v.ndim != num_channels or v.shape[2] != num_channels:
        msg = "Input image must be a 3D array with 3 channels."
        raise ValueError(msg)
    return v


type ColoredImageType[T: np.generic] = typing.Annotated[
    npt.NDArray[T], pydantic.AfterValidator(_validate_colored_image)
]
"""Colored image type."""


class Pixel(pydantic.BaseModel, frozen=True, strict=True):
    """Pixel data type.

    Attributes
    ----------
    x : pydantic.NonNegativeInt
        X-coordinate of the pixel.
    y : pydantic.NonNegativeInt
        Y-coordinate of the pixel.
    """

    x: pydantic.NonNegativeInt
    """X-coordinate of the pixel."""

    y: pydantic.NonNegativeInt
    """Y-coordinate of the pixel."""
