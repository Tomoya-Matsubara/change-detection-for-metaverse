"""Utility functions for ARKit integration."""

import typing

import numpy as np

from mcd import dtypes

_flip_yz_matrix: typing.Final[dtypes.NpArray4x4Type[np.float32]] = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
)
"""Transformation matrix to flip the Y and Z axes.

In the portrait-oriented coordinate system, the X axis points to the right, the Y axis
points down, and the Z axis points to the direction the camera is facing. To convert
from the portrait-oriented coordinate system to the right-handed coordinate system, we
need to flip the Y and Z axes (i.e., reverse the direction of the Y and Z axes).
"""

ARKIT_MATRIX: typing.Final[dtypes.NpArray4x4Type[np.float32]] = _flip_yz_matrix
"""Transformation matrix to correct the coordinate system.

The transformation matrix that needs to be multiplied to the view matrix to correct the
coordinate system.
"""


__all__ = ["ARKIT_MATRIX"]
