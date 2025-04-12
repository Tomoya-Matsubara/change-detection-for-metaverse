"""3D space data type module."""

import pydantic


class Point(pydantic.BaseModel, frozen=True, strict=True):
    """Point data type.

    Attributes
    ----------
    x : float
        X-coordinate of the point.
    y : float
        Y-coordinate of the point.
    z : float
        Z-coordinate of the point.
    """

    x: float
    """X-coordinate of the point."""

    y: float
    """Y-coordinate of the point."""

    z: float
    """Z-coordinate of the point."""
