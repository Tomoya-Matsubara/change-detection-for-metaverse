"""Camera data type."""

import numpy as np
import pydantic

from mcd.dtypes import _array


class Camera(
    pydantic.BaseModel, frozen=True, strict=True, arbitrary_types_allowed=True
):
    """Camera parameters.

    Attributes
    ----------
    intrinsic : NpArray3x3Type[np.float32]
        Intrinsic matrix.
    view_matrix : NpArray4x4Type[np.float32]
        View matrix.
    """

    intrinsic: _array.NpArray3x3Type[np.float32] = pydantic.Field(
        validation_alias=pydantic.AliasChoices("intrinsic", "intrinsics")
    )
    """Intrinsic matrix.

    The transformation from camera coordinates to image coordinates.
    """

    view_matrix: _array.NpArray4x4Type[np.float32]
    """View matrix.

    The transformation from world coordinates to camera coordinates.
    """
