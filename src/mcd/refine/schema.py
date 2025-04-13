"""Schema for refining change detection results."""

import typing

import numpy as np
import pydantic

from mcd import dtypes
from mcd.change_detection import label, result


class ChangePoint(pydantic.BaseModel, frozen=True, strict=True):
    """Point data type with change information.

    Attributes
    ----------
    image_id : dtypes.ImageId
        Image ID.
    change : result.Change
        Change type.
    point : Point
        Center of the bounding box in 3D space.
    pixel : dtypes.Pixel
        Center of the bounding box in pixel space.
    """

    image_id: dtypes.ImageId
    """Image ID."""

    change: result.Change
    """Change type."""

    point: dtypes.Point
    """Center of the bounding box in 3D space."""

    pixel: dtypes.Pixel
    """Center of the bounding box in pixel space."""


class ChangePoints(
    pydantic.BaseModel, frozen=True, strict=True, arbitrary_types_allowed=True
):
    """List of change points."""

    root: list[ChangePoint]
    """List of change points."""

    coordinates: dtypes.NpArrayNx3Type[np.float32] = pydantic.Field(init=False)
    """NumPy representation of the coordinates."""

    def __init__(self, /, **data: list[ChangePoint]) -> None:
        coordinates = np.array(
            [[point.point.x, point.point.y, point.point.z] for point in data["root"]]
        )
        super().__init__(**data, coordinates=coordinates)

    @pydantic.model_serializer
    def _serialize_model(self) -> list[ChangePoint]:
        """Serialize the model to a list of ChangePoint."""
        return self.root


class LabelInfo3d(pydantic.BaseModel, frozen=True, strict=True):
    """Label schema for change detection in 3D space.

    Attributes
    ----------
    label_id : label.LabelId | None
        Label identifier.
    label_name : label.LabelName | None
        Label name.
    pixel : dtypes.Pixel
        Pixel of the center of the bounding box.
    point : dtypes.Point
        3D point of the center of the bounding box.
    """

    label_id: label.LabelId | None = None
    """Label identifier."""

    label_name: label.LabelName | None = None
    """Label name."""

    pixel: dtypes.Pixel
    """Pixel of the center of the bounding box."""

    point: dtypes.Point
    """3D point of the center of the bounding box."""

    @pydantic.model_validator(mode="after")
    def _check_if_label_is_provided(self) -> typing.Self:
        """Check if either `label_id` or `label_name` is provided."""
        if self.label_id is None and self.label_name is None:
            message = "Either `label_id` or `label_name` must be provided."
            raise ValueError(message)
        return self

    @property
    def label(self) -> label.LabelId | label.LabelName:
        """Get the label ID or name.

        Note that the label name is preferred over the label ID.

        Returns
        -------
        label.LabelId | label.LabelName
            If the label name is provided, it is returned. Otherwise, the label ID is
            returned.
        """
        if self.label_name is not None:
            return self.label_name
        if self.label_id is not None:
            return self.label_id
        message = "Label ID or name must be provided."
        raise ValueError(message)


class SinglePairResult3d(pydantic.BaseModel, frozen=True, strict=True):
    """Result schema for change detection in 3D space of a single pair of images.

    Unlike the 2D version, origins of items in the sets is not important because the
    two scenes are expected to be aligned in 3D space.

    Attributes
    ----------
    added : set[LabelInfo3d]
        Set of labels classified as added.
    removed : set[LabelInfo3d]
        Set of labels classified as removed.
    unchanged : set[LabelInfo3d]
        Set of labels classified as unchanged.
    """

    added: set[LabelInfo3d]
    """Set of labels classified as added."""

    removed: set[LabelInfo3d]
    """Set of labels classified as removed."""

    unchanged: set[LabelInfo3d]
    """Set of labels classified as unchanged."""

    @pydantic.model_serializer
    def _serialize_model(self) -> dict[str, list[LabelInfo3d]]:
        """Serialize the model to a dictionary."""
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "unchanged": list(self.unchanged),
        }


class ChangeDetection3dResults(
    pydantic.RootModel[dict[dtypes.ImageId, SinglePairResult3d]]
):
    """Results of change detection in 3D space."""

    @pydantic.model_serializer
    def _serialize_model(self) -> dict[dtypes.ImageId, SinglePairResult3d]:
        """Serialize the model.

        Returns
        -------
        dict[dtypes.ImageId, list[SinglePairResult3d]]
            Serialized model.
        """
        return dict(self.root.items())

    def items(self) -> typing.ItemsView[dtypes.ImageId, SinglePairResult3d]:
        """Return a view of the items."""
        return self.root.items()
