"""Result schema for change detection."""

import enum

import pydantic

from mcd import dtypes
from mcd.change_detection import label


class Change(enum.IntEnum):
    """Enumeration for change types."""

    REMOVED = -1
    """Removed label."""

    UNCHANGED = 0
    """Unchanged label."""

    ADDED = 1
    """Added label."""

    def __str__(self) -> str:
        """Return the string representation of the change type."""
        return self.name.lower()

    @property
    def color(self) -> str:
        """Return the color associated with the change type."""
        match self:
            case Change.UNCHANGED:
                return "black"
            case Change.ADDED:
                return "green"
            case Change.REMOVED:
                return "red"


class SinglePairResult(pydantic.BaseModel):
    """Result schema for change detection of a single pair of images.

    Attributes
    ----------
    added : set[label.LabelInfo]
        Set of labels classified as added. All labels in this set should come from the
        second image.
    removed : set[label.LabelInfo]
        Set of labels classified as removed. All labels in this set should come from the
        first image.
    unchanged : set[label.LabelInfo]
        Set of labels classified as unchanged. All labels in this set come from the
        first image. Since this is an arbitrary choice, the labels could come from the
        second image as well, or a combination of both is also possible.
    """

    added: set[label.LabelInfo]
    """Set of labels classified as added.

    All labels in this set should come from the second image.
    """

    removed: set[label.LabelInfo]
    """Set of labels classified as removed.

    All labels in this set should come from the first image.
    """

    unchanged: set[label.LabelInfo]
    """Set of labels classified as unchanged.

    All labels in this set come from the first image. Since this is an arbitrary
    choice, the labels could come from the second image as well, or a combination of
    both is also possible.
    """

    @pydantic.model_serializer
    def _serialize_model(self) -> dict[str, list[label.LabelInfo]]:
        """Serialize the model to a dictionary."""
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "unchanged": list(self.unchanged),
        }


class ChangeDetectionResults(pydantic.BaseModel):
    """Schema for results of change detection for a set of images.

    Attributes
    ----------
    result : dict[dtypes.ImageId, SinglePairResult]
        Result of change detection for each image.
    image_height : pydantic.PositiveInt
        Height of the image.
    image_width : pydantic.PositiveInt
        Width of the image.
    """

    result: dict[dtypes.ImageId, SinglePairResult]
    """Result of change detection for each image."""

    image_height: pydantic.PositiveInt
    """Height of the image."""

    image_width: pydantic.PositiveInt
    """Width of the image."""
