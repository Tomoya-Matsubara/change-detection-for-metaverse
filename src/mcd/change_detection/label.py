"""Label schema for change detection."""

import typing
import warnings

import pydantic

from mcd.change_detection import exceptions

type LabelId = int
"""Label identifier type."""

type LabelName = str
"""Label name type."""


class BoundingBox(pydantic.BaseModel, frozen=True, strict=True):
    """Bounding box schema for change detection.

    Attributes
    ----------
    x : float
        The normalized x-coordinate of the center of the bounding box.
    y : float
        The normalized y-coordinate of the center of the bounding box.
    width : float
        The normalized width of the bounding box.
    height : float
        The normalized height of the bounding box.
    """

    x: float = pydantic.Field(ge=0.0, le=1.0)
    """The normalized x-coordinate of the center of the bounding box."""

    y: float = pydantic.Field(ge=0.0, le=1.0)
    """The normalized y-coordinate of the center of the bounding box."""

    width: float = pydantic.Field(gt=0.0, le=1.0)
    """The normalized width of the bounding box."""

    height: float = pydantic.Field(gt=0.0, le=1.0)
    """The normalized height of the bounding box."""

    @property
    def area(self) -> float:
        """Compute the area of the bounding box.

        Returns
        -------
        float
            The area of the bounding box.
        """
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Get the bounding box in the format (x1, y1, x2, y2).

        Returns
        -------
        tuple[float, float, float, float]
            The bounding box in the format (x1, y1, x2, y2).
        """
        x1 = self.x - self.width / 2
        y1 = self.y - self.height / 2
        x2 = self.x + self.width / 2
        y2 = self.y + self.height / 2

        # Due to floating-point precision computed by YOLO, the bounding box may not be
        # exactly within the range [0, 1]. Therefore, we allow a small epsilon for
        # checking the bounds.
        epsilon = 1e-6
        if x1 < -epsilon:
            warnings.warn(f"Bounding box x1 is less than 0 ({x1}).", stacklevel=2)
        if y1 < -epsilon:
            warnings.warn(f"Bounding box y1 is less than 0 ({y1}).", stacklevel=2)
        if x2 > 1 + epsilon:
            warnings.warn(f"Bounding box x2 is greater than 1 ({x2}).", stacklevel=2)
        if y2 > 1 + epsilon:
            warnings.warn(f"Bounding box y2 is greater than 1 ({y2}).", stacklevel=2)
        return x1, y1, x2, y2

    def compute_iou(self, other: typing.Self) -> float:
        """Compute the Intersection over Union (IoU) with another bounding box.

        Parameters
        ----------
        other : BoundingBox
            The other bounding box to compute the IoU with.

        Returns
        -------
        float
            The IoU between the two bounding boxes.
        """
        x1_min, y1_min, x1_max, y1_max = self.xyxy
        x2_min, y2_min, x2_max, y2_max = other.xyxy

        intersected_x_min = max(x1_min, x2_min)
        intersected_y_min = max(y1_min, y2_min)
        intersected_x_max = min(x1_max, x2_max)
        intersected_y_max = min(y1_max, y2_max)

        intersected_width = max(0, intersected_x_max - intersected_x_min)
        intersected_height = max(0, intersected_y_max - intersected_y_min)
        intersected_area = intersected_width * intersected_height

        union_area = self.area + other.area - intersected_area
        return intersected_area / union_area


class LabelInfo(pydantic.BaseModel, frozen=True, strict=True):
    """Label schema for change detection.

    Either `label_id` or `label_name` must be provided.

    Attributes
    ----------
    label_id : LabelId | None, default None
        Label identifier. Either this or `label_name` must be provided.
    label_name : LabelName | None, default None
        Label name. Either this or `label_id` must be provided.
    bounding_box : BoundingBox
        Bounding box of the label.
    """

    label_id: LabelId | None = None
    """Label identifier."""

    label_name: LabelName | None = None
    """Label name."""

    bounding_box: BoundingBox
    """Bounding box of the label."""

    @pydantic.model_validator(mode="after")
    def _check_if_label_is_provided(self) -> typing.Self:
        """Check if either `label_id` or `label_name` is provided."""
        if self.label_id is None and self.label_name is None:
            msg = "Either `label_id` or `label_name` must be provided."
            raise ValueError(msg)
        return self

    def is_label_same(self, other: typing.Self) -> bool:
        """Check if two labels are the same.

        Parameters
        ----------
        other : LabelInfo
            The other label information to compare with.

        Returns
        -------
        bool
            True if the labels are the same, False otherwise.

        Raises
        ------
        LabelInconsistentError
            Raised when the labels are inconsistent. The labels are inconsistent if
            either of the following conditions is met:

            - `label_id` is present in either label but not in the other.
            - `label_name` is present in either label but not in the other.
        """
        if self.label_id is not None:
            if other.label_id is None:
                raise exceptions.LabelInconsistentError(
                    label_type="label_id", label1=self.label_id, label2=other.label_id
                )
            return self.label_id == other.label_id

        # NOTE: Because of the model_validator, if `label_id` is None, `label_name` is
        # guaranteed to be not None.
        if other.label_name is None:
            raise exceptions.LabelInconsistentError(
                label_type="label_name", label1=self.label_name, label2=other.label_name
            )
        return self.label_name == other.label_name
