"""Configuration for object detection."""

import typing

import pydantic

from mcd.conf import _utils


class ObjectDetectionConfig(pydantic.BaseModel):
    """Configuration for object detection.

    Attributes
    ----------
    model_path : pathlib.Path
        Path to the object detection model file.
    object_detector_id : Literal["yolo"]
        ID of the object detector to use.
    """

    model_path: _utils.Path
    """Path to the object detection model file."""

    object_detector_id: typing.Literal["yolo"]
    """ID of the object detector to use."""
