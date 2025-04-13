"""Configuration for change detection."""

import typing

import pydantic


class ChangeDetectionConfig(pydantic.BaseModel, frozen=True, strict=True):
    """Configuration for change detection.

    Attributes
    ----------
    loader_id : Literal["yolo"]
        ID of the loader to use. The loader is used to load the object detection
        results.
    """

    loader_id: typing.Literal["yolo"]
    """ID of the loader to use.

    The loader is used to load the object detection results.
    """
