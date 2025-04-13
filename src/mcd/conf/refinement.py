"""Configuration for the refinement process."""

import typing

import pydantic


class RefinementConfig(pydantic.BaseModel, frozen=True, strict=True):
    """Configuration for the refinement process.

    Attributes
    ----------
    loader_id : typing.Literal["arkit_ue5"]
        ID of the refinement data loader to use. The loader is used to load the frame
        data (e.g., images, depth maps, camera parameters) for the refinement process.
    """

    loader_id: typing.Literal["arkit_ue5"]
    """ID of the refinement data loader to use.

    The loader is used to load the frame data (e.g., images, depth maps, camera
    parameters) for the refinement process.
    """
