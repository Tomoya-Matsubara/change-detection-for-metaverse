"""Data loader for source data captured by ARKit."""

import json
import pathlib
import typing

import cv2
import numpy as np
import pydantic

from mcd import dtypes
from mcd.loader import base
from mcd.utils import arkit


class _ArkitDepthMap(pydantic.BaseModel, frozen=True, strict=True):
    """Depth map captured by ARKit.

    Attributes
    ----------
    height : pydantic.PositiveInt
        Height of the depth map.
    width : pydantic.PositiveInt
        Width of the depth map.
    values : list[pydantic.NonNegativeFloat]
        Flattened values of the depth map.
    """

    height: pydantic.PositiveInt
    """Height of the depth map."""

    width: pydantic.PositiveInt
    """Width of the depth map."""

    values: list[pydantic.NonNegativeFloat]
    """Flattened values of the depth map."""

    def to_numpy(self) -> dtypes.GrayscaleImageType[np.float32]:
        """Convert the depth map to a NumPy array.

        Returns
        -------
        dtypes.GrayscaleImageType[np.float32]
            Depth map.
        """
        depth_map = np.array(self.values).reshape(self.height, self.width)
        # ARKit depth maps are rotated 90 degrees clockwise
        return cv2.rotate(depth_map, cv2.ROTATE_90_CLOCKWISE).astype(np.float32)


class _ArkitConfidenceMap(pydantic.BaseModel, frozen=True, strict=True):
    """Confidence map captured by ARKit.

    Attributes
    ----------
    height : pydantic.PositiveInt
        Height of the confidence map.
    width : pydantic.PositiveInt
        Width of the confidence map.
    values : list[typing.Literal[0, 1, 2]]
        Values of the confidence map.
    """

    height: pydantic.PositiveInt
    """Height of the confidence map."""

    width: pydantic.PositiveInt
    """Width of the confidence map."""

    values: list[typing.Literal[0, 1, 2]]
    """Values of the confidence map."""


class ArkitFrame(dtypes.Camera, frozen=True, strict=True):
    """Frame data captured by ARKit.

    Attributes
    ----------
    resolution : tuple[pydantic.PositiveInt, pydantic.PositiveInt]
        Resolution of the image in the format (height, width).
    timestamp : pydantic.NonNegativeFloat
        Timestamp of the frame.
    frame_number : pydantic.NonNegativeInt
        Frame number.
    depth_map : _ArkitDepthMap
        Depth map of the frame.
    confidence_map : _ArkitConfidenceMap
        Confidence map of the frame.
    """

    resolution: tuple[pydantic.PositiveInt, pydantic.PositiveInt]
    """Resolution of the image in the format (height, width)."""

    timestamp: pydantic.NonNegativeFloat
    """Timestamp of the frame."""

    frame_number: pydantic.NonNegativeInt
    """Frame number."""

    depth_map: _ArkitDepthMap
    """Depth map of the frame."""

    confidence_map: _ArkitConfidenceMap
    """Confidence map of the frame."""

    @pydantic.field_validator("resolution", mode="before")
    @classmethod
    def _validate_resolution(
        cls, resolution: tuple[pydantic.PositiveInt, pydantic.PositiveInt]
    ) -> tuple[pydantic.PositiveInt, pydantic.PositiveInt]:
        """Validate the resolution of the image."""
        if isinstance(resolution, list):
            return tuple(resolution)
        return resolution

    @pydantic.field_validator("view_matrix", mode="before")
    @classmethod
    def _validate_view_matrix(
        cls, view_matrix: list[list[float]]
    ) -> dtypes.NpArray4x4Type[np.float32]:
        """Validate the view matrix."""
        if isinstance(view_matrix, np.ndarray):
            return view_matrix
        return np.array(view_matrix).squeeze().T

    @pydantic.field_validator("intrinsic", mode="before")
    @classmethod
    def _validate_intrinsic(
        cls, intrinsic: list[list[float]]
    ) -> dtypes.NpArray3x3Type[np.float32]:
        """Validate the intrinsic matrix."""
        if isinstance(intrinsic, np.ndarray):
            return intrinsic
        return np.array(intrinsic).squeeze().T


class ArkitImageLoader(base.ImageLoaderBase):
    """Data loader for images captured by ARKit.

    The loader expects the following dataset structure:

    ```
    dataset/
    ├── images/
    │   ├── image_0.png
    │   └── ...
    ├── confidence/
    │   ├── image_0.png
    │   └── ...
    └── depth/
        ├── image_0.json
        └── ...
    ```

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset directory containing the images.

    Attributes
    ----------
    correction_matrix : dtypes.NpArray4x4Type[np.float32] | None
        Matrix to correct the position in camera space.
    """

    def __init__(self, dataset_path: pathlib.Path) -> None:
        super().__init__(dataset_path, correction_matrix=arkit.ARKIT_MATRIX)

        self._size: typing.Final[int] = sum(
            1 for _ in self._dataset_path.glob("images/*.png")
        )
        """Number of images in the dataset."""

    def __len__(self) -> int:
        """Get the number of images in the dataset.

        Returns
        -------
        int
            Number of images.
        """
        return self._size

    def __iter__(
        self,
    ) -> typing.Iterator[
        tuple[
            dtypes.ImageId,
            dtypes.ColoredImageType[np.uint8],
            dtypes.GrayscaleImageType[np.float32],
            dtypes.GrayscaleImageType[np.uint8] | None,
            dtypes.Camera,
        ],
    ]:
        """Iterate over the images in the dataset in chronological order.

        Yields
        ------
        tuple[
            dtypes.ImageId,
            dtypes.ColoredImageType[np.uint8],
            dtypes.GrayscaleImageType[np.float32],
            dtypes.GrayscaleImageType[np.uint8] | None,
            dtypes.Camera,
        ]
            Image ID, image, depth map, and confidence map.
        """
        # Sort the image IDs in chronological order, in other words, natural sorting.
        image_ids = sorted(
            [path.stem for path in self._dataset_path.glob("images/*.png")],
            key=lambda x: int(x.split("_")[-1]),
        )
        for image_id in image_ids:
            yield (
                image_id,
                self._load_image(image_id),
                self._load_depth_map(image_id),
                self._get_confidence_map(image_id),
                self._load_camera_parameters(image_id),
            )

    def _load_image(
        self, image_id: dtypes.ImageId
    ) -> dtypes.ColoredImageType[np.uint8]:
        """Load an image.

        Parameters
        ----------
        image_id : dtypes.ImageId
            Identifier of the image.

        Returns
        -------
        dtypes.ColoredImageType[np.uint8]
            Image.
        """
        path = self._dataset_path / "images" / f"{image_id}.png"
        image = cv2.imread(path.as_posix())
        return image.astype(np.uint8)

    def _load_depth_map(
        self, image_id: dtypes.ImageId
    ) -> dtypes.GrayscaleImageType[np.float32]:
        """Load a depth map.

        Parameters
        ----------
        image_id : dtypes.ImageId
            Identifier of the image.

        Returns
        -------
        dtypes.GrayscaleImageType[np.float32]
            Loaded depth map.
        """
        path = self._dataset_path / "depth" / f"{image_id}.json"
        with path.open("r") as file:
            frame = ArkitFrame.model_validate(json.load(file))
        depth_map = frame.depth_map.to_numpy()

        # ARKit depth maps are smaller than the image resolution
        return cv2.resize(
            depth_map, (int(frame.resolution[1]), int(frame.resolution[0]))
        ).astype(np.float32)

    def _get_confidence_map(
        self, image_id: dtypes.ImageId
    ) -> dtypes.GrayscaleImageType[np.uint8] | None:
        """Get the confidence map of the image.

        Parameters
        ----------
        image_id : dtypes.ImageId
            Identifier of the image.

        Returns
        -------
        dtypes.GrayscaleImageType[np.uint8] | None
            Confidence map or None if not available.
        """
        path = self._dataset_path / "confidence" / f"{image_id}.png"
        confidence_map = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
        if confidence_map is None:
            return None

        # NOTE: The confidence map has an extra column at the end for some reason.
        #       We remove it here.
        return confidence_map[:, :-1].astype(np.uint8)

    def _load_camera_parameters(self, image_id: dtypes.ImageId) -> dtypes.Camera:
        """Load the camera parameters of an image.

        Parameters
        ----------
        image_id : dtypes.ImageId
            Identifier of the image.

        Returns
        -------
        dtypes.Camera
            Camera parameters.
        """
        path = self._dataset_path / "depth" / f"{image_id}.json"
        with path.open("r") as file:
            frame = ArkitFrame.model_validate(json.load(file))
        return dtypes.Camera(intrinsic=frame.intrinsic, view_matrix=frame.view_matrix)
