"""Result schema for refined change detection."""

import typing

import numpy as np
import numpy.typing as npt
import pydantic

from mcd import dtypes
from mcd.change_detection import result

type ClusterId = int
"""Cluster identifier type."""


class ClusteredPoint(pydantic.BaseModel, frozen=True, strict=True):
    """Point data type with change information.

    Attributes
    ----------
    image_id : dtypes.ImageId
        Image ID.
    change : result.Change
        Change type.
    point : dtypes.Point
        Center of the bounding box in 3D space.
    pixel : dtypes.Pixel
        Center of the bounding box in pixel space.
    cluster_id : ClusterId
        Cluster identifier.
    """

    image_id: dtypes.ImageId
    """Image ID."""

    change: result.Change
    """Change type."""

    point: dtypes.Point
    """Center of the bounding box in 3D space."""

    pixel: dtypes.Pixel
    """Center of the bounding box in pixel space."""

    cluster_id: ClusterId
    """Cluster identifier."""


class PointsInCluster(
    pydantic.BaseModel, frozen=True, strict=True, arbitrary_types_allowed=True
):
    """Points information in a cluster.

    Parameters
    ----------
    points : dtypes.NpArrayNx3Type[np.float32]
        Points in 3D space.
    pixels : dtypes.NpArrayNx2Type[np.float32]
        Points in pixel space.
    changes : dtypes.NpArrayNx1Type[np.int32]
        Change types.
    image_indices : dtypes.NpArrayNx1Type[np.int32]
        Image indices. The image indices are integer representations of the image IDs.
    index_to_image_id : dict[int, dtypes.ImageId]
        Mapping from index to image ID.

    Attributes
    ----------
    points : dtypes.NpArrayNx3Type[np.float32]
        Points in 3D space.
    pixels : dtypes.NpArrayNx2Type[np.float32]
        Points in pixel space.
    changes : dtypes.NpArrayNx1Type[np.int32]
        Change types.
    image_indices : dtypes.NpArrayNx1Type[np.int32]
        Image indices. The image indices are integer representations of the image IDs.
    """

    points: dtypes.NpArrayNx3Type[np.float32]
    """Points in 3D space."""

    pixels: dtypes.NpArrayNx2Type[np.int32]
    """Points in pixel space."""

    changes: dtypes.NpArrayNx1Type[np.int32]
    """Change types."""

    image_indices: dtypes.NpArrayNx1Type[np.int32]
    """Image indices.

    The image indices are integer representations of the image IDs.
    """

    def __init__(
        self,
        points: dtypes.NpArrayNx3Type[np.float32],
        pixels: dtypes.NpArrayNx2Type[np.int32],
        changes: dtypes.NpArrayNx1Type[np.int32],
        image_indices: dtypes.NpArrayNx1Type[np.int32],
        index_to_image_id: dict[int, dtypes.ImageId],
    ) -> None:
        super().__init__(
            points=points, pixels=pixels, changes=changes, image_indices=image_indices
        )

        self._index_to_image_id: typing.Final[dict[int, dtypes.ImageId]] = (
            index_to_image_id
        )
        """Mapping from index to image ID."""

    @pydantic.field_validator("changes", mode="after")
    @classmethod
    def _validate_changes(
        cls, changes: dtypes.NpArrayNx1Type[np.int32]
    ) -> dtypes.NpArrayNx1Type[np.int32]:
        """Validate the changes."""
        if not all(change.item() in result.Change for change in changes):
            message = "The change type must be one of the allowed values."
            raise ValueError(message)
        return changes

    @pydantic.model_validator(mode="after")
    def _validate_model(self) -> typing.Self:
        """Validate the model."""
        if not (
            len(self.points)
            == len(self.pixels)
            == len(self.changes)
            == len(self.image_indices)
        ):
            message = (
                "The number of points, pixels, changes, and image indices must be "
                "the same."
            )
            raise ValueError(message)
        return self

    @property
    def unique_change_values(self) -> list[int]:
        """Get the unique change values.

        Returns
        -------
        list[int]
            Unique change values
        """
        return list(np.unique(self.changes))

    @property
    def size(self) -> int:
        """Get the number of points in the cluster.

        Returns
        -------
        int
            Number of points in the cluster.
        """
        return len(self.points)

    def iter(
        self,
    ) -> typing.Generator[
        tuple[dtypes.Point, dtypes.Pixel, result.Change, dtypes.ImageId], None, None
    ]:
        """Iterate over the points in the cluster.

        Yields
        ------
        tuple[dtypes.Point, dtypes.Pixel, result.Change, dtypes.ImageId]
            Point in 3D space, pixel, change type, and image ID.
        """
        for point, pixel, change, image_index in zip(
            self.points, self.pixels, self.changes, self.image_indices, strict=True
        ):
            image_id = self._index_to_image_id[int(image_index.squeeze())]
            yield (
                dtypes.Point(x=point[0], y=point[1], z=point[2]),
                dtypes.Pixel(x=int(pixel[0]), y=int(pixel[1])),
                result.Change(change.squeeze()),
                image_id,
            )


class ClusteredPoints(
    pydantic.BaseModel, frozen=True, strict=True, arbitrary_types_allowed=True
):
    """List of clustered points."""

    root: list[ClusteredPoint]
    """List of clustered points."""

    def __init__(self, /, **data: list[ClusteredPoint]) -> None:
        super().__init__(**data)

        unique_image_ids = {point.image_id for point in data["root"]}

        self._index_to_image_id: typing.Final[dict[int, dtypes.ImageId]] = dict(
            enumerate(unique_image_ids)
        )
        """Mapping from index to image ID in the list of clustered points."""

        self._image_id_to_index: typing.Final[dict[dtypes.ImageId, int]] = {
            image_id: index for index, image_id in self._index_to_image_id.items()
        }
        """Mapping from image ID to index in the list of clustered points."""

        coordinates_with_metadata = np.array(
            [
                [
                    point.point.x,
                    point.point.y,
                    point.point.z,
                    point.pixel.x,
                    point.pixel.y,
                    point.change.value,
                    self._image_id_to_index[point.image_id],
                    point.cluster_id,
                ]
                for point in data["root"]
            ]
        )
        self._coordinates_with_metadata: typing.Final[npt.NDArray[np.generic]] = (
            coordinates_with_metadata
        )
        """NumPy representation of the coordinates with metadata.

        Each row contains the following columns in order:
        - (3) Position in 3D space (x, y, z).
        - (2) Position in pixel space (x, y).
        - (1) Change type.
        - (1) Image ID.
        - (1) Cluster ID.
        """

    @pydantic.model_serializer
    def _serialize_model(self) -> list[ClusteredPoint]:
        """Serialize the model to a list of ClusteredPoint."""
        return self.root

    @property
    def unique_cluster_ids(self) -> list[ClusterId]:
        """Get the unique cluster identifiers.

        Returns
        -------
        list[ClusterId]
            Unique cluster identifiers.
        """
        return [
            int(cluster_id)
            for cluster_id in np.unique(self._coordinates_with_metadata[:, -1])
        ]

    def get_points_in_cluster(self, cluster_id: ClusterId) -> PointsInCluster:
        """Get the points in a cluster.

        Parameters
        ----------
        cluster_id : ClusterId
            Cluster identifier.

        Returns
        -------
        PointsInCluster
            Points information in the cluster.
        """
        coordinates_with_metadata = self._coordinates_with_metadata[
            self._coordinates_with_metadata[:, -1] == cluster_id
        ]
        points, pixels, changes, image_ids, _ = np.split(
            coordinates_with_metadata, [3, 5, 6, 7], axis=1
        )
        return PointsInCluster(
            points=points,
            pixels=pixels,
            changes=changes,
            image_indices=image_ids,
            index_to_image_id=self._index_to_image_id,
        )
