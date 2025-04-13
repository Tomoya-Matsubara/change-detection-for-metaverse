"""Unit tests for the `result` module."""

import numpy as np
import pydantic
import pytest
from numpy import testing

from mcd import dtypes
from mcd.change_detection import result as cd_result
from mcd.refine import result


class TestPointsInCluster:
    """Test suite for the `PointsInCluster` class."""

    class TestValidateChanges:
        """Test suite for the `_validate_changes()` method."""

        def test_valid_changes(self) -> None:
            """Test valid changes."""
            try:
                result.PointsInCluster(
                    points=np.array([[0.0, 0.0, 0.0]]),
                    pixels=np.array([[0, 0]]),
                    changes=np.array([[cd_result.Change.ADDED]]),
                    image_indices=np.array([[0]]),
                    index_to_image_id={0: "image_0"},
                )
            except pydantic.ValidationError:
                pytest.fail("ValidationError raised unexpectedly.")

        def test_invalid_changes(self) -> None:
            """Test invalid changes."""
            with pytest.raises(
                pydantic.ValidationError,
                match="The change type must be one of the allowed values.",
            ):
                result.PointsInCluster(
                    points=np.array([[0.0, 0.0, 0.0]]),
                    pixels=np.array([[0, 0]]),
                    changes=np.array([[100]]),
                    image_indices=np.array([[0]]),
                    index_to_image_id={},
                )

    class TestValidateModel:
        """Test suite for the `_validate_model()` method."""

        def test_valid_model(self) -> None:
            """Test a valid model."""
            try:
                result.PointsInCluster(
                    points=np.array([[0.0, 0.0, 0.0]]),
                    pixels=np.array([[0, 0]]),
                    changes=np.array([[cd_result.Change.ADDED]]),
                    image_indices=np.array([[0]]),
                    index_to_image_id={0: "image_0"},
                )
            except pydantic.ValidationError:
                pytest.fail("ValidationError raised unexpectedly.")

        def test_invalid_model(self) -> None:
            """Test an invalid model."""
            with pytest.raises(
                pydantic.ValidationError,
                match="The number of points, pixels, changes, and image indices must be"
                " the same.",
            ):
                result.PointsInCluster(
                    points=np.array([[0.0, 0.0, 0.0]]),
                    pixels=np.array([[0, 0]]),
                    changes=np.array([[cd_result.Change.ADDED]]),
                    image_indices=np.array([[0], [0]]),
                    index_to_image_id={},
                )

    @pytest.mark.parametrize(
        ("changes", "expected"),
        [
            (np.array([[cd_result.Change.ADDED]]), [cd_result.Change.ADDED]),
            (
                np.array([[cd_result.Change.ADDED], [cd_result.Change.REMOVED]]),
                [cd_result.Change.ADDED, cd_result.Change.REMOVED],
            ),
        ],
    )
    def test_unique_change_values(
        self, changes: dtypes.NpArrayNx1Type[np.int32], expected: list[int]
    ) -> None:
        """Test the `unique_change_values` property."""
        points_in_cluster = result.PointsInCluster(
            points=np.array([[0.0, 0.0, 0.0] for _ in range(len(changes))]),
            pixels=np.array([[0, 0] for _ in range(len(changes))]),
            changes=changes,
            image_indices=np.array([[0] for _ in range(len(changes))]),
            index_to_image_id={0: "image_0"},
        )
        assert sorted(points_in_cluster.unique_change_values) == sorted(expected)

    @pytest.mark.parametrize(
        ("points", "expected"),
        [
            (np.array([[0.0, 0.0, 0.0]]), 1),
            (np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), 2),
        ],
    )
    def test_size(
        self, points: dtypes.NpArrayNx3Type[np.float32], expected: int
    ) -> None:
        """Test the `size` property."""
        points_in_cluster = result.PointsInCluster(
            points=points,
            pixels=np.array([[0, 0] for _ in range(len(points))]),
            changes=np.array([[cd_result.Change.ADDED] for _ in range(len(points))]),
            image_indices=np.array([[0] for _ in range(len(points))]),
            index_to_image_id={0: "image_0"},
        )
        assert points_in_cluster.size == expected

    def test_iter(self) -> None:
        """Test the `__iter__()` method."""
        points = [dtypes.Point(x=0.0, y=0.0, z=0.0), dtypes.Point(x=1.0, y=1.0, z=1.0)]
        pixels = [dtypes.Pixel(x=0, y=0), dtypes.Pixel(x=1, y=1)]
        changes = [cd_result.Change.ADDED, cd_result.Change.REMOVED]
        image_indices = ["image_0", "image_1"]
        points_in_cluster = result.PointsInCluster(
            points=np.array([[point.x, point.y, point.z] for point in points]),
            pixels=np.array([[pixel.x, pixel.y] for pixel in pixels]),
            changes=np.array([[change.value] for change in changes]),
            image_indices=np.array([[i] for i in range(len(image_indices))]),
            index_to_image_id=dict(enumerate(image_indices)),
        )
        for i, (point, pixel, change, image_index) in enumerate(
            points_in_cluster.iter()
        ):
            assert point == points[i]
            assert pixel == pixels[i]
            assert change == changes[i]
            assert image_index == image_indices[i]


class TestClusteredPoints:
    """Test suite for the `ClusteredPoints` class."""

    def test_serialize_model(self) -> None:
        """Test the `_serialize_model()` method."""
        clustered_point_list = [
            result.ClusteredPoint(
                image_id="image_0",
                change=cd_result.Change.ADDED,
                point=dtypes.Point(x=0.0, y=0.0, z=0.0),
                pixel=dtypes.Pixel(x=0, y=0),
                cluster_id=0,
            ),
            result.ClusteredPoint(
                image_id="image_1",
                change=cd_result.Change.REMOVED,
                point=dtypes.Point(x=1.0, y=1.0, z=1.0),
                pixel=dtypes.Pixel(x=1, y=1),
                cluster_id=1,
            ),
        ]
        clustered_points = result.ClusteredPoints(root=clustered_point_list)
        assert clustered_points._serialize_model() == clustered_point_list

    @pytest.mark.parametrize(
        ("cluster_ids", "expected"), [([0, 1, 0, 1], [0, 1]), ([2, 3, 4], [2, 3, 4])]
    )
    def test_unique_cluster_ids(
        self, cluster_ids: list[result.ClusterId], expected: list[result.ClusterId]
    ) -> None:
        """Test the `unique_cluster_ids` property."""
        clustered_points = result.ClusteredPoints(
            root=[
                result.ClusteredPoint(
                    image_id="image_0",
                    change=cd_result.Change.ADDED,
                    point=dtypes.Point(x=0.0, y=0.0, z=0.0),
                    pixel=dtypes.Pixel(x=0, y=0),
                    cluster_id=cluster_id,
                )
                for cluster_id in cluster_ids
            ]
        )
        assert sorted(clustered_points.unique_cluster_ids) == sorted(expected)

    def test_get_points_in_cluster(self) -> None:
        """Test the `get_points_in_cluster()` method."""
        clustered_points = result.ClusteredPoints(
            root=[
                result.ClusteredPoint(
                    image_id="image_0",
                    change=cd_result.Change.ADDED,
                    point=dtypes.Point(x=0.0, y=0.0, z=0.0),
                    pixel=dtypes.Pixel(x=0, y=0),
                    cluster_id=0,
                ),
                result.ClusteredPoint(
                    image_id="image_1",
                    change=cd_result.Change.REMOVED,
                    point=dtypes.Point(x=1.0, y=1.0, z=1.0),
                    pixel=dtypes.Pixel(x=1, y=1),
                    cluster_id=1,
                ),
                result.ClusteredPoint(
                    image_id="image_2",
                    change=cd_result.Change.ADDED,
                    point=dtypes.Point(x=2.0, y=2.0, z=2.0),
                    pixel=dtypes.Pixel(x=2, y=2),
                    cluster_id=0,
                ),
            ]
        )
        points_in_cluster = clustered_points.get_points_in_cluster(0)
        testing.assert_array_equal(
            points_in_cluster.points, np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        )
        testing.assert_array_equal(points_in_cluster.pixels, np.array([[0, 0], [2, 2]]))
        testing.assert_array_equal(
            points_in_cluster.changes,
            np.array([[cd_result.Change.ADDED], [cd_result.Change.ADDED]]),
        )
        testing.assert_array_equal(
            points_in_cluster.image_indices,
            np.array(
                [
                    [clustered_points._image_id_to_index[image_id]]
                    for image_id in ["image_0", "image_2"]
                ]
            ),
        )
        testing.assert_array_equal(
            points_in_cluster.changes,
            np.array([[cd_result.Change.ADDED], [cd_result.Change.ADDED]]),
        )

        points_in_cluster = clustered_points.get_points_in_cluster(1)
        testing.assert_array_equal(
            points_in_cluster.points, np.array([[1.0, 1.0, 1.0]])
        )
        testing.assert_array_equal(points_in_cluster.pixels, np.array([[1, 1]]))
        testing.assert_array_equal(
            points_in_cluster.changes, np.array([[cd_result.Change.REMOVED.value]])
        )
        testing.assert_array_equal(
            points_in_cluster.image_indices,
            np.array([[clustered_points._image_id_to_index["image_1"]]]),
        )
        testing.assert_array_equal(
            points_in_cluster.changes, np.array([[cd_result.Change.REMOVED.value]])
        )
